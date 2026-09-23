// Reuses the TEXT tower's norm kernels because gemma's encoder block IS its
// text block; weights stay quantized and expand in registers inside the GEMM.
import Foundation
import Metal

public final class Gemma4MetalViT {
    public let cfg: Gemma4VisionConfig
    private let cpu: Gemma4ViT
    private let ctx: MetalContext

    private let model: Gemma4Model
    private let map: UnsafeRawPointer
    private let scales: [String: SRQ]

    public init(_ model: Gemma4Model, ctx: MetalContext) throws {
        let g = model.gguf
        cfg = try Gemma4VisionConfig(g)
        cpu = try Gemma4ViT(model)
        self.ctx = ctx
        self.model = model
        map = g.map
        var found: [String: SRQ] = [:]
        for il in 0..<cfg.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffn_gate", "ffn_up", "ffn_down"] {
                let name = "v.blk.\(il).\(tag).weight"
                found[name] = SRQ(g, name)
            }
        }
        scales = found
    }

    private func srq(_ w: GGUFTensor) -> SRQ { scales[w.name] ?? SRQ.none }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    public func forward(pixels: [Float], pos: [(Int, Int)]) throws
        -> (tower: [Float], proj: [Float], count: Int) {
        let n = pos.count
        let e = cfg.embd
        let hd = cfg.headDim
        let nH = cfg.heads
        let wide = nH * hd
        let padded = pos.map { p in p.0 < 0 && p.1 < 0 }
        let rope = cpu.ropeForGPU(pos: pos)

        let bX = ctx.makeF32(cpu.embedForGPU(pixels: pixels, pos: pos))
        let bNorm = ctx.makeF32(n * e)
        let bTmp = ctx.makeF32(n * e)
        let bQ = ctx.makeF32(n * wide)
        let bK = ctx.makeF32(n * wide)
        let bV = ctx.makeF32(n * wide)
        let bCtx = ctx.makeF32(n * wide)
        let bFF = ctx.makeF32(n * cfg.ff)
        let bUp = ctx.makeF32(n * cfg.ff)
        let bCos = ctx.makeF32(rope.cos)
        let bSin = ctx.makeF32(rope.sin)
        let bMask = ctx.makeF32(padded.map { p in p ? Float(1) : Float(0) })
        let bClamp = ctx.makeF32(n * max(e, cfg.ff))

        let seed = Array(bX.f32(n * e))
        let ran = GPUGate.shared.run("gemma vit") {
            seed.withUnsafeBytes { raw in
                _ = memcpy(bX.contents(), raw.baseAddress!, raw.count)
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: enc)
            for il in 0..<cfg.layers {
                encodeBlock(f, il, n: n, bX, bNorm, bTmp, bQ, bK, bV, bCtx,
                            bFF, bUp, bCos, bSin, bMask, bClamp)
            }
            enc.endEncoding()
            return cb
        }
        if !ran { throw GPUFault(description: "the vision tower did not run") }
        return cpu.poolAndProject(Array(bX.f32(n * e)), pos: pos,
                                  padded: padded)
    }

    private func encodeBlock(_ f: MetalEnc, _ il: Int, n: Int,
                             _ bX: MTLBuffer, _ bNorm: MTLBuffer,
                             _ bTmp: MTLBuffer, _ bQ: MTLBuffer,
                             _ bK: MTLBuffer, _ bV: MTLBuffer,
                             _ bCtx: MTLBuffer, _ bFF: MTLBuffer,
                             _ bUp: MTLBuffer, _ bCos: MTLBuffer,
                             _ bSin: MTLBuffer, _ bMask: MTLBuffer,
                             _ bClamp: MTLBuffer) {
        let e = cfg.embd
        let hd = cfg.headDim
        let nH = cfg.heads
        let wide = nH * hd
        do {
            func t(_ n: String) -> GGUFTensor {
                model.gguf.tensor("v.blk.\(il).\(n)")
            }
            f.rmsnormBatchBF16(x: bX, weightOff: off(t("ln1.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.linear(t("attn_q.weight"), X: bNorm, out: bQ,
                     off: off(t("attn_q.weight")), N: n,
                     srq: srq(t("attn_q.weight")), scratch: bClamp)
            f.linear(t("attn_k.weight"), X: bNorm, out: bK,
                     off: off(t("attn_k.weight")), N: n,
                     srq: srq(t("attn_k.weight")), scratch: bClamp)
            f.linear(t("attn_v.weight"), X: bNorm, out: bV,
                     off: off(t("attn_v.weight")), N: n,
                     srq: srq(t("attn_v.weight")), scratch: bClamp)
            f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: n * nH,
                              weightOff: off(t("attn_q_norm.weight")),
                              eps: cfg.eps)
            f.rmsnormRowsBF16(x: bK, xoff: 0, d: hd, rows: n * nH,
                              weightOff: off(t("attn_k_norm.weight")),
                              eps: cfg.eps)
            f.rmsnormRowsNoWeight(x: bV, xoff: 0, d: hd, rows: n * nH,
                                  eps: cfg.eps)
            f.gemmaVisionRope(x: bQ, cos: bCos, sin: bSin, rowStride: wide,
                              headDim: hd, nHead: nH, N: n)
            f.gemmaVisionRope(x: bK, cos: bCos, sin: bSin, rowStride: wide,
                              headDim: hd, nHead: nH, N: n)
            f.gemmaVisionAttn(q: bQ, k: bK, v: bV, mask: bMask, out: bCtx,
                              n: n, hd: hd, nHead: nH)
            f.linear(t("attn_out.weight"), X: bCtx, out: bTmp,
                     off: off(t("attn_out.weight")), N: n,
                     srq: srq(t("attn_out.weight")), scratch: bClamp)
            f.rmsnormBatchBF16(x: bTmp,
                               weightOff: off(t("post_attn_norm.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.add(x: bX, y: bNorm, n: n * e)

            f.rmsnormBatchBF16(x: bX, weightOff: off(t("ln2.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.linear(t("ffn_gate.weight"), X: bNorm, out: bFF,
                     off: off(t("ffn_gate.weight")), N: n,
                     srq: srq(t("ffn_gate.weight")), scratch: bClamp)
            f.linear(t("ffn_up.weight"), X: bNorm, out: bUp,
                     off: off(t("ffn_up.weight")), N: n,
                     srq: srq(t("ffn_up.weight")), scratch: bClamp)
            f.activateMul(cfg.activation, a: bFF, b: bUp, n: n * cfg.ff)
            f.linear(t("ffn_down.weight"), X: bFF, out: bTmp,
                     off: off(t("ffn_down.weight")), N: n,
                     srq: srq(t("ffn_down.weight")), scratch: bClamp)
            f.rmsnormBatchBF16(x: bTmp,
                               weightOff: off(t("post_ffn_norm.weight")),
                               y: bNorm, n: e, rows: n, eps: cfg.eps)
            f.add(x: bX, y: bNorm, n: n * e)
        }
    }

}
