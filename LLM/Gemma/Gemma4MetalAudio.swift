// Every norm here is F32 where the other towers carry BF16; a BF16 kernel
// pointed at one walks the weights at half stride. `off` asserts the type.
import Foundation
import Metal

public final class Gemma4MetalAudio {
    public let cfg: Gemma4AudioConfig
    private let cpu: Gemma4Audio
    private let ctx: MetalContext
    private let model: Gemma4Model
    private let map: UnsafeRawPointer
    // Held for the tower's life, so no dispatch binds a buffer the encoder
    // outlives.
    private let relKey: [MTLBuffer]
    private let qScale: [MTLBuffer]
    private let dwWeight: [MTLBuffer]
    private let scales: [String: SRQ]

    public init(_ model: Gemma4Model, ctx: MetalContext) throws {
        let g = model.gguf
        let conf = try Gemma4AudioConfig(g)
        let tower = try Gemma4Audio(model)
        cfg = conf
        cpu = tower
        self.ctx = ctx
        self.model = model
        map = g.map
        relKey = (0..<conf.layers).map { il in
            ctx.makeF32(tower.relativeSlice(il))
        }
        qScale = (0..<conf.layers).map { il in
            ctx.makeF32(tower.queryScales(il))
        }
        dwWeight = (0..<conf.layers).map { il in
            ctx.makeF32(Dense.floats(
                g.tensor("a.blk.\(il).lconv_dw.weight")))
        }
        var found: [String: SRQ] = [:]
        for il in 0..<conf.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffw1_up", "ffw1_down", "ffw2_up", "ffw2_down",
                        "lconv_start", "lconv_end"] {
                let name = "a.blk.\(il).\(tag).weight"
                found[name] = SRQ(g, name)
            }
        }
        scales = found
    }

    private func srq(_ w: GGUFTensor) -> SRQ { scales[w.name] ?? SRQ.none }

    private struct Scratch {
        let x, norm, tmp, wide, glu: MTLBuffer
        let q, k, v, out, attn, ff, clamp: MTLBuffer
    }

    public func forward(mel: [Float], frames: Int, bins: Int) throws
        -> (tower: [Float], proj: [Float], count: Int) {
        let (seed, n) = cpu.subsample(mel, frames, bins)
        let e = cfg.embd
        let s = Scratch(x: ctx.makeF32(seed), norm: ctx.makeF32(n * e),
                        tmp: ctx.makeF32(n * e),
                        wide: ctx.makeF32(n * 2 * e),
                        glu: ctx.makeF32(n * e), q: ctx.makeF32(n * e),
                        k: ctx.makeF32(n * e), v: ctx.makeF32(n * e),
                        out: ctx.makeF32(n * e), attn: ctx.makeF32(n * e),
                        ff: ctx.makeF32(n * cfg.ff),
                        clamp: ctx.makeF32(n * max(cfg.ff, 2 * e)))
        let ran = GPUGate.shared.run("gemma audio") {
            seed.withUnsafeBytes { raw in
                _ = memcpy(s.x.contents(), raw.baseAddress!, raw.count)
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: enc)
            for il in 0..<cfg.layers {
                feedForward(f, il, "ffw1", s, n)
                f.rmsnormBatch(x: s.x, weightOff: off(il, "norm_pre_attn"),
                               y: s.norm, n: e, rows: n, eps: cfg.eps)
                attention(f, il, s, n)
                f.rmsnormRows(x: s.attn, xoff: 0, d: e, rows: n,
                              weightOff: off(il, "norm_post_attn"),
                              eps: cfg.eps)
                f.add(x: s.x, y: s.attn, n: n * e)
                lightConv(f, il, s, n)
                feedForward(f, il, "ffw2", s, n)
                f.rmsnormRows(x: s.x, xoff: 0, d: e, rows: n,
                              weightOff: off(il, "norm_out"), eps: cfg.eps)
            }
            enc.endEncoding()
            return cb
        }
        if !ran { throw GPUFault(description: "the audio tower did not run") }
        let tower = cpu.outputProject(Array(s.x.f32(n * e)), n)
        return (tower, cpu.project(tower, n), n)
    }

    // A HALF-WEIGHT residual add rather than a full one.
    private func feedForward(_ f: MetalEnc, _ il: Int, _ tag: String,
                             _ s: Scratch, _ n: Int) {
        let e = cfg.embd
        let up = tensor(il, "\(tag)_up.weight")
        let down = tensor(il, "\(tag)_down.weight")
        f.rmsnormBatch(x: s.x, weightOff: off(il, "\(tag)_norm"), y: s.norm,
                       n: e, rows: n, eps: cfg.eps)
        f.linear(up, X: s.norm, out: s.ff, off: byteOff(up), N: n,
                 srq: srq(up), scratch: s.clamp)
        f.activateInPlace(cfg.activation, x: s.ff, n: n * cfg.ff)
        f.linear(down, X: s.ff, out: s.tmp, off: byteOff(down), N: n,
                 srq: srq(down), scratch: s.clamp)
        f.rmsnormRows(x: s.tmp, xoff: 0, d: e, rows: n,
                      weightOff: off(il, "\(tag)_post_norm"), eps: cfg.eps)
        f.fmaInPlace(x: s.x, y: s.tmp, n: n * e, s: cfg.residualWeight)
    }

    private func attention(_ f: MetalEnc, _ il: Int, _ s: Scratch, _ n: Int) {
        let e = cfg.embd
        let wq = tensor(il, "attn_q.weight")
        let wk = tensor(il, "attn_k.weight")
        let wv = tensor(il, "attn_v.weight")
        let wo = tensor(il, "attn_out.weight")
        f.linear(wq, X: s.norm, out: s.q, off: byteOff(wq), N: n,
                 srq: srq(wq), scratch: s.clamp)
        f.linear(wk, X: s.norm, out: s.k, off: byteOff(wk), N: n,
                 srq: srq(wk), scratch: s.clamp)
        f.linear(wv, X: s.norm, out: s.v, off: byteOff(wv), N: n,
                 srq: srq(wv), scratch: s.clamp)
        f.scaleHeadDims(x: s.q, scales: qScale[il], rows: n, width: e,
                        hd: cfg.headDim)
        f.scaleInPlace(x: s.k, n: n * e, s: cpu.keyScale)
        f.gemmaAudioAttn(q: s.q, k: s.k, v: s.v, relk: relKey[il],
                         out: s.out, rows: n, width: e, heads: cfg.heads,
                         hd: cfg.headDim, chunk: cfg.chunk,
                         context: cfg.context, past: cfg.pastHorizon,
                         cap: cfg.logitCap)
        f.linear(wo, X: s.out, out: s.attn, off: byteOff(wo), N: n,
                 srq: srq(wo), scratch: s.clamp)
    }

    private func lightConv(_ f: MetalEnc, _ il: Int, _ s: Scratch, _ n: Int) {
        let e = cfg.embd
        let start = tensor(il, "lconv_start.weight")
        let end = tensor(il, "lconv_end.weight")
        f.rmsnormBatch(x: s.x, weightOff: off(il, "lconv_norm"), y: s.norm,
                       n: e, rows: n, eps: cfg.eps)
        f.linear(start, X: s.norm, out: s.wide, off: byteOff(start), N: n,
                 srq: srq(start), scratch: s.clamp)
        f.gluRows(src: s.wide, dst: s.glu, rows: n, width: e)
        f.depthwiseCausal(x: s.glu, w: dwWeight[il], out: s.tmp, rows: n,
                          width: e, k: cfg.convKernel)
        f.rmsnormRows(x: s.tmp, xoff: 0, d: e, rows: n,
                      weightOff: off(il, "lconv_conv_norm"), eps: cfg.eps)
        f.activateInPlace(cfg.activation, x: s.tmp, n: n * e)
        f.linear(end, X: s.tmp, out: s.out, off: byteOff(end), N: n,
                 srq: srq(end), scratch: s.clamp)
        f.add(x: s.x, y: s.out, n: n * e)
    }

    private func tensor(_ il: Int, _ name: String) -> GGUFTensor {
        model.gguf.tensor("a.blk.\(il).\(name)")
    }

    private func byteOff(_ w: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(w.base - map))
    }

    private func off(_ il: Int, _ name: String) -> WeightRef {
        let w = tensor(il, "\(name).weight")
        precondition(w.type == .f32,
                     "audio norm \(w.name) is \(w.type), expected f32")
        return ctx.window(UInt64(w.base - map))
    }
}
