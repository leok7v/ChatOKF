import Accelerate
import Foundation
import Metal

public final class QwenMetalViT {
    public let cfg: QwenViTConfig
    // Holds the mmap alive: ctx wraps it bytesNoCopy.
    private let gguf: GGUF
    private let ctx: MetalContext

    private struct Layer {
        let ln1w, ln1b, qkvW, qkvB, outW, outB: MTLBuffer
        let ln2w, ln2b, upW, upB, downW, downB: MTLBuffer
    }
    private let blocks: [Layer]
    private let patchW: MTLBuffer
    private let patchWPair: MTLBuffer
    private let patchB: [Float]
    private let posEmbd: [Float]
    private let postW, postB: MTLBuffer
    private let mm0W, mm0B, mm2W, mm2B: MTLBuffer

    private struct Tables {
        let h: Int
        let w: Int
        let order: [Int]
        let posBias: MTLBuffer
        let ropeCos: MTLBuffer
        let ropeSin: MTLBuffer
    }
    private var tables: Tables?
    private var capacity = 0
    private var bRows, bX, bNorm, bQkv, bCtx, bTmp, bFF, bOut: MTLBuffer

    public init(path: String) throws {
        let g = try GGUF(path: path)
        let c = QwenViTConfig(g)
        let context = try MetalContext(g)
        try context.prewarm()
        let dev = context.device
        func f32(_ name: String) -> MTLBuffer {
            context.makeF32(Dense.floats(g.tensor(name)))
        }
        func f16(_ name: String) -> MTLBuffer {
            QwenMetalViT.halfBuf(dev, Dense.floats(g.tensor(name)))
        }
        // The two temporal conv kernels sum for a still image; conv is linear.
        let w0 = Dense.floats(g.tensor("v.patch_embd.weight"))
        let w1 = Dense.floats(g.tensor("v.patch_embd.weight.1"))
        var sum = w0
        vDSP_vadd(w0, 1, w1, 1, &sum, 1, vDSP_Length(w0.count))
        gguf = g
        cfg = c
        ctx = context
        patchW = QwenMetalViT.halfBuf(dev, sum)
        patchWPair = QwenMetalViT.halfBuf(dev, QwenViT.pairKernel(
            w0, w1, embd: c.embd, pd: c.patchDim))
        patchB = Dense.floats(g.tensor("v.patch_embd.bias"))
        posEmbd = Dense.floats(g.tensor("v.position_embd.weight"))
        postW = f32("v.post_ln.weight")
        postB = f32("v.post_ln.bias")
        mm0W = f16("mm.0.weight")
        mm0B = f32("mm.0.bias")
        mm2W = f16("mm.2.weight")
        mm2B = f32("mm.2.bias")
        blocks = (0 ..< c.layers).map { il in
            Layer(ln1w: f32("v.blk.\(il).ln1.weight"),
                  ln1b: f32("v.blk.\(il).ln1.bias"),
                  qkvW: f16("v.blk.\(il).attn_qkv.weight"),
                  qkvB: f32("v.blk.\(il).attn_qkv.bias"),
                  outW: f16("v.blk.\(il).attn_out.weight"),
                  outB: f32("v.blk.\(il).attn_out.bias"),
                  ln2w: f32("v.blk.\(il).ln2.weight"),
                  ln2b: f32("v.blk.\(il).ln2.bias"),
                  upW: f16("v.blk.\(il).ffn_up.weight"),
                  upB: f32("v.blk.\(il).ffn_up.bias"),
                  downW: f16("v.blk.\(il).ffn_down.weight"),
                  downB: f32("v.blk.\(il).ffn_down.bias"))
        }
        let n = c.patches
        capacity = n
        bRows = context.makeF32(n * 2 * c.patchDim)
        bX = context.makeF32(n * c.embd)
        bNorm = context.makeF32(n * c.embd)
        bQkv = context.makeF32(n * 3 * c.embd)
        bCtx = context.makeF32(n * c.embd)
        bTmp = context.makeF32(n * c.embd)
        bFF = context.makeF32(n * c.ff)
        bOut = context.makeF32(c.mergedTokens * c.projDim)
    }

    private func ensure(_ n: Int) {
        if n > capacity {
            let c = cfg
            capacity = n
            bRows = ctx.makeF32(n * 2 * c.patchDim)
            bX = ctx.makeF32(n * c.embd)
            bNorm = ctx.makeF32(n * c.embd)
            bQkv = ctx.makeF32(n * 3 * c.embd)
            bCtx = ctx.makeF32(n * c.embd)
            bTmp = ctx.makeF32(n * c.embd)
            bFF = ctx.makeF32(n * c.ff)
            bOut = ctx.makeF32(n / (c.merge * c.merge) * c.projDim)
        }
    }

    private func prepare(h: Int, w: Int) -> Tables {
        if tables?.h != h || tables?.w != w {
            let c = cfg
            let order = QwenViT.mergeOrderTable(h: h, w: w, merge: c.merge)
            let pos = QwenViT.positionTable(posEmbd, embd: c.embd, h: h, w: w)
            var combined = [Float](repeating: 0, count: order.count * c.embd)
            for s in 0 ..< order.count {
                for i in 0 ..< c.embd {
                    combined[s * c.embd + i] =
                        patchB[i] + pos[order[s] * c.embd + i]
                }
            }
            let rope = QwenViT.ropeTables(c, order, w: w)
            tables = Tables(h: h, w: w, order: order,
                            posBias: ctx.makeF32(combined),
                            ropeCos: ctx.makeF32(rope.cos),
                            ropeSin: ctx.makeF32(rope.sin))
        }
        return tables!
    }

    public func forward(pixels: [Float]) throws -> [Float] {
        try forward(pixels: pixels, gridH: cfg.side, gridW: cfg.side)
    }

    public func forward(pixels: [Float], gridH: Int,
                        gridW: Int) throws -> [Float] {
        let t = prepare(h: gridH, w: gridW)
        return try run(QwenViT.patchRows(pixels, cfg, t.order, w: gridW),
                       cfg.patchDim, patchW, t)
    }

    public func forward(pair a: [Float], _ b: [Float], gridH: Int,
                        gridW: Int) throws -> [Float] {
        let t = prepare(h: gridH, w: gridW)
        return try run(QwenViT.pairRows(a, b, cfg, t.order, w: gridW),
                       2 * cfg.patchDim, patchWPair, t)
    }

    private func run(_ rows: [Float], _ k: Int, _ weight: MTLBuffer,
                     _ t: Tables) throws -> [Float] {
        let n = t.order.count
        ensure(n)
        let m2 = cfg.merge * cfg.merge
        let mTok = n / m2
        rows.withUnsafeBytes { src in
            _ = memcpy(bRows.contents(), src.baseAddress!, src.count)
        }
        let ran = GPUGate.shared.run("qwen vit") { encode(k, weight, t) }
        if !ran { throw GPUFault(description: "the vision tower did not run") }
        return Array(bOut.f32(mTok * cfg.projDim))
    }

    private func encode(_ k: Int, _ weight: MTLBuffer,
                        _ t: Tables) -> MTLCommandBuffer {
        let n = t.order.count
        let embd = cfg.embd
        let hd = cfg.headDim
        let scale = 1 / Float(hd).squareRoot()
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: enc)
        f.f16Gemm(weight, X: bRows, out: bX, K: k, M: embd, N: n)
        f.add(x: bX, y: t.posBias, n: n * embd)
        for l in blocks {
            f.layerNorm(x: bX, w: l.ln1w, b: l.ln1b, y: bNorm, n: embd,
                        rows: n, eps: cfg.eps)
            f.f16Gemm(l.qkvW, X: bNorm, out: bQkv, K: embd, M: 3 * embd, N: n)
            f.addBiasRows(x: bQkv, bias: l.qkvB, m: 3 * embd, rows: n)
            f.visionRope(x: bQkv, cos: t.ropeCos, sin: t.ropeSin,
                         rowStride: 3 * embd, off: 0, headDim: hd,
                         nHead: cfg.heads, N: n)
            f.visionRope(x: bQkv, cos: t.ropeCos, sin: t.ropeSin,
                         rowStride: 3 * embd, off: embd, headDim: hd,
                         nHead: cfg.heads, N: n)
            f.visionAttn(qkv: bQkv, out: bCtx, n: n, embd: embd, hd: hd,
                         nHead: cfg.heads, scale: scale)
            f.f16Gemm(l.outW, X: bCtx, out: bTmp, K: embd, M: embd, N: n)
            f.addBiasRows(x: bTmp, bias: l.outB, m: embd, rows: n)
            f.add(x: bX, y: bTmp, n: n * embd)
            f.layerNorm(x: bX, w: l.ln2w, b: l.ln2b, y: bNorm, n: embd,
                        rows: n, eps: cfg.eps)
            f.f16Gemm(l.upW, X: bNorm, out: bFF, K: embd, M: cfg.ff, N: n)
            f.addBiasRows(x: bFF, bias: l.upB, m: cfg.ff, rows: n)
            f.gelu(x: bFF, n: n * cfg.ff)
            f.f16Gemm(l.downW, X: bFF, out: bTmp, K: cfg.ff, M: embd, N: n)
            f.addBiasRows(x: bTmp, bias: l.downB, m: embd, rows: n)
            f.add(x: bX, y: bTmp, n: n * embd)
        }
        f.layerNorm(x: bX, w: postW, b: postB, y: bNorm, n: embd, rows: n,
                    eps: cfg.eps)
        // The merger's concat is a pure reshape over merge-ordered rows.
        let m2 = cfg.merge * cfg.merge
        let mIn = embd * m2
        let mTok = n / m2
        f.f16Gemm(mm0W, X: bNorm, out: bTmp, K: mIn, M: mIn, N: mTok)
        f.addBiasRows(x: bTmp, bias: mm0B, m: mIn, rows: mTok)
        f.gelu(x: bTmp, n: mTok * mIn)
        f.f16Gemm(mm2W, X: bTmp, out: bOut, K: mIn, M: cfg.projDim, N: mTok)
        f.addBiasRows(x: bOut, bias: mm2B, m: cfg.projDim, rows: mTok)
        enc.endEncoding()
        return cb
    }

    // One vectorized vImage pass: a scalar loop over ~250M params is seconds.
    private static func halfBuf(_ dev: MTLDevice, _ v: [Float]) -> MTLBuffer {
        let buf = dev.makeBuffer(length: v.count * 2,
                                 options: .storageModeShared)!
        v.withUnsafeBufferPointer { src in
            var s = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                height: 1, width: vImagePixelCount(v.count),
                rowBytes: v.count * 4)
            var d = vImage_Buffer(data: buf.contents(), height: 1,
                                  width: vImagePixelCount(v.count),
                                  rowBytes: v.count * 2)
            _ = vImageConvert_PlanarFtoPlanar16F(&s, &d, 0)
        }
        return buf
    }
}
