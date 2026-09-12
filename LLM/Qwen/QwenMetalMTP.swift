// The nextn drafter on the GPU with its own KV pool; mirrors llama.cpp
// qwen35.cpp graph_mtp. Leaves h_nextn for the tied lm_head.
import Foundation
import Metal

final class QwenMetalMTP {
    private let w: QwenMTP
    private let cfg: QwenConfig
    private let ctx: MetalContext
    private let map: UnsafeRawPointer
    private let tokEmbd: GGUFTensor
    private let head: GGUFTensor
    let pool: MetalKVPool
    private(set) var origin = 0

    private let bEmbed, bCat, bCur, bNormed, bContrib: MTLBuffer
    private let bQFull, bQ, bGate, bK1, bV1, bAttnOut: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    let bHidden: MTLBuffer
    let bLogits: MTLBuffer

    init(_ model: QwenModel, _ w: QwenMTP, ctx: MetalContext, pageP: Int) {
        self.w = w
        self.ctx = ctx
        cfg = model.cfg
        map = model.gguf.map
        tokEmbd = model.tokEmbd
        head = model.output
        let c = model.cfg
        pool = MetalKVPool(device: ctx.device, P: pageP,
                           kvDim: c.headDim * c.nHeadKV)
        bEmbed = ctx.makeF32(c.nEmbd)
        bCat = ctx.makeF32(2 * c.nEmbd)
        bCur = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQFull = ctx.makeF32(c.headDim * 2 * c.nHead)
        bQ = ctx.makeF32(c.headDim * c.nHead)
        bGate = ctx.makeF32(c.headDim * c.nHead)
        bK1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bV1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bAttnOut = ctx.makeF32(c.headDim * c.nHead)
        bFfnGate = ctx.makeF32(c.nFF)
        bFfnUp = ctx.makeF32(c.nFF)
        bHidden = ctx.makeF32(c.nEmbd)
        bLogits = ctx.makeF32(c.nVocab)
    }

    var cached: Int { pool.len }

    var end: Int { origin + pool.len }

    func reset() { reorigin(at: 0) }

    func reorigin(at position: Int) {
        pool.truncate(to: 0)
        origin = position
    }

    func truncate(to n: Int) { pool.truncate(to: n) }

    func snapshot() -> MetalKVPool.Snapshot { pool.snapshot() }

    func restore(_ s: MetalKVPool.Snapshot, origin position: Int) {
        pool.restore(s)
        origin = position
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    func encodeStep(_ f: MetalEnc, token: Int, hidden: MTLBuffer,
                    hiddenOff: Int, ropePos: Int, head wantHead: Bool) {
        let c = cfg
        encodeInput(f, token: token, hidden: hidden, hiddenOff: hiddenOff)
        encodeAttn(f, ropePos: ropePos)
        f.add(x: bCur, y: bContrib, n: c.nEmbd)
        f.rmsnorm(x: bCur, weightOff: off(w.attnPostNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
        f.gemv(w.ffnGate, x: bNormed, out: bFfnGate, off: off(w.ffnGate))
        f.gemv(w.ffnUp, x: bNormed, out: bFfnUp, off: off(w.ffnUp))
        f.siluMul(a: bFfnGate, b: bFfnUp, n: c.nFF)
        f.gemv(w.ffnDown, x: bFfnGate, out: bContrib, off: off(w.ffnDown))
        f.add(x: bCur, y: bContrib, n: c.nEmbd)
        f.rmsnorm(x: bCur, weightOff: off(w.headNorm), out: bHidden,
                  n: c.nEmbd, eps: c.eps)
        if wantHead {
            f.gemv(head, x: bHidden, out: bLogits, off: off(head))
        }
    }

    func encodeRow(_ f: MetalEnc, token: Int, hidden: MTLBuffer,
                   ropePos: Int) {
        encodeInput(f, token: token, hidden: hidden, hiddenOff: 0)
        encodeKV(f, ropePos: ropePos)
    }

    func encodeSeedInput(_ f: MetalEnc, x: MTLBuffer, N: Int,
                         _ s: QwenMTPSeedScratch) {
        let c = cfg
        f.rmsnormBatch(x: x, weightOff: off(w.enorm), y: s.cat, n: c.nEmbd,
                       rows: N, eps: c.eps, yStride: 2 * c.nEmbd)
    }

    func encodeSeed(_ f: MetalEnc, hidden: MTLBuffer, prev: MTLBuffer,
                    N: Int, basePos: Int, _ s: QwenMTPSeedScratch) {
        let c = cfg
        let stride = MemoryLayout<Float>.stride
        let wide = 2 * c.nEmbd
        f.rmsnorm(x: prev, weightOff: off(w.hnorm), out: s.cat, n: c.nEmbd,
                  eps: c.eps, outOff: c.nEmbd * stride)
        if N > 1 {
            f.rmsnormBatch(x: hidden, weightOff: off(w.hnorm), y: s.cat,
                           n: c.nEmbd, rows: N - 1, eps: c.eps,
                           yOff: (wide + c.nEmbd) * stride, yStride: wide)
        }
        f.gemm(w.ehProj, X: s.cat, out: s.cur, off: off(w.ehProj), N: N)
        f.rmsnormBatch(x: s.cur, weightOff: off(w.attnNorm), y: s.normed,
                       n: c.nEmbd, rows: N, eps: c.eps)
        f.gemm(w.wk, X: s.normed, out: s.k1, off: off(w.wk), N: N)
        f.gemm(w.wv, X: s.normed, out: s.v1, off: off(w.wv), N: N)
        f.rmsnormRows(x: s.k1, xoff: 0, d: c.headDim, rows: N * c.nHeadKV,
                      weightOff: off(w.kNorm), eps: c.eps)
        f.ropeBatch(x: s.k1, headDim: c.headDim, nHead: c.nHeadKV,
                    nRot: c.nRot, base: c.ropeBase, basePos: basePos, N: N)
        let at = pool.len
        pool.appendBatch(N)
        f.kvAppendBatch(kCurN: s.k1, vCurN: s.v1, kAddr: pool.kAddr,
                        vAddr: pool.vAddr, pages: pool.residentPages,
                        kvDim: c.headDim * c.nHeadKV, basePos: at,
                        P: pool.P, N: N)
    }

    private func encodeInput(_ f: MetalEnc, token: Int, hidden: MTLBuffer,
                             hiddenOff: Int) {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(tokEmbd.type, c.nEmbd)
        f.dequantRow(weightOff: off(tokEmbd) + UInt64(token * rowBytes),
                     out: bEmbed, n: c.nEmbd, type: tokEmbd.type)
        f.rmsnorm(x: bEmbed, weightOff: off(w.enorm), out: bCat, n: c.nEmbd,
                  eps: c.eps)
        f.rmsnorm(x: hidden, weightOff: off(w.hnorm), out: bCat, n: c.nEmbd,
                  eps: c.eps, xOff: hiddenOff,
                  outOff: c.nEmbd * MemoryLayout<Float>.stride)
        f.gemv(w.ehProj, x: bCat, out: bCur, off: off(w.ehProj))
        f.rmsnorm(x: bCur, weightOff: off(w.attnNorm), out: bNormed,
                  n: c.nEmbd, eps: c.eps)
    }

    private func encodeKV(_ f: MetalEnc, ropePos: Int) {
        let c = cfg
        f.gemv(w.wk, x: bNormed, out: bK1, off: off(w.wk))
        f.gemv(w.wv, x: bNormed, out: bV1, off: off(w.wv))
        f.rmsnormRows(x: bK1, xoff: 0, d: c.headDim, rows: c.nHeadKV,
                      weightOff: off(w.kNorm), eps: c.eps)
        f.rope(x: bK1, headDim: c.headDim, nHead: c.nHeadKV, nRot: c.nRot,
               base: c.ropeBase, pos: ropePos)
        let tail = pool.tailForAppend()
        f.kvAppend(kCur: bK1, vCur: bV1, K: tail.k, V: tail.v,
                   kvDim: c.headDim * c.nHeadKV, pos: tail.slot)
        pool.commitAppend()
        pool.refreshTable()
    }

    private func encodeAttn(_ f: MetalEnc, ropePos: Int) {
        let c = cfg
        f.gemv(w.wq, x: bNormed, out: bQFull, off: off(w.wq))
        f.splitQGate(qFull: bQFull, q: bQ, gate: bGate, hd: c.headDim,
                     nH: c.nHead)
        f.rmsnormRows(x: bQ, xoff: 0, d: c.headDim, rows: c.nHead,
                      weightOff: off(w.qNorm), eps: c.eps)
        f.rope(x: bQ, headDim: c.headDim, nHead: c.nHead, nRot: c.nRot,
               base: c.ropeBase, pos: ropePos)
        encodeKV(f, ropePos: ropePos)
        f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                    pages: pool.residentPages, gate: bGate, out: bAttnOut,
                    hd: c.headDim, nH: c.nHead, nKV: c.nHeadKV, T: pool.len,
                    kvDim: c.headDim * c.nHeadKV, P: pool.P,
                    scale: 1 / Float(c.headDim).squareRoot(), gated: 1)
        f.gemv(w.wo, x: bAttnOut, out: bContrib, off: off(w.wo))
    }
}

struct QwenMTPSeedScratch {
    let cat, cur, normed, k1, v1: MTLBuffer

    init(ctx: MetalContext, cfg c: QwenConfig, N: Int) {
        cat = ctx.makeF32(N * 2 * c.nEmbd)
        cur = ctx.makeF32(N * c.nEmbd)
        normed = ctx.makeF32(N * c.nEmbd)
        k1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
        v1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
    }
}
