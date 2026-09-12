import Foundation
import Metal

final class Gemma4MetalAssist {
    private let w: Gemma4Assist
    private let cfg: Gemma4Config
    private let ctx: MetalContext
    private let map: UnsafeRawPointer
    private let tokEmbd: GGUFTensor
    private var scales: [String: SRQ] = [:]

    private struct Norms {
        let attn, postAttn, ffn, postFfn, qNorm: WeightRef
    }
    private var norms: [Norms] = []
    private let outNormOff: WeightRef

    private let bCat, bx, bNormed, bContrib: MTLBuffer
    private let bQ, bAttnOut, bGateNull: MTLBuffer
    private let bFfnGate, bFfnUp, bClamp: MTLBuffer
    let bLogits: MTLBuffer
    let bBack: MTLBuffer
    private let bPick: MTLBuffer
    private let bScores: MTLBuffer
    private let bClusters: MTLBuffer
    private let orderingOff: WeightRef?
    private let clusterCount: Int
    private let perCluster: Int

    static let centroids = Flags.on("centroids")
    static let topClusters = max(1, Flags.int("top-clusters") ?? 32)

    private var usesCentroids: Bool {
        Gemma4MetalAssist.centroids && w.clustered && perCluster > 0
    }

    let slidingSource: Int
    let fullSource: Int

    init(_ model: Gemma4Model, _ head: Gemma4Assist, ctx: MetalContext) {
        w = head
        cfg = model.cfg
        self.ctx = ctx
        map = model.gguf.map
        tokEmbd = model.tokEmbd
        var lastSliding = -1
        var lastFull = -1
        for il in 0..<cfg.nLayer where !cfg.isShared(il) {
            if cfg.isFull(il) { lastFull = il } else { lastSliding = il }
        }
        slidingSource = lastSliding
        fullSource = lastFull
        let maxHd = max(cfg.headDimSliding, cfg.headDimFull)
        let maxFF = head.layers.map { L in L.nFF }.max() ?? 0
        let nH = head.nHead
        bCat = ctx.makeF32(2 * head.backbone)
        bx = ctx.makeF32(head.nEmbd)
        bNormed = ctx.makeF32(head.nEmbd)
        bContrib = ctx.makeF32(head.nEmbd)
        bQ = ctx.makeF32(maxHd * nH)
        bAttnOut = ctx.makeF32(maxHd * nH)
        bGateNull = ctx.makeF32(maxHd * nH)
        bFfnGate = ctx.makeF32(maxFF)
        bFfnUp = ctx.makeF32(maxFF)
        bClamp = ctx.makeF32(max(maxFF, head.nEmbd))
        bLogits = ctx.makeF32(cfg.nVocab)
        bBack = ctx.makeF32(head.backbone)
        bPick = ctx.makeU32(1)
        clusterCount = head.centroids?.dims[1] ?? 0
        perCluster = clusterCount > 0 ? cfg.nVocab / clusterCount : 0
        bScores = ctx.makeF32(max(1, clusterCount))
        bClusters = ctx.makeU32(max(1, Gemma4MetalAssist.topClusters))
        let g = model.gguf
        orderingOff = head.tokenOrdering.map { t in
            ctx.window(UInt64(t.base - g.map))
        }
        for L in head.layers {
            for t in [L.wq, L.wo, L.ffnGate, L.ffnUp, L.ffnDown] {
                scales[t.name] = SRQ(g, t.name)
            }
        }
        scales[head.output.name] = SRQ(g, head.output.name)
        scales[head.preProj.name] = SRQ(g, head.preProj.name)
        scales[head.postProj.name] = SRQ(g, head.postProj.name)
        let context = ctx
        func normOff(_ t: GGUFTensor) -> WeightRef {
            precondition(t.type == .bf16,
                         "assist norm \(t.name) is \(t.type), expected bf16")
            return context.window(UInt64(t.base - g.map))
        }
        outNormOff = normOff(head.outputNorm)
        for L in head.layers {
            norms.append(Norms(attn: normOff(L.attnNorm),
                               postAttn: normOff(L.postAttnNorm),
                               ffn: normOff(L.ffnNorm),
                               postFfn: normOff(L.postFfnNorm),
                               qNorm: normOff(L.qNorm)))
        }
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    private func srq(_ t: GGUFTensor) -> SRQ { scales[t.name] ?? SRQ.none }

    var backboneWidth: Int { w.backbone }

    func seed(_ token: Int, hidden: UnsafePointer<Float>) {
        let dst = bCat.f32(2 * w.backbone).baseAddress!
        GQ.gather(tokEmbd, row: token, from: 0, count: w.backbone, into: dst)
        for i in 0..<w.backbone { dst[i] *= cfg.embedScale }
        memcpy(dst + w.backbone, hidden,
               w.backbone * MemoryLayout<Float>.stride)
    }

    func encodeStep(_ f: MetalEnc, pos: Int, pools: [Int: MetalKVPool]) {
        f.linear(w.preProj, x: bCat, out: bx, off: off(w.preProj),
                 srq: srq(w.preProj), scratch: bClamp)
        f.rmsnormBF16(x: bx, weightOff: norms[0].attn, out: bNormed,
                      n: w.nEmbd, eps: cfg.eps)
        for il in 0..<w.nLayer { layer(f, il, pos: pos, pools: pools) }
        if usesCentroids {
            f.gemv(w.centroids!, x: bNormed, out: bScores,
                   off: off(w.centroids!))
            f.assistTopClusters(x: bScores, out: bClusters, n: clusterCount,
                                k: Gemma4MetalAssist.topClusters)
            f.assistClusterArgmax(w.output, h: bNormed,
                                  ordering: orderingOff!,
                                  clusters: bClusters, out: bPick,
                                  off: off(w.output), dim: w.nEmbd,
                                  per: perCluster,
                                  k: Gemma4MetalAssist.topClusters)
        } else {
            f.linear(w.output, x: bNormed, out: bLogits, off: off(w.output),
                     srq: srq(w.output), scratch: bClamp)
            f.argmaxRows(x: bLogits, out: bPick, n: cfg.nVocab, rows: 1)
        }
        f.linear(w.postProj, x: bNormed, out: bBack, off: off(w.postProj),
                 srq: srq(w.postProj), scratch: bClamp)
    }

    private func layer(_ f: MetalEnc, _ il: Int, pos: Int,
                       pools: [Int: MetalKVPool]) {
        let L = w.layers[il]
        let isFull = w.isFull(il)
        let hd = isFull ? cfg.headDimFull : cfg.headDimSliding
        let nKV = isFull ? cfg.nHeadKVFull : cfg.nHeadKV
        let rot = isFull ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding
        let base = isFull ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let pool = pools[isFull ? fullSource : slidingSource]!
        f.linear(L.wq, x: bNormed, out: bQ, off: off(L.wq), srq: srq(L.wq),
                 scratch: bClamp)
        f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: w.nHead,
                          weightOff: norms[il].qNorm, eps: cfg.eps)
        f.ropeGemma(x: bQ, headDim: hd, nHead: w.nHead, rotated: rot,
                    base: base, pos: pos)
        let lo = isFull ? 0 : max(0, pos - cfg.slidingWindow + 1)
        f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                    pages: pool.residentPages, gate: bGateNull,
                    out: bAttnOut, hd: hd, nH: w.nHead, nKV: nKV,
                    T: pos + 1, kvDim: hd * nKV, P: pool.P, scale: 1,
                    gated: 0, lo: lo)
        f.linear(L.wo, x: bAttnOut, out: bContrib, off: off(L.wo),
                 srq: srq(L.wo), scratch: bClamp)
        f.normAddNormBF16(x: bx, c: bContrib, post: norms[il].postAttn,
                          pre: norms[il].ffn, y: bNormed, n: w.nEmbd,
                          rows: 1, eps: cfg.eps, s: 1)
        f.parallel {
            f.linear(L.ffnGate, x: bNormed, out: bFfnGate,
                     off: off(L.ffnGate), srq: srq(L.ffnGate),
                     scratch: bClamp)
            f.linear(L.ffnUp, x: bNormed, out: bFfnUp, off: off(L.ffnUp),
                     srq: srq(L.ffnUp), scratch: bClamp)
        }
        f.activateMul(cfg.activation, a: bFfnGate, b: bFfnUp, n: L.nFF)
        f.linear(L.ffnDown, x: bFfnGate, out: bContrib, off: off(L.ffnDown),
                 srq: srq(L.ffnDown), scratch: bClamp)
        let next = il + 1 < w.nLayer ? norms[il + 1].attn : outNormOff
        f.normAddNormBF16(x: bx, c: bContrib, post: norms[il].postFfn,
                          pre: next, y: bNormed, n: w.nEmbd, rows: 1,
                          eps: cfg.eps, s: L.layerScalar)
    }

    func logits() -> [Float] { Array(bLogits.f32(cfg.nVocab)) }

    func picked() -> Int32 {
        bPick.contents().assumingMemoryBound(to: Int32.self)[0]
    }

    func backbone() -> [Float] { Array(bBack.f32(w.backbone)) }
}
