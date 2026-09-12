import Foundation
import Metal

public final class QwenMetalEngine {
    let model: QwenModel
    let cfg: QwenConfig
    let ctx: MetalContext
    private let map: UnsafeRawPointer
    let pageP: Int

    public private(set) var pos = 0
    private(set) var ropeShift = 0
    public var sampler: Sampler?
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }

    private var gdnConv: [Int: MTLBuffer] = [:]
    private var gdnRec: [Int: MTLBuffer] = [:]
    private var kvPool: [Int: MetalKVPool] = [:]

    private var mtp: QwenMetalMTP?
    private var specN = 2
    private var plainDecode = false
    private var bSpecPrev: MTLBuffer?
    private var specBatch: BatchScratch?
    private var specLogits: MTLBuffer?
    private var specIds: MTLBuffer?
    private var specPick: MTLBuffer?
    // The GDN rollback ring: `recSlots` copies of every recurrent state, so
    // accepting m tokens is recSlot += m and rejecting costs nothing.
    private var recSlots = 1
    private var recSlot = 0
    private var specQueue: [Int32] = []
    private var specAligned: Bool {
        mtp.map { d in d.end == pos } ?? false
    }

    private let bx, bNormed, bContrib: MTLBuffer
    private let bQkv, bConvOut, bZ, bO: MTLBuffer
    private let bBetaPre, bAlphaPre, bBeta, bG: MTLBuffer
    private let bQFull, bQ, bGate, bK1, bV1, bAttnOut: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    private let bLogits: MTLBuffer

    public init(_ model: QwenModel, pageP: Int = 512) throws {
        self.model = model
        cfg = model.cfg
        map = model.gguf.map
        self.pageP = pageP
        ctx = try MetalContext(model.gguf)
        try ctx.prewarm()
        let c = cfg
        bx = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQkv = ctx.makeF32(c.convDim)
        bConvOut = ctx.makeF32(c.convDim)
        bZ = ctx.makeF32(c.valueDim)
        bO = ctx.makeF32(c.valueDim)
        bBetaPre = ctx.makeF32(c.nVHead)
        bAlphaPre = ctx.makeF32(c.nVHead)
        bBeta = ctx.makeF32(c.nVHead)
        bG = ctx.makeF32(c.nVHead)
        bQFull = ctx.makeF32(c.headDim * 2 * c.nHead)
        bQ = ctx.makeF32(c.headDim * c.nHead)
        bGate = ctx.makeF32(c.headDim * c.nHead)
        bK1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bV1 = ctx.makeF32(c.headDim * c.nHeadKV)
        bAttnOut = ctx.makeF32(c.headDim * c.nHead)
        bFfnGate = ctx.makeF32(c.nFF)
        bFfnUp = ctx.makeF32(c.nFF)
        bLogits = ctx.makeF32(c.nVocab)
        for il in 0..<c.nLayer {
            if c.isRecurrent(il) {
                gdnConv[il] = ctx.makeF32(c.convDim * (c.dConv - 1))
                gdnRec[il] = ctx.makeF32(c.nVHead * c.dState * c.dState)
            } else {
                kvPool[il] = MetalKVPool(device: ctx.device, P: pageP,
                                         kvDim: c.headDim * c.nHeadKV)
            }
        }
    }

    public func reset() {
        pos = 0
        ropeShift = 0
        stopSignal.clear()
        for (_, b) in gdnConv { memset(b.contents(), 0, b.length) }
        for (_, b) in gdnRec { memset(b.contents(), 0, b.length) }
        for (_, p) in kvPool { p.truncate(to: 0) }
        recSlot = 0
        specQueue.removeAll()
        mtp?.reset()
        if let prev = bSpecPrev { memset(prev.contents(), 0, prev.length) }
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    // The batched forward is built on the simdgroup-matrix GEMM, so it exists
    // only with matrix units; an A13 takes the token-by-token path, plain SIMD.
    var batched: Bool { ctx.matrixUnits }
    // Each chunk re-streams all weights once; capped by activation memory and
    // the serial-over-N GDN scan.
    static let prefillChunk = Flags.int("prefill-chunk") ?? 512
    static let skip: Set<String> = Set(
        (Flags.value("skip") ?? "").split(separator: ",").map(String.init))
    static func runs(_ group: String) -> Bool { !skip.contains(group) }

    public func extend(_ ids: [Int32]) -> Int32 {
        // A prior turn's Stop must not kill this one.
        stopSignal.clear()
        specQueue.removeAll()
        var out: Int32 = 0
        if ids.count > 1 {
            out = prefillBatch(ids)
        } else {
            var i = 0
            while i < ids.count && !stopSignal.raisedNow {
                if i == ids.count - 1 {
                    out = pick(forwardLogits(token: Int(ids[i]), pos: pos,
                                             seed: true))
                } else {
                    forward(token: Int(ids[i]), pos: pos, seed: true)
                }
                pos += 1
                i += 1
            }
        }
        return out
    }

    public func chunkCost(_ ids: [Int32], from first: Int,
                          want: (Int, Int32, UnsafePointer<Float>) -> Void) {
        let c = cfg
        let n = ids.count
        reset()
        let b = BatchScratch(ctx: ctx, cfg: c, N: n)
        let idsBuf = ctx.device.makeBuffer(length: n * 4,
                                           options: .storageModeShared)!
        idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: n) { p in
            for k in 0..<n { p[k] = ids[k] }
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
        f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd), out: b.x,
                     nEmbd: c.nEmbd, N: n, type: model.tokEmbd.type)
        encodeChunk(f, b, N: n, basePos: 0, pos3: nil, head: false)
        e.endEncoding()
        commitTimed(cb, "ppl.trunk")
        let sub = 64
        let hidden = ctx.makeF32(sub * c.nEmbd)
        let logits = ctx.makeF32(sub * c.nVocab)
        var at = max(first - 1, 0)
        while at < n - 1 {
            let span = min(sub, n - 1 - at)
            let src = b.normed.contents()
                .advanced(by: at * c.nEmbd * MemoryLayout<Float>.stride)
            memcpy(hidden.contents(), src,
                   span * c.nEmbd * MemoryLayout<Float>.stride)
            let hb = ctx.queue.makeCommandBuffer()!
            let he = hb.makeComputeCommandEncoder()!
            MetalEnc(ctx: ctx, e: he).gemm(model.output, X: hidden,
                                           out: logits,
                                           off: off(model.output), N: span)
            he.endEncoding()
            commitTimed(hb, "ppl.head")
            let lp = logits.contents().assumingMemoryBound(to: Float.self)
            for j in 0..<span {
                want(at + j, ids[at + j + 1], lp + j * c.nVocab)
            }
            at += span
        }
        pos += n
    }

    public func step(_ token: Int32) -> [Float] {
        let out = forwardLogits(token: Int(token), pos: pos)
        pos += 1
        return out
    }

    // A grammar mask cannot advance mid-cycle, so it stays on plain decode.
    public func setSpeculation(_ on: Bool) {
        plainDecode = !on
    }

    public func decode(_ token: Int32) -> Int32 {
        let ready = mtp != nil && !plainDecode && sampler?.logitMask == nil
        var out: Int32
        if !specQueue.isEmpty {
            out = specQueue.removeFirst()
        } else if ready && specAligned {
            specQueue = specDecode(token)
            out = specQueue.removeFirst()
        } else if ready {
            out = specPrime(token)
        } else {
            out = pick(forwardLogits(token: Int(token), pos: pos,
                                     seed: mtp != nil && !plainDecode))
            pos += 1
        }
        return out
    }

    // The drafter loaded but UNUSED: a baseline MTP cannot quietly become.
    public func decodePlain(_ token: Int32) -> Int32 {
        let out = pick(forwardLogits(token: Int(token), pos: pos))
        pos += 1
        return out
    }

    public var queued: Int { max(specQueue.count - 1, 0) }

    private func keepPrev(_ hidden: MTLBuffer, row: Int) {
        if let prev = bSpecPrev {
            let bytes = cfg.nEmbd * MemoryLayout<Float>.stride
            memcpy(prev.contents(), hidden.contents().advanced(by: row * bytes),
                   bytes)
        }
    }

    private func encodeDrafterRow(_ f: MetalEnc, token: Int, pos: Int) {
        if let d = mtp, let prev = bSpecPrev {
            if d.end != pos { d.reorigin(at: pos) }
            d.encodeRow(f, token: token, hidden: prev, ropePos: pos)
        }
    }

    private func encodeDrafterInput(_ f: MetalEnc, x: MTLBuffer, N: Int,
                                    _ s: QwenMTPSeedScratch?) {
        if let d = mtp, let s { d.encodeSeedInput(f, x: x, N: N, s) }
    }

    private func encodeDrafterSeed(_ f: MetalEnc, hidden: MTLBuffer, N: Int,
                                   basePos: Int, _ s: QwenMTPSeedScratch?) {
        if let d = mtp, let prev = bSpecPrev, let s {
            if d.end != basePos { d.reorigin(at: basePos) }
            d.encodeSeed(f, hidden: hidden, prev: prev, N: N,
                         basePos: basePos, s)
        }
    }

    func prefillBatch(_ ids: [Int32], chunk c0: Int? = nil) -> Int32 {
        let chunk = c0 ?? (ctx.matrixUnits
            ? QwenMetalEngine.prefillChunk
            : min(QwenMetalEngine.prefillChunk, MetalEnc.narrowMax))
        let c = cfg
        var out: Int32 = 0
        // One scratch set sized to the largest chunk; each chunk is its own
        // waited command buffer, so there is no cross-chunk hazard.
        let capN = min(chunk, ids.count)
        let b = BatchScratch(ctx: ctx, cfg: c, N: capN)
        let seed = mtp.map { _ in
            QwenMTPSeedScratch(ctx: ctx, cfg: c, N: capN)
        }
        let idsBuf = ctx.device.makeBuffer(length: capN * 4,
                                           options: .storageModeShared)!
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let end = min(i + chunk, ids.count)
            let N = end - i
            let basePos = pos
            idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: N) {
                p in
                for k in 0..<N { p[k] = ids[i + k] }
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
            let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
            f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd),
                         out: b.x, nEmbd: c.nEmbd, N: N,
                         type: model.tokEmbd.type)
            encodeDrafterInput(f, x: b.x, N: N, seed)
            encodeChunk(f, b, N: N, basePos: basePos, pos3: nil)
            encodeDrafterSeed(f, hidden: b.normed, N: N, basePos: basePos,
                              seed)
            e.endEncoding()
            commitTimed(cb, "prefillBatch")
            pos += N
            keepPrev(b.normed, row: N - 1)
            if end == ids.count { out = pick(Array(bLogits.f32(c.nVocab))) }
            i = end
        }
        return out
    }

    private func encodeChunk(_ f: MetalEnc, _ b: BatchScratch, N: Int,
                             basePos: Int, pos3: MTLBuffer?,
                             head: Bool = true,
                             ring: StateRing = .inPlace) {
        let c = cfg
        firstNorm(f, x: b.x, y: b.normed, rows: N, count: c.nLayer)
        for il in 0..<c.nLayer {
            let L = model.layers[il]
            if L.recurrent {
                gdnBatch(f, L, il, b, N: N, ring: ring)
            } else {
                attnBatchLayer(f, L, il, b, N: N, basePos: basePos,
                               pos3: pos3)
            }
            addNorm(f, x: b.x, r: b.contrib, y: b.normed, rows: N,
                    weightOff: off(L.attnPostNorm))
            if QwenMetalEngine.runs("ffn") {
                f.parallel {
                    f.gemm(L.ffnGate, X: b.normed, out: b.ffnGate,
                           off: off(L.ffnGate), N: N)
                    f.gemm(L.ffnUp, X: b.normed, out: b.ffnUp,
                           off: off(L.ffnUp), N: N)
                }
                if QwenMetalEngine.runs("glue") {
                    f.siluMul(a: b.ffnGate, b: b.ffnUp, n: N * c.nFF)
                }
                f.gemm(L.ffnDown, X: b.ffnGate, out: b.contrib,
                       off: off(L.ffnDown), N: N)
            }
            addNorm(f, x: b.x, r: b.contrib, y: b.normed, rows: N,
                    weightOff: nextNorm(il, count: c.nLayer))
        }
        if head {
            f.gemv(model.output, x: b.normed, out: bLogits,
                   off: off(model.output),
                   xOff: (N - 1) * c.nEmbd * MemoryLayout<Float>.stride)
        }
    }

    // The <|image_pad|> rows are replaced CPU-side, so no splice kernel exists.
    public func extendVision(_ ids: [Int32], feats: [[Float]],
                             spans: [(start: Int, gh: Int, gw: Int)])
        -> Int32 {
        stopSignal.clear()
        let c = cfg
        let base = pos
        let plan = VisionPositions.visionPositionsMulti(
            ids.count, spans, startScalar: base + ropeShift)
        ropeShift = plan.next - (base + ids.count)
        let chunk = QwenMetalEngine.prefillChunk
        let capN = min(chunk, ids.count)
        let b = BatchScratch(ctx: ctx, cfg: c, N: capN)
        let seed = mtp.map { _ in
            QwenMTPSeedScratch(ctx: ctx, cfg: c, N: capN)
        }
        let pos3Buf = ctx.device.makeBuffer(length: capN * 3 * 4,
                                            options: .storageModeShared)!
        var out: Int32 = 0
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let end = min(i + chunk, ids.count)
            let N = end - i
            let basePos = pos
            let xp = b.x.contents().assumingMemoryBound(to: Float.self)
            for k in 0..<N {
                let g = i + k
                var img = -1
                for (j, s) in spans.enumerated()
                where g >= s.start
                    && g < s.start + feats[j].count / c.nEmbd { img = j }
                if img >= 0 {
                    feats[img].withUnsafeBufferPointer { fp in
                        _ = memcpy(xp + k * c.nEmbd,
                                   fp.baseAddress!
                                       + (g - spans[img].start) * c.nEmbd,
                                   c.nEmbd * 4)
                    }
                } else {
                    QB.dequant(model.tokEmbd, row: Int(ids[g]),
                               count: c.nEmbd, into: xp + k * c.nEmbd)
                }
            }
            let pp = pos3Buf.contents().assumingMemoryBound(to: Int32.self)
            for k in 0..<N {
                let p = plan.pos[i + k]
                pp[k * 3] = p.0
                pp[k * 3 + 1] = p.1
                pp[k * 3 + 2] = p.2
            }
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
            let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
            encodeDrafterInput(f, x: b.x, N: N, seed)
            encodeChunk(f, b, N: N, basePos: basePos, pos3: pos3Buf)
            encodeDrafterSeed(f, hidden: b.normed, N: N, basePos: basePos,
                              seed)
            e.endEncoding()
            commitTimed(cb, "prefillVision")
            pos += N
            keepPrev(b.normed, row: N - 1)
            if end == ids.count { out = pick(Array(bLogits.f32(c.nVocab))) }
            i = end
        }
        return out
    }

    private func gdnBatch(_ f: MetalEnc, _ L: QwenLayer, _ il: Int,
                          _ b: BatchScratch, N: Int,
                          ring: StateRing = .inPlace) {
        let c = cfg
        let qkv = b.qkv
        let convOut = b.convOut
        let gate = b.g
        let beta = b.beta
        if QwenMetalEngine.runs("proj") {
            f.parallel {
                f.gemm(L.wqkv!, X: b.normed, out: qkv, off: off(L.wqkv!),
                       N: N)
                f.gemm(L.wqkvGate!, X: b.normed, out: b.z,
                       off: off(L.wqkvGate!), N: N)
                f.gemm(L.ssmBeta!, X: b.normed, out: b.betaPre,
                       off: off(L.ssmBeta!), N: N)
                f.gemm(L.ssmAlpha!, X: b.normed, out: b.alphaPre,
                       off: off(L.ssmAlpha!), N: N)
            }
        }
        f.parallel {
            if QwenMetalEngine.runs("gate") {
                f.gdnGate(bPre: b.betaPre, aPre: b.alphaPre,
                          dtOff: off(L.ssmDt!), aOff: off(L.ssmA!),
                          beta: beta, g: gate, nV: c.nVHead,
                          count: N * c.nVHead)
            }
            if QwenMetalEngine.runs("conv") {
                f.gdnConvBatch(qkvMixN: qkv, convState: gdnConv[il]!,
                               cwOff: off(L.ssmConv1d!), outN: convOut,
                               convDim: c.convDim, dConv: c.dConv, N: N,
                               ring: ring)
            }
        }
        // One dispatch for all N tokens replaces the 2*N tiny per-token ones
        // that made prefill CPU-bound.
        if QwenMetalEngine.runs("norm") {
            f.l2normRowsBatch(x: convOut, d: c.dState,
                              rowsPerTok: 2 * c.nKHead, tokStride: c.convDim,
                              tokens: N, eps: c.eps)
        }
        let qScale = 1 / Float(c.dState).squareRoot()
        if QwenMetalEngine.runs("scan") {
            f.gdnScanBatch(convOutN: convOut, keyDim: c.keyDim,
                           valueDim: c.valueDim, convDim: c.convDim, gN: gate,
                           betaN: beta, S: gdnRec[il]!, oN: b.o, nV: c.nVHead,
                           nK: c.nKHead, dS: c.dState, qScale: qScale, N: N,
                           ring: ring)
        }
        normSilu(f, o: b.o, z: b.z, rows: N * c.nVHead,
                 weightOff: off(L.ssmNorm!))
        f.gemm(L.ssmOut!, X: b.o, out: b.contrib, off: off(L.ssmOut!), N: N)
    }

    private func nextNorm(_ il: Int, count: Int) -> WeightRef {
        il + 1 < count ? off(model.layers[il + 1].attnNorm)
                       : off(model.outputNorm)
    }

    private func firstNorm(_ f: MetalEnc, x: MTLBuffer, y: MTLBuffer,
                           rows: Int, count: Int) {
        if QwenMetalEngine.runs("norm") {
            let w = count > 0 ? model.layers[0].attnNorm : model.outputNorm
            f.rmsnormBatch(x: x, weightOff: off(w), y: y, n: cfg.nEmbd,
                           rows: rows, eps: cfg.eps)
        }
    }

    private func addNorm(_ f: MetalEnc, x: MTLBuffer, r: MTLBuffer,
                         y: MTLBuffer, rows: Int, weightOff: WeightRef) {
        let c = cfg
        let norm = QwenMetalEngine.runs("norm")
        let glue = QwenMetalEngine.runs("glue")
        if norm && glue {
            f.addRmsnormBatch(x: x, r: r, weightOff: weightOff, y: y,
                              n: c.nEmbd, rows: rows, eps: c.eps)
        } else {
            if glue { f.add(x: x, y: r, n: rows * c.nEmbd) }
            if norm {
                f.rmsnormBatch(x: x, weightOff: weightOff, y: y, n: c.nEmbd,
                               rows: rows, eps: c.eps)
            }
        }
    }

    private func normSilu(_ f: MetalEnc, o: MTLBuffer, z: MTLBuffer,
                          rows: Int, weightOff: WeightRef) {
        let c = cfg
        let norm = QwenMetalEngine.runs("norm")
        let glue = QwenMetalEngine.runs("glue")
        if norm && glue {
            f.rmsnormRowsSilu(x: o, z: z, d: c.dState, rows: rows,
                              weightOff: weightOff, eps: c.eps)
        } else {
            if norm {
                f.rmsnormRows(x: o, xoff: 0, d: c.dState, rows: rows,
                              weightOff: weightOff, eps: c.eps)
            }
            if glue { f.mulSilu(a: o, b: z, n: rows * c.dState) }
        }
    }

    private func attnBatchLayer(_ f: MetalEnc, _ L: QwenLayer, _ il: Int,
                                _ b: BatchScratch, N: Int, basePos: Int,
                                pos3: MTLBuffer? = nil) {
        let c = cfg
        let kvDim = c.headDim * c.nHeadKV
        f.parallel {
            f.gemm(L.wq!, X: b.normed, out: c.dense ? b.q : b.qFull,
                   off: off(L.wq!), N: N)
            f.gemm(L.wk!, X: b.normed, out: b.k1, off: off(L.wk!), N: N)
            f.gemm(L.wv!, X: b.normed, out: b.v1, off: off(L.wv!), N: N)
        }
        if !c.dense {
            f.splitQGateBatch(qFullN: b.qFull, qN: b.q, gateN: b.gate,
                              hd: c.headDim, nH: c.nHead, N: N)
        }
        if QwenMetalEngine.runs("norm") {
            f.parallel {
                f.rmsnormRows(x: b.q, xoff: 0, d: c.headDim,
                              rows: N * c.nHead, weightOff: off(L.qNorm!),
                              eps: c.eps)
                f.rmsnormRows(x: b.k1, xoff: 0, d: c.headDim,
                              rows: N * c.nHeadKV, weightOff: off(L.kNorm!),
                              eps: c.eps)
            }
        }
        if let pos3 {
            f.ropeMBatch(x: b.q, pos3: pos3, headDim: c.headDim,
                         nHead: c.nHead, nRot: c.nRot, base: c.ropeBase, N: N)
            f.ropeMBatch(x: b.k1, pos3: pos3, headDim: c.headDim,
                         nHead: c.nHeadKV, nRot: c.nRot, base: c.ropeBase,
                         N: N)
        } else {
            f.ropeBatch(x: b.q, headDim: c.headDim, nHead: c.nHead,
                        nRot: c.nRot, base: c.ropeBase,
                        basePos: basePos + ropeShift, N: N)
            f.ropeBatch(x: b.k1, headDim: c.headDim, nHead: c.nHeadKV,
                        nRot: c.nRot, base: c.ropeBase,
                        basePos: basePos + ropeShift, N: N)
        }
        let pool = kvPool[il]!
        pool.appendBatch(N)
        f.kvAppendBatch(kCurN: b.k1, vCurN: b.v1, kAddr: pool.kAddr,
                        vAddr: pool.vAddr, pages: pool.residentPages,
                        kvDim: kvDim, basePos: basePos, P: pool.P, N: N)
        if QwenMetalEngine.runs("attn") {
            f.attnBatch(qN: b.q, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gateN: b.gate,
                        outN: b.attnOut, hd: c.headDim, nH: c.nHead,
                        nKV: c.nHeadKV, kvDim: kvDim, P: pool.P,
                        scale: 1 / Float(c.headDim).squareRoot(),
                        basePos: basePos, N: N, gated: c.dense ? 0 : 1)
        }
        f.gemm(L.wo!, X: b.attnOut, out: b.contrib, off: off(L.wo!), N: N)
    }

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let ropeShift: Int
        let recSlot: Int
        let conv: [Int: [Float]]
        let rec: [Int: [Float]]
        let kv: [Int: MetalKVPool.Snapshot]
        let drafter: MetalKVPool.Snapshot?
        let specOrigin: Int
        let prev: [Float]
    }

    public func bookmark() -> Bookmark {
        var conv: [Int: [Float]] = [:], rec: [Int: [Float]] = [:]
        var kv: [Int: MetalKVPool.Snapshot] = [:]
        for (il, b) in gdnConv { conv[il] = Array(b.f32(b.length / 4)) }
        for (il, b) in gdnRec { rec[il] = Array(b.f32(b.length / 4)) }
        for (il, p) in kvPool { kv[il] = p.snapshot() }
        return Bookmark(pos: pos, ropeShift: ropeShift, recSlot: recSlot,
                        conv: conv, rec: rec, kv: kv,
                        drafter: mtp?.snapshot(),
                        specOrigin: mtp?.origin ?? 0,
                        prev: bSpecPrev.map { b in Array(b.f32(cfg.nEmbd)) }
                            ?? [])
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        ropeShift = b.ropeShift
        recSlot = b.recSlot
        specQueue.removeAll()
        for (il, a) in b.conv { copyIn(a, gdnConv[il]!) }
        for (il, a) in b.rec { copyIn(a, gdnRec[il]!) }
        for (il, s) in b.kv { kvPool[il]!.restore(s) }
        if let d = mtp {
            if let s = b.drafter {
                d.restore(s, origin: b.specOrigin)
            } else {
                d.reorigin(at: b.specOrigin)
            }
        }
        if let prev = bSpecPrev {
            if b.prev.count == cfg.nEmbd {
                copyIn(b.prev, prev)
            } else {
                memset(prev.contents(), 0, prev.length)
            }
        }
    }

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putInt(&out, b.pos)
        StateBytes.putInt(&out, b.ropeShift)
        for dict in [b.conv, b.rec] {
            StateBytes.putKeyed(&out, dict) { out, _, v in
                StateBytes.putFloats(&out, v)
            }
        }
        StateBytes.putKeyed(&out, b.kv) { out, il, s in
            let pool = kvPool[il]!
            StateBytes.putInt(&out, s.len)
            StateBytes.putFloats(&out, pool.flatten(s.kPages, len: s.len))
            StateBytes.putFloats(&out, pool.flatten(s.vPages, len: s.len))
        }
        StateBytes.putInt(&out, b.specOrigin)
        StateBytes.putFloats(&out, b.prev)
        var rows = 0
        var kRows: [Float] = []
        var vRows: [Float] = []
        if let d = mtp, let s = b.drafter {
            rows = s.len
            kRows = d.pool.flatten(s.kPages, len: s.len)
            vRows = d.pool.flatten(s.vPages, len: s.len)
        }
        StateBytes.putInt(&out, rows)
        StateBytes.putFloats(&out, kRows)
        StateBytes.putFloats(&out, vRows)
        return out
    }

    // Pages are re-chunked at this engine's pageP.

    public func deserialize(_ data: Data) -> Bookmark? {
        StateBytes.read(data, named: false) { r in
            let pos = r.int()
            let ropeShift = r.int()
            let conv = StateBytes.keyed(&r) { r in r.span().array }
            let rec = StateBytes.keyed(&r) { r in r.span().array }
            let kv = StateBytes.keyed(&r) { r -> MetalKVPool.Snapshot in
                let len = r.int()
                let k = r.span()
                let v = r.span()
                let pool = MetalKVPool(device: ctx.device, P: pageP,
                                       kvDim: cfg.headDim * cfg.nHeadKV)
                pool.fill(k: k, v: v, count: len)
                return pool.snapshot()
            }
            let specOrigin = r.int()
            let prev = r.span().array
            let len = r.int()
            let k = r.span()
            let v = r.span()
            let kvDim = cfg.headDim * cfg.nHeadKV
            var drafter: MetalKVPool.Snapshot? = nil
            if len > 0 && k.count == len * kvDim && v.count == k.count {
                let pool = MetalKVPool(device: ctx.device, P: pageP,
                                       kvDim: kvDim)
                pool.fill(k: k, v: v, count: len)
                drafter = pool.snapshot()
            }
            return Bookmark(pos: pos, ropeShift: ropeShift,
                            recSlot: recSlot, conv: conv, rec: rec, kv: kv,
                            drafter: drafter, specOrigin: specOrigin,
                            prev: prev)
        }
    }

    private func copyIn(_ a: [Float], _ b: MTLBuffer) {
        a.withUnsafeBytes { raw in
            _ = memcpy(b.contents(), raw.baseAddress!, raw.count)
        }
    }

    // A greedy turn already has its answer from argmax_rows, so the row never
    // crosses to the host; a sampled one needs the whole distribution.
    private func pickRow(_ row: UnsafePointer<Float>, gpu: Int32) -> Int32 {
        var out = gpu
        if sampler != nil {
            out = pick(Array(UnsafeBufferPointer(start: row,
                                                 count: cfg.nVocab)))
        }
        return out
    }

    func pick(_ logits: [Float]) -> Int32 {
        var out: Int32
        if sampler != nil {
            var work = logits
            let picked = sampler!.sample(&work)
            sampler!.accept(picked)
            out = picked
        } else {
            out = Int32(Vectors.argmax(logits))
        }
        return out
    }

    // Debug knob: cap the layer count to bisect a GPU hang (0 = embed + norm).
    var maxLayers = Int.max
    static let timing = Flags.on("metal-timing")

    // No commit: the caller appends the lm_head in the SAME command buffer, so
    // a decoded token is ONE dispatch stream and ONE GPU sync.
    private func encodeForward(_ f: MetalEnc, token: Int, pos: Int) {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(model.tokEmbd.type, c.nEmbd)
        f.dequantRow(weightOff: off(model.tokEmbd) + UInt64(token * rowBytes),
                     out: bx, n: c.nEmbd, type: model.tokEmbd.type)
        let count = min(c.nLayer, maxLayers)
        firstNorm(f, x: bx, y: bNormed, rows: 1, count: count)
        for il in 0..<count {
            encodeLayer(f, il, pos: pos, next: nextNorm(il, count: count))
        }
    }

    private var imatrix: [String: MTLBuffer] = [:]
    private var imatrixWidth: [String: Int] = [:]

    public func imatrixSums() -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for (k, b) in imatrix { out[k] = Array(b.f32(imatrixWidth[k]!)) }
        return out
    }

    public func collectImatrix() {
        let c = cfg
        for il in 0..<c.nLayer {
            var sites = ["l\(il).in": c.nEmbd, "l\(il).post": c.nEmbd,
                         "l\(il).ffn_down": c.nFF]
            if c.isRecurrent(il) {
                sites["l\(il).ssm_out"] = c.valueDim
            } else {
                sites["l\(il).attn_out"] = c.headDim * c.nHead
            }
            for (name, n) in sites {
                let b = ctx.makeF32(n)
                memset(b.contents(), 0, b.length)
                imatrix[name] = b
                imatrixWidth[name] = n
            }
        }
    }

    private func tally(_ f: MetalEnc, _ key: String, _ src: MTLBuffer,
                       _ n: Int) {
        if let dst = imatrix[key] { f.accumSq(dst: dst, src: src, n: n) }
        if let h = hessian[key] { f.accumOuter(h: h, src: src, n: n) }
    }

    private var hessian: [String: MTLBuffer] = [:]

    static let wideHessian = Flags.on("hess-wide")

    public func collectHessians(from lo: Int, upto hi: Int) {
        let c = cfg
        for il in lo..<min(hi, c.nLayer) {
            var sites = ["in": c.nEmbd, "post": c.nEmbd]
            if QwenMetalEngine.wideHessian { sites["ffn_down"] = c.nFF }
            if c.isRecurrent(il) {
                sites["ssm_out"] = c.valueDim
            } else {
                sites["attn_out"] = c.headDim * c.nHead
            }
            for (site, n) in sites {
                let b = ctx.device.makeBuffer(
                    length: n * n * MemoryLayout<Float>.stride,
                    options: .storageModeShared)!
                memset(b.contents(), 0, b.length)
                hessian["l\(il).\(site)"] = b
            }
        }
    }

    public func hessianNames() -> [String] { Array(hessian.keys).sorted() }

    public func hessianBytes(_ name: String) -> Data {
        let b = hessian[name]!
        return Data(bytes: b.contents(), count: b.length)
    }

    private func encodeLayer(_ f: MetalEnc, _ il: Int, pos: Int,
                             next: WeightRef) {
        let c = cfg
        let L = model.layers[il]
        tally(f, "l\(il).in", bNormed, c.nEmbd)
        if L.recurrent {
            gdnLayer(f, L, il)
        } else {
            attnLayer(f, L, il, pos: pos)
        }
        addNorm(f, x: bx, r: bContrib, y: bNormed, rows: 1,
                weightOff: off(L.attnPostNorm))
        tally(f, "l\(il).post", bNormed, c.nEmbd)
        if QwenMetalEngine.runs("ffn") {
            f.parallel {
                f.gemv(L.ffnGate, x: bNormed, out: bFfnGate,
                       off: off(L.ffnGate))
                f.gemv(L.ffnUp, x: bNormed, out: bFfnUp, off: off(L.ffnUp))
            }
            if QwenMetalEngine.runs("glue") {
                f.siluMul(a: bFfnGate, b: bFfnUp, n: c.nFF)
            }
            tally(f, "l\(il).ffn_down", bFfnGate, c.nFF)
            f.gemv(L.ffnDown, x: bFfnGate, out: bContrib, off: off(L.ffnDown))
        }
        addNorm(f, x: bx, r: bContrib, y: bNormed, rows: 1, weightOff: next)
    }

    public func tapLayers(token: Int) -> [[Float]] {
        let c = cfg
        let rowBytes = GGUF.rowByteCount(model.tokEmbd.type, c.nEmbd)
        var out: [[Float]] = []
        let cb0 = ctx.queue.makeCommandBuffer()!
        let e0 = cb0.makeComputeCommandEncoder()!
        let f0 = MetalEnc(ctx: ctx, e: e0)
        f0.dequantRow(weightOff: off(model.tokEmbd) + UInt64(token * rowBytes),
                      out: bx, n: c.nEmbd, type: model.tokEmbd.type)
        firstNorm(f0, x: bx, y: bNormed, rows: 1, count: c.nLayer)
        e0.endEncoding()
        commitTimed(cb0, "tap.embed")
        out.append(Array(bx.f32(c.nEmbd)))
        for il in 0..<c.nLayer {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            encodeLayer(MetalEnc(ctx: ctx, e: e), il, pos: pos,
                        next: nextNorm(il, count: c.nLayer))
            e.endEncoding()
            commitTimed(cb, "tap.l\(il)")
            out.append(Array(bx.f32(c.nEmbd)))
        }
        pos += 1
        return out
    }

    @discardableResult
    func forward(token: Int, pos: Int, seed: Bool = false) -> MTLBuffer {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
        if seed { encodeDrafterRow(f, token: token, pos: pos) }
        encodeForward(f, token: token, pos: pos)
        e.endEncoding()
        commitTimed(cb, "forward")
        keepPrev(bNormed, row: 0)
        return bNormed
    }

    func forwardLogits(token: Int, pos: Int, seed: Bool = false) -> [Float] {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
        if seed { encodeDrafterRow(f, token: token, pos: pos) }
        encodeForward(f, token: token, pos: pos)
        f.gemv(model.output, x: bNormed, out: bLogits, off: off(model.output))
        e.endEncoding()
        commitTimed(cb, "forward+head")
        keepPrev(bNormed, row: 0)
        return Array(bLogits.f32(cfg.nVocab))
    }

    private func commitTimed(_ cb: MTLCommandBuffer, _ tag: String) {
        // iOS aborts a GPU submit made in the background; every runtime commit
        // funnels through here, so this one gate covers them all.
        BackgroundGate.shared.waitForForeground()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { fatalError("metal \(tag): \(err)") }
        if QwenMetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "\(tag) gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
    }

    private func gdnLayer(_ f: MetalEnc, _ L: QwenLayer, _ il: Int) {
        let c = cfg
        if QwenMetalEngine.runs("proj") {
            f.parallel {
                f.gemv(L.wqkv!, x: bNormed, out: bQkv, off: off(L.wqkv!))
                f.gemv(L.wqkvGate!, x: bNormed, out: bZ,
                       off: off(L.wqkvGate!))
                f.gemv(L.ssmBeta!, x: bNormed, out: bBetaPre,
                       off: off(L.ssmBeta!))
                f.gemv(L.ssmAlpha!, x: bNormed, out: bAlphaPre,
                       off: off(L.ssmAlpha!))
            }
        }
        f.parallel {
            if QwenMetalEngine.runs("gate") {
                f.gdnGate(bPre: bBetaPre, aPre: bAlphaPre,
                          dtOff: off(L.ssmDt!), aOff: off(L.ssmA!),
                          beta: bBeta, g: bG, nV: c.nVHead, count: c.nVHead)
            }
            if QwenMetalEngine.runs("conv") {
                f.gdnConv(qkvMix: bQkv, convState: gdnConv[il]!,
                          cwOff: off(L.ssmConv1d!), out: bConvOut,
                          convDim: c.convDim, dConv: c.dConv)
            }
        }
        if QwenMetalEngine.runs("norm") {
            f.l2normRowsBatch(x: bConvOut, d: c.dState,
                              rowsPerTok: 2 * c.nKHead, tokStride: c.convDim,
                              tokens: 1, eps: c.eps)
        }
        let qScale = 1 / Float(c.dState).squareRoot()
        if QwenMetalEngine.runs("scan") {
            f.gdnScanBatch(convOutN: bConvOut, keyDim: c.keyDim,
                           valueDim: c.valueDim, convDim: c.convDim, gN: bG,
                           betaN: bBeta, S: gdnRec[il]!, oN: bO, nV: c.nVHead,
                           nK: c.nKHead, dS: c.dState, qScale: qScale, N: 1)
        }
        normSilu(f, o: bO, z: bZ, rows: c.nVHead, weightOff: off(L.ssmNorm!))
        tally(f, "l\(il).ssm_out", bO, c.valueDim)
        f.gemv(L.ssmOut!, x: bO, out: bContrib, off: off(L.ssmOut!))
    }

    private func attnLayer(_ f: MetalEnc, _ L: QwenLayer, _ il: Int,
                           pos: Int) {
        let c = cfg
        // The hybrid fuses q|gate in wq and splits it; dense qwen3 projects q
        // straight.
        f.parallel {
            f.gemv(L.wq!, x: bNormed, out: c.dense ? bQ : bQFull,
                   off: off(L.wq!))
            f.gemv(L.wk!, x: bNormed, out: bK1, off: off(L.wk!))
            f.gemv(L.wv!, x: bNormed, out: bV1, off: off(L.wv!))
        }
        if !c.dense {
            f.splitQGate(qFull: bQFull, q: bQ, gate: bGate, hd: c.headDim,
                         nH: c.nHead)
        }
        if QwenMetalEngine.runs("norm") {
            f.parallel {
                f.rmsnormRows(x: bQ, xoff: 0, d: c.headDim, rows: c.nHead,
                              weightOff: off(L.qNorm!), eps: c.eps)
                f.rmsnormRows(x: bK1, xoff: 0, d: c.headDim,
                              rows: c.nHeadKV, weightOff: off(L.kNorm!),
                              eps: c.eps)
            }
        }
        f.parallel {
            f.rope(x: bQ, headDim: c.headDim, nHead: c.nHead, nRot: c.nRot,
                   base: c.ropeBase, pos: pos + ropeShift)
            f.rope(x: bK1, headDim: c.headDim, nHead: c.nHeadKV,
                   nRot: c.nRot, base: c.ropeBase, pos: pos + ropeShift)
        }
        let kvDim = c.headDim * c.nHeadKV
        let pool = kvPool[il]!
        let tail = pool.tailForAppend()
        f.kvAppend(kCur: bK1, vCur: bV1, K: tail.k, V: tail.v,
                   kvDim: kvDim, pos: tail.slot)
        pool.commitAppend()
        pool.refreshTable()
        if QwenMetalEngine.runs("attn") {
            f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.residentPages, gate: bGate, out: bAttnOut,
                        hd: c.headDim, nH: c.nHead, nKV: c.nHeadKV,
                        T: pool.len, kvDim: kvDim, P: pool.P,
                        scale: 1 / Float(c.headDim).squareRoot(),
                        gated: c.dense ? 0 : 1)
        }
        tally(f, "l\(il).attn_out", bAttnOut, c.headDim * c.nHead)
        f.gemv(L.wo!, x: bAttnOut, out: bContrib, off: off(L.wo!))
    }

    public private(set) var specCycles = 0
    public private(set) var specCommitted = 0
    public private(set) var specDrafted = 0
    public private(set) var specAccepted = 0

    public var mtpReady: Bool { mtp != nil }

    public func drainSpecTurn() -> SpecTurn? {
        var out: SpecTurn? = nil
        if specCycles > 0 {
            out = SpecTurn(cycles: specCycles,
                                  committed: specCommitted,
                                  drafted: specDrafted,
                                  accepted: specAccepted)
        }
        specCycles = 0
        specCommitted = 0
        specDrafted = 0
        specAccepted = 0
        return out
    }

    // The ring is (drafts + 2) copies of every recurrent state, so the check
    // is against THIS machine: a phone declines what a Mac takes.
    static let ringShare = Flags.double("mtp-ring-share") ?? 0.05

    public func loadMTP(drafts: Int = 2) {
        if let w = model.mtp, mtp == nil, drafts > 0, batched,
           ringFits(drafts) {
            let c = cfg
            let width = drafts + 1
            specN = drafts
            mtp = QwenMetalMTP(model, w, ctx: ctx, pageP: pageP)
            let prev = ctx.makeF32(c.nEmbd)
            memset(prev.contents(), 0, prev.length)
            bSpecPrev = prev
            specBatch = BatchScratch(ctx: ctx, cfg: c, N: width)
            specLogits = ctx.makeF32(width * c.nVocab)
            specIds = ctx.device.makeBuffer(
                length: width * 4, options: .storageModeShared)
            specPick = ctx.makeU32(width)
            recSlots = width + 1
            recSlot = 0
            for il in 0..<c.nLayer where c.isRecurrent(il) {
                gdnConv[il] = ctx.makeF32(
                    recSlots * c.convDim * (c.dConv - 1))
                gdnRec[il] = ctx.makeF32(
                    recSlots * c.nVHead * c.dState * c.dState)
            }
            Diag.shared.report(.load, "[mtp] ON: metal drafter, "
                + "n=\(drafts), "
                + "\(recSlots) state slots")
        } else if model.mtp != nil, drafts > 0, !batched {
            Diag.shared.report(.load,
                               "[mtp] OFF: the verify needs matrix units")
        } else if model.mtp != nil, drafts > 0, !ringFits(drafts) {
            Diag.shared.report(.load, "[mtp] OFF: the "
                + "\(drafts + 2)-slot ring needs "
                + "\(ringBytes(drafts) / 1_048_576) MB, over "
                + "\(Int(QwenMetalEngine.ringShare * 100))% of this machine")
        }
    }

    private func ringBytes(_ drafts: Int) -> Int {
        let c = cfg
        let per = c.convDim * (c.dConv - 1) + c.nVHead * c.dState * c.dState
        var layers = 0
        for il in 0..<c.nLayer where c.isRecurrent(il) { layers += 1 }
        return (drafts + 2) * per * layers * MemoryLayout<Float>.stride
    }

    private func ringFits(_ drafts: Int) -> Bool {
        let have = Double(ProcessInfo.processInfo.physicalMemory)
        return Double(ringBytes(drafts)) < have * QwenMetalEngine.ringShare
    }

    public func specPrime(_ token: Int32) -> Int32 {
        let out = pick(forwardLogits(token: Int(token), pos: pos, seed: true))
        pos += 1
        return out
    }

    public func specDecode(_ token: Int32) -> [Int32] {
        var out: [Int32] = []
        if let d = mtp, let prev = bSpecPrev,
           let b = specBatch, let lg = specLogits, let idsBuf = specIds {
            out = specCycle(d, prev, b, lg, idsBuf, token)
        } else {
            out = [specPrime(token)]
        }
        return out
    }

    private func specCycle(_ d: QwenMetalMTP,
                           _ prev: MTLBuffer, _ b: BatchScratch,
                           _ lg: MTLBuffer, _ idsBuf: MTLBuffer,
                           _ token: Int32) -> [Int32] {
        let c = cfg
        let p0 = pos
        let mtpBase = d.cached
        var fed: [Int32] = [token]
        var drafts: [Int32] = []
        var i = 0
        while i < specN {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
            let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
            d.encodeStep(f, token: Int(fed[i]),
                         hidden: i == 0 ? prev : d.bHidden, hiddenOff: 0,
                         ropePos: p0 + i, head: true)
            f.argmaxRows(x: d.bLogits, out: specPick!, n: c.nVocab, rows: 1)
            e.endEncoding()
            commitTimed(cb, "spec.draft")
            let dt = specPick!.contents()
                .assumingMemoryBound(to: Int32.self)[0]
            drafts.append(dt)
            fed.append(dt)
            i += 1
        }
        let width = fed.count
        idsBuf.contents().withMemoryRebound(to: Int32.self, capacity: width) {
            p in
            for k in 0..<width { p[k] = fed[k] }
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
        f.embedBatch(ids: idsBuf, weightOff: off(model.tokEmbd), out: b.x,
                     nEmbd: c.nEmbd, N: width, type: model.tokEmbd.type)
        encodeChunk(f, b, N: width, basePos: p0, pos3: nil, head: false,
                    ring: StateRing(slot0: recSlot, slots: recSlots))
        f.gemm(model.output, X: b.normed, out: lg, off: off(model.output),
               N: width)
        f.argmaxRows(x: lg, out: specPick!, n: c.nVocab, rows: width)
        e.endEncoding()
        commitTimed(cb, "spec.verify")
        let lp = lg.contents().assumingMemoryBound(to: Float.self)
        let gp = specPick!.contents().assumingMemoryBound(to: Int32.self)
        var accepted = 0
        var bonus: Int32 = 0
        var scanning = true
        while scanning {
            let r = pickRow(lp + accepted * c.nVocab, gpu: gp[accepted])
            if accepted < drafts.count && r == drafts[accepted] {
                accepted += 1
            } else {
                bonus = r
                scanning = false
            }
        }
        let m = accepted + 1
        recSlot = (recSlot + m) % recSlots
        if m < width {
            for (_, p) in kvPool { p.truncate(to: p0 + m) }
        }
        pos = p0 + m
        maintainMTP(d, prev, b, fed, from: mtpBase, at: p0, count: m)
        keepPrev(b.normed, row: m - 1)
        specCycles += 1
        specCommitted += m
        specDrafted += drafts.count
        specAccepted += accepted
        var out = Array(drafts.prefix(accepted))
        out.append(bonus)
        return out
    }

    // Rebuilds the drafter's KV over the committed tokens from the BASE
    // hiddens; draft 0 already wrote row `base` from this `prev`.
    private func maintainMTP(_ d: QwenMetalMTP, _ prev: MTLBuffer,
                             _ b: BatchScratch, _ fed: [Int32], from base: Int,
                             at p0: Int, count m: Int) {
        let c = cfg
        d.truncate(to: base + 1)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        let f = MetalEnc(ctx: ctx, e: e, concurrent: true)
        var j = 1
        while j < m {
            let stride = MemoryLayout<Float>.stride
            d.encodeStep(f, token: Int(fed[j]), hidden: b.normed,
                         hiddenOff: (j - 1) * c.nEmbd * stride,
                         ropePos: p0 + j, head: false)
            j += 1
        }
        e.endEncoding()
        commitTimed(cb, "spec.maintain")
    }

    public func logits(_ hidden: MTLBuffer) -> [Float] {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemv(model.output, x: hidden, out: bLogits,
                                      off: off(model.output))
        e.endEncoding()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        if QwenMetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "logits gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
        return Array(bLogits.f32(cfg.nVocab))
    }
}


private struct BatchScratch {
    let x, normed, contrib: MTLBuffer
    let qkv, z, betaPre, alphaPre, beta, g, convOut, o: MTLBuffer
    let qFull, q, gate, k1, v1, attnOut: MTLBuffer
    let ffnGate, ffnUp: MTLBuffer

    init(ctx: MetalContext, cfg c: QwenConfig, N: Int) {
        x = ctx.makeF32(N * c.nEmbd)
        normed = ctx.makeF32(N * c.nEmbd)
        contrib = ctx.makeF32(N * c.nEmbd)
        qkv = ctx.makeF32(N * c.convDim)
        z = ctx.makeF32(N * c.valueDim)
        betaPre = ctx.makeF32(N * c.nVHead)
        alphaPre = ctx.makeF32(N * c.nVHead)
        beta = ctx.makeF32(N * c.nVHead)
        g = ctx.makeF32(N * c.nVHead)
        convOut = ctx.makeF32(N * c.convDim)
        o = ctx.makeF32(N * c.valueDim)
        qFull = ctx.makeF32(N * c.headDim * 2 * c.nHead)
        q = ctx.makeF32(N * c.headDim * c.nHead)
        gate = ctx.makeF32(N * c.headDim * c.nHead)
        k1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
        v1 = ctx.makeF32(N * c.headDim * c.nHeadKV)
        attnOut = ctx.makeF32(N * c.headDim * c.nHead)
        ffnGate = ctx.makeF32(N * c.nFF)
        ffnUp = ctx.makeF32(N * c.nFF)
    }
}

extension QwenMetalEngine.Bookmark: BackendState {}

extension QwenMetalEngine: TextEngine {}
