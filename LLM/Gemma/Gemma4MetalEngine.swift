import Foundation
import Metal

public final class Gemma4MetalEngine {
    let model: Gemma4Model
    public let cfg: Gemma4Config
    public let ctx: MetalContext
    private let map: UnsafeRawPointer
    let pageP: Int

    public private(set) var pos = 0
    public var sampler: Sampler?
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }

    // One pool per NON-shared layer; a shared layer reads its source's.
    private var kv: [Int: MetalKVPool] = [:]
    private var assist: Gemma4MetalAssist?
    private var bLogitsN: MTLBuffer?
    private var specQueue: [Int32] = []
    private var plainDecode = false
    public private(set) var specCycles = 0
    public private(set) var specCommitted = 0
    public private(set) var specDrafted = 0
    public private(set) var specAccepted = 0
    public static let specN = max(1, Flags.int("spec-n") ?? 3)

    private let bx, bNormed, bContrib: MTLBuffer
    private let bQ, bK, bV, bAttnOut, bGateNull: MTLBuffer
    private let bFfnGate, bFfnUp: MTLBuffer
    private let bPle, bPleProj, bPleGate: MTLBuffer
    private let bLogits: MTLBuffer
    private let bClamp: MTLBuffer
    private let bXN, bNormedN, bContribN: MTLBuffer
    private let bQN, bKN, bVN, bAttnOutN, bGateNullN: MTLBuffer
    private let bFfnGateN, bFfnUpN: MTLBuffer
    private let bPleN, bPleProjN, bPleGateN, bClampN: MTLBuffer
    // The vision block of each chunk ROW, two absolute uints per row, sized 8
    // rows past the batch: the matrix attention masks a partial tail tile.
    private let bBlocks: MTLBuffer

    // 128 buys 98% of the win for a quarter of 512's scratch.
    public static let defaultBatch: Int = {
        let gb = (ProcessInfo.processInfo.physicalMemory + (1 << 29)) >> 30
        return Flags.int("gemma-batch") ?? (gb <= 4 ? 32 : 128)
    }()
    private let capacity: Int
    public var batch: Int {
        get { batchStore }
        set {
            precondition(newValue >= 1 && newValue <= capacity,
                         "batch \(newValue) exceeds the \(capacity) this "
                         + "engine allocated for; set --gemma-batch before "
                         + "constructing it")
            batchStore = newValue
        }
    }
    private var batchStore: Int
    // ONE because splitting is measured free; residency barely moves either
    // way.
    public static let prefillLayers = max(1, Flags.int("prefill-layers") ?? 1)
    public var probe: Set<Int> = []
    public var onLayer: ((Int, Int, [Float]) -> Void)?

    static let skip: Set<String> = Set(
        (Flags.value("skip") ?? "").split(separator: ",").map(String.init))
    // GPU interval, not wall clock: wall-clock t/s swings far too much on a
    // shared GPU to A/B against.
    static let timing = Flags.on("metal-timing")

    private var scales: [String: SRQ] = [:]

    private struct LayerNorms {
        let attn, postAttn, ffn, postFfn, qNorm: WeightRef
        let kNorm: WeightRef?
        let perLayerPost: WeightRef?
    }
    private var norms: [LayerNorms] = []
    private let outputNormOff: WeightRef
    private let perLayerProjNormOff: WeightRef?

    public init(_ model: Gemma4Model, pageP: Int = 512) throws {
        self.model = model
        cfg = model.cfg
        map = model.gguf.map
        self.pageP = pageP
        ctx = try MetalContext(model.gguf)
        try ctx.prewarm()
        // token_embd is gather-only because the lm_head is untied here; a TIED
        // one is walked end to end every token, where no readahead is wrong.
        if let ple = model.perLayerEmbd { model.gguf.gathered(ple) }
        if model.output.name != model.tokEmbd.name {
            model.gguf.gathered(model.tokEmbd)
        }
        let c = cfg
        // Scratch is sized to the WIDEST layer; the kv width is maximised over
        // the layers, since head width and head count vary in opposite ways.
        let maxHd = max(c.headDimSliding, c.headDimFull)
        let maxFF = (0..<c.nLayer)
            .map { il in model.layers[il].nFF }.max() ?? 0
        let maxKV = (0..<c.nLayer)
            .map { il in c.headDim(il) * model.layers[il].nHeadKV }.max() ?? 0
        // A zero-length MTLBuffer is nil rather than empty.
        let pleWidth = max(1, c.nLayer * c.perLayerDim)
        let pleDim = max(1, c.perLayerDim)
        bx = ctx.makeF32(c.nEmbd)
        bNormed = ctx.makeF32(c.nEmbd)
        bContrib = ctx.makeF32(c.nEmbd)
        bQ = ctx.makeF32(maxHd * c.nHead)
        bK = ctx.makeF32(maxKV)
        bV = ctx.makeF32(maxKV)
        bAttnOut = ctx.makeF32(maxHd * c.nHead)
        bGateNull = ctx.makeF32(maxHd * c.nHead)
        bFfnGate = ctx.makeF32(maxFF)
        bFfnUp = ctx.makeF32(maxFF)
        bPle = ctx.makeF32(pleWidth)
        bPleProj = ctx.makeF32(pleWidth)
        bPleGate = ctx.makeF32(pleDim)
        bLogits = ctx.makeF32(c.nVocab)
        bClamp = ctx.makeF32(max(maxFF, max(pleWidth, c.nEmbd)))
        // A blockwise-vision checkpoint must fit its widest image in ONE chunk:
        // a block that does not fit cannot be attended correctly at all.
        let widest = cfg.blockwiseVision
            ? (model.gguf.int("gemma4.vision.max_soft_tokens") ?? 0) : 0
        let B = ctx.matrixUnits
            ? max(Gemma4MetalEngine.defaultBatch, widest) : 1
        capacity = B
        batchStore = B
        bXN = ctx.makeF32(B * c.nEmbd)
        bNormedN = ctx.makeF32(B * c.nEmbd)
        bContribN = ctx.makeF32(B * c.nEmbd)
        bQN = ctx.makeF32(B * maxHd * c.nHead)
        bKN = ctx.makeF32(B * maxKV)
        bVN = ctx.makeF32(B * maxKV)
        bAttnOutN = ctx.makeF32(B * maxHd * c.nHead)
        bGateNullN = ctx.makeF32(B * maxHd * c.nHead)
        bFfnGateN = ctx.makeF32(B * maxFF)
        bFfnUpN = ctx.makeF32(B * maxFF)
        bPleN = ctx.makeF32(B * pleWidth)
        bPleProjN = ctx.makeF32(B * pleWidth)
        bPleGateN = ctx.makeF32(B * pleDim)
        bClampN = ctx.makeF32(B * max(maxFF, max(pleWidth, c.nEmbd)))
        bBlocks = ctx.makeU32(2 * (B + 8))
        let pages = MetalKVPool.pagesFor(model.gguf, P: pageP)
        for il in 0..<c.nLayer where !c.isShared(il) {
            let source = (0..<c.nLayer).contains { j in
                c.isShared(j) && c.sharedSource(j) == il
            }
            let pool = MetalKVPool(
                device: ctx.device, P: pageP,
                kvDim: c.headDim(il) * model.layers[il].nHeadKV,
                window: c.isFull(il) || source ? nil : c.slidingWindow,
                capacity: pages)
            pool.starved = { [stopSignal] in stopSignal.raise() }
            kv[il] = pool
        }
        let g = model.gguf
        for L in model.layers {
            for w in [L.wq, L.wo, L.wk, L.wv, L.ffnGate, L.ffnUp, L.ffnDown,
                      L.perLayerGate, L.perLayerProj] {
                if let w { scales[w.name] = SRQ(g, w.name) }
            }
        }
        scales[model.output.name] = SRQ(g, model.output.name)
        // The BF16 norm kernels read two bytes per weight, so a norm re-emitted
        // as f32 would fuse two weights into one garbage float. Fail here.
        let context = ctx
        func normOff(_ name: String) -> WeightRef {
            let t = g.tensor(name)
            precondition(t.type == .bf16,
                         "gemma norm \(name) is \(t.type), expected bf16")
            return context.window(UInt64(t.base - g.map))
        }
        func normOffIfPresent(_ name: String) -> WeightRef? {
            g.maybe(name).map { _ in normOff(name) }
        }
        outputNormOff = normOff("output_norm.weight")
        perLayerProjNormOff = normOffIfPresent("per_layer_proj_norm.weight")
        for il in 0..<c.nLayer {
            norms.append(LayerNorms(
                attn: normOff("blk.\(il).attn_norm.weight"),
                postAttn: normOff("blk.\(il).post_attn_norm.weight"),
                ffn: normOff("blk.\(il).ffn_norm.weight"),
                postFfn: normOff("blk.\(il).post_ffn_norm.weight"),
                qNorm: normOff("blk.\(il).attn_q_norm.weight"),
                kNorm: c.isShared(il)
                    ? nil : normOff("blk.\(il).attn_k_norm.weight"),
                perLayerPost: normOffIfPresent(
                    "blk.\(il).per_layer_post_norm.weight")))
        }
        if let head = model.assist {
            assist = Gemma4MetalAssist(model, head, ctx: ctx)
        }
    }

    public var hasAssist: Bool { assist != nil }

    public func drainGPUSeconds() -> Double { ctx.clock.drain() }
    public var gpuSeconds: Double { ctx.clock.elapsed }

    public func drainSpecTurn() -> SpecTurn? {
        var out: SpecTurn? = nil
        if specCycles > 0 {
            out = SpecTurn(cycles: specCycles, committed: specCommitted,
                           drafted: specDrafted, accepted: specAccepted)
        }
        specCycles = 0
        specCommitted = 0
        specDrafted = 0
        specAccepted = 0
        return out
    }

    public func verifyChunk(_ ids: [Int32]) -> [Int32] {
        let c = cfg
        let n = ids.count
        let lp = chunkLogits(ids)
        var out: [Int32] = []
        for j in 0..<n {
            out.append(Int32(Vectors.argmax(lp + j * c.nVocab, c.nVocab)))
        }
        return out
    }

    private func chunkLogits(_ ids: [Int32])
        -> UnsafeMutablePointer<Float> {
        let c = cfg
        let n = ids.count
        forwardChunk(ids, [:],
                     [(Int, Int)](repeating: (0, 0), count: n))
        pos += n
        if bLogitsN == nil
            || bLogitsN!.length < n * c.nVocab * MemoryLayout<Float>.stride {
            bLogitsN = ctx.makeF32(n * c.nVocab)
        }
        let cb = buffer("verify lm_head")
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        f.linear(model.output, X: bNormedN, out: bLogitsN!,
                 off: off(model.output), N: n, srq: srq(model.output),
                 scratch: bClampN)
        if c.logitSoftcap > 0 {
            f.softcap(x: bLogitsN!, n: n * c.nVocab, cap: c.logitSoftcap)
        }
        e.endEncoding()
        commit(cb, "verify lm_head")
        return bLogitsN!.contents().assumingMemoryBound(to: Float.self)
    }

    public func acceptPartial(_ mark: Bookmark, keep: Int) {
        pos = mark.pos + keep
        for (il, n) in mark.lens { kv[il]!.truncate(to: n + keep) }
    }

    public func chunkCost(_ ids: [Int32], from first: Int,
                          want: (Int, Int32, UnsafePointer<Float>) -> Void) {
        let n = ids.count
        let lo = max(first - 1, 0)
        reset()
        var i = 0
        while i < n {
            let take = min(batch, n - i)
            let rows = Array(ids[i..<(i + take)])
            if take == 1 {
                forward(token: Int(rows[0]), pos: pos)
                pos += 1
            } else {
                forwardChunk(rows, [:],
                             [(Int, Int)](repeating: (0, 0), count: take))
                pos += take
            }
            for j in 0..<take where i + j >= lo && i + j < n - 1 {
                if take > 1 { selectRow(j, of: take) }
                let lg = logits()
                lg.withUnsafeBufferPointer { p in
                    want(i + j, ids[i + j + 1], p.baseAddress!)
                }
            }
            i += take
        }
    }

    public func selectRow(_ j: Int, of n: Int) {
        let c = cfg
        let rows = bNormedN.f32(n * c.nEmbd)
        let dst = bNormed.f32(c.nEmbd)
        for i in 0..<c.nEmbd { dst[i] = rows[j * c.nEmbd + i] }
    }

    public func assistDraft(_ token: Int32, count: Int) -> [Int32] {
        var out: [Int32] = []
        if let head = assist {
            var last = token
            var back = hidden()
            let at = max(0, pos - 1)
            for _ in 0..<count {
                back.withUnsafeBufferPointer { h in
                    head.seed(Int(last), hidden: h.baseAddress!)
                }
                encode("assist draft") { f in
                    head.encodeStep(f, pos: at, pools: kv)
                }
                last = head.picked()
                back = head.backbone()
                out.append(last)
            }
        }
        return out
    }

    private var floor = 0

    public func retain(from position: Int) { floor = position }

    private func evictAll() {
        for (_, pool) in kv { pool.evict(floor: floor) }
    }

    public func reset() {
        pos = 0
        floor = 0
        stopSignal.clear()
        specFlush()
        let pages = kv.values.reduce(0) { sum, pool in sum + pool.pageCount }
        let bytes = kv.values.reduce(0) { sum, pool in
            sum + pool.pageBytesTotal
        }
        for (_, pool) in kv { pool.truncate(to: 0) }
        Diag.memory?("engine reset: dropped \(pages) kv pages, "
                     + "\(bytes / 1_048_576) MB")
    }

    private func buffer(_ tag: String) -> MTLCommandBuffer {
        let made = ctx.queue.makeCommandBuffer()
        if made == nil {
            Diag.shared.report("[metal] no command buffer: \(tag)")
        }
        return made!
    }

    private func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - map))
    }

    private func srq(_ w: GGUFTensor) -> SRQ { scales[w.name] ?? SRQ.none }

    public func extend(_ ids: [Int32]) -> Int32 {
        extend(ids, softAt: { _ in nil })
    }

    // `softAt` is asked EXACTLY ONCE per id and in order: a caller's feed is a
    // cursor.
    public func extend(_ ids: [Int32],
                       softAt: (Int32) -> [Float]?) -> Int32 {
        let B = batch
        stopSignal.clear()
        specFlush()
        Diag.memory?("prefill start \(ids.count) ids at pos \(pos)")
        let blocks = cfg.blockwiseVision
            ? cfg.visionBlocks(ids, from: pos)
            : [(Int, Int)](repeating: (0, 0), count: ids.count)
        var i = 0
        // The stop is the loop PREDICATE, so a Stop lands between chunks and
        // leaves `pos` and the KV consistent for the rollback.
        while i < ids.count && !stopSignal.raisedNow {
            let n = B > 1 ? chunk(blocks, at: i, want: min(B, ids.count - i))
                          : 1
            var soft: [Int: [Float]] = [:]
            for j in 0..<n {
                if let feature = softAt(ids[i + j]) { soft[j] = feature }
            }
            if n == 1 {
                if let feature = soft[0] {
                    forward(embedding: feature, pos: pos)
                } else {
                    forward(token: Int(ids[i]), pos: pos)
                }
            } else {
                forwardChunk(Array(ids[i ..< (i + n)]), soft,
                             Array(blocks[i ..< (i + n)]))
            }
            pos += n
            i += n
        }
        Diag.memory?("prefill done at pos \(pos)")
        return pick(logits())
    }

    private func chunk(_ blocks: [(Int, Int)], at i: Int, want: Int) -> Int {
        let out = Gemma4Config.chunkLength(blocks, at: i, want: want)
        precondition(out <= capacity,
                     "a \(out)-token vision block must ride one chunk and "
                     + "this engine allocated \(capacity); raise "
                     + "--gemma-batch or lower the image budget")
        return out
    }

    public func setSpeculation(_ on: Bool) {
        plainDecode = !on
    }

    public func decode(_ token: Int32) -> Int32 {
        let ready = assist != nil && Gemma4MetalEngine.specN > 1
            && !plainDecode && capacity >= Gemma4MetalEngine.specN
            && sampler?.logitMask == nil
        var out: Int32
        if !specQueue.isEmpty {
            out = specQueue.removeFirst()
        } else if ready {
            specQueue = specCycle(token)
            out = specQueue.removeFirst()
        } else {
            out = decodePlain(token)
        }
        return out
    }

    public func decodePlain(_ token: Int32) -> Int32 {
        forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits())
    }

    private func specFlush() { specQueue.removeAll() }

    public var queued: Int { max(specQueue.count - 1, 0) }

    private func pickRow(_ row: UnsafePointer<Float>, gpu: Int32) -> Int32 {
        var out = gpu
        if sampler != nil {
            out = pick(Array(UnsafeBufferPointer(start: row,
                                                 count: cfg.nVocab)))
        }
        return out
    }

    private func specCycle(_ token: Int32) -> [Int32] {
        let c = cfg
        let p0 = pos
        let drafts = assistDraft(token, count: Gemma4MetalEngine.specN - 1)
        var fed: [Int32] = [token]
        fed.append(contentsOf: drafts)
        let width = fed.count
        let lp = chunkLogits(fed)
        var accepted = 0
        var bonus: Int32 = 0
        var scanning = true
        while scanning {
            let row = lp + accepted * c.nVocab
            let r = pickRow(row, gpu: Int32(Vectors.argmax(row, c.nVocab)))
            if accepted < drafts.count && r == drafts[accepted] {
                accepted += 1
            } else {
                bonus = r
                scanning = false
            }
        }
        let m = accepted + 1
        if m < width { for (_, pool) in kv { pool.truncate(to: p0 + m) } }
        pos = p0 + m
        selectRow(m - 1, of: width)
        specCycles += 1
        specCommitted += m
        specDrafted += drafts.count
        specAccepted += accepted
        var out = Array(drafts.prefix(accepted))
        out.append(bonus)
        return out
    }

    func pick(_ values: [Float]) -> Int32 {
        var out: Int32
        if sampler != nil {
            var work = values
            out = sampler!.sample(&work)
            sampler!.accept(out)
        } else {
            out = Int32(Vectors.argmax(values))
        }
        return out
    }

    public func forward(token: Int, pos: Int) {
        seedToken(token)
        encode("forward pos=\(pos)") { f in
            if cfg.hasPerLayerInputs { buildPLE(f) }
            layers(f, pos: pos)
        }
        evictAll()
    }

    private func seedToken(_ token: Int) {
        let c = cfg
        gatherEmbed(token, into: bx.f32(c.nEmbd).baseAddress!)
        if c.hasPerLayerInputs {
            gatherPLE(token,
                      into: bPle.f32(c.nLayer * c.perLayerDim).baseAddress!)
        }
    }

    private func encode(_ tag: String, _ body: (MetalEnc) -> Void) {
        let cb = buffer(tag)
        let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        body(MetalEnc(ctx: ctx, e: e, concurrent: true))
        e.endEncoding()
        commit(cb, tag)
    }

    // Row-gathered off the mmap and never bound: a bound bytesNoCopy window is
    // WIRED into kernel_task, and per_layer_token_embd is 1260 of 2541 MB.
    private func gatherEmbed(_ token: Int,
                             into dst: UnsafeMutablePointer<Float>) {
        let c = cfg
        GQ.gather(model.tokEmbd, row: token, from: 0, count: c.nEmbd,
                  into: dst)
        for i in 0..<c.nEmbd { dst[i] *= c.embedScale }
    }

    private func gatherPLE(_ token: Int,
                           into dst: UnsafeMutablePointer<Float>) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        GQ.gather(model.perLayerEmbd!, row: token, from: 0, count: width,
                  into: dst)
        for i in 0..<width { dst[i] *= c.perLayerEmbedScale }
    }

    // A soft row gathers the PAD row of the per-layer table (HF rewrites every
    // multimodal position to pad_token_id) and enters UNSCALED.
    public func forward(embedding: [Float], pos: Int) {
        seedEmbedding(embedding)
        encode("soft forward pos=\(pos)") { f in
            if cfg.hasPerLayerInputs { buildPLE(f) }
            layers(f, pos: pos)
        }
        evictAll()
    }

    private func seedEmbedding(_ embedding: [Float]) {
        let c = cfg
        precondition(embedding.count == c.nEmbd,
                     "soft token is \(embedding.count) wide, expected "
                     + "\(c.nEmbd)")
        let dst = bx.f32(c.nEmbd)
        for i in 0..<c.nEmbd { dst[i] = embedding[i] }
        if c.hasPerLayerInputs {
            gatherPLE(c.padTokenId,
                      into: bPle.f32(c.nLayer * c.perLayerDim).baseAddress!)
        }
    }

    // DEBUG ONLY: a command buffer per LAYER, far slower than the shipping
    // path.

    public func forward(token: Int, pos: Int,
                        tap: (String, Int, [Float]) -> Void) {
        seedToken(token)
        tapped(pos: pos, tap: tap)
    }

    public func forward(embedding: [Float], pos: Int,
                        tap: (String, Int, [Float]) -> Void) {
        seedEmbedding(embedding)
        tapped(pos: pos, tap: tap)
    }

    private func tapped(pos: Int, tap: (String, Int, [Float]) -> Void) {
        let c = cfg
        tap("embed", -1, Array(bx.f32(c.nEmbd)))
        encode("tap ple") { f in
            if c.hasPerLayerInputs { buildPLE(f) }
            f.rmsnormBF16(x: bx, weightOff: norms[0].attn, out: bNormed,
                          n: c.nEmbd, eps: c.eps)
        }
        for il in 0..<c.nLayer {
            encode("tap attn \(il)") { f in layerAttn(f, il, pos: pos) }
            tap("attn", il, Array(bContrib.f32(c.nEmbd)))
            encode("tap ffn \(il)") { f in layerFfn(f, il) }
            tap("mlp", il, Array(bContrib.f32(c.nEmbd)))
            encode("tap tail \(il)") { f in layerTail(f, il) }
            tap("l_out", il, Array(bx.f32(c.nEmbd)))
        }
        tap("final", -1, hidden())
    }


    // `pos` is the LIVE position, never zero: a continuation prefills onto an
    // existing KV, so every absolute index below is basePos-relative.
    private func forwardChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                              _ blocks: [(Int, Int)]) {
        let c = cfg
        let n = ids.count
        precondition(n <= capacity,
                     "chunk of \(n) exceeds the \(capacity) allocated")
        let basePos = pos
        let hidden = bXN.f32(n * c.nEmbd)
        for (row, feature) in soft {
            precondition(feature.count == c.nEmbd,
                         "soft token is \(feature.count) wide, expected "
                         + "\(c.nEmbd)")
            for i in 0..<c.nEmbd { hidden[row * c.nEmbd + i] = feature[i] }
        }
        // Every non-shared layer reserves the same N positions BEFORE the
        // kernels run: appendBatch allocates pages and refreshes the table.
        embedChunk(ids, soft, n)
        blockChunk(blocks, n)
        if c.hasPerLayerInputs { gatherPLEChunk(ids, soft, n) }
        for (_, pool) in kv { pool.appendBatch(n) }
        encodeChunk(basePos: basePos, n: n)
        evictAll()
        let src = bNormedN.f32(n * c.nEmbd)
        let dst = bNormed.f32(c.nEmbd)
        for i in 0..<c.nEmbd { dst[i] = src[(n - 1) * c.nEmbd + i] }
    }

    // Soft rows already hold a tower feature and must NOT be scaled.
    private func embedChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                            _ n: Int) {
        let c = cfg
        let dst = bXN.f32(n * c.nEmbd).baseAddress!
        for j in 0..<n where soft[j] == nil {
            gatherEmbed(Int(ids[j]), into: dst + j * c.nEmbd)
        }
    }

    // The padding rows the matrix attention reads past n are cleared to an
    // empty range, so a tail tile's dead rows cannot widen the key sweep.
    private func blockChunk(_ blocks: [(Int, Int)], _ n: Int) {
        let dst = bBlocks.u32(2 * (n + 8))
        for j in 0..<n {
            dst[2 * j] = UInt32(blocks[j].0)
            dst[2 * j + 1] = UInt32(blocks[j].1)
        }
        for j in n..<(n + 8) {
            dst[2 * j] = 0
            dst[2 * j + 1] = 0
        }
    }

    private func gatherPLEChunk(_ ids: [Int32], _ soft: [Int: [Float]],
                                _ n: Int) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        let dst = bPleN.f32(n * width).baseAddress!
        for j in 0..<n {
            gatherPLE(soft[j] == nil ? Int(ids[j]) : c.padTokenId,
                      into: dst + j * width)
        }
    }

    private func buildPLEChunk(_ f: MetalEnc, _ n: Int) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        f.gemm(model.perLayerModelProj!, X: bXN, out: bPleProjN,
               off: off(model.perLayerModelProj!), N: n)
        f.scaleInPlace(x: bPleProjN, n: n * width,
                       s: 1 / Float(c.nEmbd).squareRoot())
        f.rmsnormRowsBF16(x: bPleProjN, xoff: 0, d: c.perLayerDim,
                          rows: n * c.nLayer,
                          weightOff: perLayerProjNormOff!, eps: c.eps)
        f.addScaled(a: bPleN, b: bPleProjN, n: n * width,
                    s: 1 / Float(2).squareRoot())
    }

    // Groups are committed WITHOUT waiting between them: one queue runs them in
    // commit order, and an error is read off every buffer, not just the last.
    private func encodeChunk(basePos: Int, n: Int) {
        let c = cfg
        let per = min(Gemma4MetalEngine.prefillLayers, c.nLayer)
        BackgroundGate.shared.waitForForeground()
        var queued: [MTLCommandBuffer] = []
        var il = 0
        while il < c.nLayer {
            let end = min(il + per, c.nLayer)
            let cb = buffer("prefill chunk n=\(n) pos=\(basePos) "
                            + "layers \(il)..<\(end)")
            let e = cb.makeComputeCommandEncoder()!
            let f = MetalEnc(ctx: ctx, e: e)
            if il == 0 && c.hasPerLayerInputs { buildPLEChunk(f, n) }
            layersChunk(f, basePos: basePos, n: n, layers: il..<end)
            e.endEncoding()
            cb.commit()
            queued.append(cb)
            if let onLayer {
                cb.waitUntilCompleted()
                emit(onLayer, end - 1, bXN, basePos: basePos, n: n)
                if end == c.nLayer {
                    emit(onLayer, c.nLayer, bNormedN, basePos: basePos, n: n)
                }
            }
            il = end
        }
        queued[queued.count - 1].waitUntilCompleted()
        for cb in queued { ctx.clock.add(cb) }
        for (_, pool) in kv { pool.touch() }
        let fault = queued.compactMap { cb in cb.error }.first
        if let fault {
            let why = "prefill chunk n=\(n) pos=\(basePos): \(fault)"
            Diag.shared.report("[metal] \(why)")
            fatalError("metal \(why)")
        }
        if Gemma4MetalEngine.timing {
            let gpu = queued.reduce(0.0) { total, cb in
                total + (cb.gpuEndTime - cb.gpuStartTime) * 1000
            }
            FileHandle.standardError.write(Data(
                "chunk n=\(n) pos=\(basePos) gpu=\(Int(gpu))ms\n".utf8))
        }
        Diag.memoryDetail?("prefill chunk n=\(n) pos=\(basePos)")
    }

    private func emit(_ tap: (Int, Int, [Float]) -> Void, _ il: Int,
                      _ buffer: MTLBuffer, basePos: Int, n: Int) {
        let c = cfg
        let rows = buffer.f32(n * c.nEmbd)
        for j in 0..<n where probe.contains(basePos + j) {
            let lo = j * c.nEmbd
            tap(il, basePos + j, Array(rows[lo..<(lo + c.nEmbd)]))
        }
    }

    private func layersChunk(_ f: MetalEnc, basePos: Int, n: Int,
                             layers: Range<Int>) {
        let c = cfg
        let width = c.nLayer * c.perLayerDim
        for il in layers {
            let L = model.layers[il]
            let nm = norms[il]
            if il == 0 {
                f.rmsnormBatchBF16(x: bXN, weightOff: nm.attn, y: bNormedN,
                                   n: c.nEmbd, rows: n, eps: c.eps)
            }
            attentionChunk(f, L, il, basePos: basePos, n: n)
            f.normAddNormBF16(x: bXN, c: bContribN, post: nm.postAttn,
                              pre: nm.ffn, y: bNormedN, n: c.nEmbd, rows: n,
                              eps: c.eps, s: 1)
            if !Gemma4MetalEngine.skip.contains("ffn") {
                f.linear(L.ffnGate, X: bNormedN, out: bFfnGateN,
                         off: off(L.ffnGate), N: n, srq: srq(L.ffnGate),
                         scratch: bClampN)
                f.linear(L.ffnUp, X: bNormedN, out: bFfnUpN, off: off(L.ffnUp),
                         N: n, srq: srq(L.ffnUp), scratch: bClampN)
                f.activateMul(c.activation, a: bFfnGateN, b: bFfnUpN,
                              n: n * L.nFF)
                f.linear(L.ffnDown, X: bFfnGateN, out: bContribN,
                         off: off(L.ffnDown), N: n, srq: srq(L.ffnDown),
                         scratch: bClampN)
            }
            if c.hasPerLayerInputs {
                f.normAddNormBF16(x: bXN, c: bContribN, post: nm.postFfn,
                                  pre: nil, y: bNormedN, n: c.nEmbd, rows: n,
                                  eps: c.eps, s: 1)
                f.linear(L.perLayerGate!, X: bXN, out: bPleGateN,
                         off: off(L.perLayerGate!), N: n,
                         srq: srq(L.perLayerGate!), scratch: bClampN)
                // Each row multiplies its OWN table row at this layer's
                // offset, so the two operands have different strides.
                f.activateMulRows(c.activation, a: bPleGateN, b: bPleN,
                                  n: c.perLayerDim, rows: n,
                                  aStride: c.perLayerDim, bStride: width,
                                  bOff: il * c.perLayerDim)
                f.linear(L.perLayerProj!, X: bPleGateN, out: bContribN,
                         off: off(L.perLayerProj!), N: n,
                         srq: srq(L.perLayerProj!), scratch: bClampN)
                f.normAddNormBF16(x: bXN, c: bContribN,
                                  post: nm.perLayerPost!, pre: nextNorm(il),
                                  y: bNormedN, n: c.nEmbd, rows: n,
                                  eps: c.eps, s: L.layerScalar)
            } else {
                f.normAddNormBF16(x: bXN, c: bContribN, post: nm.postFfn,
                                  pre: nextNorm(il), y: bNormedN, n: c.nEmbd,
                                  rows: n, eps: c.eps, s: L.layerScalar)
            }
        }
    }

    private func nextNorm(_ il: Int) -> WeightRef {
        il + 1 < cfg.nLayer ? norms[il + 1].attn : outputNormOff
    }

    private func attentionChunk(_ f: MetalEnc, _ L: Gemma4Layer, _ il: Int,
                                basePos: Int, n: Int) {
        let c = cfg
        let hd = c.headDim(il)
        let full = c.isFull(il)
        let base = full ? c.ropeBaseFull : c.ropeBaseSliding
        let rot = full ? c.rotatedPairsFull : c.rotatedPairsSliding
        f.linear(L.wq, X: bNormedN, out: bQN, off: off(L.wq), N: n,
                 srq: srq(L.wq), scratch: bClampN)
        f.rmsnormRowsBF16(x: bQN, xoff: 0, d: hd, rows: n * c.nHead,
                          weightOff: norms[il].qNorm, eps: c.eps)
        f.ropeGemmaBatch(x: bQN, headDim: hd, nHead: c.nHead, rotated: rot,
                         base: base, basePos: basePos, N: n)
        let nKV = L.nHeadKV
        let pool = kv[c.isShared(il) ? c.sharedSource(il) : il]!
        if !c.isShared(il) {
            f.linear(L.wk!, X: bNormedN, out: bKN, off: off(L.wk!), N: n,
                     srq: srq(L.wk!), scratch: bClampN)
            f.rmsnormRowsBF16(x: bKN, xoff: 0, d: hd, rows: n * nKV,
                              weightOff: norms[il].kNorm!, eps: c.eps)
            f.ropeGemmaBatch(x: bKN, headDim: hd, nHead: nKV,
                             rotated: rot, base: base, basePos: basePos, N: n)
            // A layer with no value projection takes bNormedN, not bKN: by then
            // bKN is a normed and rotated key, and the value wants neither.
            let wv = L.wv ?? L.wk!
            f.linear(wv, X: bNormedN, out: bVN, off: off(wv), N: n,
                     srq: srq(wv), scratch: bClampN)
            f.rmsnormRowsNoWeight(x: bVN, xoff: 0, d: hd,
                                  rows: n * nKV, eps: c.eps)
            f.kvAppendBatch(kCurN: bKN, vCurN: bVN, kAddr: pool.kAddr,
                            vAddr: pool.vAddr,
                            pages: pool.pages(rows: basePos, basePos + n - 1),
                            kvDim: hd * nKV, basePos: basePos,
                            P: pool.P, N: n)
        }
        // Scale 1: q_norm is the query's only normalization. The window is per
        // ROW.
        let lo = full ? 0 : max(0, basePos - c.slidingWindow + 1)
        if !Gemma4MetalEngine.skip.contains("attn") {
            f.attnBatch(qN: bQN, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.pages(rows: lo, basePos + n - 1),
                        gateN: bGateNullN,
                        outN: bAttnOutN, hd: hd, nH: c.nHead, nKV: nKV,
                        kvDim: hd * nKV, P: pool.P, scale: 1,
                        basePos: basePos, N: n, gated: 0,
                        window: full ? 0 : c.slidingWindow,
                        // BOTH layer types carry the blocks, as HF does;
                        // sliding-only breaks L5.
                        blocks: bBlocks)
        }
        f.linear(L.wo, X: bAttnOutN, out: bContribN, off: off(L.wo), N: n,
                 srq: srq(L.wo), scratch: bClampN)
    }

    private func layers(_ f: MetalEnc, pos: Int) {
        let c = cfg
        f.rmsnormBF16(x: bx, weightOff: norms[0].attn, out: bNormed,
                      n: c.nEmbd, eps: c.eps)
        for il in 0..<c.nLayer { layer(f, il, pos: pos) }
    }

    private func layer(_ f: MetalEnc, _ il: Int, pos: Int) {
        layerAttn(f, il, pos: pos)
        layerFfn(f, il)
        layerTail(f, il)
    }

    private func layerAttn(_ f: MetalEnc, _ il: Int, pos: Int) {
        attention(f, model.layers[il], il, pos: pos)
    }

    private func layerFfn(_ f: MetalEnc, _ il: Int) {
        let c = cfg
        let L = model.layers[il]
        let nm = norms[il]
        f.normAddNormBF16(x: bx, c: bContrib, post: nm.postAttn, pre: nm.ffn,
                          y: bNormed, n: c.nEmbd, rows: 1, eps: c.eps, s: 1)
        if !Gemma4MetalEngine.skip.contains("ffn") {
            f.parallel {
                f.linear(L.ffnGate, x: bNormed, out: bFfnGate,
                         off: off(L.ffnGate), srq: srq(L.ffnGate),
                         scratch: bClamp)
                f.linear(L.ffnUp, x: bNormed, out: bFfnUp,
                         off: off(L.ffnUp), srq: srq(L.ffnUp),
                         scratch: bClamp)
            }
            f.activateMul(c.activation, a: bFfnGate, b: bFfnUp, n: L.nFF)
            f.linear(L.ffnDown, x: bFfnGate, out: bContrib,
                     off: off(L.ffnDown), srq: srq(L.ffnDown),
                     scratch: bClamp)
        }
    }

    private func layerTail(_ f: MetalEnc, _ il: Int) {
        let c = cfg
        let L = model.layers[il]
        let nm = norms[il]
        if c.hasPerLayerInputs {
            f.normAddNormBF16(x: bx, c: bContrib, post: nm.postFfn, pre: nil,
                              y: bNormed, n: c.nEmbd, rows: 1, eps: c.eps,
                              s: 1)
            f.linear(L.perLayerGate!, x: bx, out: bPleGate,
                     off: off(L.perLayerGate!),
                     srq: srq(L.perLayerGate!), scratch: bClamp)
            f.activateMul(c.activation, a: bPleGate, b: bPle,
                          n: c.perLayerDim, bOff: il * c.perLayerDim)
            f.linear(L.perLayerProj!, x: bPleGate, out: bContrib,
                     off: off(L.perLayerProj!),
                     srq: srq(L.perLayerProj!), scratch: bClamp)
            f.normAddNormBF16(x: bx, c: bContrib, post: nm.perLayerPost!,
                              pre: nextNorm(il), y: bNormed, n: c.nEmbd,
                              rows: 1, eps: c.eps, s: L.layerScalar)
        } else {
            f.normAddNormBF16(x: bx, c: bContrib, post: nm.postFfn,
                              pre: nextNorm(il), y: bNormed, n: c.nEmbd,
                              rows: 1, eps: c.eps, s: L.layerScalar)
        }
    }

    // The gathered row is only the token-identity half; the projection is the
    // context half, averaged by 1/sqrt(2). The gather alone looks plausible.
    private func buildPLE(_ f: MetalEnc) {
        let c = cfg
        let n = c.nLayer * c.perLayerDim
        f.gemv(model.perLayerModelProj!, x: bx, out: bPleProj,
               off: off(model.perLayerModelProj!))
        f.scaleInPlace(x: bPleProj, n: n,
                       s: 1 / Float(c.nEmbd).squareRoot())
        f.rmsnormRowsBF16(x: bPleProj, xoff: 0, d: c.perLayerDim,
                          rows: c.nLayer,
                          weightOff: perLayerProjNormOff!, eps: c.eps)
        f.addScaled(a: bPle, b: bPleProj, n: n,
                    s: 1 / Float(2).squareRoot())
    }

    private func attention(_ f: MetalEnc, _ L: Gemma4Layer, _ il: Int,
                           pos: Int) {
        let c = cfg
        let hd = c.headDim(il)
        let full = c.isFull(il)
        let base = full ? c.ropeBaseFull : c.ropeBaseSliding
        let rot = full ? c.rotatedPairsFull : c.rotatedPairsSliding
        let nKV = L.nHeadKV
        let pool = kv[c.isShared(il) ? c.sharedSource(il) : il]!
        let own = !c.isShared(il)
        let wv = L.wv ?? L.wk
        f.parallel {
            f.linear(L.wq, x: bNormed, out: bQ, off: off(L.wq),
                     srq: srq(L.wq), scratch: bClamp)
            if own {
                f.linear(L.wk!, x: bNormed, out: bK, off: off(L.wk!),
                         srq: srq(L.wk!), scratch: bClamp)
                f.linear(wv!, x: bNormed, out: bV, off: off(wv!),
                         srq: srq(wv!), scratch: bClamp)
            }
        }
        f.parallel {
            f.rmsnormRowsBF16(x: bQ, xoff: 0, d: hd, rows: c.nHead,
                              weightOff: norms[il].qNorm, eps: c.eps)
            if own {
                f.rmsnormRowsBF16(x: bK, xoff: 0, d: hd, rows: nKV,
                                  weightOff: norms[il].kNorm!, eps: c.eps)
                f.rmsnormRowsNoWeight(x: bV, xoff: 0, d: hd, rows: nKV,
                                      eps: c.eps)
            }
        }
        f.parallel {
            f.ropeGemma(x: bQ, headDim: hd, nHead: c.nHead, rotated: rot,
                        base: base, pos: pos)
            if own {
                f.ropeGemma(x: bK, headDim: hd, nHead: nKV, rotated: rot,
                            base: base, pos: pos)
            }
        }
        if own {
            let tail = pool.tailForAppend()
            f.kvAppend(kCur: bK, vCur: bV, K: tail.k, V: tail.v,
                       kvDim: hd * nKV, pos: tail.slot)
            pool.commitAppend()
            pool.refreshTable()
        }
        // The window bounds the READ, so no eviction is needed for parity.
        let lo = full ? 0 : max(0, pos - c.slidingWindow + 1)
        if !Gemma4MetalEngine.skip.contains("attn") {
            f.attnPaged(q: bQ, kAddr: pool.kAddr, vAddr: pool.vAddr,
                        pages: pool.pages(rows: lo, pos), gate: bGateNull,
                        out: bAttnOut, hd: hd, nH: c.nHead, nKV: nKV,
                        T: pos + 1, kvDim: hd * nKV, P: pool.P, scale: 1,
                        gated: 0, lo: lo)
        }
        f.linear(L.wo, x: bAttnOut, out: bContrib, off: off(L.wo),
                 srq: srq(L.wo), scratch: bClamp)
    }

    // The final softcap is why nothing in the reference dump exceeds +/-30.
    public func logits() -> [Float] {
        let c = cfg
        let cb = buffer("lm_head")
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        f.linear(model.output, x: bNormed, out: bLogits,
                 off: off(model.output), srq: srq(model.output),
                 scratch: bClamp)
        if c.logitSoftcap > 0 {
            f.softcap(x: bLogits, n: c.nVocab, cap: c.logitSoftcap)
        }
        e.endEncoding()
        commit(cb, "lm_head")
        return Array(bLogits.f32(c.nVocab))
    }

    // Rollback is INDEX-based: pools are pos-major and append-only, so a mark
    // is the position plus each pool's length and a rewind truncates to it.
    public struct Bookmark: Sendable {
        let pos: Int
        let lens: [Int: Int]
    }

    public func bookmark() -> Bookmark {
        var lens: [Int: Int] = [:]
        for (il, pool) in kv { lens[il] = pool.len }
        return Bookmark(pos: pos, lens: lens)
    }

    public func restore(_ b: Bookmark) {
        specFlush()
        var reached = b.pos
        for (il, n) in b.lens {
            let pool = kv[il]!
            pool.truncate(to: min(n, pool.addressableLength))
            reached = min(reached, pool.len)
        }
        pos = kv.isEmpty ? b.pos : reached
    }

    public func attach(_ dir: URL) throws {
        var at = 0
        var rows: [Int: (first: Int, len: Int)] = [:]
        let header = dir.appendingPathComponent("state")
        if let data = try? Data(contentsOf: header, options: .mappedIfSafe) {
            let read: Void? = StateBytes.read(data) { r in
                at = r.int()
                rows = StateBytes.keyed(&r) { r in (first: r.int(), len: r.int()) }
            }
            if read == nil { at = 0; rows = [:] }
        }
        for (il, pool) in kv {
            let slot = rows[il] ?? (first: 0, len: 0)
            try pool.attach(dir.appendingPathComponent("pool.\(il)"),
                            first: slot.first, len: slot.len)
        }
        specFlush()
        stopSignal.clear()
        pos = at
        floor = at
        if let (il, pool) = kv.min(by: { a, b in a.key < b.key }) {
            Diag.memory?("attach pos \(at) pool.\(il) len \(pool.len) "
                         + pool.probe)
        }
    }

    public func export(_ dir: URL) throws {
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        for (il, pool) in kv {
            pool.export(dir.appendingPathComponent("pool.\(il)"))
        }
        try header().write(to: dir.appendingPathComponent("state"),
                           options: .atomic)
    }

    private func header() -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, pos)
        var lens: [Int: Int] = [:]
        for (il, pool) in kv { lens[il] = pool.len }
        StateBytes.putKeyed(&out, lens) { out, il, len in
            StateBytes.putInt(&out, kv[il]!.firstLive(below: len))
            StateBytes.putInt(&out, len)
        }
        return out
    }

    public func flush(_ dir: URL) throws {
        for (_, pool) in kv { pool.writeBack() }
        try header().write(to: dir.appendingPathComponent("state"),
                           options: .atomic)
        if let (il, pool) = kv.min(by: { a, b in a.key < b.key }) {
            Diag.memory?("flush pos \(pos) pool.\(il) len \(pool.len) "
                         + pool.probe)
        }
    }

    public func detach() {
        for (_, pool) in kv { pool.detach() }
        pos = 0
        floor = 0
    }

    public var stateBytes: Int {
        kv.values.reduce(0) { sum, pool in sum + pool.liveBytes }
    }

    public func hidden() -> [Float] { Array(bNormed.f32(cfg.nEmbd)) }

    // NO memory report here: one call per token, and the hook writes to stderr
    // and the log synchronously. The prefill path carries the reports.
    private func commit(_ cb: MTLCommandBuffer, _ tag: String) {
        BackgroundGate.shared.waitForForeground()
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        ctx.clock.add(cb)
        for (_, pool) in kv { pool.touch() }
        if let err = cb.error {
            Diag.shared.report("[metal] \(tag): \(err)")
            fatalError("metal \(tag): \(err)")
        }
        if Gemma4MetalEngine.timing {
            let wall = Date().timeIntervalSince(t0) * 1000
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            FileHandle.standardError.write(Data(
                "\(tag) gpu=\(Int(gpu))ms wall=\(Int(wall))ms\n".utf8))
        }
    }
}

extension Gemma4MetalEngine: TextEngine {}
