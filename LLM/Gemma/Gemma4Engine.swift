import Foundation

// Sliding layers may evict, except the one a shared layer reads from: L13
// must keep an unwindowed history because layers 15+ read past its window.
final class GemmaKV {
    private(set) var k: [[Float]] = []
    private(set) var v: [[Float]] = []
    private(set) var first = 0
    private let keep: Int?

    init(keep: Int?) { self.keep = keep }

    func append(_ kk: [Float], _ vv: [Float]) {
        k.append(kk)
        v.append(vv)
        if let keep, k.count > keep {
            k.removeFirst()
            v.removeFirst()
            first += 1
        }
    }

    var end: Int { first + k.count }

    func reset() {
        k.removeAll()
        v.removeAll()
        first = 0
    }

    struct Snapshot: Sendable {
        let k: [[Float]]
        let v: [[Float]]
        let first: Int
    }
    func snapshot() -> Snapshot { Snapshot(k: k, v: v, first: first) }
    func restore(_ s: Snapshot) {
        k = s.k
        v = s.v
        first = s.first
    }
}

public final class Gemma4Engine {
    let model: Gemma4Model
    public let cfg: Gemma4Config
    var kv: [Int: GemmaKV] = [:]
    public private(set) var pos = 0
    public var sampler: Sampler?
    // The same lock-backed stop the GPU engine carries: this forward is
    // synchronous, so Task cancellation never reaches it.
    private let stopSignal = MetalStopSignal()
    public func requestStop() { stopSignal.raise() }
    public func shouldStop() -> Bool { stopSignal.raisedNow }
    private var pleRow: [Float]
    private let srq: [String: SRQ]

    public init(_ model: Gemma4Model) {
        self.model = model
        cfg = model.cfg
        pleRow = [Float](repeating: 0,
                         count: cfg.nLayer * cfg.perLayerDim)
        var scales: [String: SRQ] = [:]
        for L in model.layers {
            for w in [L.wq, L.wo, L.wk, L.wv, L.ffnGate, L.ffnUp, L.ffnDown,
                      L.perLayerGate, L.perLayerProj] {
                if let w { scales[w.name] = SRQ(model.gguf, w.name) }
            }
        }
        scales[model.output.name] = SRQ(model.gguf, model.output.name)
        srq = scales
        for il in 0..<cfg.nLayer where !cfg.isShared(il) {
            let source = cfg.layerFull.indices.contains(il)
                && (0..<cfg.nLayer).contains { j in
                    cfg.isShared(j) && cfg.sharedSource(j) == il
                }
            let keep = (cfg.isFull(il) || source) ? nil : cfg.slidingWindow
            kv[il] = GemmaKV(keep: keep)
        }
    }

    public func reset() {
        pos = 0
        stopSignal.clear()
        for (_, store) in kv { store.reset() }
    }

    public func extend(_ ids: [Int32]) -> Int32 {
        stopSignal.clear()
        var hidden = [Float]()
        var i = 0
        // Stop as the loop predicate, so it lands on a token boundary.
        while i < ids.count && !stopSignal.raisedNow {
            hidden = forward(token: Int(ids[i]), pos: pos)
            pos += 1
            i += 1
        }
        // A stop before the first token leaves no hidden; the caller throws
        // either way.
        return hidden.isEmpty ? 0 : pick(logits(hidden))
    }

    public func decode(_ token: Int32) -> Int32 {
        let hidden = forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits(hidden))
    }

    public func extend(_ ids: [Int32],
                       softAt: (Int32) -> [Float]?) -> Int32 {
        stopSignal.clear()
        let e = cfg.nEmbd
        let blocks = cfg.blockwiseVision
            ? cfg.visionBlocks(ids, from: pos)
            : [(Int, Int)](repeating: (0, 0), count: ids.count)
        // The per-layer-embedding checkpoints stay on the token path: their PLE
        // is gathered and averaged per token, which the batched layers lack.
        let width = cfg.hasPerLayerInputs ? 1 : Gemma4Engine.batch
        var hidden = [Float]()
        var i = 0
        while i < ids.count && !stopSignal.raisedNow {
            let n = Gemma4Config.chunkLength(
                blocks, at: i, want: min(width, ids.count - i))
            if n == 1 {
                if let feature = softAt(ids[i]) {
                    hidden = forward(embedding: feature, pos: pos)
                } else {
                    hidden = forward(token: Int(ids[i]), pos: pos)
                }
            } else {
                var x = [Float](repeating: 0, count: n * e)
                for j in 0..<n {
                    let id = ids[i + j]
                    // A soft row enters UNSCALED; embed_scale is for text.
                    let row = softAt(id) ?? embed(Int(id))
                    for c in 0..<e { x[j * e + c] = row[c] }
                }
                layersBatch(&x, n: n, basePos: pos,
                            Array(blocks[i..<(i + n)]))
                hidden = GK.rmsnorm(
                    Array(x[((n - 1) * e)..<(n * e)]),
                    model.outputNorm, cfg.eps)
            }
            pos += n
            i += n
        }
        return hidden.isEmpty ? 0 : pick(logits(hidden))
    }

    static let batch = max(1, Flags.int("gemma-batch") ?? 128)

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

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, b.pos)
        StateBytes.putKeyed(&out, b.kv) { out, _, s in
            StateBytes.putInt(&out, s.first)
            StateBytes.putInt(&out, s.k.count)
            for row in s.k { StateBytes.putFloats(&out, row) }
            for row in s.v { StateBytes.putFloats(&out, row) }
        }
        return out
    }

    public func deserialize(_ data: Data) -> Bookmark? {
        StateBytes.read(data) { r in
            let pos = r.int()
            let kv = StateBytes.keyed(&r) { r -> GemmaKV.Snapshot in
                let first = r.int()
                let rows = r.int()
                var k: [[Float]] = [], v: [[Float]] = []
                for _ in 0..<rows { k.append(r.span().array) }
                for _ in 0..<rows { v.append(r.span().array) }
                return GemmaKV.Snapshot(k: k, v: v, first: first)
            }
            return Bookmark(pos: pos, kv: kv)
        }
    }

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let kv: [Int: GemmaKV.Snapshot]
    }

    public func bookmark() -> Bookmark {
        var s: [Int: GemmaKV.Snapshot] = [:]
        for (il, store) in kv { s[il] = store.snapshot() }
        return Bookmark(pos: pos, kv: s)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        for (il, s) in b.kv { kv[il]!.restore(s) }
    }

    // Every quantized projection routes through here, so a site cannot skip
    // the clamp: the shapes are identical either way, only the answer changes.
    private func linear(_ w: GGUFTensor, _ x: [Float],
                        _ out: inout [Float]) {
        let s = srq[w.name] ?? SRQ.none
        var h = x
        SRQ.apply(&h, s.input)
        GQ.matvec(w, x: h, out: &out)
        SRQ.apply(&out, s.output)
    }

    func embed(_ token: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        GQ.dequantSpan(model.tokEmbd, row: token, from: 0,
                       count: cfg.nEmbd, into: &out)
        let s = cfg.embedScale
        for i in 0..<cfg.nEmbd { out[i] *= s }
        return out
    }

    // The gathered table is only the token-identity half; HF also projects the
    // scaled embedding, norms each layer slice and averages by 1/sqrt(2).
    private func buildPLE(_ token: Int, _ embed: [Float]) {
        GQ.dequantSpan(model.perLayerEmbd!, row: token, from: 0,
                       count: pleRow.count, into: &pleRow)
        let s = cfg.perLayerEmbedScale
        for i in 0..<pleRow.count { pleRow[i] *= s }

        var proj = [Float](repeating: 0, count: pleRow.count)
        GQ.matvec(model.perLayerModelProj!, x: embed, out: &proj)
        let ps = 1 / Float(cfg.nEmbd).squareRoot()
        for i in 0..<proj.count { proj[i] *= ps }
        GK.rmsnormRows(&proj, d: cfg.perLayerDim, rows: cfg.nLayer,
                       w: model.perLayerProjNorm, eps: cfg.eps)

        let half = 1 / Float(2).squareRoot()
        for i in 0..<pleRow.count { pleRow[i] = (proj[i] + pleRow[i]) * half }
    }

    private func mlp(_ x: [Float], _ L: Gemma4Layer) -> [Float] {
        var gate = [Float](repeating: 0, count: L.nFF)
        var up = [Float](repeating: 0, count: L.nFF)
        linear(L.ffnGate, x, &gate)
        linear(L.ffnUp, x, &up)
        let act = cfg.activation
        for i in 0..<L.nFF { gate[i] = act.apply(gate[i]) * up[i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.ffnDown, gate, &out)
        return out
    }

    private func perLayer(_ x: [Float], _ L: Gemma4Layer,
                          _ il: Int) -> [Float] {
        let d = cfg.perLayerDim
        var g = [Float](repeating: 0, count: d)
        linear(L.perLayerGate!, x, &g)
        let off = il * d
        let act = cfg.activation
        for i in 0..<d { g[i] = act.apply(g[i]) * pleRow[off + i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.perLayerProj!, g, &out)
        return out
    }

    // A vision block's tokens read FORWARD across the block, so every key must
    // be in the store before any query runs; per-token forward cannot do that.

    private func layersBatch(_ x: inout [Float], n: Int, basePos: Int,
                             _ blocks: [(Int, Int)]) {
        let e = cfg.nEmbd
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            var h = x
            GK.rmsnormRows(&h, d: e, rows: n, w: L.attnNorm, eps: cfg.eps)
            var attn = attentionBatch(h, L, il, basePos: basePos, n: n, blocks)
            GK.rmsnormRows(&attn, d: e, rows: n, w: L.postAttnNorm,
                           eps: cfg.eps)
            for i in 0..<(n * e) { x[i] += attn[i] }

            h = x
            GK.rmsnormRows(&h, d: e, rows: n, w: L.ffnNorm, eps: cfg.eps)
            var ff = mlpBatch(h, L, n)
            GK.rmsnormRows(&ff, d: e, rows: n, w: L.postFfnNorm, eps: cfg.eps)
            for i in 0..<(n * e) { x[i] += ff[i] }

            let s = L.layerScalar
            for i in 0..<(n * e) { x[i] *= s }
        }
    }

    // SRQ is elementwise, so a batch clamps exactly as N single rows would.
    private func linearBatch(_ w: GGUFTensor, _ X: [Float], _ n: Int,
                             _ outDim: Int) -> [Float] {
        let s = srq[w.name] ?? SRQ.none
        var h = X
        SRQ.apply(&h, s.input)
        var out = [Float](repeating: 0, count: n * outDim)
        GQ.matmul(w, X: h, N: n, out: &out)
        SRQ.apply(&out, s.output)
        return out
    }

    private func mlpBatch(_ h: [Float], _ L: Gemma4Layer,
                          _ n: Int) -> [Float] {
        var gate = linearBatch(L.ffnGate, h, n, L.nFF)
        let up = linearBatch(L.ffnUp, h, n, L.nFF)
        let act = cfg.activation
        for i in 0..<(n * L.nFF) { gate[i] = act.apply(gate[i]) * up[i] }
        return linearBatch(L.ffnDown, gate, n, cfg.nEmbd)
    }

    private func attentionBatch(_ h: [Float], _ L: Gemma4Layer, _ il: Int,
                                basePos: Int, n: Int,
                                _ blocks: [(Int, Int)]) -> [Float] {
        let hd = cfg.headDim(il)
        let nH = cfg.nHead
        let nKV = L.nHeadKV
        let full = cfg.isFull(il)
        let base = full ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let rot = full ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding
        var q = linearBatch(L.wq, h, n, hd * nH)
        GK.rmsnormRows(&q, d: hd, rows: n * nH, w: L.qNorm, eps: cfg.eps)
        ropeRows(&q, hd, nH, rot, base, basePos, n)
        let store = kv[cfg.isShared(il) ? cfg.sharedSource(il) : il]!
        if !cfg.isShared(il) {
            var k = linearBatch(L.wk!, h, n, hd * nKV)
            var v = linearBatch(L.wv ?? L.wk!, h, n, hd * nKV)
            GK.rmsnormRows(&k, d: hd, rows: n * nKV, w: L.kNorm!, eps: cfg.eps)
            ropeRows(&k, hd, nKV, rot, base, basePos, n)
            GK.rmsnormRowsNoWeight(&v, d: hd, rows: n * nKV, eps: cfg.eps)
            let width = hd * nKV
            for j in 0..<n {
                store.append(Array(k[(j * width)..<((j + 1) * width)]),
                             Array(v[(j * width)..<((j + 1) * width)]))
            }
        }
        var ctx = [Float](repeating: 0, count: n * hd * nH)
        for j in 0..<n {
            let row = Array(q[(j * hd * nH)..<((j + 1) * hd * nH)])
            let out = attend(row, store, il, pos: basePos + j,
                             block: blocks[j], hd: hd,
                             nH: nH, nKV: nKV)
            for i in 0..<(hd * nH) { ctx[j * hd * nH + i] = out[i] }
        }
        return linearBatch(L.wo, ctx, n, cfg.nEmbd)
    }

    private func ropeRows(_ t: inout [Float], _ hd: Int, _ heads: Int,
                          _ rot: Int, _ base: Float, _ basePos: Int,
                          _ n: Int) {
        let width = hd * heads
        for j in 0..<n {
            var row = Array(t[(j * width)..<((j + 1) * width)])
            GK.rope(&row, headDim: hd, nHead: heads, rotated: rot,
                    base: base, pos: basePos + j)
            for i in 0..<width { t[j * width + i] = row[i] }
        }
    }

    private func attention(_ n: [Float], _ L: Gemma4Layer, _ il: Int,
                           pos: Int, block: (Int, Int) = (0, 0)) -> [Float] {
        let hd = cfg.headDim(il)
        let nH = cfg.nHead
        let nKV = L.nHeadKV
        let full = cfg.isFull(il)
        let base = full ? cfg.ropeBaseFull : cfg.ropeBaseSliding
        let rot = full ? cfg.rotatedPairsFull : cfg.rotatedPairsSliding

        var q = [Float](repeating: 0, count: hd * nH)
        linear(L.wq, n, &q)
        GK.rmsnormRows(&q, d: hd, rows: nH, w: L.qNorm, eps: cfg.eps)
        GK.rope(&q, headDim: hd, nHead: nH, rotated: rot, base: base,
                pos: pos)

        let store = kv[cfg.isShared(il) ? cfg.sharedSource(il) : il]!
        if !cfg.isShared(il) {
            var k = [Float](repeating: 0, count: hd * nKV)
            linear(L.wk!, n, &k)
            // V is projected again from the same input, not copied out of `k`,
            // which keeps k_norm and rope below from reaching it.
            var v = [Float](repeating: 0, count: hd * nKV)
            linear(L.wv ?? L.wk!, n, &v)
            GK.rmsnormRows(&k, d: hd, rows: nKV, w: L.kNorm!, eps: cfg.eps)
            GK.rope(&k, headDim: hd, nHead: nKV, rotated: rot, base: base,
                    pos: pos)
            // v_norm is Gemma4RMSNorm(with_scale=False): a real op that leaves
            // no tensor in the checkpoint, so a tensor scan cannot see it.
            GK.rmsnormRowsNoWeight(&v, d: hd, rows: nKV, eps: cfg.eps)
            store.append(k, v)
        }

        let attnOut = attend(q, store, il, pos: pos, block: block,
                             hd: hd, nH: nH, nKV: nKV)
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        linear(L.wo, attnOut, &out)
        return out
    }

    // The RANGE is the mask: a vision block always CONTAINS its query, so the
    // union with [lo, pos] is contiguous, and the window still bounds it below.
    private func attend(_ q: [Float], _ store: GemmaKV, _ il: Int,
                        pos: Int, block: (Int, Int), hd: Int, nH: Int,
                        nKV: Int) -> [Float] {
        let full = cfg.isFull(il)
        let lo = full ? store.first
                      : max(store.first, pos - cfg.slidingWindow + 1)
        let hi = min(store.end, max(pos + 1, block.1))
        let group = nH / nKV
        var out = [Float](repeating: 0, count: hd * nH)
        var scores = [Float](repeating: 0, count: max(hi - lo, 1))
        for h in 0..<nH {
            let kvh = h / group
            let qh = h * hd
            var t = lo
            while t < hi {
                let row = store.k[t - store.first]
                var s: Float = 0
                for i in 0..<hd { s += q[qh + i] * row[kvh * hd + i] }
                // scaling = 1.0: q_norm is the only normalization, no
                // 1/sqrt(head_dim).
                scores[t - lo] = s
                t += 1
            }
            GK.softmaxInPlace(&scores, hi - lo)
            t = lo
            while t < hi {
                let w = scores[t - lo]
                let row = store.v[t - store.first]
                for i in 0..<hd { out[qh + i] += w * row[kvh * hd + i] }
                t += 1
            }
        }
        return out
    }

    @discardableResult
    public func forward(token: Int, pos: Int,
                        tap: ((String, Int, [Float]) -> Void)? = nil)
        -> [Float] {
        let x = embed(token)
        if cfg.hasPerLayerInputs { buildPLE(token, x) }
        return layers(x, pos: pos, tap: tap)
    }

    // A soft token gathers the PAD row of the per-layer table (HF rewrites
    // every multimodal position to pad_token_id) and enters UNSCALED.
    @discardableResult
    public func forward(embedding: [Float], pos: Int,
                        tap: ((String, Int, [Float]) -> Void)? = nil)
        -> [Float] {
        if cfg.hasPerLayerInputs { buildPLE(cfg.padTokenId, embedding) }
        return layers(embedding, pos: pos, tap: tap)
    }

    private func layers(_ start: [Float], pos: Int,
                        tap: ((String, Int, [Float]) -> Void)?) -> [Float] {
        var x = start
        tap?("embed", -1, x)
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            var r = x
            var h = GK.rmsnorm(x, L.attnNorm, cfg.eps)
            h = attention(h, L, il, pos: pos)
            tap?("attn", il, h)
            h = GK.rmsnorm(h, L.postAttnNorm, cfg.eps)
            for i in 0..<cfg.nEmbd { h[i] += r[i] }

            r = h
            var f = GK.rmsnorm(h, L.ffnNorm, cfg.eps)
            f = mlp(f, L)
            tap?("mlp", il, f)
            f = GK.rmsnorm(f, L.postFfnNorm, cfg.eps)
            for i in 0..<cfg.nEmbd { f[i] += r[i] }

            if cfg.hasPerLayerInputs {
                r = f
                var p = perLayer(f, L, il)
                p = GK.rmsnorm(p, L.perLayerPostNorm, cfg.eps)
                for i in 0..<cfg.nEmbd { p[i] += r[i] }
                f = p
            }

            let s = L.layerScalar
            for i in 0..<cfg.nEmbd { f[i] *= s }
            x = f
            tap?("l_out", il, x)
        }
        let out = GK.rmsnorm(x, model.outputNorm, cfg.eps)
        tap?("final", -1, out)
        return out
    }

    // HF applies tanh(logits / cap) * cap, so nothing in the reference dump
    // exceeds 30.
    public func logits(_ hidden: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nVocab)
        linear(model.output, hidden, &out)
        let cap = cfg.logitSoftcap
        if cap > 0 {
            for i in 0..<out.count { out[i] = tanhf(out[i] / cap) * cap }
        }
        return out
    }
}

extension Gemma4Engine.Bookmark: BackendState {}

extension Gemma4Engine: TextEngine {}
