import Foundation

struct PieceBPE {

    private struct Symbol {
        var start: Int
        var length: Int
        var id: Int32
        var fixed: Bool
    }

    private let index: PieceIndex
    private let byteId: [Int32]
    private let added: [(bytes: [UInt8], id: Int32)]
    private let addedHeads: [Bool]
    let unkId: Int32

    static func from(gguf g: GGUF) -> PieceBPE? {
        PieceIndex.from(gguf: g).map { index in
            PieceBPE(index: index,
                     types: g.ints("tokenizer.ggml.token_type") ?? [],
                     unkId: Int32(g.int("tokenizer.ggml.unknown_token_id")
                                  ?? 3))
        }
    }

    private init(index: PieceIndex, types: [Int], unkId: Int32) {
        self.index = index
        self.unkId = unkId
        byteId = (0..<256).map { b in
            index.id(of: String(format: "<0x%02X>", b)) ?? -1
        }
        var atomic: [(bytes: [UInt8], id: Int32)] = []
        for (id, type) in types.enumerated() where type == 3 || type == 4 {
            atomic.append((index.bytes(of: Int32(id)), Int32(id)))
        }
        atomic.sort { a, b in a.bytes.count > b.bytes.count }
        added = atomic
        var heads = [Bool](repeating: false, count: 256)
        for entry in atomic where !entry.bytes.isEmpty {
            heads[Int(entry.bytes[0])] = true
        }
        addedHeads = heads
    }

    private static let marker: [UInt8] = [0xE2, 0x96, 0x81]

    static func replaced(_ text: String) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(text.utf8.count + 3)
        for b in text.utf8 {
            if b == 0x20 {
                out.append(contentsOf: marker)
            } else {
                out.append(b)
            }
        }
        return out
    }

    private static func matches(_ bytes: [UInt8], at i: Int,
                                _ pattern: [UInt8]) -> Bool {
        var k = 0
        while k < pattern.count && i + k < bytes.count
              && bytes[i + k] == pattern[k] {
            k += 1
        }
        return k == pattern.count
    }

    private func atomic(_ bytes: [UInt8], at i: Int)
        -> (bytes: [UInt8], id: Int32)? {
        var out: (bytes: [UInt8], id: Int32)? = nil
        if addedHeads[Int(bytes[i])] {
            out = added.first { entry in
                PieceBPE.matches(bytes, at: i, entry.bytes)
            }
        }
        return out
    }

    private static func charLength(_ bytes: [UInt8], _ i: Int) -> Int {
        var n = i + 1
        while n < bytes.count && (bytes[n] & 0xC0) == 0x80 { n += 1 }
        return n - i
    }

    private static func words(_ bytes: [UInt8]) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var start = 0
        var i = 0
        while i < bytes.count {
            let n = charLength(bytes, i)
            let atMarker = n == 3 && bytes[i] == marker[0]
                && bytes[i + 1] == marker[1] && bytes[i + 2] == marker[2]
            if atMarker && i > start {
                out.append(start..<i)
                start = i
            }
            i += n
        }
        if start < bytes.count { out.append(start..<bytes.count) }
        return out
    }

    private func symbols(_ p: UnsafeRawPointer, _ bytes: [UInt8],
                         _ word: Range<Int>) -> [Symbol] {
        var out: [Symbol] = []
        var i = word.lowerBound
        while i < word.upperBound {
            let n = PieceBPE.charLength(bytes, i)
            if let piece = index.lookup(p + i, n) {
                out.append(Symbol(start: i, length: n, id: piece.id,
                                  fixed: false))
            } else {
                for k in 0..<n {
                    let id = byteId[Int(bytes[i + k])]
                    out.append(Symbol(start: i + k, length: 1,
                                      id: id >= 0 ? id : unkId, fixed: true))
                }
            }
            i += n
        }
        return out
    }

    private func bestPair(_ p: UnsafeRawPointer, _ syms: [Symbol])
        -> (at: Int, id: Int32)? {
        var best: (at: Int, id: Int32, score: Float)? = nil
        for i in 0..<(syms.count - 1) {
            let a = syms[i], b = syms[i + 1]
            let n = a.length + b.length
            if !a.fixed && !b.fixed && n <= index.maxPieceBytes,
               let piece = index.lookup(p + a.start, n),
               best == nil || piece.score > best!.score {
                best = (i, piece.id, piece.score)
            }
        }
        return best.map { found in (found.at, found.id) }
    }

    private func merged(_ p: UnsafeRawPointer, _ bytes: [UInt8],
                        _ word: Range<Int>) -> [Int32] {
        var syms = symbols(p, bytes, word)
        while syms.count > 1, let pair = bestPair(p, syms) {
            syms[pair.at].length += syms[pair.at + 1].length
            syms[pair.at].id = pair.id
            syms.remove(at: pair.at + 1)
        }
        return syms.map { symbol in symbol.id }
    }

    private func pieces(_ bytes: [UInt8], _ range: Range<Int>) -> [Int32] {
        var ids: [Int32] = []
        if !range.isEmpty {
            var chunk = Array(bytes[range])
            if !chunk.starts(with: PieceBPE.marker) {
                chunk.insert(contentsOf: PieceBPE.marker, at: 0)
            }
            chunk.withUnsafeBytes { raw in
                let p = raw.baseAddress!
                for word in PieceBPE.words(chunk) {
                    ids.append(contentsOf: merged(p, chunk, word))
                }
            }
        }
        return ids
    }

    func encode(_ text: String, limit: Int = Int.max) -> [Int32] {
        let bytes = PieceBPE.replaced(text)
        var ids: [Int32] = []
        var start = 0
        var i = 0
        while i < bytes.count {
            if let hit = atomic(bytes, at: i) {
                ids.append(contentsOf: pieces(bytes, start..<i))
                ids.append(hit.id)
                i += hit.bytes.count
                start = i
            } else {
                i += 1
            }
        }
        ids.append(contentsOf: pieces(bytes, start..<bytes.count))
        if ids.count > limit { ids.removeLast(ids.count - limit) }
        return ids
    }
}

final class ModernBERT {
    struct Layer {
        let attnNorm: [Float]?
        let wqkv: GGUFTensor
        let wo: GGUFTensor
        let ffnNorm: [Float]
        let wUp: GGUFTensor
        let wDown: GGUFTensor
        let window: Int
    }

    let nLayer: Int
    let nEmbd: Int
    let nFF: Int
    let nHead: Int
    let nCtx: Int
    let lnEps: Float
    let ropeBase: Float
    let bosId: Int32
    let eosId: Int32

    private let gguf: GGUF
    private let tokEmb: GGUFTensor
    private let embNorm: [Float]
    private let outNorm: [Float]
    private let layers: [Layer]
    private let bpe: PieceBPE?
    private var rowScratch: [Float]

    var dim: Int { nEmbd }

    static let arch = "modern-bert"

    static func isModernBert(_ g: GGUF) -> Bool {
        g.string("general.architecture") == ModernBERT.arch
    }

    init(gguf g: GGUF) {
        let a = ModernBERT.arch
        nLayer = g.int(a + ".block_count") ?? 13
        nEmbd = g.int(a + ".embedding_length") ?? 384
        nFF = g.int(a + ".feed_forward_length") ?? 1152
        nHead = g.int(a + ".attention.head_count") ?? 6
        nCtx = g.int(a + ".context_length") ?? 8192
        lnEps = Float(g.double(a + ".attention.layer_norm_epsilon") ?? 1e-5)
        ropeBase = Float(g.double(a + ".rope.freq_base") ?? 160000)
        let half = (g.int(a + ".attention.sliding_window") ?? 128) / 2
        let every = g.int(a + ".attention.sliding_window_pattern") ?? 3
        bosId = Int32(g.int("tokenizer.ggml.bos_token_id") ?? 2)
        eosId = Int32(g.int("tokenizer.ggml.eos_token_id") ?? 1)
        bpe = PieceBPE.from(gguf: g)
        gguf = g
        tokEmb = g.tensor("token_embd.weight")
        embNorm = EncoderKernels.dequant(g.tensor("token_embd_norm.weight"))
        outNorm = EncoderKernels.dequant(g.tensor("output_norm.weight"))
        rowScratch = [Float](repeating: 0, count: nEmbd)
        let n = nLayer
        layers = (0..<n).map { il in
            func w(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
            return Layer(
                attnNorm: g.maybe("blk.\(il).attn_norm.weight")
                    .map { t in EncoderKernels.dequant(t) },
                wqkv: w("attn_qkv.weight"), wo: w("attn_output.weight"),
                ffnNorm: EncoderKernels.dequant(w("ffn_norm.weight")),
                wUp: w("ffn_up.weight"), wDown: w("ffn_down.weight"),
                window: il % every == 0 ? 0 : half)
        }
    }

    func tokenize(_ text: String) -> [Int32] {
        var ids: [Int32] = [bosId]
        if let bpe { ids.append(contentsOf: bpe.encode(text, limit: nCtx - 2)) }
        ids.append(eosId)
        return ids
    }

    private func normalized(_ x: [Float], T: Int, _ w: [Float]) -> [Float] {
        var h = x
        for t in 0..<T {
            EncoderKernels.layerNorm(&h, off: t * nEmbd, n: nEmbd,
                                     eps: lnEps, w, nil)
        }
        return h
    }

    private struct Rope {
        let cos: [Float]
        let sin: [Float]
        let half: Int
    }

    private func rope(T: Int) -> Rope {
        let hd = nEmbd / nHead
        let half = hd / 2
        var cos = [Float](repeating: 0, count: T * half)
        var sin = [Float](repeating: 0, count: T * half)
        for d in 0..<half {
            let inv = 1 / powf(ropeBase, Float(2 * d) / Float(hd))
            for t in 0..<T {
                let angle = Float(t) * inv
                cos[t * half + d] = cosf(angle)
                sin[t * half + d] = sinf(angle)
            }
        }
        return Rope(cos: cos, sin: sin, half: half)
    }

    private func rotate(_ x: inout [Float], T: Int, _ r: Rope) {
        let hd = nEmbd / nHead
        for t in 0..<T {
            for h in 0..<nHead {
                let at = t * nEmbd + h * hd
                for d in 0..<r.half {
                    let c = r.cos[t * r.half + d]
                    let s = r.sin[t * r.half + d]
                    let lo = x[at + d]
                    let hi = x[at + d + r.half]
                    x[at + d] = lo * c - hi * s
                    x[at + d + r.half] = hi * c + lo * s
                }
            }
        }
    }

    private func split(_ qkv: [Float], T: Int)
        -> (q: [Float], k: [Float], v: [Float]) {
        let ne = nEmbd
        var q = [Float](repeating: 0, count: T * ne)
        var k = [Float](repeating: 0, count: T * ne)
        var v = [Float](repeating: 0, count: T * ne)
        for t in 0..<T {
            for d in 0..<ne {
                q[t * ne + d] = qkv[t * 3 * ne + d]
                k[t * ne + d] = qkv[t * 3 * ne + ne + d]
                v[t * ne + d] = qkv[t * 3 * ne + 2 * ne + d]
            }
        }
        return (q, k, v)
    }

    private func gated(_ up: [Float], T: Int) -> [Float] {
        var out = [Float](repeating: 0, count: T * nFF)
        let root2: Float = 1.4142135623730951
        for t in 0..<T {
            for i in 0..<nFF {
                let a = up[t * 2 * nFF + i]
                let g = up[t * 2 * nFF + nFF + i]
                out[t * nFF + i] = 0.5 * a * (1 + erff(a / root2)) * g
            }
        }
        return out
    }

    private func encoderLayer(_ L: Layer, _ x: inout [Float], T: Int,
                              _ r: Rope) {
        let ne = nEmbd
        let scale = 1 / Float(ne / nHead).squareRoot()
        let h = L.attnNorm.map { w in normalized(x, T: T, w) } ?? x
        var qkv = [Float](repeating: 0, count: T * 3 * ne)
        EncoderKernels.linearBatch(L.wqkv, nil, h, T: T, inn: ne, out: 3 * ne,
                                   &qkv)
        var (q, k, v) = split(qkv, T: T)
        rotate(&q, T: T, r)
        rotate(&k, T: T, r)
        var ctx = [Float](repeating: 0, count: T * ne)
        EncoderKernels.attention(q, k, v, &ctx, T: T, ne: ne, heads: nHead,
                                 scale: scale, window: L.window)
        var proj = [Float](repeating: 0, count: T * ne)
        EncoderKernels.linearBatch(L.wo, nil, ctx, T: T, inn: ne, out: ne,
                                   &proj)
        for i in 0..<(T * ne) { x[i] += proj[i] }
        let m = normalized(x, T: T, L.ffnNorm)
        var up = [Float](repeating: 0, count: T * 2 * nFF)
        EncoderKernels.linearBatch(L.wUp, nil, m, T: T, inn: ne,
                                   out: 2 * nFF, &up)
        EncoderKernels.linearBatch(L.wDown, nil, gated(up, T: T), T: T,
                                   inn: nFF, out: ne, &proj)
        for i in 0..<(T * ne) { x[i] += proj[i] }
    }

    func encode(ids: [Int32]) -> [Float] {
        let T = ids.count
        let ne = nEmbd
        var x = [Float](repeating: 0, count: T * ne)
        if T > 0 {
            for t in 0..<T {
                EncoderKernels.dequant(tokEmb, Int(ids[t]) * ne, ne,
                                       into: &rowScratch)
                for d in 0..<ne { x[t * ne + d] = rowScratch[d] }
                EncoderKernels.layerNorm(&x, off: t * ne, n: ne, eps: lnEps,
                                         embNorm, nil)
            }
            let r = rope(T: T)
            for l in 0..<nLayer { encoderLayer(layers[l], &x, T: T, r) }
            for t in 0..<T {
                EncoderKernels.layerNorm(&x, off: t * ne, n: ne, eps: lnEps,
                                         outNorm, nil)
            }
        }
        return x
    }

    func encode(_ text: String) -> [Float] {
        encode(ids: tokenize(text))
    }

    func embed(_ text: String) -> [Float] {
        var out = [Float](repeating: 0, count: nEmbd)
        EncoderKernels.pooled(encode(text), width: nEmbd, into: &out)
        return out
    }
}

public final class ModernBertEmbedder: Embedder {

    private let gguf: GGUF
    private let model: ModernBERT
    public static let modelName = "bekko-embedding-v1-a25m"
    public let name = ModernBertEmbedder.modelName

    public var dim: Int { model.dim }
    public var queryPrefix: String { "" }
    public var passagePrefix: String { "" }
    public var relevanceFloor: Float { 0.42 }
    public var reranks: Bool { false }
    public var duplicateAt: Float { 0.64 }
    public var mergeAt: Float { 0.60 }
    public var relateAt: Float { 0.35 }
    public var clusterAt: Float { 0.30 }

    public static func load(ggufPath: String) -> ModernBertEmbedder? {
        var result: ModernBertEmbedder? = nil
        if let gguf = try? GGUF(path: ggufPath), ModernBERT.isModernBert(gguf) {
            result = ModernBertEmbedder(gguf: gguf)
        }
        return result
    }

    private init(gguf: GGUF) {
        self.gguf = gguf
        model = ModernBERT(gguf: gguf)
    }

    public func embed(_ text: String) -> [Float] { model.embed(text) }

    public func states(_ text: String) -> [Float] { model.encode(text) }

    public func pooled(_ states: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: model.dim)
        EncoderKernels.pooled(states, width: model.dim, into: &out)
        return out
    }

    public func tokens(_ text: String) -> [Int32] { model.tokenize(text) }
}
