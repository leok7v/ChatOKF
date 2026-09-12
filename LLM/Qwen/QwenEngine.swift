import Foundation

public final class QwenEngine {
    let model: QwenModel
    let cfg: QwenConfig
    var gdn: [Int: GDNState] = [:]
    var kvc: [Int: KVCache] = [:]
    public private(set) var pos = 0
    public var sampler: Sampler?

    public init(_ model: QwenModel) {
        self.model = model
        cfg = model.cfg
        for il in 0..<cfg.nLayer {
            if cfg.isRecurrent(il) { gdn[il] = GDNState(cfg) } else { kvc[il] = KVCache(cfg) }
        }
    }

    public func reset() {
        pos = 0
        for il in 0..<cfg.nLayer {
            if cfg.isRecurrent(il) { gdn[il] = GDNState(cfg) } else { kvc[il] = KVCache(cfg) }
        }
    }

    // Prefill onto the CURRENT state; no reset.
    public func extend(_ ids: [Int32]) -> Int32 {
        var hidden = [Float]()
        for id in ids { hidden = forward(token: Int(id), pos: pos); pos += 1 }
        return pick(logits(hidden))
    }

    public func decode(_ token: Int32) -> Int32 {
        let hidden = forward(token: Int(token), pos: pos)
        pos += 1
        return pick(logits(hidden))
    }

    func pick(_ logits: [Float]) -> Int32 {
        if sampler != nil {
            var work = logits
            let picked = sampler!.sample(&work)
            sampler!.accept(picked)
            return picked
        }
        return Int32(Vectors.argmax(logits))
    }

    // The GDN recurrence is copied whole, since it cannot be paged; the KV
    // shares append-only page snapshots, so only a partial tail copies.

    public func serialize(_ b: Bookmark) -> Data {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, b.pos)
        StateBytes.putKeyed(&out, b.gdn) { out, _, s in
            StateBytes.putFloats(&out, s.conv)
            StateBytes.putFloats(&out, s.rec)
        }
        StateBytes.putKeyed(&out, b.kv) { out, _, s in
            StateBytes.putInt(&out, s.len)
            StateBytes.putInt(&out, s.kPages.count)
            for page in s.kPages { StateBytes.putFloats(&out, page) }
            for page in s.vPages { StateBytes.putFloats(&out, page) }
        }
        return out
    }

    public func deserialize(_ data: Data) -> Bookmark? {
        StateBytes.read(data) { r in
            let pos = r.int()
            let gdn = StateBytes.keyed(&r) {
                r -> (conv: [Float], rec: [Float]) in
                let conv = r.span().array
                let rec = r.span().array
                return (conv: conv, rec: rec)
            }
            let kv = StateBytes.keyed(&r) { r -> KVCache.Snapshot in
                let len = r.int()
                let n = r.int()
                var kp: [[Float]] = [], vp: [[Float]] = []
                for _ in 0..<n { kp.append(r.span().array) }
                for _ in 0..<n { vp.append(r.span().array) }
                return KVCache.Snapshot(kPages: kp, vPages: vp, len: len)
            }
            return Bookmark(pos: pos, gdn: gdn, kv: kv)
        }
    }

    public struct Bookmark: @unchecked Sendable {
        let pos: Int
        let gdn: [Int: (conv: [Float], rec: [Float])]
        let kv: [Int: KVCache.Snapshot]
    }

    public func bookmark() -> Bookmark {
        var g: [Int: (conv: [Float], rec: [Float])] = [:]
        for (il, s) in gdn { g[il] = (s.conv, s.rec) }
        var k: [Int: KVCache.Snapshot] = [:]
        for (il, c) in kvc { k[il] = c.snapshot() }
        return Bookmark(pos: pos, gdn: g, kv: k)
    }

    public func restore(_ b: Bookmark) {
        pos = b.pos
        for (il, s) in b.gdn { gdn[il]!.conv = s.conv; gdn[il]!.rec = s.rec }
        for (il, s) in b.kv { kvc[il]!.restore(s) }
    }

    func embed(_ token: Int) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        out.withUnsafeMutableBufferPointer { ob in
            QB.dequant(model.tokEmbd, row: token, count: cfg.nEmbd,
                       into: ob.baseAddress!)
        }
        return out
    }

    func ffn(_ x: [Float], _ L: QwenLayer) -> [Float] {
        var gate = [Float](repeating: 0, count: cfg.nFF)
        var up = [Float](repeating: 0, count: cfg.nFF)
        QB.matvec(L.ffnGate, x: x, out: &gate)
        QB.matvec(L.ffnUp, x: x, out: &up)
        for i in 0..<cfg.nFF { gate[i] = silu(gate[i]) * up[i] }
        var out = [Float](repeating: 0, count: cfg.nEmbd)
        QB.matvec(L.ffnDown, x: gate, out: &out)
        return out
    }

    // Logits are computed separately, only for the tokens that get sampled.
    @discardableResult
    public func forward(token: Int, pos: Int, tap: ((String, Int, [Float]) -> Void)? = nil) -> [Float] {
        var x = embed(token)
        tap?("embed", -1, x)
        for il in 0..<cfg.nLayer {
            let L = model.layers[il]
            let normed = Kern.rmsnorm(x, F32T.ptr(L.attnNorm), cfg.eps)
            tap?("attn_norm", il, normed)
            let attn: [Float]
            if L.recurrent {
                attn = GDN.step(normed, L, gdn[il]!, cfg)
            } else {
                attn = Attn.step(normed, L, kvc[il]!, pos: pos, cfg)
            }
            tap?("attn_out", il, attn)
            for i in 0..<cfg.nEmbd { x[i] += attn[i] }
            let ffnRes = x
            let postNormed = Kern.rmsnorm(x, F32T.ptr(L.attnPostNorm), cfg.eps)
            let f = ffn(postNormed, L)
            x = ffnRes
            for i in 0..<cfg.nEmbd { x[i] += f[i] }
            tap?("l_out", il, x)
        }
        return Kern.rmsnorm(x, F32T.ptr(model.outputNorm), cfg.eps)
    }

    public func logits(_ hidden: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: cfg.nVocab)
        QB.matvec(model.output, x: hidden, out: &out)
        return out
    }

}

extension QwenEngine.Bookmark: BackendState {}

extension QwenEngine: TextEngine {}
