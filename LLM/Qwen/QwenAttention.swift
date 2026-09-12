// One token at a time; mirrors qwen35.cpp build_layer_attn. The partial
// MRoPE is NEOX for text.
import Foundation

final class KVCache {
    static let pageP = 512
    let kvDim: Int
    private let P: Int
    private(set) var kPages: [[Float]] = []
    private(set) var vPages: [[Float]] = []
    private(set) var len = 0

    init(_ cfg: QwenConfig) {
        kvDim = cfg.headDim * cfg.nHeadKV
        P = KVCache.pageP
    }

    func append(_ kk: [Float], _ vv: [Float]) {
        let slot = len % P
        if slot == 0 {
            kPages.append([Float](repeating: 0, count: P * kvDim))
            vPages.append([Float](repeating: 0, count: P * kvDim))
        }
        let pi = kPages.count - 1
        let off = slot * kvDim
        for i in 0..<kvDim { kPages[pi][off + i] = kk[i]; vPages[pi][off + i] = vv[i] }
        len += 1
    }

    func kBase(_ t: Int, _ kvh: Int, _ headDim: Int) -> ArraySlice<Float> {
        let off = (t % P) * kvDim + kvh * headDim
        return kPages[t / P][off ..< off + headDim]
    }
    func vBase(_ t: Int, _ kvh: Int, _ headDim: Int) -> ArraySlice<Float> {
        let off = (t % P) * kvDim + kvh * headDim
        return vPages[t / P][off ..< off + headDim]
    }

    func truncate(to newLength: Int) {
        len = newLength
        let need = (newLength + P - 1) / P
        while kPages.count > need { kPages.removeLast(); vPages.removeLast() }
    }

    // A bookmark shares completed pages; only the tail copies on next append.
    struct Snapshot: Sendable {
        let kPages: [[Float]]
        let vPages: [[Float]]
        let len: Int
    }
    func snapshot() -> Snapshot { Snapshot(kPages: kPages, vPages: vPages, len: len) }
    func restore(_ s: Snapshot) {
        kPages = s.kPages; vPages = s.vPages; len = s.len
    }
}

enum Attn {
    static func step(_ n: [Float], _ L: QwenLayer, _ kv: KVCache, pos: Int, _ cfg: QwenConfig) -> [Float] {
        let hd = cfg.headDim, nH = cfg.nHead, nKV = cfg.nHeadKV
        let group = nH / nKV

        // The hybrid fuses [q(hd) | gate(hd)] per head into wq and gates the
        // output by sigmoid(gate); dense qwen3 leaves `gate` empty.
        var q = [Float](repeating: 0, count: hd * nH)
        var gate: [Float] = []
        if cfg.dense {
            QB.matvec(L.wq!, x: n, out: &q)
        } else {
            var qFull = [Float](repeating: 0, count: hd * 2 * nH)
            QB.matvec(L.wq!, x: n, out: &qFull)
            gate = [Float](repeating: 0, count: hd * nH)
            for h in 0..<nH {
                let src = h * hd * 2
                for i in 0..<hd {
                    q[h * hd + i] = qFull[src + i]
                    gate[h * hd + i] = qFull[src + hd + i]
                }
            }
        }
        var kCur = [Float](repeating: 0, count: hd * nKV)
        QB.matvec(L.wk!, x: n, out: &kCur)
        var vCur = [Float](repeating: 0, count: hd * nKV)
        QB.matvec(L.wv!, x: n, out: &vCur)

        let qn = F32T.ptr(L.qNorm!), kn = F32T.ptr(L.kNorm!)
        Kern.rmsnormRows(&q, d: hd, rows: nH, w: qn, eps: cfg.eps)
        Kern.rmsnormRows(&kCur, d: hd, rows: nKV, w: kn, eps: cfg.eps)

        Kern.ropeNeox(&q, headDim: hd, nHead: nH, nRot: cfg.nRot, base: cfg.ropeBase, pos: pos)
        Kern.ropeNeox(&kCur, headDim: hd, nHead: nKV, nRot: cfg.nRot, base: cfg.ropeBase, pos: pos)

        kv.append(kCur, vCur)
        let T = kv.len
        let scale = 1 / Float(hd).squareRoot()

        var attnOut = [Float](repeating: 0, count: hd * nH)
        for h in 0..<nH {
            let kvh = h / group
            let qh = h * hd
            var scores = [Float](repeating: 0, count: T)
            for t in 0..<T {
                var s: Float = 0
                for (i, kti) in kv.kBase(t, kvh, hd).enumerated() {
                    s += q[qh + i] * kti
                }
                scores[t] = s * scale
            }
            Kern.softmaxInPlace(&scores, T)
            for t in 0..<T {
                let w = scores[t]
                for (i, vti) in kv.vBase(t, kvh, hd).enumerated() {
                    attnOut[qh + i] += w * vti
                }
            }
        }

        if !cfg.dense {
            for i in 0..<(hd * nH) { attnOut[i] *= sigmoidf(gate[i]) }
        }

        var out = [Float](repeating: 0, count: cfg.nEmbd)
        QB.matvec(L.wo!, x: attnOut, out: &out)
        return out
    }
}
