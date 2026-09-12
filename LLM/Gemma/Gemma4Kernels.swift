// Kept apart from Kern so the ternary path's numerics cannot drift; the weight
// is [Float] because every gemma norm is BF16, and rope takes a PAIR COUNT.
import Foundation

// A model-shaped VALUE like the rope base; a name not implemented stops the
// load.

enum GemmaActivation: String {
    case geluTanh = "gelu_pytorch_tanh"
    case silu = "silu"

    @inline(__always)
    func apply(_ v: Float) -> Float {
        let out: Float
        switch self {
        case .geluTanh: out = GK.geluTanh(v)
        case .silu: out = v / (1 + expf(-v))
        }
        return out
    }

    // Absence THROWS rather than defaulting: a file arrives from a download, so
    // the caller tells the user, and a guessed curve would read fluent.
    static func read(_ g: GGUF, _ key: String) throws -> GemmaActivation {
        let name = g.string(key) ?? ""
        let found = GemmaActivation(rawValue: name)
        if found == nil {
            throw GGUFErr.parse(
                "\(key) is \(name.isEmpty ? "absent" : name); this build "
                + "implements gelu_pytorch_tanh and silu. Re-emit the file "
                + "with a repack that records it.")
        }
        return found!
    }
}

enum GK {

    // normed * weight with NO `1 + w`: gemma 1/2/3 used (1 + w) and llama.cpp's
    // converter bakes the +1 in; gemma 4 does not, so neither may add it.
    static func rmsnorm(_ x: [Float], _ w: [Float], _ eps: Float) -> [Float] {
        let d = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let scale = 1 / (ss / Float(d) + eps).squareRoot()
        var o = [Float](repeating: 0, count: d)
        for i in 0..<d { o[i] = x[i] * scale * w[i] }
        return o
    }

    static func rmsnormRows(_ x: inout [Float], d: Int, rows: Int,
                            w: [Float], eps: Float) {
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for r in 0..<rows {
                let off = r * d
                var ss: Float = 0
                for i in 0..<d { ss += p[off + i] * p[off + i] }
                let s = 1 / (ss / Float(d) + eps).squareRoot()
                for i in 0..<d { p[off + i] = p[off + i] * s * w[i] }
            }
        }
    }

    // with_scale=False: V goes through this on every non-shared layer.
    static func rmsnormRowsNoWeight(_ x: inout [Float], d: Int, rows: Int,
                                    eps: Float) {
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for r in 0..<rows {
                let off = r * d
                var ss: Float = 0
                for i in 0..<d { ss += p[off + i] * p[off + i] }
                let s = 1 / (ss / Float(d) + eps).squareRoot()
                for i in 0..<d { p[off + i] *= s }
            }
        }
    }

    static func softmaxInPlace(_ x: inout [Float], _ n: Int) {
        var mx = -Float.greatestFiniteMagnitude
        for i in 0..<n { mx = max(mx, x[i]) }
        var s: Float = 0
        for i in 0..<n { let e = expf(x[i] - mx); x[i] = e; s += e }
        let inv = 1 / s
        for i in 0..<n { x[i] *= inv }
    }

    @inline(__always)
    static func geluTanh(_ v: Float) -> Float {
        let t = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        return 0.5 * v * (1 + tanhf(t))
    }

    // Pair (j, j + headDim/2) rotates for j < `rotated` and is left untouched
    // beyond it, which is exactly what a zero inverse frequency does.
    static func rope(_ x: inout [Float], headDim: Int, nHead: Int,
                     rotated: Int, base: Float, pos: Int) {
        let half = headDim / 2
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for j in 0..<rotated {
                let freq = powf(base, -2 * Float(j) / Float(headDim))
                let ang = Float(pos) * freq
                let c = cosf(ang), s = sinf(ang)
                for h in 0..<nHead {
                    let off = h * headDim + j
                    let a = p[off]
                    let bb = p[off + half]
                    p[off] = a * c - bb * s
                    p[off + half] = a * s + bb * c
                }
            }
        }
    }
}
