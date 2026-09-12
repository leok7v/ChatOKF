import Foundation

@inline(__always) func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
@inline(__always) func sigmoidf(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
@inline(__always) func softplusf(_ x: Float) -> Float {
    // numerically stable: relu(x) + log1p(exp(-|x|))  (matches ggml_softplus)
    max(x, 0) + log1pf(expf(-abs(x)))
}

enum Kern {
    // llama.cpp build_norm multiplies by the stored weight directly (no 1+w).
    static func rmsnorm(_ x: [Float], _ w: UnsafePointer<Float>, _ eps: Float) -> [Float] {
        let d = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let scale = 1 / (ss / Float(d) + eps).squareRoot()
        var o = [Float](repeating: 0, count: d)
        for i in 0..<d { o[i] = x[i] * scale * w[i] }
        return o
    }

    static func rmsnormRows(_ x: inout [Float], d: Int, rows: Int, w: UnsafePointer<Float>, eps: Float) {
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

    // ggml_l2_norm has no weight and uses sum, not mean.
    static func l2normRows(_ x: inout [Float], d: Int, rows: Int, eps: Float) {
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for r in 0..<rows {
                let off = r * d
                var ss: Float = 0
                for i in 0..<d { ss += p[off + i] * p[off + i] }
                let s = 1 / (ss + eps).squareRoot()
                for i in 0..<d { p[off + i] *= s }
            }
        }
    }

    static func softmaxInPlace(_ x: inout [Float], _ n: Int, off: Int = 0) {
        var mx = -Float.greatestFiniteMagnitude
        for i in 0..<n { mx = max(mx, x[off + i]) }
        var s: Float = 0
        for i in 0..<n { let e = expf(x[off + i] - mx); x[off + i] = e; s += e }
        let inv = 1 / s
        for i in 0..<n { x[off + i] *= inv }
    }

    static func ropeNeox(_ x: inout [Float], headDim: Int, nHead: Int,
                         nRot: Int, base: Float, pos: Int) {
        let half = nRot / 2
        x.withUnsafeMutableBufferPointer { b in
            let p = b.baseAddress!
            for h in 0..<nHead {
                let hoff = h * headDim
                for i in 0..<half {
                    let freq = powf(base, -2 * Float(i) / Float(nRot))
                    let ang = Float(pos) * freq
                    let c = cosf(ang), s = sinf(ang)
                    let a = p[hoff + i]
                    let bb = p[hoff + i + half]
                    p[hoff + i] = a * c - bb * s
                    p[hoff + i + half] = a * s + bb * c
                }
            }
        }
    }

    static func matvecF32(_ a: UnsafePointer<Float>, x: [Float], m: Int, k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m)
        x.withUnsafeBufferPointer { xb in
            let xp = xb.baseAddress!
            out.withUnsafeMutableBufferPointer { ob in
                for r in 0..<m {
                    var acc: Float = 0
                    let off = r * k
                    for i in 0..<k { acc += a[off + i] * xp[i] }
                    ob[r] = acc
                }
            }
        }
        return out
    }
}
