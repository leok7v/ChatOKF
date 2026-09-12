import Foundation

public enum Vectors {

    public static func argmax(_ v: [Float]) -> Int {
        v.withUnsafeBufferPointer { b in
            b.baseAddress.map { p in argmax(p, b.count) } ?? 0
        }
    }

    public static func argmax(_ v: UnsafePointer<Float>, _ n: Int) -> Int {
        var bi = 0
        var bv = -Float.greatestFiniteMagnitude
        for i in 0..<n where v[i] > bv {
            bv = v[i]
            bi = i
        }
        return bi
    }

    public static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }
}
