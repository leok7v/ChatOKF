import Foundation

public enum BF16 {
    static func decode(_ raw: UnsafeRawPointer, _ n: Int,
                       _ out: UnsafeMutablePointer<Float>) {
        for i in 0..<n {
            let h = (raw + i * 2).loadUnaligned(as: UInt16.self)
            out[i] = Float(bitPattern: UInt32(h) << 16)
        }
    }
}

public enum FP8 {
    public static let e4m3: [Float] = {
        var table = [Float](repeating: 0, count: 256)
        for bits in 0..<256 {
            let sign: Float = (bits & 0x80) != 0 ? -1 : 1
            let exp = (bits >> 3) & 0x0F
            let man = bits & 0x07
            let value: Float
            if exp == 0x0F && man == 0x07 {
                value = Float.nan
            } else if exp == 0 {
                value = sign * Float(man) * exp2(Float(-9))
            } else {
                value = sign * (1.0 + Float(man) / 8.0) * exp2(Float(exp - 7))
            }
            table[bits] = value
        }
        return table
    }()

    static func decode(_ raw: UnsafePointer<UInt8>, _ n: Int,
                       _ out: UnsafeMutablePointer<Float>) {
        e4m3.withUnsafeBufferPointer { lut in
            let t = lut.baseAddress!
            for i in 0..<n { out[i] = t[Int(raw[i])] }
        }
    }
}
