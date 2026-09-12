import Foundation

enum IQ2XXS {

    static let qk = 256
    static let blockBytes = 66

    static func dot(_ block: UnsafeRawPointer,
                    _ x: UnsafePointer<Float>) -> Float {
        var acc: Float = 0
        let d = Float(block.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let a0 = block.loadUnaligned(fromByteOffset: 2 + ib * 8,
                                         as: UInt32.self)
            let a1 = block.loadUnaligned(fromByteOffset: 2 + ib * 8 + 4,
                                         as: UInt32.self)
            let db = d * (0.5 + Float(a1 >> 28)) * 0.25
            var sum: Float = 0
            for l in 0..<4 {
                let entry = IQ2Grid.xxs[Int((a0 >> (8 * l)) & 0xFF)]
                let signs = IQ2Grid.ksigns[Int((a1 >> (7 * l)) & 127)]
                let at = x + ib * 32 + l * 8
                for j in 0..<8 {
                    let g = Float((entry >> (8 * j)) & 0xFF)
                    let v = (signs & IQ2Grid.kmask[j]) != 0 ? -at[j] : at[j]
                    sum += g * v
                }
            }
            acc += db * sum
        }
        return acc
    }

    static func dequant(_ block: UnsafeRawPointer,
                        into out: UnsafeMutablePointer<Float>) {
        let d = Float(block.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let a0 = block.loadUnaligned(fromByteOffset: 2 + ib * 8,
                                         as: UInt32.self)
            let a1 = block.loadUnaligned(fromByteOffset: 2 + ib * 8 + 4,
                                         as: UInt32.self)
            let db = d * (0.5 + Float(a1 >> 28)) * 0.25
            for l in 0..<4 {
                let entry = IQ2Grid.xxs[Int((a0 >> (8 * l)) & 0xFF)]
                let signs = IQ2Grid.ksigns[Int((a1 >> (7 * l)) & 127)]
                for j in 0..<8 {
                    let g = Float((entry >> (8 * j)) & 0xFF)
                    let s: Float = (signs & IQ2Grid.kmask[j]) != 0 ? -1 : 1
                    out[ib * 32 + l * 8 + j] = db * g * s
                }
            }
        }
    }
}

enum IQ1M {

    static let qk = 256
    static let blockBytes = 56

    // The f16 scale is spread four nibbles at a time across the scales
    // shorts, so it has to be reassembled before anything else can be read.
    static func scale(_ block: UnsafeRawPointer) -> (Float, [UInt16]) {
        var sc = [UInt16](repeating: 0, count: 4)
        for i in 0..<4 {
            sc[i] = block.loadUnaligned(fromByteOffset: 48 + i * 2,
                                        as: UInt16.self)
        }
        let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
            | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
        return (Float(Float16(bitPattern: bits)), sc)
    }

    static func dequant(_ block: UnsafeRawPointer,
                        into out: UnsafeMutablePointer<Float>) {
        let (d, sc) = scale(block)
        for ib in 0..<8 {
            let s = sc[ib / 2]
            let shift = 6 * (ib % 2)
            let dl1 = d * Float(2 * ((s >> shift) & 7) + 1)
            let dl2 = d * Float(2 * ((s >> (shift + 3)) & 7) + 1)
            let h0 = block.load(fromByteOffset: 32 + ib * 2, as: UInt8.self)
            let h1 = block.load(fromByteOffset: 33 + ib * 2, as: UInt8.self)
            for l in 0..<4 {
                let q = block.load(fromByteOffset: ib * 4 + l, as: UInt8.self)
                let h = l < 2 ? h0 : h1
                let hi = (l % 2) == 0 ? (Int(h) << 8) : (Int(h) << 4)
                let idx = Int(q) | (hi & 0x700)
                let neg = (h & (l % 2 == 0 ? 0x08 : 0x80)) != 0
                let delta = neg ? -IQ1Grid.delta : IQ1Grid.delta
                let dl = l < 2 ? dl1 : dl2
                let entry = IQ1Grid.gpu[idx]
                for j in 0..<8 {
                    let n = Float((entry >> (8 * (j % 4) + 4 * (j / 4))) & 0xF)
                    out[ib * 32 + l * 8 + j] = dl * (n - 1 + delta)
                }
            }
        }
    }

    static func dot(_ block: UnsafeRawPointer,
                    _ x: UnsafePointer<Float>) -> Float {
        var w = [Float](repeating: 0, count: qk)
        var acc: Float = 0
        w.withUnsafeMutableBufferPointer { b in
            dequant(block, into: b.baseAddress!)
            for i in 0..<qk { acc += b[i] * x[i] }
        }
        return acc
    }
}
