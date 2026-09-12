import Foundation

enum IQ1 {

    static let qk = 256
    static let blockBytes = 50

    static func dot(_ block: UnsafeRawPointer,
                    _ x: UnsafePointer<Float>) -> Float {
        let d = Float(block.loadUnaligned(as: Float16.self))
        var acc: Float = 0
        for ib in 0..<8 {
            let h = block.loadUnaligned(fromByteOffset: 34 + ib * 2,
                                        as: UInt16.self)
            let dl = d * Float(2 * ((h >> 12) & 7) + 1)
            let delta = (h & 0x8000) != 0 ? -IQ1Grid.delta : IQ1Grid.delta
            var lo: Float = 0
            var hi: Float = 0
            var sx: Float = 0
            for l in 0..<4 {
                let q = block.load(fromByteOffset: 2 + ib * 4 + l,
                                   as: UInt8.self)
                let idx = Int(q) | ((Int(h >> (3 * l)) & 7) << 8)
                let entry = IQ1Grid.gpu[idx]
                let at = x + ib * 32 + l * 8
                for j in 0..<8 {
                    let n = (entry >> (8 * (j % 4) + 4 * (j / 4))) & 0xF
                    let v = at[j]
                    sx += v
                    lo += n == 1 ? v : 0
                    hi += n == 2 ? v : 0
                }
            }
            acc += dl * (lo + 2 * hi + (delta - 1) * sx)
        }
        return acc
    }

    static func block(_ p: UnsafeRawPointer,
                      _ x: UnsafePointer<Float>) -> Float {
        dot(p, x)
    }

    static func dequant(_ block: UnsafeRawPointer,
                        into out: UnsafeMutablePointer<Float>) {
        let d = Float(block.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let h = block.loadUnaligned(fromByteOffset: 34 + ib * 2,
                                        as: UInt16.self)
            let dl = d * Float(2 * ((h >> 12) & 7) + 1)
            let delta = (h & 0x8000) != 0 ? -IQ1Grid.delta : IQ1Grid.delta
            for l in 0..<4 {
                let q = block.load(fromByteOffset: 2 + ib * 4 + l,
                                   as: UInt8.self)
                let idx = Int(q) | ((Int(h >> (3 * l)) & 7) << 8)
                let entry = IQ1Grid.gpu[idx]
                for j in 0..<8 {
                    let shift = 8 * (j % 4) + 4 * (j / 4)
                    let n = Float((entry >> shift) & 0xF)
                    out[ib * 32 + l * 8 + j] = dl * (n - 1 + delta)
                }
            }
        }
    }
}
