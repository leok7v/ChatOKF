import Foundation

// The K-quants inside unsloth's mixed UD- files: an f16 scale, an f16
// minimum where affine, packed 6-bit sub-scales, no codebook.
enum KQuant {

    static let qk = 256

    // { uint8 scales[16]; uint8 qs[64]; half d; half dmin }: ggml's order.
    static func q2K(_ b: UnsafeRawPointer,
                    into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(fromByteOffset: 80, as: Float16.self))
        let dmin = Float(b.loadUnaligned(fromByteOffset: 82, as: Float16.self))
        var y = 0
        var qOff = 16
        var isc = 0
        for _ in 0..<2 {
            var shift = 0
            for _ in 0..<4 {
                for half in 0..<2 {
                    let sc = b.load(fromByteOffset: isc, as: UInt8.self)
                    isc += 1
                    let dl = d * Float(sc & 0xF)
                    let ml = dmin * Float(sc >> 4)
                    for l in 0..<16 {
                        let q = b.load(fromByteOffset: qOff + half * 16 + l,
                                       as: UInt8.self)
                        out[y] = dl * Float((q >> shift) & 3) - ml
                        y += 1
                    }
                }
                shift += 2
            }
            qOff += 32
        }
    }

    // { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d }
    static func q3K(_ b: UnsafeRawPointer,
                    into out: UnsafeMutablePointer<Float>) {
        let dAll = Float(b.loadUnaligned(fromByteOffset: 108, as: Float16.self))
        var aux = [UInt32](repeating: 0, count: 4)
        for i in 0..<3 {
            aux[i] = b.loadUnaligned(fromByteOffset: 96 + i * 4, as: UInt32.self)
        }
        let kmask1: UInt32 = 0x0303_0303
        let kmask2: UInt32 = 0x0f0f_0f0f
        let tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        var scales = [Int8](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                scales[i * 4 + j] = Int8(bitPattern:
                    UInt8((aux[i] >> (8 * j)) & 0xFF))
            }
        }
        var y = 0
        var qOff = 32
        var isc = 0
        var m: UInt8 = 1
        for _ in 0..<2 {
            var shift = 0
            for _ in 0..<4 {
                for half in 0..<2 {
                    let dl = dAll * Float(Int(scales[isc]) - 32)
                    isc += 1
                    for l in 0..<16 {
                        let at = half * 16 + l
                        let q = b.load(fromByteOffset: qOff + at, as: UInt8.self)
                        let h = b.load(fromByteOffset: at, as: UInt8.self)
                        let bump: Float = (h & m) != 0 ? 0 : 4
                        out[y] = dl * (Float((q >> shift) & 3) - bump)
                        y += 1
                    }
                }
                shift += 2
                m <<= 1
            }
            qOff += 32
        }
    }

    private static func scaleMin(_ b: UnsafeRawPointer,
                                 _ j: Int) -> (Float, Float) {
        func s(_ i: Int) -> UInt8 {
            b.load(fromByteOffset: 4 + i, as: UInt8.self)
        }
        let sc: UInt8
        let mn: UInt8
        if j < 4 {
            sc = s(j) & 63
            mn = s(j + 4) & 63
        } else {
            sc = (s(j + 4) & 0xF) | ((s(j - 4) >> 6) << 4)
            mn = (s(j + 4) >> 4) | ((s(j) >> 6) << 4)
        }
        return (Float(sc), Float(mn))
    }

    // { half d; half dmin; uint8 scales[12]; uint8 qs[128] }
    static func q4K(_ b: UnsafeRawPointer,
                    into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        let dmin = Float(b.loadUnaligned(fromByteOffset: 2, as: Float16.self))
        var y = 0
        var qOff = 16
        var isc = 0
        for _ in 0..<4 {
            let (s1, m1) = scaleMin(b, isc)
            let (s2, m2) = scaleMin(b, isc + 1)
            let d1 = d * s1, o1 = dmin * m1
            let d2 = d * s2, o2 = dmin * m2
            for l in 0..<32 {
                let q = b.load(fromByteOffset: qOff + l, as: UInt8.self)
                out[y + l] = d1 * Float(q & 0xF) - o1
                out[y + 32 + l] = d2 * Float(q >> 4) - o2
            }
            y += 64
            qOff += 32
            isc += 2
        }
    }

    static func q5K(_ b: UnsafeRawPointer,
                    into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        let dmin = Float(b.loadUnaligned(fromByteOffset: 2, as: Float16.self))
        var y = 0
        var qOff = 48
        var isc = 0
        var u1: UInt8 = 1
        var u2: UInt8 = 2
        for _ in 0..<4 {
            let (s1, m1) = scaleMin(b, isc)
            let (s2, m2) = scaleMin(b, isc + 1)
            let d1 = d * s1, o1 = dmin * m1
            let d2 = d * s2, o2 = dmin * m2
            for l in 0..<32 {
                let q = b.load(fromByteOffset: qOff + l, as: UInt8.self)
                let h = b.load(fromByteOffset: 16 + l, as: UInt8.self)
                let lo = Float((q & 0xF) + ((h & u1) != 0 ? 16 : 0))
                let hi = Float((q >> 4) + ((h & u2) != 0 ? 16 : 0))
                out[y + l] = d1 * lo - o1
                out[y + 32 + l] = d2 * hi - o2
            }
            y += 64
            qOff += 32
            isc += 2
            u1 <<= 2
            u2 <<= 2
        }
    }

    static func q6K(_ b: UnsafeRawPointer,
                    into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(fromByteOffset: 208, as: Float16.self))
        for n in 0..<2 {
            for l in 0..<32 {
                let h = Int(b.load(fromByteOffset: 128 + n * 32 + l,
                                   as: UInt8.self))
                let a = Int(b.load(fromByteOffset: n * 64 + l, as: UInt8.self))
                let c = Int(b.load(fromByteOffset: n * 64 + l + 32,
                                   as: UInt8.self))
                for r in 0..<4 {
                    let byte = (r % 2) == 0 ? a : c
                    let nib = r < 2 ? (byte & 0xF) : (byte >> 4)
                    let q = nib | (((h >> (2 * r)) & 3) << 4)
                    let s = Int(b.load(fromByteOffset: 192 + n * 8 + 2 * r
                                       + l / 16, as: Int8.self))
                    out[n * 128 + r * 32 + l] = d * Float(s) * Float(q - 32)
                }
            }
        }
    }
}
