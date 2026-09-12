import Foundation

// All four share one shape: a grid of unsigned magnitudes, a sign pattern
// from ksigns_iq2xs, and a sub-scale per 32 weights off the f16 d.
enum IQX {

    static let qk = 256

    static func sign(_ signs: UInt8, _ j: Int, _ v: Float) -> Float {
        (signs & IQ2Grid.kmask[j]) != 0 ? -v : v
    }

    // { half d; uint16 qs[32]; uint8 scales[8] }
    static func iq2XS(_ b: UnsafeRawPointer,
                      into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let sc = b.load(fromByteOffset: 66 + ib, as: UInt8.self)
            let db0 = d * (0.5 + Float(sc & 0xF)) * 0.25
            let db1 = d * (0.5 + Float(sc >> 4)) * 0.25
            for l in 0..<4 {
                let q = b.loadUnaligned(fromByteOffset: 2 + (ib * 4 + l) * 2,
                                        as: UInt16.self)
                let entry = IQ3Grid.xs2[Int(q & 511)]
                let signs = IQ2Grid.ksigns[Int(q >> 9)]
                let dl = l / 2 == 0 ? db0 : db1
                for j in 0..<8 {
                    let g = Float((entry >> (8 * j)) & 0xFF)
                    out[ib * 32 + l * 8 + j] = sign(signs, j, dl * g)
                }
            }
        }
    }

    // { half d; uint8 qs[32]; uint8 qh[8]; uint8 signs[32]; uint8 scales[8] }
    static func iq2S(_ b: UnsafeRawPointer,
                     into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let sc = b.load(fromByteOffset: 74 + ib, as: UInt8.self)
            let db0 = d * (0.5 + Float(sc & 0xF)) * 0.25
            let db1 = d * (0.5 + Float(sc >> 4)) * 0.25
            let qh = b.load(fromByteOffset: 66 + ib, as: UInt8.self)
            for l in 0..<4 {
                let q = b.load(fromByteOffset: 2 + ib * 4 + l, as: UInt8.self)
                let idx = Int(q) | ((Int(qh) << (8 - 2 * l)) & 0x300)
                let entry = IQ3Grid.s2[idx]
                let signs = b.load(fromByteOffset: 34 + ib * 4 + l,
                                   as: UInt8.self)
                let dl = l / 2 == 0 ? db0 : db1
                for j in 0..<8 {
                    let g = Float((entry >> (8 * j)) & 0xFF)
                    out[ib * 32 + l * 8 + j] = sign(signs, j, dl * g)
                }
            }
        }
    }

    // { half d; uint8 qs[64]; uint8 scales_and_signs[32] }
    static func iq3XXS(_ b: UnsafeRawPointer,
                       into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        for ib in 0..<8 {
            let aux = b.loadUnaligned(fromByteOffset: 66 + ib * 4,
                                      as: UInt32.self)
            let db = d * (0.5 + Float(aux >> 28)) * 0.5
            for l in 0..<4 {
                let signs = IQ2Grid.ksigns[Int((aux >> (7 * l)) & 127)]
                let g1 = IQ3Grid.xxs3[Int(b.load(
                    fromByteOffset: 2 + ib * 8 + 2 * l, as: UInt8.self))]
                let g2 = IQ3Grid.xxs3[Int(b.load(
                    fromByteOffset: 2 + ib * 8 + 2 * l + 1, as: UInt8.self))]
                let at = ib * 32 + l * 8
                for j in 0..<4 {
                    out[at + j] = sign(signs, j,
                        db * Float((g1 >> (8 * j)) & 0xFF))
                    out[at + j + 4] = sign(signs, j + 4,
                        db * Float((g2 >> (8 * j)) & 0xFF))
                }
            }
        }
    }

    // { half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4] }
    static func iq3S(_ b: UnsafeRawPointer,
                     into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        for pair in 0..<4 {
            let sc = b.load(fromByteOffset: 106 + pair, as: UInt8.self)
            let db1 = d * Float(1 + 2 * (sc & 0xF))
            let db2 = d * Float(1 + 2 * (sc >> 4))
            for half in 0..<2 {
                let ib = pair * 2 + half
                let qh = b.load(fromByteOffset: 66 + ib, as: UInt8.self)
                let db = half == 0 ? db1 : db2
                for l in 0..<4 {
                    let base = 2 + ib * 8 + 2 * l
                    let lo = Int(b.load(fromByteOffset: base, as: UInt8.self))
                    let hi = Int(b.load(fromByteOffset: base + 1,
                                        as: UInt8.self))
                    let g1 = IQ3Grid.s3[lo | ((Int(qh) << (8 - 2 * l)) & 256)]
                    let g2 = IQ3Grid.s3[hi | ((Int(qh) << (7 - 2 * l)) & 256)]
                    let signs = b.load(fromByteOffset: 74 + ib * 4 + l,
                                       as: UInt8.self)
                    let at = ib * 32 + l * 8
                    for j in 0..<4 {
                        out[at + j] = sign(signs, j,
                            db * Float((g1 >> (8 * j)) & 0xFF))
                        out[at + j + 4] = sign(signs, j + 4,
                            db * Float((g2 >> (8 * j)) & 0xFF))
                    }
                }
            }
        }
    }

    // { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] }
    static func iq4XS(_ b: UnsafeRawPointer,
                      into out: UnsafeMutablePointer<Float>) {
        let d = Float(b.loadUnaligned(as: Float16.self))
        let sh = b.loadUnaligned(fromByteOffset: 2, as: UInt16.self)
        for ib in 0..<8 {
            let sl = b.load(fromByteOffset: 4 + ib / 2, as: UInt8.self)
            let ls = Int((sl >> (4 * (ib % 2))) & 0xF)
                | ((Int(sh >> (2 * ib)) & 3) << 4)
            let dl = d * Float(ls - 32)
            for j in 0..<16 {
                let q = b.load(fromByteOffset: 8 + ib * 16 + j, as: UInt8.self)
                out[ib * 32 + j] = dl * Float(IQ3Grid.nl4[Int(q & 0xF)])
                out[ib * 32 + 16 + j] = dl * Float(IQ3Grid.nl4[Int(q >> 4)])
            }
        }
    }
}
