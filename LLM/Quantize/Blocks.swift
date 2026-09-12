import Foundation

// One table from ggml type to block decoder, so a new type is added once
// and matvec, the goldens and the Metal upload path all pick it up.
enum Blocks {

    typealias Decode = @Sendable (UnsafeRawPointer,
                                  UnsafeMutablePointer<Float>) -> Void

    static func decoder(_ t: GGUFType) -> Decode? {
        let out: Decode?
        switch t {
        case .iq1_s: out = IQ1.dequant
        case .iq1_m: out = IQ1M.dequant
        case .iq2_xxs: out = IQ2XXS.dequant
        case .iq2_xs: out = IQX.iq2XS
        case .iq2_s: out = IQX.iq2S
        case .iq3_xxs: out = IQX.iq3XXS
        case .iq3_s: out = IQX.iq3S
        case .iq4_xs: out = IQX.iq4XS
        case .q2k: out = KQuant.q2K
        case .q3k: out = KQuant.q3K
        case .q4k: out = KQuant.q4K
        case .q5k: out = KQuant.q5K
        case .q6k: out = KQuant.q6K
        default: out = nil
        }
        return out
    }

    static func superBlocked(_ t: GGUFType) -> Bool { decoder(t) != nil }

    static func codebook(_ t: GGUFType) -> Bool {
        let out: Bool
        switch t {
        case .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s,
             .iq3_xxs, .iq3_s, .iq4_xs, .iq4_nl: out = true
        default: out = false
        }
        return out
    }

    static let superBlock = 256
}
