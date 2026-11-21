import Foundation

enum HadamardErr: Error, CustomStringConvertible {
    case invalid(String)

    var description: String {
        let out: String
        switch self {
        case let .invalid(m): out = "prism.hadamard: \(m)"
        }
        return out
    }
}

struct HadamardFold {
    struct Grouping: Equatable {
        let hd: Int
        let nk: Int
        let rep: Int
    }

    struct Transform {
        let block: Int
        let signs: [Float]?
        let grouping: Grouping?
    }

    let version: Int
    let block: Int
    let tied: Bool
    let explicit: Bool
    let signs: [Int: [Float]]
    let rotated: Set<String>
    let inverse: Set<String>
    let grouping: Grouping?

    static let key = "prism.hadamard."

    static let kinds: Set<String> = [
        "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        "ssm_out",
    ]

    static func foldable(_ name: String) -> Bool {
        var out = name == "output.weight"
        if !out, name.hasPrefix("blk.") {
            let parts = name.split(separator: ".",
                                   omittingEmptySubsequences: false)
            out = parts.count == 4 && !parts[1].isEmpty
                && parts[1].allSatisfy { c in c.isNumber }
                && parts[3] == "weight"
                && kinds.contains(String(parts[2]))
        }
        return out
    }

    static func read(_ g: GGUF, nV: Int, nK: Int) throws -> HadamardFold? {
        var out: HadamardFold? = nil
        if let version = g.int(key + "version") {
            out = try HadamardFold(g, version: version, nV: nV, nK: nK)
        } else if g.bool(key + "tied_output") == true {
            throw HadamardErr.invalid("tied_output requires version 2")
        }
        return out
    }

    private init(_ g: GGUF, version: Int, nV: Int, nK: Int) throws {
        let k = HadamardFold.key
        let tied = g.bool(k + "tied_output") ?? false
        let block = g.int(k + "block_size") ?? 0
        let mode = g.string(k + "sign_mode") ?? ""
        let names = g.strings(k + "weight_names") ?? []
        let latent = g.strings(k + "inverse_weight_names") ?? []
        let grouped = g.bool(k + "gdn_v_grouped") ?? false
        if let f = HadamardFold.headerFault(g, version: version, tied: tied,
                                           block: block, mode: mode,
                                           names: names) {
            throw HadamardErr.invalid(f)
        }
        let signs = mode == "explicit"
            ? try HadamardFold.readSigns(g, block: block) : [:]
        let sets = try HadamardFold.readNames(g, names: names, latent: latent,
                                             tied: tied)
        let grouping = try HadamardFold.readGrouping(
            g, grouped: grouped, rotated: sets.rotated, nV: nV, nK: nK)
        if let f = HadamardFold.tensorFault(
            g, names: sets.rotated.union(sets.inverse), block: block,
            signs: signs, explicit: mode == "explicit") {
            throw HadamardErr.invalid(f)
        }
        self.version = version
        self.block = block
        self.tied = tied
        self.explicit = mode == "explicit"
        self.signs = signs
        self.rotated = sets.rotated
        self.inverse = sets.inverse
        self.grouping = grouping
    }

    private static func headerFault(_ g: GGUF, version: Int, tied: Bool,
                                    block: Int, mode: String,
                                    names: [String]) -> String? {
        let transform = g.string(key + "transform") ?? ""
        let axis = g.string(key + "axis") ?? ""
        var out: String? = nil
        if version != 1 && version != 2 {
            out = "unsupported version \(version)"
        } else if (version == 2) != tied {
            out = "version 2 requires tied_output=true; version 1 forbids it"
        } else if block <= 0 || block & (block - 1) != 0 {
            out = "invalid block_size \(block)"
        } else if transform != "normalized-sylvester-walsh-hadamard" {
            out = "unsupported transform \(transform)"
        } else if axis != "input-last-dimension" {
            out = "unsupported axis \(axis)"
        } else if mode != "identity" && mode != "explicit" {
            out = "unsupported sign_mode \(mode)"
        } else if names.isEmpty {
            out = "weight_names is empty"
        }
        return out
    }

    private static func readSigns(_ g: GGUF,
                                  block: Int) throws -> [Int: [Float]] {
        let widths = g.ints(key + "sign_widths") ?? []
        let values = g.ints(key + "sign_values") ?? []
        var out: [Int: [Float]] = [:]
        var fault: String? = widths.isEmpty
            ? "sign_mode is explicit but sign_widths is empty" : nil
        var off = 0
        for w in widths where fault == nil {
            if w <= 0 || w % block != 0 || off + w > values.count {
                fault = "invalid sign width \(w)"
            } else if out[w] != nil {
                fault = "duplicate sign width \(w)"
            } else {
                let slice = values[off ..< off + w]
                if slice.contains(where: { v in v != 1 && v != -1 }) {
                    fault = "sign values must be +/-1"
                }
                out[w] = slice.map { v in Float(v) }
                off += w
            }
        }
        if fault == nil && off != values.count {
            fault = "sign_values length mismatch"
        }
        if let fault { throw HadamardErr.invalid(fault) }
        return out
    }

    private static func readNames(_ g: GGUF, names: [String],
                                  latent: [String], tied: Bool) throws
        -> (rotated: Set<String>, inverse: Set<String>) {
        var rotated: Set<String> = []
        var inverse: Set<String> = []
        var fault = rotatedFault(names, &rotated)
        if fault == nil { fault = latentFault(latent, rotated, &inverse) }
        if fault == nil { fault = tiedFault(g, tied: tied, inverse) }
        if fault == nil && tied { rotated.insert("token_embd.weight") }
        if let fault { throw HadamardErr.invalid(fault) }
        return (rotated, inverse)
    }

    private static func rotatedFault(_ names: [String],
                                     _ rotated: inout Set<String>) -> String? {
        var out: String? = nil
        for name in names where out == nil {
            if !foldable(name) {
                out = "weight '\(name)' is not on a verified "
                    + "Hadamard-aware matmul path"
            } else if !rotated.insert(name).inserted {
                out = "duplicate weight \(name)"
            }
        }
        return out
    }

    private static func latentFault(_ latent: [String],
                                    _ rotated: Set<String>,
                                    _ inverse: inout Set<String>) -> String? {
        var out: String? = nil
        for name in latent where out == nil {
            if name != "token_embd.weight" {
                out = "weight '\(name)' is not a verified "
                    + "inverse-after-lookup table"
            } else if rotated.contains(name) || !inverse.insert(name).inserted {
                out = "duplicate inverse weight \(name)"
            }
        }
        return out
    }

    private static func tiedFault(_ g: GGUF, tied: Bool,
                                  _ inverse: Set<String>) -> String? {
        let latent = inverse.contains("token_embd.weight")
        let head = g.maybe("output.weight") != nil
        var out: String? = nil
        if tied && head {
            out = "tied_output requires output.weight to be absent"
        } else if tied && !latent {
            out = "tied_output requires a latent token embedding"
        } else if !tied && latent && !head {
            out = "a tied Hadamard output requires version 2 and "
                + "tied_output=true"
        }
        return out
    }

    private static func readGrouping(_ g: GGUF, grouped: Bool,
                                     rotated: Set<String>, nV: Int,
                                     nK: Int) throws -> Grouping? {
        let outs = rotated.filter { name in name.contains(".ssm_out.") }
        let widths = Set(outs.compactMap { name in g.maybe(name)?.dims[0] })
        var out: Grouping? = nil
        var fault: String? = nil
        if grouped, let width = widths.first {
            let fits = widths.count == 1 && nK > 0 && nV > 0 && nV % nK == 0
                && width % nV == 0
            if fits {
                out = Grouping(hd: width / nV, nk: nK, rep: nV / nK)
            } else {
                fault = "bad GDN head geometry for gdn_v_grouped"
            }
        }
        if let fault { throw HadamardErr.invalid(fault) }
        return out
    }

    private static func tensorFault(_ g: GGUF, names: Set<String>, block: Int,
                                    signs: [Int: [Float]],
                                    explicit: Bool) -> String? {
        var out: String? = nil
        for name in names.sorted() where out == nil {
            if let t = g.maybe(name) {
                if t.dims[0] % block != 0 {
                    out = "block size \(block) does not divide input "
                        + "dimension \(t.dims[0]) for \(name)"
                } else if explicit && signs[t.dims[0]] == nil {
                    out = "no sign vector for width \(t.dims[0]) (\(name))"
                }
            } else {
                out = "weight not found: \(name)"
            }
        }
        return out
    }

    func transform(_ t: GGUFTensor) -> Transform? {
        var out: Transform? = nil
        if rotated.contains(t.name) {
            let perm = t.name.contains(".ssm_out.") ? grouping : nil
            out = Transform(block: block, signs: signs[t.dims[0]],
                            grouping: perm)
        }
        return out
    }

    func inverseTransform(_ t: GGUFTensor) -> Transform? {
        inverse.contains(t.name)
            ? Transform(block: block, signs: signs[t.dims[0]], grouping: nil)
            : nil
    }
}

enum Hadamard {
    static func dense(_ n: Int) -> [Float] {
        let scale = 1 / Float(n).squareRoot()
        var out = [Float](repeating: 0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                out[r * n + c] = (r & c).nonzeroBitCount & 1 == 0
                    ? scale : -scale
            }
        }
        return out
    }

    static func butterfly(_ x: inout [Float], at: Int, n: Int) {
        var h = 1
        while h < n {
            var i = 0
            while i < n {
                for j in i ..< i + h {
                    let a = x[at + j]
                    let b = x[at + j + h]
                    x[at + j] = a + b
                    x[at + j + h] = a - b
                }
                i += 2 * h
            }
            h *= 2
        }
    }

    static func rows(_ x: inout [Float], width: Int, block: Int,
                     signs: [Float]?, inverse: Bool) {
        precondition(width % block == 0 && x.count % width == 0,
                     "hadamard: block \(block) must divide the width "
                     + "\(width), and the width the count \(x.count)")
        let scale = 1 / Float(block).squareRoot()
        for r in 0 ..< x.count / width {
            let base = r * width
            for i in 0..<width where !inverse {
                x[base + i] *= signs?[i] ?? 1
            }
            for i in 0..<width { x[base + i] *= scale }
            var b = 0
            while b < width {
                butterfly(&x, at: base + b, n: block)
                b += block
            }
            for i in 0..<width where inverse {
                x[base + i] *= signs?[i] ?? 1
            }
        }
    }

    static func grouped(_ x: inout [Float], width: Int,
                        _ g: HadamardFold.Grouping) {
        precondition(g.hd * g.nk * g.rep == width,
                     "hadamard: the grouping \(g) does not span \(width)")
        var tmp = [Float](repeating: 0, count: width)
        for r in 0 ..< x.count / width {
            let base = r * width
            for k in 0..<g.nk {
                for rep in 0..<g.rep {
                    for d in 0..<g.hd {
                        tmp[d + g.hd * (rep + g.rep * k)] =
                            x[base + d + g.hd * (k + g.nk * rep)]
                    }
                }
            }
            for i in 0..<width { x[base + i] = tmp[i] }
        }
    }

    static func apply(_ x: inout [Float], width: Int,
                      _ t: HadamardFold.Transform, inverse: Bool) {
        if !inverse, let g = t.grouping { grouped(&x, width: width, g) }
        rows(&x, width: width, block: t.block, signs: t.signs,
             inverse: inverse)
    }

    static func input(_ x: [Float], _ w: GGUFTensor,
                      _ fold: HadamardFold?) -> [Float] {
        var out = x
        if let t = fold?.transform(w) {
            apply(&out, width: w.dims[0], t, inverse: false)
        }
        return out
    }

    static func gathered(_ x: inout [Float], _ table: GGUFTensor,
                         _ fold: HadamardFold?) {
        if let t = fold?.inverseTransform(table) {
            apply(&x, width: table.dims[0], t, inverse: true)
        }
    }
}
