import Foundation

// SentencePiece-flavoured and driven by the GGUF metadata (metaspace, byte
// fallback, ignore_merges, token types): a second implementation, not a branch.
public struct GemmaTokenizer: Sendable {
    private let vocab: [String: Int32]
    private let pieces: [String]
    private let types: [Int]
    private let table: MergeTable
    private let byteId: [Int32]
    private let metaspace: String
    private let byteFallback: Bool
    private let ignoreMerges: Bool
    private let specials: SpecialIndex

    public let eosId: Int32
    // A runtime comparing against the scalar alone runs past <end_of_turn>.
    public let eosIds: Set<Int32>
    public let bosId: Int32
    public let padId: Int32
    public let unkId: Int32

    init(gguf g: GGUF) throws {
        let listed = g.strings("tokenizer.ggml.tokens")
        if listed == nil {
            throw GGUFErr.parse("tokenizer.ggml.tokens missing")
        }
        let toks = listed!
        pieces = toks
        types = g.ints("tokenizer.ggml.token_type") ?? []
        metaspace = g.string("tokenizer.ggml.metaspace") ?? "\u{2581}"
        byteFallback = g.bool("tokenizer.ggml.byte_fallback") ?? false
        ignoreMerges = g.bool("tokenizer.ggml.ignore_merges") ?? false

        var v = [String: Int32](minimumCapacity: toks.count)
        for (i, p) in toks.enumerated() { v[p] = Int32(i) }
        vocab = v

        table = MergeTable(g.strings("tokenizer.ggml.merges") ?? [])

        var bytes = [Int32](repeating: -1, count: 256)
        for (i, p) in toks.enumerated()
        where i < types.count && types[i] == 6 {
            let hex = p.dropFirst(3).dropLast()
            if let b = UInt8(hex, radix: 16) { bytes[Int(b)] = Int32(i) }
        }
        byteId = bytes

        var sp: [(String, Int32)] = []
        for (i, p) in toks.enumerated()
        where i < types.count && types[i] == 3 {
            sp.append((p, Int32(i)))
        }
        sp.sort { a, b in a.0.count > b.0.count }
        specials = SpecialIndex(sp)

        let ids = g.ints("tokenizer.ggml.eos_token_ids")?
            .map { id in Int32(id) }
        let scalar = Int32(g.int("tokenizer.ggml.eos_token_id") ?? 1)
        eosId = scalar
        eosIds = Set(ids ?? [scalar])
        bosId = Int32(g.int("tokenizer.ggml.bos_token_id") ?? 2)
        padId = Int32(g.int("tokenizer.ggml.padding_token_id") ?? 0)
        unkId = Int32(g.int("tokenizer.ggml.unknown_token_id") ?? 3)
    }

    public var vocabCount: Int { pieces.count }

    public var bosToken: String {
        let i = Int(bosId)
        return i >= 0 && i < pieces.count ? pieces[i] : ""
    }

    public func encode(_ text: String, addSpecial: Bool = true) -> [Int32] {
        var out: [Int32] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            let hit = addSpecial ? specials.first(in: rest) : nil
            if let hit {
                let head = rest[rest.startIndex ..< hit.range.lowerBound]
                out.append(contentsOf: encodePlain(String(head)))
                out.append(hit.id)
                rest = rest[hit.range.upperBound...]
            } else {
                out.append(contentsOf: encodePlain(String(rest)))
                rest = rest[rest.endIndex...]
            }
        }
        return out
    }

    private func encodePlain(_ text: String) -> [Int32] {
        var out: [Int32] = []
        if !text.isEmpty {
            let normalized = text.replacingOccurrences(of: " ",
                                                       with: metaspace)
            if ignoreMerges, let id = vocab[normalized] {
                out = [id]
            } else {
                for sym in table.symbols(normalized) {
                    if let id = vocab[sym] {
                        out.append(id)
                    } else {
                        out.append(contentsOf: fallback(sym))
                    }
                }
            }
        }
        return out
    }

    private func fallback(_ sym: String) -> [Int32] {
        var out: [Int32] = []
        if byteFallback {
            for b in sym.utf8 {
                let id = byteId[Int(b)]
                out.append(id >= 0 ? id : unkId)
            }
        } else {
            out.append(unkId)
        }
        return out
    }

    // The gate proved all three are needed: byte fallback for the emoji probe,
    // the metaspace rewrite for the leading-space one.
    public func decodeBytes(_ ids: [Int32]) -> [UInt8] {
        var out: [UInt8] = []
        for id in ids {
            let i = Int(id)
            if i >= 0 && i < pieces.count {
                let t = i < types.count ? types[i] : 1
                if t == 6 {
                    let hex = pieces[i].dropFirst(3).dropLast()
                    if let b = UInt8(hex, radix: 16) { out.append(b) }
                } else if t == 3 {
                    out.append(contentsOf: pieces[i].utf8)
                } else {
                    out.append(contentsOf: pieces[i]
                        .replacingOccurrences(of: metaspace, with: " ").utf8)
                }
            }
        }
        return out
    }

    public func decode(_ ids: [Int32]) -> String {
        String(decoding: decodeBytes(ids), as: UTF8.self)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] { decodeBytes([id]) }
}

extension GemmaTokenizer: Tokenizing {}
