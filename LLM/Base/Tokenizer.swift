import Foundation

public struct Tokenizer: Sendable {
    // Regex matching is thread-safe; Regex is just not marked Sendable yet.
    private struct RegexBox: @unchecked Sendable {
        let regex: Regex<AnyRegexOutput>
    }

    private let vocab: [String: Int32]
    private let idToBytes: [Int32: [UInt8]]
    private let table: MergeTable
    private let byteToUni: [UInt8: Character]
    private let specials: SpecialIndex
    private let splitRegex: RegexBox
    public let eosId: Int32
    public private(set) var eosIds: Set<Int32>

    public init(modelsDir: URL) throws {
        let url = modelsDir.appendingPathComponent("tokenizer.json")
        let raw = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: raw)
            as! [String: Any]
        let model = root["model"] as! [String: Any]

        var v = [String: Int32](minimumCapacity: 250_000)
        for (piece, id) in model["vocab"] as! [String: Int] {
            v[piece] = Int32(id)
        }
        var joined: [String] = []
        for m in model["merges"] as! [Any] {
            var key = m as? String
            if key == nil, let pair = m as? [String], pair.count == 2 {
                key = pair[0] + " " + pair[1]
            }
            if let key { joined.append(key) }
        }

        let b2u = Tokenizer.bytesToUnicode()
        var u2b = [Character: UInt8](minimumCapacity: 256)
        for (b, c) in b2u { u2b[c] = b }

        var special: [(String, Int32)] = []
        for t in root["added_tokens"] as! [[String: Any]] {
            let content = t["content"] as! String
            let id = Int32(t["id"] as! Int)
            v[content] = id
            special.append((content, id))
        }
        special.sort { $0.0.count > $1.0.count }

        let specialSet = Set(special.map { $0.0 })
        var i2b = [Int32: [UInt8]](minimumCapacity: 250_000)
        for (piece, id) in v {
            if specialSet.contains(piece) {
                i2b[id] = Array(piece.utf8)
            } else {
                var bytes: [UInt8] = []
                bytes.reserveCapacity(piece.count)
                for ch in piece where u2b[ch] != nil {
                    bytes.append(u2b[ch]!)
                }
                i2b[id] = bytes
            }
        }

        let pre = root["pre_tokenizer"] as! [String: Any]
        let seq = pre["pretokenizers"] as! [[String: Any]]
        let patt = (seq[0]["pattern"] as! [String: Any])["Regex"]
            as! String

        self.vocab = v
        self.idToBytes = i2b
        self.table = MergeTable(joined)
        self.byteToUni = b2u
        self.specials = SpecialIndex(special)
        self.splitRegex = RegexBox(regex: try Regex(patt))
        self.eosId = v["<|im_end|>"] ?? 0
        self.eosIds = [self.eosId]
        addStops(Tokenizer.stopIds(besideSet: modelsDir))
    }

    public mutating func addStops(_ ids: [Int32]) {
        for id in ids where id >= 0 { eosIds.insert(id) }
    }

    public static func stopIds(besideSet dir: URL) -> [Int32] {
        stopIds(generationConfig:
            dir.appendingPathComponent("generation_config.json"))
    }

    public static func stopIds(generationConfig url: URL) -> [Int32] {
        stopIds(generationConfigText: (try? Data(contentsOf: url))
            .flatMap { d in String(data: d, encoding: .utf8) })
    }

    public static func stopIds(generationConfigText text: String?) -> [Int32] {
        var out: [Int32] = []
        if let text, let data = text.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data))
               as? [String: Any] {
            if let one = root["eos_token_id"] as? Int {
                out = [Int32(one)]
            } else if let many = root["eos_token_id"] as? [Int] {
                out = many.map(Int32.init)
            }
        }
        return out
    }

    static let qwenPretokenizer =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    init(gguf g: GGUF) throws {
        let listed = g.strings("tokenizer.ggml.tokens")
        if listed == nil {
            throw GGUFErr.parse("tokenizer.ggml.tokens missing")
        }
        let tokens = listed!
        let merges = g.strings("tokenizer.ggml.merges") ?? []
        let types = g.ints("tokenizer.ggml.token_type") ?? []

        var v = [String: Int32](minimumCapacity: tokens.count)
        for (i, piece) in tokens.enumerated() { v[piece] = Int32(i) }

        let b2u = Tokenizer.bytesToUnicode()
        var u2b = [Character: UInt8](minimumCapacity: 256)
        for (b, c) in b2u { u2b[c] = b }

        var special: [(String, Int32)] = []
        var specialSet = Set<String>()
        for (i, piece) in tokens.enumerated() {
            let t = i < types.count ? types[i] : 1
            if t == 3 || t == 4 { special.append((piece, Int32(i))); specialSet.insert(piece) }
        }
        special.sort { $0.0.count > $1.0.count }

        var i2b = [Int32: [UInt8]](minimumCapacity: tokens.count)
        for (piece, id) in v {
            if specialSet.contains(piece) {
                i2b[id] = Array(piece.utf8)
            } else {
                var bytes: [UInt8] = []
                bytes.reserveCapacity(piece.count)
                for ch in piece where u2b[ch] != nil { bytes.append(u2b[ch]!) }
                i2b[id] = bytes
            }
        }

        self.vocab = v
        self.idToBytes = i2b
        self.table = MergeTable(merges)
        self.byteToUni = b2u
        self.specials = SpecialIndex(special)
        self.splitRegex = RegexBox(regex: try Regex(Tokenizer.qwenPretokenizer))
        let stops = g.ints("tokenizer.ggml.eos_token_id")
        self.eosId = Int32(stops?.first
            ?? (g.int("tokenizer.ggml.eos_token_id") ?? 0))
        self.eosIds = Set((stops ?? [Int(self.eosId)]).map(Int32.init))
        addStops((g.ints("tokenizer.ggml.eos_token_ids") ?? [])
            .map(Int32.init))
        addStops([specialSet.contains(Tokenizer.endOfText)
            ? (v[Tokenizer.endOfText] ?? -1) : -1])
        addStops(Tokenizer.stopIds(
            generationConfigText: g.string(Tokenizer.generationConfigKey)))
    }

    public static let generationConfigKey = "general.generation_config_json"

    static let endOfText = "<|endoftext|>"

    static func bytesToUnicode() -> [UInt8: Character] {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var map = [UInt8: Character](minimumCapacity: 256)
        for (b, c) in zip(bs, cs) {
            map[UInt8(b)] = Character(UnicodeScalar(c)!)
        }
        return map
    }

    public func encode(_ text: String,
                       addSpecial: Bool = true) -> [Int32] {
        var out: [Int32] = []
        if addSpecial {
            var rest = Substring(text)
            while !rest.isEmpty {
                if let hit = specials.first(in: rest) {
                    let head = rest[rest.startIndex ..< hit.range.lowerBound]
                    out.append(contentsOf: encodePlain(String(head)))
                    out.append(hit.id)
                    rest = rest[hit.range.upperBound...]
                } else {
                    out.append(contentsOf: encodePlain(String(rest)))
                    rest = rest[rest.endIndex...]
                }
            }
        } else {
            out = encodePlain(text)
        }
        return out
    }

    private func encodePlain(_ text: String) -> [Int32] {
        var out: [Int32] = []
        if !text.isEmpty {
            for match in text.matches(of: splitRegex.regex) {
                var mapped = ""
                for byte in String(text[match.range]).utf8 {
                    mapped.append(byteToUni[byte]!)
                }
                for sym in table.symbols(mapped) where vocab[sym] != nil {
                    out.append(vocab[sym]!)
                }
            }
        }
        return out
    }


    public func decode(_ ids: [Int32]) -> String {
        String(decoding: decodeBytes(ids), as: UTF8.self)
    }

    public func decodeBytes(_ ids: [Int32]) -> [UInt8] {
        var bytes: [UInt8] = []
        for id in ids where idToBytes[id] != nil {
            bytes.append(contentsOf: idToBytes[id]!)
        }
        return bytes
    }

    public var vocabCount: Int { idToBytes.count }

    public func vocabBytes() -> [[UInt8]] {
        var out = [[UInt8]](repeating: [], count: idToBytes.count)
        for (id, bytes) in idToBytes where Int(id) < out.count && id >= 0 {
            out[Int(id)] = bytes
        }
        return out
    }

    // A SCALAR boundary, not a grapheme one: a multi-scalar emoji streams
    // scalar by scalar and the UI reassembles the cluster.
    public static func completeUTF8Count(_ b: [UInt8]) -> Int {
        var j = b.count - 1
        var cont = 0
        while j >= 0, (b[j] & 0xC0) == 0x80 { cont += 1; j -= 1 }
        var n = b.count
        if j >= 0 {
            let lead = b[j]
            let need: Int
            if lead & 0x80 == 0 {
                need = 0
            } else if lead & 0xE0 == 0xC0 {
                need = 1
            } else if lead & 0xF0 == 0xE0 {
                need = 2
            } else if lead & 0xF8 == 0xF0 {
                need = 3
            } else {
                need = 0
            }
            n = cont == need ? b.count : j
        }
        return n
    }
}

// Diffs by BYTE length, not grapheme count: a trailing scalar that merges
// into the current grapheme (U+FE0F, a ZWJ) does not grow the count.
public struct StreamDecoder {
    private var shownBytes = 0

    public init() {}

    public mutating func step(_ ids: [Int32], _ tokenizer: Tokenizer) -> String {
        let bytes = tokenizer.decodeBytes(ids)
        let n = Tokenizer.completeUTF8Count(bytes)
        var piece = ""
        if n > shownBytes {
            piece = String(decoding: bytes[shownBytes ..< n], as: UTF8.self)
            shownBytes = n
        }
        return piece
    }
}

extension Tokenizer: Tokenizing {
    public var bosToken: String { "" }
}
