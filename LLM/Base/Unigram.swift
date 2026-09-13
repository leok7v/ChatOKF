import Foundation

struct Unigram {

    static let space = "\u{2581}"

    private let gguf: GGUF
    private let base: UnsafeRawPointer
    private let starts: [Int32]
    private let scores: [Float]
    private let slots: [Int32]
    private let mask: Int
    private let maxPieceBytes: Int
    let unkId: Int32
    let unkScore: Float

    static func from(gguf g: GGUF) -> Unigram? {
        var result: Unigram? = nil
        if g.string("tokenizer.ggml.model") == "llama",
           let table = g.stringTable("tokenizer.ggml.tokens"),
           let scores = g.doubles("tokenizer.ggml.scores"),
           table.count == scores.count, table.count > 0 {
            let unk = Int32(g.int("tokenizer.ggml.unknown_token_id") ?? 3)
            result = Unigram(gguf: g, table: table, scores: scores,
                             unkId: unk)
        }
        return result
    }

    private init(gguf g: GGUF, table: GGUFStringTable, scores: [Double],
                 unkId: Int32) {
        var capacity = 1
        while capacity < table.count * 2 { capacity *= 2 }
        let wrap = capacity - 1
        var index = [Int32](repeating: -1, count: capacity)
        var offsets: [Int32] = []
        offsets.reserveCapacity(table.count)
        var longest = 0
        table.forEach { offset, bytes in
            let id = Int32(offsets.count)
            offsets.append(Int32(offset))
            longest = max(longest, bytes.count)
            var slot = Int(Unigram.fnv(bytes.baseAddress!, bytes.count)
                           & UInt64(wrap))
            while index[slot] >= 0 { slot = (slot + 1) & wrap }
            index[slot] = id
        }
        gguf = g
        base = table.base
        starts = offsets
        self.scores = scores.map { score in Float(score) }
        slots = index
        mask = wrap
        maxPieceBytes = longest
        self.unkId = unkId
        unkScore = (self.scores.min() ?? 0) - 10
    }

    private static func fnv(_ p: UnsafeRawPointer, _ n: Int) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for i in 0..<n {
            h ^= UInt64(p.load(fromByteOffset: i, as: UInt8.self))
            h = h &* 0x0000_0100_0000_01b3
        }
        return h
    }

    private func lookup(_ p: UnsafeRawPointer, _ n: Int)
        -> (id: Int32, score: Float)? {
        var slot = Int(Unigram.fnv(p, n) & UInt64(mask))
        var out: (id: Int32, score: Float)? = nil
        while out == nil && slots[slot] >= 0 {
            let id = slots[slot]
            let entry = base + Int(starts[Int(id)])
            let len = Int(entry.loadUnaligned(as: UInt64.self))
            if len == n && memcmp(entry + 8, p, n) == 0 {
                out = (id, scores[Int(id)])
            } else {
                slot = (slot + 1) & mask
            }
        }
        return out
    }

    static func normalize(_ text: String) -> String {
        let folded = text.precomposedStringWithCompatibilityMapping
        var out = ""
        out.reserveCapacity(folded.count + 8)
        var wantSpace = true
        for character in folded {
            if character.isWhitespace {
                wantSpace = true
            } else {
                if wantSpace { out += space }
                wantSpace = false
                out.append(character)
            }
        }
        return out
    }

    private static func boundaries(_ bytes: [UInt8]) -> [Bool] {
        var out = [Bool](repeating: false, count: bytes.count + 1)
        out[bytes.count] = true
        for i in 0..<bytes.count {
            out[i] = (bytes[i] & 0xC0) != 0x80
        }
        return out
    }

    private func relax(_ bytes: UnsafeRawPointer, _ count: Int, _ at: [Bool],
                       from: Int, _ best: inout [Float],
                       _ backId: inout [Int32], _ backFrom: inout [Int])
        -> Bool {
        let top = min(count, from + maxPieceBytes)
        var matched = false
        var upto = from + 1
        while upto <= top {
            if at[upto], let piece = lookup(bytes + from, upto - from) {
                matched = true
                let candidate = best[from] + piece.score
                if candidate > best[upto] {
                    best[upto] = candidate
                    backId[upto] = piece.id
                    backFrom[upto] = from
                }
            }
            upto += 1
        }
        return matched
    }

    private static func backtrack(_ backId: [Int32], _ backFrom: [Int],
                                  limit: Int) -> [Int32] {
        var out: [Int32] = []
        var position = backId.count - 1
        while position > 0 && backFrom[position] >= 0 {
            out.append(backId[position])
            position = backFrom[position]
        }
        out.reverse()
        if out.count > limit { out.removeLast(out.count - limit) }
        return out
    }

    func encode(_ text: String, limit: Int = Int.max) -> [Int32] {
        let bytes = Array(Unigram.normalize(text).utf8)
        let count = bytes.count
        let at = Unigram.boundaries(bytes)
        let floor = -Float.greatestFiniteMagnitude
        var best = [Float](repeating: floor, count: count + 1)
        var backId = [Int32](repeating: unkId, count: count + 1)
        var backFrom = [Int](repeating: -1, count: count + 1)
        var ids: [Int32] = []
        if count > 0 {
            best[0] = 0
            bytes.withUnsafeBytes { raw in
                let p = raw.baseAddress!
                for from in 0..<count {
                    if best[from] > floor && at[from] {
                        let matched = relax(p, count, at, from: from, &best,
                                            &backId, &backFrom)
                        if !matched {
                            var upto = from + 1
                            while upto < count && !at[upto] { upto += 1 }
                            let candidate = best[from] + unkScore
                            if candidate > best[upto] {
                                best[upto] = candidate
                                backId[upto] = unkId
                                backFrom[upto] = from
                            }
                        }
                    }
                }
            }
            ids = Unigram.backtrack(backId, backFrom, limit: limit)
        }
        return ids
    }
}
