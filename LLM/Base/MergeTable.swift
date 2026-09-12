import Foundation

struct CharPair: Hashable {
    let a: Character
    let b: Character
}

struct MergeTable: Sendable {

    private struct Rule {
        let rank: Int32
        let merged: Int32
    }

    private let symbol: [String]
    private let charId: [Character: Int32]
    private let rule: [UInt64: Rule]
    private let crossable: Set<CharPair>

    private static func key(_ a: Int32, _ b: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))
    }

    init(_ merges: [String]) {
        var text: [String] = []
        var ids = [String: Int32](minimumCapacity: merges.count * 2)
        var rules = [UInt64: Rule](minimumCapacity: merges.count)
        var cross = Set<CharPair>(minimumCapacity: 64_000)

        func intern(_ s: String) -> Int32 {
            let known = ids[s]
            let out: Int32
            if let known {
                out = known
            } else {
                out = Int32(text.count)
                ids[s] = out
                text.append(s)
            }
            return out
        }

        for (i, m) in merges.enumerated() {
            if let space = m.firstIndex(of: " ") {
                let a = String(m[..<space])
                let b = String(m[m.index(after: space)...])
                if let last = a.last, let first = b.first {
                    let pair = MergeTable.key(intern(a), intern(b))
                    rules[pair] = Rule(rank: Int32(i), merged: intern(a + b))
                    cross.insert(CharPair(a: last, b: first))
                }
            }
        }
        var chars = [Character: Int32](minimumCapacity: 8192)
        for (i, s) in text.enumerated() where s.count == 1 {
            chars[s.first!] = Int32(i)
        }
        symbol = text
        charId = chars
        rule = rules
        crossable = cross
    }

    func symbols(_ text: String) -> [String] {
        var out: [String] = []
        for segment in segments(text) {
            out.append(contentsOf: merged(segment))
        }
        return out
    }

    func segments(_ text: String) -> [ArraySlice<Character>] {
        let chars = Array(text)
        var out: [ArraySlice<Character>] = []
        var start = 0
        var i = 0
        while i + 1 < chars.count {
            if !crossable.contains(CharPair(a: chars[i], b: chars[i + 1])) {
                out.append(chars[start ... i])
                start = i + 1
            }
            i += 1
        }
        if start < chars.count { out.append(chars[start...]) }
        return out
    }

    func merged(_ token: ArraySlice<Character>) -> [String] {
        var extra: [String] = []
        var word: [Int32] = []
        word.reserveCapacity(token.count)
        for ch in token {
            if let id = charId[ch] {
                word.append(id)
            } else {
                word.append(Int32(symbol.count + extra.count))
                extra.append(String(ch))
            }
        }
        var merging = word.count >= 2
        while merging {
            var minRank = Int32.max
            var at = -1
            for i in 0 ..< (word.count - 1) {
                let found = rule[MergeTable.key(word[i], word[i + 1])]
                if let found, found.rank < minRank {
                    minRank = found.rank
                    at = i
                }
            }
            if at < 0 {
                merging = false
            } else {
                let a = word[at]
                let b = word[at + 1]
                let into = rule[MergeTable.key(a, b)]!.merged
                var next: [Int32] = []
                next.reserveCapacity(word.count)
                var i = 0
                while i < word.count {
                    if i < word.count - 1 && word[i] == a && word[i + 1] == b {
                        next.append(into)
                        i += 2
                    } else {
                        next.append(word[i])
                        i += 1
                    }
                }
                word = next
                merging = word.count >= 2
            }
        }
        var out: [String] = []
        out.reserveCapacity(word.count)
        for id in word {
            let i = Int(id)
            out.append(i < symbol.count ? symbol[i] : extra[i - symbol.count])
        }
        return out
    }
}
