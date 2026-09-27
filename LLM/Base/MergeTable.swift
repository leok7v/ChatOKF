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

    private struct Chain {
        var sym: [Int32]
        var next: [Int32]
        var prev: [Int32]
        var live: [Bool]

        init(_ word: [Int32]) {
            let n = word.count
            sym = word
            next = (0 ..< n).map { i in Int32(i + 1 < n ? i + 1 : -1) }
            prev = (0 ..< n).map { i in Int32(i - 1) }
            live = [Bool](repeating: true, count: n)
        }

        mutating func join(_ at: Int, into: Int32) {
            let right = Int(next[at])
            let after = next[right]
            sym[at] = into
            live[right] = false
            next[at] = after
            if after >= 0 { prev[Int(after)] = Int32(at) }
        }

        func symbols() -> [Int32] {
            var out: [Int32] = []
            var at = sym.isEmpty ? -1 : 0
            while at >= 0 {
                out.append(sym[at])
                at = Int(next[at])
            }
            return out
        }
    }

    private static func entry(_ rank: Int32, _ at: Int) -> UInt64 {
        UInt64(UInt32(bitPattern: rank)) << 32 | UInt64(UInt32(at))
    }

    private static func smallerChild(_ heap: [UInt64], _ i: Int) -> Int {
        var child = 2 * i + 1
        if child + 1 < heap.count && heap[child + 1] < heap[child] {
            child += 1
        }
        return child
    }

    private static func push(_ heap: inout [UInt64], _ key: UInt64) {
        heap.append(key)
        var i = heap.count - 1
        while i > 0 && heap[(i - 1) / 2] > heap[i] {
            heap.swapAt(i, (i - 1) / 2)
            i = (i - 1) / 2
        }
    }

    private static func pop(_ heap: inout [UInt64]) -> UInt64 {
        let top = heap[0]
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var i = 0
            var child = smallerChild(heap, i)
            while child < heap.count && heap[child] < heap[i] {
                heap.swapAt(i, child)
                i = child
                child = smallerChild(heap, i)
            }
        }
        return top
    }

    private func pairRule(_ chain: Chain, _ at: Int) -> Rule? {
        var found: Rule? = nil
        if chain.live[at] && chain.next[at] >= 0 {
            let right = Int(chain.next[at])
            found = rule[MergeTable.key(chain.sym[at], chain.sym[right])]
        }
        return found
    }

    private func enqueue(_ heap: inout [UInt64], _ chain: Chain, _ at: Int) {
        if let found = pairRule(chain, at) {
            MergeTable.push(&heap, MergeTable.entry(found.rank, at))
        }
    }

    private func mergeRound(_ heap: inout [UInt64], _ chain: inout Chain,
                            _ touched: inout [Int]) {
        let rank = heap[0] >> 32
        while !heap.isEmpty && heap[0] >> 32 == rank {
            let at = Int(MergeTable.pop(&heap) & 0xffff_ffff)
            if let found = pairRule(chain, at),
               UInt64(UInt32(bitPattern: found.rank)) == rank {
                chain.join(at, into: found.merged)
                touched.append(at)
                if chain.prev[at] >= 0 { touched.append(Int(chain.prev[at])) }
            }
        }
    }

    private func mergeAll(_ word: [Int32]) -> [Int32] {
        var chain = Chain(word)
        var heap: [UInt64] = []
        var i = 0
        while i + 1 < word.count {
            enqueue(&heap, chain, i)
            i += 1
        }
        var touched: [Int] = []
        while !heap.isEmpty {
            mergeRound(&heap, &chain, &touched)
            for at in touched { enqueue(&heap, chain, at) }
            touched.removeAll(keepingCapacity: true)
        }
        return chain.symbols()
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
        let ids = word.count >= 2 ? mergeAll(word) : word
        var out: [String] = []
        out.reserveCapacity(ids.count)
        for id in ids {
            let i = Int(id)
            out.append(i < symbol.count ? symbol[i] : extra[i - symbol.count])
        }
        return out
    }
}
