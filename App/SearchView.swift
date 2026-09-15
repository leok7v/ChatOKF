import Chat
import Foundation

enum ConversationSearch {

    static let minimumQuery = 2

    static func words(_ query: String) -> [String] {
        query.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init)
    }

    static func active(_ query: String) -> Bool {
        let all = words(query)
        return !all.isEmpty
            && all.reduce(0) { sum, word in sum + word.count }
                >= minimumQuery
    }

    static func rank(_ list: [ConversationStore.Convo],
                     _ index: [UUID: [String: Int]],
                     _ query: String) -> [ConversationStore.Convo] {
        let wanted = words(query).map { word in word.lowercased() }
        var scored: [(convo: ConversationStore.Convo, score: Int)] = []
        scored.reserveCapacity(list.count)
        for convo in list {
            let counts = index[convo.id] ?? [:]
            var total = 0
            var landed = 0
            for want in wanted {
                let gain = score(want, counts)
                total += gain
                if gain > 0 { landed += 1 }
            }
            if landed == wanted.count { scored.append((convo, total)) }
        }
        return scored
            .sorted { a, b in
                a.score == b.score
                    ? a.convo.updated > b.convo.updated
                    : a.score > b.score
            }
            .map { pair in pair.convo }
    }

    static func score(_ want: String,
                      _ counts: [String: Int]) -> Int {
        var total = 0
        for (word, count) in counts {
            if word == want {
                total += 100 * count
            } else if word.hasPrefix(want) {
                total += 10 * count
            } else if word.contains(want) {
                total += count
            }
        }
        return total
    }

    static func reason(_ convo: ConversationStore.Convo,
                       _ query: String) -> String? {
        var result: String? = nil
        for word in words(query) where result == nil {
            if convo.title.range(of: word,
                                 options: .caseInsensitive) == nil {
                result = found(word, in: convo)
            }
        }
        return result
    }

    private static func found(_ word: String,
                              in convo: ConversationStore.Convo) -> String? {
        var result: String? = nil
        for m in convo.messages where result == nil {
            if let r = m.text.range(of: word, options: .caseInsensitive) {
                result = snippet(m.text, r)
            }
        }
        return result
    }

    static func snippet(_ text: String,
                        _ range: Range<String.Index>) -> String {
        let pad = 24
        let lo = text.index(range.lowerBound, offsetBy: -pad,
                            limitedBy: text.startIndex) ?? text.startIndex
        let hi = text.index(range.upperBound, offsetBy: pad,
                            limitedBy: text.endIndex) ?? text.endIndex
        var out = String(text[lo ..< hi])
            .replacingOccurrences(of: "\n", with: " ")
        if lo > text.startIndex { out = "\u{2026}" + out }
        if hi < text.endIndex { out += "\u{2026}" }
        return out
    }

}

enum MemorySearch {

    static func active(_ query: String) -> Bool {
        ConversationSearch.active(query)
    }

    static func rank(_ rows: [MemoryRow], _ query: String) -> [MemoryRow] {
        let wanted = ConversationSearch.words(query)
        var scored: [(row: MemoryRow, score: Int)] = []
        scored.reserveCapacity(rows.count)
        for row in rows {
            let counts = weigh(row)
            var total = 0
            var landed = 0
            for want in wanted {
                let gain = ConversationSearch.score(want, counts)
                total += gain
                if gain > 0 { landed += 1 }
            }
            if landed == wanted.count { scored.append((row, total)) }
        }
        return scored
            .sorted { a, b in
                a.score == b.score
                    ? a.row.updated > b.row.updated
                    : a.score > b.score
            }
            .map { pair in pair.row }
    }

    private static let titleWeight = 5

    private static func weigh(_ row: MemoryRow) -> [String: Int] {
        var counts: [String: Int] = [:]
        add(row.title, titleWeight, &counts)
        add(row.id.replacingOccurrences(of: "/", with: " "), titleWeight,
            &counts)
        add(row.description, 1, &counts)
        add(row.tags.joined(separator: " "), 1, &counts)
        return counts
    }

    private static func add(_ text: String, _ weight: Int,
                            _ counts: inout [String: Int]) {
        for word in ConversationSearch.words(text) {
            counts[word, default: 0] += weight
        }
    }

}
