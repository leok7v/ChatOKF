import Foundation

public enum TopicTitle {

    public static let stopWords: Set<String> = [
        "about", "after", "all", "also", "and", "any", "are", "but", "can",
        "did", "does", "each", "for", "from", "get", "give", "had", "has",
        "have", "her", "here", "him", "his", "how", "into", "its", "just",
        "like", "make", "many", "may", "more", "most", "much", "not", "now",
        "one", "only", "other", "our", "out", "over", "own", "same", "she",
        "should", "some", "such", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "use", "used",
        "using", "very", "want", "wants", "was", "way", "were", "what",
        "when", "where", "which", "while", "who", "why", "will", "with",
        "would", "you", "your",
    ]

    public static let words = 4
    public static let shortest = 3
    public static let repeats = 2

    public static func from(_ texts: [String]) -> String {
        var counts: [String: Int] = [:]
        var seenAt: [String: Int] = [:]
        var at = 0
        for text in texts {
            for token in text.lowercased().split(whereSeparator: { c in
                !c.isLetter && !c.isNumber
            }) {
                let word = String(token)
                at += 1
                let topical = word.count >= TopicTitle.shortest
                    && !TopicTitle.stopWords.contains(word)
                    && word.contains(where: { c in c.isLetter })
                if topical {
                    counts[word, default: 0] += 1
                    if seenAt[word] == nil { seenAt[word] = at }
                }
            }
        }
        counts = counts.filter { pair in pair.value >= TopicTitle.repeats }
        let ranked = counts.sorted { a, b in
            a.value != b.value ? a.value > b.value
                : (seenAt[a.key] ?? 0) < (seenAt[b.key] ?? 0)
        }
        let picked = ranked.prefix(TopicTitle.words)
            .map { pair in pair.key }
            .sorted { a, b in (seenAt[a] ?? 0) < (seenAt[b] ?? 0) }
        return picked
            .map { word in word.prefix(1).uppercased() + word.dropFirst() }
            .joined(separator: " ")
    }
}
