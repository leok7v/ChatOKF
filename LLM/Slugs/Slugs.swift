import Foundation

protocol VectorIndex {
    var count: Int { get }
    func search(_ emb: [Float], topK: Int) -> [(index: Int, distance: Int)]
    func record(_ i: Int) -> (id: String, title: String)
}

// Both read straight from the GGUF mmap, so a live engine pins only ~36 MB
// of clean pages: cheap per query or held indefinitely.
public final class WikiSlugs {
    private let gguf: GGUF
    let embedder: MiniLM
    private let index: SignIndex

    public var articleCount: Int { index.count }

    public static var bundledModel: URL? {
        Res.url("minilm", "gguf",
               dev: URL(fileURLWithPath: #filePath)
                   .deletingLastPathComponent())
    }

    // Fails when the file is not a GGUF or carries no SLGX index trailer.
    public init?(ggufPath: String) {
        guard let g = try? GGUF(path: ggufPath),
              let idx = SignIndex(gguf: g) else {
            return nil
        }
        gguf = g
        embedder = MiniLM(gguf: g)
        index = idx
    }

    public func embed(_ text: String) -> [Float] {
        embedder.embed(text)
    }

    public func query(_ text: String, topK: Int = 5) -> [SlugHit] {
        let emb = embedder.embed(text)
        var hits: [SlugHit] = []
        for hit in index.search(emb, topK: topK) {
            let r = index.record(hit.index)
            hits.append(SlugHit(id: r.id, title: r.title,
                                distance: hit.distance))
        }
        return hits
    }

    // Exact-title rescue for an embedding miss: the LONGEST title (>= 3 chars)
    // found as whole words. A linear scan, so only on a low-confidence result.
    public func titleMatch(_ text: String) -> SlugHit? {
        let hay = " " + WikiSlugs.foldWords(text) + " "
        var best: (id: String, title: String)? = nil
        var bestLen = 2
        for i in 0 ..< index.count {
            let r = index.record(i)
            let folded = WikiSlugs.foldWords(r.title)
            if folded.count > bestLen,
               !WikiSlugs.functionWords.contains(folded),
               hay.contains(" " + folded + " ") {
                best = r
                bestLen = folded.count
            }
        }
        return best.map { r in
            SlugHit(id: r.id, title: r.title, distance: 0)
        }
    }

    // Single function words are titles too (the corpus has "This") and never
    // a question's SUBJECT; multi-word titles pass, "The Who" is real.
    private static let functionWords: Set<String> = [
        "a", "an", "the", "this", "that", "these", "those", "it", "its",
        "is", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "have", "has", "had", "will", "would", "can", "could",
        "should", "shall", "may", "might", "must", "and", "or", "but",
        "not", "no", "yes", "if", "then", "than", "so", "as", "of", "in",
        "on", "at", "to", "for", "with", "by", "from", "about", "into",
        "over", "under", "again", "there", "here", "when", "where", "why",
        "how", "what", "who", "whom", "which", "you", "your", "yours",
        "me", "my", "mine", "we", "us", "our", "ours", "he", "him", "his",
        "she", "her", "hers", "they", "them", "their", "theirs", "all",
        "any", "some", "such", "only", "very", "just", "also", "tell",
        "say", "says", "know", "like", "please", "thing", "things",
    ]

    // Whole-word match: "St. Louis" and "st louis" compare equal.
    static func foldWords(_ s: String) -> String {
        var out = ""
        var pendingSpace = false
        for c in s.lowercased() {
            if c.isLetter || c.isNumber {
                if pendingSpace && !out.isEmpty { out.append(" ") }
                pendingSpace = false
                out.append(c)
            } else {
                pendingSpace = true
            }
        }
        return out
    }
}
