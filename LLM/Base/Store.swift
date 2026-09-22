import Foundation

public protocol Embedder: AnyObject {
    var dim: Int { get }
    var name: String { get }
    var queryPrefix: String { get }
    var passagePrefix: String { get }
    var relevanceFloor: Float { get }
    func embed(_ text: String) -> [Float]
    func states(_ text: String) -> [Float]
    func tokens(_ text: String) -> [Int32]
}

public extension Embedder {
    func embedQuery(_ text: String) -> [Float] {
        embed(queryPrefix + text)
    }

    func embedPassage(_ text: String) -> [Float] {
        embed(passagePrefix + text)
    }
}

public final class BertEmbedder: Embedder {

    private let gguf: GGUF
    private let model: MiniLM
    private let multilingual: Bool
    public let name: String

    public var dim: Int { model.dim }
    public var queryPrefix: String { multilingual ? "query: " : "" }
    public var passagePrefix: String { multilingual ? "passage: " : "" }
    public var relevanceFloor: Float { multilingual ? 0.806 : 0.5 }

    public static var bundledMultilingual: URL? {
        MiniLM.bundledMultilingual
    }

    public static func load(ggufPath: String) -> BertEmbedder? {
        var result: BertEmbedder? = nil
        if let gguf = try? GGUF(path: ggufPath) {
            result = BertEmbedder(gguf: gguf)
        }
        return result
    }

    private init(gguf: GGUF) {
        self.gguf = gguf
        model = MiniLM(gguf: gguf)
        multilingual = gguf.string("tokenizer.ggml.model") == "llama"
        name = multilingual ? "multilingual-e5-small" : "all-MiniLM-L6-v2"
    }

    public func embed(_ text: String) -> [Float] { model.embed(text) }

    public func states(_ text: String) -> [Float] { model.encode(text) }

    public func tokens(_ text: String) -> [Int32] { model.tokenize(text) }
}

public enum Trust: String {
    case unverified
    case unconfirmed
    case machine
    case human
}

public struct Concept {
    public var id: String
    public var path: URL
    public var type: String
    public var title: String
    public var description: String
    public var tags: [String]
    public var status: String
    public var staleAfter: Date?
    public var generatedBy: String
    public var verifiers: [String]
    public var extraFrontmatter: [String]
    public var body: String
    public var links: [String]
    public var backlinks: [String] = []
    public var hash: UInt64
    public var vector: [Float] = []
    public var bodyVector: [Float] = []

    public var trust: Trust {
        var result = Trust.unverified
        if !verifiers.isEmpty {
            result = .machine
            for actor in verifiers where actor.hasPrefix("human:") {
                result = .human
            }
        } else if !generatedBy.isEmpty
                    && !generatedBy.hasPrefix("human:") {
            result = .unconfirmed
        }
        return result
    }

    public var passage: String {
        Concept.passage(title: title, description: description, tags: tags,
                        type: type)
    }

    public static func passage(title: String, description: String,
                               tags: [String], type: String) -> String {
        var out = title
        if !description.isEmpty { out += ". " + description }
        if !tags.isEmpty {
            out += " [" + tags.joined(separator: ", ") + "]"
        }
        if !type.isEmpty { out += " (" + type + ")" }
        return out
    }

    public var bodyPassage: String { String(body.prefix(2000)) }

    public var isDeprecated: Bool { status == "deprecated" }

    public var isStale: Bool {
        var result = false
        if let deadline = staleAfter { result = Date() >= deadline }
        return result
    }
}

public struct Hit {
    public let concept: Concept
    public let score: Float
    public let relevance: Float
    public let terms: [String]
}

public struct GrepHit {
    public let id: String
    public let title: String
    public let line: Int
    public let text: String
}

public struct RemovedLink: Sendable {
    public let referrer: String
    public let markup: String
    public let label: String

    public init(referrer: String, markup: String, label: String) {
        self.referrer = referrer
        self.markup = markup
        self.label = label
    }
}

public struct Proposal {
    public let kind: Kind
    public let score: Float
    public let members: [String]

    public enum Kind: String {
        case dangling
        case merge
        case link
        case skill
        case orphan
    }
}

public struct WriteReport {
    public let url: URL
    public let candidates: [String]
    public let strayArea: String
}

public struct Paragraph {
    public let from: Int
    public let to: Int
    public let text: String
}

public struct Filter {
    public var type: String = ""
    public var tags: [String] = []
    public var area: String = ""
    public var deprecated: Bool = false

    public init(type: String = "", tags: [String] = [], area: String = "",
                deprecated: Bool = false) {
        self.type = type
        self.tags = tags
        self.area = area
        self.deprecated = deprecated
    }

    public var isEmpty: Bool {
        type.isEmpty && tags.isEmpty && area.isEmpty
    }

    public func admits(_ concept: Concept) -> Bool {
        let live = deprecated || !concept.isDeprecated
        let typed = type.isEmpty
            || concept.type.lowercased() == type.lowercased()
        let placed = area.isEmpty || concept.id.hasPrefix(area + "/")
        var tagged = true
        for tag in tags where !concept.tags.contains(tag) {
            tagged = false
        }
        return live && typed && placed && tagged
    }
}

public struct SearchResult {
    public let hits: [Hit]
    public let embedSeconds: Double
    public let scanSeconds: Double
    public let literalSeconds: Double
    public let standout: Float
}

struct Parsed {
    var fields: [String: String] = [:]
    var lists: [String: [String]] = [:]
    var unknown: [String] = []
    var body: String = ""
}

enum Frontmatter {

    static let known: Set<String> = ["type", "title", "description",
                                     "tags", "status", "stale_after"]

    static func unquote(_ text: String) -> String {
        var out = text
        let quoted = out.count >= 2
            && ((out.hasPrefix("\"") && out.hasSuffix("\""))
                || (out.hasPrefix("'") && out.hasSuffix("'")))
        if quoted { out = String(out.dropFirst().dropLast()) }
        return out
    }

    private static func byValues(_ text: String) -> [String] {
        var out: [String] = []
        var rest = Substring(text)
        while let found = rest.range(of: "by:") {
            let head = rest[rest.startIndex..<found.lowerBound].last
            let standalone = head == nil || !(head!.isLetter
                || head! == "-" || head! == "_")
            let after = rest[found.upperBound...]
            let actor = unquote(String(after.prefix { character in
                character != "," && character != "}"
            }).trimmingCharacters(in: .whitespaces))
            if standalone && !actor.isEmpty { out.append(actor) }
            rest = after
        }
        return out
    }

    static func actors(_ lines: [String], under key: String) -> [String] {
        var out: [String] = []
        var inside = false
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let indented = line.hasPrefix(" ") || line.hasPrefix("\t")
            if !indented { inside = trimmed.hasPrefix(key + ":") }
            if inside { out += byValues(trimmed) }
        }
        return out
    }

    static func dropping(_ lines: [String], keys: [String]) -> [String] {
        var out: [String] = []
        var inside = false
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let indented = line.hasPrefix(" ") || line.hasPrefix("\t")
            if !indented {
                inside = keys.contains { key in trimmed.hasPrefix(key + ":") }
            }
            if !inside { out.append(line) }
        }
        return out
    }

    private static func closingIndex(_ lines: [String]) -> Int {
        var found = -1
        if lines.first?.trimmingCharacters(in: .whitespaces) == "---" {
            var i = 1
            while i < lines.count && found < 0 {
                if lines[i].trimmingCharacters(in: .whitespaces) == "---" {
                    found = i
                }
                i += 1
            }
        }
        return found
    }

    private static func splitList(_ text: String) -> [String] {
        let inner = String(text.dropFirst().dropLast())
        return inner.components(separatedBy: ",")
            .map { item in
                unquote(item.trimmingCharacters(in: .whitespaces))
            }
            .filter { item in !item.isEmpty }
    }

    private static func absorb(_ line: String, _ key: inout String,
                               _ out: inout Parsed) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let indented = line.hasPrefix(" ") || line.hasPrefix("\t")
        let listItem = indented && trimmed.hasPrefix("- ")
        let colon = trimmed.firstIndex(of: ":")
        if listItem && !key.isEmpty {
            let item = String(trimmed.dropFirst(2))
                .trimmingCharacters(in: .whitespaces)
            if known.contains(key) {
                out.lists[key, default: []].append(unquote(item))
            } else {
                out.unknown.append(line)
            }
        } else if indented && !key.isEmpty {
            if known.contains(key) {
                let prior = out.fields[key] ?? ""
                out.fields[key] = prior.isEmpty
                    ? trimmed : prior + " " + trimmed
            } else {
                out.unknown.append(line)
            }
        } else if let colon = colon {
            key = String(trimmed[trimmed.startIndex..<colon])
                .trimmingCharacters(in: .whitespaces)
            let rest = String(trimmed[trimmed.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            if !known.contains(key) {
                out.unknown.append(line)
            } else if rest.hasPrefix("[") && rest.hasSuffix("]") {
                out.lists[key] = splitList(rest)
            } else if !rest.isEmpty {
                out.fields[key] = unquote(rest)
            }
        } else {
            out.unknown.append(line)
        }
    }

    static func parse(_ text: String) -> Parsed {
        var out = Parsed()
        let lines = text.components(separatedBy: "\n")
        let end = closingIndex(lines)
        if end < 0 {
            out.body = text
        } else {
            var key = ""
            for i in 1..<end {
                if !lines[i].trimmingCharacters(in: .whitespaces).isEmpty {
                    absorb(lines[i], &key, &out)
                }
            }
            out.body = lines[(end + 1)...].joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return out
    }

    private static func wrap(_ key: String, _ value: String) -> String {
        let limit = 72
        var out = key + ":"
        var column = out.count
        for word in value.split(separator: " ") {
            if column + 1 + word.count > limit {
                out += "\n  " + word
                column = 2 + word.count
            } else {
                out += " " + word
                column += 1 + word.count
            }
        }
        return out + "\n"
    }

    static func emit(_ concept: Concept) -> String {
        var out = "---\ntype: " + concept.type + "\n"
        if !concept.title.isEmpty {
            out += wrap("title", concept.title)
        }
        if !concept.description.isEmpty {
            out += wrap("description", concept.description)
        }
        if !concept.tags.isEmpty {
            out += "tags: [" + concept.tags.joined(separator: ", ") + "]\n"
        }
        if !concept.status.isEmpty {
            out += "status: " + concept.status + "\n"
        }
        if let deadline = concept.staleAfter {
            out += "stale_after: " + Store.iso.string(from: deadline) + "\n"
        }
        for line in concept.extraFrontmatter { out += line + "\n" }
        out += "---\n\n" + concept.body + "\n"
        return out
    }
}

public final class Store {

    nonisolated(unsafe) public static let iso: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static let linkPattern = try! NSRegularExpression(
        pattern: "\\[[^\\]]*\\]\\(([^)\\s]+\\.md)\\)", options: [])

    static let reserved: Set<String> = ["index", "log"]

    static let okfSpecVersion = "0.2"
    static let versionKey = "okf_version:"

    public let root: URL
    private let embedder: Embedder
    private let sidecar: URL
    public private(set) var concepts: [Concept] = []
    public private(set) var byId: [String: Int] = [:]
    public private(set) var embeddedCount = 0
    public private(set) var loadSeconds = 0.0
    public private(set) var okfVersion = ""
    private var anyPostings: [String: [Int]] = [:]
    private var headPostings: [String: Set<Int>] = [:]

    public init(root: URL, embedder: Embedder) {
        self.root = root
        self.embedder = embedder
        sidecar = root.appendingPathComponent(
            ".okf-vectors-" + embedder.name + ".bin")
    }

    public var owner: Concept? {
        var result = concepts.first { concept in
            concept.tags.contains("owner")
        }
        if result == nil {
            result = concepts.first { concept in concept.type == "Person" }
        }
        return result
    }

    public func concept(_ id: String) -> Concept? {
        var result: Concept? = nil
        if let index = byId[id] { result = concepts[index] }
        return result
    }

    private func isConcept(_ id: String) -> Bool {
        let leaf = (id as NSString).lastPathComponent
        return !Store.reserved.contains(leaf)
    }

    private func conceptId(_ url: URL) -> String {
        let base = root.resolvingSymlinksInPath().path
        var path = url.resolvingSymlinksInPath().path
        if path.hasPrefix(base) {
            path = String(path.dropFirst(base.count))
        }
        while path.hasPrefix("/") { path = String(path.dropFirst()) }
        if path.hasSuffix(".md") { path = String(path.dropLast(3)) }
        return path
    }

    private func markdownFiles() -> [URL] {
        var out: [URL] = []
        let manager = FileManager.default
        let walker = manager.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])
        if let walker = walker {
            for case let url as URL in walker {
                if url.pathExtension == "md" { out.append(url) }
            }
        }
        return out.sorted { left, right in left.path < right.path }
    }

    static func resolve(_ target: String, from id: String) -> String {
        var out = String(target.dropLast(3))
        let directory = (id as NSString).deletingLastPathComponent
        if out.hasPrefix("/") {
            out = String(out.dropFirst())
        } else if out.hasPrefix("./") || !out.contains("/") {
            let leaf = out.hasPrefix("./") ? String(out.dropFirst(2)) : out
            out = directory.isEmpty ? leaf : directory + "/" + leaf
        }
        return out
    }

    static func parseLinks(_ body: String, from id: String) -> [String] {
        let range = NSRange(body.startIndex..<body.endIndex, in: body)
        let matches = linkPattern.matches(in: body, options: [],
                                          range: range)
        var out: [String] = []
        var seen = Set<String>()
        for match in matches {
            if let span = Range(match.range(at: 1), in: body) {
                let target = resolve(String(body[span]), from: id)
                if !seen.contains(target) {
                    seen.insert(target)
                    out.append(target)
                }
            }
        }
        return out
    }

    private func parse(_ text: String, id: String, path: URL,
                       hash: UInt64) -> Concept {
        let parsed = Frontmatter.parse(text)
        var stale: Date? = nil
        if let raw = parsed.fields["stale_after"] {
            stale = Store.iso.date(from: raw)
        }
        return Concept(
            id: id, path: path,
            type: parsed.fields["type"] ?? "Concept",
            title: parsed.fields["title"] ?? id,
            description: parsed.fields["description"] ?? "",
            tags: parsed.lists["tags"] ?? [],
            status: parsed.fields["status"] ?? "",
            staleAfter: stale,
            generatedBy: Frontmatter.actors(parsed.unknown,
                                            under: "generated").first ?? "",
            verifiers: Frontmatter.actors(parsed.unknown,
                                          under: "verified"),
            extraFrontmatter: parsed.unknown,
            body: parsed.body,
            links: Store.parseLinks(parsed.body, from: id),
            hash: hash)
    }

    private func embedded(_ concept: Concept,
                          _ cache: [String: CacheEntry]) -> Concept {
        var out = concept
        let hit = cache[concept.id]
        if let hit = hit, hit.hash == concept.hash,
           hit.vectors.count == embedder.dim * 2 {
            out.vector = Array(hit.vectors[0..<embedder.dim])
            out.bodyVector = Array(hit.vectors[embedder.dim...])
        } else {
            out.vector = embedder.embedPassage(concept.passage)
            out.bodyVector = concept.body.isEmpty
                ? out.vector : embedder.embedPassage(concept.bodyPassage)
            embeddedCount += 1
        }
        return out
    }

    private func deriveBacklinks() {
        for i in concepts.indices {
            for target in concepts[i].links {
                if let j = byId[target], j != i {
                    concepts[j].backlinks.append(concepts[i].id)
                }
            }
        }
        for i in concepts.indices {
            concepts[i].backlinks = Array(Set(concepts[i].backlinks))
                .sorted()
        }
    }

    public func load() {
        let started = Date()
        let cache = readCache()
        concepts = []
        byId = [:]
        embeddedCount = 0
        for url in markdownFiles() {
            let data = try? Data(contentsOf: url)
            let id = conceptId(url)
            if let data = data, isConcept(id) {
                let text = String(decoding: data, as: UTF8.self)
                let parsed = parse(text, id: id, path: url,
                                   hash: fnv1a(data))
                byId[id] = concepts.count
                concepts.append(embedded(parsed, cache))
            }
        }
        deriveBacklinks()
        buildPostings()
        var fresh: [String: CacheEntry] = [:]
        for concept in concepts {
            fresh[concept.id] = CacheEntry(
                hash: concept.hash,
                vectors: concept.vector + concept.bodyVector)
        }
        writeCache(fresh)
        okfVersion = readVersion()
        loadSeconds = Date().timeIntervalSince(started)
    }

    private func readVersion() -> String {
        var out = ""
        let url = root.appendingPathComponent("index.md")
        if let text = try? String(contentsOf: url, encoding: .utf8) {
            for line in Frontmatter.parse(text).unknown {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix(Store.versionKey) {
                    out = Frontmatter.unquote(String(trimmed.dropFirst(
                        Store.versionKey.count))
                        .trimmingCharacters(in: .whitespaces))
                }
            }
        }
        return out
    }

    public func writeIndex() throws -> URL {
        var out = "---\n" + Store.versionKey + " \""
            + Store.okfSpecVersion + "\"\n---\n\n# Areas\n\n"
        for area in areas() where !area.area.isEmpty {
            out += "* [" + area.area + "](" + area.area + "/)"
            out += area.themes.isEmpty
                ? "\n" : " - " + area.themes.joined(separator: ", ") + "\n"
        }
        let loose = concepts.filter { concept in
            Store.area(of: concept.id).isEmpty
        }
        if !loose.isEmpty {
            out += "\n# Concepts\n\n"
            for concept in loose.sorted(by: { left, right in
                left.id < right.id
            }) {
                out += "* [" + concept.title + "](" + concept.id + ".md)"
                out += concept.description.isEmpty
                    ? "\n" : " - " + concept.description + "\n"
            }
        }
        let url = root.appendingPathComponent("index.md")
        try out.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    static let distinctiveAt = 3

    static let literalFloor = 2

    static let stopWords: Set<String> = [
        "the", "and", "for", "with", "that", "this", "from", "what",
        "when", "where", "which", "who", "how", "why", "into", "than",
        "then", "they", "them", "their", "there", "here", "have", "has",
        "had", "was", "were", "been", "being", "are", "not", "but",
        "all", "any", "some", "such", "only", "very", "just", "also",
        "can", "could", "should", "would", "will", "does", "did", "out",
        "over", "under", "about", "one", "two", "you", "your", "our",
        "his", "her", "its", "get", "got", "use", "used", "make", "made",
        "like", "does", "doing", "each", "own", "off", "per", "via"]

    static func words(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for character in text.lowercased() {
            if character.isLetter || character.isNumber {
                current.append(character)
            } else {
                if current.count >= 3 { out.append(current) }
                current = ""
            }
        }
        if current.count >= 3 { out.append(current) }
        return out
    }

    private func buildPostings() {
        anyPostings = [:]
        headPostings = [:]
        for (index, concept) in concepts.enumerated() {
            let head = Store.words(concept.title + " "
                + concept.description + " "
                + concept.tags.joined(separator: " "))
            for word in Set(head) {
                headPostings[word, default: []].insert(index)
            }
            let all = Set(head).union(Store.words(concept.body))
            for word in all {
                anyPostings[word, default: []].append(index)
            }
        }
    }

    public func literalRanking(_ query: String)
        -> [(index: Int, weight: Int, terms: [String])] {
        var weights: [Int: Int] = [:]
        var matched: [Int: [String]] = [:]
        for word in Set(Store.words(query)) {
            let holders = anyPostings[word] ?? []
            let rare = !holders.isEmpty
                && holders.count <= Store.distinctiveAt
                && !Store.stopWords.contains(word)
            if rare {
                for index in holders {
                    let inHead = headPostings[word]?.contains(index) == true
                    weights[index, default: 0] += inHead ? 3 : 1
                    matched[index, default: []].append(word)
                }
            }
        }
        let ranked = weights.filter { entry in
            entry.value >= Store.literalFloor
        }.sorted { left, right in
            left.value != right.value
                ? left.value > right.value : left.key < right.key
        }
        return ranked.map { entry in
            (entry.key, entry.value, (matched[entry.key] ?? []).sorted())
        }
    }

    static let fusionK: Float = 10

    private static func fuse(_ rankings: [[Int]]) -> [Int: Float] {
        var out: [Int: Float] = [:]
        for ranking in rankings {
            for (rank, index) in ranking.enumerated() {
                out[index, default: 0] += 1 / (fusionK + Float(rank))
            }
        }
        return out
    }

    private func denseRanking(_ query: String, _ retired: Bool)
        -> [(index: Int, score: Float)] {
        let vector = embedder.embedQuery(query)
        var scored: [(index: Int, score: Float)] = []
        scored.reserveCapacity(concepts.count)
        let width = embedder.dim
        vector.withUnsafeBufferPointer { query in
            let base = query.baseAddress!
            for i in concepts.indices
                where retired || !concepts[i].isDeprecated {
                let abstract = Store.dot(base, concepts[i].vector, width)
                let body = Store.dot(base, concepts[i].bodyVector, width)
                scored.append((i, max(abstract, body * 0.95)))
            }
        }
        scored.sort { left, right in
            (-left.score, left.index) < (-right.score, right.index)
        }
        return scored
    }

    public func search(_ queries: [String], filter: Filter,
                       limit: Int) -> SearchResult {
        let started = Date()
        var rankings: [[Int]] = []
        var best: [Int: Float] = [:]
        var standout: Float = 0
        for query in queries {
            let dense = denseRanking(query, filter.deprecated)
            rankings.append(dense.map { entry in entry.index })
            for entry in dense where entry.score > best[entry.index] ?? -1 {
                best[entry.index] = entry.score
            }
            standout = max(standout, Store.standout(
                dense.map { entry in entry.score }))
        }
        let embedded = Date()
        var terms: [Int: Set<String>] = [:]
        for query in queries {
            let literal = literalRanking(query).filter { entry in
                filter.deprecated || !concepts[entry.index].isDeprecated
            }
            rankings.append(literal.map { entry in entry.index })
            for entry in literal {
                terms[entry.index, default: []].formUnion(entry.terms)
            }
        }
        let done = Date()
        let fused = Store.fuse(rankings)
        let order = fused.sorted { left, right in
            left.value != right.value
                ? left.value > right.value : left.key < right.key
        }
        let admitted = order.filter { entry in
            filter.admits(concepts[entry.key])
        }
        let hits = rerank(queries, admitted.prefix(limit).map { entry in
            Hit(concept: concepts[entry.key],
                score: best[entry.key] ?? 0, relevance: 0,
                terms: (terms[entry.key] ?? []).sorted())
        })
        return SearchResult(
            hits: hits,
            embedSeconds: embedded.timeIntervalSince(started),
            scanSeconds: done.timeIntervalSince(embedded),
            literalSeconds: Date().timeIntervalSince(done),
            standout: standout)
    }

    static let bodyDiscount: Float = 0.95
    static let relevanceChars = 600

    private func tokenStates(_ prefix: String, _ text: String) -> [Float] {
        let width = embedder.dim
        let skip = embedder.tokens(prefix).count - 1
        let states = embedder.states(prefix + text)
        let kept = max(0, states.count / width - skip - 1)
        var out = Array(states[(skip * width)..<((skip + kept) * width)])
        for t in 0..<kept {
            var square: Float = 0
            for d in 0..<width {
                square += out[t * width + d] * out[t * width + d]
            }
            let inverse = square > 0 ? 1 / square.squareRoot() : 0
            for d in 0..<width { out[t * width + d] *= inverse }
        }
        return out
    }

    private func maxSim(_ query: [Float], _ document: [Float]) -> Float {
        let width = embedder.dim
        let asked = query.count / width
        let given = document.count / width
        var total: Float = 0
        query.withUnsafeBufferPointer { queries in
            document.withUnsafeBufferPointer { documents in
                for i in 0..<asked {
                    var best: Float = -1
                    for j in 0..<given {
                        let dot = Store.dot(
                            queries.baseAddress! + i * width,
                            documents.baseAddress! + j * width, width)
                        if dot > best { best = dot }
                    }
                    total += best
                }
            }
        }
        return asked > 0 && given > 0 ? total / Float(asked) : 0
    }

    private func relevance(_ concept: Concept,
                           _ queries: [[Float]]) -> Float {
        let abstract = tokenStates(embedder.passagePrefix, concept.passage)
        let body = concept.body.isEmpty
            ? [] : tokenStates(embedder.passagePrefix,
                               String(concept.body.prefix(
                                   Store.relevanceChars)))
        var out: Float = 0
        for query in queries {
            let scored = max(maxSim(query, abstract),
                             maxSim(query, body) * Store.bodyDiscount)
            if scored > out { out = scored }
        }
        return out
    }

    private func rerank(_ queries: [String], _ hits: [Hit]) -> [Hit] {
        let asked = queries.map { query in
            tokenStates(embedder.queryPrefix, query)
        }
        let scored = hits.enumerated().map { rank, hit in
            (rank, Hit(concept: hit.concept, score: hit.score,
                       relevance: relevance(hit.concept, asked),
                       terms: hit.terms))
        }
        return scored.sorted { left, right in
            (-left.1.relevance, left.0) < (-right.1.relevance, right.0)
        }.map { entry in entry.1 }
    }

    public func relevant(_ hit: Hit) -> Bool {
        hit.relevance >= embedder.relevanceFloor
    }

    public func confident(_ result: SearchResult) -> Bool {
        result.hits.first.map { top in relevant(top) } ?? false
    }

    static func standout(_ scores: [Float]) -> Float {
        var result: Float = 0
        if scores.count > 2 {
            var total: Float = 0
            for score in scores { total += score }
            let mean = total / Float(scores.count)
            var sumSquares: Float = 0
            for score in scores {
                sumSquares += (score - mean) * (score - mean)
            }
            let deviation = (sumSquares / Float(scores.count)).squareRoot()
            if deviation > 0 {
                result = (scores[0] - mean) / deviation
            }
        }
        return result
    }

    private static func dot(_ query: UnsafePointer<Float>,
                            _ vector: [Float], _ width: Int) -> Float {
        var total: Float = -1
        if vector.count == width {
            vector.withUnsafeBufferPointer { stored in
                total = Store.dot(query, stored.baseAddress!, width)
            }
        }
        return total
    }

    private static func dot(_ left: UnsafePointer<Float>,
                            _ right: UnsafePointer<Float>,
                            _ width: Int) -> Float {
        var total: Float = 0
        for i in 0..<width { total += left[i] * right[i] }
        return total
    }

    public func grep(_ pattern: String, limit: Int) -> [GrepHit] {
        let regex = try? NSRegularExpression(
            pattern: pattern, options: [.caseInsensitive])
        var out: [GrepHit] = []
        if let regex = regex {
            for concept in concepts {
                let haystack = concept.title + "\n" + concept.description
                    + "\n" + concept.body
                var line = 0
                for text in haystack.components(separatedBy: "\n") {
                    line += 1
                    let range = NSRange(text.startIndex..<text.endIndex,
                                        in: text)
                    let matched = out.count < limit && regex.firstMatch(
                        in: text, options: [], range: range) != nil
                    if matched {
                        out.append(GrepHit(
                            id: concept.id, title: concept.title,
                            line: line,
                            text: text.trimmingCharacters(in: .whitespaces)))
                    }
                }
            }
        }
        return out
    }

    static let mergeAt: Float = 0.85
    static let relateAt: Float = 0.60
    static let clusterAt: Float = 0.55

    private func similarity(_ left: Int, _ right: Int) -> Float {
        var total: Float = 0
        let a = concepts[left].vector
        let b = concepts[right].vector
        if a.count == b.count {
            for i in 0..<a.count { total += a[i] * b[i] }
        }
        return total
    }

    private func linked(_ left: Int, _ right: Int) -> Bool {
        concepts[left].links.contains(concepts[right].id)
            || concepts[right].links.contains(concepts[left].id)
    }

    private func closePairs() -> [(left: Int, right: Int, score: Float)] {
        var out: [(left: Int, right: Int, score: Float)] = []
        for i in concepts.indices {
            for j in concepts.indices where j > i {
                let score = similarity(i, j)
                if score >= Store.relateAt {
                    out.append((i, j, score))
                }
            }
        }
        return out.sorted { left, right in
            (-left.score, left.left, left.right)
                < (-right.score, right.left, right.right)
        }
    }

    private func looseCluster(_ seed: Int,
                              _ spoken: Set<String>) -> Proposal? {
        var neighbours: [(index: Int, score: Float)] = []
        for other in concepts.indices where other != seed {
            let score = similarity(seed, other)
            if score >= Store.clusterAt && !spoken.contains(
                concepts[other].id) {
                neighbours.append((other, score))
            }
        }
        neighbours.sort { left, right in
            (-left.score, left.index) < (-right.score, right.index)
        }
        let members = [seed] + neighbours.prefix(4).map { hit in hit.index }
        var present = 0
        var possible = 0
        for i in members.indices {
            for j in members.indices where j > i {
                possible += 1
                if linked(members[i], members[j]) { present += 1 }
            }
        }
        var result: Proposal? = nil
        if members.count >= 3 && present * 2 < possible {
            let mean = neighbours.prefix(4).reduce(Float(0)) {
                running, hit in running + hit.score
            } / Float(min(4, neighbours.count))
            result = Proposal(
                kind: .skill, score: mean,
                members: members.map { index in concepts[index].id })
        }
        return result
    }

    private func rank(_ kind: Proposal.Kind) -> Int {
        var out = 0
        switch kind {
        case .dangling: out = 4
        case .merge:    out = 3
        case .skill:    out = 2
        case .link:     out = 1
        case .orphan:   out = 0
        }
        return out
    }

    private func pairProposals(_ kind: Proposal.Kind,
                               _ spoken: inout Set<String>) -> [Proposal] {
        var out: [Proposal] = []
        for pair in closePairs() {
            let ids = [concepts[pair.left].id, concepts[pair.right].id]
            let fresh = !spoken.contains(ids[0])
                && !spoken.contains(ids[1])
            let duplicate = pair.score >= Store.mergeAt
            let wanted = kind == .merge
                ? duplicate
                : !duplicate && !linked(pair.left, pair.right)
            if fresh && wanted {
                out.append(Proposal(kind: kind, score: pair.score,
                                    members: ids))
                spoken.formUnion(ids)
            }
        }
        return out
    }

    private func skillProposals(
        _ spoken: inout Set<String>) -> [Proposal] {
        var out: [Proposal] = []
        for seed in concepts.indices {
            if !spoken.contains(concepts[seed].id) {
                if let cluster = looseCluster(seed, spoken) {
                    out.append(cluster)
                    spoken.formUnion(cluster.members)
                }
            }
        }
        return out
    }

    private func orphanProposals(
        _ spoken: inout Set<String>) -> [Proposal] {
        var out: [Proposal] = []
        for concept in concepts {
            let alone = concept.links.isEmpty
                && concept.backlinks.isEmpty
                && !spoken.contains(concept.id)
            if alone {
                out.append(Proposal(kind: .orphan, score: 0,
                                    members: [concept.id]))
                spoken.insert(concept.id)
            }
        }
        return out
    }

    private func danglingProposals() -> [Proposal] {
        var out: [Proposal] = []
        for concept in concepts {
            for link in concept.links where self.concept(link) == nil {
                out.append(Proposal(kind: .dangling, score: 0,
                                    members: [concept.id, link]))
            }
        }
        return out
    }

    public func dream(limit: Int) -> [Proposal] {
        var spoken = Set<String>()
        var out = danglingProposals()
        out += pairProposals(.merge, &spoken)
        out += skillProposals(&spoken)
        out += pairProposals(.link, &spoken)
        out += orphanProposals(&spoken)
        let ranked = out.sorted { left, right in
            (-rank(left.kind), -left.score, left.members.joined())
                < (-rank(right.kind), -right.score,
                   right.members.joined())
        }
        return Array(ranked.prefix(limit))
    }

    public static func area(of id: String) -> String {
        var out = "."
        if let slash = id.firstIndex(of: "/") {
            out = String(id[id.startIndex..<slash])
        }
        return out
    }

    public func neighbours(of id: String, limit: Int) -> [String] {
        var out: [String] = []
        if let index = byId[id] {
            let already = Set(concepts[index].links
                + concepts[index].backlinks + [id])
            var scored: [(index: Int, score: Float)] = []
            for other in concepts.indices where other != index {
                if !already.contains(concepts[other].id) {
                    scored.append((other, similarity(index, other)))
                }
            }
            scored.sort { left, right in
                (-left.score, left.index) < (-right.score, right.index)
            }
            out = scored.prefix(limit).map { entry in
                concepts[entry.index].id
            }
        }
        return out
    }

    public func nearest(title: String, description: String, tags: [String],
                        type: String, limit: Int)
        -> [(id: String, score: Float)] {
        let vector = embedder.embedPassage(Concept.passage(
            title: title, description: description, tags: tags, type: type))
        var scored: [(index: Int, score: Float)] = []
        let width = embedder.dim
        vector.withUnsafeBufferPointer { passage in
            let base = passage.baseAddress!
            for i in concepts.indices where !concepts[i].isDeprecated {
                scored.append((i, Store.dot(base, concepts[i].vector, width)))
            }
        }
        scored.sort { left, right in
            (-left.score, left.index) < (-right.score, right.index)
        }
        return scored.prefix(limit).map { entry in
            (concepts[entry.index].id, entry.score)
        }
    }

    static let duplicateAt: Float = 0.90
    static let duplicateWords = 6

    public func duplicate(title: String, description: String, tags: [String],
                          type: String) -> (id: String, score: Float)? {
        var out: (id: String, score: Float)? = nil
        let top = nearest(title: title, description: description,
                          tags: tags, type: type, limit: 1).first
        let words = literalRanking(title + " " + description).first
        if let top, let words, top.score >= Store.duplicateAt,
           words.weight >= Store.duplicateWords,
           concepts[words.index].id == top.id {
            out = top
        }
        return out
    }

    public func report(for id: String) -> WriteReport? {
        var result: WriteReport? = nil
        if let concept = concept(id) {
            let candidates = neighbours(of: id, limit: 5)
            var tally: [String: Int] = [:]
            for candidate in candidates {
                tally[Store.area(of: candidate), default: 0] += 1
            }
            let leader = tally.sorted { left, right in
                (-left.value, left.key) < (-right.value, right.key)
            }.first
            var stray = ""
            if let leader = leader, leader.value >= 3,
               leader.key != Store.area(of: id) {
                stray = leader.key
            }
            result = WriteReport(url: concept.path,
                                 candidates: candidates,
                                 strayArea: stray)
        }
        return result
    }

    static func paragraphs(_ body: String) -> [Paragraph] {
        var out: [Paragraph] = []
        let bytes = Array(body.utf8)
        var start = 0
        var index = 0
        while index <= bytes.count {
            let split = index + 1 < bytes.count
                && bytes[index] == 10 && bytes[index + 1] == 10
            if split || index == bytes.count {
                if index > start {
                    out.append(Paragraph(
                        from: start, to: index,
                        text: String(decoding: bytes[start..<index],
                                     as: UTF8.self)))
                }
                start = index + 2
                index += 2
            } else {
                index += 1
            }
        }
        return out
    }

    public func window(_ concept: Concept, about: String, limit: Int)
        -> (from: Int, to: Int) {
        let parts = Store.paragraphs(concept.body)
        var from = 0
        var to = min(concept.body.utf8.count, limit)
        if parts.count > 1 && limit > 0 {
            let query = embedder.embedQuery(about)
            var best = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for (index, part) in parts.enumerated() {
                let vector = embedder.embedPassage(part.text)
                var total: Float = 0
                for i in 0..<min(query.count, vector.count) {
                    total += query[i] * vector[i]
                }
                if total > bestScore {
                    bestScore = total
                    best = index
                }
            }
            var low = best
            var high = best
            var size = parts[best].to - parts[best].from
            var growing = true
            while growing {
                let takeLow = low > 0
                    && (high == parts.count - 1
                        || parts[low - 1].to - parts[low - 1].from
                            <= parts[high + 1].to - parts[high + 1].from)
                let nextSize = takeLow && low > 0
                    ? parts[low - 1].to - parts[low - 1].from
                    : (high < parts.count - 1
                        ? parts[high + 1].to - parts[high + 1].from : 0)
                if (low > 0 || high < parts.count - 1)
                    && size + nextSize + 2 <= limit {
                    if takeLow { low -= 1 } else { high += 1 }
                    size += nextSize + 2
                } else {
                    growing = false
                }
            }
            from = parts[low].from
            to = parts[high].to
        }
        return (from, to)
    }

    public func areas() -> [(area: String, count: Int, themes: [String])] {
        var members: [String: [Int]] = [:]
        for (index, concept) in concepts.enumerated() {
            members[Store.area(of: concept.id), default: []]
                .append(index)
        }
        var out: [(area: String, count: Int, themes: [String])] = []
        for (area, indices) in members {
            var frequency: [String: Int] = [:]
            for index in indices {
                for tag in concepts[index].tags {
                    if tag != area { frequency[tag, default: 0] += 1 }
                }
            }
            let ranked = frequency.sorted { left, right in
                left.value != right.value
                    ? left.value > right.value : left.key < right.key
            }
            out.append((area, indices.count,
                        ranked.prefix(6).map { entry in entry.key }))
        }
        return out.sorted { left, right in left.area < right.area }
    }

    public func ids(inArea area: String) -> [String] {
        concepts.map { concept in concept.id }
            .filter { id in id.hasPrefix(area + "/") }
            .sorted()
    }

    public func write(id: String, type: String, title: String,
                      description: String, tags: [String], body: String,
                      status: String = "", staleAfter: Date? = nil,
                      adding lines: [String] = [],
                      dropping keys: [String] = []) throws -> URL {
        let url = root.appendingPathComponent(id + ".md")
        var extra: [String] = []
        if let existing = try? String(contentsOf: url, encoding: .utf8) {
            extra = Frontmatter.dropping(Frontmatter.parse(existing).unknown,
                                         keys: keys)
        }
        extra += lines
        let concept = Concept(
            id: id, path: url, type: type, title: title,
            description: description, tags: tags, status: status,
            staleAfter: staleAfter,
            generatedBy: Frontmatter.actors(extra,
                                            under: "generated").first ?? "",
            verifiers: Frontmatter.actors(extra, under: "verified"),
            extraFrontmatter: extra, body: body,
            links: [], hash: 0)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let existed = self.concept(id) != nil
        try Frontmatter.emit(concept).write(to: url, atomically: true,
                                            encoding: .utf8)
        log(existed ? .update : .creation, id: id, title: title)
        return url
    }

    public func deprecate(id: String) throws -> [String] {
        var referrers: [String] = []
        if var target = concept(id) {
            referrers = target.backlinks
            target.status = "deprecated"
            try Frontmatter.emit(target).write(
                to: target.path, atomically: true, encoding: .utf8)
            log(.deprecation, id: id, title: target.title)
        } else {
            throw StoreError.noSuchConcept(id)
        }
        return referrers
    }

    public func purge(id: String) throws -> [String] {
        var referrers: [String] = []
        if let target = concept(id) {
            referrers = target.backlinks
            try FileManager.default.removeItem(at: target.path)
            pruneEmptyDirectories(from: target.path
                .deletingLastPathComponent())
            log(.deletion, id: id, title: target.title)
        } else {
            throw StoreError.noSuchConcept(id)
        }
        return referrers
    }

    public func unlink(_ id: String) throws -> [RemovedLink] {
        var out: [RemovedLink] = []
        if let target = concept(id) {
            for referrer in target.backlinks {
                if var source = concept(referrer) {
                    let cut = Store.collapse(source.body, to: id,
                                             from: referrer)
                    if !cut.removed.isEmpty {
                        source.body = cut.body
                        source.links = Store.parseLinks(cut.body,
                                                        from: referrer)
                        try Frontmatter.emit(source).write(
                            to: source.path, atomically: true,
                            encoding: .utf8)
                        out += cut.removed.map { link in
                            RemovedLink(referrer: referrer,
                                        markup: link.markup,
                                        label: link.label)
                        }
                    }
                }
            }
        } else {
            throw StoreError.noSuchConcept(id)
        }
        return out
    }

    public func relink(_ removed: [RemovedLink]) {
        var byReferrer: [String: [RemovedLink]] = [:]
        for link in removed {
            byReferrer[link.referrer, default: []].append(link)
        }
        for (referrer, links) in byReferrer {
            if var source = concept(referrer) {
                var body = source.body
                for link in links
                where !link.label.isEmpty && !body.contains(link.markup) {
                    if let at = body.range(of: link.label) {
                        body.replaceSubrange(at, with: link.markup)
                    }
                }
                if body != source.body {
                    source.body = body
                    source.links = Store.parseLinks(body, from: referrer)
                    try? Frontmatter.emit(source).write(
                        to: source.path, atomically: true, encoding: .utf8)
                }
            }
        }
    }

    static func collapse(_ body: String, to id: String, from referrer: String)
        -> (body: String, removed: [(markup: String, label: String)]) {
        let range = NSRange(body.startIndex..<body.endIndex, in: body)
        let matches = linkPattern.matches(in: body, options: [], range: range)
        var out = ""
        var removed: [(markup: String, label: String)] = []
        var cursor = body.startIndex
        for match in matches {
            let whole = Range(match.range, in: body)
            let target = Range(match.range(at: 1), in: body)
            if let whole, let target,
               Store.resolve(String(body[target]), from: referrer) == id {
                let markup = String(body[whole])
                let label = Store.linkLabel(markup)
                out += String(body[cursor..<whole.lowerBound]) + label
                cursor = whole.upperBound
                removed.append((markup, label))
            }
        }
        return (out + String(body[cursor...]), removed)
    }

    static func linkLabel(_ markup: String) -> String {
        var out = markup
        if markup.hasPrefix("["), let close = markup.firstIndex(of: "]") {
            out = String(
                markup[markup.index(after: markup.startIndex)..<close])
        }
        return out
    }

    enum Change: String {
        case creation = "Creation"
        case update = "Update"
        case deprecation = "Deprecation"
        case deletion = "Deletion"
    }

    func log(_ kind: Change, id: String, title: String) {
        let url = root.appendingPathComponent("log.md")
        let today = String(Store.iso.string(from: Date()).prefix(10))
        let heading = "## " + today
        let label = title.isEmpty ? id : title
        let entry = "* **" + kind.rawValue + "**: ["
            + label + "](/" + id + ".md)"
        var lines = ((try? String(contentsOf: url, encoding: .utf8))
            ?? "# Update Log (UTC)\n").components(separatedBy: "\n")
        if lines.first?.hasPrefix("# ") != true {
            lines.insert("# Update Log (UTC)", at: 0)
        }
        var at = 1
        while at < lines.count && !lines[at].hasPrefix("## ") { at += 1 }
        if at >= lines.count || lines[at] != heading {
            lines.insert(contentsOf: ["", heading], at: 1)
            at = 2
        }
        lines.insert(entry, at: at + 1)
        let text = lines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }

    private func pruneEmptyDirectories(from directory: URL) {
        var at = directory
        var pruning = at.path.hasPrefix(root.path) && at.path != root.path
        while pruning {
            let entries = try? FileManager.default
                .contentsOfDirectory(atPath: at.path)
            if entries?.isEmpty == true {
                try? FileManager.default.removeItem(at: at)
                at = at.deletingLastPathComponent()
                pruning = at.path != root.path
            } else {
                pruning = false
            }
        }
    }

    struct CacheEntry {
        let hash: UInt64
        let vectors: [Float]
    }

    private static let magic: UInt32 = 0x5646_4B4F

    private func readCache() -> [String: CacheEntry] {
        var out: [String: CacheEntry] = [:]
        let width = embedder.dim * 2
        let data = try? Data(contentsOf: sidecar)
        if let data = data, data.count >= 12 {
            var cursor = ByteReader(data: data)
            let header = cursor.u32() == Store.magic
                && Int(cursor.u32()) == width
            let count = Int(cursor.u32())
            var i = 0
            while header && i < count && cursor.has(12) {
                let hash = cursor.u64()
                let length = Int(cursor.u32())
                if cursor.has(length + width * 4) {
                    let id = cursor.string(length)
                    out[id] = CacheEntry(hash: hash,
                                         vectors: cursor.floats(width))
                    i += 1
                } else {
                    i = count
                }
            }
        }
        return out
    }

    private func writeCache(_ cache: [String: CacheEntry]) {
        var data = Data()
        appendLE(&data, Store.magic)
        appendLE(&data, UInt32(embedder.dim * 2))
        appendLE(&data, UInt32(cache.count))
        for (id, entry) in cache.sorted(by: { left, right in
            left.key < right.key
        }) {
            appendLE(&data, entry.hash)
            let bytes = Array(id.utf8)
            appendLE(&data, UInt32(bytes.count))
            data.append(contentsOf: bytes)
            for value in entry.vectors { appendLE(&data, value) }
        }
        try? data.write(to: sidecar)
    }
}

public enum StoreError: Error, CustomStringConvertible {
    case noSuchConcept(String)

    public var description: String {
        var out = ""
        switch self {
        case let .noSuchConcept(id): out = "no such concept: " + id
        }
        return out
    }
}

struct ByteReader {
    let data: Data
    var offset = 0

    func has(_ count: Int) -> Bool { offset + count <= data.count }

    mutating func u32() -> UInt32 {
        let value = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }
        offset += 4
        return value
    }

    mutating func u64() -> UInt64 {
        let value = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
        }
        offset += 8
        return value
    }

    mutating func string(_ count: Int) -> String {
        let start = data.startIndex + offset
        let out = String(decoding: data[start..<(start + count)],
                         as: UTF8.self)
        offset += count
        return out
    }

    mutating func floats(_ count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                out[i] = raw.loadUnaligned(
                    fromByteOffset: offset + i * 4, as: Float.self)
            }
        }
        offset += count * 4
        return out
    }
}

func appendLE<T>(_ data: inout Data, _ value: T) {
    var copy = value
    withUnsafeBytes(of: &copy) { bytes in
        data.append(contentsOf: bytes)
    }
}

func fnv1a(_ data: Data) -> UInt64 {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    data.withUnsafeBytes { raw in
        for byte in raw {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
    }
    return hash
}
