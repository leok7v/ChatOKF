import Foundation
import LLM
import Observation

@MainActor @Observable public final class Memories {

    public struct Recall {
        public let titles: [String]
        public let ids: [String]
        public let block: String
        public let standout: Float
        public let tokens: Int
        public let seconds: Double
        public let searchSeconds: Double
        public let readSeconds: [Double]
        public var silent: Bool { seconds <= Memories.budgetSeconds }
    }

    public struct Note {
        public let id: String
        public let title: String
        public let text: String
    }

    public static let folder = Flags.value("memories-folder")
        ?? "memories.noindex"
    static let enabledKey = "totalRecall"
    static let backupKey = "backupMemories"
    static let bytesPerToken = 3.5
    static let recallLimit = 3
    static let bodyCap = 600
    public static var supported: Bool { true }

    public static let defaultRoot: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        return support.appendingPathComponent(folder, isDirectory: true)
    }()

    nonisolated public static let budgetSeconds: Double =
        Flags.double("recall-budget") ?? 5

    public let root: URL

    public var enabled: Bool {
        didSet {
            UserDefaults.standard.set(enabled, forKey: Memories.enabledKey)
        }
    }

    public var backup: Bool {
        didSet {
            UserDefaults.standard.set(backup, forKey: Memories.backupKey)
            Memories.exclude(root, !backup)
        }
    }

    let owner: MemoryStore
    private var opening: Task<Void, Never>?
    var concepts: [String: Concept] = [:]
    private var floor: Float = 0

    public private(set) var isOpen = false
    public internal(set) var list: [MemoryRow] = []
    public internal(set) var trashed: [MemoryRow] = []
    public private(set) var revision = 0

    public init(root: URL = Memories.defaultRoot) {
        let d = UserDefaults.standard
        self.root = root
        owner = MemoryStore(root: root)
        enabled = d.object(forKey: Memories.enabledKey) as? Bool ?? true
        backup = d.bool(forKey: Memories.backupKey)
    }

    public var active: Bool {
        Memories.supported && enabled && Flags.on("memories")
    }

    public var count: Int { list.count }

    public func open() {
        if !isOpen, opening == nil, active {
            let excluded = !backup
            let owner = self.owner
            opening = Task(priority: .utility) { [weak self] in
                let opened = await owner.open(excludedFromBackup: excluded)
                if let self, let opened, !Task.isCancelled {
                    self.adopt(opened.snapshot)
                    self.isOpen = true
                    Diag.shared.report(.load, String(
                        format: "[memories] %d concepts, re-embedded %d, "
                            + "%.2fs at %@", opened.snapshot.concepts.count,
                        opened.embedded, opened.seconds, owner.root.path))
                }
                if !Task.isCancelled { self?.opening = nil }
            }
        }
    }

    public func awaitOpen() async {
        open()
        await opening?.value
    }

    private var adopted = 0

    func adopt(_ snapshot: MemorySnapshot?) {
        if let snapshot, snapshot.serial > adopted { apply(snapshot) }
    }

    private func apply(_ snapshot: MemorySnapshot) {
        adopted = snapshot.serial
        list = snapshot.list
        trashed = snapshot.trashed
        floor = snapshot.floor
        concepts = [:]
        for concept in snapshot.concepts { concepts[concept.id] = concept }
        revision += 1
    }

    nonisolated static func exclude(_ root: URL, _ on: Bool) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = on
        var url = root
        try? url.setResourceValues(values)
    }

    private func close() {
        opening?.cancel()
        opening = nil
        isOpen = false
        apply(MemorySnapshot.empty)
    }

    public func reopen() {
        close()
        open()
    }

    public func forgetAll() {
        close()
        let owner = self.owner
        Task { await owner.erase() }
    }

    public static func erase() {
        try? FileManager.default.removeItem(at: defaultRoot)
    }

    public func map() async -> String {
        await owner.map()
    }

    static func line(_ concept: Concept) -> String {
        var out = "- " + concept.title
        if concept.description.lowercased().hasPrefix(Memories.opening) {
            out = "- " + concept.description
        } else if !concept.description.isEmpty {
            out += ": " + concept.description
        }
        let body = concept.body.replacingOccurrences(of: "\n", with: " ")
        if !body.isEmpty {
            out += "\n  " + String(body.prefix(Memories.bodyCap))
            if body.count > Memories.bodyCap { out += " ..." }
        }
        return out + "\n"
    }

    func relevant(_ hit: Hit) -> Bool {
        hit.relevance >= floor
    }

    public func recall(_ question: String, also: [String], pp: Double,
                       excluding read: Set<String>) async -> Recall? {
        var out: Recall? = nil
        if isOpen, !concepts.isEmpty {
            let queries = [question] + also.filter { text in
                !text.isEmpty
            } + Memories.clauses(question)
            let ticket = owner.searches.take()
            let found = await owner.search(
                ticket, queries, limit: Memories.recallLimit + read.count)
            if let found {
                out = recalled(found, pp: pp, excluding: read)
            } else {
                Diag.shared.report(.turn, "[recall] superseded")
            }
        } else if !isOpen, active {
            Diag.shared.report(.turn, "[recall] the store is not open yet")
            open()
        }
        return out
    }

    private func recalled(_ result: SearchResult, pp: Double,
                          excluding read: Set<String>) -> Recall? {
        var out: Recall? = nil
        let searched = result.embedSeconds + result.scanSeconds
            + result.literalSeconds
        let fresh = result.hits.filter { hit in
            !read.contains(hit.concept.id) && relevant(hit)
        }.prefix(Memories.recallLimit)
        if !fresh.isEmpty {
            var block = "What you remember about me from our earlier "
                + "chats (in these notes \"the user\" is me, the person "
                + "writing to you); use what bears on my message:\n"
            for hit in fresh { block += Memories.line(hit.concept) }
            block += "\n"
            let tokens = Int(Double(block.utf8.count)
                             / Memories.bytesPerToken)
            let reads = fresh.map { hit in
                Memories.seconds(bytes: hit.concept.passage.utf8.count
                                     + hit.concept.body.utf8.count, pp)
            }
            let titles = fresh.map { hit in hit.concept.title }
            out = Recall(titles: titles,
                         ids: fresh.map { hit in hit.concept.id },
                         block: block, standout: result.standout,
                         tokens: tokens,
                         seconds: pp > 0 ? Double(tokens) / pp : 0,
                         searchSeconds: searched, readSeconds: reads)
        } else {
            let unread = result.hits.first { hit in
                !read.contains(hit.concept.id)
            }
            Diag.shared.report(.turn, String(
                format: "[recall] nothing: top unread %.3f of %.3f, "
                    + "standout %.1f, %d of %d concepts fit, "
                    + "searched %.2fs",
                unread?.relevance ?? 0, floor, result.standout, fresh.count,
                concepts.count, searched))
        }
        return out
    }

    static func seconds(bytes: Int, _ pp: Double) -> Double {
        pp > 0 ? Double(bytes) / bytesPerToken / pp : 0
    }

    public func note(_ id: String) -> Note? {
        var out: Note? = nil
        if let concept = concepts[id] {
            var text = "My note \"" + concept.title + "\""
            if !concept.description.isEmpty {
                text += ": " + concept.description
            }
            text += "\n\n" + concept.body
            out = Note(id: concept.id, title: concept.title, text: text)
        }
        return out
    }

    public struct Draft: Sendable {
        public let area: String
        public let title: String
        public let description: String
        public let isPrivate: Bool

        public init(area: String = "", title: String, description: String,
                    isPrivate: Bool = false) {
            self.area = area
            self.title = title
            self.description = description
            self.isPrivate = isPrivate
        }
    }

    public struct Remembered: Identifiable, Sendable {
        public let id: String
        public let title: String
    }

    public static let extractionInstruction =
        "From the user's LAST message only, list the durable facts the "
        + "user stated about themselves: preferences, possessions, people, "
        + "pets, places, plans, habits, health or money facts that will "
        + "still be true in six months. Only what the user said, never what "
        + "the assistant supplied, and never a question they asked, a "
        + "lookup, a calculation, a what-if, a story or small talk. If "
        + "there is nothing, reply NONE. Otherwise write one short sentence "
        + "per fact, three at most, each on its own line, and begin every "
        + "sentence with the words The user."

    static let draftLimit = 3
    static let opening = "the user"
    static let titleWords = 5
    static let trailing: Set<String> = [
        "and", "or", "is", "are", "was", "a", "an", "the", "who", "that",
        "which", "in", "at", "of", "to", "with", "as", "for", "on", "has",
    ]

    nonisolated static func leading(_ text: String,
                                    sentences: Int) -> String {
        var ends = 0
        var cut = text.endIndex
        var i = text.startIndex
        while i < text.endIndex && ends < sentences {
            let next = text.index(after: i)
            let closes = ".!?".contains(text[i])
                && (next == text.endIndex || text[next].isWhitespace)
            if closes {
                ends += 1
                cut = next
            }
            i = next
        }
        return String(text[..<(ends == sentences ? cut : text.endIndex)])
    }

    static func title(of sentence: String) -> String {
        var words = sentence.split(separator: " ").map { word in
            word.trimmingCharacters(in: .punctuationCharacters)
        }.filter { word in !word.isEmpty }
        words = Array(words.dropFirst(2).prefix(Memories.titleWords))
        while let last = words.last,
              Memories.trailing.contains(last.lowercased()) {
            words.removeLast()
        }
        let joined = words.joined(separator: " ")
        return joined.prefix(1).uppercased() + joined.dropFirst()
    }

    static func parseDrafts(_ raw: String) -> [Draft] {
        var out: [Draft] = []
        for line in raw.components(separatedBy: "\n")
        where out.count < Memories.draftLimit {
            let stated = line.trimmingCharacters(
                in: CharacterSet(charactersIn: " \t-*`\"0123456789.)"))
            let fact = Memories.leading(stated, sentences: 1)
            let title = Memories.title(of: fact)
            if fact.lowercased().hasPrefix(Memories.opening),
               !title.isEmpty {
                out.append(Draft(title: title, description: fact))
            }
        }
        return out
    }

    nonisolated static let selfWords: Set<String> = [
        "i", "my", "me", "mine", "myself", "we", "our", "us", "remember",
        "\u{044F}", "\u{043C}\u{0435}\u{043D}\u{044F}",
        "\u{043C}\u{043D}\u{0435}", "\u{043C}\u{043E}\u{0439}",
        "\u{043C}\u{043E}\u{044F}", "\u{043C}\u{043E}\u{0451}",
        "\u{043C}\u{043E}\u{0435}", "\u{043C}\u{043E}\u{0438}",
        "\u{043C}\u{044B}", "\u{043D}\u{0430}\u{0441}",
        "\u{043D}\u{0430}\u{0448}",
        "\u{0437}\u{0430}\u{043F}\u{043E}\u{043C}\u{043D}\u{0438}",
        "ich", "mein", "meine", "mich", "mir", "wir", "unser",
        "yo", "mi", "mis", "nosotros", "je", "mon", "ma", "mes", "moi",
        "nous",
    ]

    nonisolated static func speaksOfSelf(_ said: String) -> Bool {
        let lower = said.lowercased()
        let tokens = lower.split(whereSeparator: { c in !c.isLetter })
        let foreign = lower.unicodeScalars.contains { s in
            s.properties.isAlphabetic && s.value > 0x052F
        }
        let stated = lower.trimmingCharacters(in: .whitespacesAndNewlines)
        let asking = stated.hasSuffix("?")
            && Memories.leading(stated, sentences: 1) == stated
        return !asking && (foreign || tokens.contains { word in
            Memories.selfWords.contains(String(word))
        })
    }

    nonisolated static let askings = [
        "wants to", "would like", "asked", "is asking", "is interested",
        "is considering", "is curious", "wonders", "prefers to be called",
        "needs to know",
    ]

    nonisolated static func numbers(_ text: String) -> Set<String> {
        Set(text.split(whereSeparator: { c in !c.isNumber })
            .map(String.init))
    }

    nonisolated static let clauseWords = 3
    nonisolated static let clauseLimit = 3

    nonisolated static func clauses(_ question: String) -> [String] {
        let parts = question
            .replacingOccurrences(of: " and ", with: ",")
            .split(whereSeparator: { c in ",;?.!".contains(c) })
            .map { part in part.trimmingCharacters(in: .whitespaces) }
            .filter { part in
                part.split(separator: " ").count >= Memories.clauseWords
            }
        return parts.count > 1
            ? Array(parts.prefix(Memories.clauseLimit)) : []
    }

    public struct Coverage {
        public let covered: Bool
        public let known: [(id: String, title: String)]
        public let seconds: Double
    }

    public func coverage(_ text: String) async -> Coverage {
        var covered = false
        var known: [(id: String, title: String)] = []
        let began = Date()
        if isOpen, !concepts.isEmpty {
            let ticket = owner.searches.take()
            let found = await owner.search(ticket, [text],
                                           limit: Memories.recallLimit)
            if let found {
                let facts = found.hits.filter { hit in
                    hit.concept.type != ConversationNote.type
                }
                covered = facts.first.map { top in relevant(top) } ?? false
                known = facts.filter { hit in relevant(hit) }
                    .map { hit in (hit.concept.id, hit.concept.title) }
            } else {
                Diag.shared.report(.turn, "[extract] coverage superseded")
            }
        }
        return Coverage(covered: covered, known: known,
                        seconds: Date().timeIntervalSince(began))
    }

    static let quotedLimit = 600

    public static func extractionInstruction(
        known: [(id: String, title: String)], said: String = "") -> String {
        var out = extractionInstruction
        if !said.isEmpty {
            out = "The user's last message was:\n\"\"\"\n"
                + String(said.prefix(Memories.quotedLimit))
                + "\n\"\"\"\n" + out
        }
        if !known.isEmpty {
            out += "\nAlready on file: " + known.map { note in note.title }
                .joined(separator: "; ") + ". Leave out a fact one of them "
                + "already covers."
        }
        return out
    }

    public private(set) var seenIds: Set<String> = []
    public private(set) var noted: [Remembered] = []

    public func forgetSeen() {
        seenIds = []
        noted = []
    }

    public func takeNoted() -> [Remembered] {
        let out = noted
        noted = []
        return out
    }

    nonisolated static let generic: Set<String> = [
        "about", "after", "again", "always", "another", "because", "before",
        "being", "between", "could", "every", "known", "might", "never",
        "often", "other", "should", "since", "still", "their", "there",
        "these", "things", "those", "through", "under", "until", "usually",
        "where", "which", "while", "would",
    ]

    nonisolated static func words(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init)
            .filter { word in
                (word.count >= 5 && !Memories.generic.contains(word))
                    || (word.count == 4 && word.allSatisfy { c in c.isNumber })
            }
    }

    nonisolated static let stem = 5

    nonisolated static func grounded(_ draft: Draft, in said: String)
        -> Bool {
        let heard = Set(Memories.words(said).map { word in
            String(word.prefix(Memories.stem))
        })
        let claimed = Memories.words(draft.title + " " + draft.description)
        let spoken = claimed.filter { word in
            heard.contains(String(word.prefix(Memories.stem)))
        }
        let stated = draft.description.lowercased()
        let asking = Memories.askings.contains { phrase in
            stated.contains(phrase)
        }
        let counted = Memories.numbers(draft.description)
            .isSubset(of: Memories.numbers(said))
        return !claimed.isEmpty && spoken.count * 2 >= claimed.count
            && !asking && counted
    }

    public func remember(_ drafts: [Draft], said: String,
                         source: UUID?, excluding seen: Set<String>)
        async -> [Remembered] {
        var out: [Remembered] = []
        if isOpen {
            let got = await owner.remember(
                drafts, said: said, source: source,
                excluding: seen.union(seenIds), by: modelName)
            adopt(got.snapshot)
            out = got.kept
        }
        return out
    }

    public func keep(_ note: ConversationNote, title: String,
                     concluded: String) async {
        if isOpen {
            adopt(await owner.keep(note, title: title, concluded: concluded,
                                   by: modelName))
        }
    }

    nonisolated static func generatedLine(_ by: String) -> String {
        "generated: { by: " + by + ", at: "
            + Store.iso.string(from: Date()) + " }"
    }

    public var modelName = "model"

    func tool(_ name: String, _ args: [ToolArg]) async -> String {
        let ticket = name == "memory_search" ? owner.searches.take() : 0
        let reply = await owner.tool(name, args, ticket: ticket,
                                     by: modelName)
        seenIds.formUnion(reply.seen)
        if let note = reply.noted {
            noted.removeAll { seen in seen.id == note.id }
            noted.append(note)
        }
        adopt(reply.snapshot)
        return reply.text
    }

    nonisolated static func searchText(_ store: Store, _ result: SearchResult,
                                       _ shown: [Hit]) -> String {
        var out = ""
        if shown.isEmpty {
            out = "none of the user's notes is about this; answer from your "
                + "own knowledge\n"
        } else {
            for hit in shown { out += StoreText.line(store, hit) }
            let hidden = result.hits.count - shown.count
            if hidden > 0 {
                out += "\(hidden) other note(s) came up and none is about "
                    + "this\n"
            }
        }
        return out
    }
}
