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

    public static let folder = "memories.noindex"
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
        if !concept.description.isEmpty { out += ": " + concept.description }
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
            }
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
            var block = "Notes remembered about this user that may "
                + "bear on the message below:\n"
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
        public let id: String
        public let type: String
        public let title: String
        public let description: String
        public let tags: [String]
        public let body: String
    }

    public struct Remembered: Identifiable, Sendable {
        public let id: String
        public let title: String
    }

    public static let extractionInstruction =
        "From the user's LAST message and the answer to it, write what is "
        + "worth keeping about this user, at most five entries. Two kinds: "
        + "a durable fact they stated about themselves (preferences, "
        + "possessions, people, places, plans, habits, health or money "
        + "facts, still true in six months), and an interest, a subject "
        + "they asked to have explained or explored, one entry per subject "
        + "under the area interest saying what they asked, so a question "
        + "about X and Y is two entries, interest/x and interest/y, and a "
        + "third names the field both belong to when it is clear. Not a "
        + "lookup, a calculation, a translation or small talk, and never a "
        + "fact only the assistant supplied. "
        + "If there is nothing, reply NONE. Otherwise reply only entries in "
        + "this exact form:\n"
        + "### area/name\ntype: Note\ntitle: a few words\n"
        + "description: one sentence stating the fact, or what was asked\n"
        + "tags: comma separated; include private for medical, financial "
        + "or address facts\nbody:\none to three sentences\n"
        + "The id is an area word, a slash and a dashed name, like "
        + "person/coffee, house/roof-leak or interest/black-holes; areas are "
        + "person, house, work, family, health, interest and the like."

    static func parseDrafts(_ raw: String) -> [Draft] {
        var out: [Draft] = []
        var fields: [String: String] = [:]
        var body: [String] = []
        var id = ""
        var inBody = false
        func flush() {
            let title = fields["title"] ?? ""
            let draft = Draft(
                id: MemoryTools.validId(id)
                    ? id : MemoryTools.repaired(id, title),
                type: fields["type"] ?? "Note", title: title,
                description: fields["description"] ?? "",
                tags: MemoryTools.list(fields["tags"], ","),
                body: body.joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines))
            if MemoryTools.validId(draft.id), !draft.title.isEmpty,
               !draft.description.isEmpty {
                out.append(draft)
            }
            fields = [:]
            body = []
            inBody = false
        }
        for line in raw.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let marked = trimmed.hasPrefix("### ")
            let bare = marked
                ? String(trimmed.dropFirst(4)).trimmingCharacters(
                    in: .whitespaces)
                : trimmed
            if marked || (!inBody && MemoryTools.validId(bare.lowercased())) {
                if !id.isEmpty { flush() }
                id = bare.lowercased()
            } else if id.isEmpty || trimmed.hasPrefix("```") {
                inBody = inBody && !trimmed.hasPrefix("```")
            } else if inBody {
                body.append(line)
            } else if trimmed.hasPrefix("body:") {
                inBody = true
                let rest = String(trimmed.dropFirst(5))
                    .trimmingCharacters(in: .whitespaces)
                if !rest.isEmpty { body.append(rest) }
            } else if let colon = trimmed.firstIndex(of: ":") {
                let key = String(trimmed[..<colon]).lowercased()
                fields[key] = String(trimmed[trimmed.index(after: colon)...])
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        if !id.isEmpty { flush() }
        return out
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
                covered = found.hits.first.map { top in relevant(top) }
                    ?? false
                known = found.hits.filter { hit in relevant(hit) }
                    .map { hit in (hit.concept.id, hit.concept.title) }
            } else {
                Diag.shared.report(.turn, "[extract] coverage superseded")
            }
        }
        return Coverage(covered: covered, known: known,
                        seconds: Date().timeIntervalSince(began))
    }

    public static func extractionInstruction(
        known: [(id: String, title: String)]) -> String {
        var out = extractionInstruction
        if !known.isEmpty {
            out += "\nNotes already on file: " + known.map { note in
                note.id + " (" + note.title + ")"
            }.joined(separator: "; ") + ". A fact one of them already "
                + "covers is not new, leave it out; a fact that adds to one "
                + "of them uses that note's id."
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

    nonisolated static func grounded(_ draft: Draft, in said: String)
        -> Bool {
        let heard = Memories.words(said)
        let claimed = Memories.words(draft.title + " " + draft.description
                                     + " " + draft.body)
        return claimed.contains { word in
            heard.contains { spoken in
                spoken.hasPrefix(word) || word.hasPrefix(spoken)
            }
        }
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
