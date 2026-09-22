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

    private struct Opened: @unchecked Sendable {
        let embedder: BertEmbedder
        let store: Store
        let seconds: Double
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

    private var embedder: BertEmbedder?
    private(set) var store: Store?
    private var opening: Task<Void, Never>?

    public internal(set) var list: [MemoryRow] = []
    public internal(set) var trashed: [MemoryRow] = []

    public init(root: URL = Memories.defaultRoot) {
        let d = UserDefaults.standard
        self.root = root
        enabled = d.object(forKey: Memories.enabledKey) as? Bool ?? true
        backup = d.bool(forKey: Memories.backupKey)
    }

    public var active: Bool {
        Memories.supported && enabled && Flags.on("memories")
    }

    public var isOpen: Bool { store != nil }

    public var count: Int { list.count }

    public func open() {
        if store == nil, opening == nil, active {
            let root = self.root
            let excluded = !backup
            opening = Task { [weak self] in
                let opened = await Task.detached(priority: .utility) {
                    Memories.load(root, excludedFromBackup: excluded)
                }.value
                if let self, let opened, !Task.isCancelled {
                    self.embedder = opened.embedder
                    self.store = opened.store
                    self.purgeExpired()
                    self.refresh()
                    Diag.shared.report(.load, String(
                        format: "[memories] %d concepts, re-embedded %d, "
                            + "%.2fs at %@", opened.store.concepts.count,
                        opened.store.embeddedCount, opened.seconds,
                        root.path))
                }
                self?.opening = nil
            }
        }
    }

    public func awaitOpen() async {
        open()
        await opening?.value
    }

    nonisolated private static func load(_ root: URL,
                                         excludedFromBackup: Bool) -> Opened? {
        var out: Opened? = nil
        if let url = BertEmbedder.bundledMultilingual,
           let embedder = BertEmbedder.load(ggufPath: url.path) {
            let fm = FileManager.default
            try? fm.createDirectory(at: root,
                                    withIntermediateDirectories: true)
            Memories.exclude(root, excludedFromBackup)
            let t0 = Date()
            let store = Store(root: root, embedder: embedder)
            store.load()
            out = Opened(embedder: embedder, store: store,
                         seconds: Date().timeIntervalSince(t0))
        }
        return out
    }

    nonisolated private static func exclude(_ root: URL, _ on: Bool) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = on
        var url = root
        try? url.setResourceValues(values)
    }

    public func reopen() {
        opening?.cancel()
        opening = nil
        store = nil
        list = []
        trashed = []
        open()
    }

    public var map: String {
        var out = ""
        if let store, !store.concepts.isEmpty {
            out = StoreText.map(store, ids: false)
        }
        return out
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

    public func recall(_ question: String, also: [String], pp: Double,
                       excluding read: Set<String>) -> Recall? {
        var out: Recall? = nil
        if let store, let embedder, !store.concepts.isEmpty {
            let queries = [question] + also.filter { text in
                !text.isEmpty
            }
            let began = Date()
            let result = store.search(queries, filter: Filter(),
                                      limit: Memories.recallLimit + read.count)
            let searched = Date().timeIntervalSince(began)
            let fresh = result.hits.filter { hit in
                !read.contains(hit.concept.id) && store.relevant(hit)
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
                    unread?.relevance ?? 0, embedder.relevanceFloor,
                    result.standout, fresh.count, store.concepts.count,
                    searched))
            }
        } else if store == nil, active {
            Diag.shared.report(.turn, "[recall] the store is not open yet")
            open()
        }
        return out
    }

    static func seconds(bytes: Int, _ pp: Double) -> Double {
        pp > 0 ? Double(bytes) / bytesPerToken / pp : 0
    }

    public func note(_ id: String) -> Note? {
        var out: Note? = nil
        if let concept = store?.concept(id) {
            var text = "My note \"" + concept.title + "\""
            if !concept.description.isEmpty {
                text += ": " + concept.description
            }
            text += "\n\n" + concept.body
            out = Note(id: concept.id, title: concept.title, text: text)
        }
        return out
    }

    public struct Draft {
        public let id: String
        public let type: String
        public let title: String
        public let description: String
        public let tags: [String]
        public let body: String
    }

    public struct Remembered: Identifiable {
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

    public func coverage(_ text: String) -> Coverage {
        var covered = false
        var known: [(id: String, title: String)] = []
        let began = Date()
        if let store, !store.concepts.isEmpty {
            let result = store.search([text], filter: Filter(),
                                      limit: Memories.recallLimit)
            covered = store.confident(result)
            known = result.hits.filter { hit in store.relevant(hit) }
                .map { hit in (hit.concept.id, hit.concept.title) }
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
        -> [Remembered] {
        var out: [Remembered] = []
        if let store {
            for draft in drafts where !Memories.grounded(draft, in: said) {
                Diag.shared.report(.turn, "[extract] " + draft.id
                                   + " shares no word with what the user said")
            }
            for draft in drafts where Memories.grounded(draft, in: said) {
                var id = draft.id
                if store.concept(id) == nil,
                   let same = store.duplicate(
                       title: draft.title, description: draft.description,
                       tags: draft.tags, type: draft.type) {
                    Diag.shared.report(.turn, String(
                        format: "[extract] %@ restates %@ (%.3f)", draft.id,
                        same.id, same.score))
                    id = same.id
                }
                let existing = store.concept(id)
                var lines = [Memories.generatedLine(modelName)]
                if let source {
                    lines += ["sources:",
                              "  - resource: chatokf://conversation/"
                                  + source.uuidString]
                }
                var wrote: URL? = nil
                if existing?.trust != .human, !seen.contains(id),
                   !seenIds.contains(id) {
                    wrote = try? store.write(
                        id: id, type: draft.type, title: draft.title,
                        description: draft.description, tags: draft.tags,
                        body: draft.body, status: "",
                        adding: existing == nil ? lines : [])
                }
                if wrote != nil {
                    out.append(Remembered(id: id, title: draft.title))
                }
            }
            if !out.isEmpty {
                store.load()
                refresh()
            }
        }
        return out
    }

    static func generatedLine(_ by: String) -> String {
        "generated: { by: " + by + ", at: "
            + Store.iso.string(from: Date()) + " }"
    }

    public var modelName = "model"

    func tool(_ name: String, _ args: [ToolArg]) -> String {
        var out = "error: memories are not open"
        if let store {
            switch name {
            case "memory_search":
                let queries = [MemoryTools.arg(args, "query") ?? ""]
                    + MemoryTools.list(MemoryTools.arg(args, "also"), ";")
                let filter = Filter(area: MemoryTools.arg(args, "area") ?? "")
                let limit = Int(MemoryTools.arg(args, "limit") ?? "") ?? 8
                if queries[0].isEmpty {
                    out = "error: memory_search needs a query"
                } else {
                    let found = store.search(queries, filter: filter,
                                             limit: max(1, limit))
                    seenIds.formUnion(found.hits.map { hit in hit.concept.id })
                    out = StoreText.search(store, queries: queries,
                                           filter: filter, limit: max(1, limit))
                }
            case "memory_read":
                let id = MemoryTools.arg(args, "id") ?? ""
                if let concept = store.concept(id) {
                    seenIds.insert(id)
                    out = StoreText.read(
                        store, concept, about: MemoryTools.arg(args, "about"),
                        limit: Int(MemoryTools.arg(args, "limit") ?? "")
                            ?? 2048,
                        offset: Int(MemoryTools.arg(args, "offset") ?? "")
                            ?? 0)
                } else {
                    out = "error: no such note: " + id
                }
            case "memory_create", "memory_update":
                out = save(name == "memory_update", args)
            case "memory_forget":
                out = retire(MemoryTools.arg(args, "id") ?? "")
            default:
                out = "error: no tool named " + name
            }
        }
        return out
    }

    private func save(_ existing: Bool, _ args: [ToolArg]) -> String {
        let id = MemoryTools.arg(args, "id") ?? ""
        let found = store?.concept(id) != nil
        var out = ""
        if !MemoryTools.validId(id) {
            out = "error: an id is area/name, letters, digits and dashes"
        } else if found && !existing {
            out = id + " already exists; use memory_update to replace it"
        } else if !found && existing {
            out = "no such note: " + id + "; use memory_create to add it"
        } else if let store {
            let type = MemoryTools.arg(args, "type") ?? "Note"
            let title = MemoryTools.arg(args, "title") ?? id
            let description = MemoryTools.arg(args, "description") ?? ""
            let tags = MemoryTools.list(MemoryTools.arg(args, "tags"), ",")
            let same = found ? nil : store.duplicate(
                title: title, description: description, tags: tags,
                type: type)
            if let same {
                out = id + " restates " + same.id + " ("
                    + (store.concept(same.id)?.title ?? "") + "); "
                    + "memory_read it, and memory_update it only if this "
                    + "adds something new\n"
            } else {
                do {
                    _ = try store.write(
                        id: id, type: type, title: title,
                        description: description, tags: tags,
                        body: MemoryTools.arg(args, "body") ?? "",
                        status: "",
                        adding: [Memories.generatedLine(modelName)],
                        dropping: found ? ["generated", "verified"] : [])
                    store.load()
                    refresh()
                    seenIds.insert(id)
                    noted.removeAll { note in note.id == id }
                    noted.append(Remembered(id: id, title: title))
                    out = StoreText.saved(store, id: id, existed: found)
                } catch {
                    out = "error: write failed: \(error)"
                }
            }
        }
        return out
    }

    private func retire(_ id: String) -> String {
        var out = "error: memories are not open"
        if let store {
            do {
                let referrers = try store.deprecate(id: id)
                store.load()
                refresh()
                out = StoreText.retired(store, id: id, referrers: referrers)
            } catch {
                out = "\(error)"
            }
        }
        return out
    }

    public func forgetAll() {
        opening?.cancel()
        opening = nil
        store = nil
        list = []
        trashed = []
        try? FileManager.default.removeItem(at: root)
    }

    public static func erase() {
        try? FileManager.default.removeItem(at: defaultRoot)
    }
}
