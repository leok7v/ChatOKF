import Foundation

public final class Supersedable: @unchecked Sendable {
    private let lock = NSLock()
    private var issued = 0

    public init() {}

    public func take() -> Int {
        lock.lock()
        issued += 1
        let ticket = issued
        lock.unlock()
        return ticket
    }

    public func current(_ ticket: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ticket == issued
    }
}

struct MemorySnapshot: Sendable {
    let serial: Int
    let concepts: [Concept]
    let list: [MemoryRow]
    let trashed: [MemoryRow]
    let floor: Float

    static let empty = MemorySnapshot(serial: 0, concepts: [], list: [],
                                      trashed: [], floor: 0)
}

actor MemoryStore {

    struct Opened: Sendable {
        let snapshot: MemorySnapshot
        let embedded: Int
        let seconds: Double
    }

    struct ToolReply: Sendable {
        var text: String
        var seen: [String] = []
        var noted: Memories.Remembered? = nil
        var snapshot: MemorySnapshot? = nil
    }

    let root: URL
    nonisolated let searches = Supersedable()
    private var embedder: Embedder?
    private(set) var store: Store?

    init(root: URL) {
        self.root = root
    }

    func open(excludedFromBackup: Bool) -> Opened? {
        var out: Opened? = nil
        if embedder == nil, let path = EncoderPack.path {
            embedder = ModernBertEmbedder.load(ggufPath: path)
        }
        if embedder == nil {
            Diag.shared.report(.load, "[memories] not open: no encoder on "
                               + "disk yet")
        }
        if let embedder {
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            Memories.exclude(root, excludedFromBackup)
            sweepSidecars(keeping: Store.sidecarName(for: embedder.name))
            let t0 = Date()
            let loaded = Store(root: root, embedder: embedder)
            loaded.load()
            store = loaded
            purgeExpired()
            out = Opened(snapshot: snapshot(), embedded: loaded.embeddedCount,
                         seconds: Date().timeIntervalSince(t0))
        }
        return out
    }

    private func sweepSidecars(keeping name: String) {
        let fm = FileManager.default
        let found = (try? fm.contentsOfDirectory(atPath: root.path)) ?? []
        for file in found
        where file.hasPrefix(Store.sidecarPrefix) && file != name {
            try? fm.removeItem(at: root.appendingPathComponent(file))
            Diag.shared.report(.load, "[memories] removed " + file)
        }
    }

    func erase() {
        store = nil
        try? FileManager.default.removeItem(at: root)
    }

    func snapshot() -> MemorySnapshot {
        snapshot(trashed: readTrash())
    }

    private var snapshots = 0

    func snapshot(trashed: [TrashedMemory]) -> MemorySnapshot {
        let concepts = store?.concepts ?? []
        snapshots += 1
        return MemorySnapshot(
            serial: snapshots,
            concepts: concepts,
            list: concepts.map { concept in MemoryStore.row(concept) }
                .sorted { a, b in a.updated > b.updated },
            trashed: trashed
                .sorted { a, b in a.trashedAt > b.trashedAt }
                .map { entry in MemoryStore.row(entry) },
            floor: embedder?.relevanceFloor ?? 0)
    }

    func map() -> String {
        var out = ""
        if let store, !store.concepts.isEmpty {
            out = StoreText.map(store, ids: false)
        }
        return out
    }

    func search(_ ticket: Int, _ queries: [String],
                filter: Filter = Filter(), limit: Int) -> SearchResult? {
        var out: SearchResult? = nil
        if let store, searches.current(ticket) {
            out = store.search(queries, filter: filter, limit: limit)
        }
        return out
    }

    func remember(_ drafts: [Memories.Draft], said: String, source: UUID?,
                  excluding seen: Set<String>, by model: String)
        -> (kept: [Memories.Remembered], snapshot: MemorySnapshot?) {
        var kept: [Memories.Remembered] = []
        if let store {
            for draft in drafts where !Memories.grounded(draft, in: said) {
                Diag.shared.report(.turn, "[extract] \"" + draft.title
                                   + "\" is not what the user said")
            }
            for draft in drafts where Memories.grounded(draft, in: said) {
                if let id = write(store, draft, source: source,
                                  excluding: seen, by: model) {
                    kept.append(Memories.Remembered(id: id,
                                                    title: draft.title))
                    store.reload(id: id)
                }
            }
        }
        return (kept, kept.isEmpty ? nil : snapshot())
    }

    private func write(_ store: Store, _ draft: Memories.Draft,
                       source: UUID?, excluding seen: Set<String>,
                       by model: String) -> String? {
        let area = draft.area.isEmpty
            ? self.area(for: draft.title + ". " + draft.description)
            : draft.area
        let kept = draft.isPrivate
            || MemoryStore.privateAreas.contains(area)
        let tags = [area] + (kept ? [MemoryRow.privateTag] : [])
        var id = area + "/" + MemoryTools.slug(draft.title, words: 6)
        if store.concept(id) == nil, let same = store.duplicate(
            title: draft.title, description: draft.description,
            tags: tags, type: MemoryStore.factType),
           store.concept(same.id)?.type != ConversationNote.type {
            Diag.shared.report(.turn, String(
                format: "[extract] %@ restates %@ (%.3f)", id, same.id,
                same.score))
            id = same.id
        }
        let existing = store.concept(id)
        var lines = [Memories.generatedLine(model)]
        if let source {
            lines += ["sources:",
                      "  - resource: chatokf://conversation/"
                          + source.uuidString]
        }
        var wrote: URL? = nil
        if existing?.trust != .human, !seen.contains(id),
           MemoryTools.validId(id) {
            wrote = try? store.write(
                id: id, type: MemoryStore.factType, title: draft.title,
                description: draft.description, tags: tags,
                body: "", status: "",
                adding: existing == nil ? lines : [])
        }
        return wrote == nil ? nil : id
    }

    static let factType = "Note"
    static let privateAreas: Set<String> = ["health", "money"]

    static let areas: [(name: String, gloss: String)] = [
        ("person", "the user: name, age, tastes, preferences, habits"),
        ("family", "family, partner, children, parents, friends"),
        ("pets", "pets and animals the user keeps"),
        ("home", "house, flat, garden, car, things the user owns"),
        ("work", "job, profession, projects, colleagues, studies"),
        ("health", "health, illness, medicine, diet, exercise"),
        ("money", "money, income, savings, debts, purchases"),
        ("travel", "trips, places lived in or visited, plans to go"),
    ]

    private var areaVectors: [[Float]] = []

    func area(for text: String) -> String {
        var out = MemoryStore.areas[0].name
        if let embedder {
            if areaVectors.isEmpty {
                areaVectors = MemoryStore.areas.map { area in
                    embedder.embedPassage(area.name + ": " + area.gloss)
                }
            }
            let asked = embedder.embedQuery(text)
            var best = -Float.greatestFiniteMagnitude
            for (at, vector) in areaVectors.enumerated() {
                var score: Float = 0
                for i in 0..<min(asked.count, vector.count) {
                    score += asked[i] * vector[i]
                }
                if score > best {
                    best = score
                    out = MemoryStore.areas[at].name
                }
            }
        }
        return out
    }

    func scores(_ query: String, _ texts: [String]) -> [Float] {
        var out: [Float] = []
        if let embedder {
            let asked = embedder.embedQuery(query)
            out = texts.map { text in
                let vector = embedder.embedPassage(text)
                var score: Float = 0
                for i in 0..<min(asked.count, vector.count) {
                    score += asked[i] * vector[i]
                }
                return score
            }
        }
        return out
    }

    func keep(_ note: ConversationNote, title: String, concluded: String,
              by model: String) -> MemorySnapshot? {
        var out: MemorySnapshot? = nil
        let id = ConversationNote.id(note.conversation)
        if let store, !note.asked.isEmpty {
            let earlier = store.concept(id)
            let fresh = earlier == nil
            let lines = [Memories.generatedLine(model), "sources:",
                         "  - resource: " + MemoryStore.sourceMark
                             + note.conversation.uuidString]
            let reached = note.reached(concluded,
                                       after: earlier?.body ?? "")
            let wrote = try? store.write(
                id: id, type: ConversationNote.type, title: title,
                description: note.description(reached),
                tags: [ConversationNote.tag], body: note.body(reached),
                status: "", adding: fresh ? lines : [])
            if wrote != nil {
                store.reload(id: id)
                out = snapshot()
            }
        }
        return out
    }

    func tool(_ name: String, _ args: [ToolArg], ticket: Int,
              by model: String) -> ToolReply {
        var out = ToolReply(text: "error: memories are not open")
        if let store {
            switch name {
            case MemoryTools.search:
                out = searchTool(store, args, ticket: ticket)
            case MemoryTools.grep:
                out = grepTool(store, args)
            case MemoryTools.read:
                out = readTool(store, args)
            case MemoryTools.create, MemoryTools.update:
                out = save(store, name == MemoryTools.update, args, by: model)
            case MemoryTools.forget:
                out = retire(store, MemoryTools.arg(args, "id") ?? "")
            default:
                out = ToolReply(text: "error: no tool named " + name)
            }
        }
        return out
    }

    static let searchLines = 8

    private func searchTool(_ store: Store, _ args: [ToolArg],
                            ticket: Int) -> ToolReply {
        let queries = [MemoryTools.arg(args, "query") ?? ""]
            + MemoryTools.list(MemoryTools.arg(args, "also"), ";")
        let filter = Filter(area: MemoryTools.arg(args, "area") ?? "")
        let limit = Int(MemoryTools.arg(args, "limit") ?? "") ?? 8
        var out = ToolReply(text: "error: " + MemoryTools.search
                            + " needs a query")
        if !queries[0].isEmpty {
            if let found = search(ticket, queries, filter: filter,
                                  limit: max(1, limit)) {
                let shown = found.hits.filter { hit in store.relevant(hit) }
                var text = Memories.searchText(store, found, shown)
                let terms = NoteFind.contentTerms(queries[0])
                if !terms.isEmpty {
                    text += "lines anywhere in the notes that hold the words "
                        + "of the question:\n"
                        + NoteFind.lines(terms, asked: queries[0],
                                         limit: MemoryStore.searchLines,
                                         in: store.concepts)
                }
                out = ToolReply(text: text,
                                seen: shown.map { hit in hit.concept.id })
            } else {
                out.text = "error: the search was superseded"
            }
        }
        return out
    }

    private func grepTool(_ store: Store, _ args: [ToolArg]) -> ToolReply {
        let asked = MemoryTools.arg(args, "words") ?? ""
        let limit = Int(MemoryTools.arg(args, "limit") ?? "")
            ?? NoteFind.lineLimit
        let terms = NoteFind.terms(asked)
        var out = ToolReply(text: "error: " + MemoryTools.grep
                            + " needs one or more words")
        if !terms.isEmpty {
            out = ToolReply(text: NoteFind.lines(terms, asked: asked,
                                                 limit: limit,
                                                 in: store.concepts))
        }
        return out
    }

    private var readEnds: [String: Int] = [:]

    func beginTurn() {
        readEnds = [:]
    }

    private func resolve(_ store: Store, _ asked: String)
        -> (id: String, line: Int?) {
        var id = asked
        var glued: Int? = nil
        if store.concept(id) == nil, let colon = id.lastIndex(of: ":"),
           let n = Int(id[id.index(after: colon)...]) {
            glued = n
            id = String(id[..<colon])
        }
        if store.concept(id) == nil {
            let leaf = (id as NSString).lastPathComponent.lowercased()
            let same = store.concepts.filter { concept in
                (concept.id as NSString).lastPathComponent.lowercased()
                    == leaf
            }
            if same.count == 1 { id = same[0].id }
        }
        return (id, glued)
    }

    private func readTool(_ store: Store, _ args: [ToolArg]) -> ToolReply {
        let found = resolve(store, MemoryTools.arg(args, "id") ?? "")
        var out = ToolReply(text: "error: no such note: " + found.id
                            + "; ids are exactly as " + MemoryTools.grep
                            + " and " + MemoryTools.search + " print them")
        if let concept = store.concept(found.id) {
            let count = Int(MemoryTools.arg(args, "lines") ?? "")
                ?? NoteFind.readLines
            let about = MemoryTools.arg(args, "about") ?? ""
            var start = found.line
                ?? Int(MemoryTools.arg(args, "line") ?? "") ?? 1
            if !about.isEmpty, found.line == nil,
               MemoryTools.arg(args, "line") == nil {
                let span = store.window(concept, about: about, limit: 1)
                start = NoteFind.lineOf(concept, byteOffset: span.from)
            }
            let key = concept.id + "#" + about + "#" + String(start)
            if let last = readEnds[key] { start = last + 1 }
            let read = NoteFind.read(concept, from: start, count: count,
                                     more: MemoryTools.read)
            readEnds[key] = read.end
            out = ToolReply(text: read.text, seen: [concept.id])
        }
        return out
    }

    static let pageType = "Page"
    static let pictureType = "Picture"
    static let documentType = "Document"
    static let fetchedKey = "fetched:"
    static let freshDays = 7.0
    static let descriptionLimit = 200

    static func fetchedAt(_ concept: Concept) -> Date? {
        var out: Date? = nil
        for line in concept.extraFrontmatter
        where line.hasPrefix(MemoryStore.fetchedKey) {
            out = Store.iso.date(from: String(line.dropFirst(
                MemoryStore.fetchedKey.count))
                .trimmingCharacters(in: .whitespaces))
        }
        return out
    }

    func page(_ name: String, _ args: [ToolArg]) -> String? {
        var out: String? = nil
        if let store, let id = MemoryTools.pageId(name, args),
           let concept = store.concept(id),
           let at = MemoryStore.fetchedAt(concept),
           Date().timeIntervalSince(at) < MemoryStore.freshDays * 86400 {
            let day = String(Store.iso.string(from: at).prefix(10))
            out = "From the user's notes, fetched on \(day) (note \(id)):"
                + "\n" + concept.body + "\n"
        }
        return out
    }

    private func stamped(_ source: String) -> [String] {
        [MemoryStore.fetchedKey + " " + Store.iso.string(from: Date()),
         "sources:", "  - resource: " + source]
    }

    func keepPage(_ name: String, _ args: [ToolArg], text: String)
        -> MemorySnapshot? {
        var out: MemorySnapshot? = nil
        if let store, !MemoryTools.pageFailed(text),
           let id = MemoryTools.pageId(name, args) {
            let lines = text.components(separatedBy: "\n")
            let source = MemoryTools.arg(args, "url")
                ?? "wikipedia:" + (MemoryTools.arg(args, "query") ?? "")
            let wrote = try? store.write(
                id: id, type: MemoryStore.pageType,
                title: MemoryTools.pageTitle(name, args, text),
                description: NoteFind.cut(
                    lines.dropFirst().joined(separator: "\n"),
                    limit: MemoryStore.descriptionLimit),
                tags: ["pages"], body: text, adding: stamped(source))
            if wrote != nil {
                store.reload(id: id)
                out = snapshot()
                Diag.shared.report(.turn, "[notes] kept page " + id)
            }
        }
        return out
    }

    func keepPicture(names: [String], stamp: String, caption: String,
                     answer: String) -> MemorySnapshot? {
        var out: MemorySnapshot? = nil
        if let store, !caption.isEmpty, let first = names.first {
            let id = "pictures/" + NoteFind.slug(first) + "-" + stamp
            let body = "Picture " + names.joined(separator: ", ") + "\n\n"
                + caption + "\n\nWhat was said about it:\n" + answer
            let wrote = try? store.write(
                id: id, type: MemoryStore.pictureType, title: first,
                description: NoteFind.cut(
                    caption, limit: MemoryStore.descriptionLimit),
                tags: ["pictures"], body: body,
                adding: stamped("picture:" + first))
            if wrote != nil {
                store.reload(id: id)
                out = snapshot()
                Diag.shared.report(.turn, "[notes] kept picture " + id)
            }
        }
        return out
    }

    func importMarkdown(from dir: String) -> (count: Int,
                                               snapshot: MemorySnapshot?) {
        let fm = FileManager.default
        let names = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { name in name.hasSuffix(".md") }.sorted()
        var count = 0
        for name in names {
            let path = dir + "/" + name
            if let text = try? String(contentsOfFile: path, encoding: .utf8),
               keepDocument(name: String(name.dropLast(3)),
                            text: NoteFind.shaped(text)) != nil {
                count += 1
            }
        }
        return (count, count > 0 ? snapshot() : nil)
    }

    func keepDocument(name: String, text: String) -> MemorySnapshot? {
        var out: MemorySnapshot? = nil
        if let store, !text.isEmpty {
            let id = "docs/" + NoteFind.slug(name)
            var lines = text.components(separatedBy: "\n")
            if lines.first?.trimmingCharacters(in: .whitespaces) == "---",
               let end = lines.dropFirst().firstIndex(where: { line in
                   line.trimmingCharacters(in: .whitespaces) == "---"
               }) {
                lines = Array(lines[(end + 1)...])
            }
            let first = lines.first { line in
                let t = line.trimmingCharacters(in: .whitespaces)
                return !t.isEmpty && !t.hasPrefix("#") && !t.hasPrefix("<!--")
                    && !t.hasPrefix(">") && !t.hasPrefix("*")
                    && !t.hasPrefix("|")
            } ?? ""
            let wrote = try? store.write(
                id: id, type: MemoryStore.documentType, title: name,
                description: NoteFind.cut(
                    first, limit: MemoryStore.descriptionLimit),
                tags: ["docs"], body: text,
                adding: stamped("document:" + name))
            if wrote != nil {
                store.reload(id: id)
                out = snapshot()
                Diag.shared.report(.turn, "[notes] kept document " + id
                    + " (\(text.utf8.count) bytes)")
            }
        }
        return out
    }

    private func save(_ store: Store, _ existing: Bool, _ args: [ToolArg],
                      by model: String) -> ToolReply {
        let id = MemoryTools.arg(args, "id") ?? ""
        let found = store.concept(id) != nil
        var out = ToolReply(text: "")
        if !MemoryTools.validId(id) {
            out.text = "error: the id needs an area word, a slash and a "
                + "short dashed name. If the user did not ask you to remember "
                + "this, do not save it: answer the user instead."
        } else if found && !existing {
            out.text = id + " already exists; use " + MemoryTools.update
                + " to replace it"
        } else if !found && existing {
            out.text = "no such note: " + id + "; use " + MemoryTools.create
                + " to add it"
        } else {
            let type = MemoryTools.arg(args, "type") ?? "Note"
            let title = MemoryTools.arg(args, "title") ?? id
            let description = MemoryTools.arg(args, "description") ?? ""
            let tags = MemoryTools.list(MemoryTools.arg(args, "tags"), ",")
            let same = found ? nil : store.duplicate(
                title: title, description: description, tags: tags,
                type: type)
            if let same {
                out.text = id + " restates " + same.id + " ("
                    + (store.concept(same.id)?.title ?? "") + "); "
                    + MemoryTools.read + " it, and " + MemoryTools.update
                    + " it only if this adds something new\n"
            } else {
                do {
                    _ = try store.write(
                        id: id, type: type, title: title,
                        description: description, tags: tags,
                        body: MemoryTools.arg(args, "body") ?? "", status: "",
                        adding: [Memories.generatedLine(model)],
                        dropping: found ? ["generated", "verified"] : [])
                    store.reload(id: id)
                    out = ToolReply(
                        text: StoreText.saved(store, id: id, existed: found),
                        seen: [id],
                        noted: Memories.Remembered(id: id, title: title),
                        snapshot: snapshot())
                } catch {
                    out.text = "error: write failed: \(error)"
                }
            }
        }
        return out
    }

    private func retire(_ store: Store, _ id: String) -> ToolReply {
        var out = ToolReply(text: "")
        do {
            let referrers = try store.deprecate(id: id)
            store.reload(id: id)
            out = ToolReply(text: StoreText.retired(store, id: id,
                                                    referrers: referrers),
                            snapshot: snapshot())
        } catch {
            out.text = "\(error)"
        }
        return out
    }
}
