import Foundation
import LLM

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
    private var embedder: BertEmbedder?
    private(set) var store: Store?

    init(root: URL) {
        self.root = root
    }

    func open(excludedFromBackup: Bool) -> Opened? {
        var out: Opened? = nil
        if embedder == nil, let url = BertEmbedder.bundledMultilingual {
            embedder = BertEmbedder.load(ggufPath: url.path)
        }
        if let embedder {
            try? FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            Memories.exclude(root, excludedFromBackup)
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
            case "memory_search":
                out = searchTool(store, args, ticket: ticket)
            case "memory_read":
                out = readTool(store, args)
            case "memory_create", "memory_update":
                out = save(store, name == "memory_update", args, by: model)
            case "memory_forget":
                out = retire(store, MemoryTools.arg(args, "id") ?? "")
            default:
                out = ToolReply(text: "error: no tool named " + name)
            }
        }
        return out
    }

    private func searchTool(_ store: Store, _ args: [ToolArg],
                            ticket: Int) -> ToolReply {
        let queries = [MemoryTools.arg(args, "query") ?? ""]
            + MemoryTools.list(MemoryTools.arg(args, "also"), ";")
        let filter = Filter(area: MemoryTools.arg(args, "area") ?? "")
        let limit = Int(MemoryTools.arg(args, "limit") ?? "") ?? 8
        var out = ToolReply(text: "error: memory_search needs a query")
        if !queries[0].isEmpty {
            if let found = search(ticket, queries, filter: filter,
                                  limit: max(1, limit)) {
                let shown = found.hits.filter { hit in store.relevant(hit) }
                out = ToolReply(text: Memories.searchText(store, found, shown),
                                seen: shown.map { hit in hit.concept.id })
            } else {
                out.text = "error: the search was superseded"
            }
        }
        return out
    }

    private func readTool(_ store: Store, _ args: [ToolArg]) -> ToolReply {
        let id = MemoryTools.arg(args, "id") ?? ""
        var out = ToolReply(text: "error: no such note: " + id)
        if let concept = store.concept(id) {
            out = ToolReply(
                text: StoreText.read(
                    store, concept, about: MemoryTools.arg(args, "about"),
                    limit: Int(MemoryTools.arg(args, "limit") ?? "") ?? 2048,
                    offset: Int(MemoryTools.arg(args, "offset") ?? "") ?? 0),
                seen: [id])
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
            out.text = id + " already exists; use memory_update to replace it"
        } else if !found && existing {
            out.text = "no such note: " + id + "; use memory_create to add it"
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
                    + "memory_read it, and memory_update it only if this "
                    + "adds something new\n"
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
