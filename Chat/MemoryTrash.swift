import Foundation
import LLM

public struct MemoryRow: Identifiable, Sendable {

    public static let privateTag = "private"

    public let id: String
    public let area: String
    public let title: String
    public let description: String
    public let tags: [String]
    public let source: UUID?
    public let updated: Date

    public var isPrivate: Bool { tags.contains(MemoryRow.privateTag) }
}

public struct MemoryNote: Sendable {
    public let row: MemoryRow
    public let type: String
    public let body: String
    public let links: [String]
    public let backlinks: [String]
    public let retired: Bool
}

struct TrashedMemory: Codable, Sendable {

    struct Cut: Codable, Sendable {
        let referrer: String
        let markup: String
        let label: String
    }

    let id: String
    let title: String
    let description: String
    let tags: [String]
    let source: String?
    let updated: Date
    let trashedAt: Date
    let unlinked: [Cut]
}

extension Memories {

    nonisolated public static let trashFolder = ".trash"
    nonisolated public static let retention = ConversationStore.trashRetention

    public func notes(from conversation: UUID) -> [MemoryRow] {
        list.filter { row in row.source == conversation }
    }

    public func note(detail id: String) -> MemoryNote? {
        var out: MemoryNote? = nil
        if let concept = concepts[id],
           let row = list.first(where: { row in row.id == id }) {
            out = MemoryNote(row: row, type: concept.type,
                             body: concept.body, links: concept.links,
                             backlinks: concept.backlinks,
                             retired: concept.isDeprecated)
        }
        return out
    }

    public func trash(_ id: String) async {
        adopt(await owner.trash(id))
    }

    public func restore(_ id: String) async {
        adopt(await owner.restore(id))
    }

    public func deleteForever(_ id: String) async {
        adopt(await owner.deleteForever(id))
    }

    public func emptyTrash() async {
        adopt(await owner.emptyTrash())
    }
}

extension MemoryStore {

    static let sourceMark = "chatokf://conversation/"

    func trash(_ id: String) -> MemorySnapshot? {
        var out: MemorySnapshot? = nil
        if let store, let concept = store.concept(id) {
            let cuts = (try? store.unlink(id)) ?? []
            let row = MemoryStore.row(concept)
            let to = trashURL(id)
            let fm = FileManager.default
            try? fm.createDirectory(at: to.deletingLastPathComponent(),
                                    withIntermediateDirectories: true)
            try? fm.removeItem(at: to)
            try? fm.moveItem(at: concept.path, to: to)
            prune(concept.path.deletingLastPathComponent(), under: root)
            var kept = readTrash().filter { entry in entry.id != id }
            kept.append(TrashedMemory(
                id: row.id, title: row.title, description: row.description,
                tags: row.tags, source: row.source?.uuidString,
                updated: row.updated, trashedAt: Date(),
                unlinked: cuts.map { cut in
                    TrashedMemory.Cut(referrer: cut.referrer,
                                      markup: cut.markup, label: cut.label)
                }))
            writeTrash(kept)
            store.reload(id: id)
            for referrer in Set(cuts.map { cut in cut.referrer }) {
                store.reload(id: referrer)
            }
            out = snapshot(trashed: kept)
            Diag.shared.report(.turn, "[memories] trashed \(id), "
                + "\(cuts.count) link(s) collapsed")
        }
        return out
    }

    func restore(_ id: String) -> MemorySnapshot? {
        let kept = readTrash()
        var out: MemorySnapshot? = nil
        if let store, let entry = kept.first(where: { e in e.id == id }) {
            let to = root.appendingPathComponent(id + ".md")
            let fm = FileManager.default
            try? fm.createDirectory(at: to.deletingLastPathComponent(),
                                    withIntermediateDirectories: true)
            try? fm.moveItem(at: trashURL(id), to: to)
            store.relink(entry.unlinked.map { cut in
                RemovedLink(referrer: cut.referrer, markup: cut.markup,
                            label: cut.label)
            })
            let remaining = kept.filter { e in e.id != id }
            writeTrash(remaining)
            store.reload(id: id)
            for referrer in Set(entry.unlinked.map { cut in cut.referrer }) {
                store.reload(id: referrer)
            }
            out = snapshot(trashed: remaining)
        }
        return out
    }

    func deleteForever(_ id: String) -> MemorySnapshot {
        let url = trashURL(id)
        try? FileManager.default.removeItem(at: url)
        prune(url.deletingLastPathComponent(), under: trashDir())
        let remaining = readTrash().filter { entry in entry.id != id }
        writeTrash(remaining)
        return snapshot(trashed: remaining)
    }

    func emptyTrash() -> MemorySnapshot {
        try? FileManager.default.removeItem(at: trashDir())
        return snapshot()
    }

    func purgeExpired() {
        let now = Date()
        let kept = readTrash()
        let live = kept.filter { entry in
            now.timeIntervalSince(entry.trashedAt) <= Memories.retention
        }
        if live.count != kept.count {
            for entry in kept where !live.contains(where: { e in
                e.id == entry.id
            }) {
                try? FileManager.default.removeItem(at: trashURL(entry.id))
            }
            writeTrash(live)
        }
    }

    static func row(_ concept: Concept) -> MemoryRow {
        MemoryRow(id: concept.id, area: Store.area(of: concept.id),
                  title: concept.title, description: concept.description,
                  tags: concept.tags,
                  source: MemoryStore.source(concept.extraFrontmatter),
                  updated: MemoryStore.modified(concept.path))
    }

    static func row(_ entry: TrashedMemory) -> MemoryRow {
        MemoryRow(id: entry.id, area: Store.area(of: entry.id),
                  title: entry.title, description: entry.description,
                  tags: entry.tags,
                  source: entry.source.flatMap { text in
                      UUID(uuidString: text)
                  },
                  updated: entry.updated)
    }

    static func source(_ lines: [String]) -> UUID? {
        var out: UUID? = nil
        for line in lines where out == nil {
            if let at = line.range(of: MemoryStore.sourceMark) {
                out = UUID(uuidString: String(line[at.upperBound...])
                    .trimmingCharacters(in: .whitespaces))
            }
        }
        return out
    }

    static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? Date()
    }

    func trashDir() -> URL {
        root.appendingPathComponent(Memories.trashFolder, isDirectory: true)
    }

    func trashURL(_ id: String) -> URL {
        trashDir().appendingPathComponent(id + ".md")
    }

    private func manifestURL() -> URL {
        trashDir().appendingPathComponent("trash.json")
    }

    func readTrash() -> [TrashedMemory] {
        let data = try? Data(contentsOf: manifestURL())
        return data.flatMap { bytes in
            try? JSONDecoder().decode([TrashedMemory].self, from: bytes)
        } ?? []
    }

    func writeTrash(_ entries: [TrashedMemory]) {
        let url = manifestURL()
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(entries) {
            try? data.write(to: url)
        }
    }

    private func prune(_ directory: URL, under base: URL) {
        var at = directory
        var pruning = at.path.hasPrefix(base.path) && at.path != base.path
        while pruning {
            let entries = try? FileManager.default
                .contentsOfDirectory(atPath: at.path)
            if entries?.isEmpty == true {
                try? FileManager.default.removeItem(at: at)
                at = at.deletingLastPathComponent()
                pruning = at.path != base.path
            } else {
                pruning = false
            }
        }
    }
}
