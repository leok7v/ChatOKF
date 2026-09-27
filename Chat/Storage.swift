import Foundation
import LLM

public struct KeptSize: Sendable, Identifiable, Equatable {
    public let id: UUID
    public let state: Int
    public let soft: Int
    public let attachments: Int

    public var total: Int { state + soft + attachments }
}

public struct StorageReport: Sendable, Equatable {
    public var kept: [KeptSize] = []
    public var trash = 0

    public init(kept: [KeptSize] = [], trash: Int = 0) {
        self.kept = kept
        self.trash = trash
    }

    public var total: Int {
        kept.reduce(0) { sum, size in sum + size.total }
    }

    public var anything: Bool { total + trash > 0 }

    public func size(of id: UUID) -> Int {
        kept.first { size in size.id == id }?.total ?? 0
    }
}

struct StorageCitations: Sendable {
    let id: UUID
    let docs: [String]
    let soft: [String]
}

extension Session {

    public func refreshStorage() {
        let store = ConversationStore.shared
        let live = store.list.map { convo in Session.citations(convo) }
        let trashed = store.trashed.map { convo in Session.citations(convo) }
        let root = store.root
        storageAsked += 1
        let serial = storageAsked
        Task { [weak self] in
            let report = await Task.detached {
                Session.measured(live: live, trashed: trashed, root: root)
            }.value
            if let self, serial == self.storageAsked { self.storage = report }
        }
    }

    static func citations(_ convo: ConversationStore.Convo)
        -> StorageCitations {
        var docs: [String] = []
        var soft: [String] = []
        for m in convo.messages {
            for doc in m.docs ?? [] {
                if let name = ConversationFiles.keptDocName(
                    ConversationFiles.restoredURL(doc.path)) {
                    docs.append(name)
                }
            }
            if let name = m.soft { soft.append(name) }
        }
        return StorageCitations(id: convo.id, docs: docs, soft: soft)
    }

    nonisolated static func measured(live: [StorageCitations],
                                     trashed: [StorageCitations],
                                     root: URL) -> StorageReport {
        var states: [UUID: Int] = [:]
        for url in Session.parkedURLs() {
            let head = url.lastPathComponent.split(separator: ".").first
            if let id = head.flatMap({ text in UUID(uuidString: String(text)) }) {
                states[id, default: 0] += ChatSession.allocated(url)
            }
        }
        var kept: [KeptSize] = []
        for convo in live {
            let size = KeptSize(id: convo.id, state: states[convo.id] ?? 0,
                                soft: Session.softBytes(convo.soft),
                                attachments: Session.docBytes(convo.docs))
            if size.total > 0 { kept.append(size) }
        }
        var trash = 0
        let trashDir = root.appendingPathComponent("trash", isDirectory: true)
        for convo in trashed {
            let json = trashDir.appendingPathComponent(
                convo.id.uuidString + ".json")
            trash += Session.fileBytes(json) + Session.docBytes(convo.docs)
                + Session.softBytes(convo.soft) + (states[convo.id] ?? 0)
        }
        return StorageReport(kept: kept, trash: trash)
    }

    nonisolated private static func docBytes(_ names: [String]) -> Int {
        var seen: Set<String> = []
        var out = 0
        for name in names where seen.insert(name).inserted {
            out += Session.treeBytes(
                Session.attachments.appendingPathComponent(name))
        }
        return out
    }

    nonisolated private static func softBytes(_ names: [String]) -> Int {
        names.reduce(0) { sum, name in
            sum + Session.fileBytes(SoftFile.url(name))
        }
    }

    nonisolated static func treeBytes(_ url: URL) -> Int {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey,
                                      .isRegularFileKey]
        var out = 0
        if let walk = fm.enumerator(at: url, includingPropertiesForKeys: keys,
                                    options: [.skipsHiddenFiles]) {
            for case let file as URL in walk {
                let values = try? file.resourceValues(forKeys: Set(keys))
                if values?.isRegularFile == true {
                    out += values?.totalFileAllocatedSize ?? 0
                }
            }
        }
        return out
    }

    nonisolated private static func fileBytes(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?
            .totalFileAllocatedSize ?? 0
    }

}
