import Foundation
import LLM

@MainActor @Observable public final class ConversationStore {

    public static let shared = ConversationStore(root: defaultRoot)

    public struct Round: Codable, Sendable {
        public let emitted: String
        public let label: String
        public let symbol: String
        public let args: String
        public let result: String?

        public init(emitted: String, label: String, symbol: String,
                    args: String, result: String?) {
            self.emitted = emitted
            self.label = label
            self.symbol = symbol
            self.args = args
            self.result = result
        }
    }

    public struct Msg: Codable, Sendable {
        public let fromUser: Bool
        public let text: String
        public let reasoning: String
        public let rounds: [Round]
        public let images: [Data]
        public let loopStopped: Bool
        // A synthesized Codable throws on a missing key instead of using
        // the property default, so every added field must be Optional.
        public let clips: [String]?
        public let docs: [StoredDoc]?
        public let posters: [Data]?
        public let soft: String?
        public let prompt: String?

        public init(fromUser: Bool, text: String, reasoning: String,
                    rounds: [Round], images: [Data], loopStopped: Bool,
                    clips: [String]?, docs: [StoredDoc]?,
                    posters: [Data]?, soft: String? = nil,
                    prompt: String? = nil) {
            self.fromUser = fromUser
            self.text = text
            self.reasoning = reasoning
            self.rounds = rounds
            self.images = images
            self.loopStopped = loopStopped
            self.clips = clips
            self.docs = docs
            self.posters = posters
            self.soft = soft
            self.prompt = prompt
        }
    }

    public struct StoredDoc: Codable, Sendable {
        public let path: String
        public let bytes: Int
        public let short: Bool?
        public let total: Int?
        public let read: Int?
        public let cut: String?

        public init(path: String, bytes: Int, short: Bool?, total: Int?,
                    read: Int? = nil, cut: String? = nil) {
            self.path = path
            self.bytes = bytes
            self.short = short
            self.total = total
            self.read = read
            self.cut = cut
        }
    }

    public struct Trace: Codable, Sendable {
        public let kind: String
        public let t0: Date
        public let t1: Date
        public let ctx: Int
        public let tokens: Int
        public let summary: String
        public let text: String

        public init(kind: String, t0: Date, t1: Date, ctx: Int, tokens: Int,
                    summary: String, text: String) {
            self.kind = kind
            self.t0 = t0
            self.t1 = t1
            self.ctx = ctx
            self.tokens = tokens
            self.summary = summary
            self.text = text
        }
    }

    public struct Convo: Codable, Identifiable, Sendable {
        public let id: UUID
        public var title: String
        public let created: Date
        public var updated: Date
        public var messages: [Msg]
        public var trace: [Trace]? = nil
        public var trashedAt: Date? = nil
        public var extracted: Date? = nil
        public var followup: String? = nil

        public init(id: UUID, title: String, created: Date, updated: Date,
                    messages: [Msg], trace: [Trace]? = nil,
                    trashedAt: Date? = nil, extracted: Date? = nil,
                    followup: String? = nil) {
            self.id = id
            self.title = title
            self.created = created
            self.updated = updated
            self.messages = messages
            self.trace = trace
            self.trashedAt = trashedAt
            self.extracted = extracted
            self.followup = followup
        }
    }

    nonisolated public static let trashRetention: TimeInterval =
        30 * 24 * 3600

    public static let defaultRoot: URL = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("conversations", isDirectory: true)

    public let root: URL
    let files: ConversationFiles

    public private(set) var list: [Convo] = []
    public private(set) var trashed: [Convo] = []
    public private(set) var words: [UUID: [String: Int]] = [:]
    public private(set) var revision = 0

    private var loading: Task<Void, Never>?
    private var lastCommit: Task<Void, Never>?

    public init(root: URL) {
        self.root = root
        files = ConversationFiles(root: root)
        reload()
    }

    public func reload() {
        let files = self.files
        loading = Task { [weak self] in
            let snapshot = await files.reload()
            if let self, !Task.isCancelled {
                self.list = snapshot.list
                self.trashed = snapshot.trashed
                self.words = snapshot.words
                self.revision += 1
            }
        }
    }

    public func loaded() async {
        await loading?.value
    }

    public func settled() async {
        await loading?.value
        await lastCommit?.value
    }

    @discardableResult
    public func commit(id: UUID, title: String?, fallbackTitle: String,
                       messages: [Message], trace: [TraceEvent],
                       extracted: Date?,
                       followup: String? = nil) -> Task<Void, Never> {
        let files = self.files
        let task = Task { [weak self] in
            await self?.loaded()
            let prior = self?.list.first { convo in convo.id == id }
            let indexed = await files.commit(
                id: id, title: title ?? prior?.title ?? fallbackTitle,
                created: prior?.created,
                extracted: extracted ?? prior?.extracted,
                messages: messages, trace: trace, followup: followup)
            if let indexed { self?.upsert(indexed) }
        }
        lastCommit = task
        return task
    }

    public func save(_ convo: Convo) async {
        upsert(await files.save(convo))
    }

    public func rename(_ id: UUID, to title: String) async -> Bool {
        let renamed = await files.rename(id, to: title)
        if let renamed { upsert(renamed) }
        return renamed != nil
    }

    public func load(_ id: UUID) async -> Convo? {
        await files.load(id)
    }

    // Not `load`: a trashed conversation read through it and saved back
    // would land in `list` and undelete itself.
    public func loadTrashed(_ id: UUID) async -> Convo? {
        await files.loadTrashed(id)
    }

    func open(_ id: UUID) async -> ConversationFiles.Restored? {
        let ticket = files.opens.take()
        return await files.open(id, ticket: ticket)
    }

    public func trash(_ id: UUID) async {
        if let convo = await files.trash(id) { moved(convo) }
    }

    public func trashAll() async {
        let ids = list.map { convo in convo.id }
        for convo in await files.trashAll(ids) { moved(convo) }
    }

    private func moved(_ convo: Convo) {
        list.removeAll { existing in existing.id == convo.id }
        words[convo.id] = nil
        trashed.insert(convo, at: 0)
        revision += 1
    }

    public func restore(_ id: UUID) async {
        if let indexed = await files.restore(id) {
            trashed.removeAll { existing in existing.id == id }
            upsert(indexed)
        }
    }

    public func deleteForever(_ id: UUID) async {
        await files.deleteForever(id)
        trashed.removeAll { convo in convo.id == id }
        revision += 1
    }

    public func emptyTrash() async {
        await files.emptyTrash(trashed.map { convo in convo.id })
        trashed = []
        revision += 1
    }

    public func eraseAll() {
        try? FileManager.default.removeItem(at: root)
        list = []
        trashed = []
        words = [:]
        revision += 1
    }

    private func upsert(_ indexed: ConversationFiles.Indexed) {
        var next = list.filter { existing in existing.id != indexed.convo.id }
        next.append(indexed.convo)
        list = next.sorted { a, b in a.updated > b.updated }
        words[indexed.convo.id] = indexed.words
        revision += 1
    }

}
