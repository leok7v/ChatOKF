import Foundation

@MainActor @Observable public final class ConversationStore {

    public static let shared = ConversationStore()

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

        public init(fromUser: Bool, text: String, reasoning: String,
                    rounds: [Round], images: [Data], loopStopped: Bool,
                    clips: [String]?, docs: [StoredDoc]?,
                    posters: [Data]?) {
            self.fromUser = fromUser
            self.text = text
            self.reasoning = reasoning
            self.rounds = rounds
            self.images = images
            self.loopStopped = loopStopped
            self.clips = clips
            self.docs = docs
            self.posters = posters
        }
    }

    public struct StoredDoc: Codable, Sendable {
        public let path: String
        public let bytes: Int
        public let short: Bool?
        public let total: Int?

        public init(path: String, bytes: Int, short: Bool?, total: Int?) {
            self.path = path
            self.bytes = bytes
            self.short = short
            self.total = total
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

        public init(id: UUID, title: String, created: Date, updated: Date,
                    messages: [Msg], trace: [Trace]? = nil,
                    trashedAt: Date? = nil, extracted: Date? = nil) {
            self.id = id
            self.title = title
            self.created = created
            self.updated = updated
            self.messages = messages
            self.trace = trace
            self.trashedAt = trashedAt
            self.extracted = extracted
        }
    }

    public static let trashRetention: TimeInterval = 30 * 24 * 3600

    public private(set) var list: [Convo] = []
    public private(set) var trashed: [Convo] = []

    public private(set) var words: [UUID: [String: Int]] = [:]

    private init() {
        reload()
    }

    public func reload() {
        purgeExpired()
        list = read(conversationsDir()).sorted { a, b in a.updated > b.updated }
        trashed = read(trashDir()).sorted { a, b in
            (a.trashedAt ?? a.updated) > (b.trashedAt ?? b.updated)
        }
        words = [:]
        for convo in list { words[convo.id] = Self.wordCounts(convo) }
    }

    private func read(_ dir: URL) -> [Convo] {
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? []
        var convos: [Convo] = []
        for name in names where name.hasSuffix(".json") {
            if let convo = decodeConvo(dir.appendingPathComponent(name)) {
                convos.append(convo)
            }
        }
        return convos
    }

    public func save(_ convo: Convo) {
        if let data = try? JSONEncoder().encode(convo) {
            try? data.write(to: fileURL(convo.id))
        }
        upsert(convo)
    }

    public func trash(_ id: UUID) {
        if var convo = decodeConvo(fileURL(id)) {
            convo.trashedAt = Date()
            if let data = try? JSONEncoder().encode(convo) {
                try? data.write(to: trashURL(id))
                try? FileManager.default.removeItem(at: fileURL(id))
                list.removeAll { existing in existing.id == id }
                words[id] = nil
                trashed.insert(convo, at: 0)
            }
        }
    }

    public func trashAll() {
        for convo in list { trash(convo.id) }
    }

    public func restore(_ id: UUID) {
        if var convo = decodeConvo(trashURL(id)) {
            convo.trashedAt = nil
            if let data = try? JSONEncoder().encode(convo) {
                try? data.write(to: fileURL(id))
                try? FileManager.default.removeItem(at: trashURL(id))
                trashed.removeAll { existing in existing.id == id }
                upsert(convo)
            }
        }
    }

    public func deleteForever(_ id: UUID) {
        try? FileManager.default.removeItem(at: trashURL(id))
        trashed.removeAll { convo in convo.id == id }
    }

    public func emptyTrash() {
        for convo in trashed {
            try? FileManager.default.removeItem(at: trashURL(convo.id))
        }
        trashed = []
    }

    public func eraseAll() {
        try? FileManager.default.removeItem(at: conversationsDir())
        list = []
        trashed = []
        words = [:]
    }

    // A stamp rather than the file's mtime, which a copy or a restore from
    // backup would reset.
    private func purgeExpired() {
        let now = Date()
        for convo in read(trashDir()) {
            let since = now.timeIntervalSince(convo.trashedAt ?? now)
            if since > Self.trashRetention {
                try? FileManager.default.removeItem(at: trashURL(convo.id))
            }
        }
    }

    public func load(_ id: UUID) -> Convo? {
        decodeConvo(fileURL(id))
    }

    // Not `load`: a trashed conversation read through it and saved back
    // would land in `list` and undelete itself.
    public func loadTrashed(_ id: UUID) -> Convo? {
        decodeConvo(trashURL(id))
    }

    private func upsert(_ convo: Convo) {
        var next = list.filter { existing in existing.id != convo.id }
        next.append(convo)
        list = next.sorted { a, b in a.updated > b.updated }
        words[convo.id] = Self.wordCounts(convo)
    }

    private static let titleWeight = 5

    private static func wordCounts(_ convo: Convo) -> [String: Int] {
        var counts: [String: Int] = [:]
        add(convo.title, titleWeight, &counts)
        for m in convo.messages { add(m.text, 1, &counts) }
        return counts
    }

    private static func add(_ text: String, _ weight: Int,
                            _ counts: inout [String: Int]) {
        for token in text.lowercased().split(whereSeparator: { c in
            !c.isLetter && !c.isNumber
        }) {
            counts[String(token), default: 0] += weight
        }
    }

    private func decodeConvo(_ url: URL) -> Convo? {
        (try? JSONDecoder().decode(Convo.self, from: Data(contentsOf: url)))
    }

    private func conversationsDir() -> URL {
        let fm = FileManager.default
        let base = fm.urls(for: .applicationSupportDirectory,
                           in: .userDomainMask)[0]
            .appendingPathComponent("conversations", isDirectory: true)
        try? fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func trashDir() -> URL {
        let fm = FileManager.default
        let base = conversationsDir()
            .appendingPathComponent("trash", isDirectory: true)
        try? fm.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func fileURL(_ id: UUID) -> URL {
        conversationsDir().appendingPathComponent("\(id.uuidString).json")
    }

    private func trashURL(_ id: UUID) -> URL {
        trashDir().appendingPathComponent("\(id.uuidString).json")
    }

}
