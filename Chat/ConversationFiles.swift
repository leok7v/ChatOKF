import CoreGraphics
import Foundation
import LLM
import MD

actor ConversationFiles {

    struct Snapshot: Sendable {
        let list: [ConversationStore.Convo]
        let trashed: [ConversationStore.Convo]
        let words: [UUID: [String: Int]]
    }

    struct Indexed: Sendable {
        let convo: ConversationStore.Convo
        let words: [String: Int]
    }

    struct Restored: Sendable {
        let messages: [Message]
        let trace: [TraceEvent]
        let followup: String
    }

    let root: URL
    nonisolated let opens = Supersedable()
    private var jpegs: [ObjectIdentifier: (image: CGImage, data: Data)] = [:]

    init(root: URL) {
        self.root = root
    }

    private var trashDir: URL {
        root.appendingPathComponent("trash", isDirectory: true)
    }

    private func fileURL(_ id: UUID) -> URL {
        root.appendingPathComponent("\(id.uuidString).json")
    }

    private func trashURL(_ id: UUID) -> URL {
        trashDir.appendingPathComponent("\(id.uuidString).json")
    }

    private func prepared() {
        try? FileManager.default.createDirectory(
            at: trashDir, withIntermediateDirectories: true)
    }

    func reload() -> Snapshot {
        prepared()
        let now = Date()
        var trashed: [ConversationStore.Convo] = []
        for convo in read(trashDir) {
            let since = now.timeIntervalSince(convo.trashedAt ?? now)
            if since > ConversationStore.trashRetention {
                try? FileManager.default.removeItem(at: trashURL(convo.id))
            } else {
                trashed.append(convo)
            }
        }
        let list = read(root).sorted { a, b in a.updated > b.updated }
        var words: [UUID: [String: Int]] = [:]
        for convo in list {
            words[convo.id] = ConversationFiles.wordCounts(convo)
        }
        return Snapshot(
            list: list.map { convo in ConversationFiles.light(convo) },
            trashed: trashed.sorted { a, b in
                (a.trashedAt ?? a.updated) > (b.trashedAt ?? b.updated)
            }.map { convo in ConversationFiles.light(convo) },
            words: words)
    }

    static func light(_ convo: ConversationStore.Convo)
        -> ConversationStore.Convo {
        var out = convo
        out.trace = nil
        out.messages = convo.messages.map { m in
            ConversationStore.Msg(
                fromUser: m.fromUser, text: m.text, reasoning: "",
                rounds: [], images: [], loopStopped: m.loopStopped,
                clips: m.clips, docs: m.docs, posters: nil, soft: m.soft)
        }
        return out
    }

    private func read(_ dir: URL) -> [ConversationStore.Convo] {
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir.path)) ?? []
        var convos: [ConversationStore.Convo] = []
        for name in names where name.hasSuffix(".json") {
            if let convo = decode(dir.appendingPathComponent(name)) {
                convos.append(convo)
            }
        }
        return convos
    }

    private func decode(_ url: URL) -> ConversationStore.Convo? {
        (try? JSONDecoder().decode(ConversationStore.Convo.self,
                                   from: Data(contentsOf: url)))
    }

    private func write(_ convo: ConversationStore.Convo, to url: URL) -> Bool {
        var wrote = false
        if let data = try? JSONEncoder().encode(convo) {
            wrote = (try? data.write(to: url)) != nil
        }
        return wrote
    }

    func load(_ id: UUID) -> ConversationStore.Convo? {
        decode(fileURL(id))
    }

    func loadTrashed(_ id: UUID) -> ConversationStore.Convo? {
        decode(trashURL(id))
    }

    func save(_ convo: ConversationStore.Convo) -> Indexed {
        prepared()
        _ = write(convo, to: fileURL(convo.id))
        return Indexed(convo: ConversationFiles.light(convo),
                       words: ConversationFiles.wordCounts(convo))
    }

    func rename(_ id: UUID, to title: String) -> Indexed? {
        var out: Indexed? = nil
        if var convo = load(id) {
            convo.title = title
            out = save(convo)
        }
        return out
    }

    func commit(id: UUID, title: String, created: Date?, extracted: Date?,
                messages: [Message], trace: [TraceEvent],
                followup: String?) -> Indexed? {
        var out: Indexed? = nil
        if !FileManager.default.fileExists(atPath: trashURL(id).path) {
            let now = Date()
            let convo = ConversationStore.Convo(
                id: id, title: title, created: created ?? now, updated: now,
                messages: messages.map { m in stored(m) },
                trace: trace.map { e in ConversationFiles.storedTrace(e) },
                extracted: extracted, followup: followup)
            pruneJpegs(to: messages)
            out = save(convo)
        }
        return out
    }

    func open(_ id: UUID, ticket: Int) -> Restored? {
        var out: Restored? = nil
        if opens.current(ticket), let convo = load(id) ?? loadTrashed(id) {
            out = Restored(
                messages: convo.messages.map { s in restored(s) },
                trace: (convo.trace ?? []).map { t in
                    ConversationFiles.restoredTrace(t)
                },
                followup: convo.followup ?? "")
        }
        return out
    }

    func trash(_ id: UUID) -> ConversationStore.Convo? {
        var out: ConversationStore.Convo? = nil
        if var convo = load(id) {
            convo.trashedAt = Date()
            prepared()
            if write(convo, to: trashURL(id)) {
                try? FileManager.default.removeItem(at: fileURL(id))
                out = ConversationFiles.light(convo)
            }
        }
        return out
    }

    func trashAll(_ ids: [UUID]) -> [ConversationStore.Convo] {
        ids.compactMap { id in trash(id) }
    }

    func restore(_ id: UUID) -> Indexed? {
        var out: Indexed? = nil
        if var convo = loadTrashed(id) {
            convo.trashedAt = nil
            if write(convo, to: fileURL(id)) {
                try? FileManager.default.removeItem(at: trashURL(id))
                out = Indexed(convo: ConversationFiles.light(convo),
                              words: ConversationFiles.wordCounts(convo))
            }
        }
        return out
    }

    func deleteForever(_ id: UUID) {
        try? FileManager.default.removeItem(at: trashURL(id))
    }

    func emptyTrash(_ ids: [UUID]) {
        for id in ids { deleteForever(id) }
    }

    func sweep(_ dir: URL, keeping cited: Set<String>) {
        let fm = FileManager.default
        let kept = (try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        for entry in kept where !cited.contains(entry.lastPathComponent) {
            try? fm.removeItem(at: entry)
        }
    }

    private static let titleWeight = 5

    static func wordCounts(_ convo: ConversationStore.Convo) -> [String: Int] {
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

    private func jpeg(_ cg: CGImage) -> Data? {
        let key = ObjectIdentifier(cg)
        var out = jpegs[key]?.data
        if out == nil, let made = VisionPreprocess.jpeg(cg) {
            jpegs[key] = (cg, made)
            out = made
        }
        return out
    }

    private func pruneJpegs(to messages: [Message]) {
        var live = Set<ObjectIdentifier>()
        for m in messages {
            for cg in m.images + m.posters { live.insert(ObjectIdentifier(cg)) }
        }
        jpegs = jpegs.filter { entry in live.contains(entry.key) }
    }

    private func decoded(_ data: Data) -> CGImage? {
        let cg = VisionPreprocess.image(data)
        if let cg { jpegs[ObjectIdentifier(cg)] = (cg, data) }
        return cg
    }

    private static let bundleNameMark = "bundle:"
    private static let storeRelativeMark = "store:"

    private static var storeRoot: String {
        Session.attachments.path + "/"
    }

    static func storedPath(_ url: URL) -> String {
        var out = url.path
        if url.path.hasPrefix(Bundle.main.bundlePath) {
            out = bundleNameMark + url.lastPathComponent
        } else if url.path.hasPrefix(storeRoot) {
            out = storeRelativeMark +
                String(url.path.dropFirst(storeRoot.count))
        }
        return out
    }

    static func keptDocName(_ url: URL) -> String? {
        var out: String? = nil
        if url.path.hasPrefix(storeRoot) {
            out = String(url.path.dropFirst(storeRoot.count))
                .split(separator: "/").first.map(String.init)
        }
        return out
    }

    static func restoredURL(_ stored: String) -> URL {
        var out = URL(fileURLWithPath: stored)
        if stored.hasPrefix(bundleNameMark) {
            let name =
                String(stored.dropFirst(bundleNameMark.count)) as NSString
            out = Bundle.main.url(forResource: name.deletingPathExtension,
                                  withExtension: name.pathExtension)
                ?? URL(fileURLWithPath: name as String)
        } else if stored.hasPrefix(storeRelativeMark) {
            out = URL(fileURLWithPath: storeRoot +
                String(stored.dropFirst(storeRelativeMark.count)))
        }
        return out
    }

    private func stored(_ m: Message) -> ConversationStore.Msg {
        ConversationStore.Msg(
            fromUser: m.fromUser, text: m.text, reasoning: m.reasoning,
            rounds: m.toolRounds.map { r in
                ConversationStore.Round(
                    emitted: r.emitted, label: r.label, symbol: r.symbol,
                    args: r.args, result: r.result)
            },
            images: m.images.compactMap { cg in jpeg(cg) },
            loopStopped: m.loopStopped,
            clips: nil,
            docs: m.docs.map { ref in
                ConversationStore.StoredDoc(
                    path: ConversationFiles.storedPath(ref.url),
                    bytes: ref.bytes, short: ref.short, total: ref.total,
                    read: ref.read, cut: ref.cut)
            },
            posters: m.posters.compactMap { cg in jpeg(cg) },
            soft: m.soft?.lastPathComponent,
            prompt: m.prompt.isEmpty || m.prompt == m.text ? nil : m.prompt)
    }

    private func restored(_ s: ConversationStore.Msg) -> Message {
        var m = Message(fromUser: s.fromUser, text: s.text)
        m.reasoning = s.reasoning
        m.loopStopped = s.loopStopped
        m.prompt = s.prompt ?? ""
        m.soft = s.soft.map { name in SoftFile.url(name) }
        m.images = s.images.compactMap { data in decoded(data) }
        m.posters = (s.posters ?? []).compactMap { data in decoded(data) }
        m.docs = (s.docs ?? []).map { d in
            DocRef(url: ConversationFiles.restoredURL(d.path), bytes: d.bytes,
                   short: d.short ?? false, total: d.total ?? d.bytes,
                   read: d.read, cut: d.cut ?? "")
        }
        m.toolRounds = s.rounds.enumerated().map { pair in
            ToolRound(id: pair.offset, emitted: pair.element.emitted,
                      label: pair.element.label, symbol: pair.element.symbol,
                      args: pair.element.args, result: pair.element.result)
        }
        let answer = MarkdownStream()
        answer.append(s.text)
        m.answerDoc = answer.finish()
        if !s.reasoning.isEmpty {
            let reasoning = MarkdownStream()
            reasoning.append(s.reasoning)
            m.reasoningDoc = reasoning.finish()
        }
        return m
    }

    private static let traceTextKinds: Set<TraceEvent.Kind> =
        [.toolCall, .toolResult, .inject, .diag]

    static func storedTrace(_ e: TraceEvent) -> ConversationStore.Trace {
        let text = traceTextKinds.contains(e.kind)
            ? String(e.text.prefix(4000)) : ""
        return ConversationStore.Trace(
            kind: e.kind.rawValue, t0: e.t0, t1: e.t1, ctx: e.ctx,
            tokens: e.tokens, summary: e.summary, text: text)
    }

    static func restoredTrace(_ t: ConversationStore.Trace) -> TraceEvent {
        TraceEvent(kind: TraceEvent.Kind(rawValue: t.kind) ?? .diag,
                   t0: t.t0, t1: t.t1, ctx: t.ctx, tokens: t.tokens,
                   summary: t.summary, text: t.text)
    }
}
