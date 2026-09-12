import Foundation
import LLM

extension Session {

    public static let attachments: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        let dir = support.appendingPathComponent("attachments",
                                                 isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    public func commitCurrent(generatedTitle: String?, fallbackTitle: String,
                              messages: [Message], traceEvents: [TraceEvent],
                              currentConversationId: UUID?, readOnly: Bool)
        -> UUID? {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let worth = !readOnly && messages.count >= 2 && chars > 200
        var result = currentConversationId
        if worth {
            let id = currentConversationId ?? UUID()
            let prior = ConversationStore.shared.load(id)
            let now = Date()
            let convo = ConversationStore.Convo(
                id: id,
                title: generatedTitle ?? prior?.title ?? fallbackTitle,
                created: prior?.created ?? now, updated: now,
                messages: messages.map { m in Session.stored(m) },
                trace: traceEvents.map { e in Session.storedTrace(e) })
            ConversationStore.shared.save(convo)
            result = id
        }
        return result
    }

    public func openConversation(_ id: UUID)
        -> (messages: [Message], traceEvents: [TraceEvent])? {
        let store = ConversationStore.shared
        var result: (messages: [Message], traceEvents: [TraceEvent])? = nil
        if let convo = store.load(id) ?? store.loadTrashed(id) {
            result = (convo.messages.map { s in Session.restored(s) },
                      (convo.trace ?? []).map { t in Session.restoredTrace(t) })
        }
        return result
    }

    public func sweepAttachments(liveMessages: [Message], liveDocURLs: [URL]) {
        var cited = Set<String>()
        let store = ConversationStore.shared
        for convo in store.list + store.trashed {
            for doc in convo.messages.flatMap({ m in m.docs ?? [] }) {
                if let name = Session.keptDocName(
                    Session.restoredURL(doc.path)) {
                    cited.insert(name)
                }
            }
        }
        for doc in liveMessages.flatMap({ m in m.docs }) {
            if let name = Session.keptDocName(doc.url) { cited.insert(name) }
        }
        for url in liveDocURLs {
            if let name = Session.keptDocName(url) { cited.insert(name) }
        }
        let fm = FileManager.default
        let kept = (try? fm.contentsOfDirectory(
            at: Session.attachments,
            includingPropertiesForKeys: nil)) ?? []
        for entry in kept where !cited.contains(entry.lastPathComponent) {
            try? fm.removeItem(at: entry)
        }
    }

    private static let bundleNameMark = "bundle:"
    private static let storeRelativeMark = "store:"

    private static var storeRoot: String {
        Session.attachments.path + "/"
    }

    private static func storedPath(_ url: URL) -> String {
        var out = url.path
        if url.path.hasPrefix(Bundle.main.bundlePath) {
            out = bundleNameMark + url.lastPathComponent
        } else if url.path.hasPrefix(storeRoot) {
            out = storeRelativeMark +
                String(url.path.dropFirst(storeRoot.count))
        }
        return out
    }

    private static func keptDocName(_ url: URL) -> String? {
        var out: String? = nil
        if url.path.hasPrefix(storeRoot) {
            out = String(url.path.dropFirst(storeRoot.count))
                .split(separator: "/").first.map(String.init)
        }
        return out
    }

    private static func restoredURL(_ stored: String) -> URL {
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

    private static func stored(_ m: Message) -> ConversationStore.Msg {
        ConversationStore.Msg(
            fromUser: m.fromUser, text: m.text, reasoning: m.reasoning,
            rounds: m.toolRounds.map { r in
                ConversationStore.Round(
                    emitted: r.emitted, label: r.label, symbol: r.symbol,
                    args: r.args, result: r.result)
            },
            images: m.images.compactMap { cg in VisionPreprocess.jpeg(cg) },
            loopStopped: m.loopStopped,
            clips: nil,
            docs: m.docs.map { ref in
                ConversationStore.StoredDoc(path: storedPath(ref.url),
                                            bytes: ref.bytes,
                                            short: ref.short)
            },
            posters: m.posters.compactMap { cg in VisionPreprocess.jpeg(cg) })
    }

    private static func restored(_ s: ConversationStore.Msg) -> Message {
        var m = Message(fromUser: s.fromUser, text: s.text)
        m.reasoning = s.reasoning
        m.loopStopped = s.loopStopped
        m.images = s.images.compactMap { data in VisionPreprocess.image(data) }
        m.posters = (s.posters ?? []).compactMap { data in
            VisionPreprocess.image(data)
        }
        m.docs = (s.docs ?? []).map { d in
            DocRef(url: restoredURL(d.path), bytes: d.bytes,
                  short: d.short ?? false)
        }
        m.toolRounds = s.rounds.enumerated().map { pair in
            ToolRound(id: pair.offset, emitted: pair.element.emitted,
                      label: pair.element.label, symbol: pair.element.symbol,
                      args: pair.element.args, result: pair.element.result)
        }
        m.answerStream.append(s.text)
        m.answerDoc = m.answerStream.finish()
        if !s.reasoning.isEmpty {
            m.reasoningStream.append(s.reasoning)
            m.reasoningDoc = m.reasoningStream.finish()
        }
        return m
    }

    private static let traceTextKinds: Set<TraceEvent.Kind> =
        [.toolCall, .toolResult, .inject, .diag]

    private static func storedTrace(_ e: TraceEvent)
        -> ConversationStore.Trace {
        let text = traceTextKinds.contains(e.kind)
            ? String(e.text.prefix(4000)) : ""
        return ConversationStore.Trace(
            kind: e.kind.rawValue, t0: e.t0, t1: e.t1, ctx: e.ctx,
            tokens: e.tokens, summary: e.summary, text: text)
    }

    private static func restoredTrace(_ t: ConversationStore.Trace)
        -> TraceEvent {
        TraceEvent(kind: TraceEvent.Kind(rawValue: t.kind) ?? .diag,
                   t0: t.t0, t1: t.t1, ctx: t.ctx, tokens: t.tokens,
                   summary: t.summary, text: t.text)
    }

}
