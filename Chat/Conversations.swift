import Foundation
import LLM

extension Session {

    nonisolated public static let attachments: URL = {
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
                              currentConversationId: UUID?, readOnly: Bool,
                              extracted: Date? = nil,
                              followup: String? = nil)
        -> UUID? {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let worth = !readOnly && messages.count >= 2 && chars > 200
        var result = currentConversationId
        if worth {
            let id = currentConversationId ?? UUID()
            ConversationStore.shared.commit(
                id: id, title: generatedTitle, fallbackTitle: fallbackTitle,
                messages: messages, trace: traceEvents, extracted: extracted,
                followup: followup)
            result = id
        }
        return result
    }

    public struct Opened: Sendable {
        public let messages: [Message]
        public let traceEvents: [TraceEvent]
        public let followup: String
    }

    public func openConversation(_ id: UUID) async -> Opened? {
        var result: Opened? = nil
        if let restored = await ConversationStore.shared.open(id) {
            result = Opened(messages: restored.messages,
                            traceEvents: restored.trace,
                            followup: restored.followup)
        }
        return result
    }

    public func sweepAttachments(liveMessages: [Message],
                                 liveDocURLs: [URL]) async {
        var cited = Set<String>()
        var soft = Set<String>()
        let store = ConversationStore.shared
        for convo in store.list + store.trashed {
            for m in convo.messages {
                for doc in m.docs ?? [] {
                    if let name = ConversationFiles.keptDocName(
                        ConversationFiles.restoredURL(doc.path)) {
                        cited.insert(name)
                    }
                }
                if let name = m.soft { soft.insert(name) }
            }
        }
        for m in liveMessages {
            for doc in m.docs {
                if let name = ConversationFiles.keptDocName(doc.url) {
                    cited.insert(name)
                }
            }
            if let name = m.soft?.lastPathComponent { soft.insert(name) }
        }
        for url in liveDocURLs {
            if let name = ConversationFiles.keptDocName(url) {
                cited.insert(name)
            }
        }
        await store.files.sweep(Session.attachments, keeping: cited)
        await store.files.sweep(SoftFile.dir, keeping: soft)
    }

}
