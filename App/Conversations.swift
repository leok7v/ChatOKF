import Chat
import Foundation
import LLM
import MD

extension ChatModel {

    func commitCurrent() {
        currentConversationId = session.commitCurrent(
            generatedTitle: generatedTitle, fallbackTitle: conversationTitle(),
            messages: messages, traceEvents: traceEvents,
            currentConversationId: currentConversationId, readOnly: readOnly,
            extracted: extractedAt)
    }

    func openConversation(_ id: UUID) {
        commitCurrent()
        if !busy, let restored = session.openConversation(id) {
            messages = restored.messages
            traceEvents = restored.traceEvents
            currentConversationId = id
            generatedTitle = nil
            followupHint = ""
            readOnly = true
            statsLabel = ""
        }
    }

    func renameConversation(_ id: UUID, to title: String) {
        let name = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty, var convo = ConversationStore.shared.load(id) {
            convo.title = name
            ConversationStore.shared.save(convo)
            if id == currentConversationId { generatedTitle = name }
        }
    }

    func deleteConversation(_ id: UUID) {
        ConversationStore.shared.trash(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func restoreConversation(_ id: UUID) {
        ConversationStore.shared.restore(id)
    }

    func deleteForever(_ id: UUID) {
        ConversationStore.shared.deleteForever(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func emptyTrash() {
        let gone = ConversationStore.shared.trashed.map { convo in convo.id }
        ConversationStore.shared.emptyTrash()
        for id in gone { closeIfShowing(id) }
        sweepAttachments()
    }

    // The trash can be read before it is emptied, so a destroyed
    // conversation may be the one on screen.
    private func closeIfShowing(_ id: UUID) {
        if id == currentConversationId {
            currentConversationId = nil
            generatedTitle = nil
            messages = []
            traceEvents = []
            newChat()
        }
    }

    func clearAllConversations() {
        ConversationStore.shared.trashAll()
        currentConversationId = nil
        if readOnly { newChat() }
        sweepAttachments()
    }

    func sweepAttachments() {
        session.sweepAttachments(
            liveMessages: messages,
            liveDocURLs: attachedDocs.compactMap { d in d.url })
    }

    var transcriptDocument: Markdown.Document {
        let stream = MarkdownStream()
        for m in messages {
            let who = m.fromUser ? "**You**\n\n" : "**ChatOKF**\n\n"
            stream.append(who + m.text + "\n\n")
        }
        return stream.finish()
    }

    var transcriptTitle: String { conversationTitle() }

    func conversationTitle() -> String {
        var title = generatedTitle ?? ""
        if title.isEmpty {
            title = TopicTitle.from(messages.map { m in m.text })
        }
        if title.isEmpty { title = ChatModel.timestampTitle() }
        return title
    }

    private static func timestampTitle() -> String {
        let f = DateFormatter()
        f.locale = Locale.current
        f.dateStyle = .none
        f.timeStyle = .short
        return f.string(from: Date())
    }

}
