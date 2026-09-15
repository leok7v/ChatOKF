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
        let leaving = liveConversation
        if !busy, let restored = session.openConversation(id) {
            messages = restored.messages
            traceEvents = restored.traceEvents
            currentConversationId = id
            generatedTitle = nil
            followupHint = ""
            heldSend = nil
            remembered = []
            extractedAt = nil
            readOnly = true
            statsLabel = ""
            if leaving != nil || session.hasParked(id) {
                genTask = Task { @MainActor in
                    if let leaving { await session.parkCurrent(leaving) }
                    await resumeParked(id)
                    genTask = nil
                }
            }
        }
    }

    private func resumeParked(_ id: UUID) async {
        if session.hasParked(id), await session.resumeParked(
            id, sessionConfig(), onEvent: { [weak self] e in
                self?.recordTrace(e)
            }) {
            generatedTitle = ConversationStore.shared.load(id)?.title
            readOnly = false
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
        session.dropParked(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func restoreConversation(_ id: UUID) {
        ConversationStore.shared.restore(id)
    }

    func deleteForever(_ id: UUID) {
        ConversationStore.shared.deleteForever(id)
        session.dropParked(id)
        closeIfShowing(id)
        sweepAttachments()
    }

    func emptyTrash() {
        let gone = ConversationStore.shared.trashed.map { convo in convo.id }
        ConversationStore.shared.emptyTrash()
        for id in gone {
            session.dropParked(id)
            closeIfShowing(id)
        }
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
        let gone = ConversationStore.shared.list.map { convo in convo.id }
        ConversationStore.shared.trashAll()
        for id in gone { session.dropParked(id) }
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
            stream.append(ConversationExport.block(
                fromUser: m.fromUser, text: m.text, reasoning: m.reasoning))
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
