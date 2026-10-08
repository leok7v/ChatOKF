import Chat
import Foundation
import LLM

extension ChatModel {

    func commitCurrent() {
        if transcriptDirty || currentConversationId == nil {
            currentConversationId = session.commitCurrent(
                generatedTitle: generatedTitle,
                fallbackTitle: conversationTitle(),
                messages: messages, traceEvents: traceEvents,
                currentConversationId: currentConversationId,
                readOnly: readOnly, extracted: extractedAt,
                followup: followupHint.isEmpty ? nil : followupHint)
            transcriptDirty = false
        }
    }

    func openConversation(_ id: UUID) {
        commitCurrent()
        let leaving = liveConversation
        if !locked {
            let running = replaying ? genTask : nil
            if running != nil { session.requestStop() }
            Task { @MainActor in
                if let running {
                    running.cancel()
                    _ = await running.value
                }
                if !locked, let restored = await session.openConversation(id) {
                    let plan = await session.resumePlan(id, restored.messages)
                    if case .needsModel(let name) = plan,
                       let kin = ChatModel.onDiskKin(name) {
                        resumeAfterLoad = id
                        flashNote("Switching to \(Models.display(kin))")
                        switchModel(kin)
                    } else {
                        show(restored, as: id)
                        resumeShown(id, plan, leaving)
                    }
                }
            }
        }
    }

    static func onDiskKin(_ name: String) -> String? {
        let downloaded = Models.downloaded
        return Models.every.first { other in
            downloaded.contains(other)
                && Session.towerFamily(other) == Session.towerFamily(name)
        }
    }

    private func show(_ opened: Session.Opened, as id: UUID) {
        transcriptSerial += 1
        messages = opened.messages
        traceEvents = opened.traceEvents
        currentConversationId = id
        generatedTitle = nil
        followupHint = offersFollowupHint ? opened.followup : ""
        heldSend = nil
        remembered = []
        extractedAt = nil
        readOnly = true
        savedLabel = ChatModel.savedLabel(opened.traceEvents)
        statsLabel = ""
    }

    private func resumeShown(_ id: UUID, _ plan: ResumePlan,
                             _ leaving: UUID?) {
        var live = false
        switch plan {
        case .parked, .replay: live = true
        case .needsModel(let name): offerDownload(name, for: id)
        case .readOnly(let why): noteReadOnly(why)
        }
        if leaving != nil || live {
            genSerial += 1
            let serial = genSerial
            genTask = Task { @MainActor in
                if let leaving { await session.parkCurrent(leaving) }
                if case .parked = plan { await resumeParked(id) }
                if case .replay(let turns) = plan { await replay(id, turns) }
                if genSerial == serial { genTask = nil }
            }
        }
    }

    private func noteReadOnly(_ why: String) {
        let note = "read only: " + why
        savedLabel += savedLabel.isEmpty ? note : "  " + note
        flashNote("Read only: " + why)
    }

    private func offerDownload(_ name: String, for id: UUID) {
        if Models.all.contains(name), ModelCatalog.source(name) != nil {
            resumeAfterLoad = id
            resumeAsk = name
        } else {
            noteReadOnly("needs " + Models.display(name))
        }
    }

    private func resumeParked(_ id: UUID) async {
        if session.hasParked(id), await session.resumeParked(
            id, sessionConfig(), onEvent: { [weak self] e in
                self?.recordTrace(e)
            }) {
            generatedTitle = ChatModel.storedTitle(id)
            readOnly = false
            statsLabel = savedLabel
        }
    }

    private static func storedTitle(_ id: UUID) -> String? {
        ConversationStore.shared.list.first { convo in convo.id == id }?.title
    }

    static let resumingNotice = "Resuming the conversation"

    private func replay(_ id: UUID, _ turns: [PlannedTurn]) async {
        replaying = true
        prefilling = true
        let ok = await session.replay(turns, budget: imageBudget.tokens,
                                      sessionConfig()) { [weak self] e in
            self?.recordTrace(e)
        }
        replaying = false
        prefilling = false
        if currentConversationId == id {
            if ok {
                generatedTitle = ChatModel.storedTitle(id)
                readOnly = false
                statsLabel = savedLabel
            } else if !Task.isCancelled {
                noteReadOnly("could not resume")
            }
        }
    }

    func renameConversation(_ id: UUID, to title: String) {
        let name = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty {
            Task { @MainActor in
                if await ConversationStore.shared.rename(id, to: name),
                   id == currentConversationId {
                    generatedTitle = name
                }
            }
        }
    }

    func deleteConversation(_ id: UUID) {
        closeIfShowing(id)
        Task { @MainActor in
            await ConversationStore.shared.trash(id)
            session.dropParked(id)
            await session.memories.trash(ConversationNote.id(id))
            await sweepAttachments()
        }
    }

    func restoreConversation(_ id: UUID) {
        Task { @MainActor in
            await ConversationStore.shared.restore(id)
            await session.memories.restore(ConversationNote.id(id))
            session.refreshStorage()
        }
    }

    func deleteForever(_ id: UUID) {
        closeIfShowing(id)
        Task { @MainActor in
            await ConversationStore.shared.deleteForever(id)
            session.dropParked(id)
            await session.memories.deleteForever(ConversationNote.id(id))
            await sweepAttachments()
        }
    }

    func emptyTrash() {
        let gone = ConversationStore.shared.trashed.map { convo in convo.id }
        for id in gone { closeIfShowing(id) }
        Task { @MainActor in
            await ConversationStore.shared.emptyTrash()
            for id in gone {
                session.dropParked(id)
                await session.memories.deleteForever(ConversationNote.id(id))
            }
            await sweepAttachments()
        }
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
        currentConversationId = nil
        if readOnly { newChat() }
        Task { @MainActor in
            await ConversationStore.shared.trashAll()
            for id in gone {
                session.dropParked(id)
                await session.memories.trash(ConversationNote.id(id))
            }
            await sweepAttachments()
        }
    }

    func sweepAttachments() async {
        await session.sweepAttachments(
            liveMessages: messages,
            liveDocURLs: attachedDocs.compactMap { d in d.url })
        session.refreshStorage()
    }

    struct TitleKey: Equatable {
        let conversation: UUID?
        let count: Int
        let generated: String?
        let turn: Int
    }

    var transcriptTitle: String { conversationTitle() }

    func conversationTitle() -> String {
        let key = TitleKey(conversation: currentConversationId,
                           count: messages.count, generated: generatedTitle,
                           turn: settledTurns)
        var title = titleCache?.key == key ? titleCache?.title ?? "" : ""
        if title.isEmpty {
            title = generatedTitle ?? ""
            if title.isEmpty {
                title = TopicTitle.from(messages.map { m in m.text })
            }
            if title.isEmpty { title = ChatModel.timestampTitle() }
            titleCache = (key, title)
        }
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
