import Chat
import SwiftUI
import UniformTypeIdentifiers

enum AppTheme: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var scheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
    var icon: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light: return "sun.max"
        case .dark: return "moon"
        }
    }
}

struct Sidebar: View {

    let model: ChatModel
    @Binding var theme: AppTheme
    let onClose: () -> Void
    let onOpen: (UUID) -> Void
    let onNewChat: () -> Void
    let onSettings: () -> Void
    let onRename: (ConversationStore.Convo) -> Void

    @State private var armedDelete: UUID?
    @State private var disarmTask: Task<Void, Never>?
    @State private var query = ""
    @State private var showingTrash = false
    @State private var armedEmpty = false
    @State private var exportFile: ExportFile?
    @State private var showExporter = false
    @State private var exportName = "Conversation"

    private var hasHistory: Bool { !ConversationStore.shared.list.isEmpty }
    private var trash: [ConversationStore.Convo] {
        ConversationStore.shared.trashed
    }

    var body: some View {
        let items = shown
        return VStack(spacing: 0) {
            closeRow
            newChatRow
            if showingTrash {
                trashHeader
                Divider()
                if trash.isEmpty { emptyTrash } else { history(trash) }
            } else if hasHistory {
                searchField
                Divider()
                if items.isEmpty {
                    noMatches
                } else {
                    history(items)
                }
            } else {
                emptyState
            }
            Divider()
            trashRow
            footer
        }
        .fileExporter(isPresented: $showExporter, document: exportFile,
                      contentType: .pdf,
                      defaultFilename: exportName) { _ in }
    }

    private var shown: [ConversationStore.Convo] {
        let store = ConversationStore.shared
        return ConversationSearch.active(query)
            ? ConversationSearch.rank(store.list, store.words, query)
            : store.list
    }

    // The trash's own list replaces the history rather than sitting under
    // it, so the date sections and search field need not say which they filter.
    private var trashHeader: some View {
        HStack(spacing: 8) {
            Button(action: toggleTrash) {
                Label("Trash", systemImage: "chevron.left")
                    .appFont(.callout)
            }
            .buttonStyle(.plain)
            Spacer()
            emptyButton
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .animation(.easeInOut(duration: 0.15), value: armedEmpty)
    }

    @ViewBuilder
    private var emptyButton: some View {
        if !trash.isEmpty {
            if armedEmpty {
                Button { model.emptyTrash(); armedEmpty = false } label: {
                    capsuleLabel("Empty")
                }
                .buttonStyle(.plain)
                .help("Click to empty the trash; this cannot be undone")
            } else {
                Button { armedEmpty = true; scheduleDisarmEmpty() } label: {
                    Image(systemName: "trash").foregroundStyle(.tint)
                }
                .buttonStyle(.plain)
                .help("Empty the trash")
            }
        }
    }

    // Shaped after the system's own swipe action, which is drawn by the
    // list and cannot be restyled to match this one.
    private func capsuleLabel(_ text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "trash")
            Text(text)
        }
        .appFont(.caption)
        .fontWeight(.semibold)
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.red, in: Capsule())
        .fixedSize()
    }

    @ViewBuilder
    private var trashRow: some View {
        if !trash.isEmpty, !showingTrash {
            HStack {
                Button(action: toggleTrash) {
                    Text("Trash")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                emptyButton
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .animation(.easeInOut(duration: 0.15), value: armedEmpty)
        }
    }

    private func toggleTrash() {
        showingTrash.toggle()
        armedEmpty = false
        armedDelete = nil
    }

    private var emptyTrash: some View {
        VStack(spacing: 8) {
            Image(systemName: "trash")
                .appFont(.largeTitle)
                .foregroundStyle(.tertiary)
            Text("Trash is empty")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            Text("Deleted chats are kept for 30 days.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .appFont(.largeTitle)
                .foregroundStyle(.tertiary)
            Text("No conversations yet")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            Text("Your chats will appear here.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    @ViewBuilder
    private var closeRow: some View {
        if isOS {
            HStack {
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
        } else {
            Color.clear.frame(height: 12)
        }
    }

    private var newChatRow: some View {
        Button(action: onNewChat) {
            Label("New Chat", systemImage: "square.and.pencil")
                .foregroundStyle(.tint)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model.busy)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search", text: $query)
                .textFieldStyle(.plain)
            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var noMatches: some View {
        VStack(spacing: 6) {
            Text("No matches").foregroundStyle(.secondary)
            Text("for \u{201C}\(query)\u{201D}")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private var rowInsets: EdgeInsets? {
        isOS ? EdgeInsets(top: 2, leading: 8, bottom: 2, trailing: 8) : nil
    }

    private func history(_ items: [ConversationStore.Convo]) -> some View {
        List {
            if ConversationSearch.active(query) || showingTrash {
                ForEach(items) { convo in
                    historyRow(convo)
                        .listRowBackground(Color.clear)
                        .listRowInsets(rowInsets)
                        .deleteDisabled(model.busy)
                }
                .onDelete { offsets in delete(offsets, in: items) }
            } else {
                ForEach(Sidebar.groups(items)) { group in
                    Section(group.title) {
                        ForEach(group.items) { convo in
                            historyRow(convo)
                                .listRowBackground(Color.clear)
                                .listRowInsets(rowInsets)
                                .deleteDisabled(model.busy)
                        }
                        .onDelete { offsets in
                            delete(offsets, in: group.items)
                        }
                    }
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .modifier(TightSections())
    }

    private func isCurrent(_ convo: ConversationStore.Convo) -> Bool {
        convo.id == model.currentConversationId
    }

    private func historyRow(_ convo: ConversationStore.Convo) -> some View {
        HStack(spacing: 6) {
            Button { armedDelete = nil; open(convo) } label: { row(convo) }
                .buttonStyle(.plain)
                .disabled(model.busy)
                .help(rowHelp)
            // The armed capsule confirms BOTH the trash button and the
            // context menu, so it is not macOS-only like the trash button.
            if armedDelete == convo.id {
                Button { confirmArmed(convo) } label: { capsuleLabel("Delete") }
                    .buttonStyle(.plain)
                    .disabled(model.busy)
                    .help("Click to delete; this cannot be undone")
            } else if !isOS {
                Button { requestDelete(convo) } label: {
                    Image(systemName: "trash")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .disabled(model.busy)
                .help(trashHelp)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isCurrent(convo) ? Color.accentColor.opacity(0.15)
                                     : .clear,
                    in: RoundedRectangle(cornerRadius: 7))
        .animation(.easeInOut(duration: 0.15), value: armedDelete)
        .contextMenu { menu(convo) }
    }

    @ViewBuilder
    private func menu(_ convo: ConversationStore.Convo) -> some View {
        if showingTrash {
            Button { model.restoreConversation(convo.id) } label: {
                Label("Restore", systemImage: "arrow.uturn.backward")
            }
            Divider()
            Button(role: .destructive) { requestDelete(convo) } label: {
                Label("Delete Forever", systemImage: "trash")
            }
        } else {
            Button { onRename(convo) } label: {
                Label("Rename\u{2026}", systemImage: "pencil")
            }
            ShareLink(item: ConversationPDF(convo: convo),
                      preview: SharePreview(convo.title)) {
                Label("Share PDF", systemImage: "square.and.arrow.up")
            }
            if !isOS {
                Button { save(convo) } label: {
                    Label("Save PDF\u{2026}",
                          systemImage: "square.and.arrow.down")
                }
            }
            Divider()
            Button(role: .destructive) { requestDelete(convo) } label: {
                Label("Delete", systemImage: "trash")
            }
            .disabled(model.busy)
        }
    }

    private func save(_ convo: ConversationStore.Convo) {
        var ready = convo
        if convo.id == model.currentConversationId {
            model.commitCurrent()
            ready = ConversationStore.shared.load(convo.id) ?? convo
        }
        exportName = ConversationExport.filename(ready.title)
        Task { @MainActor in
            if let data = await ConversationExport.pdf(ready) {
                exportFile = ExportFile(data: data)
                showExporter = true
            }
        }
    }

    private func open(_ convo: ConversationStore.Convo) {
        onOpen(convo.id)
    }

    private var trashHelp: String {
        let what = showingTrash ? "Delete this conversation forever"
                                : "Delete conversation"
        return model.busy ? "Available once this turn has finished" : what
    }

    private var rowHelp: String {
        model.busy ? "Available once this turn has finished"
                   : "Open this conversation"
    }

    private func requestDelete(_ convo: ConversationStore.Convo) {
        if model.confirmDeleteConversation {
            armedDelete = convo.id
            scheduleDisarm(convo.id)
        } else {
            remove(convo.id)
        }
    }

    private func confirmArmed(_ convo: ConversationStore.Convo) {
        disarmTask?.cancel()
        armedDelete = nil
        remove(convo.id)
    }

    private func remove(_ id: UUID) {
        if showingTrash {
            model.deleteForever(id)
        } else {
            model.deleteConversation(id)
        }
    }

    // Six, not four: from the context menu the row is only visible again
    // once the menu has finished dismissing.
    private static let disarmSeconds = 6.0

    private func scheduleDisarm(_ id: UUID) {
        disarmTask?.cancel()
        disarmTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(Sidebar.disarmSeconds))
            if !Task.isCancelled, armedDelete == id { armedDelete = nil }
        }
    }

    private func scheduleDisarmEmpty() {
        disarmTask?.cancel()
        disarmTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(Sidebar.disarmSeconds))
            if !Task.isCancelled { armedEmpty = false }
        }
    }

    struct Group: Identifiable {
        let id: String
        let items: [ConversationStore.Convo]
        var title: String { id }
    }

    private static func groups(_ list: [ConversationStore.Convo]) -> [Group] {
        let cal = Calendar.current
        let now = Date()
        let titles = ["Today", "Yesterday", "Previous 7 Days", "Older"]
        var buckets: [[ConversationStore.Convo]] = [[], [], [], []]
        for convo in list {
            let idx: Int
            if cal.isDateInToday(convo.updated) {
                idx = 0
            } else if cal.isDateInYesterday(convo.updated) {
                idx = 1
            } else if let days = cal.dateComponents(
                [.day], from: convo.updated, to: now).day, days < 7 {
                idx = 2
            } else {
                idx = 3
            }
            buckets[idx].append(convo)
        }
        var result: [Group] = []
        for i in buckets.indices where !buckets[i].isEmpty {
            result.append(Group(id: titles[i], items: buckets[i]))
        }
        return result
    }

    private func row(_ convo: ConversationStore.Convo) -> some View {
        let reason = ConversationSearch.active(query)
            ? ConversationSearch.reason(convo, query) : nil
        let current = isCurrent(convo)
        return VStack(alignment: .leading, spacing: 2) {
            Text(convo.title)
                .lineLimit(1)
                .fontWeight(current ? .semibold : .regular)
                .foregroundStyle(current ? Color.accentColor : .primary)
            Text(reason ?? Sidebar.when(convo.updated))
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }

    private func delete(_ offsets: IndexSet,
                        in items: [ConversationStore.Convo]) {
        let ids = offsets.map { i in items[i].id }
        for id in ids { remove(id) }
    }

    private var footer: some View {
        VStack(spacing: 18) {
            Picker("Theme", selection: $theme) {
                ForEach(AppTheme.allCases) { option in
                    Image(systemName: option.icon).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Button(action: onSettings) {
                Label("Settings", systemImage: "gearshape")
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 16)
    }

    private static func when(_ date: Date) -> String {
        let f = RelativeDateTimeFormatter()
        return f.localizedString(for: date, relativeTo: Date())
    }

}
