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

enum SidebarTab: String, CaseIterable, Identifiable {
    case chats = "Chats"
    case memories = "Memories"
    var id: String { rawValue }
}

struct SidebarRow: Identifiable {
    let id: String
    let title: String
    let subtitle: String
    let detail: String
    let marked: Bool
    let current: Bool
}

struct SidebarSection: Identifiable {
    let id: String
    let rows: [SidebarRow]
    var title: String { id }
}

struct Sidebar: View {

    let model: ChatModel
    @Binding var theme: AppTheme
    let onClose: () -> Void
    let onOpen: (UUID) -> Void
    let onNewChat: () -> Void
    let onSettings: () -> Void
    let onRename: (ConversationStore.Convo) -> Void

    @State private var armedDelete: String?
    @State private var disarmTask: Task<Void, Never>?
    @State private var query = ""
    @State private var showingTrash = false
    @State private var armedEmpty = false
    @State private var chosen: SidebarTab = .chats
    @State private var openNote: MemoryRow?
    @State private var exportFile: ExportFile?
    @State private var showExporter = false
    @State private var exportName = "Conversation"

    private var tab: SidebarTab { model.memoriesOn ? chosen : .chats }

    private var searching: Bool { ConversationSearch.active(query) }

    private var flat: Bool { searching || showingTrash }

    private var hasItems: Bool {
        tab == .chats ? !ConversationStore.shared.list.isEmpty
                      : !model.memoryList.isEmpty
    }

    private var trashCount: Int {
        tab == .chats ? ConversationStore.shared.trashed.count
                      : model.memoryTrash.count
    }

    var body: some View {
        let items = sections
        return VStack(spacing: 0) {
            closeRow
            newChatRow
            tabPicker
            if showingTrash {
                trashHeader
                Divider()
                if items.isEmpty { emptyTrash } else { history(items) }
            } else if hasItems {
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
        .sheet(item: $openNote) { row in
            MemoryNoteView(model: model, row: row, onOpen: onOpen,
                           onClose: { openNote = nil })
        }
    }

    @ViewBuilder
    private var tabPicker: some View {
        if model.memoriesOn {
            Picker("Show", selection: $chosen) {
                ForEach(SidebarTab.allCases) { option in
                    Text(option.rawValue).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 16)
            .padding(.bottom, 4)
            .onChange(of: chosen) { _, _ in
                query = ""
                showingTrash = false
                armedDelete = nil
                armedEmpty = false
            }
        }
    }

    private var sections: [SidebarSection] {
        tab == .chats ? chatSections : memorySections
    }

    private var chatSections: [SidebarSection] {
        let store = ConversationStore.shared
        let items = showingTrash
            ? store.trashed
            : (searching ? ConversationSearch.rank(store.list, store.words,
                                                   query)
                         : store.list)
        var out: [SidebarSection] = []
        if flat {
            let rows = items.map { convo in chatRow(convo) }
            if !rows.isEmpty { out = [SidebarSection(id: "", rows: rows)] }
        } else {
            out = Sidebar.dated(items).map { group in
                SidebarSection(id: group.title,
                               rows: group.items.map { c in chatRow(c) })
            }
        }
        return out
    }

    private var memorySections: [SidebarSection] {
        let items = showingTrash
            ? model.memoryTrash
            : (searching ? MemorySearch.rank(model.memoryList, query)
                         : model.memoryList)
        var out: [SidebarSection] = []
        if flat {
            let rows = items.map { note in memoryRow(note) }
            if !rows.isEmpty { out = [SidebarSection(id: "", rows: rows)] }
        } else {
            out = Sidebar.byArea(items).map { group in
                SidebarSection(id: Sidebar.areaTitle(group.area),
                               rows: group.notes.map { n in memoryRow(n) })
            }
        }
        return out
    }

    private func chatRow(_ convo: ConversationStore.Convo) -> SidebarRow {
        let reason = searching
            ? ConversationSearch.reason(convo, query) : nil
        return SidebarRow(id: convo.id.uuidString, title: convo.title,
                          subtitle: reason ?? Sidebar.when(convo.updated),
                          detail: "", marked: false,
                          current: convo.id == model.currentConversationId)
    }

    private func memoryRow(_ note: MemoryRow) -> SidebarRow {
        SidebarRow(id: note.id, title: note.title,
                   subtitle: note.description,
                   detail: note.id + "   " + Sidebar.when(note.updated),
                   marked: note.isPrivate, current: false)
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
        if trashCount > 0 {
            if armedEmpty {
                Button { emptyNow(); armedEmpty = false } label: {
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

    private func emptyNow() {
        if tab == .chats {
            model.emptyTrash()
        } else {
            model.emptyMemoriesTrash()
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
        if trashCount > 0, !showingTrash {
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
            Text(tab == .chats ? "Deleted chats are kept for 30 days."
                               : "Deleted memories are kept for 30 days.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: tab == .chats
                  ? "bubble.left.and.bubble.right" : "brain")
                .appFont(.largeTitle)
                .foregroundStyle(.tertiary)
            Text(tab == .chats ? "No conversations yet"
                               : "Nothing remembered yet")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            Text(tab == .chats
                 ? "Your chats will appear here."
                 : "Notes the assistant keeps will appear here.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
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

    private func history(_ sections: [SidebarSection]) -> some View {
        let loose = sections.count == 1 && sections[0].title.isEmpty
        return List {
            if loose {
                ForEach(sections[0].rows) { row in listed(row) }
                    .onDelete { offsets in
                        delete(offsets, in: sections[0].rows)
                    }
            } else {
                ForEach(sections) { section in
                    Section(section.title) {
                        ForEach(section.rows) { row in listed(row) }
                            .onDelete { offsets in
                                delete(offsets, in: section.rows)
                            }
                    }
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .modifier(TightSections())
    }

    private func listed(_ row: SidebarRow) -> some View {
        historyRow(row)
            .listRowBackground(Color.clear)
            .listRowInsets(rowInsets)
            .deleteDisabled(blocked(row))
    }

    private func blocked(_ row: SidebarRow) -> Bool {
        tab == .memories ? model.busy : (model.busy && row.current)
    }

    private func historyRow(_ row: SidebarRow) -> some View {
        HStack(spacing: 6) {
            Button { armedDelete = nil; open(row) } label: { label(row) }
                .buttonStyle(.plain)
                .disabled(model.busy)
                .help(rowHelp)
            // The armed capsule confirms BOTH the trash button and the
            // context menu, so it is not macOS-only like the trash button.
            if armedDelete == row.id {
                Button { confirmArmed(row) } label: { capsuleLabel("Delete") }
                    .buttonStyle(.plain)
                    .disabled(blocked(row))
                    .help("Click to delete; this cannot be undone")
            } else if !isOS {
                Button { requestDelete(row) } label: {
                    Image(systemName: "trash")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .disabled(blocked(row))
                .help(trashHelp(row))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(row.current ? Color.accentColor.opacity(0.15) : .clear,
                    in: RoundedRectangle(cornerRadius: 7))
        .animation(.easeInOut(duration: 0.15), value: armedDelete)
        .contextMenu { menu(row) }
    }

    private func label(_ row: SidebarRow) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                if row.marked {
                    Image(systemName: "lock.fill")
                        .appFont(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(row.title)
                    .lineLimit(1)
                    .fontWeight(row.current ? .semibold : .regular)
                    .foregroundStyle(row.current ? Color.accentColor
                                                 : .primary)
            }
            Text(row.subtitle)
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            if !row.detail.isEmpty {
                Text(row.detail)
                    .appFont(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }

    @ViewBuilder
    private func menu(_ row: SidebarRow) -> some View {
        if tab == .chats {
            chatMenu(row)
        } else {
            memoryMenu(row)
        }
    }

    @ViewBuilder
    private func chatMenu(_ row: SidebarRow) -> some View {
        if showingTrash {
            if let convo = Sidebar.convo(row.id) {
                Button { model.restoreConversation(convo.id) } label: {
                    Label("Restore", systemImage: "arrow.uturn.backward")
                }
                Divider()
            }
            Button(role: .destructive) { requestDelete(row) } label: {
                Label("Delete Forever", systemImage: "trash")
            }
        } else {
            if let convo = Sidebar.convo(row.id) {
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
            }
            Button(role: .destructive) { requestDelete(row) } label: {
                Label("Delete", systemImage: "trash")
            }
            .disabled(blocked(row))
        }
    }

    @ViewBuilder
    private func memoryMenu(_ row: SidebarRow) -> some View {
        if showingTrash {
            Button { model.restoreMemory(row.id) } label: {
                Label("Restore", systemImage: "arrow.uturn.backward")
            }
            Divider()
            Button(role: .destructive) { requestDelete(row) } label: {
                Label("Delete Forever", systemImage: "trash")
            }
        } else {
            if let source = Sidebar.sourceChat(model, row.id) {
                Button { onOpen(source) } label: {
                    Label("Open the Chat", systemImage: "bubble.left")
                }
                .disabled(model.busy)
                Divider()
            }
            Button(role: .destructive) { requestDelete(row) } label: {
                Label("Delete", systemImage: "trash")
            }
            .disabled(blocked(row))
        }
    }

    static func convo(_ id: String) -> ConversationStore.Convo? {
        let store = ConversationStore.shared
        let uuid = UUID(uuidString: id)
        return (store.list + store.trashed).first { c in c.id == uuid }
    }

    static func sourceChat(_ model: ChatModel, _ id: String) -> UUID? {
        var out: UUID? = nil
        if let source = model.memoryList.first(where: { n in n.id == id })?
            .source,
           ConversationStore.shared.list.contains(where: { c in
               c.id == source
           }) {
            out = source
        }
        return out
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

    private func open(_ row: SidebarRow) {
        if tab == .chats {
            if let id = UUID(uuidString: row.id) { onOpen(id) }
        } else {
            openNote = (showingTrash ? model.memoryTrash : model.memoryList)
                .first { note in note.id == row.id }
        }
    }

    private func trashHelp(_ row: SidebarRow) -> String {
        let noun = tab == .chats ? "conversation" : "memory"
        let what = showingTrash ? "Delete this \(noun) forever"
                                : "Delete \(noun)"
        return blocked(row) ? "Available once this turn has finished" : what
    }

    private var rowHelp: String {
        let what = tab == .chats ? "Open this conversation" : "Read this note"
        return model.busy ? "Available once this turn has finished" : what
    }

    private func requestDelete(_ row: SidebarRow) {
        if tab == .memories || model.confirmDeleteConversation {
            armedDelete = row.id
            scheduleDisarm(row.id)
        } else {
            remove(row.id)
        }
    }

    private func confirmArmed(_ row: SidebarRow) {
        disarmTask?.cancel()
        armedDelete = nil
        remove(row.id)
    }

    private func remove(_ id: String) {
        if tab == .chats {
            if let uuid = UUID(uuidString: id) {
                if showingTrash {
                    model.deleteForever(uuid)
                } else {
                    model.deleteConversation(uuid)
                }
            }
        } else if showingTrash {
            model.deleteMemoryForever(id)
        } else {
            model.forgetMemory(id)
        }
    }

    // Six, not four: from the context menu the row is only visible again
    // once the menu has finished dismissing.
    private static let disarmSeconds = 6.0

    private func scheduleDisarm(_ id: String) {
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

    private static func dated(_ list: [ConversationStore.Convo]) -> [Group] {
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

    static func byArea(_ notes: [MemoryRow])
        -> [(area: String, notes: [MemoryRow])] {
        var members: [String: [MemoryRow]] = [:]
        for note in notes { members[note.area, default: []].append(note) }
        return members.keys.sorted().map { area in
            (area, members[area] ?? [])
        }
    }

    static func areaTitle(_ area: String) -> String {
        area == "." ? "Loose" : area.replacingOccurrences(of: "-", with: " ")
            .capitalized
    }

    private func delete(_ offsets: IndexSet, in rows: [SidebarRow]) {
        let ids = offsets.map { i in rows[i].id }
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

    static func when(_ date: Date) -> String {
        let f = RelativeDateTimeFormatter()
        return f.localizedString(for: date, relativeTo: Date())
    }

}
