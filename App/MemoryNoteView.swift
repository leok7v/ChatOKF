import Chat
import SwiftUI

struct MemoryNoteView: View {

    let model: ChatModel
    let row: MemoryRow
    let onOpen: (UUID) -> Void
    let onClose: () -> Void

    private var note: MemoryNote? { model.memoryNote(row.id) }

    private var source: UUID? {
        var out: UUID? = nil
        if let id = row.source,
           ConversationStore.shared.list.contains(where: { c in c.id == id }) {
            out = id
        }
        return out
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    heading
                    if let note {
                        detail(note)
                    } else {
                        trashed
                    }
                    footer
                }
                .padding(20)
                .frame(maxWidth: 720, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(minWidth: isOS ? 0 : 520, minHeight: isOS ? 0 : 420)
    }

    private var header: some View {
        HStack {
            Label("Memory", systemImage: "brain")
                .appFont(.headline)
            Spacer()
            DoneButton(action: onClose)
            EscapeToClose(action: onClose)
        }
        .padding(12)
    }

    private var heading: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                if row.isPrivate {
                    Image(systemName: "lock.fill")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(row.title)
                    .appFont(.title3)
                    .bold()
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !row.description.isEmpty {
                Text(row.description)
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(row.id + "   " + Sidebar.when(row.updated))
                .appFont(.caption)
                .foregroundStyle(.tertiary)
            if !row.tags.isEmpty {
                Text(row.tags.joined(separator: ", "))
                    .appFont(.caption)
                    .foregroundStyle(.tertiary)
            }
            if row.isPrivate {
                Text("Marked private: it holds a medical, financial or "
                    + "address detail. It stays on this device, and the "
                    + "assistant still reads it.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private func detail(_ note: MemoryNote) -> some View {
        if note.retired {
            Text("Retired: the assistant no longer finds this note, and "
                + "the notes that mention it still resolve.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        Text(note.body)
            .appFont(.body)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
        linked("Mentions", note.links)
        linked("Mentioned by", note.backlinks)
    }

    @ViewBuilder
    private func linked(_ title: String, _ ids: [String]) -> some View {
        if !ids.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .appFont(.caption)
                    .bold()
                    .foregroundStyle(.secondary)
                ForEach(ids, id: \.self) { id in
                    Text(model.memoryNote(id)?.row.title ?? id)
                        .appFont(.callout)
                }
            }
        }
    }

    private var trashed: some View {
        Text("In the trash, where it stays for 30 days. The assistant "
            + "cannot read it from here. Restore it to read it again.")
            .appFont(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var footer: some View {
        if note == nil {
            Button {
                model.restoreMemory(row.id)
                onClose()
            } label: {
                Label("Restore", systemImage: "arrow.uturn.backward")
            }
        } else if let source {
            Button {
                onOpen(source)
                onClose()
            } label: {
                Label("Open the Chat", systemImage: "bubble.left")
            }
            .disabled(model.busy)
        }
    }

}
