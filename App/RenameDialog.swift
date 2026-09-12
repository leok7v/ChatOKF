import SwiftUI

struct RenameDialog: View {

    let title: String
    let onCommit: (String) -> Void
    let onCancel: () -> Void

    @State private var text = ""

    var body: some View {
        ZStack {
            Color.black.opacity(0.35)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture(perform: onCancel)
            card
        }
        .onAppear { text = title }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Conversation")
                .appFont(.headline)
            RenameField(text: $text, onSubmit: commit, onCancel: onCancel)
                .frame(height: 22)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 7))
                .overlay {
                    RoundedRectangle(cornerRadius: 7)
                        .stroke(.separator, lineWidth: 0.5)
                }
            HStack(spacing: 10) {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Rename", action: commit)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(maxWidth: 380)
        .background(.regularMaterial,
                    in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(.separator, lineWidth: 0.5)
        }
        .padding(24)
    }

    private func commit() {
        let name = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { onCancel() } else { onCommit(name) }
    }

}
