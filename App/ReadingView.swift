import MD
import SwiftUI

struct ReadingView: View {

    let document: Markdown.Document
    let title: String
    var style: MarkdownStyle = .default
    let onClose: () -> Void

    @State private var find = MarkdownFindController()
    @State private var query = ""
    @State private var matchCount = 0
    @State private var currentMatch = 0

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            findBar
            Divider()
            MarkdownTextView(document, style: style, find: find)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var header: some View {
        HStack {
            Text(title.isEmpty ? "Transcript" : title)
                .appFont(.headline)
                .lineLimit(1)
            Spacer()
            Button("Done", action: onClose)
                .keyboardShortcut(.defaultAction)
            Button("", action: onClose)
                .keyboardShortcut(.cancelAction)
                .hidden()
        }
        .padding(12)
    }

    private var findBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Find in transcript", text: $query)
                .textFieldStyle(.plain)
                .onSubmit { currentMatch = find.findNext() }
                .task(id: query) { await runFind() }
            if matchCount > 0 {
                Text("\(currentMatch)/\(matchCount)")
                    .appFont(.caption).monospacedDigit()
                    .foregroundStyle(.secondary)
            } else if !query.isEmpty {
                Text("none")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
            Button { currentMatch = find.findPrevious() } label: {
                Image(systemName: "chevron.up")
            }
            .disabled(matchCount == 0)
            Button { currentMatch = find.findNext() } label: {
                Image(systemName: "chevron.down")
            }
            .disabled(matchCount == 0)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    private func runFind() async {
        try? await Task.sleep(for: .milliseconds(120))
        if !Task.isCancelled {
            matchCount = find.find(query)
            currentMatch = find.currentMatch
        }
    }

}
