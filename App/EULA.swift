import Chat
import SwiftUI

struct EULAView: View {

    let onAgree: () -> Void
    @State private var atBottom = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(Array(EULAView.paragraphs.enumerated()),
                            id: \.offset) { _, paragraph in
                        Text(paragraph)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(20)
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
            }
            // visibleRect, not contentOffset + containerSize: that form
            // under-counts by the top inset and leaves Accept unreachable.
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.visibleRect.maxY >= geo.contentSize.height - 24
            } action: { _, bottom in
                if bottom { atBottom = true }
            }
            Divider()
            Button(action: onAgree) {
                Text(atBottom ? "Accept" : "Scroll to the end to continue")
                    .frame(maxWidth: .infinity)
            }
            .keyboardShortcut(.defaultAction)
            .controlSize(.large)
            .disabled(!atBottom)
            .padding(16)
        }
    }

    private static let paragraphs: [AttributedString] = agreement
        .components(separatedBy: "\n\n")
        .map { part in
            part.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { part in !part.isEmpty }
        .map { part in
            (try? AttributedString(markdown: part, options: .init(
                interpretedSyntax: .inlineOnlyPreservingWhitespace)))
                ?? AttributedString(part)
        }

}

private let agreement = Texts.text("eula")
