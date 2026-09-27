import Chat
import Foundation
import MD
import SwiftUI
import UniformTypeIdentifiers

struct TranscriptActions: View {

    let messages: [Message]
    let title: String
    @Binding var renderMarkdown: Bool
    @Binding var exporting: Bool
    let onFind: () -> Void
    let onDebug: (() -> Void)?
    @State private var exportFile: ExportFile?
    @State private var exportType: UTType = .pdf
    @State private var rendering = false
    @State private var copied = false
    @State private var copiedReset: Task<Void, Never>?

    var body: some View {
        HStack(spacing: 14) {
            Button { renderMarkdown.toggle() } label: {
                Image(systemName: renderMarkdown ? "doc.plaintext"
                                                 : "doc.richtext")
            }
            .help(renderMarkdown ? "Show as plain text" : "Show as Markdown")
            Button(action: onFind) {
                Image(systemName: "magnifyingglass")
            }
            .help("Find in chat")
            Button(action: copy) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            }
            .help("Copy transcript")
            Menu {
                ShareLink(item: TranscriptPDF(messages: messages, title: title),
                          preview: SharePreview(title)) {
                    Label("Share as PDF", systemImage: "square.and.arrow.up")
                }
                ShareLink(item: TranscriptHTML(messages: messages,
                                               title: title),
                          preview: SharePreview(title)) {
                    Label("Share as HTML", systemImage: "square.and.arrow.up")
                }
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .menuIndicator(.hidden)
            .help("Share as PDF or HTML")
            if !isOS {
                Menu {
                    Button("Save as PDF") { beginSave(.pdf) }
                    Button("Save as HTML") { beginSave(.html) }
                } label: {
                    Image(systemName: rendering ? "hourglass"
                                                : "square.and.arrow.down")
                }
                .menuIndicator(.hidden)
                .disabled(rendering)
                .help("Save as PDF or HTML")
            }
            if let onDebug {
                Button(action: onDebug) {
                    Image(systemName: "ladybug")
                }
                .help("For Nerds")
            }
        }
        .fileExporter(isPresented: $exporting, document: exportFile,
                      contentType: exportType,
                      defaultFilename: exportName) { _ in }
    }

    private var exportName: String { ConversationExport.filename(title) }

    private func beginSave(_ type: UTType) {
        rendering = true
        Task { @MainActor in
            let data = await ConversationExport.rendered(
                messages: messages, title: title, as: type)
            rendering = false
            if let data, !data.isEmpty {
                exportFile = ExportFile(data: data)
                exportType = type
                exporting = true
            }
        }
    }

    private func copy() {
        copiedReset?.cancel()
        copiedReset = Task { @MainActor in
            await MarkdownCopy.put(messages: messages, title: title)
            copied = true
            try? await Task.sleep(for: .seconds(1.2))
            if !Task.isCancelled { copied = false }
        }
    }

}
