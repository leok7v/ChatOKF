import MD
import SwiftUI
import UniformTypeIdentifiers

struct AnswerActions: View {

    let text: String
    let title: String

    @State private var copied = false
    @State private var copiedReset: Task<Void, Never>?
    @State private var exportFile: ExportFile?
    @State private var exporting = false
    @State private var rendering = false
    @ScaledMetric(relativeTo: .body) private var slot: CGFloat = 26

    var body: some View {
        HStack(spacing: isOS ? slot : 12) {
            Button(action: copy) {
                glyph(copied ? "checkmark" : "doc.on.doc")
            }
            .help("Copy this answer")
            ShareLink(item: AnswerPDF(text: text, title: title),
                      preview: SharePreview(title)) {
                glyph("square.and.arrow.up")
            }
            .help("Share this answer as a PDF")
            if !isOS {
                Button(action: save) {
                    glyph(rendering ? "hourglass" : "square.and.arrow.down")
                }
                .disabled(rendering)
                .help("Save this answer as a PDF")
            }
        }
        .buttonStyle(.plain)
        .appFont(.caption)
        .foregroundStyle(.secondary)
        .padding(.leading, 2)
        .fileExporter(isPresented: $exporting, document: exportFile,
                      contentType: .pdf,
                      defaultFilename: ConversationExport.filename(title)) {
            _ in
        }
    }

    private func glyph(_ symbol: String) -> some View {
        Image(systemName: symbol)
            .frame(width: slot, height: slot)
            .contentShape(Rectangle())
    }

    private func copy() {
        MarkdownCopy.put(ConversationExport.document(text: text),
                         title: title)
        copied = true
        copiedReset?.cancel()
        copiedReset = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.2))
            if !Task.isCancelled { copied = false }
        }
    }

    private func save() {
        rendering = true
        Task { @MainActor in
            let data = await MarkdownPDF.export(
                ConversationExport.document(text: text), title: title)
            rendering = false
            if let data, !data.isEmpty {
                exportFile = ExportFile(data: data)
                exporting = true
            }
        }
    }

}
