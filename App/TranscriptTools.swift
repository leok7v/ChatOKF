import Foundation
import MD
import SwiftUI
import UniformTypeIdentifiers

struct TranscriptActions: View {

    let document: Markdown.Document
    let title: String
    @Binding var renderMarkdown: Bool
    @Binding var exporting: Bool
    let onFind: () -> Void
    let onDebug: (() -> Void)?
    @State private var pdfURL: URL?
    @State private var htmlURL: URL?
    @State private var exportFile: ExportFile?
    @State private var exportType: UTType = .pdf
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
                shareRow(pdfURL, kind: "PDF")
                shareRow(htmlURL, kind: "HTML")
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .menuIndicator(.hidden)
            .help("Share as PDF or HTML")
            if !isOS {
                Menu {
                    saveRow("Save as PDF", pdfURL, .pdf)
                    saveRow("Save as HTML", htmlURL, .html)
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .menuIndicator(.hidden)
                .help("Save as PDF or HTML")
            }
            if let onDebug {
                Button(action: onDebug) {
                    Image(systemName: "ladybug")
                }
                .help("For Nerds")
            }
        }
        .task(id: document) { await prepareExports() }
        .fileExporter(isPresented: $exporting, document: exportFile,
                      contentType: exportType,
                      defaultFilename: exportName) { _ in }
    }

    private var exportName: String { ConversationExport.filename(title) }

    @ViewBuilder
    private func saveRow(_ label: String, _ url: URL?,
                         _ type: UTType) -> some View {
        Button(label) { beginSave(url, type) }
            .disabled(url == nil)
    }

    private func beginSave(_ url: URL?, _ type: UTType) {
        if let url, let data = try? Data(contentsOf: url) {
            exportFile = ExportFile(data: data)
            exportType = type
            exporting = true
        }
    }

    @ViewBuilder
    private func shareRow(_ url: URL?, kind: String) -> some View {
        if let url {
            ShareLink(item: url) {
                Label("Share as \(kind)", systemImage: "square.and.arrow.up")
            }
        } else {
            Button { } label: {
                Label("Preparing \(kind)\u{2026}",
                      systemImage: "square.and.arrow.up")
            }
            .disabled(true)
        }
    }

    private func copy() {
        setClipboard(Markdown.plainText(document))
        copied = true
        copiedReset?.cancel()
        copiedReset = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.2))
            if !Task.isCancelled { copied = false }
        }
    }

    private func prepareExports() async {
        let safe = exportName
        async let pdf = MarkdownPDF.export(document, title: title)
        async let html = Markdown.htmlPrefetching(document, title: title)
        let (pdfData, htmlText) = await (pdf, html)
        pdfURL = pdfData.flatMap { data in
            writeTemp(data, name: "\(safe).pdf")
        }
        htmlURL = writeTemp(Data(htmlText.utf8), name: "\(safe).html")
    }

    private func writeTemp(_ data: Data, name: String) -> URL? {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("TranscriptTools", isDirectory: true)
        var result: URL?
        do {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent(name)
            try data.write(to: url, options: .atomic)
            result = url
        } catch {
            result = nil
        }
        return result
    }

}
