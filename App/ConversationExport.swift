import Chat
import CoreTransferable
import Foundation
import MD
import SwiftUI
import UniformTypeIdentifiers

struct ExportFile: FileDocument {

    static let readableContentTypes: [UTType] = []
    static let writableContentTypes: [UTType] = [.pdf, .html]
    let data: Data

    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws { data = Data() }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }

}

enum ConversationExport {

    static func filename(_ s: String) -> String {
        let base = s.isEmpty ? "Conversation" : s
        return base.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
    }

    static func document(_ convo: ConversationStore.Convo)
        -> Markdown.Document {
        let stream = MarkdownStream()
        for m in convo.messages {
            let who = m.fromUser ? "**You**\n\n" : "**ChatOKF**\n\n"
            stream.append(who + m.text + "\n\n")
        }
        return stream.finish()
    }

    static func pdf(_ convo: ConversationStore.Convo) async -> Data? {
        await MarkdownPDF.export(document(convo), title: convo.title)
    }

    // A real file on disk: a share EXTENSION runs out of process and reads
    // the attachment by URL; handed raw bytes, Mail and Gmail attach nothing.
    static func pdfFile(_ convo: ConversationStore.Convo) async throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("Conversations", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(filename(convo.title) + ".pdf")
        let data = await pdf(convo)
        // Never a zero-byte stand-in: an empty PDF is one some receivers
        // accept and show as a broken attachment.
        if let data, !data.isEmpty {
            try data.write(to: url, options: .atomic)
        } else {
            throw ExportFailure.render
        }
        return url
    }

}

enum ExportFailure: Error { case render }

// The exporting closure is async and the share sheet calls it only once a
// destination is chosen, so nothing is rendered until something asks.
struct ConversationPDF: Transferable {

    let convo: ConversationStore.Convo

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .pdf) { item in
            SentTransferredFile(try await ConversationExport
                .pdfFile(item.convo))
        }
        .suggestedFileName { item in
            ConversationExport.filename(item.convo.title) + ".pdf"
        }
    }

}
