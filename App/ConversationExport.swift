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

    static let reasoningKey = "exportReasoning"

    static var includesReasoning: Bool {
        UserDefaults.standard.bool(forKey: reasoningKey)
    }

    static func block(fromUser: Bool, text: String,
                      reasoning: String) -> String {
        var out = fromUser ? "**You**\n\n" : "**ChatOKF**\n\n"
        if !fromUser, includesReasoning, !reasoning.isEmpty {
            out += "_Thoughts_\n\n"
            for line in reasoning.split(separator: "\n",
                                        omittingEmptySubsequences: false) {
                out += "> " + line + "\n"
            }
            out += "\n"
        }
        return out + text + "\n\n"
    }

    static func document(_ convo: ConversationStore.Convo)
        -> Markdown.Document {
        let stream = MarkdownStream()
        for m in convo.messages {
            stream.append(block(fromUser: m.fromUser, text: m.text,
                                reasoning: m.reasoning))
        }
        return stream.finish()
    }

    static func pdf(_ convo: ConversationStore.Convo) async -> Data? {
        await MarkdownPDF.export(document(convo), title: convo.title)
    }

    static func document(text: String) -> Markdown.Document {
        let stream = MarkdownStream()
        stream.append(text)
        return stream.finish()
    }

    static func pdfFile(text: String, title: String) async throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("Answers", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(filename(title) + ".pdf")
        let data = await MarkdownPDF.export(document(text: text),
                                            title: title)
        if let data, !data.isEmpty {
            try data.write(to: url, options: .atomic)
        } else {
            throw ExportFailure.render
        }
        return url
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

enum MarkdownCopy {

    static func put(_ document: Markdown.Document, title: String) {
        setClipboard(Markdown.plainText(document),
                     html: Markdown.html(document, title: title))
    }

}

struct AnswerPDF: Transferable {

    let text: String
    let title: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .pdf) { item in
            SentTransferredFile(try await ConversationExport
                .pdfFile(text: item.text, title: item.title))
        }
        .suggestedFileName { item in
            ConversationExport.filename(item.title) + ".pdf"
        }
    }

}

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
