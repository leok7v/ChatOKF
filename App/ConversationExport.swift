import Chat
import CoreTransferable
import Foundation
import LLM
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
        for (i, m) in convo.messages.enumerated() {
            stream.append(block(fromUser: m.fromUser, text: m.text,
                                reasoning: m.reasoning))
            stream.append(attached(m, i))
        }
        return stream.finish()
    }

    static func document(messages: [Message]) -> Markdown.Document {
        let stream = MarkdownStream()
        for m in messages {
            stream.append(block(fromUser: m.fromUser, text: m.text,
                                reasoning: m.reasoning))
        }
        return stream.finish()
    }

    private static func attachURL(_ i: Int, _ kind: String,
                                  _ j: Int) -> URL {
        URL(string: "chatokf://attachment/\(i)/\(kind)/\(j)")!
    }

    private static func attached(_ m: ConversationStore.Msg,
                                 _ i: Int) -> String {
        var out = ""
        for j in m.images.indices {
            out += "![Picture \(j + 1)](\(attachURL(i, "image", j)))\n\n"
        }
        for j in (m.posters ?? []).indices {
            out += "![Video \(j + 1)](\(attachURL(i, "video", j)))\n\n"
            out += "_Video_\n\n"
        }
        for doc in m.docs ?? [] {
            out += "_" + trace(doc) + "_\n\n"
        }
        return out
    }

    static func trace(_ doc: ConversationStore.StoredDoc) -> String {
        let name = doc.path.split(separator: "/").last.map(String.init)
            ?? doc.path
        let total = doc.total ?? doc.bytes
        var out = name + " \u{2014} " + size(total)
        if let read = doc.read, read < total {
            out = name + " \u{2014} read " + size(read) + " of "
                + size(total)
        }
        if let cut = doc.cut, !cut.isEmpty {
            out += cut == "memory" ? ", stopped: out of memory"
                                   : ", stopped by the user"
        }
        return out
    }

    private static func size(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes),
                                  countStyle: .file)
    }

    static func blobs(_ convo: ConversationStore.Convo) -> [URL: Data] {
        var out: [URL: Data] = [:]
        for (i, m) in convo.messages.enumerated() {
            for (j, data) in m.images.enumerated() {
                out[attachURL(i, "image", j)] = data
            }
            for (j, data) in (m.posters ?? []).enumerated() {
                out[attachURL(i, "video", j)] = data
            }
        }
        return out
    }

    static func pdf(_ convo: ConversationStore.Convo) async -> Data? {
        var decoded: [URL: CGImage] = [:]
        for (url, data) in blobs(convo) {
            if let cg = VisionPreprocess.image(data) { decoded[url] = cg }
        }
        return MarkdownPDF.data(document(convo), title: convo.title,
                                images: decoded)
    }

    static func document(text: String) -> Markdown.Document {
        let stream = MarkdownStream()
        stream.append(text)
        return stream.finish()
    }

    static func rendered(messages: [Message], title: String,
                         as type: UTType) async -> Data? {
        let document = document(messages: messages)
        var out: Data? = nil
        if type == .pdf {
            out = await MarkdownPDF.export(document, title: title)
        } else {
            let html = await Markdown.htmlPrefetching(document, title: title)
            out = Data(html.utf8)
        }
        return out
    }

    private static func written(_ data: Data?, folder: String,
                                name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(folder, isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        if let data, !data.isEmpty {
            try data.write(to: url, options: .atomic)
        } else {
            throw ExportFailure.render
        }
        return url
    }

    static func pdfFile(text: String, title: String) async throws -> URL {
        try written(await MarkdownPDF.export(document(text: text),
                                             title: title),
                    folder: "Answers", name: filename(title) + ".pdf")
    }

    static func pdfFile(_ id: UUID) async throws -> URL {
        let convo = await saved(id)
        var data: Data? = nil
        if let convo { data = await pdf(convo) }
        return try written(data, folder: "Conversations",
                           name: filename(convo?.title ?? "") + ".pdf")
    }

    @MainActor private static func saved(_ id: UUID) async
        -> ConversationStore.Convo? {
        let store = ConversationStore.shared
        var out = await store.load(id)
        if out == nil { out = await store.loadTrashed(id) }
        return out
    }

    static func file(messages: [Message], title: String,
                     as type: UTType) async throws -> URL {
        try written(await rendered(messages: messages, title: title,
                                   as: type),
                    folder: "Transcripts",
                    name: filename(title) + "." + (type == .pdf ? "pdf"
                                                                : "html"))
    }

}

enum ExportFailure: Error { case render }

@MainActor enum MarkdownCopy {

    static func put(messages: [Message], title: String) async {
        await put(title: title) {
            ConversationExport.document(messages: messages)
        }
    }

    static func put(text: String, title: String) async {
        await put(title: title) { ConversationExport.document(text: text) }
    }

    private static func put(title: String,
                            _ make: @escaping @Sendable ()
                                -> Markdown.Document) async {
        let rendered = await Task.detached {
            let document = make()
            return (plain: Markdown.plainText(document),
                    html: Markdown.html(document, title: title))
        }.value
        setClipboard(rendered.plain, html: rendered.html)
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

struct ConversationPDF: Transferable {

    let id: UUID
    let title: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .pdf) { item in
            SentTransferredFile(try await ConversationExport
                .pdfFile(item.id))
        }
        .suggestedFileName { item in
            ConversationExport.filename(item.title) + ".pdf"
        }
    }

}

struct TranscriptPDF: Transferable {

    let messages: [Message]
    let title: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .pdf) { item in
            SentTransferredFile(try await ConversationExport.file(
                messages: item.messages, title: item.title, as: .pdf))
        }
        .suggestedFileName { item in
            ConversationExport.filename(item.title) + ".pdf"
        }
    }

}

struct TranscriptHTML: Transferable {

    let messages: [Message]
    let title: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .html) { item in
            SentTransferredFile(try await ConversationExport.file(
                messages: item.messages, title: item.title, as: .html))
        }
        .suggestedFileName { item in
            ConversationExport.filename(item.title) + ".html"
        }
    }

}
