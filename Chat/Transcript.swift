import CoreGraphics
import Foundation
import LLM
import MD

public struct Doc: Identifiable, Sendable {
    public let id = UUID()
    public let name: String
    public let content: String
    public let url: URL?
    public let short: Bool

    public init(name: String, content: String, url: URL?, short: Bool) {
        self.name = name
        self.content = content
        self.url = url
        self.short = short
    }
}

public struct DocRef: Hashable, Sendable {
    public let url: URL
    public let bytes: Int
    public let short: Bool

    public init(url: URL, bytes: Int, short: Bool) {
        self.url = url
        self.bytes = bytes
        self.short = short
    }
}

public struct ImageAttachment: Identifiable, @unchecked Sendable {
    public let id = UUID()
    public let name: String
    public let file: String
    public let data: Data
    public let thumbnail: CGImage?

    public init(name: String, file: String, data: Data,
                thumbnail: CGImage?) {
        self.name = name
        self.file = file
        self.data = data
        self.thumbnail = thumbnail
    }
}

public struct ClipAttachment: Identifiable, @unchecked Sendable {
    public let id = UUID()
    public let name: String
    public let file: String
    public let url: URL
    public let isVideo: Bool
    public var thumbnail: CGImage?

    public init(name: String, file: String, url: URL, isVideo: Bool,
                thumbnail: CGImage?) {
        self.name = name
        self.file = file
        self.url = url
        self.isVideo = isVideo
        self.thumbnail = thumbnail
    }

    public func attached(_ media: any MediaEncoder, budget: Int,
                         onFrame: (@Sendable (VideoPeek) -> Void)? = nil)
        async throws -> Attached {
        var out = Attached()
        if isVideo {
            var seen = 0
            out.append(try await media.video(url: url, budget: budget) {
                img, _ in
                if let onFrame,
                   let peek = VideoPeek(index: seen, full: img) {
                    onFrame(peek)
                }
                seen += 1
            })
            for heard in (try? await media.audio(url: url)) ?? [] {
                out.parts.append(.audio)
                out.spans.append(heard)
            }
        } else {
            for heard in try await media.audio(url: url) {
                out.parts.append(.audio)
                out.spans.append(heard)
            }
        }
        return out
    }
}

public struct ToolRound: Identifiable {
    public let id: Int
    public let emitted: String
    public let label: String
    public let symbol: String
    public let args: String
    public var result: String?

    public init(id: Int, emitted: String, label: String, symbol: String,
                args: String, result: String?) {
        self.id = id
        self.emitted = emitted
        self.label = label
        self.symbol = symbol
        self.args = args
        self.result = result
    }
}

public struct Message: Identifiable {
    public let id = UUID()
    public let fromUser: Bool
    public var text: String
    public var images: [CGImage] = []
    public var clips: [URL] = []
    public var posters: [CGImage] = []
    public var docs: [DocRef] = []
    public var toolRounds: [ToolRound] = []
    public var reasoning = ""
    public var placeholder = false
    public var loopStopped = false
    public var answerDoc = Markdown.Document.empty
    public var reasoningDoc = Markdown.Document.empty
    public let answerStream = MarkdownStream()
    public let reasoningStream = MarkdownStream()

    public init(fromUser: Bool, text: String, images: [CGImage] = [],
                clips: [URL] = [], posters: [CGImage] = [],
                docs: [DocRef] = [],
                toolRounds: [ToolRound] = [], reasoning: String = "",
                placeholder: Bool = false, loopStopped: Bool = false,
                answerDoc: Markdown.Document = .empty,
                reasoningDoc: Markdown.Document = .empty) {
        self.fromUser = fromUser
        self.text = text
        self.images = images
        self.clips = clips
        self.posters = posters
        self.docs = docs
        self.toolRounds = toolRounds
        self.reasoning = reasoning
        self.placeholder = placeholder
        self.loopStopped = loopStopped
        self.answerDoc = answerDoc
        self.reasoningDoc = reasoningDoc
    }
}
