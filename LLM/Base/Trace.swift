import Foundation

public struct TraceEvent: Sendable, Identifiable {
    public enum Kind: String, Sendable {
        case user
        case render
        case prefill
        case prime
        case vision
        case decode
        case toolCall = "tool"
        case toolResult = "result"
        case inject
        case rewind
        case reset
        case answer
        case diag
    }

    public let id = UUID()
    public let kind: Kind
    public let t0: Date
    public let t1: Date
    public let ctx: Int
    public let tokens: Int
    public let summary: String
    public let text: String
    // A tiny JPEG (<= 128 px) so the debug view shows WHAT was attached.
    public let image: Data?

    public init(kind: Kind, t0: Date, t1: Date, ctx: Int, tokens: Int,
                summary: String, text: String, image: Data? = nil) {
        self.kind = kind
        self.t0 = t0
        self.t1 = t1
        self.ctx = ctx
        self.tokens = tokens
        self.summary = summary
        self.text = text
        self.image = image
    }
}
