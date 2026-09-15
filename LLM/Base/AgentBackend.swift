import Foundation


public enum ContentPart: Sendable {
    case text(String)
    case image
    case audio
    case video

    public var noun: String {
        switch self {
        case .text: "text"
        case .image: "image"
        case .audio: "audio"
        case .video: "video"
        }
    }
}

public struct AgentMessage: Sendable {
    public var role: String
    public var content: String
    public var contentParts: [ContentPart]?
    public var toolCalls: [AgentToolCall]
    public var reasoning: String?
    public var name: String?

    public init(role: String, content: String,
                contentParts: [ContentPart]? = nil,
                toolCalls: [AgentToolCall] = [], reasoning: String? = nil,
                name: String? = nil) {
        self.role = role
        self.content = content
        self.contentParts = contentParts
        self.toolCalls = toolCalls
        self.reasoning = reasoning
        self.name = name
    }
}

public struct AgentToolCall: Sendable {
    public var name: String
    public var arguments: [ToolArg]

    public init(name: String, arguments: [ToolArg]) {
        self.name = name
        self.arguments = arguments
    }
}

public struct TurnMetrics: Sendable {
    public let ctx: Int
    public let thinkTokens: Int
    public let contentTokens: Int
    public let pp: Double
    public let tg: Double
    public let endReason: String
    public let overrun: Int
    public let stopToken: Int32?
    public let prefillDone: Int
    public let prefillTotal: Int
    public let readFraction: Double
    public let readStop: String
    public init(ctx: Int, thinkTokens: Int, contentTokens: Int,
                pp: Double = 0, tg: Double = 0, endReason: String = "",
                overrun: Int = 0, stopToken: Int32? = nil,
                prefillDone: Int = 0, prefillTotal: Int = 0,
                readFraction: Double = 1, readStop: String = "") {
        self.ctx = ctx
        self.thinkTokens = thinkTokens
        self.contentTokens = contentTokens
        self.pp = pp
        self.tg = tg
        self.endReason = endReason
        self.overrun = overrun
        self.stopToken = stopToken
        self.prefillDone = prefillDone
        self.prefillTotal = prefillTotal
        self.readFraction = readFraction
        self.readStop = readStop
    }
}

public protocol AgentBackend: Sendable {
    func encode(_ text: String) -> [Int32]
    func tokenBytes(_ id: Int32) -> [UInt8]
    func text(_ ids: [Int32]) -> String
    var eos: Int32 { get }
    var eosIds: Set<Int32> { get }
    func reset() async
    func useSampler(_ s: Sampler?) async
    func extend(_ ids: [Int32]) async throws -> Int32
    func mark() async throws
    func rewind() async throws
    var position: Int { get async }
    func decode(_ token: Int32) async throws -> Int32
    func requestStop()
    func shouldStop() -> Bool
    func queuedCount() async -> Int
    func drainSpecTurn() -> SpecTurn?
    func useSpeculation(_ on: Bool)
    // `ids` carries every span's placeholder already expanded to its block;
    // `spans` lays the tower rows over those positions.
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32
    func supportsSoftTokens() async -> Bool
    var bosToken: String { get }
    func supportsVision() async -> Bool
    func saveState() async throws -> any BackendState
    func loadState(_ state: any BackendState) async throws
    func checkpoint() async throws -> any BackendState
    func rollback(_ state: any BackendState) async throws
    func attach(_ dir: URL) async throws
    func detach() async
    var stateBytes: Int { get async }
    func park(to dir: URL, meta: Data) async throws
    func resume(from dir: URL) async throws -> Data
    func prime(from cooked: URL, into live: URL) async throws -> Data
    func precook(to cooked: URL, meta: Data) async throws
}

public protocol BackendState: Sendable {}

public struct NullBackendState: BackendState {}

public extension AgentBackend {
    var eosIds: Set<Int32> { [eos] }
    func requestStop() {}
    func shouldStop() -> Bool { false }
    func queuedCount() async -> Int { 0 }
    func drainSpecTurn() -> SpecTurn? { nil }
    func useSpeculation(_ on: Bool) {}
    func supportsVision() async -> Bool { false }
    func supportsSoftTokens() async -> Bool { false }
    var bosToken: String { "" }
    func extendSoft(_ ids: [Int32], spans: [SoftSpan]) async throws -> Int32 {
        throw EngineError.missingModel("soft tokens")
    }
    func attach(_ dir: URL) async throws {}
    func detach() async {}
    var stateBytes: Int { get async { 0 } }
    func park(to dir: URL, meta: Data) async throws {}
    func resume(from dir: URL) async throws -> Data {
        throw EngineError.missingModel("state")
    }
    func prime(from cooked: URL, into live: URL) async throws -> Data {
        throw EngineError.missingModel("state")
    }
    func precook(to cooked: URL, meta: Data) async throws {}
    func checkpoint() async throws -> any BackendState {
        try await saveState()
    }
    func rollback(_ state: any BackendState) async throws {
        try await loadState(state)
    }
}
