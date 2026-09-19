import Foundation

public protocol TextEngine: AnyObject {
    associatedtype Bookmark: Sendable
    var pos: Int { get }
    var sampler: Sampler? { get set }
    var queued: Int { get }
    func reset()
    func extend(_ ids: [Int32]) -> Int32
    func decode(_ token: Int32) -> Int32
    func requestStop()
    func shouldStop() -> Bool
    func drainSpecTurn() -> SpecTurn?
    func drainGPUSeconds() -> Double
    var gpuSeconds: Double { get }
    func setSpeculation(_ on: Bool)
    func bookmark() -> Bookmark
    func restore(_ b: Bookmark)
    func retain(from position: Int)
    func attach(_ dir: URL) throws
    func flush(_ dir: URL) throws
    func export(_ dir: URL) throws
    func detach()
    var stateBytes: Int { get }
}

public extension TextEngine {
    var queued: Int { 0 }
    func retain(from position: Int) {}
    func requestStop() {}
    func shouldStop() -> Bool { false }
    func drainSpecTurn() -> SpecTurn? { nil }
    func drainGPUSeconds() -> Double { 0 }
    var gpuSeconds: Double { 0 }
    func setSpeculation(_ on: Bool) {}
    func attach(_ dir: URL) throws { throw EngineError.missingModel("state") }
    func flush(_ dir: URL) throws {}
    func export(_ dir: URL) throws {}
    func detach() {}
    var stateBytes: Int { 0 }
}

public protocol Tokenizing: Sendable {
    var eosId: Int32 { get }
    var eosIds: Set<Int32> { get }
    var bosToken: String { get }
    var vocabCount: Int { get }
    func encode(_ text: String, addSpecial: Bool) -> [Int32]
    func decodeBytes(_ ids: [Int32]) -> [UInt8]
    func decode(_ ids: [Int32]) -> String
}

public class EngineBackend<E: TextEngine, T: Tokenizing>: AgentBackend,
    @unchecked Sendable {
    let engine: E
    let tokenizer: T
    var savedMark: E.Bookmark?

    public init(engine: E, tokenizer: T) {
        self.engine = engine
        self.tokenizer = tokenizer
    }

    public var eos: Int32 { tokenizer.eosId }
    public var eosIds: Set<Int32> { tokenizer.eosIds }
    public var bosToken: String { tokenizer.bosToken }
    public var position: Int { get async { engine.pos } }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }

    public func tokenBytes(_ id: Int32) -> [UInt8] { tokenizer.decodeBytes([id]) }
    public func text(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    public func reset() async {
        savedMark = nil
        engine.reset()
    }

    public func useSampler(_ s: Sampler?) async { engine.sampler = s }

    public func extend(_ ids: [Int32]) async throws -> Int32 {
        let out = engine.extend(ids)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func decode(_ token: Int32) async throws -> Int32 {
        engine.decode(token)
    }

    public func requestStop() { engine.requestStop() }
    public func shouldStop() -> Bool { engine.shouldStop() }
    public func queuedCount() async -> Int { engine.queued }
    public func drainSpecTurn() -> SpecTurn? { engine.drainSpecTurn() }
    public func drainGPUSeconds() -> Double { engine.drainGPUSeconds() }
    public var gpuSeconds: Double { engine.gpuSeconds }
    public func useSpeculation(_ on: Bool) { engine.setSpeculation(on) }

    public func supportsSoftTokens() async -> Bool { false }

    public func extendSoft(_ ids: [Int32],
                         spans: [SoftSpan]) async throws -> Int32 {
        throw EngineError.missingModel("soft tokens")
    }

    public func supportsVision() async -> Bool { false }

    public func mark() async throws {
        savedMark = engine.bookmark()
        engine.retain(from: engine.pos)
    }

    public func rewind() async throws {
        if let m = savedMark { engine.restore(m) }
    }

    public struct State: BackendState { let bookmark: E.Bookmark }

    public struct Turn: BackendState {
        let bookmark: E.Bookmark
        let mark: E.Bookmark?
    }

    public func saveState() async throws -> any BackendState {
        State(bookmark: engine.bookmark())
    }

    public func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { engine.restore(s.bookmark) }
    }

    public private(set) var attachedDir: URL?

    public var stateBytes: Int { get async { engine.stateBytes } }

    public func attach(_ dir: URL) async throws {
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try engine.attach(dir)
        attachedDir = dir
        savedMark = nil
    }

    public func detach() async {
        engine.detach()
        attachedDir = nil
        savedMark = nil
    }

    public func park(to dir: URL, meta: Data) async throws {
        if let from = attachedDir {
            try engine.flush(from)
            try meta.write(to: from.appendingPathComponent("meta.json"),
                           options: .atomic)
            engine.detach()
            attachedDir = nil
            savedMark = nil
            if from.standardizedFileURL != dir.standardizedFileURL {
                try? FileManager.default.removeItem(at: dir)
                try FileManager.default.moveItem(at: from, to: dir)
            }
        }
    }

    public func resume(from dir: URL) async throws -> Data {
        let meta = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        try await attach(dir)
        return meta
    }

    public func prime(from cooked: URL, into live: URL) async throws -> Data {
        engine.detach()
        attachedDir = nil
        savedMark = nil
        try? FileManager.default.removeItem(at: live)
        try EngineBackend.clone(cooked, to: live)
        return try await resume(from: live)
    }

    public func precook(to cooked: URL, meta: Data) async throws {
        if attachedDir != nil {
            try? FileManager.default.removeItem(at: cooked)
            try engine.export(cooked)
            try meta.write(to: cooked.appendingPathComponent("meta.json"),
                           options: .atomic)
        }
    }

    static func clone(_ from: URL, to: URL) throws {
        try FileManager.default.createDirectory(
            at: to.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        if clonefile(from.path, to.path, 0) != 0 {
            try FileManager.default.copyItem(at: from, to: to)
        }
    }

    public func checkpoint() async throws -> any BackendState {
        Turn(bookmark: engine.bookmark(), mark: savedMark)
    }

    public func rollback(_ state: any BackendState) async throws {
        if let t = state as? Turn {
            engine.restore(t.bookmark)
            savedMark = t.mark
        }
    }
}
