import Foundation

public protocol TextEngine: AnyObject {
    associatedtype Bookmark: Sendable
    associatedtype Parked: BackendState
    var pos: Int { get }
    var sampler: Sampler? { get set }
    var queued: Int { get }
    func reset()
    func extend(_ ids: [Int32]) -> Int32
    func decode(_ token: Int32) -> Int32
    func requestStop()
    func shouldStop() -> Bool
    func drainSpecTurn() -> SpecTurn?
    func setSpeculation(_ on: Bool)
    func bookmark() -> Bookmark
    func restore(_ b: Bookmark)
    func serialize(_ b: Bookmark) -> Data
    func deserialize(_ data: Data) -> Parked?
    func adopt(_ parked: Parked)
}

public extension TextEngine {
    var queued: Int { 0 }
    func requestStop() {}
    func shouldStop() -> Bool { false }
    func drainSpecTurn() -> SpecTurn? { nil }
    func setSpeculation(_ on: Bool) {}
}

public extension TextEngine where Parked == Bookmark {
    func adopt(_ parked: Parked) { restore(parked) }
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
    public func useSpeculation(_ on: Bool) { engine.setSpeculation(on) }

    public func supportsSoftTokens() async -> Bool { false }

    public func extendSoft(_ ids: [Int32],
                         spans: [SoftSpan]) async throws -> Int32 {
        throw EngineError.missingModel("soft tokens")
    }

    public func supportsVision() async -> Bool { false }

    public func mark() async throws { savedMark = engine.bookmark() }

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
        if let s = state as? State {
            engine.restore(s.bookmark)
        } else if let p = state as? E.Parked {
            engine.adopt(p)
        }
    }

    public func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State {
            out = engine.serialize(s.bookmark)
        } else if let b = state as? E.Bookmark {
            out = engine.serialize(b)
        }
        return out
    }

    public func deserializeState(_ data: Data) async throws
        -> any BackendState {
        let parked = engine.deserialize(data)
        if parked == nil {
            throw GGUFErr.parse("parked state is not this build's format")
        }
        return parked!
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
