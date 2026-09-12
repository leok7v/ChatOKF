import Foundation

public enum EngineError: Error {
    case missingModel(String)
    case contextFull
    case stopped
}

public struct SpecTurn: Sendable {
    public let cycles: Int
    public let committed: Int
    public let drafted: Int
    public let accepted: Int

    public init(cycles: Int, committed: Int, drafted: Int, accepted: Int) {
        self.cycles = cycles
        self.committed = committed
        self.drafted = drafted
        self.accepted = accepted
    }

    public var tokensPerCycle: Double {
        Double(committed) / Double(max(cycles, 1))
    }
    public var acceptRate: Double {
        Double(accepted) / Double(max(drafted, 1))
    }
}
