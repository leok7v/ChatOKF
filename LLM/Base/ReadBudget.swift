import Foundation

public final class ReadBudget: @unchecked Sendable {
    public typealias Ranker = @Sendable (String, [String]) async -> [Float]

    public static let shared = ReadBudget()
    static let seconds = Flags.double("read-budget") ?? 12
    static let bytesPerToken = 3.5
    static let floorChars = 2000
    static let ceilingChars = 24000
    static let turnFactor = 2.5
    static let defaultRate = 100.0
    static let leastRead = 500

    private let lock = NSLock()
    private var rate = ReadBudget.defaultRate
    private var spent = 0
    private var rankerBox: Ranker?
    private var asked = ""

    init() {}

    public func setRate(_ pp: Double) {
        if pp > 0 {
            lock.lock()
            rate = pp
            lock.unlock()
        }
    }

    private func article() -> Int {
        let raw = rate * ReadBudget.seconds * ReadBudget.bytesPerToken
        return min(ReadBudget.ceilingChars,
                   max(ReadBudget.floorChars, Int(raw)))
    }

    public var articleChars: Int {
        lock.lock()
        defer { lock.unlock() }
        return article()
    }

    public var turnChars: Int {
        lock.lock()
        defer { lock.unlock() }
        return Int(Double(article()) * ReadBudget.turnFactor)
    }

    public var room: Int {
        lock.lock()
        defer { lock.unlock() }
        return max(0, Int(Double(article()) * ReadBudget.turnFactor) - spent)
    }

    public var exhausted: Bool { room < ReadBudget.leastRead }

    public func charge(_ chars: Int) {
        lock.lock()
        spent += max(0, chars)
        lock.unlock()
    }

    public func beginTurn() {
        lock.lock()
        spent = 0
        lock.unlock()
    }

    public func setQuestion(_ text: String) {
        lock.lock()
        asked = text
        lock.unlock()
    }

    public var question: String {
        lock.lock()
        defer { lock.unlock() }
        return asked
    }

    public var ranker: Ranker? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return rankerBox
        }
        set {
            lock.lock()
            rankerBox = newValue
            lock.unlock()
        }
    }
}
