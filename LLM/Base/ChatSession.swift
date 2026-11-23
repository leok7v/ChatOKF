import CryptoKit
import Foundation

final class GrammarGate: @unchecked Sendable {
    private let vocab: GrammarVocab
    private let structuralOnly: Bool
    var grammar: Grammar?
    // Armed IS "a live grammar exists", so no parallel flag can drift from it.
    var armed: Bool { grammar != nil }

    init(vocab: GrammarVocab, structuralOnly: Bool) {
        self.vocab = vocab
        self.structuralOnly = structuralOnly
    }

    func disarm() {
        grammar = nil
    }

    func mask(_ logits: inout [Float]) {
        if let g = grammar {
            g.maskLogits(vocab, &logits, structuralOnly: structuralOnly)
        }
    }
}

private final class Collected: @unchecked Sendable {
    private let lock = NSLock()
    private var acc = ""

    func add(_ s: String) -> Int {
        lock.lock()
        acc += s
        let n = acc.count
        lock.unlock()
        return n
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return acc
    }
}

private final class FedBackend: AgentBackend, @unchecked Sendable {
    private let inner: any AgentBackend
    private let lock = NSLock()
    private var ids: [Int32] = []
    private var marked: Int?

    struct State: BackendState {
        let inner: any BackendState
        let ids: [Int32]
        let marked: Int?
    }

    init(_ inner: any AgentBackend) {
        self.inner = inner
    }

    var fed: [Int32] {
        lock.lock()
        defer { lock.unlock() }
        return ids
    }

    private func update(_ change: (inout [Int32], inout Int?) -> Void) {
        lock.lock()
        change(&ids, &marked)
        lock.unlock()
    }

    private func held() -> (ids: [Int32], marked: Int?) {
        lock.lock()
        defer { lock.unlock() }
        return (ids, marked)
    }

    func adopt(_ list: [Int32]) {
        update { ids, _ in ids = list }
    }

    var markedAt: Int? { held().marked }

    static let unknown = Int32.min

    func settle(_ stop: Int32?) async {
        let extra = await inner.position - held().ids.count
        let ran: [Int32] = extra == 1 ? [stop ?? FedBackend.unknown]
            : Array(repeating: FedBackend.unknown, count: max(0, extra))
        update { ids, _ in ids += ran }
    }

    var eos: Int32 { inner.eos }
    var eosIds: Set<Int32> { inner.eosIds }
    var bosToken: String { inner.bosToken }
    var controlIds: Set<Int32> { inner.controlIds }
    var gpuSeconds: Double { inner.gpuSeconds }
    var position: Int { get async { await inner.position } }
    var stateBytes: Int { get async { await inner.stateBytes } }

    func encode(_ text: String) -> [Int32] { inner.encode(text) }
    func tokenBytes(_ id: Int32) -> [UInt8] { inner.tokenBytes(id) }
    func text(_ ids: [Int32]) -> String { inner.text(ids) }
    func useSampler(_ s: Sampler?) async { await inner.useSampler(s) }
    func requestStop() { inner.requestStop() }
    func shouldStop() -> Bool { inner.shouldStop() }
    func queuedCount() async -> Int { await inner.queuedCount() }
    func drainSpecTurn() -> SpecTurn? { inner.drainSpecTurn() }
    func drainGPUSeconds() -> Double { inner.drainGPUSeconds() }
    func useSpeculation(_ on: Bool) { inner.useSpeculation(on) }
    func supportsSoftTokens() async -> Bool {
        await inner.supportsSoftTokens()
    }
    func supportsVision() async -> Bool { await inner.supportsVision() }

    func reset() async {
        await inner.reset()
        update { ids, marked in
            ids = []
            marked = nil
        }
    }

    private func laid(_ list: [Int32], before: Int,
                      _ run: () async throws -> Int32) async throws -> Int32 {
        var count = list.count
        let result: Result<Int32, any Error>
        do {
            result = .success(try await run())
        } catch {
            let after = await inner.position
            count = min(list.count, max(0, after - before))
            result = .failure(error)
        }
        let kept = Array(list.prefix(count))
        update { ids, _ in ids += kept }
        return try result.get()
    }

    func extend(_ list: [Int32]) async throws -> Int32 {
        let before = await inner.position
        return try await laid(list, before: before) {
            try await inner.extend(list)
        }
    }

    func extendSoft(_ list: [Int32],
                    spans: [SoftSpan]) async throws -> Int32 {
        let before = await inner.position
        return try await laid(list, before: before) {
            try await inner.extendSoft(list, spans: spans)
        }
    }

    func decode(_ token: Int32) async throws -> Int32 {
        let out = try await inner.decode(token)
        update { ids, _ in ids.append(token) }
        return out
    }

    func mark() async throws {
        try await inner.mark()
        update { ids, marked in marked = ids.count }
    }

    func rewind() async throws {
        try await inner.rewind()
        update { ids, marked in
            if let at = marked { ids = Array(ids.prefix(at)) }
        }
    }

    func shift(keep: Int, count: Int) async -> Bool {
        let moved = await inner.shift(keep: keep, count: count)
        if moved {
            update { ids, marked in
                ids.removeSubrange(keep ..< keep + count)
                marked = marked.flatMap { at in
                    at >= keep + count ? at - count : nil
                }
            }
        }
        return moved
    }

    func saveState() async throws -> any BackendState {
        let held = held()
        return State(inner: try await inner.saveState(), ids: held.ids,
                     marked: held.marked)
    }

    func loadState(_ state: any BackendState) async throws {
        let s = state as? State
        try await inner.loadState(s?.inner ?? state)
        if let s { update { ids, _ in ids = s.ids } }
    }

    func checkpoint() async throws -> any BackendState {
        let held = held()
        return State(inner: try await inner.checkpoint(), ids: held.ids,
                     marked: held.marked)
    }

    func rollback(_ state: any BackendState) async throws {
        let s = state as? State
        try await inner.rollback(s?.inner ?? state)
        if let s {
            update { ids, marked in
                ids = s.ids
                marked = s.marked
            }
        }
    }

    func attach(_ dir: URL) async throws {
        try await inner.attach(dir)
        update { _, marked in marked = nil }
    }

    func detach() async {
        await inner.detach()
        update { _, marked in marked = nil }
    }

    func park(to dir: URL, meta: Data) async throws {
        try await inner.park(to: dir, meta: meta)
        update { _, marked in marked = nil }
    }

    func resume(from dir: URL) async throws -> Data {
        let meta = try await inner.resume(from: dir)
        update { _, marked in marked = nil }
        return meta
    }

    func prime(from cooked: URL, into live: URL) async throws -> Data {
        let meta = try await inner.prime(from: cooked, into: live)
        update { _, marked in marked = nil }
        return meta
    }

    func precook(to cooked: URL, meta: Data) async throws {
        try await inner.precook(to: cooked, meta: meta)
    }
}

final class SeamFaults: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return n
    }

    func add(_ what: String) {
        lock.lock()
        n += 1
        lock.unlock()
        Diag.shared.report("seam: " + what)
    }
}

final class Claims: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return n
    }

    func add(_ delta: Int) {
        lock.lock()
        n += delta
        lock.unlock()
    }
}

public struct ToolRoundEvent: Sendable {
    public let round: Int
    public let name: String
    public let resolved: String?
    public let params: [ToolArg]
    public let result: String?
}

public actor ChatSession {
    private let backend: FedBackend
    private let template: String
    private let presets: SamplingPresets
    private let vocabSize: Int
    private let samplerSeed: UInt64
    private let overthinkTokens: Set<Int32>
    private let overthinkLambda: Float
    private let wireTokens: Set<Int32>
    private let bannedControl: Set<Int32>
    private let reasoningOpeners: Set<Int32>
    private var toolSpecs: [ToolSpec] { current.runner?.tools ?? [] }
    enum GrammarMode { case off, full, structural }
    private static let grammarMode: GrammarMode = {
        switch Flags.value("tool-grammar") {
        case "full", "1": return .full
        case "off", "0": return .off
        default: return .structural
        }
    }()
    static let digitsExempt = Flags.on("digit-exempt")
    static let budgetNote = Flags.value("reasoning-note") ?? ""
    static let dryMultiplier = Float(Flags.double("dry-multiplier") ?? 0.8)
    private var gate: GrammarGate?
    private var grammarVocab: GrammarVocab?
    private let toolDialectXML: Bool
    private let wire: ChatWire
    private var logitTap: (@Sendable ([Float]) -> Void)?
    static let maxToolRounds = 10
    static let maxFruitless = 2
    // While a call is open nothing streams and the loop breaker never trips, so
    // a model free-running a huge value is an invisible forever-decode.
    static let maxOpenCallBytes = 8192
    static let toolBudgetNudge =
        "You have reached the tool-call limit for this turn. Do not call any "
        + "more tools; answer the user now using the information you already "
        + "have."
    private var history: [AgentMessage]
    // Append-only mirror of the tokens in the KV up to the turn mark, so park,
    // resume and serialize can round-trip the sequence. Empty is fresh.
    private var committed: [Int32]
    private var pending: [Int32] = []
    private var systemMark = 0
    private var userMark: Mark?
    private let context: Int
    private let shiftMode: ShiftMode
    private var starts: [Start] = []

    public enum ShiftMode: String, Sendable { case exact, soft, off }

    public static let shiftKnob =
        ShiftMode(rawValue: Flags.value("shift") ?? "") ?? .exact

    private struct Start: Codable {
        let at: Int
        let message: Int
        let tools: Int
    }
    private var softTurns = 0
    private var toolTokens = 0
    private var visionContext: Bool {
        history.contains { m in m.contentParts != nil }
    }
    private var genStartsThink = false
    private let readGuard: (@Sendable () -> String?)?
    private let chatGuard: (@Sendable () -> String?)?
    public enum TurnOutcome: String, Sendable {
        case answered, stopped, answerless, memory
    }

    public static let memoryEnd = "memory"
    static let memoryEvery = 16

    struct Turn: Sendable {
        var enableThinking: Bool
        var suppressReasoning: Bool
        var reasoningEffort: String?
        var maxTokens: Int
        var maxReasoning: Int
        var softReasoningCap: Int
        var runner: (any ToolRunner)?
        var traceSink: (@Sendable (TraceEvent) -> Void)?
        var forceEndThink = false
        var quickTurn = false
        var metaTurn = false
        var stoppable: String? = nil
        var readFraction = 1.0
        var readStop = ""
        var turnSampler: Sampler? = nil
        var lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0,
                                      contentTokens: 0)
        var turnOutcome: TurnOutcome = .answered

        var meta: Turn {
            var out = self
            out.enableThinking = false
            out.suppressReasoning = true
            out.maxReasoning = 0
            out.softReasoningCap = 0
            out.runner = nil
            out.traceSink = nil
            out.metaTurn = true
            return out
        }

        mutating func open() {
            forceEndThink = false
            quickTurn = false
            turnOutcome = .answered
            stoppable = nil
            readFraction = 1
            readStop = ""
        }
    }

    private var turn: Turn
    private var aside: Turn?

    private var current: Turn {
        get { aside ?? turn }
        set {
            if aside == nil {
                turn = newValue
            } else {
                aside = newValue
            }
        }
    }

    public var turnOutcome: TurnOutcome { turn.turnOutcome }

    public var turnRolledBack: Bool { turnOutcome != .answered }
    public var lastMetrics: TurnMetrics { turn.lastMetrics }
    private let systemStable: String
    private let systemTail: String
    private var priming: Task<Void, Never>?

    public init(backend: any AgentBackend, template: String, system: String,
                systemTail: String = "", vocabSize: Int,
                presets: SamplingPresets = .greedy,
                enableThinking: Bool = false,
                suppressReasoning: Bool = false,
                reasoningEffort: String? = nil, maxTokens: Int = .max,
                maxReasoning: Int = 0, softReasoningCap: Int = 0,
                overthink: Float = 0, seed: UInt64 = 0,
                runner: (any ToolRunner)? = nil,
                readGuard: (@Sendable () -> String?)? = nil,
                chatGuard: (@Sendable () -> String?)? = nil,
                breakers: Set<Int32>? = nil,
                grammarVocab: GrammarVocab? = nil, context: Int = 0,
                shift: ShiftMode = ChatSession.shiftKnob) {
        self.backend = FedBackend(backend)
        self.context = max(context, 0)
        self.shiftMode = shift
        self.readGuard = readGuard
        self.chatGuard = chatGuard
        self.breakers = breakers
        self.grammarVocab = grammarVocab
        self.template = template
        let wire = ChatWire.derive(template)
        self.wire = wire
        self.openTag = Array(wire.toolCallOpen.utf8)
        self.closeTag = Array(wire.toolCallClose.utf8)
        self.thinkOpen = Array(wire.reasoningOpen.utf8)
        self.thinkClose = Array(wire.reasoningClose.utf8)
        self.toolDialectXML = template.contains("<function=")
        self.systemStable = system
        self.systemTail = systemTail
        self.presets = presets
        self.vocabSize = max(vocabSize, 1)
        self.turn = Turn(enableThinking: enableThinking,
                         suppressReasoning: suppressReasoning,
                         reasoningEffort: reasoningEffort,
                         maxTokens: maxTokens, maxReasoning: maxReasoning,
                         softReasoningCap: softReasoningCap, runner: runner)
        self.overthinkLambda = overthink
        self.samplerSeed = seed
        var markers: Set<Int32> = []
        if overthink != 0 {
            for marker in Sampler.overthinkMarkers {
                for variant in [marker, " " + marker] {
                    let ids = backend.encode(variant)
                    if ids.count == 1 { markers.insert(ids[0]) }
                }
            }
        }
        self.overthinkTokens = markers
        var specials: Set<Int32> = []
        for marker in wire.penaltyExemptCandidates {
            let ids = backend.encode(marker)
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        for ch in "0123456789." where ChatSession.digitsExempt {
            let ids = backend.encode(String(ch))
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        self.wireTokens = specials
        self.bannedControl = ChatSession.bannedControl(backend, template)
        var openers: Set<Int32> = []
        if wire.derivedReasoning,
           let first = backend.encode(wire.reasoningOpen).first {
            let text = String(decoding: backend.tokenBytes(first),
                              as: UTF8.self)
            if text.count >= 3, wire.reasoningOpen.hasPrefix(text) {
                openers.insert(first)
            }
        }
        self.reasoningOpeners = openers
        self.history = [AgentMessage(role: "system",
                                     content: system + systemTail)]
        self.committed = []
    }

    public func setThinking(_ on: Bool) {
        turn.enableThinking = on
    }

    static func bannedControl(_ backend: any AgentBackend,
                              _ template: String) -> Set<Int32> {
        let probe = [
            AgentMessage(role: "system", content: "s"),
            AgentMessage(role: "user", content: "u"),
            AgentMessage(role: "assistant", content: "a",
                         toolCalls: [AgentToolCall(
                             name: "f",
                             arguments: [ToolArg(name: "k", value: "v")])],
                         reasoning: "r"),
            AgentMessage(role: "tool", content: "t", name: "f"),
            AgentMessage(role: "user", content: "w"),
        ]
        var seen: Set<Int32> = []
        for thinking in [true, false] {
            let text = (try? renderPrompt(
                template: template, messages: probe, tools: [],
                addGenerationPrompt: true, enableThinking: thinking,
                reasoningEffort: nil, bosToken: backend.bosToken)) ?? ""
            seen.formUnion(backend.encode(text))
        }
        return backend.controlIds.subtracting(seen)
            .subtracting(backend.eosIds)
    }

    public func setSuppressReasoning(_ on: Bool) {
        turn.suppressReasoning = on
    }

    public func setReasoningEffort(_ level: String?) {
        turn.reasoningEffort = level
    }

    public func setSpeculation(_ on: Bool) {
        backend.useSpeculation(on)
    }

    public func setTools(_ runner: (any ToolRunner)?) {
        turn.runner = runner
    }

    public func setTrace(_ sink: (@Sendable (TraceEvent) -> Void)?) {
        turn.traceSink = sink
    }

    public func setLogitTap(_ tap: (@Sendable ([Float]) -> Void)?) {
        logitTap = tap
    }

    public func supportsSoftTokens() async -> Bool {
        await backend.supportsSoftTokens()
    }

    private func trace(_ kind: TraceEvent.Kind, from t0: Date? = nil,
                       until t1: Date? = nil, ctx: Int = -1, tokens: Int = 0,
                       summary: String, text: String = "",
                       image: Data? = nil) {
        if let sink = current.traceSink {
            let now = Date()
            sink(TraceEvent(kind: kind, t0: t0 ?? now, t1: t1 ?? now, ctx: ctx,
                            tokens: tokens, summary: summary, text: text,
                            image: image))
        }
    }

    public func requestQuickAnswer() {
        current.forceEndThink = true
        current.quickTurn = true
    }

    static let quickSkip = "skipped: the user asked for the answer now. Do "
        + "not call any more tools; answer from what you already have."

    public func setReasoningCaps(soft: Int, hard: Int) {
        turn.softReasoningCap = soft
        turn.maxReasoning = hard
    }

    public nonisolated func requestStop() {
        backend.requestStop()
    }

    public func reset() async {
        unanswered = ""
        history = Array(history.prefix(1))
        committed = []
        pending = []
        systemMark = 0
        starts = []
        softTurns = 0
        toolTokens = 0
        attachmentCounts = [:]
        await backend.reset()
        Diag.memory?("session reset")
        turn.lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0,
                                       contentTokens: 0)
        trace(.reset, ctx: 0, summary: "new chat")
    }

    public struct ChatContext: Sendable {
        let state: any BackendState
        let history: [AgentMessage]
        let committed: [Int32]
        let attachments: [String: Int]
    }

    public func park() async throws -> ChatContext {
        try await toReply()
        return ChatContext(state: try await backend.saveState(),
                           history: history, committed: committed,
                           attachments: attachmentCounts)
    }

    public func resume(_ context: ChatContext) async throws {
        try await backend.loadState(context.state)
        history = context.history
        committed = context.committed
        pending = committed
        attachmentCounts = context.attachments
        try await markReply()
    }

    static let seamFaults = SeamFaults()

    private func checkFed(_ want: [Int32], _ site: String) {
        let fed = backend.fed
        if fed != want {
            ChatSession.seamFaults.add(
                "\(site) fed \(fed.count) ids, want \(want.count)")
        }
    }

    private struct Mark {
        let state: any BackendState
        let ids: [Int32]
        let messages: Int
        let tools: Int
    }

    private func saveMark() async throws -> Mark {
        let ids = backend.fed
        return Mark(state: try await backend.checkpoint(), ids: ids,
                    messages: history.count, tools: toolTokens)
    }

    private func loadMark(_ mark: Mark) async throws {
        try await backend.rollback(mark.state)
    }

    private func markUser() async {
        userMark = try? await saveMark()
    }

    private func markReply() async throws {
        try await backend.mark()
        checkFed(committed, "mark")
    }

    private func toReply() async throws {
        pending = committed
        _ = try await catchUp()
    }

    private func catchUp() async throws -> Int32 {
        var next = backend.eos
        let fed = backend.fed
        if ChatSession.common(fed, pending) < fed.count {
            try await restore()
        }
        let from = backend.fed.count
        if from < pending.count {
            next = try await backend.extend(Array(pending[from...]))
        }
        return next
    }

    private func settle() async {
        await backend.settle(current.lastMetrics.stopToken)
        pending = backend.fed
    }

    static func common(_ a: [Int32], _ b: [Int32]) -> Int {
        var same = 0
        while same < a.count && same < b.count && a[same] == b[same] {
            same += 1
        }
        return same
    }

    private func restore() async throws {
        let fed = backend.fed
        let same = ChatSession.common(fed, pending)
        if !fed.isEmpty, let at = backend.markedAt, at <= same {
            try await backend.rewind()
            trace(.rewind, ctx: await backend.position,
                  summary: "rewind to turn mark")
        } else {
            let stale = await backend.position
            if stale > 0 {
                Diag.shared.report(
                    "RESET reprefill (history=\(history.count) msgs)")
                trace(.reset, ctx: 0,
                      summary: "fresh turn over stale state (\(stale) tok)")
            }
            await backend.reset()
        }
    }

    private struct ContextMeta: Codable {
        let stamp: String
        let committed: [Int32]
        let roles: [String]
        let contents: [String]
        let attachments: [String: Int]
        let bytes: Int?
        let primed: Int?
        let soft: Int?
        let tools: Int?
        let starts: [Start]?
    }

    public private(set) var lastSaved: (tokens: Int, bytes: Int)?

    public var committedCount: Int { committed.count }

    public var messageCount: Int { history.count }

    public var replayTokens: Int {
        max(0, committed.count - systemMark - toolTokens)
    }

    public var hasSoftTurns: Bool { softTurns > 0 }

    public private(set) var liveDir: URL?

    public func attach(live dir: URL) async throws {
        try await backend.attach(dir)
        liveDir = dir
    }

    public func detach() async {
        await backend.detach()
    }

    private func metaBytes(_ stamp: String) async throws -> Data {
        try JSONBytes.reproducible(ContextMeta(
            stamp: stamp, committed: committed,
            roles: history.map { m in m.role },
            contents: history.map { m in m.content },
            attachments: attachmentCounts,
            bytes: await backend.stateBytes, primed: systemMark,
            soft: softTurns, tools: toolTokens, starts: starts))
    }

    private func adoptMeta(_ data: Data, stamp: String) async -> Bool {
        var loaded = false
        if let meta = try? JSONDecoder().decode(ContextMeta.self, from: data),
           meta.stamp == stamp {
            var restored: [AgentMessage] = []
            for i in 0 ..< meta.roles.count {
                restored.append(AgentMessage(role: meta.roles[i],
                                             content: meta.contents[i]))
            }
            history = restored
            committed = meta.committed
            attachmentCounts = meta.attachments
            systemMark = meta.primed ?? 0
            softTurns = meta.soft ?? 0
            toolTokens = meta.tools ?? 0
            starts = meta.starts ?? []
            turn.lastMetrics = TurnMetrics(ctx: await backend.position,
                                      thinkTokens: 0, contentTokens: 0)
            loaded = true
        }
        return loaded
    }

    public static func allocated(_ dir: URL) -> Int {
        let fm = FileManager.default
        let meta = try? JSONDecoder().decode(
            ContextMeta.self,
            from: Data(contentsOf: dir.appendingPathComponent("meta.json")))
        let files = (try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]))
            ?? []
        let onDisk = files.reduce(0) { sum, url in
            sum + ((try? url.resourceValues(
                forKeys: [.totalFileAllocatedSizeKey]))?
                .totalFileAllocatedSize ?? 0)
        }
        return max(onDisk, meta?.bytes ?? 0)
    }

    public func allocatedLive() -> Int {
        liveDir.map { dir in ChatSession.allocated(dir) } ?? 0
    }

    public func park(to dir: URL, stamp: String) async throws {
        await priming?.value
        try await saveReplyMark(to: dir, stamp: stamp)
    }

    public func resume(from dir: URL, stamp: String) async throws -> Bool {
        let loaded = try await loadReplyMark(from: dir, stamp: stamp)
        if loaded {
            liveDir = dir
            lastSaved = (committed.count, ChatSession.allocated(dir))
        } else {
            await backend.detach()
        }
        return loaded
    }

    private func saveReplyMark(to dir: URL, stamp: String) async throws {
        try await toReply()
        try await backend.park(to: dir, meta: await metaBytes(stamp))
        lastSaved = (committed.count, ChatSession.allocated(dir))
    }

    private func loadReplyMark(from dir: URL,
                               stamp: String) async throws -> Bool {
        let meta = try await backend.resume(from: dir)
        let loaded = await adoptMeta(meta, stamp: stamp)
        if loaded {
            backend.adopt(committed)
            pending = committed
            try await markReply()
        }
        return loaded
    }

    private func saveSystemMark(to url: URL, stamp: String) async throws {
        try await backend.precook(to: url, meta: await metaBytes(stamp))
        lastSaved = (committed.count, ChatSession.allocated(url))
    }

    private func loadSystemMark(from url: URL, stamp: String) async -> Bool {
        var loaded = false
        if let live = liveDir,
           let meta = try? await backend.prime(from: url, into: live) {
            loaded = await adoptMeta(meta, stamp: stamp)
        }
        return loaded
    }

    private func render(_ messages: [AgentMessage], tools: [ToolSpec] = [],
                        generation: Bool = false,
                        plain: Bool = false) -> String {
        (try? renderPrompt(
            template: template, messages: messages, tools: tools,
            addGenerationPrompt: generation,
            enableThinking: plain ? false : current.enableThinking,
            reasoningEffort: plain ? nil : current.reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
    }

    private struct Block {
        let closed: String
        let gen: String
    }

    private func block(_ messages: [AgentMessage],
                       after context: [AgentMessage]?,
                       tools: [ToolSpec] = []) -> Block {
        let head = context.map { c in render(c) } ?? ""
        let all = (context ?? []) + messages
        let shut = render(all, tools: tools)
        let full = render(all, tools: tools, generation: true)
        if full.isEmpty {
            Diag.shared.report("template rendered nothing for these variables")
        }
        if !full.hasPrefix(shut) {
            Diag.shared.report("generation prompt is not a render suffix")
        }
        let closed = shut.hasPrefix(head)
            ? String(shut.dropFirst(head.count)) : shut
        let gen = full.hasPrefix(shut)
            ? String(full.dropFirst(shut.count)) : ""
        return Block(closed: closed, gen: gen)
    }

    private func seeded(_ gen: String) -> String {
        var out = gen
        if current.metaTurn { out += ChatSession.titleSeed(out, wire) }
        out = reasoningSeed(out)
        genStartsThink = wire.startsInReasoning(genPrompt: out, enabled: true)
        return out
    }

    private func systemRenders() -> (full: String, prefix: String) {
        let probe = AgentMessage(role: "user", content: "x")
        let both = render([history[0], probe], tools: toolSpecs)
        // No tools and no reasoning, so these bytes are the user TURN alone; a
        // probe carrying either is not a suffix of `both` on Qwen3.8.
        var probeText = render([probe], plain: true)
        let lead = render([], plain: true)
        if !lead.isEmpty, probeText.hasPrefix(lead) {
            probeText = String(probeText.dropFirst(lead.count))
        }
        var full = ""
        if !both.isEmpty, !probeText.isEmpty, both.hasSuffix(probeText) {
            full = String(both.dropLast(probeText.count))
        }
        if full.isEmpty {
            Diag.shared.report("no precook prefix under this template")
        }
        var prefix = full
        if !systemTail.isEmpty,
           let at = full.range(of: systemTail, options: .backwards) {
            prefix = String(full[..<at.lowerBound])
        }
        return (full, prefix)
    }

    private static func stamp(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8))
            .map { b in String(format: "%02x", b) }.joined()
    }

    public var precookStamp: String {
        let prefix = systemRenders().prefix
        return prefix.isEmpty ? "" : ChatSession.stamp(prefix)
    }

    public func precook(to url: URL) async throws {
        let r = systemRenders()
        if committed.isEmpty && !r.prefix.isEmpty {
            let t0 = Date()
            let prefixIds = backend.encode(r.prefix)
            let fullIds = backend.encode(r.full)
            if let liveDir { try await backend.attach(liveDir) }
            await backend.reset()
            _ = try await backend.extend(prefixIds)
            committed = prefixIds
            Diag.memory?("precook before save")
            try await saveSystemMark(to: url,
                                     stamp: ChatSession.stamp(r.prefix))
            Diag.memory?("precook after save")
            if fullIds.count >= prefixIds.count,
               Array(fullIds.prefix(prefixIds.count)) == prefixIds {
                let tail = Array(fullIds[prefixIds.count...])
                if !tail.isEmpty { _ = try await backend.extend(tail) }
            } else {
                await backend.reset()
                _ = try await backend.extend(fullIds)
            }
            committed = fullIds
            pending = fullIds
            systemMark = fullIds.count
            try await markReply()
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: fullIds.count,
                  summary: String(format: "precooked system prefix (%.1fs)",
                                  Date().timeIntervalSince(t0)))
        }
    }

    public func primeOrCook(at url: URL, resetFirst: Bool = false) {
        let running = priming
        priming = Task {
            await running?.value
            if resetFirst { await reset() }
            if await prime(from: url) == false {
                try? await precook(to: url)
            }
            if ChatSession.dryMultiplier > 0 { _ = sequenceBreakers() }
        }
    }

    private var engineUsers = 0
    private let claims = Claims()
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    private func enterEngine() { engineUsers += 1 }

    private func enterClaimed() async -> Bool {
        await priming?.value
        enterEngine()
        claims.add(-1)
        let live = !Task.isCancelled
        if !live { dropUnentered() }
        return live
    }

    private func dropUnentered() {
        current.open()
        current.turnOutcome = .stopped
        current.lastMetrics = TurnMetrics(
            ctx: current.lastMetrics.ctx, thinkTokens: 0, contentTokens: 0,
            endReason: "cancelled")
        trace(.rewind, summary: "turn stopped before it entered the engine")
    }

    private func leaveEngine() {
        engineUsers -= 1
        if engineUsers == 0 {
            let waiting = idleWaiters
            idleWaiters = []
            for waiter in waiting { waiter.resume() }
        }
    }

    public func quiesce() async {
        backend.requestStop()
        while engineUsers > 0 || claims.count > 0 {
            await withCheckedContinuation { c in idleWaiters.append(c) }
        }
    }

    public func endPriming() async {
        backend.requestStop()
        priming?.cancel()
        await priming?.value
        priming = nil
    }

    public func awaitPriming() async {
        await priming?.value
    }

    public func prime(from url: URL) async -> Bool {
        let t0 = Date()
        let r = systemRenders()
        var ok = false
        var primed = false
        if !r.prefix.isEmpty {
            primed = await loadSystemMark(
                from: url, stamp: ChatSession.stamp(r.prefix))
            ok = primed
        }
        if ok {
            backend.adopt(committed)
            let fullIds = backend.encode(r.full)
            let n = committed.count
            // The restored state must hold exactly the committed prefix, which
            // must open today's render; a no-op serialization fails here.
            ok = await backend.position == n && n <= fullIds.count
                && Array(fullIds.prefix(n)) == committed
            if ok, n < fullIds.count {
                ok = (try? await backend.extend(
                    Array(fullIds[n...]))) != nil
            }
            if ok {
                committed = fullIds
                pending = fullIds
                systemMark = fullIds.count
                ok = (try? await markReply()) != nil
            }
            history = [AgentMessage(role: "system",
                                    content: systemStable + systemTail)]
        }
        if primed && !ok {
            if let liveDir { try? await backend.attach(liveDir) }
            await backend.reset()
            committed = []
            pending = []
        }
        if ok {
            lastSaved = (committed.count, ChatSession.allocated(url))
            Diag.memory?("primed \(committed.count) ids from cache")
            trace(.prime, from: t0, ctx: await backend.position,
                  tokens: committed.count,
                  summary: String(format: "primed from cache (%.2fs)",
                                  Date().timeIntervalSince(t0)))
        }
        return ok
    }

    public nonisolated func reply(
        _ user: String, stoppable: String? = nil,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onTool: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            self.claims.add(1)
            let task = Task {
                await self.runTurn(user, stoppable: stoppable,
                                   onReasoning: onReasoning,
                                   onTool: onTool,
                                   onToolRound: onToolRound) { piece in
                    cont.yield(piece)
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    public nonisolated func replySoft(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool = true,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            self.claims.add(1)
            let task = Task {
                await self.runSoftTurn(
                    user, parts: parts, spans: spans, labelled: labelled,
                    onReasoning: onReasoning, onToolRound: onToolRound
                ) { piece in cont.yield(piece) }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    private func reasoningSeed(_ gen: String) -> String {
        var out = gen
        let wanted = current.enableThinking && !current.suppressReasoning
            && !current.metaTurn
            && wire.derivedReasoning && !wire.opensReasoning(gen)
            && !wire.closesReasoning(gen)
        if wanted { out += wire.reasoningOpen + "\n" }
        return out
    }

    static func titleSeed(_ gen: String, _ wire: ChatWire) -> String {
        var out = ""
        if !wire.closesReasoning(gen), wire.opensReasoning(gen) {
            out = wire.reasoningClose
        }
        return out + ChatSession.metaBlock
    }

    static let metaBlock = "```text\n"

    private func oneShot(_ instruction: String, stopAfter: Int,
                         _ label: String) async -> String {
        await priming?.value
        var raw = ""
        if history.count > 1 {
            enterEngine()
            defer { leaveEngine() }
            let began = Date()
            let outer = aside
            aside = current.meta
            _ = try? await catchUp()
            await markUser()
            let opened = pending
            let messages = history.count
            let asked = await askIds(instruction)
            pending = asked.map { ids in opened + ids }
                ?? committed + relayIds(instruction)
            raw = await metaReply(stopAfter: stopAfter)
            await backend.settle(current.lastMetrics.stopToken)
            let spent = current.lastMetrics
            aside = outer
            history = Array(history.prefix(messages))
            pending = asked == nil ? committed : opened
            if asked != nil, let mark = userMark {
                try? await loadMark(mark)
            }
            Diag.shared.report(.turn, String(
                format: "%@ %@ raw=%@ think=%d content=%d in %.1fs", label,
                asked == nil ? "relaid" : "appended",
                raw.debugDescription, spent.thinkTokens, spent.contentTokens,
                Date().timeIntervalSince(began)))
        }
        return raw
    }

    private static let metaUser = "ZqUserZq"
    private static let metaBody = "ZqBodyZq"

    private func relayIds(_ instruction: String) -> [Int32] {
        let last = history.last.map { m in
            m.role == "assistant" ? [m] : []
        } ?? []
        let made = block(last + [AgentMessage(role: "user",
                                              content: instruction)],
                         after: [])
        return backend.encode(made.closed) + backend.encode(seeded(made.gen))
    }

    private func askIds(_ instruction: String,
                        fenced: Bool = true) async -> [Int32]? {
        var out: [Int32]? = nil
        let end = current.lastMetrics
        let position = await backend.position
        if position == end.ctx, end.endReason != "tool-call",
           pending.count > committed.count {
            let probe = AgentMessage(role: "user",
                                     content: ChatSession.metaUser)
            let prior = AgentMessage(role: "assistant",
                                     content: ChatSession.metaBody)
            let ask = AgentMessage(role: "user", content: instruction)
            let asked = render([probe, prior, ask])
            let full = render([probe, prior, ask], generation: true)
            if let body = asked.range(of: ChatSession.metaBody),
               full.hasPrefix(asked) {
                var gen = String(full.dropFirst(asked.count))
                gen += ChatSession.titleSeed(gen, wire)
                if !fenced { gen.removeLast(ChatSession.metaBlock.count) }
                let laid = backend.encode(String(asked[body.upperBound...]))
                var head: [Int32]? = nil
                if end.overrun == 0 {
                    head = laid
                } else if end.overrun == 1, laid.first == end.stopToken {
                    head = Array(laid.dropFirst())
                }
                genStartsThink = wire.startsInReasoning(genPrompt: gen,
                                                        enabled: true)
                out = head.map { ids in ids + backend.encode(gen) }
            }
        }
        return out
    }

    private func metaReply(stopAfter: Int) async -> String {
        await installTurnSampler(vision: visionContext)
        let collected = Collected()
        let brake = backend
        if let seed = try? await catchUp() {
            checkFed(pending, "meta append")
            _ = await decodeStep(seed: seed, pp: 0, tally: (0, 0),
                                 onReasoning: nil) { piece in
                if collected.add(piece) > stopAfter { brake.requestStop() }
            }
        }
        return collected.text
    }

    private func judgeIds(_ ask: String) async
        -> (ids: [Int32]?, path: String) {
        var out: (ids: [Int32]?, path: String) = (nil, "relay")
        if history.count == 1 {
            let made = block([AgentMessage(role: "user", content: ask)],
                             after: [])
            out = (backend.encode(made.closed + made.gen), "fresh")
        } else if pending.count > committed.count {
            out = (await askIds(ask, fenced: false), "append")
        } else {
            out.ids = relayIds(ask)
        }
        return out
    }

    private struct JudgeCost {
        var path = ""
        var kv = 0
        var caught = 0
        var caughtSeconds = 0.0
        var renderSeconds = 0.0
        var head = 0
        var headSeconds = 0.0
        var tails: [(tokens: Int, seconds: Double)] = []

        var line: String {
            let each = tails.map { tail in
                String(format: "%d in %.2fs", tail.tokens, tail.seconds)
            }.joined(separator: ", ")
            let longest = kv + head + (tails.map { t in t.tokens }.max() ?? 0)
            return String(
                format: "%@, prompt %d tok, %d in the KV, caught up %d in "
                    + "%.2fs, rendered in %.2fs, head %d in %.2fs, tails %@",
                path, longest, kv, caught, caughtSeconds, renderSeconds,
                head, headSeconds, each.isEmpty ? "none" : each)
        }
    }

    private func laidTimed(_ ids: [Int32]) async throws -> Double {
        let began = Date()
        if !ids.isEmpty { _ = try await backend.extend(ids) }
        return Date().timeIntervalSince(began)
    }

    private func judgeShares(_ laid: [[Int32]], _ wanted: [Set<Int32>],
                             _ tap: LogitTap, _ cost: JudgeCost) async throws
        -> (shares: [[Double]], cost: JudgeCost) {
        var out: (shares: [[Double]], cost: JudgeCost) = ([], cost)
        var head = laid.count > 1 ? laid[0] : []
        for ids in laid {
            head = Array(head.prefix(Judge.common(head, ids)))
        }
        out.cost.head = head.count
        out.cost.headSeconds = try await laidTimed(head)
        var rest = try await saveMark()
        for (index, ids) in laid.enumerated() {
            let tail = Array(ids[head.count...])
            out.cost.tails.append((tail.count, try await laidTimed(tail)))
            out.shares.append(Judge.shares(tap.logits, wanted))
            try await loadMark(rest)
            if index + 1 < laid.count { rest = try await saveMark() }
        }
        return out
    }

    public func judge(frame: String, items: [String], options: [String],
                      label: String = "judge",
                      seed: String = "") async -> [[Double]] {
        await priming?.value
        enterEngine()
        defer { leaveEngine() }
        let began = Date()
        var out: [[Double]] = []
        var cost = JudgeCost()
        let fed = backend.fed
        _ = try? await catchUp()
        cost.caught = pending.count - ChatSession.common(fed, pending)
        cost.caughtSeconds = Date().timeIntervalSince(began)
        let position = await backend.position
        cost.kv = position
        if position > 0, let save = try? await saveMark() {
            let outer = aside
            var asked = current
            asked.enableThinking = false
            aside = asked
            let rendering = Date()
            var laid: [[Int32]] = []
            for item in items {
                let ask = frame.replacingOccurrences(of: "{item}", with: item)
                let made = await judgeIds(ask)
                cost.path = made.path
                if let ids = made.ids {
                    laid.append(ids + (seed.isEmpty ? []
                                       : backend.encode(seed)))
                }
            }
            cost.renderSeconds = Date().timeIntervalSince(rendering)
            aside = outer
            if laid.count == items.count, !laid.isEmpty {
                let helper = Judge(backend: backend, template: template,
                                   vocabSize: vocabSize)
                let wanted = options.map { option in helper.ids(option) }
                let tap = LogitTap()
                var sampler = Sampler(vocabSize: vocabSize, config: .greedy)
                sampler.logitMask = { logits in tap.keep(logits) }
                await backend.useSampler(sampler)
                let got = try? await judgeShares(laid, wanted, tap, cost)
                out = got?.shares ?? []
                cost = got?.cost ?? cost
                await backend.useSampler(current.turnSampler)
            }
            try? await loadMark(save)
        }
        Diag.shared.report(.turn, String(
            format: "[judge] %@ %d item(s) %@, %.2fs", label, items.count,
            cost.line, Date().timeIntervalSince(began)))
        return out
    }

    public func extractNotes(_ instruction: String) async -> String {
        await oneShot(instruction, stopAfter: 2400, "extract")
    }

    public func conclude(_ instruction: String) async -> String {
        await oneShot(instruction, stopAfter: 320, "conclude")
    }

    public func makeTitle() async -> String {
        let raw = await oneShot(ChatSession.titleInstruction, stopAfter: 160,
                                "makeTitle")
        return ChatSession.cleanTitle(raw)
    }

    public func makeFollowup() async -> String {
        let raw = await oneShot(ChatSession.followUpInstruction,
                                stopAfter: 280, "makeFollowup")
        let hint = ChatSession.cleanFollowup(raw)
        return alreadyAsked(hint) ? "" : hint
    }

    private func alreadyAsked(_ hint: String) -> Bool {
        let candidate = ChatSession.bagOfWords(hint)
        var repeated = candidate.isEmpty
        for m in history where m.role == "user" && !repeated {
            let prior = ChatSession.bagOfWords(m.content)
            let union = candidate.union(prior).count
            repeated = union > 0
                && Double(candidate.intersection(prior).count)
                    / Double(union) >= ChatSession.echoOverlap
        }
        return repeated
    }

    static let echoOverlap = 0.6

    static func bagOfWords(_ s: String) -> Set<String> {
        Set(s.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init))
    }

    static let titleInstruction: String = {
        let text = Flags.value("title-instruction") ?? ""
        return text.isEmpty ? defaultTitleInstruction : text
    }()

    static let defaultTitleInstruction =
        "Give this conversation a title of two to four words that names "
        + "the subject it is about. Plain words on one line. Reply with "
        + "the title only."

    static func withoutMarkup(_ raw: String) -> String {
        var out = ""
        var rest = Substring(raw)
        while let lt = rest.firstIndex(of: "<") {
            let tail = rest[lt...]
            if let gt = tail.firstIndex(of: ">") {
                out += rest[..<lt]
                rest = tail[tail.index(after: gt)...]
            } else {
                out += rest
                rest = rest[rest.endIndex...]
            }
        }
        return out + rest
    }

    static func cleanTitle(_ raw: String) -> String {
        let line = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .first.map(String.init) ?? ""
        let trimmed = line.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`.*_#"))
        var body = trimmed
        if let r = body.range(of: "title:",
                              options: [.caseInsensitive, .anchored]) {
            body = String(body[r.upperBound...])
                .trimmingCharacters(in: .whitespaces)
        }
        var words: [String] = []
        var chars = 0
        for word in body.split(separator: " ").prefix(6)
        where chars + word.count <= 40 {
            words.append(String(word))
            chars += word.count + 1
        }
        let prose = words.contains { word in
            word.contains(where: { c in c.isLetter })
        }
        let deliberating = ChatSession.deliberation.contains { phrase in
            body.lowercased().contains(phrase)
        }
        let usable = prose && words.count > 1 && !deliberating
        return usable ? words.joined(separator: " ") : ""
    }

    static let deliberation = [
        "the user wants", "the user is asking", "the user would",
        "the conversation is about", "short title for", "thinking process",
    ]

    static let followUpInstruction: String = {
        let text = Flags.value("followup-instruction") ?? ""
        return text.isEmpty ? defaultFollowUpInstruction : text
    }()

    static let defaultFollowUpInstruction =
        "Pick one specific detail from your last answer that the user would "
        + "most want to know more about, and write the question they would "
        + "type to ask about it. One sentence ending in a question mark. "
        + "Never repeat a question already asked. Output only the question."

    static let followupMin = 12
    static let followupMax = 140

    static func cleanFollowup(_ raw: String) -> String {
        let lines = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .map { line in line.trimmingCharacters(in: .whitespaces) }
            .filter { line in !line.isEmpty }
        let asked = lines.filter { line in line.contains("?") }
        var out = ""
        if asked.count == 1, let line = asked.first {
            out = ChatSession.oneQuestion(line)
        }
        return out
    }

    private static let assistantVoice = [
        "would you like", "do you want", "shall i", "can i help",
        "is there anything", "let me know", "should i", "like me to",
        "your last answer",
    ]

    private static func oneQuestion(_ line: String) -> String {
        var body = line
        if let mark = body.firstIndex(of: "?") {
            body = String(body[...mark])
        }
        if let colon = body.range(of: ": "),
           !body[..<colon.lowerBound].contains("?"),
           body.distance(from: body.startIndex, to: colon.lowerBound) <= 40 {
            body = String(body[colon.upperBound...])
        }
        body = body.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`*_-#>0123456789."))
        let latex = body.contains(where: { c in "{}\\".contains(c) })
        for tex in ["{", "}", "\\"] + (latex ? ["$"] : []) {
            body = body.replacingOccurrences(of: tex, with: "")
        }
        let lower = body.lowercased()
        let markup = body.contains(where: { c in "{}<>".contains(c) })
            || body.contains("**")
        let aboutTheUser = lower.hasPrefix("the user")
            || lower.hasPrefix("user ")
        let assistant = ChatSession.assistantVoice.contains { phrase in
            lower.contains(phrase)
        }
        let deliberating = ChatSession.deliberation.contains { phrase in
            lower.contains(phrase)
        }
        let usable = !aboutTheUser && !assistant && !markup && !deliberating
            && body.hasSuffix("?")
            && body.count >= ChatSession.followupMin
            && body.count <= ChatSession.followupMax
        return usable ? body : ""
    }

    private func runTurn(_ user: String, stoppable: String?,
                         onReasoning: (@Sendable (String) -> Void)?,
                         onTool: (@Sendable (String) -> Void)?,
                         onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
                         _ yield: @Sendable (String) -> Void) async {
        if await enterClaimed() {
            let full = await enterTurn(adding: backend.encode(
                ChatSession.folded(unanswered, user)).count)
            current.stoppable = stoppable
            let asked = ChatSession.folded(unanswered, user)
            unanswered = ""
            ReadBudget.shared.setQuestion(asked)
            trace(.user, ctx: await backend.position,
                  summary: String(asked.prefix(80)), text: asked)
            history.append(AgentMessage(role: "user", content: asked))
            await installTurnSampler(vision: visionContext)
            await runSeed(full: full, soft: [], onReasoning: onReasoning,
                          onTool: onTool, onToolRound: onToolRound, yield)
            await settle()
        }
        leaveEngine()
    }

    private func runSoftTurn(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool,
        onReasoning: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        if await enterClaimed() {
            let rows = spans.reduce(0) { sum, span in sum + span.rows }
            let full = await enterTurn(
                adding: backend.encode(user).count + rows)
            softTurns += 1
            ReadBudget.shared.setQuestion(user)
            trace(.user, ctx: await backend.position,
                  summary: String(user.prefix(80))
                      + " [\(spans.count) attachment(s), \(rows) soft]",
                  text: user)
            history.append(AgentMessage(
                role: "user", content: user,
                contentParts: numbered(parts, labelled)))
            await installTurnSampler(vision: true)
            await runSeed(full: full, soft: spans, onReasoning: onReasoning,
                          onTool: nil, onToolRound: onToolRound, yield)
            await settle()
        }
        leaveEngine()
    }

    private func runSeed(
        full: String?, soft: [SoftSpan],
        onReasoning: (@Sendable (String) -> Void)?,
        onTool: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        if let starved = full ?? chatGuard?() {
            await starve(starved)
        } else {
            await runRounds(soft: soft, onReasoning: onReasoning,
                            onTool: onTool, onToolRound: onToolRound, yield)
        }
    }

    private func starve(_ why: String) async {
        trace(.inject, summary: "chat starved: " + why)
        current.lastMetrics = TurnMetrics(
            ctx: await backend.position, thinkTokens: 0, contentTokens: 0,
            endReason: ChatSession.memoryEnd)
        await rollbackTurn(why: .memory)
    }

    static func emptyOutcome(_ endReason: String,
                             _ readFraction: Double) -> TurnOutcome {
        let out: TurnOutcome
        if endReason == ChatSession.memoryEnd {
            out = .memory
        } else if ChatSession.userStopped(endReason) && readFraction == 1 {
            out = .stopped
        } else {
            out = .answerless
        }
        return out
    }

    private func runRounds(
        soft: [SoftSpan],
        onReasoning: (@Sendable (String) -> Void)?,
        onTool: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        let first = await seedOnce(soft: soft)
        if first.stopped {
            await rollbackTurn()
        } else {
            var pp = first.pp
            var tally = (think: 0, content: 0)
            var round = 0
            var pagedURL: String? = nil
            var paged = 0
            var spent: [String: String] = [:]
            var fruitless = 0
            var step = await decodeStep(seed: first.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
            tally = (current.lastMetrics.thinkTokens,
                     current.lastMetrics.contentTokens)
            while let call = step.pending, round < ChatSession.maxToolRounds,
                  fruitless < ChatSession.maxFruitless {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                let url = resolved == "fetch_url"
                    ? call.params.first { p in
                        ["url", "link", "u"].contains(p.name)
                    }?.value
                    : nil
                if url != nil && url == pagedURL && paged < 8 {
                    paged += 1
                } else {
                    round += 1
                    paged = 0
                }
                pagedURL = url
                onTool?(call.functionName)
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: nil))
                trace(.toolCall, summary: call.functionName
                          + (resolved == call.functionName || resolved == nil
                             ? "" : " -> \(resolved!)"),
                      text: wire.toolCallOpen + call.rawBlock
                          + wire.toolCallClose)
                toolLog("tool call \(round): \(call.functionName) "
                    + call.params.map { "\($0.name)=\($0.value)" }
                        .joined(separator: " ")
                    + " | raw="
                    + call.rawBlock.replacingOccurrences(of: "\n", with: "\\n"))
                let toolT0 = Date()
                let signature = ChatSession.callSignature(
                    resolved ?? call.functionName,
                    sanitizedArgs(call, resolved))
                var result = spent[signature].map { prior in
                    ChatSession.repeatedCall(prior)
                } ?? ""
                if result.isEmpty, current.quickTurn {
                    result = ChatSession.quickSkip
                    toolLog("tool call \(round): skipped for Quick Answer")
                } else if result.isEmpty {
                    result = await runTool(call, resolved: resolved)
                    if (result.hasPrefix("error")
                        && spent.values.contains(result))
                        || Tools.fruitless(result) {
                        fruitless += 1
                    }
                    spent[signature] = result
                } else {
                    fruitless += 1
                    toolLog("tool call \(round): REFUSED as an exact repeat")
                }
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: result))
                trace(.toolResult, from: toolT0,
                      summary: "\(resolved ?? call.functionName) -> "
                          + "\(result.count) chars",
                      text: result)
                toolLog("tool result \(round): \(result.count) chars: "
                    + result.prefix(160))
                let cont = await toolContinuation(
                    call: call, resolved: resolved,
                    preamble: step.preamble, result: result)
                if cont.pp > 0 { pp = cont.pp }
                step = await decodeStep(seed: cont.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
                tally = (current.lastMetrics.thinkTokens,
                         current.lastMetrics.contentTokens)
            }
            if let call = step.pending {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                onToolRound?(ToolRoundEvent(
                    round: round + 1, name: call.functionName,
                    resolved: resolved, params: call.params,
                    result: ChatSession.toolBudgetNudge))
                trace(.toolResult, summary: "tool budget spent -> nudge",
                      text: ChatSession.toolBudgetNudge)
                let cont = await toolContinuation(
                    call: call, resolved: resolved, preamble: step.preamble,
                    result: ChatSession.toolBudgetNudge)
                if cont.pp > 0 { pp = cont.pp }
                step = await decodeStep(seed: cont.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
                if step.pending != nil {
                    history.append(AgentMessage(role: "assistant",
                                                content: step.preamble))
                }
            }
            let empty = history.last.map { last in
                last.role == "assistant" && last.content.isEmpty
            } ?? false
            if empty {
                await rollbackTurn(why: ChatSession.emptyOutcome(
                    current.lastMetrics.endReason, current.readFraction))
            }
        }
    }

    private func seedOnce(
        soft: [SoftSpan]
    ) async -> (seed: Int32, pp: Double, stopped: Bool) {
        let t0 = Date()
        var seed = backend.eos
        var added = 0
        var stopped = false
        do {
            let r = try await seedTurn(
                soft: soft, stoppable: soft.isEmpty ? current.stoppable : nil)
            seed = r.seed
            added = r.added
        } catch EngineError.stopped {
            stopped = true
        } catch {
            seed = backend.eos
        }
        let sec = Date().timeIntervalSince(t0)
        let pp = sec > 0 && added > 0 ? Double(added) / sec : 0
        trace(.prefill, from: t0,
              ctx: await backend.position, tokens: added,
              summary: stopped ? "stopped mid-prefill"
                               : String(format: "%.0f t/s", pp))
        return (seed, pp, stopped)
    }

    private struct Laid {
        var ids: [Int32]
        var tail = 0
        var spanStart = 0
        var soft: [SoftSpan] = []
        var chunked = true
        var reading = false
    }

    private func turnMessages() -> [AgentMessage] {
        history.count == 2 ? [history[1]] : Array(history.suffix(2))
    }

    private func seedTurn(soft: [SoftSpan], stoppable: String?,
                          decoding: Bool = true) async throws
        -> (seed: Int32, added: Int, laid: Int) {
        let fresh = committed.isEmpty
        if fresh { systemMark = 0 }
        let made = fresh ? block(history, after: nil, tools: toolSpecs)
                         : block(turnMessages(), after: [])
        let gen = decoding ? backend.encode(seeded(made.gen)) : []
        var laying = Laid(ids: backend.encode(made.closed), soft: soft,
                          chunked: soft.isEmpty)
        let span = stoppable.map { text in
            text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let span, !span.isEmpty, let at = made.closed.range(of: span) {
            let read = backend.encode(String(made.closed[..<at.upperBound]))
            let tail = backend.encode(String(made.closed[at.upperBound...]))
            laying.ids = read + tail
            laying.tail = tail.count
            laying.reading = true
            laying.spanStart = backend.encode(
                String(made.closed[..<at.lowerBound])).count
        }
        if !soft.isEmpty {
            laying.ids = Continuation.expandSpans(laying.ids, soft)
        }
        let opens = userStart(made.closed, laying.ids)
        let keep = committed.count
        let fed = try await feed(
            laying, gen: gen, site: "seed",
            shown: (fresh ? "fresh (system + tools + user)"
                           : "delta (prev answer + user)",
                     made.closed + made.gen))
        if let opens {
            starts.append(Start(at: keep + opens, message: history.count - 1,
                                tools: toolTokens))
        }
        return fed
    }

    private func userStart(_ closed: String, _ ids: [Int32]) -> Int? {
        var out: Int? = nil
        if !wire.turnOpen.isEmpty,
           let at = closed.range(of: wire.turnOpen, options: .backwards) {
            let head = backend.encode(String(closed[..<at.lowerBound]))
            if ids.starts(with: head) { out = head.count }
        }
        return out
    }

    static let replyRoom = 2048

    private func shift(adding need: Int) async -> String? {
        let room = min(context / 4, ChatSession.replyRoom)
        let kept = starts.filter { s in s.at <= committed.count }
        let over = context > 0 && pending.count + need + room > context
        let full = over && shiftMode == .off && !kept.isEmpty
            ? "the conversation reached the context of \(context) tokens"
            : nil
        if let full {
            Diag.shared.report(.turn, "[chat] cut: " + full)
        } else if over, kept.count > 1 {
            await dropOldest(kept, need: need, room: room)
        }
        return full
    }

    private func dropOldest(_ kept: [Start], need: Int, room: Int) async {
        let goal = context - context / 4
        var to = 1
        while to + 1 < kept.count
              && pending.count - (kept[to].at - kept[0].at) + need + room
                  > goal {
            to += 1
        }
        let t0 = Date()
        let soft = await softCut(from: kept[0], to: kept[to])
        drop(from: kept[0], to: kept[to])
        if soft {
            checkSoft()
        } else {
            await relayKept()
        }
        Diag.shared.report(.turn, String(
            format: "[shift] %@ dropped %d messages, %d tokens, %d kept "
                + "of %d in %.0f ms", soft ? "soft" : "exact",
            kept[to].message - kept[0].message, kept[to].at - kept[0].at,
            committed.count, context, Date().timeIntervalSince(t0) * 1000))
    }

    private func softCut(from first: Start, to last: Start) async -> Bool {
        var moved = false
        if shiftMode == .soft, backend.fed.starts(with: committed) {
            moved = await backend.shift(keep: first.at,
                                        count: last.at - first.at)
        }
        return moved
    }

    private func checkSoft() {
        let fed = backend.fed
        let mark = backend.markedAt
        if mark != committed.count || !fed.starts(with: committed) {
            ChatSession.seamFaults.add(
                "soft shift fed \(fed.count) ids, mark \(mark ?? -1), "
                + "committed \(committed.count)")
        }
    }

    private func drop(from first: Start, to last: Start) {
        let count = last.at - first.at
        let messages = last.message - first.message
        let tools = last.tools - first.tools
        committed.removeSubrange(first.at ..< last.at)
        history.removeSubrange(first.message ..< last.message)
        toolTokens -= tools
        starts = starts.filter { s in s.at >= last.at }.map { s in
            Start(at: s.at - count, message: s.message - messages,
                  tools: s.tools - tools)
        }
        pending = committed
        trace(.inject, summary: "context full: dropped \(messages) "
              + "messages, \(count) tokens")
    }

    private func relayKept() async {
        do {
            try await layKept()
        } catch {
            Diag.shared.report("[shift] the kept exchanges did not lay "
                               + "again: \(error)")
        }
    }

    private func layKept() async throws {
        try await restore()
        let from = backend.fed.count
        if from < committed.count {
            _ = try await lay(Array(committed[from...]))
        }
        try await markReply()
    }

    private func feed(_ laying: Laid, gen: [Int32], site: String,
                      shown: (summary: String, text: String)) async throws
        -> (seed: Int32, added: Int, laid: Int) {
        let keep = committed.count
        pending = committed + laying.ids + gen
        try await restore()
        trace(.render, tokens: laying.ids.count + gen.count,
              summary: shown.summary, text: shown.text)
        let from = backend.fed.count
        var next = backend.eos
        if from < keep {
            next = try await backend.extend(Array(pending[from ..< keep]))
        }
        let start = max(from, keep)
        var laid: [Int32] = []
        if start < keep + laying.ids.count {
            let got = try await layBlock(laying, skip: start - keep)
            next = got.next
            laid = got.laid
        }
        committed = Array(pending.prefix(start)) + laid
        try await markReply()
        pending = committed
        var seed = next
        if !gen.isEmpty {
            seed = laying.chunked && current.readFraction < 1
                ? try await lay(gen) : try await backend.extend(gen)
        }
        pending += gen
        checkFed(pending, site)
        return (seed, max(0, keep - from) + laid.count + gen.count, laid.count)
    }

    private func layBlock(_ laying: Laid, skip: Int) async throws
        -> (next: Int32, laid: [Int32]) {
        let ids = Array(laying.ids.dropFirst(skip))
        let tail = min(laying.tail, ids.count)
        var out = (next: backend.eos, laid: ids)
        if !laying.soft.isEmpty {
            out.next = try await backend.extendSoft(ids, spans: laying.soft)
        } else if laying.chunked {
            out = try await extendChunked(
                Array(ids.dropLast(tail)), tail: Array(ids.suffix(tail)),
                spanStart: max(0, laying.spanStart - skip),
                reading: laying.reading)
        } else {
            out.next = try await backend.extend(ids)
        }
        return out
    }

    public struct ReplayTurn: Sendable {
        public let prompt: String
        public let answer: String
        public let parts: [ContentPart]
        public let spans: [SoftSpan]
        public let labelled: Bool

        public init(prompt: String, answer: String, parts: [ContentPart],
                    spans: [SoftSpan], labelled: Bool) {
            self.prompt = prompt
            self.answer = answer
            self.parts = parts
            self.spans = spans
            self.labelled = labelled
        }
    }

    public func replay(_ turns: [ReplayTurn]) async throws -> Int {
        await priming?.value
        enterEngine()
        defer { leaveEngine() }
        let t0 = Date()
        var laid = 0
        var i = 0
        var refused: String? = nil
        while i < turns.count && !Task.isCancelled && refused == nil {
            let turn = turns[i]
            let need = backend.encode(turn.prompt + turn.answer).count
                + turn.spans.reduce(0) { sum, span in sum + span.rows }
            refused = await shift(adding: need) ?? chatGuard?()
            if refused == nil {
                laid += try await replayed(turn)
                i += 1
            }
        }
        if let refused {
            Diag.shared.report(.turn, "[replay] refused at turn \(i + 1) of "
                + "\(turns.count): " + refused)
            throw EngineError.contextFull
        } else if i < turns.count {
            throw CancellationError()
        }
        let ctx = await backend.position
        current.lastMetrics = TurnMetrics(ctx: ctx, thinkTokens: 0,
                                          contentTokens: 0)
        trace(.prefill, from: t0, ctx: ctx, tokens: laid,
              summary: String(format: "replayed %d turn(s) (%.1fs)",
                              turns.count, Date().timeIntervalSince(t0)))
        return laid
    }

    private func replayed(_ turn: ReplayTurn) async throws -> Int {
        let soft = !turn.spans.isEmpty
        if soft { softTurns += 1 }
        history.append(AgentMessage(
            role: "user", content: turn.prompt,
            contentParts: soft ? numbered(turn.parts, turn.labelled) : nil))
        let laid = try await seedTurn(soft: turn.spans, stoppable: nil,
                                      decoding: false).laid
        history.append(AgentMessage(role: "assistant", content: turn.answer))
        return laid
    }

    static let prefillChunk = 1024

    static func readingStopped(_ fraction: Double, _ why: String) -> String {
        let cause = why == "memory"
            ? "because the device ran out of memory" : "by the user"
        return "\n[Reading stopped \(cause) at \(Int(fraction * 100)) "
            + "percent of the attached text; answer from what was read.]\n"
    }

    private func extendChunked(_ ids: [Int32], tail: [Int32],
                               spanStart: Int, reading: Bool) async throws
        -> (next: Int32, laid: [Int32]) {
        var next = backend.eos
        var laid = ids
        if ids.count <= ChatSession.prefillChunk && !reading {
            next = try await backend.extend(ids)
        } else {
            let startCtx = await backend.position
            let total = ids.count + tail.count
            current.lastMetrics = TurnMetrics(
                ctx: startCtx, thinkTokens: 0, contentTokens: 0,
                prefillDone: 0, prefillTotal: total)
            var done = 0
            var rest = tail
            var cutAt: Int? = nil
            while done < ids.count && cutAt == nil {
                let starved = done > 0 ? readGuard?() : nil
                if done > 0 && backend.shouldStop() {
                    cutAt = done
                    current.readStop = "user"
                } else if let starved {
                    cutAt = done
                    current.readStop = "memory"
                    trace(.inject, summary: "reading starved: " + starved)
                } else {
                    let end = min(ids.count, done + ChatSession.prefillChunk)
                    let from = done
                    let chunkStart = Date()
                    do {
                        next = try await backend.extend(Array(ids[done..<end]))
                        done = end
                    } catch EngineError.stopped {
                        cutAt = await backend.position - startCtx
                        current.readStop = "user"
                    }
                    let sec = Date().timeIntervalSince(chunkStart)
                    let at = cutAt ?? done
                    current.lastMetrics = TurnMetrics(
                        ctx: startCtx + at, thinkTokens: 0, contentTokens: 0,
                        pp: sec > 0 ? Double(at - from) / sec : 0,
                        prefillDone: at, prefillTotal: total)
                }
                if cutAt != nil && !reading { throw EngineError.stopped }
            }
            if let at = cutAt {
                let span = max(1, ids.count - spanStart)
                current.readFraction = min(1, Double(max(0, at - spanStart))
                                      / Double(span))
                laid = Array(ids[..<at])
                rest = backend.encode(
                    ChatSession.readingStopped(current.readFraction,
                                               current.readStop)) + tail
                trace(.inject, summary: "reading stopped at \(at) of "
                      + "\(ids.count) tokens")
            }
            if !rest.isEmpty {
                next = try await lay(rest)
                laid += rest
            }
        }
        return (next, laid)
    }

    private func lay(_ ids: [Int32]) async throws -> Int32 {
        var next = backend.eos
        var rest = ids
        while !rest.isEmpty {
            let step = try await layStep(rest)
            next = step.next
            rest = step.rest
        }
        return next
    }

    private func layStep(_ ids: [Int32]) async throws
        -> (next: Int32, rest: [Int32]) {
        let before = await backend.position
        var out = (next: backend.eos, rest: [Int32]())
        do {
            out.next = try await backend.extend(ids)
        } catch EngineError.stopped {
            let laid = await backend.position - before
            if laid <= 0 { throw EngineError.stopped }
            out.rest = Array(ids.dropFirst(laid))
        }
        return out
    }

    private var attachmentCounts: [String: Int] = [:]

    private static func noun(_ part: ContentPart) -> String? {
        switch part {
        case .image: "Picture"
        case .audio: "Audio"
        case .video: "Video"
        case .text: nil
        }
    }

    private func numbered(_ parts: [ContentPart],
                          _ labelled: Bool = true) -> [ContentPart] {
        var out: [ContentPart] = []
        for part in parts {
            if let noun = ChatSession.noun(part), labelled {
                let n = (attachmentCounts[noun] ?? 0) + 1
                attachmentCounts[noun] = n
                out.append(.text("\(noun) \(n): "))
            }
            out.append(part)
        }
        return out
    }

    static func userStopped(_ endReason: String) -> Bool {
        endReason == "cancelled" || endReason == "stop"
    }

    private func enterTurn(adding need: Int) async -> String? {
        current.open()
        current.runner?.beginTurn()
        let fed = backend.fed
        if !fed.starts(with: pending) {
            ChatSession.seamFaults.add(
                "open fed \(fed.count) ids, pending \(pending.count)")
        }
        let full = await shift(adding: need)
        await markUser()
        return full
    }

    private var unanswered = ""

    static func folded(_ earlier: String, _ user: String) -> String {
        earlier.isEmpty ? user
            : "[My earlier message, which got no answer:]\n" + earlier
                + "\n\n" + user
    }

    private func rollbackTurn(why: TurnOutcome = .stopped) async {
        if let mark = userMark {
            if why == .answerless,
               let asked = history.last(where: { m in m.role == "user" }),
               history.count > mark.messages {
                unanswered = asked.content
            }
            try? await loadMark(mark)
            pending = mark.ids
            committed = Array(mark.ids.prefix(backend.markedAt ?? 0))
            history = Array(history.prefix(mark.messages))
            toolTokens = mark.tools
            starts = starts.filter { s in s.message < history.count }
        }
        current.turnOutcome = why
        trace(.rewind, ctx: await backend.position,
              summary: "turn rollback (\(why.rawValue))")
    }

    private static let greedyConfig: SamplerConfig = {
        var c = SamplerConfig.default
        c.temperature = 0
        return c
    }()

    private func installTurnSampler(vision: Bool) async {
        let reasons = current.enableThinking && !current.suppressReasoning
        var turnConfig = current.metaTurn
            ? ChatSession.greedyConfig
            : presets.select(thinking: reasons, vision: vision)
        if !turnConfig.setMask.contains(.dryMultiplier) {
            turnConfig.dryMultiplier = ChatSession.dryMultiplier
        }
        if samplerSeed != 0 { turnConfig.seed = samplerSeed }
        var sampler = Sampler(vocabSize: vocabSize, config: turnConfig)
        sampler.penaltyExempt = wireTokens
        sampler.banned = bannedControl
        if !reasons { sampler.banned.formUnion(reasoningOpeners) }
        if turnConfig.dryMultiplier > 0 {
            sampler.dryBreakers = sequenceBreakers()
        }
        if overthinkLambda != 0 && reasons && !overthinkTokens.isEmpty {
            sampler.overthinkTokens = overthinkTokens
            sampler.overthinkLambda = overthinkLambda
        }
        if let tap = logitTap {
            sampler.logitMask = { logits in tap(logits) }
        }
        current.turnSampler = sampler
        if let gate = ensureGate() { gate.disarm() }
        await backend.useSampler(sampler)
    }

    static let breakerBytes = Set("\n:\"*|".utf8)

    private var breakers: Set<Int32>?

    public static func sequenceBreakers(_ backend: any AgentBackend,
                                        vocabSize: Int) -> Set<Int32> {
        var out: Set<Int32> = []
        for id in 0 ..< Int32(vocabSize)
        where backend.tokenBytes(id).contains(where: { byte in
            ChatSession.breakerBytes.contains(byte)
        }) {
            out.insert(id)
        }
        return out
    }

    public static func grammarVocab(_ backend: any AgentBackend,
                                    vocabSize: Int,
                                    template: String) -> GrammarVocab? {
        var out: GrammarVocab? = nil
        if ChatSession.grammarMode != .off, template.contains("<function=") {
            var toks: [[UInt8]] = []
            toks.reserveCapacity(vocabSize)
            for id in 0 ..< vocabSize {
                toks.append(backend.tokenBytes(Int32(id)))
            }
            out = GrammarVocab(toks)
        }
        return out
    }

    private func sequenceBreakers() -> Set<Int32> {
        if breakers == nil {
            let began = Date()
            let out = ChatSession.sequenceBreakers(backend,
                                                   vocabSize: vocabSize)
            breakers = out
            Diag.shared.report(.perf, String(
                format: "[dry] %d sequence breakers of %d tokens in %.0f ms",
                out.count, vocabSize,
                Date().timeIntervalSince(began) * 1000))
        }
        return breakers ?? []
    }

    private func installSampler(masked: Bool, verbatim: Bool) async {
        if var sampler = current.turnSampler {
            if masked, let gate {
                sampler.logitMask = { logits in gate.mask(&logits) }
            }
            sampler.verbatim = verbatim
            await backend.useSampler(sampler)
        }
    }

    private func ensureGate() -> GrammarGate? {
        var result: GrammarGate? = nil
        if ChatSession.grammarMode != .off && toolDialectXML {
            if grammarVocab == nil {
                grammarVocab = ChatSession.grammarVocab(
                    backend, vocabSize: vocabSize, template: template)
            }
            if let vocab = grammarVocab {
                let g = gate ?? GrammarGate(
                    vocab: vocab,
                    structuralOnly: ChatSession.grammarMode == .structural)
                gate = g
                result = g
            }
        }
        return result
    }

    private func advanceGrammar(_ gate: GrammarGate, _ token: Int32,
                                _ bytes: [UInt8]) {
        if gate.armed {
            gate.grammar?.advance(backend.tokenBytes(token))
            if gate.grammar?.dead ?? true { gate.grammar = nil }
        } else if let off = stillOpen(bytes) {
            let g = Grammar.toolCall()
            g.advance(Array(bytes[off...]))
            gate.grammar = g
        }
    }

    private let openTag: [UInt8]
    private let closeTag: [UInt8]
    private let thinkOpen: [UInt8]
    private let thinkClose: [UInt8]
    // No trailing '>': a spaced "</function >" is still a complete body. A
    // repair for the Qwen XML dialect specifically, not part of any wire.
    private static let closeFn = Array("</function".utf8)

    enum LeadingThink { case pending, isThink(Int), notThink }

    static func leadingThink(_ b: [UInt8], _ open: [UInt8]) -> LeadingThink {
        var i = 0
        while i < b.count && ChatSession.isBlank(b[i]) { i += 1 }
        var j = 0
        while j < open.count && i < b.count
              && (b[i] == open[j]
                  || (j > 0 && open[j - 1] == 0x3E && ChatSession.isBlank(b[i]))) {
            if b[i] == open[j] { j += 1 }
            i += 1
        }
        let result: LeadingThink
        if j == open.count {
            result = .isThink(i)
        } else if i == b.count {
            result = .pending
        } else {
            result = .notThink
        }
        return result
    }

    private static func isBlank(_ c: UInt8) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D
    }

    private static func partialSuffix(_ b: [UInt8], _ pat: [UInt8],
                                      _ n: Int) -> Int {
        var hold = 0
        var len = min(pat.count - 1, n)
        while hold == 0 && len > 0 {
            var j = 0
            while j < len && b[n - len + j] == pat[j] { j += 1 }
            if j == len { hold = len }
            len -= 1
        }
        return hold
    }
    private func stillOpen(_ b: [UInt8]) -> Int? {
        var result: Int? = nil
        if let open = ChatSession.lastIndex(b, openTag),
           ChatSession.index(b, closeTag, open + openTag.count) == nil {
            result = open
        }
        return result
    }

    private static func index(_ b: [UInt8], _ pat: [UInt8],
                              _ from: Int) -> Int? {
        var result: Int? = nil
        var i = max(from, 0)
        let last = b.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private static func lastIndex(_ b: [UInt8], _ pat: [UInt8]) -> Int? {
        var result: Int? = nil
        var i = b.count - pat.count
        while result == nil && i >= 0 {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i -= 1
        }
        return result
    }

    private func decodeStep(
        seed: Int32, pp: Double, tally: (think: Int, content: Int),
        onReasoning: (@Sendable (String) -> Void)?,
        _ yield: @Sendable (String) -> Void)
        async -> (pending: ToolCall?, preamble: String) {
        let toolsActive = current.runner != nil && !toolSpecs.isEmpty
        // A completed call parks the grammar in its absorbing match state;
        // armed into the next step it would disable spec decode all turn.
        if let gate, gate.armed {
            gate.disarm()
            await installSampler(masked: false, verbatim: false)
        }
        var pending: ToolCall? = nil
        let startCtx = await backend.position
        var cur = seed
        var ids: [Int32] = []
        var bytes: [UInt8] = []
        var emitted = 0
        var closeAt: Int? = nil
        let startsInThink = current.enableThinking && genStartsThink
        var inThinkRegion = startsInThink
        var thinkDecided = startsInThink
        var openerPending = startsInThink
        var wsDone = !startsInThink
        var thinkSearch = 0
        var toolAt: Int? = nil
        // Outlives toolAt being cleared by the quoted-pair recovery below.
        var sawToolOpen = false
        var toolSearch = 0
        var toolCloseSearch = 0
        var toolReopenSearch = 0
        var think = 0
        var content = 0
        var undecided = 0
        var steps = 0
        var fed = startCtx
        var stop = false
        var starved: String? = nil
        var thinkRescues = 0
        let g0 = Date()
        var steady: Date? = nil
        let gpu0 = backend.gpuSeconds
        current.lastMetrics = TurnMetrics(
            ctx: startCtx, thinkTokens: tally.think,
            contentTokens: tally.content, pp: pp, tg: 0,
            readFraction: current.readFraction, readStop: current.readStop)
        while !backend.eosIds.contains(cur) && !stop && !Task.isCancelled
              && !backend.shouldStop() {
            ids.append(cur)
            bytes.append(contentsOf: backend.tokenBytes(cur))
            let wasArmed = gate?.armed ?? false
            let wasOpen = toolAt != nil
            if toolsActive, let gate { advanceGrammar(gate, cur, bytes) }
            if !thinkDecided {
                switch ChatSession.leadingThink(bytes, thinkOpen) {
                case .isThink(let past):
                    inThinkRegion = true
                    think += undecided
                    undecided = 0
                    emitted = past
                    wsDone = false          // skip the ws after its </think>
                    thinkDecided = true
                case .notThink:
                    thinkDecided = true
                    content += undecided
                    undecided = 0
                case .pending:
                    break
                }
            }
            if openerPending {
                switch ChatSession.leadingThink(bytes, thinkOpen) {
                case .isThink(let past):
                    emitted = past
                    openerPending = false
                case .notThink:
                    openerPending = false
                case .pending:
                    break
                }
            }
            let wasReasoning = inThinkRegion && closeAt == nil
            if inThinkRegion && closeAt == nil {
                closeAt = ChatSession.index(bytes, thinkClose,
                                            thinkSearch)
                if closeAt == nil {
                    thinkSearch = max(thinkSearch,
                        bytes.count - thinkClose.count + 1)
                }
            }
            let inThink = inThinkRegion && closeAt == nil
            if !thinkDecided {
                undecided += 1
            } else if inThink {
                think += 1
            } else {
                content += 1
            }
            let n = Tokenizer.completeUTF8Count(bytes)
            if toolsActive && toolAt == nil {
                toolAt = ChatSession.index(
                    bytes, openTag, toolSearch)
                sawToolOpen = sawToolOpen || toolAt != nil
                if toolAt == nil {
                    toolSearch = max(toolSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let openHold = toolsActive && toolAt == nil
                ? ChatSession.partialSuffix(bytes, openTag, n)
                : 0
            if inThink {
                let hold = max(openHold, ChatSession.partialSuffix(
                    bytes, thinkClose, n))
                let end = min(toolAt ?? Int.max, n - hold)
                if end > emitted, !openerPending {
                    onReasoning?(String(decoding: bytes[emitted ..< end],
                                        as: UTF8.self))
                    emitted = end
                }
            } else if thinkDecided {
                if let at = closeAt, emitted <= at {
                    let flush = min(at, toolAt ?? Int.max)
                    if flush > emitted {
                        onReasoning?(String(decoding: bytes[emitted ..< flush],
                                            as: UTF8.self))
                    }
                    emitted = at + thinkClose.count
                }
                while !wsDone && emitted < n {
                    let b = bytes[emitted]
                    if b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D {
                        emitted += 1
                    } else {
                        wsDone = true
                    }
                }
                if wsDone {
                    var hold = openHold
                    var markAt = Int.max
                    var markLen = 0
                    // A close with no opener ANYWHERE this round closes
                    // nothing; a quoted pair keeps its close, one preceded it.
                    var marks = [thinkClose, thinkOpen]
                    if toolsActive {
                        if !sawToolOpen { marks.append(closeTag) }
                    } else {
                        marks.append(openTag)
                        marks.append(closeTag)
                    }
                    for pat in marks where !pat.isEmpty {
                        hold = max(hold,
                                   ChatSession.partialSuffix(bytes, pat, n))
                        if let at = ChatSession.index(bytes, pat, emitted),
                           at < markAt {
                            markAt = at
                            markLen = pat.count
                        }
                    }
                    let end = min(toolAt ?? Int.max, n - hold)
                    let upto = min(end, markAt)
                    if upto > emitted {
                        yield(String(decoding: bytes[emitted ..< upto],
                                     as: UTF8.self))
                        emitted = upto
                    }
                    if markAt == emitted, markAt + markLen <= n {
                        if markLen == closeTag.count, !sawToolOpen {
                            Diag.shared.report("suppressed a stray close tag")
                        }
                        emitted = markAt + markLen
                    }
                }
            }
            steps += 1
            if steady == nil { steady = Date() }
            if steps % 8 == 0 {
                current.lastMetrics = TurnMetrics(
                    ctx: startCtx + steps, thinkTokens: tally.think + think,
                    contentTokens: tally.content + content, pp: pp,
                    tg: ChatSession.steadyRate(steps, steady),
                    readFraction: current.readFraction,
                    readStop: current.readStop)
            }
            if steps % ChatSession.memoryEvery == 0 { starved = chatGuard?() }
            let thought = tally.think + think
            let softOver = current.softReasoningCap > 0
                && thought >= current.softReasoningCap
                && bytes.last == 0x0A
            let looping = Continuation.isLooping(ids,
                tokenBytes: { [backend] id in backend.tokenBytes(id) })
            let loopRescue = looping && inThinkRegion && closeAt == nil
                && toolAt == nil && thinkRescues == 0
            // toolAt non-nil suppresses injection: a call is mid-flight and
            // </think> must not land inside its markup.
            let overflow = inThink && toolAt == nil
                && (current.forceEndThink || current.suppressReasoning
                    || softOver || loopRescue
                    || (current.maxReasoning > 0
                        && thought >= current.maxReasoning))
            if toolsActive && pending == nil, let open = toolAt {
                if toolCloseSearch < open + openTag.count {
                    toolCloseSearch = open + openTag.count
                }
                if toolReopenSearch < open + openTag.count {
                    toolReopenSearch = open + openTag.count
                }
                let at = ChatSession.index(bytes, closeTag,
                                           toolCloseSearch)
                let reopen = ChatSession.index(bytes, openTag,
                                               toolReopenSearch)
                if let reopen, at == nil || reopen < at! {
                    pending = Tools.parse(Substring(String(
                        decoding: bytes[
                            (open + openTag.count) ..< reopen],
                        as: UTF8.self)))
                    if pending == nil {
                        toolAt = reopen
                        toolSearch = reopen
                        toolCloseSearch = reopen + openTag.count
                        toolReopenSearch = reopen + openTag.count
                    }
                } else if let at {
                    let end = at + closeTag.count
                    pending = completedCall(String(
                        decoding: bytes[open ..< end], as: UTF8.self))
                    if pending == nil {
                        toolCloseSearch = end
                        toolSearch = end
                        toolReopenSearch = end
                        toolAt = nil
                    }
                } else {
                    toolCloseSearch = max(toolCloseSearch,
                        bytes.count - closeTag.count + 1)
                    toolReopenSearch = max(toolReopenSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let armed = gate?.armed ?? false
            let draftEnded = wasReasoning && closeAt != nil
            if armed != wasArmed || (toolAt != nil) != wasOpen || draftEnded {
                await installSampler(masked: armed, verbatim: toolAt != nil)
            }
            let openRunaway = toolAt.map { at in
                bytes.count - at > ChatSession.maxOpenCallBytes
            } == true
            stop = content >= current.maxTokens
                || (looping && !loopRescue)
                || pending != nil
                || openRunaway
                || starved != nil
            if overflow && !stop {
                if await backend.queuedCount() > 0 {
                    // Spec-committed tokens the transcript has not seen yet:
                    // </think> injected now would land after them; drain first.
                    cur = (try? await backend.decode(cur)) ?? backend.eos
                    fed += 1
                } else {
                    current.forceEndThink = false
                    if loopRescue { thinkRescues = 1 }
                    // The injected close marker is found next pass, and the
                    // whitespace skip eats its newlines, so none of it streams.
                    let close = backend.encode(
                        ChatSession.budgetNote + wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (think \(think))")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                    fed += close.count
                }
            } else if !stop {
                cur = (try? await backend.decode(cur)) ?? backend.eos
                fed += 1
                if backend.eosIds.contains(cur), inThinkRegion,
                   closeAt == nil,
                   toolAt == nil, thinkRescues == 0,
                   await backend.queuedCount() == 0 {
                    thinkRescues = 1
                    let close = backend.encode(
                        wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (eos inside think)")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                    fed += close.count
                }
            }
        }
        if pending == nil, backend.eosIds.contains(cur), !Task.isCancelled,
           !backend.shouldStop(), let open = toolAt,
           ChatSession.index(bytes, ChatSession.closeFn, open) != nil {
            pending = Tools.parse(Substring(String(
                decoding: bytes[(open + openTag.count)...],
                as: UTF8.self)))
            if pending != nil {
                toolLog("eos-cut call rescued (body complete)")
            }
        }
        let whole = Tokenizer.completeUTF8Count(bytes)
        let readable = min(toolAt ?? whole, whole)
        if pending == nil, emitted < readable, thinkDecided,
           closeAt != nil || !inThinkRegion {
            let tail = String(decoding: bytes[emitted ..< readable],
                              as: UTF8.self)
            if !strippedOfToolBlocks(tail).isEmpty {
                yield(tail)
                emitted = readable
            }
        }

        let reason: String
        if pending != nil {
            reason = "tool-call"
        } else if Task.isCancelled {
            reason = "cancelled"
        } else if backend.shouldStop() {
            reason = "stop"
        } else if backend.eosIds.contains(cur) {
            reason = "eos"
        } else if let starved {
            reason = ChatSession.memoryEnd
            trace(.inject, summary: "chat starved: " + starved)
        } else if content >= current.maxTokens {
            reason = "max-tokens"
        } else if toolAt.map({ at in
            bytes.count - at > ChatSession.maxOpenCallBytes }) == true {
            reason = "tool-runaway"
        } else {
            reason = "loop-breaker"
        }
        toolLog("decode end: \(reason) after \(steps) tokens "
            + "(think \(think), content \(content))")
        var preamble = ""
        var committedAnswer: String? = nil
        if pending == nil {
            // An opener that never resolved must not enter history as prose; a
            // quoted pair the recovery released has already cleared toolAt.
            let raw = toolAt.map { at in
                String(decoding: bytes[..<at], as: UTF8.self)
            } ?? backend.text(ids)
            var answer = answerText(raw, sawThink: inThinkRegion)
            if strippedOfToolBlocks(answer).isEmpty {
                answer = ""
            } else {
                for pat in [wire.toolCallOpen, wire.toolCallClose]
                where !pat.isEmpty && answer.contains(pat) {
                    Diag.shared.report(
                        "stripped stray tool markup from an answer")
                    answer = answer.replacingOccurrences(of: pat, with: "")
                }
                answer = answer.trimmingCharacters(
                    in: .whitespacesAndNewlines)
            }
            history.append(AgentMessage(role: "assistant", content: answer))
            committedAnswer = answer
        } else {
            let decoded = backend.text(ids)
            var body = inThinkRegion ? "" : decoded
            if inThinkRegion,
               let c = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[c.upperBound...])
            }
            if let tc = body.range(of: wire.toolCallOpen) {
                body = String(body[..<tc.lowerBound])
            }
            preamble = body.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let ctx = await backend.position
        let tg = ChatSession.steadyRate(steps, steady)
        let gpuSec = backend.gpuSeconds - gpu0
        let tgGPU = gpuSec > 0 ? Double(steps) / gpuSec : 0
        current.lastMetrics = TurnMetrics(
            ctx: ctx, thinkTokens: tally.think + think,
            contentTokens: tally.content + content, pp: pp, tg: tg,
            tgGPU: tgGPU, endReason: reason, overrun: ctx - fed,
            stopToken: cur, readFraction: current.readFraction,
            readStop: current.readStop)
        trace(.decode, from: g0, ctx: ctx, tokens: steps,
              summary: reason + String(format:
                  " (think %d, content %d, tg %.1f t/s)", think, content, tg),
              text: backend.text(ids))
        if let committedAnswer {
            trace(.answer, tokens: content,
                  summary: "\(committedAnswer.count) chars",
                  text: committedAnswer)
        }
        return (pending, preamble)
    }

    static func steadyRate(_ steps: Int, _ since: Date?) -> Double {
        var out = 0.0
        if let since, steps > 1 {
            let sec = Date().timeIntervalSince(since)
            if sec > 0 { out = Double(steps - 1) / sec }
        }
        return out
    }

    private func completedCall(_ text: String) -> ToolCall? {
        var result: ToolCall? = nil
        if let body = Tools.findToolCall(in: text, from: 0,
                                         open: wire.toolCallOpen,
                                         close: wire.toolCallClose) {
            result = Tools.parse(text[body])
        }
        return result
    }

    private func toolLog(_ s: String) {
        Diag.shared.report(.tools, s)
    }

    private func sanitizedArgs(_ call: ToolCall,
                               _ resolved: String?) -> [ToolArg] {
        let known = resolved
            .flatMap { name in toolSpecs.first { s in s.name == name } }
            .map { spec in Tools.parameterNames(spec.parametersJSON) }
        var out: [ToolArg] = []
        for arg in call.params where !arg.value.isEmpty {
            if let known {
                if let canon = Tools.canonicalArgName(arg.name, known),
                   !out.contains(where: { a in a.name == canon }) {
                    out.append(ToolArg(name: canon, value: arg.value))
                }
            } else {
                out.append(arg)
            }
        }
        return out
    }

    static func callSignature(_ name: String, _ args: [ToolArg]) -> String {
        name + "\u{1}" + args.map { a in a.name + "=" + a.value }
            .sorted().joined(separator: "\u{1}")
    }

    static func repeatedCall(_ prior: String) -> String {
        "You already made this exact call. It returned: \(prior)\n"
            + "Do NOT repeat it. Change the arguments, use a different tool, "
            + "or answer the user now from what you already have."
    }

    private func runTool(_ call: ToolCall,
                         resolved: String?) async -> String {
        var result = "error: no tool current.runner"
        if let runner = current.runner {
            if let resolved {
                if let blocked = Tools.sanitize(resolved, call.params) {
                    result = blocked
                } else {
                    result = await runner.execute(resolved, call.params)
                }
            } else {
                let names = toolSpecs.map { spec in spec.name }
                result = "error: no tool named '\(call.functionName)'. "
                    + "Available tools: \(names.joined(separator: ", ")). "
                    + "If none of them fit, answer the user directly from your "
                    + "own knowledge."
            }
        }
        return result
    }

    private static let toolAliases: [String: String] = [
        "tavily_search": "web_search",
        "google_web_search": "web_search",
        "tavily_fetch": "fetch_url",
    ]

    static func resolveTool(_ name: String, _ available: [String]) -> String? {
        var result: String? = nil
        let lower = name.lowercased()
        if available.contains(name) {
            result = name
        } else if let alias = toolAliases[lower], available.contains(alias) {
            result = alias
        } else {
            var best: (name: String, dist: Int)? = nil
            for cand in available {
                let d = editDistance(lower, cand.lowercased())
                if best == nil || d < best!.dist { best = (cand, d) }
            }
            if let best, best.dist <= max(1, name.count / 4) {
                result = best.name
            }
        }
        return result
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let x = Array(a), y = Array(b)
        var result: Int
        if x.isEmpty {
            result = y.count
        } else if y.isEmpty {
            result = x.count
        } else {
            var prev = Array(0 ... y.count)
            var cur = [Int](repeating: 0, count: y.count + 1)
            for i in 1 ... x.count {
                cur[0] = i
                for j in 1 ... y.count {
                    let cost = x[i - 1] == y[j - 1] ? 0 : 1
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                                 prev[j - 1] + cost)
                }
                swap(&prev, &cur)
            }
            result = prev[y.count]
        }
        return result
    }

    private func toolContinuation(call: ToolCall, resolved: String?,
                                  preamble: String, result: String)
        async -> (seed: Int32, pp: Double) {
        let user = history.last { m in m.role == "user" }
            ?? AgentMessage(role: "user", content: "")
        let round = [
            AgentMessage(role: "assistant", content: preamble,
                         toolCalls: [AgentToolCall(
                             name: call.functionName,
                             arguments: sanitizedArgs(call, resolved))]),
            AgentMessage(role: "tool", content: result,
                         name: resolved ?? call.functionName),
        ]
        let made = block(round, after: [user])
        let genText = seeded(made.gen)
        let gen = backend.encode(genText)
        var seed = backend.eos
        var pp = 0.0
        do {
            let t0 = Date()
            let fed = try await feed(
                Laid(ids: backend.encode(made.closed), chunked: false),
                gen: gen, site: "tool round",
                shown: ("tool continuation (call + response)",
                        made.closed + genText))
            toolTokens += fed.laid
            seed = fed.seed
            let sec = Date().timeIntervalSince(t0)
            let total = fed.laid + gen.count
            pp = sec > 0 ? Double(total) / sec : 0
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: total, summary: String(format: "%.0f t/s", pp))
        } catch {
            seed = backend.eos
        }
        return (seed, pp)
    }

    private func strippedOfToolBlocks(_ s: String) -> String {
        var out = s
        while let open = out.range(of: wire.toolCallOpen),
              let close = out.range(
                  of: wire.toolCallClose,
                  range: open.upperBound ..< out.endIndex) {
            out.removeSubrange(open.lowerBound ..< close.upperBound)
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func answerText(_ decoded: String, sawThink: Bool) -> String {
        var body = ""
        if sawThink {
            if let close = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[close.upperBound...])
            }
        } else {
            body = decoded
        }
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
