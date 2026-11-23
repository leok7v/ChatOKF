import CoreGraphics
import Observation
import XCTest
@testable import Chat
import LLM

private final class MockBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let scripts: [[Int32]]
    private let bytesOf: [Int32: [UInt8]]
    private var round = -1
    private var cursor = 0
    private var pos = 0
    private var savedMark = 0
    var stopNextExtend = false
    var pauseOnExtend = false
    var pauseOnDecode = false
    private let lock = NSLock()
    private var paused = false
    private var gate: CheckedContinuation<Void, Never>?
    private var stopped = false

    var isPaused: Bool {
        lock.lock()
        defer { lock.unlock() }
        return paused
    }

    private func pause() async {
        await withCheckedContinuation {
            (k: CheckedContinuation<Void, Never>) in
            lock.lock()
            gate = k
            paused = true
            lock.unlock()
        }
    }

    init(scripts: [[Int32]], vocab: [Int32: [UInt8]]) {
        self.scripts = scripts
        self.bytesOf = vocab
    }

    func encode(_ text: String) -> [Int32] {
        text.utf8.map { byte in Int32(byte) }
    }

    func tokenBytes(_ id: Int32) -> [UInt8] { bytesOf[id] ?? [] }

    func text(_ ids: [Int32]) -> String {
        var b: [UInt8] = []
        for id in ids { b.append(contentsOf: bytesOf[id] ?? []) }
        return String(decoding: b, as: UTF8.self)
    }

    var position: Int { get async { pos } }

    func reset() async {
        round += 1
        cursor = 1
        pos = 0
        stopped = false
    }

    func useSampler(_ s: Sampler?) async {}

    func requestStop() { stopped = true }
    func shouldStop() -> Bool { stopped }

    private func script() -> [Int32] {
        round >= 0 && round < scripts.count ? scripts[round] : []
    }

    func extend(_ ids: [Int32]) async throws -> Int32 {
        stopped = false
        if pauseOnExtend {
            pauseOnExtend = false
            await pause()
        }
        if stopNextExtend {
            stopNextExtend = false
            throw EngineError.stopped
        }
        if stopped {
            throw EngineError.stopped
        }
        pos += ids.count
        let s = script()
        return s.isEmpty ? eos : s[0]
    }

    func mark() async throws { savedMark = pos }

    func rewind() async throws {
        round += 1
        cursor = 1
        pos = savedMark
    }

    func release() {
        lock.lock()
        let waiting = gate
        gate = nil
        paused = false
        lock.unlock()
        waiting?.resume()
    }

    func decode(_ token: Int32) async throws -> Int32 {
        if pauseOnDecode {
            pauseOnDecode = false
            await pause()
        }
        let s = script()
        let out = cursor < s.count ? s[cursor] : eos
        cursor += 1
        pos += 1
        return out
    }

    struct State: BackendState { let pos: Int; let mark: Int }
    func saveState() async throws -> any BackendState {
        State(pos: pos, mark: savedMark)
    }
    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { pos = s.pos; savedMark = s.mark }
    }
}

private final class TestRunner: ToolRunner, @unchecked Sendable {
    let tools: [ToolSpec]
    private let reply: String

    init(reply: String) {
        self.tools = [
            ToolSpec(name: "calculator", description: "evaluate math.",
                     parametersJSON: "{\"type\":\"object\",\"properties\":"
                        + "{\"expression\":{\"type\":\"string\"}}}"),
        ]
        self.reply = reply
    }

    func execute(_ name: String, _ args: [ToolArg]) async -> String { reply }
}

@MainActor final class SessionDriverTests: XCTestCase {

    private let template = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}"
        + "{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
        + "{%- endif -%}"

    private func vocab(_ pairs: [(Int32, String)]) -> [Int32: [UInt8]] {
        var out: [Int32: [UInt8]] = [:]
        for (id, s) in pairs { out[id] = Array(s.utf8) }
        return out
    }

    private func driven(_ backend: MockBackend, runner: (any ToolRunner)? = nil)
        -> Session {
        let session = Session(modelName: "test", systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        session.toolRunnerOverride = runner
        let config = Session.SessionConfig(
            thinking: false, reasoningEffortRaw: "medium",
            reasoningEffortSlot: 1, thinkTokenCap: 200)
        session.makeSession(config) { _ in }
        return session
    }

    private func drain(_ events: AsyncStream<TurnEvent>) async -> [TurnEvent] {
        var out: [TurnEvent] = []
        for await event in events { out.append(event) }
        return out
    }

    private func finalOutcome(_ events: [TurnEvent])
        -> ChatSession.TurnOutcome? {
        var outcome: ChatSession.TurnOutcome? = nil
        if case .finished(let found, _) = events.last { outcome = found }
        return outcome
    }

    // A plain turn: the driver's sendText streams the model's answer as
    // `.answer` pieces and ends on `.finished(.answered, _)`.
    func testPlainTurnStreamsAnswerAndFinishes() async throws {
        let backend = MockBackend(scripts: [[1]],
                                  vocab: vocab([(1, "hello there")]))
        let session = driven(backend)
        let sent = session.sendText(prompt: "hi", display: "hi", docs: [],
                                    thinkTokenCap: 200, thinkingActive: false)
        let events = await drain(try XCTUnwrap(sent).events)
        let answer = events.compactMap { event -> String? in
            if case .answer(let piece) = event { return piece }
            return nil
        }.joined()
        XCTAssertEqual(answer, "hello there")
        let outcome = try XCTUnwrap(finalOutcome(events),
                                    "turn did not end on .finished")
        XCTAssertEqual(outcome, .answered)
    }

    // A turn stopped mid-prefill rolls back: `.finished(.stopped, _)` with
    // no answer pieces, the outcome the view model rolls its bubbles back on.
    func testStoppedTurnFinishesWithNoAnswer() async throws {
        let backend = MockBackend(
            scripts: [[1], [2]], vocab: vocab([(1, "one"), (2, "two")]))
        let session = driven(backend)
        _ = await drain(try XCTUnwrap(session.sendText(
            prompt: "first", display: "first", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        backend.stopNextExtend = true
        let events = await drain(try XCTUnwrap(session.sendText(
            prompt: "second", display: "second", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        let answered = events.contains { event in
            if case .answer = event { return true }
            return false
        }
        XCTAssertFalse(answered, "a rolled-back turn must yield no answer")
        let outcome = try XCTUnwrap(finalOutcome(events),
                                    "turn did not end on .finished")
        XCTAssertEqual(outcome, .stopped)
    }

    // An immediate EOS with no user stop ends answerless rather than
    // stopped, the outcome the view keeps the two bubbles for.
    func testEmptyDecodeEndsAnswerless() async throws {
        let backend = MockBackend(
            scripts: [[1], []], vocab: vocab([(1, "one")]))
        let session = driven(backend)
        _ = await drain(try XCTUnwrap(session.sendText(
            prompt: "first", display: "first", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        let events = await drain(try XCTUnwrap(session.sendText(
            prompt: "second", display: "second", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        let outcome = try XCTUnwrap(finalOutcome(events),
                                    "turn did not end on .finished")
        XCTAssertEqual(outcome, .answerless)
    }

    // A tool round reaches the event stream as `.toolRound`, resolved to
    // the advertised tool, before the final answer streams.
    func testToolRoundReachesTheEventStream() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, call), (2, "The answer is 4.")]))
        let session = driven(backend, runner: TestRunner(reply: "4"))
        let sent = session.sendText(prompt: "what is 2 + 2?",
                                    display: "what is 2 + 2?", docs: [],
                                    thinkTokenCap: 200, thinkingActive: false)
        let events = await drain(try XCTUnwrap(sent).events)
        let rounds = events.compactMap { event -> ToolRoundEvent? in
            if case .toolRound(let round) = event { return round }
            return nil
        }
        XCTAssertTrue(rounds.contains { round in
            round.resolved == "calculator"
        })
        let answer = events.compactMap { event -> String? in
            if case .answer(let piece) = event { return piece }
            return nil
        }.joined()
        XCTAssertEqual(answer, "The answer is 4.")
        let committed = await session.session!.committedCount
        let replayed = await session.session!.replayTokens
        XCTAssertLessThan(replayed, committed)
        XCTAssertGreaterThan(replayed, 0)
    }

    // A Stop raised while paused inside prefill (before any token decoded)
    func testStopMidPrefillRollsBackToFinished() async throws {
        let backend = MockBackend(scripts: [[1]], vocab: vocab([(1, "x")]))
        backend.pauseOnExtend = true
        let session = driven(backend)
        let sent = try XCTUnwrap(session.sendText(
            prompt: "count", display: "count", docs: [],
            thinkTokenCap: 200, thinkingActive: false))
        var collected: [TurnEvent] = []
        let consumer = Task { @MainActor in
            for await event in sent.events { collected.append(event) }
        }
        while !backend.isPaused { await Task.yield() }
        session.stop()
        backend.release()
        await consumer.value
        let answered = collected.contains { event in
            if case .answer = event { return true }
            return false
        }
        XCTAssertFalse(answered, "a rolled-back turn must yield no answer")
        let outcome = try XCTUnwrap(finalOutcome(collected),
                                    "turn did not end on .finished")
        XCTAssertEqual(outcome, .stopped)
    }

    // A Stop raised AFTER text has streamed must keep it: only an empty
    // assistant message rolls back, so `.finished(.answered, _)` is expected.
    func testStopAfterAnswerStreamedKeepsTheAnswer() async throws {
        let backend = MockBackend(
            scripts: [[1, 2]], vocab: vocab([(1, "hello "), (2, "there")]))
        backend.pauseOnDecode = true
        let session = driven(backend)
        let sent = try XCTUnwrap(session.sendText(
            prompt: "count", display: "count", docs: [],
            thinkTokenCap: 200, thinkingActive: false))
        var collected: [TurnEvent] = []
        let consumer = Task { @MainActor in
            for await event in sent.events { collected.append(event) }
        }
        while !backend.isPaused { await Task.yield() }
        session.stop()
        backend.release()
        await consumer.value
        let answer = collected.compactMap { event -> String? in
            if case .answer(let piece) = event { return piece }
            return nil
        }.joined()
        XCTAssertEqual(answer, "hello ")
        let cancelled = collected.contains { event in
            if case .cancelled = event { return true }
            return false
        }
        XCTAssertFalse(cancelled, "text already on screen must not be "
            + "attributed to a bare cancel")
        let outcome = try XCTUnwrap(finalOutcome(collected),
                                    "turn did not end on .finished")
        XCTAssertEqual(outcome, .answered)
    }

    func testVoiceRulesRideTheSystemTextWithTheToolsOn() {
        let backend = MockBackend(scripts: [], vocab: [:])
        let session = Session(modelName: "test", systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        session.toolRunnerOverride = TestRunner(reply: "4")
        let plain = Session.SessionConfig(
            thinking: true, reasoningEffortRaw: "medium",
            reasoningEffortSlot: 1, thinkTokenCap: 200)
        session.makeSession(plain) { _ in }
        XCTAssertTrue(session.systemText.hasPrefix("You are a bot."))
        XCTAssertFalse(session.systemText.contains("voice companion"))
        XCTAssertFalse(session.voiceRules)
        var voiced = plain
        voiced.voice = true
        session.makeSession(voiced) { _ in }
        XCTAssertTrue(session.systemText.hasPrefix("You are a bot."))
        XCTAssertTrue(session.systemText.hasSuffix("directly."))
        XCTAssertTrue(session.systemText.contains("exactly one sentence"))
        XCTAssertTrue(session.systemText.contains("Your tools are Calculator:"))
        XCTAssertTrue(session.voiceRules)
    }

    func testVoiceRulesNameEveryToolOnceAndNoToolsWhenThereAreNone() {
        let rules = Session.voiceRules(tools: [
            "web_search", "fetch_url", "calculator", "get_current_time",
            "get_news", "get_weather", "wikipedia_query", MemoryTools.search])
        for label in ["Web Search", "URL Fetch", "Calculator", "Current Time",
                      "News", "Weather", "Wikipedia", "Note Search"] {
            XCTAssertEqual(rules.components(separatedBy: label).count, 2,
                           label)
        }
        XCTAssertTrue(rules.contains("Web Search, URL Fetch, Calculator"))
        XCTAssertFalse(Session.voiceRules(tools: []).contains("Your tools"))
        XCTAssertTrue(Session.voiceRules(tools: ["unknown_tool"])
            .contains("unknown_tool"))
    }

    func testMetaTaskClearingIsObservable() async throws {
        let backend = MockBackend(scripts: [[1]], vocab: vocab([(1, "hi")]))
        let session = driven(backend)
        _ = await drain(try XCTUnwrap(session.sendText(
            prompt: "hi", display: "hi", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        session.runMetaTurns(titled: true, wantsFollowup: false,
                             extraction: nil, onTitle: { _ in },
                             onFollowup: { _ in }, onRemembered: { _ in })
        XCTAssertTrue(session.metaTaskRunning)
        await withCheckedContinuation {
            (k: CheckedContinuation<Void, Never>) in
            withObservationTracking {
                _ = session.metaTaskRunning
            } onChange: {
                k.resume()
            }
        }
        XCTAssertFalse(session.metaTaskRunning)
    }

}

@MainActor final class ReplayTests: XCTestCase {

    private let template = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}"
        + "{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
        + "{%- endif -%}"

    private let config = Session.SessionConfig(
        thinking: false, reasoningEffortRaw: "medium",
        reasoningEffortSlot: 1, thinkTokenCap: 200)

    private func driver() -> Session {
        let backend = MockBackend(scripts: [], vocab: [:])
        let session = Session(modelName: "test", systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        return session
    }

    func testAReplayLaysEachTurnAsItsDeltaOverThePrimedPrefix() async throws {
        let session = driver()
        await session.newChatEngine(config) { _ in }
        await session.awaitPrimed()
        let primed = await session.session!.committedCount
        XCTAssertGreaterThan(primed, 0)
        let turns = [
            PlannedTurn(prompt: "first", answer: "one", soft: nil, images: []),
            PlannedTurn(prompt: "second", answer: "two", soft: nil,
                        images: []),
        ]
        var rendered: [String] = []
        let ok = await session.replay(turns, budget: 280, config) { e in
            if e.kind == .render { rendered.append(e.text) }
        }
        XCTAssertTrue(ok)
        let first = "<|im_start|>user\nfirst<|im_end|>"
        let second = "<|im_start|>assistant\none<|im_end|>"
            + "<|im_start|>user\nsecond<|im_end|>"
        let gen = "<|im_start|>assistant"
        XCTAssertEqual(rendered, [first + gen, second + gen])
        let after = await session.session!.committedCount
        XCTAssertEqual(after, primed + first.utf8.count + second.utf8.count)
        let replayed = await session.session!.replayTokens
        XCTAssertEqual(replayed, first.utf8.count + second.utf8.count)
        let soft = await session.session!.hasSoftTurns
        XCTAssertFalse(soft)
    }

    private func user(_ text: String, soft: URL? = nil,
                      posters: Int = 0) -> Message {
        var m = Message(fromUser: true, text: text, soft: soft)
        m.posters = Array(repeating: try! ReplayTests.square(), count: posters)
        return m
    }

    private static func square() throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: 4, height: 4, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        return try XCTUnwrap(ctx.makeImage())
    }

    private func answer(_ text: String) -> Message {
        Message(fromUser: false, text: text)
    }

    func testTextTurnsReplayAndAnUnansweredOneIsSkipped() {
        let plan = Session.plan(
            [user("a"), answer("A"), user("dropped"), answer(""),
             user("b"), answer("B")],
            loaded: "gemma-4-E4B", sees: true, stamps: [:])
        if case .replay(let turns) = plan {
            XCTAssertEqual(turns.map { t in t.prompt }, ["a", "b"])
            XCTAssertEqual(turns.map { t in t.answer }, ["A", "B"])
        } else {
            XCTFail("text turns must replay")
        }
    }

    func testAKinStampReplaysAndAForeignOneNamesItsModel() {
        let kept = URL(fileURLWithPath: "/tmp/kept.soft")
        let kin = Session.plan(
            [user("look", soft: kept), answer("A dog.")],
            loaded: "gemma-4-E4B-MTP", sees: true,
            stamps: [kept: "gemma-4-E4B.712cfb35"])
        if case .replay(let turns) = kin {
            XCTAssertEqual(turns.first?.soft, kept)
        } else {
            XCTFail("the MTP build shares the E4B towers")
        }
        let foreign = Session.plan(
            [user("look", soft: kept), answer("A dog.")],
            loaded: "gemma-4-E2B", sees: true,
            stamps: [kept: "gemma-4-E4B.712cfb35"])
        if case .needsModel(let name) = foreign {
            XCTAssertEqual(name, "gemma-4-E4B")
        } else {
            XCTFail("another family's rows need their model")
        }
        XCTAssertEqual(Session.stampModel("Qwen3.5-2B.2e65cddd"), "Qwen3.5-2B")
        XCTAssertEqual(Session.towerFamily("Qwen3.8-27B-IQ1_S"),
                       Session.towerFamily("Qwen3.8-27B-Q4_K_S"))
    }

    func testACancelledReplayStopsInsteadOfLayingOn() async throws {
        let backend = MockBackend(scripts: [], vocab: [:])
        let session = Session(modelName: "test", systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        let turns = (1 ... 3).map { i in
            PlannedTurn(prompt: "ask \(i)", answer: "answer \(i)",
                        soft: nil, images: [])
        }
        backend.pauseOnExtend = true
        let config = config
        let job = Task { @MainActor in
            await session.replay(turns, budget: 280, config) { _ in }
        }
        while !backend.isPaused { await Task.yield() }
        job.cancel()
        backend.release()
        let ok = await job.value
        XCTAssertFalse(ok, "a cancelled replay laid every turn and resumed")
    }

    func testAProcessWithNoMemoryLimitHasNoHeadroom() {
        XCTAssertNil(Footprint.headroom(0),
                     "0 is no limit, not 0 bytes left")
        XCTAssertEqual(Footprint.headroom(300 << 20), 300 << 20)
    }

    func testWhatWasNotKeptReadsOnly() {
        let missing = URL(fileURLWithPath: "/tmp/gone.soft")
        let gone = Session.plan([user("look", soft: missing), answer("A.")],
                                loaded: "gemma-4-E4B", sees: true, stamps: [:])
        if case .readOnly = gone {} else { XCTFail("a lost file reads only") }
        let spoken = Session.plan([user("Spoken, 4.2s"), answer("Hi.")],
                                  loaded: "gemma-4-E4B", sees: true,
                                  stamps: [:])
        if case .readOnly = spoken {} else { XCTFail("speech reads only") }
        let clip = Session.plan([user("watch", posters: 1), answer("Dogs.")],
                                loaded: "gemma-4-E4B", sees: true, stamps: [:])
        if case .readOnly = clip {} else { XCTFail("a clip reads only") }
        let empty = Session.plan([user("a"), answer("")],
                                 loaded: "gemma-4-E4B", sees: true, stamps: [:])
        if case .readOnly = empty {} else { XCTFail("nothing to lay") }
    }

}

private final class LaneBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let lock = NSLock()
    private var inside = 0
    private var overlaps = 0
    private var pos = 0
    private var marked = 0
    private var fed: [UInt8] = []
    private var holding = false
    private var held: CheckedContinuation<Void, Never>?
    private var heard: (@Sendable (String) -> Void)?

    var crossed: Int { lock.withLock { overlaps } }

    var isHeld: Bool { lock.withLock { held != nil } }

    func holdNextExtend() { lock.withLock { holding = true } }

    func onFeed(_ hook: @escaping @Sendable (String) -> Void) {
        lock.withLock { heard = hook }
    }

    func release() {
        let waiting = lock.withLock {
            () -> CheckedContinuation<Void, Never>? in
            let k = held
            held = nil
            return k
        }
        waiting?.resume()
    }

    private func hold() async {
        let wanted = lock.withLock { () -> Bool in
            let w = holding
            holding = false
            return w
        }
        if wanted {
            await withCheckedContinuation {
                (k: CheckedContinuation<Void, Never>) in
                lock.withLock { held = k }
            }
        }
    }

    var fedText: String {
        lock.withLock { String(decoding: fed, as: UTF8.self) }
    }

    func forget() { lock.withLock { fed = [] } }

    private func guarded<T>(_ body: () throws -> T) async rethrows -> T {
        lock.withLock {
            inside += 1
            if inside > 1 { overlaps += 1 }
        }
        for _ in 0 ..< 3 { await Task.yield() }
        defer { lock.withLock { inside -= 1 } }
        return try body()
    }

    func encode(_ text: String) -> [Int32] {
        text.utf8.map { byte in Int32(byte) }
    }

    func tokenBytes(_ id: Int32) -> [UInt8] {
        id >= 0 && id < 256 ? [UInt8(id)] : []
    }

    func text(_ ids: [Int32]) -> String {
        String(decoding: ids.flatMap { id in tokenBytes(id) }, as: UTF8.self)
    }

    var position: Int {
        get async { await guarded { lock.withLock { pos } } }
    }

    func reset() async {
        await guarded { lock.withLock { pos = 0 } }
    }

    func useSampler(_ s: Sampler?) async {}

    func extend(_ ids: [Int32]) async throws -> Int32 {
        await hold()
        await guarded {
            lock.withLock {
                pos += ids.count
                fed += ids.map { id in UInt8(truncatingIfNeeded: id) }
            }
        }
        let hook = lock.withLock { heard }
        hook?(text(ids))
        return eos
    }

    func decode(_ token: Int32) async throws -> Int32 {
        await guarded { lock.withLock { pos += 1 } }
        return eos
    }

    func mark() async throws {
        await guarded { lock.withLock { marked = pos } }
    }

    func rewind() async throws {
        await guarded { lock.withLock { pos = marked } }
    }

    struct State: BackendState { let pos: Int; let mark: Int }

    func saveState() async throws -> any BackendState {
        await guarded { lock.withLock { State(pos: pos, mark: marked) } }
    }

    func loadState(_ state: any BackendState) async throws {
        await guarded {
            if let s = state as? State {
                lock.withLock {
                    pos = s.pos
                    marked = s.mark
                }
            }
        }
    }

    func attach(_ dir: URL) async throws {
        try await guarded { throw EngineError.missingModel("lane") }
    }

    func detach() async { await guarded {} }

    func park(to dir: URL, meta: Data) async throws { await guarded {} }

    func resume(from dir: URL) async throws -> Data {
        try await guarded { throw EngineError.missingModel("lane") }
    }

    func prime(from cooked: URL, into live: URL) async throws -> Data {
        try await guarded { throw EngineError.missingModel("lane") }
    }

    func precook(to cooked: URL, meta: Data) async throws { await guarded {} }
}

private struct SplitMix {
    private var state: UInt64

    init(_ seed: UInt64) { state = seed }

    mutating func below(_ n: Int) -> Int {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Int((z ^ (z >> 31)) % UInt64(n))
    }
}

@MainActor final class SessionLaneTests: XCTestCase {

    private let template = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}"
        + "{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
        + "{%- if enable_thinking -%}[T]{%- endif -%}"
        + "{%- endif -%}"

    private func config(_ thinking: Bool) -> Session.SessionConfig {
        Session.SessionConfig(thinking: thinking, reasoningEffortRaw: "medium",
                              reasoningEffortSlot: 1, thinkTokenCap: 200)
    }

    private func turn(_ session: Session) -> Task<Void, Never> {
        let sent = session.sendText(prompt: "hi", display: "hi", docs: [],
                                    thinkTokenCap: 200, thinkingActive: false)
        let events = sent?.events ?? AsyncStream { k in k.finish() }
        return Task { @MainActor in
            for await _ in events {}
        }
    }

    private func fire(_ op: Int, _ session: Session, _ rng: inout SplitMix,
                      _ want: inout Bool, _ waits: inout [Task<Void, Never>]) {
        switch op {
        case 0:
            let asked = rng.below(2) == 1
            let leaving = rng.below(2) == 1 ? UUID() : nil
            waits.append(session.newChat(parking: leaving, config(asked)) {
                _ in
            })
            want = asked && session.modelSupportsThinking
        case 1:
            want = rng.below(2) == 1
            session.pushThinking(want, resetIfFresh: rng.below(2) == 1)
        case 2:
            session.pushReasoningEffort(rng.below(2) == 1 ? "high" : "low")
        case 3:
            session.pushTools()
        case 4:
            session.primeSession(resetFirst: true)
        case 5:
            waits.append(turn(session))
        case 6:
            session.runMetaTurns(titled: true, wantsFollowup: true,
                                 extraction: nil, onTitle: { _ in },
                                 onFollowup: { _ in }, onRemembered: { _ in })
        case 7:
            session.stop()
        case 8:
            let on = rng.below(2) == 1
            waits.append(Task { await session.pushSpeculation(on) })
        default:
            let id = UUID()
            waits.append(Task { await session.parkCurrent(id) })
        }
    }

    private func shuffled(_ seed: UInt64) async {
        let backend = LaneBackend()
        let session = Session(modelName: "lane-test",
                              systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        session.makeSession(config(false)) { _ in }
        session.primeSession()
        var rng = SplitMix(seed)
        var want = false
        var waits: [Task<Void, Never>] = []
        for _ in 0 ..< 48 {
            fire(rng.below(10), session, &rng, &want, &waits)
            for _ in 0 ..< rng.below(4) { await Task.yield() }
        }
        for wait in waits { await wait.value }
        await session.awaitPrimed()
        await session.awaitMeta()
        backend.forget()
        await turn(session).value
        XCTAssertEqual(backend.crossed, 0, "seed \(seed): the engine was "
            + "driven by two operations at once")
        XCTAssertEqual(backend.fedText.contains("[T]"), want,
                       "seed \(seed): the last thinking setting lost")
    }

    func testEveryEngineChangeRunsInOneLaneAndTheLastSettingWins() async {
        for seed in UInt64(1) ... 16 {
            await shuffled(seed)
        }
    }

    func testATurnStoppedBeforeItEntersTheEngineNeverEntersIt() async throws {
        let backend = LaneBackend()
        let session = Session(modelName: "lane-test",
                              systemPrompt: "You are a bot.")
        session.installBackend(backend, template: template, vocabSize: 256,
                               presets: .greedy)
        session.makeSession(config(false)) { _ in }
        let chat = try XCTUnwrap(session.session)
        let late = expectation(description: "the stopped turn fed the engine")
        late.isInverted = true
        backend.onFeed { text in
            if text.contains("user\nhi") { late.fulfill() }
        }
        backend.holdNextExtend()
        await chat.primeOrCook(at: FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString))
        while !backend.isHeld { await Task.yield() }
        let sent = try XCTUnwrap(session.sendText(
            prompt: "hi", display: "hi", docs: [], thinkTokenCap: 200,
            thinkingActive: false))
        var collected: [TurnEvent] = []
        let consumer = Task { @MainActor in
            for await event in sent.events { collected.append(event) }
        }
        session.stop()
        backend.release()
        await consumer.value
        await fulfillment(of: [late], timeout: 1)
        var outcome: ChatSession.TurnOutcome? = nil
        if case .finished(let found, _) = collected.last { outcome = found }
        XCTAssertEqual(outcome, .stopped)
        XCTAssertEqual(backend.crossed, 0)
    }

}
