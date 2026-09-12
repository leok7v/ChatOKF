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

    func testMetaTaskClearingIsObservable() async throws {
        let backend = MockBackend(scripts: [[1]], vocab: vocab([(1, "hi")]))
        let session = driven(backend)
        _ = await drain(try XCTUnwrap(session.sendText(
            prompt: "hi", display: "hi", docs: [],
            thinkTokenCap: 200, thinkingActive: false)).events)
        session.runMetaTurns(titled: true, wantsFollowup: false,
                             onTitle: { _ in }, onFollowup: { _ in })
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
