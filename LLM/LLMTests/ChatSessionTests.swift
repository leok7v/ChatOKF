import XCTest
@testable import LLM

// Offline structural checks for the ChatSession turn loop (no model): a
private final class MockBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let scripts: [[Int32]]
    // Replay the scripts round-robin (a model that never stops calling tools),
    // to drive the runaway tool loop against maxToolRounds.
    private let cycle: Bool
    private let bytesOf: [Int32: [UInt8]]
    private var round = -1
    private var cursor = 0
    private var pos = 0
    private var savedMark = 0
    private var freshTurn = false
    private(set) var extendedTokens = 0
    private(set) var rewinds = 0
    // When set, the next extend throws EngineError.stopped once (mid-prefill
    // Stop), so a test can exercise the turn rollback.
    var stopNextExtend = false

    init(scripts: [[Int32]], vocab: [Int32: [UInt8]], cycle: Bool = false) {
        self.scripts = scripts
        self.bytesOf = vocab
        self.cycle = cycle
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
        freshTurn = false
    }

    func useSampler(_ s: Sampler?) async {}

    private func script() -> [Int32] {
        let idx = cycle && !scripts.isEmpty ? round % scripts.count : round
        return idx >= 0 && idx < scripts.count ? scripts[idx] : []
    }

    func extend(_ ids: [Int32]) async throws -> Int32 {
        if stopNextExtend {
            stopNextExtend = false
            throw EngineError.stopped
        }
        if freshTurn {
            round += 1
            cursor = 1
            freshTurn = false
        }
        pos += ids.count
        extendedTokens += ids.count
        let s = script()
        return s.isEmpty ? eos : s[0]
    }

    func mark() async throws { savedMark = pos }

    func rewind() async throws {
        rewinds += 1
        round += 1
        cursor = 1
        pos = savedMark
        freshTurn = false
    }

    func decode(_ token: Int32) async throws -> Int32 {
        let s = script()
        let out = cursor < s.count ? s[cursor] : eos
        cursor += 1
        pos += 1
        return out
    }

    struct State: BackendState { let pos: Int; let mark: Int }
    func saveState() async throws -> any BackendState {
        freshTurn = true
        return State(pos: pos, mark: savedMark)
    }
    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State { pos = s.pos; savedMark = s.mark }
    }
}

// Models the KV as the literal byte stream of every extended/decoded token,
private final class TapeBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32
    private let scripts: [[Int32]]
    private let bytesOf: [Int32: [UInt8]]
    private let commits: Int
    private var tape: [Int32] = []
    private var marked = 0
    private var round = -1
    private var cursor = 0
    private var decoding = false
    private var freshTurn = false
    private(set) var rewinds = 0
    private(set) var longest: [Int32] = []

    private static let imagePad: Int32 = 999
    private static let junk: Int32 = 998
    private static let imEnd = "<|im_end|>"

    init(scripts: [[Int32]], vocab: [Int32: [UInt8]], eos: Int32 = -1,
         commits: Int = 0) {
        self.scripts = scripts
        self.bytesOf = vocab
        self.eos = eos
        self.commits = commits
    }

    // Sleep per extend (ms): holds a cook's prefill open across a suspension
    var extendDelayMs: UInt64 = 0

    func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        let pieces = text.components(separatedBy: "<|image_pad|>")
        for (i, piece) in pieces.enumerated() {
            if i > 0 { out.append(TapeBackend.imagePad) }
            if eos >= 0, i == 0, piece.hasPrefix(TapeBackend.imEnd) {
                out.append(eos)
                out.append(contentsOf: piece
                    .dropFirst(TapeBackend.imEnd.count).utf8
                    .map { byte in Int32(byte) })
            } else {
                out.append(contentsOf: piece.utf8.map { byte in Int32(byte) })
            }
        }
        return out
    }

    func supportsSoftTokens() async -> Bool { true }

    func extendSoft(_ ids: [Int32],
                    spans: [SoftSpan]) async throws -> Int32 {
        try await extend(ids)
    }

    func tokenBytes(_ id: Int32) -> [UInt8] {
        var out: [UInt8] = []
        if id == TapeBackend.junk {
            out = Array("~".utf8)
        } else if id >= 0 && id < 256 {
            out = [UInt8(id)]
        } else if id == eos {
            out = Array(TapeBackend.imEnd.utf8)
        } else {
            out = bytesOf[id] ?? []
        }
        return out
    }

    func text(_ ids: [Int32]) -> String {
        var b: [UInt8] = []
        for id in ids { b.append(contentsOf: tokenBytes(id)) }
        return String(decoding: b, as: UTF8.self)
    }

    var kvText: String { text(tape) }

    var longestText: String { text(longest) }

    var position: Int { get async { tape.count } }

    private func grew() {
        if tape.count > longest.count { longest = tape }
    }

    func reset() async {
        tape = []
        round += 1
        cursor = 1
        decoding = false
        freshTurn = false
    }

    func useSampler(_ s: Sampler?) async {}

    private func script() -> [Int32] {
        round >= 0 && round < scripts.count ? scripts[round] : []
    }

    private func nextScripted() -> Int32 {
        let s = script()
        let out = cursor < s.count ? s[cursor] : eos
        cursor += 1
        return out
    }

    func extend(_ ids: [Int32]) async throws -> Int32 {
        if extendDelayMs > 0 {
            try? await Task.sleep(nanoseconds: extendDelayMs * 1_000_000)
        }
        if freshTurn {
            round += 1
            cursor = 1
            decoding = false
            freshTurn = false
        }
        tape.append(contentsOf: ids)
        grew()
        let s = script()
        return decoding ? nextScripted() : (s.isEmpty ? eos : s[0])
    }

    func mark() async throws { marked = tape.count }

    func rewind() async throws {
        rewinds += 1
        tape = Array(tape.prefix(marked))
        round += 1
        cursor = 1
        decoding = false
        freshTurn = false
    }

    func decode(_ token: Int32) async throws -> Int32 {
        tape.append(token)
        decoding = true
        let out = nextScripted()
        if out == eos && commits > 0 {
            tape.append(eos)
            for _ in 1 ..< max(commits, 1) { tape.append(TapeBackend.junk) }
        }
        grew()
        return out
    }

    struct State: BackendState { let tape: [Int32]; let marked: Int }
    func saveState() async throws -> any BackendState {
        freshTurn = true
        return State(tape: tape, marked: marked)
    }
    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            tape = s.tape
            marked = s.marked
        }
    }

    // Byte state for the precook tests: the tape + mark round-trip
    // losslessly, the offline stand-in for Engine.serialize/deserialize.
    private struct Blob: Codable { let tape: [Int32]; let marked: Int }
    func serializeState(_ state: any BackendState) async -> Data {
        var out = Data()
        if let s = state as? State {
            out = (try? JSONBytes.reproducible(
                Blob(tape: s.tape, marked: s.marked))) ?? Data()
        }
        return out
    }
    func deserializeState(_ data: Data) async throws -> any BackendState {
        let b = try JSONDecoder().decode(Blob.self, from: data)
        return State(tape: b.tape, marked: b.marked)
    }
}

// Advertises get_current_time + calculator and records which tools ran, so the
// tool loop runs offline with no clock/network dependence in the assertions.
private final class RecordingRunner: ToolRunner, @unchecked Sendable {
    let tools: [ToolSpec]
    private let reply: String
    private(set) var calls: [String] = []
    private(set) var beginTurns = 0

    func beginTurn() { beginTurns += 1 }

    init(reply: String) {
        self.tools = [
            ToolSpec(name: "get_current_time", description: "the time.",
                     parametersJSON: "{\"type\":\"object\",\"properties\":{}}"),
            // Advertises expression like the production spec: the re-lay keeps
            ToolSpec(name: "calculator", description: "evaluate math.",
                     parametersJSON: "{\"type\":\"object\",\"properties\":"
                        + "{\"expression\":{\"type\":\"string\"}}}"),
        ]
        self.reply = reply
    }

    func execute(_ name: String, _ args: [ToolArg]) async -> String {
        calls.append(name)
        return reply
    }
}

final class ChatSessionTests: XCTestCase {
    private func vocab(_ pairs: [(Int32, String)]) -> [Int32: [UInt8]] {
        var out: [Int32: [UInt8]] = [:]
        for (id, s) in pairs { out[id] = Array(s.utf8) }
        return out
    }

    private let template = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}"
        + "{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
        + "{%- endif -%}"

    private func drain(_ stream: AsyncStream<String>) async -> String {
        var out = ""
        for await piece in stream { out += piece }
        return out
    }

    func testStreamsRepliesAcrossTurns() async throws {
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, "First answer."), (2, "Second answer.")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        let a1 = await drain(session.reply("first"))
        let a2 = await drain(session.reply("second"))
        XCTAssertEqual(a1, "First answer.")
        XCTAssertEqual(a2, "Second answer.")
    }

    // makeTitle appends a throwaway title turn onto the live KV and rolls it
    func testMakeTitleLeavesKVUnchanged() async throws {
        let backend = TapeBackend(
            scripts: [[1001], [1002], [1003]],
            vocab: vocab([(1001, "The sky scatters blue light."),
                          (1002, "Sky Color Question"),
                          (1003, "Yes exactly right.")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue in daytime"))
        let kvBefore = backend.kvText
        let posBefore = await backend.position
        let title = await session.makeTitle()
        XCTAssertEqual(backend.kvText, kvBefore,
                       "title gen must leave the KV byte-identical")
        let posAfter = await backend.position
        XCTAssertEqual(posAfter, posBefore)
        XCTAssertEqual(title, "Sky Color Question")
        let a2 = await drain(session.reply("is that the reason"))
        XCTAssertEqual(a2, "Yes exactly right.",
                       "the conversation must continue after a title gen")
    }

    func testMakeTitleAppendsOntoTheAnswer() async throws {
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, "The sky scatters blue light."),
                          (2, "Sky Color Question")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let rewinds = backend.rewinds
        let fed = backend.extendedTokens
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        XCTAssertEqual(backend.rewinds, rewinds, "the title turn rewound")
        XCTAssertLessThan(backend.extendedTokens - fed,
                          ChatSession.titleInstruction.utf8.count + 80,
                          "the title turn laid more than its own turn")
    }

    func testMakeTitleLaysOnlyTheInstructionTurn() async throws {
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, "The sky scatters blue light."),
                          (1002, "Sky Color Question")]))
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        let peak = backend.longestText
        XCTAssertTrue(peak.contains(
            "The sky scatters blue light.<|im_end|>\n<|im_start|>user\n"
            + ChatSession.titleInstruction + "<|im_end|>\n"
            + "<|im_start|>assistant\n" + ChatSession.metaBlock
            + "Sky Color Question"),
            "the title turn is not appended onto the answer:\n\(peak)")
        XCTAssertEqual(peak.components(separatedBy: "blue light.").count, 2,
                       "the answer was laid twice:\n\(peak)")
    }

    func testMakeTitleAppendsPastACommittedStop() async throws {
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, "The sky scatters blue light."),
                          (1002, "Sky Color Question")]),
            eos: 900, commits: 1)
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        XCTAssertEqual(backend.rewinds, 0, "a committed stop forced a re-lay")
        let peak = backend.longestText
        XCTAssertTrue(peak.contains(
            "blue light.<|im_end|>\n<|im_start|>user\n"), peak)
        XCTAssertFalse(peak.contains("<|im_end|><|im_end|>"),
                       "the stop token was laid twice:\n\(peak)")
    }

    func testMakeTitleRelaysWhenTheKVRanPastTheStop() async throws {
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, "The sky scatters blue light."),
                          (1002, "Sky Color Question")]),
            eos: 900, commits: 2)
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        XCTAssertEqual(backend.rewinds, 1,
                       "a KV that ran past the answer must be rewound")
    }

    func testMakeTitleAppendsAfterAnInjectedClose() async throws {
        let backend = TapeBackend(
            scripts: [[1001, 1002, 1003], [1004]],
            vocab: vocab([(1001, "why\n"), (1002, "because\n"),
                          (1003, "The sky scatters blue light."),
                          (1004, "Sky Color Question")]))
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256, enableThinking: true, softReasoningCap: 2)
        let answer = await drain(session.reply("why is the sky blue"))
        XCTAssertEqual(answer, "The sky scatters blue light.")
        let m = await session.lastMetrics
        XCTAssertEqual(m.thinkTokens, 2, "the soft cap did not fire")
        XCTAssertEqual(m.overrun, 0, "an injected close miscounted the KV")
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        XCTAssertEqual(backend.rewinds, 0,
                       "a capped answer forced the meta turn to re-lay")
        XCTAssertTrue(backend.longestText.contains(
            "blue light.<|im_end|>\n<|im_start|>user\n"), backend.longestText)
    }

    func testMetaTurnOpensATextBlockOnEveryWire() async throws {
        let gemma = try ChatSessionTests.fixture("gemma4-chat-template.jinja")
        let voc = vocab([(1001, "The sky scatters blue light."),
                         (1002, "Sky Color Question\n```")])
        let onGemma = TapeBackend(scripts: [[1001], [1002]], vocab: voc)
        let g = ChatSession(
            backend: onGemma, template: gemma, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        _ = await drain(g.reply("why is the sky blue"))
        let gTitle = await g.makeTitle()
        XCTAssertEqual(gTitle, "Sky Color Question")
        XCTAssertTrue(onGemma.longestText.contains(
            "<|turn>model\n```text\nSky Color Question"),
            onGemma.longestText)
        let onQwen = TapeBackend(scripts: [[1001], [1002]], vocab: voc)
        let q = ChatSession(
            backend: onQwen, template: chatml, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(q.reply("why is the sky blue"))
        let qTitle = await q.makeTitle()
        XCTAssertEqual(qTitle, "Sky Color Question")
        XCTAssertTrue(onQwen.longestText.contains(
            "<|im_start|>assistant\n```text\nSky Color Question"),
            onQwen.longestText)
        XCTAssertEqual(ChatSession.cleanFollowup(
            "Why is the sunset red?\n```"), "Why is the sunset red?")
    }

    func testMakeTitleAppendsWithRealQwenTemplate() async throws {
        guard let tmpl = try realQwenTemplate() else {
            throw XCTSkip("no Qwen3.5-0.8B set on disk")
        }
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, "The sky scatters blue light."),
                          (1002, "Sky Color Question")]))
        let session = ChatSession(
            backend: backend, template: tmpl, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertEqual(title, "Sky Color Question")
        XCTAssertEqual(backend.rewinds, 0,
                       "the Qwen template forced the meta turn to re-lay")
        XCTAssertTrue(backend.longestText.contains(
            "blue light.<|im_end|>\n<|im_start|>user\n"), backend.longestText)
    }

    private func leading(_ text: String) -> String {
        let open = Array("<|channel>thought".utf8)
        let out: String
        switch ChatSession.leadingThink(Array(text.utf8), open) {
        case .isThink(let at): out = "think@\(at)"
        case .pending: out = "pending"
        case .notThink: out = "content"
        }
        return out
    }

    func testSpacedChannelNameStillOpensReasoning() {
        XCTAssertEqual(leading("<|channel>thought\nplan"), "think@17")
        XCTAssertEqual(leading("<|channel> thought\nplan"), "think@18")
        XCTAssertEqual(leading("<|channel> "), "pending")
        XCTAssertEqual(leading("<|channel> thing"), "content")
        XCTAssertEqual(leading("< |channel>thought"), "content")
    }

    private static func fixture(_ name: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/" + name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testSpacedChannelOpenerRoutesToReasoning() async throws {
        let template = try ChatSessionTests.fixture(
            "gemma4-chat-template.jinja")
        let backend = MockBackend(
            scripts: [[1, 2, 3, 4]],
            vocab: vocab([(1, "<|channel>"), (2, " thought\nplanning"),
                          (3, "<channel|>"), (4, "The answer.")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("go", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(content, "The answer.")
        XCTAssertTrue(reasoning.text.contains("planning"), reasoning.text)
        XCTAssertFalse(reasoning.text.contains("<|channel>"), reasoning.text)
    }

    // CONCRETE reproduction of the observed "<think>" title (diag: makeTitle
    func testTitleEmptyThinkBlockDoesNotLeak() async throws {
        let backend = TapeBackend(
            scripts: [[1], [2001, 2002, 2003]],
            vocab: vocab([(1, "The sky scatters blue light."),
                          (2001, "<think>"),
                          (2002, "\n\n"),
                          (2003, "</think>")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertFalse(title.contains("<think>"),
                       "empty think block leaked into the title: '\(title)'")
        XCTAssertEqual(title, "",
                       "no real content -> empty title, first-message fallback")
    }

    func testNarratedMetaTurnsAreRejected() {
        let followup = "The\nThe user wants me to pick one specific detail "
            + "from my last answer (\"After 22 years, the balance in the "
            + "savings account will be $2,925.26.\") that they might want "
            + "to know more about. What is the formula?"
        XCTAssertEqual(ChatSession.cleanFollowup(followup), "")
        let planned = "The\nThinking Process:\n\n1.  **Analyze the Request:**"
            + " The user wants me to pick one specific detail from my *last "
            + "answer*. How does the interest compound?"
        XCTAssertEqual(ChatSession.cleanFollowup(planned), "")
        XCTAssertEqual(ChatSession.cleanTitle(
            "Thinking\nThinking Process:\n\n1. **Analyze the Request:**"), "")
        XCTAssertEqual(ChatSession.cleanTitle("Savings"), "",
                       "a one-word title is the fallback's job")
        XCTAssertEqual(ChatSession.cleanFollowup(
            "What is the difference between simple and compound interest?"),
            "What is the difference between simple and compound interest?")
        XCTAssertEqual(ChatSession.cleanFollowup(
            "What is the exact value of $(1.05)^{22}$?"),
            "What is the exact value of (1.05)^22?")
        XCTAssertEqual(ChatSession.cleanFollowup(
            "How does \"each brother has 2 sisters\" fix the count?"),
            "How does \"each brother has 2 sisters\" fix the count?")
        XCTAssertEqual(ChatSession.cleanFollowup(
            "How is the balance calculated to reach $2,925.26?"),
            "How is the balance calculated to reach $2,925.26?")
    }

    func testStrayThinkCloseDoesNotReachTheAnswer() async throws {
        let bytes = Array("Total is 1.10:\n</think>\n\nSo the ball is 0.05."
            .utf8).map { b in Int32(b) }
        let backend = TapeBackend(scripts: [bytes], vocab: vocab([]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: false)
        let out = await drain(session.reply("how much is the ball"))
        XCTAssertFalse(out.contains("</think>"),
                       "stray close marker reached the answer: '\(out)'")
        XCTAssertFalse(out.contains("</think"),
                       "a partial close marker leaked: '\(out)'")
        XCTAssertTrue(out.contains("Total is 1.10:"),
                      "text before the stray marker was dropped: '\(out)'")
        XCTAssertTrue(out.contains("So the ball is 0.05."),
                      "text after the stray marker was dropped: '\(out)'")
    }

    func testUnparseableToolMarkupKeepsTheTail() async throws {
        let text = "Here goes <tool_call>7. Tool Call Generation"
            + "</tool_call> and the answer is 4."
        let backend = MockBackend(scripts: [[1]], vocab: vocab([(1, text)]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let out = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(runner.calls, [], "junk markup must not dispatch")
        XCTAssertTrue(out.contains("Here goes"),
                      "text before the stray markup was dropped: '\(out)'")
        XCTAssertTrue(out.contains("and the answer is 4."),
                      "text after the stray markup was dropped: '\(out)'")
    }

    func testStrayToolMarkupIsStrippedWithoutTools() async throws {
        let text = "<tool_call>7. Tool Call Generation: two calls needed."
        let backend = MockBackend(scripts: [[1]], vocab: vocab([(1, text)]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        let out = await drain(session.reply("research this"))
        XCTAssertFalse(out.contains("<tool_call>"),
                       "marker reached the answer with tools off: '\(out)'")
        XCTAssertTrue(out.contains("Tool Call Generation"),
                      "the prose was dropped: '\(out)'")
    }

    // The ASCII-split variant of the same case: the model emits the block as
    func testTitleAsciiSplitThinkDoesNotLeak() async throws {
        let bytes = Array("<think>\n\n</think>".utf8).map { b in Int32(b) }
        let backend = TapeBackend(
            scripts: [[1], bytes],
            vocab: vocab([(1, "The sky scatters blue light.")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("why is the sky blue"))
        let title = await session.makeTitle()
        XCTAssertFalse(title.contains("<think>"),
                       "ASCII-split think leaked into the title: '\(title)'")
        XCTAssertEqual(title, "")
    }

    // The offline tool loop: the model emits a <tool_call> for the calculator,
    func testToolCallDispatchesThenAnswers() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, call), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(runner.calls, ["calculator"])
        XCTAssertEqual(answer, "The answer is 4.")
        // The tool round's decodeStep is the LAST publisher of lastMetrics;
        // it must carry the turn's real prefill rate, not clobber it with 0.
        let m = await session.lastMetrics
        XCTAssertGreaterThan(m.pp, 0, "tool round zeroed the prefill rate")
    }

    // Thread-safe accumulator for onToolRound events (they fire on the
    // session's task while the test drains the stream on another).
    private final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [ToolRoundEvent] = []
        func add(_ e: ToolRoundEvent) {
            lock.lock(); items.append(e); lock.unlock()
        }
        var list: [ToolRoundEvent] {
            lock.lock(); defer { lock.unlock() }; return items
        }
    }

    // onToolRound surfaces each round twice -- a start event (result nil,
    func testToolRoundEventsCarryArgsAndResult() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, call), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let events = EventBox()
        _ = await drain(session.reply("what is 2 + 2?",
                                      onToolRound: { e in events.add(e) }))
        let got = events.list
        XCTAssertEqual(got.count, 2, "want start + completion")
        XCTAssertEqual(got.first?.round, 1)
        XCTAssertEqual(got.first?.name, "calculator")
        XCTAssertEqual(got.first?.resolved, "calculator")
        XCTAssertNil(got.first?.result, "start event carries no result")
        XCTAssertEqual(got.first?.params.first?.name, "expression")
        XCTAssertEqual(got.first?.params.first?.value, "2 + 2")
        XCTAssertEqual(got.last?.result, "4")
    }

    // ChatML with the real template's whitespace (newline after every
    private let chatml = "{% for m in messages %}"
        + "{% if m.role == 'tool' %}"
        + "<|im_start|>user\n<tool_response>\n{{ m.content }}\n"
        + "</tool_response><|im_end|>\n"
        + "{% else %}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}"
        + "{% for tc in m.tool_calls %}"
        + "<tool_call><function={{ tc.name }}></function></tool_call>"
        + "{% endfor %}"
        + "<|im_end|>\n"
        + "{% endif %}"
        + "{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    // The exact bytes a tool round leaves in the KV are the TEMPLATE'S OWN
    func testToolRoundKVStreamIsWellFormedChatML() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = TapeBackend(
            scripts: [[1001], [1002], [1003]],
            vocab: vocab([(1001, call), (1002, "The answer is 4."),
                          (1003, "Six.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let a1 = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(a1, "The answer is 4.")
        let kv1 = backend.kvText
        XCTAssertTrue(
            kv1.contains("what is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
                + "<tool_call><function=calculator></function></tool_call>"
                + "<|im_end|>\n<|im_start|>user\n<tool_response>\n4\n"
                + "</tool_response><|im_end|>\n"),
            "tool round is not the template's own render:\n\(kv1)")
        let a2 = await drain(session.reply("and 3 + 3?"))
        XCTAssertEqual(a2, "Six.")
        let kv2 = backend.kvText
        XCTAssertFalse(
            kv2.contains("<|im_start|>assistant\n<|im_start|>assistant"),
            "doubled assistant opener after a tool turn:\n\(kv2)")
        XCTAssertTrue(
            kv2.contains("</tool_response><|im_end|>\n<|im_start|>assistant\n"
                + "The answer is 4.<|im_end|>"),
            "next turn's delta did not open the answer correctly:\n\(kv2)")
    }

    // A full tool turn driven by the REAL shipped Qwen3.5 template (skips when
    private func realQwenTemplate() throws -> String? {
        let base = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        let subs = (try? fm.contentsOfDirectory(atPath: base.path)) ?? []
        let tmplURL = subs
            .map { sha in base.appendingPathComponent(sha)
                .appendingPathComponent("chat_template.jinja") }
            .first { url in fm.fileExists(atPath: url.path) }
        return try tmplURL.map { url in
            try String(contentsOf: url, encoding: .utf8)
        }
    }

    func testToolRoundWithRealQwenTemplate() async throws {
        guard let tmpl = try realQwenTemplate() else {
            throw XCTSkip("no Qwen3.5-0.8B set on disk")
        }
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = TapeBackend(
            scripts: [[1001], [1002], [1003]],
            vocab: vocab([(1001, call), (1002, "The answer is 4."),
                          (1003, "Six.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: tmpl, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let a1 = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(a1, "The answer is 4.")
        let kv1 = backend.kvText
        let wire: String = [
            "what is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
            "<think>\n\n</think>\n\n<tool_call>\n<function=calculator>",
            "\n<parameter=expression>\n2 + 2\n</parameter>\n",
            "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n",
            "<tool_response>\n4\n</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ].joined()
        XCTAssertTrue(kv1.contains(wire),
            "tool round is not the real template's render:\n\(kv1)")
        let a2 = await drain(session.reply("and 3 + 3?"))
        XCTAssertEqual(a2, "Six.")
        let kv2 = backend.kvText
        XCTAssertFalse(
            kv2.contains("<|im_start|>assistant\n<|im_start|>assistant"),
            "doubled assistant opener after a tool turn:\n\(kv2)")
        XCTAssertTrue(
            kv2.contains("</tool_response><|im_end|>\n<|im_start|>assistant\n"
                + "The answer is 4.<|im_end|>\n<|im_start|>user\n"
                + "and 3 + 3?<|im_end|>\n"),
            "next turn's delta did not continue correctly:\n\(kv2)")
    }

    // A model that only ever emits tool calls stops at maxToolRounds: after the
    func testToolLoopStopsAtRoundCap() async throws {
        let call =
            "<tool_call><function=get_current_time></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1]], vocab: vocab([(1, call)]), cycle: true)
        let runner = RecordingRunner(reply: "12:00")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("loop"))
        XCTAssertEqual(runner.calls, ["get_current_time"])
        XCTAssertEqual(answer, "")
        // ANSWERLESS, not stopped: the rounds are work the user watched, and
        // the app deletes the two transcript bubbles only for `.stopped`.
        let outcome = await session.turnOutcome
        XCTAssertEqual(outcome, .answerless,
                       "a turn that ran \(ChatSession.maxToolRounds) tool "
                       + "rounds and reached no answer was reported as if "
                       + "nothing had happened")
    }

    func testIdenticalRepeatIsRefusedNotRerun() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2x + 1.00</parameter>"
            + "</function></tool_call>"
        let backend = MockBackend(
            scripts: [[1]], vocab: vocab([(1, call)]), cycle: true)
        let runner = RecordingRunner(reply: "error: not a number")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        _ = await drain(session.reply("how much is the ball"))
        XCTAssertEqual(runner.calls, ["calculator"],
                       "the same call ran more than once")
    }

    func testDifferentArgumentsStillRun() async throws {
        func call(_ expr: String) -> String {
            "<tool_call><function=calculator>"
                + "<parameter=expression>\(expr)</parameter>"
                + "</function></tool_call>"
        }
        let backend = MockBackend(
            scripts: [[1], [2], [3]],
            vocab: vocab([(1, call("1+1")), (2, call("2+2")),
                          (3, "Done.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("add"))
        XCTAssertEqual(runner.calls, ["calculator", "calculator"])
        XCTAssertEqual(answer, "Done.")
    }

    // At the round cap, the budget nudge gives the model one last generation:
    // here it answers instead of calling again, so the turn ends with real text
    func testToolLoopNudgeYieldsAnswerAtCap() async throws {
        let call = "<tool_call><function=calculator></function></tool_call>"
        // A call script per round through the cap (seed + one per rewind), then
        let calls = Array(repeating: [Int32(1)],
                          count: ChatSession.maxToolRounds + 1)
        let backend = MockBackend(
            scripts: calls + [[2]],
            vocab: vocab([(1, call), (2, "Answering from what I have.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("loop then answer"))
        XCTAssertEqual(runner.calls, ["calculator"])
        XCTAssertEqual(answer, "Answering from what I have.")
        let outcome = await session.turnOutcome
        XCTAssertEqual(outcome, .answered)
    }

    // O(delta) proof: turn two rewinds to the prior mark and re-prefills only
    func testTurnTwoRewindsAndExtendsOnlyDelta() async throws {
        let sys = String(repeating: "You are a careful assistant. ", count: 8)
        let backend = MockBackend(
            scripts: [[100], [101]],
            vocab: vocab([(100, "First answer."), (101, "Second answer.")]))
        let session = ChatSession(
            backend: backend, template: template, system: sys, vocabSize: 256)
        let before1 = backend.extendedTokens
        _ = await drain(session.reply("first"))
        let turn1 = backend.extendedTokens - before1
        XCTAssertEqual(backend.rewinds, 0, "turn 1 resets, never rewinds")
        let before2 = backend.extendedTokens
        _ = await drain(session.reply("second"))
        let turn2 = backend.extendedTokens - before2
        XCTAssertGreaterThan(backend.rewinds, 0, "turn 2 rewinds to the mark")
        XCTAssertLessThan(turn2, turn1, "turn 2 skips the shared prefix")
    }

    // A <think>...</think> reply (thinking mode) splits into reasoning vs
    // content token counts.
    func testParkResumeRestoresConversation() async throws {
        let backend = MockBackend(
            scripts: [[1], [2], [3]],
            vocab: vocab([(1, "a-one"), (2, "b-one"), (3, "a-two")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("A first"))     // conversation A
        let ctxA = try await session.park()
        await session.reset()                          // conversation B
        _ = await drain(session.reply("B first"))
        let rewindsBefore = backend.rewinds
        try await session.resume(ctxA)                 // back to A
        _ = await drain(session.reply("A second"))     // must rewind into A
        XCTAssertGreaterThan(backend.rewinds, rewindsBefore,
            "resumed A did not continue its conversation")
    }

    // A <think>...</think> reply routes reasoning to the onReasoning callback
    // and the answer to the content stream -- the two distinct streams.
    func testReasoningRoutesToCallback() async throws {
        let backend = MockBackend(
            scripts: [[10, 11, 12, 13]],
            vocab: vocab([(10, "why "), (11, "because"),
                          (12, "</think>"), (13, "answer")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("go", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertTrue(reasoning.text.contains("why"),
                      "reasoning not captured: \(reasoning.text)")
        XCTAssertTrue(content.contains("answer"), "answer missing: \(content)")
        XCTAssertFalse(content.contains("why"), "reasoning leaked into content")
    }

    // Reasoning-none: a small model that RE-OPENS <think> despite the baked
    func testReasoningNoneStrayThinkStaysOutOfContent() async throws {
        let backend = MockBackend(
            scripts: [[20, 21, 22, 23, 24]],
            vocab: vocab([(20, "<think>"), (21, "why "), (22, "because"),
                          (23, "</think>\n\n"), (24, "Sky Blue")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: false)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("go", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(content, "Sky Blue")
        XCTAssertFalse(content.contains("<think>"),
                       "stray think leaked into content: \(content)")
        XCTAssertTrue(reasoning.text.contains("why"),
                      "stray reasoning not captured: \(reasoning.text)")
        // The opener is MARKUP: it must not open the disclosure either.
        XCTAssertFalse(reasoning.text.contains("<think>"),
                       "opener leaked into reasoning: \(reasoning.text)")
    }

    // Thread-safe accumulator for the onReasoning callback (it fires on the
    // session's task while the test consumes the content stream on another).
    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var acc = ""
        func add(_ s: String) { lock.lock(); acc += s; lock.unlock() }
        var text: String { lock.lock(); defer { lock.unlock() }; return acc }
    }

    // Precooked-prompt cache: save a conversation to a file with a content
    func testSaveLoadContextStamped() async throws {
        let fm = FileManager.default
        let url = fm.temporaryDirectory
            .appendingPathComponent("ctx_\(UUID().uuidString).bin")
        let saver = MockBackend(scripts: [[1]], vocab: vocab([(1, "a-one")]))
        let s1 = ChatSession(backend: saver, template: template,
                             system: "You are a bot.", vocabSize: 256)
        _ = await drain(s1.reply("hello"))
        try await s1.saveContext(to: url, stamp: "v1")

        let loader = MockBackend(scripts: [[2]], vocab: vocab([(2, "next")]))
        let s2 = ChatSession(backend: loader, template: template,
                             system: "You are a bot.", vocabSize: 256)
        let hit = try await s2.loadContext(from: url, stamp: "v1")
        XCTAssertTrue(hit, "matching stamp did not load")
        let miss = try await s2.loadContext(from: url, stamp: "v2")
        XCTAssertFalse(miss, "changed stamp loaded a stale context")

        let before = loader.rewinds
        _ = await drain(s2.reply("again"))
        XCTAssertGreaterThan(loader.rewinds, before,
            "loaded context did not continue its conversation")
        try? fm.removeItem(at: url)
    }

    // Precook + prime: cooking persists the prefilled system+tools prefix; a
    func testPrecookPrimeKVMatchesPlain() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("precook_\(UUID().uuidString).ctx")
        let voc = vocab([(1001, "Hi there.")])
        let plainB = TapeBackend(scripts: [[1001]], vocab: voc)
        let plain = ChatSession(backend: plainB, template: chatml,
                                system: "You are a bot.",
                                systemTail: "\nNow: T2", vocabSize: 256)
        _ = await drain(plain.reply("hello"))

        let cookB = TapeBackend(scripts: [[1001]], vocab: voc)
        let cook = ChatSession(backend: cookB, template: chatml,
                               system: "You are a bot.",
                               systemTail: "\nNow: T1", vocabSize: 256)
        try await cook.precook(to: url)

        let primeB = TapeBackend(scripts: [[1001]], vocab: voc)
        let primed = ChatSession(backend: primeB, template: chatml,
                                 system: "You are a bot.",
                                 systemTail: "\nNow: T2", vocabSize: 256)
        let hit = await primed.prime(from: url)
        XCTAssertTrue(hit, "stable prefix did not prime")
        _ = await drain(primed.reply("hello"))
        XCTAssertEqual(primeB.kvText, plainB.kvText,
            "primed KV diverges from a plain session's")
        XCTAssertTrue(primeB.kvText.contains("Now: T2"))
        XCTAssertFalse(primeB.kvText.contains("Now: T1"),
            "cook-day tail leaked into the primed KV")
        try? FileManager.default.removeItem(at: url)
    }

    // The reasoning sentence sits INSIDE the tools system block when tools are
    // present and in a block of its OWN when they are not, as Qwen3.8 does.
    private let effortTemplate = """
        {%- set ri = '' %}
        {%- if enable_thinking %}
        {%- if reasoning_effort|default('xhigh') != 'medium' %}
        {%- set ri = 'Reasoning effort is set to ' ~ \
        reasoning_effort|default('xhigh') ~ '.' %}
        {%- endif %}{%- endif %}
        {%- if tools %}
        {{- '<|im_start|>system\\n' }}
        {%- if ri %}{{- ri + '\\n\\n' }}{%- endif %}
        {{- '# Tools\\n' }}
        {%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}
        {%- if messages[0].role == 'system' %}
        {{- '\\n\\n' + messages[0].content }}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
        {%- elif messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + (ri + '\\n\\n' if ri else '') \
        + messages[0].content + '<|im_end|>\\n' }}
        {%- elif ri %}
        {{- '<|im_start|>system\\n' + ri + '<|im_end|>\\n' }}
        {%- endif %}
        {%- for m in messages %}{%- if m.role != 'system' %}
        {{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}
        {%- endif %}{%- endfor %}
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}
        {%- endif %}
        """

    // A 4B emitted exactly this, and it rendered in the bubble.
    func testUnmatchedToolCloseNeverReachesTheAnswer() async throws {
        let voc = vocab([(1, "The answer is 42."),
                         (2, "\n</tool_call>\n"),
                         (3, "Anything else?")])
        let backend = MockBackend(scripts: [[1, 2, 3]], vocab: voc)
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256,
            runner: SafeToolRunner(slugsPath: nil, wikipedia: false,
                                   network: false))
        let answer = await drain(session.reply("go"))
        XCTAssertFalse(answer.contains("</tool_call>"),
                       "unmatched close reached the reader: \(answer)")
        XCTAssertTrue(answer.contains("42"), answer)
        XCTAssertTrue(answer.contains("Anything else?"), answer)
    }

    // The guard must not touch a REAL call: its close belongs to an opener.
    func testMatchedToolCallStillRuns() async throws {
        let voc = vocab([(1, "<tool_call>\n<function=get_current_time>\n"),
                         (2, "</function>\n</tool_call>"),
                         (3, "Done.")])
        let backend = MockBackend(scripts: [[1, 2], [3]], vocab: voc)
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256,
            runner: SafeToolRunner(slugsPath: nil, wikipedia: false,
                                   network: false))
        // Round 1 is reached only via the rewind a completed call triggers.
        let answer = await drain(session.reply("go"))
        XCTAssertTrue(answer.contains("Done."),
                      "the call did not run: \(answer)")
    }

    func testPrecookSurvivesReasoningInstructions() async throws {
        let voc = vocab([(1001, "ok")])
        let runner = SafeToolRunner(slugsPath: nil, wikipedia: false,
                                    network: false)
        for effort in ["low", "xhigh", "medium"] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("precook_\(UUID().uuidString).ctx")
            let cook = ChatSession(
                backend: TapeBackend(scripts: [[1001]], vocab: voc),
                template: effortTemplate, system: "You are a bot.",
                systemTail: "\nNow: T1", vocabSize: 256,
                enableThinking: true, reasoningEffort: effort, runner: runner)
            try await cook.precook(to: url)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path),
                "effort=\(effort) cooked nothing")
            let primed = ChatSession(
                backend: TapeBackend(scripts: [[1001]], vocab: voc),
                template: effortTemplate, system: "You are a bot.",
                systemTail: "\nNow: T2", vocabSize: 256,
                enableThinking: true, reasoningEffort: effort, runner: runner)
            let hit = await primed.prime(from: url)
            XCTAssertTrue(hit, "effort=\(effort) did not prime")
            try? FileManager.default.removeItem(at: url)
        }
    }

    // What the cache may and may NOT be keyed on, asked of the render itself.
    func testPrecookStampFollowsTheRenderNotTheFlags() async throws {
        let voc = vocab([(1001, "ok")])
        func stamp(_ think: Bool, _ tools: Bool) async -> String {
            let s = ChatSession(
                backend: TapeBackend(scripts: [[1001]], vocab: voc),
                template: effortTemplate, system: "You are a bot.",
                vocabSize: 256, enableThinking: think,
                reasoningEffort: "medium",
                runner: SafeToolRunner(slugsPath: nil, wikipedia: false,
                                       network: tools))
            return await s.precookStamp
        }
        let onNoTools = await stamp(true, false)
        let offNoTools = await stamp(false, false)
        let onTools = await stamp(true, true)
        XCTAssertFalse(onNoTools.isEmpty)
        XCTAssertEqual(onNoTools, offNoTools,
            "thinking is in the key but moves no bytes")
        XCTAssertNotEqual(onNoTools, onTools,
            "the tool tier moves bytes but was not in the key")
    }

    // The sentence is IN the cooked prefix, so the levels cannot share a file.
    func testPrecookStampMovesWithEffort() async throws {
        let voc = vocab([(1001, "ok")])
        var bytes: [String: Int] = [:]
        for effort in ["low", "xhigh"] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("precook_\(UUID().uuidString).ctx")
            let cook = ChatSession(
                backend: TapeBackend(scripts: [[1001]], vocab: voc),
                template: effortTemplate, system: "You are a bot.",
                vocabSize: 256, enableThinking: true, reasoningEffort: effort)
            try await cook.precook(to: url)
            let other = ChatSession(
                backend: TapeBackend(scripts: [[1001]], vocab: voc),
                template: effortTemplate, system: "You are a bot.",
                vocabSize: 256, enableThinking: true,
                reasoningEffort: effort == "low" ? "xhigh" : "low")
            let crossed = await other.prime(from: url)
            XCTAssertFalse(crossed, "\(effort) primed the other level's file")
            bytes[effort] = (try? Data(contentsOf: url))?.count ?? 0
            try? FileManager.default.removeItem(at: url)
        }
        XCTAssertNotEqual(bytes["low"], bytes["xhigh"],
            "the two levels cooked identical files")
    }

    // A changed system prompt misses the stamp: prime returns false and the
    // session is untouched (a plain fresh first turn follows).
    func testPrimeMissesOnChangedPrompt() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("precook_\(UUID().uuidString).ctx")
        let voc = vocab([(1001, "ok")])
        let cook = ChatSession(
            backend: TapeBackend(scripts: [[1001]], vocab: voc),
            template: chatml, system: "You are a bot.", vocabSize: 256)
        try await cook.precook(to: url)
        let other = ChatSession(
            backend: TapeBackend(scripts: [[1001]], vocab: voc),
            template: chatml, system: "You are a DOG.", vocabSize: 256)
        let hit = await other.prime(from: url)
        XCTAssertFalse(hit, "a changed prompt must miss the stamp")
        try? FileManager.default.removeItem(at: url)
    }

    // A backend whose byte serialization is a no-op (MockBackend keeps the
    func testPrimeRejectsStatelessBackend() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("precook_\(UUID().uuidString).ctx")
        let voc = vocab([(1, "ok")])
        let cook = ChatSession(
            backend: MockBackend(scripts: [[1]], vocab: voc),
            template: chatml, system: "You are a bot.", vocabSize: 256)
        try await cook.precook(to: url)
        let primed = ChatSession(
            backend: MockBackend(scripts: [[1]], vocab: voc),
            template: chatml, system: "You are a bot.", vocabSize: 256)
        let hit = await primed.prime(from: url)
        XCTAssertFalse(hit, "stateless backend must not claim a prime")
        try? FileManager.default.removeItem(at: url)
    }

    // A send DURING the cook must wait it out, not interleave: the cook's
    func testSendDuringCookWaitsForGate() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("precook_\(UUID().uuidString).ctx")
        let voc = vocab([(1001, "ok")])
        let plainB = TapeBackend(scripts: [[1001]], vocab: voc)
        let plain = ChatSession(backend: plainB, template: chatml,
                                system: "You are a bot.", vocabSize: 256)
        _ = await drain(plain.reply("hello"))

        // Two script rounds: the cook's reset consumes round 0, the turn's
        // rewind advances to round 1.
        let raceB = TapeBackend(scripts: [[1001], [1001]], vocab: voc)
        raceB.extendDelayMs = 50
        let raced = ChatSession(backend: raceB, template: chatml,
                                system: "You are a bot.", vocabSize: 256)
        await raced.primeOrCook(at: url)          // cook path: no file yet
        let answer = await drain(raced.reply("hello"))
        XCTAssertEqual(answer, "ok")
        XCTAssertEqual(raceB.kvText, plainB.kvText,
            "a send interleaved with the cook (torn KV)")
        try? FileManager.default.removeItem(at: url)
    }

    // Precook + prime against the REAL shipped Qwen3.5 template (skips when no
    // set is on disk).
    func testPrecookPrimeWithRealQwenTemplate() async throws {
        guard let tmpl = try realQwenTemplate() else {
            throw XCTSkip("no Qwen3.5-0.8B set on disk")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("precook_\(UUID().uuidString).ctx")
        let voc = vocab([(1001, "Four.")])
        let plainB = TapeBackend(scripts: [[1001]], vocab: voc)
        let plain = ChatSession(backend: plainB, template: tmpl,
                                system: "You are a bot.",
                                systemTail: "\nNow: T2", vocabSize: 256)
        _ = await drain(plain.reply("2+2?"))

        let cook = ChatSession(
            backend: TapeBackend(scripts: [[1001]], vocab: voc),
            template: tmpl, system: "You are a bot.",
            systemTail: "\nNow: T1", vocabSize: 256)
        try await cook.precook(to: url)

        let primeB = TapeBackend(scripts: [[1001]], vocab: voc)
        let primed = ChatSession(backend: primeB, template: tmpl,
                                 system: "You are a bot.",
                                 systemTail: "\nNow: T2", vocabSize: 256)
        let hit = await primed.prime(from: url)
        XCTAssertTrue(hit, "real-template prefix did not prime")
        _ = await drain(primed.reply("2+2?"))
        XCTAssertEqual(primeB.kvText, plainB.kvText,
            "primed KV diverges from plain under the real template")
        try? FileManager.default.removeItem(at: url)
    }

    // The n-gram loop breaker ends a runaway: a backend that emits a 2-token
    func testLoopBreakerEndsRunaway() async throws {
        let cycle: [Int32] = (0 ..< 80).map { i in i % 2 == 0 ? 7 : 8 }
        let backend = MockBackend(
            scripts: [cycle], vocab: vocab([(7, "a"), (8, "b")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        let answer = await drain(session.reply("go"))
        // 5x the 2-gram "ab" (10 tokens) trips the breaker; far short of 80.
        XCTAssertLessThan(answer.count, 24, "loop breaker did not stop: \(answer)")
        XCTAssertGreaterThan(answer.count, 0, "stopped before any output")
    }

    // A speculative backend with committed-but-undelivered tokens: the soft
    private final class SpecQueueMock: AgentBackend, @unchecked Sendable {
        let eos: Int32 = -1
        private(set) var pending = 2
        private(set) var pendingAtInject = -1
        private var pos = 0
        private var step = 0
        // Endless "line\n" think tokens keep softOver armed every iteration.
        private let think: Int32 = 900

        func encode(_ text: String) -> [Int32] {
            text.utf8.map { byte in Int32(byte) }
        }
        func tokenBytes(_ id: Int32) -> [UInt8] {
            id == think ? Array("line\n".utf8)
                        : (id >= 0 && id < 256 ? [UInt8(id)] : [])
        }
        func text(_ ids: [Int32]) -> String {
            var b: [UInt8] = []
            for id in ids { b.append(contentsOf: tokenBytes(id)) }
            return String(decoding: b, as: UTF8.self)
        }
        var position: Int { get async { pos } }
        func reset() async { pos = 0 }
        func useSampler(_ s: Sampler?) async {}
        func extend(_ ids: [Int32]) async throws -> Int32 {
            pos += ids.count
            if text(ids).contains("</think>") && pendingAtInject < 0 {
                pendingAtInject = pending
            }
            // Post-injection the answer ends at once (eos).
            return pendingAtInject >= 0 ? eos : think
        }
        func mark() async throws {}
        func rewind() async throws {}
        func decode(_ token: Int32) async throws -> Int32 {
            if pending > 0 { pending -= 1 }
            pos += 1
            step += 1
            return think
        }
        func queuedCount() async -> Int { pending }
        struct S: BackendState {}
        func saveState() async throws -> any BackendState { S() }
        func loadState(_ state: any BackendState) async throws {}
    }

    func testThinkInjectionWaitsForSpecQueue() async throws {
        let backend = SpecQueueMock()
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true, softReasoningCap: 1)
        _ = await drain(session.reply("go"))
        XCTAssertEqual(backend.pendingAtInject, 0,
            "</think> was injected with spec tokens still queued")
    }

    // A markdown table separator row repeats "|---" once per column -- far past
    func testLoopBreakerToleratesStructuralRuns() {
        let bytesOf: [Int32: [UInt8]] = [
            1: Array("|".utf8), 2: Array("---".utf8), 3: Array("so".utf8),
        ]
        let bytes: (Int32) -> [UInt8] = { id in bytesOf[id] ?? [] }
        let tableRow: [Int32] = Array(
            repeating: [1, 2], count: 20).flatMap { $0 }
        XCTAssertFalse(
            Continuation.isLooping(tableRow, tokenBytes: bytes),
            "20-column table separator tripped the loop breaker")
        let runaway: [Int32] = Array(
            repeating: [1, 2], count: 30).flatMap { $0 }
        XCTAssertTrue(
            Continuation.isLooping(runaway, tokenBytes: bytes),
            "a structural runaway must still trip, just later")
        let textLoop: [Int32] = Array(repeating: 3, count: 6)
        XCTAssertTrue(
            Continuation.isLooping(textLoop, tokenBytes: bytes),
            "ordinary text cycles keep the tight threshold")
        XCTAssertTrue(
            Continuation.isLooping(tableRow),
            "without a byte view the tight threshold applies")
    }

    // A whole re-emitted sentence cycles with a period the 4-gram scan cannot
    func testLoopBreakerCatchesLongPhraseCycles() {
        let block: [Int32] = [10, 11, 12, 13, 14, 15, 16]
        XCTAssertFalse(Continuation.isLooping(block + block),
                       "one repetition is not a loop")
        XCTAssertTrue(Continuation.isLooping(block + block + block),
                      "three identical 7-token blocks must trip")
        let paragraph: [Int32] = (100 ..< 145).map { i in Int32(i) }
        XCTAssertTrue(
            Continuation.isLooping(paragraph + paragraph + paragraph),
            "a 45-token paragraph cycle must trip")
    }

    // A thinking turn that ends (Stop / EOS) BEFORE </think> must commit an
    // EMPTY answer.
    func testInterruptedThinkingCommitsNoAnswer() async throws {
        let think = (10 ..< 210).map { i in Int32(i) }   // no </think>
        var pairs = think.map { id in (id, "r") }
        pairs.append((200, "ok"))
        let backend = MockBackend(scripts: [think, [200]], vocab: vocab(pairs))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        _ = await drain(session.reply("go"))     // ends mid-think, no </think>
        let before = backend.extendedTokens
        _ = await drain(session.reply("next"))
        let delta = backend.extendedTokens - before
        XCTAssertLessThan(delta, 130,
            "interrupted reasoning was committed and re-prefilled: \(delta)")
    }

    // A prefill cancelled by Stop (extend throws EngineError.stopped) fully
    func testPrefillStopRollsBackTurn() async throws {
        let backend = MockBackend(
            scripts: [[1], [98], [2]],
            vocab: vocab([(1, "one"), (98, "x"), (2, "two")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        _ = await drain(session.reply("first"))          // commits turn 1
        backend.stopNextExtend = true                    // stop turn 2
        let rolledOut = await drain(session.reply("second"))
        XCTAssertTrue(rolledOut.isEmpty, "rolled-back turn emitted output")
        let rolledBack = await session.turnRolledBack
        XCTAssertTrue(rolledBack, "turn not flagged rolled back")
        let before = backend.rewinds
        let a3 = await drain(session.reply("third"))     // must continue turn 1
        XCTAssertGreaterThan(backend.rewinds, before,
            "recovered turn did not rewind into the prior conversation")
        XCTAssertEqual(a3, "two", "conversation did not recover after rollback")
    }

    // A turn that decodes to NOTHING (Stop after the prefill, before the first
    func testEmptyAnswerRollsBackTurn() async throws {
        let backend = MockBackend(
            scripts: [[1], [], [2]],
            vocab: vocab([(1, "one"), (2, "two")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256)
        let a1 = await drain(session.reply("first"))
        XCTAssertEqual(a1, "one")
        let empty = await drain(session.reply("second"))   // decodes nothing
        XCTAssertTrue(empty.isEmpty, "empty turn emitted output")
        let rolledBack = await session.turnRolledBack
        XCTAssertTrue(rolledBack, "empty turn was not rolled back")
        let before = backend.rewinds
        let a3 = await drain(session.reply("third"))
        XCTAssertGreaterThan(backend.rewinds, before,
            "recovered turn did not rewind into the prior conversation")
        XCTAssertEqual(a3, "two",
                       "conversation did not recover after the empty turn")
    }

    // Tool-name resolution: exact match, a close typo snaps to the real tool,
    func testResolveToolFuzzyAndUnknown() {
        let tools = ["get_current_time", "calculator",
                     "wikipedia_query", "web_search"]
        XCTAssertEqual(ChatSession.resolveTool("calculator", tools),
                       "calculator")
        XCTAssertEqual(ChatSession.resolveTool("web_serch", tools),
                       "web_search")
        XCTAssertEqual(ChatSession.resolveTool("calcultor", tools),
                       "calculator")
        // Known alias: tavily_search (QwenPaw's real Tavily tool) and the
        // observed google_web_search map to ours when it is advertised.
        XCTAssertEqual(ChatSession.resolveTool("tavily_search", tools),
                       "web_search")
        XCTAssertEqual(ChatSession.resolveTool("google_web_search", tools),
                       "web_search")
        // ... but not when web_search is not offered (airplane) -- stays
        // unknown.
        XCTAssertNil(ChatSession.resolveTool("tavily_search",
                                             ["get_current_time", "calculator"]))
        XCTAssertNil(ChatSession.resolveTool("wikipedia-search-function", tools))
        XCTAssertNil(ChatSession.resolveTool("send_email", tools))
    }

    // Thread-safe accumulator for trace events (the sink fires on the
    // session's actor while the test drains the stream elsewhere).
    private final class TraceBox: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [TraceEvent] = []
        func add(_ e: TraceEvent) { lock.lock(); items.append(e); lock.unlock() }
        var list: [TraceEvent] {
            lock.lock(); defer { lock.unlock() }; return items
        }
    }

    // The structured trace covers a whole tool turn, in order: user, fresh
    func testTraceEventsCoverToolTurn() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, call), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let box = TraceBox()
        await session.setTrace { e in box.add(e) }
        _ = await drain(session.reply("what is 2 + 2?"))
        let kinds = box.list.map { e in e.kind }
        XCTAssertEqual(kinds, [.user, .render, .prefill, .decode,
                               .toolCall, .toolResult, .rewind, .render,
                               .prefill, .decode, .answer],
                       "trace sequence off: \(kinds)")
        let decodes = box.list.filter { e in e.kind == .decode }
        XCTAssertTrue(decodes.first?.summary.hasPrefix("tool-call") == true)
        XCTAssertTrue(decodes.last?.summary.hasPrefix("eos") == true)
        let render = box.list.first { e in e.kind == .render }
        XCTAssertTrue(render?.text.contains("what is 2 + 2?") == true,
                      "render payload is not the laid delta")
        XCTAssertEqual(box.list.last?.text, "The answer is 4.")
    }

    // A tool call emitted INSIDE <think> without ever closing it (observed
    func testToolCallInsideThinkDispatches() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1, 2], [3]],
            vocab: vocab([(1, "Let me compute. "), (2, call),
                          (3, "</think>It is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true, runner: runner)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("2+2?", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(runner.calls, ["calculator"])
        XCTAssertTrue(content.contains("It is 4."), "answer lost: \(content)")
        XCTAssertFalse(reasoning.text.contains("<tool_call>"),
            "raw call markup leaked into reasoning: \(reasoning.text)")
        XCTAssertFalse(content.contains("<tool_call>"),
            "raw call markup leaked into content: \(content)")
    }

    // EOS while still inside <think> with no content is never a finished turn
    func testEosInsideThinkRescued() async throws {
        let backend = TapeBackend(
            scripts: [[1001, -1, 1002]],
            vocab: vocab([(1001, "Still weighing it. "),
                          (1002, "So the summary stands. ")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("summarize",
                                   onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertTrue(reasoning.text.contains("Still weighing it."),
            "the pre-rescue tokens are reasoning: \(reasoning.text)")
        XCTAssertEqual(content.trimmingCharacters(in: .whitespaces),
                       "So the summary stands.",
            "the post-rescue decode must become the answer")
    }

    func testAnswerlessTurnIsNotReportedAsAUserStop() async throws {
        let backend = TapeBackend(
            scripts: [[1001]],
            vocab: vocab([(1001, "Still weighing it. ")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let answer = await drain(session.reply("summarize"))
        XCTAssertEqual(answer, "", "the turn produced no answer")
        let outcome = await session.turnOutcome
        XCTAssertEqual(outcome, .answerless,
                       "no answer is not the same as the user pressing Stop")
    }

    // A second <tool_call> opener before the first block closes (observed
    func testReopenImplicitlyClosesToolCall() async throws {
        let merged = "<tool_call><function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function>\n"
            + "<tool_call><function=get_current_time></function></tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, merged), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let events = EventBox()
        let answer = await drain(session.reply(
            "2+2?", onToolRound: { e in events.add(e) }))
        XCTAssertEqual(runner.calls, ["calculator"],
            "the merged span must dispatch the FIRST call alone")
        XCTAssertEqual(events.list.first?.params.map { p in p.name },
                       ["expression"],
            "second call's params bled into the first")
        XCTAssertEqual(answer, "The answer is 4.")
    }

    // A call opener that never resolves (EOS mid-call) must NOT be committed to
    func testUnterminatedCallNotCommitted() async throws {
        let openerOnly = "<tool_call><function=web_search>"
            + "<parameter=query>x</parameter>"     // no </tool_call>, EOS
        let backend = TapeBackend(
            scripts: [[1001, 1002], [1003]],
            vocab: vocab([(1001, "I looked around. "), (1002, openerOnly),
                          (1003, "Done.")]))
        let runner = RecordingRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let a1 = await drain(session.reply("go"))
        XCTAssertEqual(runner.calls, [], "unterminated call must not run")
        XCTAssertFalse(a1.contains("<tool_call>"),
            "raw opener leaked into the stream: \(a1)")
        _ = await drain(session.reply("next"))
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("I looked around.<|im_end|>"),
            "the pre-opener text was not committed: \(kv)")
        XCTAssertFalse(kv.contains("<tool_call>"),
            "unterminated call markup entered the KV: \(kv)")
    }

    // The observed 4B failure shape: a call whose body COMPLETED (</function>
    // emitted, spaced opener and all) but EOS landed before </tool_call>.
    func testEosCutCompleteBodyDispatches() async throws {
        let cut = "<tool_call>\n< function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, cut), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("2+2?"))
        XCTAssertEqual(runner.calls, ["calculator"],
            "EOS-cut complete-body call must dispatch")
        XCTAssertEqual(answer, "The answer is 4.")
    }

    // Sequential offset-pages of one URL are a single logical lookup: twelve
    func testFetchPagingChainCountsAsOneRound() async throws {
        func page(_ offset: Int) -> String {
            "<tool_call><function=fetch_url>"
                + "<parameter=url>example.com/a</parameter>"
                + "<parameter=offset>\(offset)</parameter>"
                + "</function></tool_call>"
        }
        var vocabPairs: [(Int32, String)] = [(2000, "Done.")]
        var scripts: [[Int32]] = []
        for i in 0 ..< 12 {
            vocabPairs.append((Int32(1000 + i), page(i * 100)))
            scripts.append([Int32(1000 + i)])
        }
        scripts.append([2000])
        let backend = MockBackend(scripts: scripts, vocab: vocab(vocabPairs))
        let runner = SchemaRunner(
            name: "fetch_url",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"url\":{\"type\":\"string\"},"
                + "\"offset\":{\"type\":\"integer\"}},"
                + "\"required\":[\"url\"]}",
            reply: "page text")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("read example.com/a"))
        XCTAssertEqual(runner.calls.count, 12,
            "the paging chain must run every page")
        XCTAssertEqual(answer, "Done.")
    }

    // Per-turn tool state (the wikipedia dedupe memo) is turn-scoped: the
    // session calls the runner's beginTurn exactly once per user turn.
    func testRunnerBeginTurnPerTurn() async throws {
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, "Hi."), (2, "Again.")]))
        let runner = RecordingRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        _ = await drain(session.reply("a"))
        XCTAssertEqual(runner.beginTurns, 1)
        _ = await drain(session.reply("b"))
        XCTAssertEqual(runner.beginTurns, 2)
    }

    // Round 5 of the observed rot: a spaced close ("</function >") with junk
    // trailing it, then EOS.
    func testEosCutSpacedCloseDispatches() async throws {
        let cut = "<tool_call>\n<function=calculator>\n"
            + "<parameter=expression>\n2 + 2\n</parameter>\n"
            + "</function > 0, \" junk trailing"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, cut), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("2+2?"))
        XCTAssertEqual(runner.calls, ["calculator"],
            "spaced-close EOS-cut call must dispatch")
        XCTAssertEqual(answer, "The answer is 4.")
    }

    // A COMPLETE but unparseable block (no function tag) as the whole answer:
    func testUnparseableBlockAloneNotCommitted() async throws {
        let junk = "<tool_call>\nno function here\n</tool_call>"
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, junk), (1002, "Later.")]))
        let runner = RecordingRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let a1 = await drain(session.reply("go"))
        XCTAssertEqual(runner.calls, [], "junk block must not dispatch")
        XCTAssertEqual(a1, "", "junk block leaked into the stream: \(a1)")
        _ = await drain(session.reply("next"))
        let kv = backend.kvText
        XCTAssertFalse(kv.contains("no function here"),
            "unparseable block entered the KV: \(kv)")
    }

    // The canonical re-lay carries ONLY the resolved tool's advertised
    func testRelayFiltersHallucinatedArgs() async throws {
        let call = "<tool_call><function=lookup>"
            + "<parameter=q>dark matter</parameter>"
            + "<parameter=empty></parameter>"
            + "<parameter=arg_value>[\"junk\"]</parameter>"
            + "</function></tool_call>"
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, call), (1002, "Found it.")]))
        // A template that renders the arguments JSON, so the laid wire is
        // assertable; a runner whose spec advertises only `query`.
        let argsTemplate = "{% for m in messages %}"
            + "{% if m.role == 'tool' %}"
            + "<|im_start|>user\n<tool_response>\n{{ m.content }}\n"
            + "</tool_response><|im_end|>\n"
            + "{% else %}"
            + "<|im_start|>{{ m.role }}\n{{ m.content }}"
            + "{% for tc in m.tool_calls %}"
            + "<tool_call>{{ tc.name }} {{ tc.arguments | tojson }}"
            + "</tool_call>"
            + "{% endfor %}"
            + "<|im_end|>\n"
            + "{% endif %}"
            + "{% endfor %}"
            + "{% if add_generation_prompt %}<|im_start|>assistant\n"
            + "{% endif %}"
        let runner = SchemaRunner(
            name: "lookup",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"query\":{\"type\":\"string\"}},"
                + "\"required\":[\"query\"]}",
            reply: "ok")
        let session = ChatSession(
            backend: backend, template: argsTemplate,
            system: "You are a bot.", vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("find dark matter"))
        XCTAssertEqual(answer, "Found it.")
        let kv = backend.kvText
        XCTAssertTrue(kv.contains("\"query\": \"dark matter\""),
            "the alias was not canonicalized into the re-lay: \(kv)")
        XCTAssertFalse(kv.contains("arg_value"),
            "hallucinated argument entered the KV: \(kv)")
        XCTAssertFalse(kv.contains("empty"),
            "empty argument entered the KV: \(kv)")
    }

    // One advertised tool with a real schema, for the re-lay filter test.
    private final class SchemaRunner: ToolRunner, @unchecked Sendable {
        let tools: [ToolSpec]
        private let reply: String
        private(set) var calls: [String] = []

        init(name: String, parametersJSON: String, reply: String) {
            self.tools = [ToolSpec(name: name, description: "test tool.",
                                   parametersJSON: parametersJSON)]
            self.reply = reply
        }

        func execute(_ name: String, _ args: [ToolArg]) async -> String {
            calls.append(name)
            return reply
        }
    }

    // A loop INSIDE <think> (a "0.00000..." digit run the digit-exempt sampler
    func testThinkLoopRescuedAndToolStillRuns() async throws {
        let call = "<tool_call><function=calculator>"
            + "<parameter=expression>6000000 / 25</parameter>"
            + "</function></tool_call>"
        // Digits are structural bytes, so the digit run must pass the
        // RELAXED threshold (structuralReps) before the breaker sees it.
        var script: [Int32] = Array(repeating: 1007, count: 24)
        script.append(1001)
        let backend = TapeBackend(
            scripts: [script, [1003, -1, 1002]],
            vocab: vocab([(1007, "0"), (1001, call),
                          (1003, "Working it out. "),
                          (1002, "240000 mice.")]))
        let runner = RecordingRunner(reply: "240000")
        let session = ChatSession(
            backend: backend, template: chatml, system: "You are a bot.",
            vocabSize: 256, enableThinking: true, runner: runner)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("mice?",
                                   onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(runner.calls, ["calculator"],
            "the rescued turn must still dispatch the call")
        XCTAssertTrue(content.contains("240000 mice."),
            "answer lost after think-loop rescue: \(content)")
    }

    // An open <tool_call> whose body free-runs varied (non-repeating) content
    func testOpenCallRunawayEndsTurn() async throws {
        var vocabPairs: [(Int32, String)] = [
            (1, "Working on it. "),
            (2, "<tool_call><function=calculator>"
                + "<parameter=expression>"),
        ]
        var script: [Int32] = [1, 2]
        // ~19KB of varied body bytes, no cycle for the breaker to catch.
        for i in 0 ..< 1600 {
            vocabPairs.append((Int32(100 + i), "value\(i * 7 + 13)x "))
            script.append(Int32(100 + i))
        }
        let backend = MockBackend(
            scripts: [script], vocab: vocab(vocabPairs))
        let runner = RecordingRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("go"))
        XCTAssertEqual(runner.calls, [], "truncated call must not run")
        XCTAssertEqual(answer.trimmingCharacters(in: .whitespaces),
                       "Working on it.")
        let m = await session.lastMetrics
        XCTAssertEqual(m.endReason, "tool-runaway")
        XCTAssertLessThan(m.thinkTokens + m.contentTokens, 1100,
            "the cap did not cut the runaway early")
    }

    // A quoted pair with no <function=> body is not a call: it must not
    func testQuotedToolCallPairStreamsOn() async throws {
        let backend = MockBackend(
            scripts: [[1, 2]],
            vocab: vocab([(1, "A call looks like <tool_call>x</tool_call>. "),
                          (2, "Use it wisely.")]))
        let runner = RecordingRunner(reply: "unused")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("how do tool calls look?"))
        XCTAssertEqual(runner.calls, [], "quoted pair dispatched a tool")
        XCTAssertTrue(answer.contains("Use it wisely."),
            "stream froze after a quoted tool-call pair: \(answer)")
        XCTAssertTrue(answer.contains("<tool_call>x</tool_call>"),
            "quoted markup vanished from the answer: \(answer)")
    }

    // "</think>" split across two tokens: the partial "</th" tail must be held
    func testSplitThinkMarkerDoesNotLeak() async throws {
        let backend = MockBackend(
            scripts: [[10, 11, 12, 13]],
            vocab: vocab([(10, "why"), (11, "</th"),
                          (12, "ink>"), (13, "answer")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("go", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(reasoning.text, "why", "marker leaked into reasoning")
        XCTAssertEqual(content, "answer")
    }

    func testThinkSplitMetrics() async throws {
        // Thinking mode: the decoded stream is reasoning until </think>, then
        // the answer.
        let backend = MockBackend(
            scripts: [[10, 11, 12, 13]],
            vocab: vocab([(10, "reasoning "), (11, "here"),
                          (12, "</think>"), (13, "The answer.")]))
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let streamed = await drain(session.reply("go"))
        let m = await session.lastMetrics
        XCTAssertEqual(m.thinkTokens, 2)
        XCTAssertEqual(m.contentTokens, 2)
        XCTAssertTrue(streamed.contains("The answer."))
        XCTAssertFalse(streamed.contains("here"), "reasoning in content stream")
    }

    private let visionChatml = "{% for m in messages %}"
        + "<|im_start|>{{ m.role }}\n"
        + "{% if m.content is string %}{{ m.content }}{% else %}"
        + "{% for item in m.content %}"
        + "{% if item.type == 'image' %}"
        + "<|vision_start|><|image_pad|><|vision_end|>"
        + "{% elif item.type == 'text' %}{{ item.text }}{% endif %}"
        + "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    func testSoftTurnKeepsTheTemplatesImagePad() async throws {
        let backend = TapeBackend(
            scripts: [[1001]],
            vocab: vocab([(1001, "A dog on a beach."),
                          (999, "<|image_pad|>")]))
        let session = ChatSession(
            backend: backend, template: visionChatml,
            system: "You are a bot.", vocabSize: 256)
        let pad = backend.encode("<|image_pad|>")[0]
        let span = SoftSpan(placeholder: pad,
                            ids: [Int32](repeating: pad, count: 3),
                            features: [Float](repeating: 0, count: 6))
        let answer = await drain(session.replySoft(
            "what is this?", parts: [.image, .text("what is this?")],
            spans: [span], labelled: false))
        XCTAssertEqual(answer, "A dog on a beach.")
        XCTAssertTrue(backend.kvText.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|>"
            + "<|vision_end|>what is this?"), backend.kvText)
    }

    // The older-Qwen3 JSON dialect: a <tool_call> body carrying {"name":..,
    func testParseJSONDialectToolCall() {
        let body = "\n{\"name\": \"calculator\", "
            + "\"arguments\": {\"expression\": \"2 + 2\"}}\n"
        let call = Tools.parse(Substring(body))
        XCTAssertEqual(call?.functionName, "calculator")
        XCTAssertEqual(call?.params.first?.name, "expression")
        XCTAssertEqual(call?.params.first?.value, "2 + 2")
    }

    // A JSON integer argument must stringify WITHOUT a ".0" tail, so a tool
    func testParseJSONDialectNumericArg() {
        let body = "{\"name\": \"fetch_url\", \"arguments\": "
            + "{\"url\": \"example.com\", \"offset\": 16384}}"
        let call = Tools.parse(Substring(body))
        XCTAssertEqual(call?.functionName, "fetch_url")
        let offset = call?.params.first { p in p.name == "offset" }?.value
        XCTAssertEqual(offset, "16384", "integer arg gained a .0 tail")
    }

    // The XML dialect (QwenPaw/3.5/3.6) still parses unchanged: the sniff
    // routes a '<'-leading body to the <function=> scan.
    func testParseXMLDialectStillWorks() {
        let body = "<function=calculator>"
            + "<parameter=expression>2 + 2</parameter></function>"
        let call = Tools.parse(Substring(body))
        XCTAssertEqual(call?.functionName, "calculator")
        XCTAssertEqual(call?.params.first?.value, "2 + 2")
    }

    // The offline tool loop over a JSON-dialect model: the model emits a JSON
    func testJSONDialectToolCallDispatchesThenAnswers() async throws {
        let call = "<tool_call>\n{\"name\": \"calculator\", "
            + "\"arguments\": {\"expression\": \"2 + 2\"}}\n</tool_call>"
        let backend = MockBackend(
            scripts: [[1], [2]],
            vocab: vocab([(1, call), (2, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: template, system: "You are a bot.",
            vocabSize: 256, runner: runner)
        let answer = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(runner.calls, ["calculator"])
        XCTAssertEqual(answer, "The answer is 4.")
    }

    // A JSON-dialect chat template (the 1.7B's own shape: it renders tool_calls
    // as {"name":.., "arguments":..|tojson}).
    private let jsonChatml = "{% for m in messages %}"
        + "{% if m.role == 'tool' %}"
        + "<|im_start|>user\n<tool_response>\n{{ m.content }}\n"
        + "</tool_response><|im_end|>\n"
        + "{% else %}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}"
        + "{% for tc in m.tool_calls %}"
        + "<tool_call>\n{\"name\": \"{{ tc.name }}\", \"arguments\": "
        + "{{ tc.arguments | tojson }}}\n</tool_call>"
        + "{% endfor %}"
        + "<|im_end|>\n"
        + "{% endif %}"
        + "{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    func testJSONDialectToolRoundRelaysJSON() async throws {
        let call = "<tool_call>\n{\"name\": \"calculator\", "
            + "\"arguments\": {\"expression\": \"2 + 2\"}}\n</tool_call>"
        let backend = TapeBackend(
            scripts: [[1001], [1002]],
            vocab: vocab([(1001, call), (1002, "The answer is 4.")]))
        let runner = RecordingRunner(reply: "4")
        let session = ChatSession(
            backend: backend, template: jsonChatml,
            system: "You are a bot.", vocabSize: 256, runner: runner)
        let a1 = await drain(session.reply("what is 2 + 2?"))
        XCTAssertEqual(a1, "The answer is 4.")
        XCTAssertTrue(backend.kvText.contains(
            "<tool_call>\n{\"name\": \"calculator\", \"arguments\": "
            + "{\"expression\": \"2 + 2\"}}\n</tool_call>"),
            "JSON tool call did not re-lay in the JSON dialect:\n"
            + backend.kvText)
    }

    // Older Qwen3 (the dense 1.7B) template bakes a CLOSED empty
    // <think></think> into the gen prompt EVEN with thinking on.
    private let closedThinkTemplate = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n{%- endif -%}"

    func testClosedThinkGenPromptDecodesToContent() async throws {
        let backend = MockBackend(
            scripts: [[1, 2, 3]],
            vocab: vocab([(1, "Once upon "), (2, "a time "),
                          (3, "the end.")]))
        let session = ChatSession(
            backend: backend, template: closedThinkTemplate,
            system: "You are a bot.", vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("tell me a fairytale",
                                   onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertEqual(content, "Once upon a time the end.",
            "closed-think gen prompt misrouted the answer to reasoning")
        XCTAssertEqual(reasoning.text, "",
            "no reasoning region, yet text reached the reasoning channel")
    }

    // The complement: an OPEN-think gen prompt (Qwen3.5 thinking on) still
    func testOpenThinkGenPromptRoutesReasoning() async throws {
        let openThink = "{%- for m in messages -%}"
            + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            + "{%- endfor -%}{%- if add_generation_prompt -%}"
            + "<|im_start|>assistant\n<think>\n{%- endif -%}"
        let backend = MockBackend(
            scripts: [[10, 11, 12, 13]],
            vocab: vocab([(10, "reasoning "), (11, "here"),
                          (12, "</think>"), (13, "The answer.")]))
        let session = ChatSession(
            backend: backend, template: openThink, system: "You are a bot.",
            vocabSize: 256, enableThinking: true)
        let reasoning = Box()
        var content = ""
        let stream = session.reply("go", onReasoning: { r in reasoning.add(r) })
        for await piece in stream { content += piece }
        XCTAssertTrue(reasoning.text.contains("reasoning here"),
                      "open-think prompt lost its reasoning: \(reasoning.text)")
        XCTAssertEqual(content, "The answer.")
    }
}
