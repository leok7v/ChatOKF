import Chat
import LLM
import XCTest
@testable import ChatOKF

final class ModelNamingTests: XCTestCase {

    private let both = ["gemma-4-12B", "gemma-4-12B-MTP"]
    private let one = ["gemma-4-12B-MTP", "Ternary-Bonsai-1.7B"]

    func testMtpIsHiddenWhenNothingSharesTheFamily() {
        XCTAssertEqual(Models.display("gemma-4-12B-MTP", among: one),
                       "Gemma 12B")
    }

    func testMtpAppearsWhenBothBuildsAreVisible() {
        XCTAssertEqual(Models.display("gemma-4-12B", among: both),
                       "Gemma 12B")
        XCTAssertEqual(Models.display("gemma-4-12B-MTP", among: both),
                       "Gemma 12B MTP")
    }

    func testQuantSuffixOnlyWhenRungsCollide() {
        let rungs = ["Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ4_XS"]
        XCTAssertEqual(Models.display("Qwen3.8-27B-IQ1_S", among: rungs),
                       "Qwen3.8 27B 1-bit")
        XCTAssertEqual(Models.display("Qwen3.8-27B-IQ1_S",
                                      among: ["Qwen3.8-27B-IQ1_S"]),
                       "Qwen3.8 27B")
    }

    func testProseNameNeverCarriesATag() {
        XCTAssertEqual(Models.display("gemma-4-E2B-MTP"), "Gemma E2B")
        XCTAssertEqual(Models.display("gemma-4-E4B-MTP"), "Gemma E4B")
    }

    func testTheSpeechFloorFallbackIsTheKnobDefault() {
        let knob = Flags.registry.first { k in k.name == "speech-floor" }
        XCTAssertEqual(knob?.defaultValue,
                       String(VoicePlayer.speechFloorDefault))
    }

    func testOnlyTheHybridsOf4BAndUpShiftAFullContext() {
        for name in ["Qwen3.5-4B", "Qwen3.5-9B", "Ternary-Bonsai-2-4B",
                     "Ternary-Bonsai-2-9B", "Ternary-Bonsai-2-27B"] {
            XCTAssertEqual(Models.shift(name), .soft, name)
        }
        for name in ["Qwen3.5-2B", "Ternary-Bonsai-2-2B", "gemma-4-E2B",
                     "gemma-4-E4B", "gemma-4-12B", "Qwen3.8-27B-IQ1_S",
                     "Ternary-Bonsai-1.7B", "Ternary-Bonsai-27B"] {
            XCTAssertEqual(Models.shift(name), .off, name)
        }
    }

}

private final class Calls: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var count: Int { lock.withLock { n } }
    func bump() { lock.withLock { n += 1 } }
}

@MainActor final class WarmedTests: XCTestCase {

    func testAWarmThatMadeNothingIsTriedAgain() async {
        let warmed = Warmed<Int>()
        let calls = Calls()
        let first = await warmed.take {
            calls.bump()
            return nil
        }.value
        XCTAssertNil(first)
        var second: Int? = nil
        let deadline = Date().addingTimeInterval(2)
        while second == nil && Date() < deadline {
            second = await warmed.take {
                calls.bump()
                return 7
            }.value
            await Task.yield()
        }
        XCTAssertEqual(second, 7, "a failed load stayed cached")
        let kept = await warmed.take { 9 }.value
        XCTAssertEqual(kept, 7, "a good load was not kept")
    }

}

final class AttachmentRefsTests: XCTestCase {

    func testInsertThenStripLeavesThePlainText() {
        let r = AttachmentRefs.insert("a.png", into: "look", at: 4)
        XCTAssertTrue(r.text.contains("a.png"))
        XCTAssertEqual(AttachmentRefs.names(in: r.text), ["a.png"])
        XCTAssertEqual(
            AttachmentRefs.stripped(r.text)
                .trimmingCharacters(in: .whitespaces),
            "look @a.png")
    }

    func testScrubRemovesTheWholeToken() {
        let r = AttachmentRefs.insert("a.png", into: "", at: 0)
        XCTAssertEqual(
            AttachmentRefs.scrub("a.png", from: r.text)
                .trimmingCharacters(in: .whitespaces), "")
    }

    func testSubstituteReplacesByName() {
        let r = AttachmentRefs.insert("doc.md", into: "read", at: 4)
        let out = AttachmentRefs.substitute(r.text) { name in
            "<\(name)>"
        }
        XCTAssertTrue(out.contains("<doc.md>"))
    }

}

@MainActor final class ZoomTests: XCTestCase {

    func testZoomClampsToTheDeclaredLimit() {
        XCTAssertEqual(ChatModel.clampZoom(9), ChatModel.zoomLimit)
        XCTAssertEqual(ChatModel.clampZoom(-9), -ChatModel.zoomLimit)
    }

    func testNotchRoundTripsThroughScale() {
        for notch in -ChatModel.zoomLimit ... ChatModel.zoomLimit {
            let scale = ChatModel.zoomScale(notch)
            XCTAssertEqual(ChatModel.zoomNotch(nearest: scale), notch)
        }
    }

}

@MainActor final class DownloadBarTests: XCTestCase {

    func testThePlanIsTheModelBytesOnly() throws {
        let bytes = try XCTUnwrap(ModelCatalog.source(Models.start)?.bytes)
        XCTAssertEqual(ChatModel.plannedBytes(Models.start), bytes)
        XCTAssertEqual(ChatModel.plannedBytes("no-such-model"), 0)
    }

    func testTheBarFillsWhenTheModelLands() throws {
        let bytes = try XCTUnwrap(ModelCatalog.source(Models.start)?.bytes)
        let first = ChatModel.barProgress(done: 0, total: 0, planned: bytes)
        XCTAssertEqual(first.total, bytes)
        let last = ChatModel.barProgress(done: bytes, total: bytes,
                                         planned: bytes)
        XCTAssertEqual(last.done, last.total)
    }

    func testTheMissingPacksComeBeforeTheModel() {
        let steps = ChatModel.downloadSteps(
            "m", packs: [.encoder, .voice, .hearing])
        XCTAssertEqual(steps, [.pack(.encoder), .pack(.voice),
                               .pack(.hearing), .model("m")])
        XCTAssertEqual(ChatModel.downloadSteps("m", packs: []), [.model("m")])
    }

}

@MainActor final class StoredMessageTests: XCTestCase {

    private func msg(posters: [Data]?) -> ConversationStore.Msg {
        ConversationStore.Msg(
            fromUser: true, text: "watch this", reasoning: "",
            rounds: [], images: [], loopStopped: false, clips: nil,
            docs: nil, posters: posters)
    }

    func testAPosterSurvivesTheRoundTrip() throws {
        let frame = Data([0xFF, 0xD8, 0xFF])
        let data = try JSONEncoder().encode(msg(posters: [frame]))
        let back = try JSONDecoder().decode(ConversationStore.Msg.self,
                                            from: data)
        XCTAssertEqual(back.posters, [frame])
    }

    func testARecordWrittenBeforePostersStillDecodes() throws {
        let json = "{\"fromUser\":true,\"text\":\"watch this\","
            + "\"reasoning\":\"\",\"rounds\":[],\"images\":[],"
            + "\"loopStopped\":false}"
        let back = try JSONDecoder().decode(ConversationStore.Msg.self,
                                            from: Data(json.utf8))
        XCTAssertNil(back.posters)
        XCTAssertEqual(back.text, "watch this")
    }

}

@MainActor final class ConversationSearchTests: XCTestCase {

    func testShortQueryIsNotActive() {
        XCTAssertFalse(ConversationSearch.active("a"))
        XCTAssertTrue(ConversationSearch.active("ab"))
    }

    func testRankingRequiresEveryWordToLand() {
        let id = UUID()
        let convo = ConversationStore.Convo(
            id: id, title: "Harvest report", created: Date(),
            updated: Date(), messages: [])
        let index = [id: ["harvest": 5, "report": 5]]
        XCTAssertEqual(
            ConversationSearch.rank([convo], index, "harvest").count, 1)
        XCTAssertEqual(
            ConversationSearch.rank([convo], index, "harvest wombat").count,
            0)
    }

}

@MainActor final class SavedStatsTests: XCTestCase {

    private func event(_ kind: TraceEvent.Kind, tokens: Int, ctx: Int,
                       seconds: Double, summary: String = "") -> TraceEvent {
        let t0 = Date(timeIntervalSince1970: 1000)
        return TraceEvent(kind: kind, t0: t0,
                          t1: t0.addingTimeInterval(seconds), ctx: ctx,
                          tokens: tokens, summary: summary, text: "")
    }

    func testSavedStatsAverageOverTheWholeTranscript() {
        let events = [
            event(.user, tokens: 0, ctx: 10, seconds: 0),
            event(.prefill, tokens: 200, ctx: 210, seconds: 2),
            event(.decode, tokens: 100, ctx: 310, seconds: 5,
                  summary: "eos (think 40, content 60, tg 20.0 t/s)"),
            event(.answer, tokens: 60, ctx: -1, seconds: 0),
            event(.user, tokens: 0, ctx: 310, seconds: 0),
            event(.prefill, tokens: 100, ctx: 410, seconds: 1),
            event(.decode, tokens: 50, ctx: 460, seconds: 5,
                  summary: "loop-breaker"),
        ]
        let stats = ChatModel.savedStats(events)
        XCTAssertEqual(stats, ChatModel.SavedStats(
            ctx: 460, think: 40, content: 110, pp: 100, tg: 15, turns: 2))
        XCTAssertTrue(ChatModel.savedLabel(events).hasSuffix("2 turns"))
    }

    func testATranscriptWithoutATraceHasNoLabel() {
        XCTAssertNil(ChatModel.savedStats([]))
        XCTAssertEqual(ChatModel.savedLabel(
            [event(.user, tokens: 0, ctx: 0, seconds: 0)]), "")
    }

    func testASingleTurnDoesNotSayOneTurn() {
        let label = ChatModel.savedLabel([
            event(.user, tokens: 0, ctx: 10, seconds: 0),
            event(.prefill, tokens: 100, ctx: 110, seconds: 1),
            event(.decode, tokens: 50, ctx: 160, seconds: 5,
                  summary: "eos (think 0, content 50, tg 10.0 t/s)"),
        ])
        XCTAssertFalse(label.contains("turn"))
        XCTAssertTrue(label.hasPrefix("\u{21C4} 160"))
    }

}

private final class TalkBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let lock = NSLock()
    private var pos = 0
    private var marked = 0

    func encode(_ text: String) -> [Int32] {
        text.utf8.map { byte in Int32(byte) }
    }

    func tokenBytes(_ id: Int32) -> [UInt8] {
        id >= 0 && id < 256 ? [UInt8(id)] : []
    }

    func text(_ ids: [Int32]) -> String {
        String(decoding: ids.flatMap { id in tokenBytes(id) }, as: UTF8.self)
    }

    var position: Int { get async { lock.withLock { pos } } }

    func reset() async { lock.withLock { pos = 0 } }

    func useSampler(_ s: Sampler?) async {}

    func extend(_ ids: [Int32]) async throws -> Int32 {
        await Task.yield()
        lock.withLock { pos += ids.count }
        return 111
    }

    func decode(_ token: Int32) async throws -> Int32 {
        await Task.yield()
        lock.withLock { pos += 1 }
        return token == 111 ? 107 : eos
    }

    func mark() async throws { lock.withLock { marked = pos } }

    func rewind() async throws { lock.withLock { pos = marked } }

    struct State: BackendState { let pos: Int; let mark: Int }

    func saveState() async throws -> any BackendState {
        lock.withLock { State(pos: pos, mark: marked) }
    }

    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            lock.withLock {
                pos = s.pos
                marked = s.mark
            }
        }
    }

    func attach(_ dir: URL) async throws {
        throw EngineError.missingModel("talk")
    }
}

private final class ParkBackend: AgentBackend, @unchecked Sendable {
    let eos: Int32 = -1
    private let lock = NSLock()
    private var pos = 0
    private var marked = 0
    private var dir: URL?
    private var hold = false
    private var gate: CheckedContinuation<Void, Never>?

    var held: Bool { lock.withLock { gate != nil } }

    func holdNextResume() { lock.withLock { hold = true } }

    func release() {
        let waiting = lock.withLock {
            let out = gate
            gate = nil
            return out
        }
        waiting?.resume()
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

    var position: Int { get async { lock.withLock { pos } } }

    func reset() async { lock.withLock { pos = 0 } }

    func useSampler(_ s: Sampler?) async {}

    func extend(_ ids: [Int32]) async throws -> Int32 {
        await Task.yield()
        lock.withLock { pos += ids.count }
        return 111
    }

    func decode(_ token: Int32) async throws -> Int32 {
        await Task.yield()
        lock.withLock { pos += 1 }
        return token == 111 ? 107 : eos
    }

    func mark() async throws { lock.withLock { marked = pos } }

    func rewind() async throws { lock.withLock { pos = marked } }

    struct State: BackendState { let pos: Int; let mark: Int }

    func saveState() async throws -> any BackendState {
        lock.withLock { State(pos: pos, mark: marked) }
    }

    func loadState(_ state: any BackendState) async throws {
        if let s = state as? State {
            lock.withLock {
                pos = s.pos
                marked = s.mark
            }
        }
    }

    func attach(_ dir: URL) async throws {
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        lock.withLock { self.dir = dir }
    }

    func detach() async { lock.withLock { dir = nil } }

    func park(to dir: URL, meta: Data) async throws {
        let from = lock.withLock { self.dir }
        if let from {
            try meta.write(to: from.appendingPathComponent("meta.json"))
            if from.standardizedFileURL != dir.standardizedFileURL {
                try? FileManager.default.removeItem(at: dir)
                try FileManager.default.moveItem(at: from, to: dir)
            }
            lock.withLock { self.dir = nil }
        }
    }

    func resume(from dir: URL) async throws -> Data {
        let waits = lock.withLock {
            let out = hold
            hold = false
            return out
        }
        if waits {
            await withCheckedContinuation {
                (k: CheckedContinuation<Void, Never>) in
                lock.withLock { gate = k }
            }
        }
        let meta = try Data(contentsOf: dir.appendingPathComponent(
            "meta.json"))
        try await attach(dir)
        return meta
    }
}

@MainActor final class ChatModelOrderTests: XCTestCase {

    private let template = "{%- if messages[0].role == 'system' -%}"
        + "<|im_start|>system\n{{ messages[0].content }}<|im_end|>\n"
        + "{%- endif -%}"
        + "{%- for m in messages -%}"
        + "{%- if m.role != 'system' -%}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{%- endif -%}{%- endfor -%}"
        + "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
        + "{%- endif -%}"

    private func talking(_ backend: any AgentBackend = TalkBackend())
        -> ChatModel {
        let model = ChatModel()
        model.session.modelName = "ui-order-test"
        model.session.installBackend(backend, template: template,
                                     vocabSize: 256, presets: .greedy)
        model.session.makeSession(model.sessionConfig()) { _ in }
        model.session.primeSession()
        return model
    }

    private func settle(_ model: ChatModel) async {
        let deadline = Date().addingTimeInterval(10)
        while (model.busy || model.pendingSend != nil) && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(5))
        }
        await model.session.awaitPrimed()
        await model.session.awaitMeta()
        XCTAssertFalse(model.busy, "the model is stuck busy")
    }

    private func say(_ model: ChatModel, _ text: String) {
        model.input = text
        model.caret = text.utf16.count
        model.send()
    }

    func testAStopBeforeTheTurnStartsCancelsTheTurn() async {
        let model = talking()
        await settle(model)
        say(model, "hi")
        XCTAssertTrue(model.showsStop)
        model.stop()
        await settle(model)
        XCTAssertTrue(model.messages.isEmpty,
                      "the stopped ask ran as a turn: \(model.messages.count)")
    }

    func testAStopOutsideATurnDoesNotOutliveIt() async {
        let model = talking()
        await settle(model)
        model.newChat()
        model.stop()
        await settle(model)
        say(model, "hi")
        XCTAssertTrue(model.showsStop, "the next turn shows no Stop")
        XCTAssertFalse(model.canSend, "Send is live beside a turn")
        await settle(model)
        XCTAssertEqual(model.messages.count, 2)
    }

    func testAReadOnlyConversationTakesNoTurn() async {
        let model = talking()
        await settle(model)
        model.messages = [Message(fromUser: true, text: "kept"),
                          Message(fromUser: false, text: "answer")]
        model.readOnly = true
        say(model, "hi")
        await settle(model)
        XCTAssertEqual(model.messages.count, 2,
                       "a turn ran in a read-only conversation")
    }

    func testANewChatTakesNothingFromTheTurnItCut() async {
        var faults: [String] = []
        for run in 0 ..< 4 {
            let model = talking()
            await settle(model)
            UserDefaults.standard.removeObject(forKey: "tg.ui-order-test")
            say(model, "hi")
            let deadline = Date().addingTimeInterval(5)
            while Session.storedTG("ui-order-test") == 0
                    && Date() < deadline {
                await Task.yield()
            }
            model.newChat()
            let settled = model.settledTurns
            await settle(model)
            if !model.messages.isEmpty || !model.statsLabel.isEmpty
                || model.settledTurns != settled {
                faults.append("run \(run): \(model.messages.count) "
                    + "message(s), stats \(model.statsLabel.debugDescription)"
                    + ", \(model.settledTurns - settled) turn(s) settled")
            }
        }
        XCTAssertEqual(faults, [])
    }

    func testAnOpenGivesWayToANewChatTappedDuringItsRead() async {
        let model = talking()
        await settle(model)
        let id = UUID()
        let said = String(repeating: "an exchange long enough to keep ",
                          count: 8)
        let store = ConversationStore.shared
        await store.commit(
            id: id, title: "ui-order-test", fallbackTitle: "ui-order-test",
            messages: [Message(fromUser: true, text: said),
                       Message(fromUser: false, text: said)],
            trace: [], extracted: nil).value
        var landed: [Int] = []
        for gap in 0 ..< 16 {
            model.openConversation(id)
            for _ in 0 ..< gap { await Task.yield() }
            model.newChat()
            let deadline = Date().addingTimeInterval(0.3)
            while model.currentConversationId != id && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(10))
            }
            await settle(model)
            if model.currentConversationId == id { landed.append(gap) }
            model.newChat()
            await settle(model)
        }
        await store.trash(id)
        await store.deleteForever(id)
        XCTAssertEqual(landed, [], "the open landed over the new chat")
    }

    private static let said = String(
        repeating: "an exchange long enough to keep ", count: 8)

    private static func parkDir(_ id: UUID) -> URL {
        Session.parked.appendingPathComponent(
            id.uuidString + ".ui-order-test.local", isDirectory: true)
    }

    private func forget(_ id: UUID, _ model: ChatModel) async {
        model.session.dropParked(id)
        try? FileManager.default.removeItem(at: Session.parked
            .appendingPathComponent("live.ui-order-test.local"))
        await ConversationStore.shared.trash(id)
        await ConversationStore.shared.deleteForever(id)
    }

    func testAParkThatDoesNotResumeFallsBackToTheReplay() async throws {
        let model = talking()
        await settle(model)
        let id = UUID()
        await ConversationStore.shared.commit(
            id: id, title: "ui-order-test", fallbackTitle: "ui-order-test",
            messages: [Message(fromUser: true, text: ChatModelOrderTests.said),
                       Message(fromUser: false, text: "kept")],
            trace: [], extracted: nil).value
        try FileManager.default.createDirectory(
            at: ChatModelOrderTests.parkDir(id),
            withIntermediateDirectories: true)
        XCTAssertTrue(model.session.hasParked(id))
        model.openConversation(id)
        let deadline = Date().addingTimeInterval(5)
        while model.currentConversationId != id && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        await settle(model)
        XCTAssertEqual(model.currentConversationId, id)
        XCTAssertFalse(model.readOnly, "a park that failed left the "
            + "conversation read only with no word: \(model.savedLabel)")
        await forget(id, model)
    }

    func testAResumeTheUserLeftWritesNothingOnTheNewChat() async throws {
        let defaults = UserDefaults.standard
        let resumable = defaults.object(forKey: "flag.resumable")
        let pp = defaults.object(forKey: "pp.ui-order-test")
        defer {
            defaults.set(resumable, forKey: "flag.resumable")
            defaults.set(pp, forKey: "pp.ui-order-test")
        }
        Flags.store("resumable", true)
        let backend = ParkBackend()
        let model = talking(backend)
        await settle(model)
        say(model, ChatModelOrderTests.said)
        await settle(model)
        let id = try XCTUnwrap(model.currentConversationId)
        defaults.set(0.001, forKey: "pp.ui-order-test")
        model.newChat()
        await settle(model)
        XCTAssertTrue(model.session.hasParked(id), "the chat did not park")
        backend.holdNextResume()
        model.openConversation(id)
        let deadline = Date().addingTimeInterval(5)
        while !backend.held && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertTrue(backend.held, "the resume never started")
        model.newChat()
        backend.release()
        await settle(model)
        XCTAssertNil(model.currentConversationId)
        XCTAssertNil(model.generatedTitle,
                     "the left chat's title landed on the new chat")
        XCTAssertEqual(model.statsLabel, "")
        await forget(id, model)
    }

}
