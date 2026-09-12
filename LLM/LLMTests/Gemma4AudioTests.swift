import Foundation
import Testing
@testable import LLM

// The gemma-4 speech path end to end, on REAL audio: decode the file, log-mel
private let audioFixtures = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("fixtures/audio")

struct Gemma4AudioTests {
    private static func fixture(_ name: String) -> URL {
        audioFixtures.appendingPathComponent(name)
    }

    // One clip through the whole path, decoded GREEDILY -- no sampler, so the
    // engine takes the argmax and the result is reproducible.
    private func transcribe(_ path: String, _ file: String,
                            _ ask: String) async throws -> String {
        let chat = try GemmaChat(ggufPath: path)
        let mel = chat.melConfig
        let pcm = try await AudioFile.samples(
            url: Gemma4AudioTests.fixture(file),
            sampleRate: Double(mel.sampleRate))
        let engine = try Gemma4MetalEngine(chat.model)
        let spans = try Gemma4Media(chat, ctx: engine.ctx).audio(pcm)
        let parts = [ContentPart](repeating: .audio, count: spans.count)
        let prompt = try renderPrompt(
            template: chat.chatTemplate,
            messages: [AgentMessage(role: "user", content: ask,
                                    contentParts: parts + [.text(ask)])],
            tools: [], addGenerationPrompt: true, enableThinking: false,
            bosToken: chat.bosToken)
        let ids = Continuation.expandSpans(chat.encode(prompt), spans)
        engine.reset()
        let feed = SoftFeed(spans)
        var next = engine.extend(ids, softAt: { id in feed.row(id) })
        var out: [Int32] = []
        while out.count < 24 && !chat.eosIds.contains(next) {
            out.append(next)
            next = engine.decode(next)
        }
        return chat.decode(out)
    }

    private static let ask = "Transcribe this audio exactly."

    // A CONTENT word rather than the whole string: the sentence is what the
    @Test(needsGemmaWeights)
    func englishClipTranscribes() async throws {
        let path = try #require(gemmaGgufPath)
        let said = try await transcribe(
            path, "en-lets-leave-tomorrow.wav", Gemma4AudioTests.ask)
        let why = "expected \"Let's leave tomorrow.\", heard: \(said)"
        #expect(said.lowercased().contains("tomorrow"), "\(why)")
        #expect(said.lowercased().contains("leave"), "\(why)")
    }

    // The same sentence in French.
    @Test(needsGemmaWeights)
    func frenchClipTranscribes() async throws {
        let path = try #require(gemmaGgufPath)
        let said = try await transcribe(
            path, "fr-partons-demain.wav", Gemma4AudioTests.ask)
        let why = "expected \"Partons demain.\", heard: \(said)"
        #expect(said.lowercased().contains("demain"), "\(why)")
        #expect(said.lowercased().contains("partons"), "\(why)")
    }

    // A clip past the tower's ceiling is CUT, not refused: it comes back as
    @Test(needsGemmaWeights) func overlongClipIsChunked() throws {
        let path = try #require(gemmaGgufPath)
        let chat = try GemmaChat(ggufPath: path)
        let wire = try chat.audioWire()
        let rate = chat.melConfig.sampleRate
        let tooLong = Int(Double(rate) * (wire.maxSeconds + 5))
        let media = Gemma4Media(chat)
        let spans = try media.audio([Float](repeating: 0, count: tooLong))
        #expect(spans.count > 1,
                "a clip past the ceiling must be cut: \(spans.count) span(s)")
        #expect(spans.allSatisfy { s in s.rows > 0 })
    }
}