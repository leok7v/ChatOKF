import Foundation
import Testing
@testable import LLM

let chatGgufPath = TestWeights.find("Qwen3.5-4B")

let needsChatWeights = ConditionTrait.enabled(
    if: chatGgufPath != nil,
    Comment(rawValue: TestWeights.missing("Qwen3.5-4B")))

struct BatAndBallTests {

    static let question = "A bat and a ball cost 1.10 dollars in total. "
        + "The bat costs 1.00 dollar more than the ball. "
        + "How much does the ball cost?"

    private func answer(tools: Bool) async throws -> String {
        let path = try #require(chatGgufPath)
        let chat = try QwenMetalChat(ggufPath: path)
        let session = ChatSession(
            backend: chat.backend(), template: chat.chatTemplate,
            system: "You are a helpful assistant.",
            vocabSize: chat.tokenizer.vocabCount,
            presets: chat.samplingPresets, enableThinking: true,
            maxTokens: 2048,
            runner: tools ? SafeToolRunner(slugsPath: nil, wikipedia: false,
                                           network: false) : nil)
        var out = ""
        for await piece in session.reply(BatAndBallTests.question) {
            out += piece
        }
        return out
    }

    private func solved(_ text: String) -> Bool {
        text.contains("0.05") || text.contains("5 cents")
            || text.contains("five cents")
    }

    @Test(needsChatWeights)
    func ballCostsFiveCentsWithoutTools() async throws {
        let out = try await answer(tools: false)
        #expect(solved(out), "answer: \(out)")
    }

    // Advertising tools must not COST the model the answer it can reach on its
    @Test(needsChatWeights)
    func ballCostsFiveCentsWithTools() async throws {
        let out = try await answer(tools: true)
        #expect(solved(out), "answer: \(out)")
    }

}
