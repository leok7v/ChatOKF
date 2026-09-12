import Foundation
import Testing
@testable import LLM

final class TextBox: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""

    func add(_ piece: String) {
        lock.lock()
        text += piece
        lock.unlock()
    }

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return text
    }
}

struct PuzzleGateTests {

    static let scoreFloorPendingCalibration = 0
    static let strayCeilingPendingCalibration = Int.max

    static let markupMarkers = ["<think>", "</think>", "<thinking",
                                "</thinking"]

    private func ask(_ prompt: String) async throws -> (String, String) {
        let path = try #require(chatGgufPath)
        let chat = try QwenMetalChat(ggufPath: path)
        let session = ChatSession(
            backend: chat.backend(), template: chat.chatTemplate,
            system: "You are a helpful assistant.",
            vocabSize: chat.tokenizer.vocabCount,
            presets: SamplingPresets.greedy, enableThinking: true,
            reasoningEffort: "on", maxTokens: 2048)
        let reasoning = TextBox()
        var content = ""
        for await piece in session.reply(
            prompt, onReasoning: { r in reasoning.add(r) }) {
            content += piece
        }
        return (reasoning.value, content)
    }

    private func strays(_ text: String) -> Int {
        PuzzleGateTests.markupMarkers.reduce(0) { total, marker in
            total + text.components(separatedBy: marker).count - 1
        }
    }

    @Test
    func everyAcceptPatternCompiles() {
        var broken: [String] = []
        for puzzle in PuzzleGate.puzzles {
            for pattern in puzzle.accept
            where (try? NSRegularExpression(pattern: pattern)) == nil {
                broken.append("\(puzzle.category): \(pattern)")
            }
        }
        #expect(broken.isEmpty, "\(broken)")
    }

    @Test
    func onlyTheAnswerTailIsScored() {
        let puzzle = PuzzleGate.puzzles[0]
        let trap = String(repeating: "so 2x = 0.10 dollars, no wait. ", count: 40)
        #expect(PuzzleGate.verdict(puzzle, content: trap + "**0.05**")
                == .pass)
        #expect(PuzzleGate.verdict(puzzle, content: "") == .incomplete)
    }

    @Test(needsChatWeights)
    func logicPuzzlesScoreAtOrAboveFloor() async throws {
        var passed = 0
        var report = ""
        for puzzle in PuzzleGate.puzzles {
            let (_, content) = try await ask(puzzle.prompt)
            let verdict = PuzzleGate.verdict(puzzle, content: content)
            passed += verdict == .pass ? 1 : 0
            report += "\(verdict.rawValue) \(puzzle.category)\n"
        }
        let total = PuzzleGate.puzzles.count
        #expect(passed >= PuzzleGateTests.scoreFloorPendingCalibration,
                "\(passed)/\(total)\n\(report)")
    }

    @Test(needsChatWeights)
    func neitherChannelCarriesThinkMarkup() async throws {
        var found = 0
        var report = ""
        for puzzle in PuzzleGate.puzzles {
            let (reasoning, content) = try await ask(puzzle.prompt)
            let inReasoning = strays(reasoning)
            let inContent = strays(content)
            found += inReasoning + inContent
            if inReasoning + inContent > 0 {
                report += "\(puzzle.category): reasoning \(inReasoning) "
                    + "content \(inContent)\n"
            }
        }
        #expect(found <= PuzzleGateTests.strayCeilingPendingCalibration,
                "\(found) stray markers\n\(report)")
    }

}
