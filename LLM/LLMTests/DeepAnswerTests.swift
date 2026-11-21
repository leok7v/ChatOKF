import XCTest
@testable import LLM

@MainActor final class DeepAnswerTests: XCTestCase {

    struct Entry: Decodable {
        let question: String
        let id: String
        let phrase: String
        let paragraph: Int
    }

    static let fixture = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/fixtures/deep-answers.json")

    private static func flat(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ")
    }

    func testTheWindowCarriesTheDeepAnswerTheHeadMissed() async throws {
        let weights = try EncoderTests.cloned(EncoderTests.a25mRepo,
                                              EncoderTests.a25mFile)
        let entries = try JSONDecoder().decode(
            [Entry].self, from: Data(contentsOf: DeepAnswerTests.fixture))
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "deep-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: StoreReferenceTests.fixture, to: root)
        let memories = Memories(root: root)
        memories.open(encoder: weights)
        await memories.awaitOpen()
        XCTAssertTrue(memories.isOpen)
        var found = 0
        var first = 0
        var carried = 0
        var headCarried = 0
        var misses: [String] = []
        for entry in entries {
            let recall = await memories.recall(entry.question, also: [],
                                               pp: 100, excluding: [])
            let ids = recall?.ids ?? []
            let phrase = DeepAnswerTests.flat(entry.phrase)
            if ids.contains(entry.id) { found += 1 }
            if ids.first == entry.id { first += 1 }
            if DeepAnswerTests.flat(recall?.block ?? "").contains(phrase) {
                carried += 1
            } else if ids.contains(entry.id) {
                misses.append(entry.id + ":" + String(entry.paragraph))
            }
            if let concept = memories.concepts[entry.id],
               DeepAnswerTests.flat(Memories.line(concept, window: nil))
                   .contains(phrase) {
                headCarried += 1
            }
        }
        print("[deep] \(entries.count) questions: note recalled \(found), "
              + "first \(first), answer in the block \(carried), answer in "
              + "the head \(headCarried); recalled without the answer: "
              + misses.joined(separator: " "))
        XCTAssertEqual(headCarried, 0)
        XCTAssertGreaterThan(carried, headCarried)
        try? fm.removeItem(at: root)
    }
}
