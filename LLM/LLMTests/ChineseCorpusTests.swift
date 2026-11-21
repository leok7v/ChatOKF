import XCTest
@testable import LLM

final class ChineseCorpusTests: XCTestCase {

    struct Asked: Decodable {
        let question: String
        let id: String
    }

    static let fixture = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-data-zh", isDirectory: true)
    static let questions = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/fixtures/zh-questions.json")

    static func asked() throws -> [Asked] {
        var out = try JSONDecoder().decode(
            [Asked].self, from: Data(contentsOf: ChineseCorpusTests.questions))
        let calibrated = CalibrationTests.translated
            .first { set in set.language == "Chinese" }?.questions ?? []
        for (i, question) in calibrated.enumerated() {
            out.append(Asked(question: question,
                             id: CalibrationTests.sampleAnswers[i]))
        }
        return out
    }

    func testChineseQuestionsFindChineseNotesWithLiteralTerms() throws {
        let weights = try EncoderTests.cloned(EncoderTests.a25mRepo,
                                              EncoderTests.a25mFile)
        guard let embedder = ModernBertEmbedder.load(ggufPath: weights) else {
            throw XCTSkip("not a ModernBERT GGUF")
        }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "zh-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: ChineseCorpusTests.fixture, to: root)
        let store = Store(root: root, embedder: embedder)
        store.load()
        let asked = try ChineseCorpusTests.asked()
        var found = 0
        var passed = 0
        var termed = 0
        var misses: [String] = []
        for ask in asked {
            let top = store.search([ask.question], filter: Filter(), limit: 3)
                .hits.first
            if top?.concept.id == ask.id {
                found += 1
                if let top, store.relevant(top) { passed += 1 }
            } else {
                misses.append(ask.id + "->" + (top?.concept.id ?? "-"))
            }
            if top?.terms.isEmpty == false { termed += 1 }
        }
        print("[zh] \(asked.count) questions over \(store.concepts.count) "
              + "Chinese notes, re-embedded \(store.embeddedCount): found "
              + "\(found), pass \(passed), top hit with literal terms "
              + "\(termed); misses: " + misses.joined(separator: " "))
        XCTAssertEqual(store.concepts.count, 50)
        XCTAssertGreaterThanOrEqual(found * 2, asked.count)
        try? fm.removeItem(at: root)
    }
}
