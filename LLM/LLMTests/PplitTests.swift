import XCTest
@testable import LLM

final class PplitTests: XCTestCase {

    func testCorpusParserMirrorsPplit() {
        let single = "# comment\n<|@PROMPT@|>\nq1\nmore\n<|@RESPONSE@|>\na1\n"
            + "<|@PROMPT@|>\nq2\n<|@RESPONSE@|>\na2\nline2\n"
        let convs = parsePplitCorpus(single)
        XCTAssertEqual(convs.count, 2)
        XCTAssertEqual(convs[0].count, 1)
        XCTAssertEqual(convs[0][0].prompt, "q1\nmore")
        XCTAssertEqual(convs[1][0].response, "a2\nline2\n")
        let multi = "<|@CONV@|>\n<|@PROMPT@|>\nq1\n<|@RESPONSE@|>\na1\n"
            + "<|@PROMPT@|>\nq2\n<|@RESPONSE@|>\na2\n<|@CONV@|>\n"
            + "<|@PROMPT@|>\nq3\n<|@RESPONSE@|>\na3\n"
        let talks = parsePplitCorpus(multi)
        XCTAssertEqual(talks.count, 2)
        XCTAssertEqual(talks[0].count, 2)
        XCTAssertEqual(pplitMessages(talks[0], upto: 1).map { m in m.role },
                       ["user", "assistant", "user"])
    }

    func testTheVendoredCorpusParses() throws {
        let text = try String(contentsOfFile: pplitCorpusBeside(),
                              encoding: .utf8)
        let convs = parsePplitCorpus(text)
        XCTAssertGreaterThan(convs.count, 50)
        for conv in convs {
            for turn in conv {
                XCTAssertFalse(turn.prompt.isEmpty)
                XCTAssertFalse(turn.response.isEmpty)
            }
        }
    }

    func testTheSmallestModelOnDiskScoresChatFramed() throws {
        let text = try String(contentsOfFile: pplitCorpusBeside(),
                              encoding: .utf8)
        let convs = Array(parsePplitCorpus(text).prefix(2))
        let found = ModelCatalog.ggufFiles.keys.sorted()
            .compactMap { name in TestWeights.find(name) }
        let smallest = found.min { a, b in size(a) < size(b) }
        guard let path = smallest else {
            throw XCTSkip("no catalog model on disk")
        }
        let scorer = try pplitScorer(path)
        let (totals, _) = try pplitScore(scorer, convs, ctx: 8192, cap: 320,
                                         dump: nil, against: nil) { _, _, _ in }
        print("[pplit] " + totals.line(
            (path as NSString).lastPathComponent, convs: convs.count))
        XCTAssertGreaterThan(totals.n, 0, "scored nothing")
        XCTAssertGreaterThan(totals.ppl, 1, "a perplexity below 1 is not one")
        XCTAssertLessThan(totals.ppl, 100,
                          "ppl \(totals.ppl) is not a chat-framed number")
        XCTAssertGreaterThan(totals.top1, 0)
        XCTAssertLessThanOrEqual(totals.top1, totals.top5)
        XCTAssertLessThanOrEqual(totals.top5, totals.top10)
    }

    private func size(_ path: String) -> Int {
        (try? FileManager.default.attributesOfItem(atPath: path))
            .flatMap { a in a[.size] as? Int } ?? Int.max
    }

}
