import XCTest
@testable import LLM

final class JudgeTests: XCTestCase {

    func testSharesAreSoftmaxMassOverEachOptionsTokens() {
        let logits: [Float] = [0, 1, 2, 3]
        let shares = Judge.shares(logits, [[3], [1, 2], [9]])
        let total = (0..<4).reduce(0.0) { sum, i in sum + exp(Double(i)) }
        XCTAssertEqual(shares[0], exp(3.0) / total, accuracy: 1e-9)
        XCTAssertEqual(shares[1], (exp(1.0) + exp(2.0)) / total,
                       accuracy: 1e-9)
        XCTAssertEqual(shares[2], 0)
    }

    func testCommonPrefixLength() {
        XCTAssertEqual(Judge.common([1, 2, 3], [1, 2, 4]), 2)
        XCTAssertEqual(Judge.common([], [1]), 0)
        XCTAssertEqual(Judge.common([5, 6], [5, 6]), 2)
    }

    func testSpellingsCoverCaseAndALeadingSpace() {
        let forms = Judge.spellings("yes")
        XCTAssertTrue(forms.contains("Yes"))
        XCTAssertTrue(forms.contains(" YES"))
        XCTAssertEqual(forms.count, 8)
    }
}
