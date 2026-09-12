import XCTest
@testable import MD

// Dollar amounts in prose are not inline math; the sentence is verbatim
// from a Qwen3.5-4B transcript.
final class CurrencyTests: XCTestCase {

    private func texts(_ s: String) -> [String] {
        TeX.split(s).compactMap { seg in
            if case let .text(t) = seg { return t }
            return nil
        }
    }

    private func maths(_ s: String) -> [String] {
        TeX.split(s).compactMap { seg in
            if case let .math(m, _) = seg { return m }
            return nil
        }
    }

    func testAPriceIsNotAFormula() {
        let s = "The bat costs $1.10 and the ball costs $0.05."
        XCTAssertEqual(maths(s), [])
        XCTAssertEqual(texts(s).joined(), s)
    }

    func testTwoPricesDoNotPairIntoOneSpan() {
        let s = "The total is $1.10, and the bat costs $1 more than the ball."
        XCTAssertEqual(maths(s), [])
        XCTAssertEqual(texts(s).joined(), s)
    }

    func testRealInlineMathStillParses() {
        XCTAssertEqual(maths("The sum is $b + (1 + b) = 1.10$ exactly."),
                       ["b + (1 + b) = 1.10"])
        XCTAssertEqual(maths("so $2b + 1 = 1.10$."), ["2b + 1 = 1.10"])
        XCTAssertEqual(maths("let $x$ be free"), ["x"])
    }

    // The whole transcript line: currency and algebra interleaved.
    func testMixedProseKeepsMoneyAndTypesetsAlgebra() {
        let s = "The total cost is $1.10, and the bat costs $1 more than "
            + "the ball. Then the bat = $1 + b$. The sum is "
            + "$b + (1 + b) = 1.10$. So $2b + 1 = 1.10$. Subtract 1: "
            + "$2b = 0.10$. Divide by 2: $b = 0.05$. The ball costs $0.05."
        XCTAssertEqual(maths(s), ["1 + b", "b + (1 + b) = 1.10",
                                  "2b + 1 = 1.10", "2b = 0.10", "b = 0.05"])
        let joined = texts(s).joined()
        XCTAssertTrue(joined.contains("$1.10, and the bat costs $1 more"),
                      "money lost its sign: \(joined)")
        XCTAssertTrue(joined.hasSuffix("The ball costs $0.05."),
                      "trailing amount lost its sign: \(joined)")
    }

    // A span opened by a space is prose ("cost $ 5"), and an escaped \$
    // was already literal before this rule.
    func testSpacedAndEscapedDollarsStayText() {
        XCTAssertEqual(maths("cost $ 5 $ each"), [])
        XCTAssertEqual(texts("a \\$5 bill").joined(), "a $5 bill")
    }

    func testDisplayMathIsUntouched() {
        XCTAssertEqual(maths("see $$x^2$$ here"), ["x^2"])
    }

    func testAnUnclosedDollarStaysText() {
        let s = "it costs $5"
        XCTAssertEqual(maths(s), [])
        XCTAssertEqual(texts(s).joined(), s)
    }
}
