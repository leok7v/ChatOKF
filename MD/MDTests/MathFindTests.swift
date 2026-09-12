import XCTest
@testable import MD

final class MathFindTests: XCTestCase {

    private func paragraphText(_ doc: Markdown.Document) -> String {
        var out = ""
        for item in doc.items {
            if case .paragraph(let a) = item.block {
                out += String(a.characters)
            }
        }
        return out
    }

    func testMathOnConvertsToUnicode() {
        let doc = Markdown.parse("value $x^2$ here", math: true)
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("\u{00B2}"), "expected superscript two")
        XCTAssertFalse(text.contains("$"), "delimiters should be consumed")
    }

    func testMathOffLeavesLiteral() {
        let doc = Markdown.parse("value $x^2$ here", math: false)
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("$x^2$"), "math must stay literal")
        XCTAssertFalse(text.contains("\u{00B2}"))
    }

    func testStreamHonorsMathFlag() {
        let src = "sum $x^2$ done"
        let stream = MarkdownStream(math: false)
        for ch in src { stream.append(String(ch)) }
        let got = stream.finish().items.map { i in i.block }
        let expected = Markdown.parse(src, math: false)
            .items.map { i in i.block }
        XCTAssertEqual(got, expected)
        XCTAssertTrue(paragraphText(stream.finish()).contains("$x^2$"))
    }

    func testBackslashParenInline() {
        let doc = Markdown.parse(#"value \(e^{ix}\) here"#, math: true)
        let text = paragraphText(doc)
        XCTAssertFalse(text.contains("\\("), "delimiters consumed")
        XCTAssertTrue(text.contains("e\u{2071}\u{02E3}"),
                      "superscript ix expected in: \(text)")
    }

    func testBackslashBracketDisplay() {
        let src = #"\[ i = e^{i\pi/2} \quad \text{(since } x\text{)} \]"#
        let doc = Markdown.parse(src, math: true)
        let text = paragraphText(doc)
        XCTAssertFalse(text.contains("\\["))
        XCTAssertFalse(text.contains("\\pi"))
        XCTAssertTrue(text.contains("e^(i\u{03C0}/2)"),
                      "kept caret for unmappable script in: \(text)")
    }

    func testAlignedEnvironment() {
        let src = "\\[\n\\begin{aligned}\ni^i &= (e^{i\\pi/2})^i \\\\\n"
            + "&= e^{-\\pi/2}\n\\end{aligned}\n\\]"
        let doc = Markdown.parse(src, math: true)
        let text = paragraphText(doc)
        XCTAssertFalse(text.contains("begin"), "got: \(text)")
        XCTAssertFalse(text.contains("&"), "got: \(text)")
        XCTAssertFalse(text.contains("\\"), "got: \(text)")
        XCTAssertTrue(text.contains("e^(-\u{03C0}/2)"), "got: \(text)")
    }

    func testBoldWrappedMathKeepsEmphasis() {
        let doc = Markdown.parse(#"**\(i^i \approx 0.208\)** rest"#,
                                 math: true)
        var strong = false
        for item in doc.items {
            if case .paragraph(let a) = item.block {
                for run in a.runs {
                    let intent = run.inlinePresentationIntent ?? []
                    if intent.contains(.stronglyEmphasized) { strong = true }
                }
            }
        }
        XCTAssertTrue(strong, "bold must survive around a math span")
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("\u{2248} 0.208"), "got: \(text)")
        XCTAssertFalse(text.contains("**"))
    }

    func testFunctionNamesAndDegrees() {
        let doc = Markdown.parse(
            #"\(e^{i} = \cos(1) + i\sin(1)\) at \(57.3^{\circ}\)"#,
            math: true)
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("cos(1)"), "got: \(text)")
        XCTAssertFalse(text.contains("\\cos"))
        XCTAssertTrue(text.contains("57.3\u{00B0}"), "got: \(text)")
    }

    func testMathBoldMarker() {
        let doc = Markdown.parse(
            #"\(\cos(1) \approx \mathbf{0.5403}\)"#, math: true)
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("0.5403"), "got: \(text)")
        XCTAssertFalse(text.contains("mathbf"))
    }

    func testUnclosedBackslashParenStaysLiteral() {
        let doc = Markdown.parse(#"an open \(e^{ix} span"#, math: true)
        let text = paragraphText(doc)
        XCTAssertTrue(text.contains("e^{ix}"), "got: \(text)")
    }

    func testFindRangesCount() {
        let matches = markdownFindRanges(in: "aXaXa", query: "X",
                                         caseSensitive: false)
        XCTAssertEqual(matches.count, 2)
        XCTAssertEqual(matches[0], NSRange(location: 1, length: 1))
        XCTAssertEqual(matches[1], NSRange(location: 3, length: 1))
    }

    func testFindCaseSensitivity() {
        let ci = markdownFindRanges(in: "Hello hello", query: "hello",
                                    caseSensitive: false)
        let cs = markdownFindRanges(in: "Hello hello", query: "hello",
                                    caseSensitive: true)
        XCTAssertEqual(ci.count, 2)
        XCTAssertEqual(cs.count, 1)
    }

    func testFindNonOverlappingAndEmpty() {
        XCTAssertEqual(
            markdownFindRanges(in: "aaa", query: "aa",
                               caseSensitive: false).count, 1)
        XCTAssertEqual(
            markdownFindRanges(in: "abc", query: "",
                               caseSensitive: false).count, 0)
    }
}
