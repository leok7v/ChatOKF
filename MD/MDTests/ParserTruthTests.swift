import XCTest
@testable import MD

final class ParserTruthTests: XCTestCase {

    private func blocks(_ md: String) -> [Markdown.Block] {
        Markdown.parse(md).items.map { item in item.block }
    }

    private func text(_ block: Markdown.Block?) -> String {
        var out = ""
        switch block {
            case .paragraph(let a)?: out = String(a.characters)
            case .heading(_, let a)?: out = String(a.characters)
            default: out = ""
        }
        return out
    }

    func testDisplaysOnOneLineAreSeparateDisplays() {
        let line = "$$\\sqrt{2} \\over 3$$ $$\\sum_{i=1}^{n} i^2$$ "
            + "$$ \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} $$ "
            + "$$ \\begin{cases} x = 1 \\\\ y = 2 \\end{cases} $$"
        let got = blocks(line)
        XCTAssertEqual(got, [
            .math("\\sqrt{2} \\over 3"),
            .math("\\sum_{i=1}^{n} i^2"),
            .math("\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"),
            .math("\\begin{cases} x = 1 \\\\ y = 2 \\end{cases}"),
        ])
    }

    func testTextAfterADisplayIsKept() {
        let got = blocks("$$a$$ is the area")
        XCTAssertEqual(got.count, 2)
        XCTAssertEqual(got.first, .math("a"))
        XCTAssertEqual(text(got.last), "is the area")
        XCTAssertEqual(blocks("$$a$$$$"), [.math("a")])
    }

    func testAClosingOnTheLastLineKeepsItsTail() {
        let got = blocks("$$\nx^2\n$$ so x is squared")
        XCTAssertEqual(got.first, .math("x^2"))
        XCTAssertEqual(text(got.last), "so x is squared")
    }

    func testTextAfterAOneLineCommentIsKept() {
        let got = blocks("<!-- note --> <!-- more --> visible text")
        XCTAssertEqual(got.count, 1)
        XCTAssertEqual(text(got.first), "visible text")
    }

    func testALongerFenceClosesOnlyOnALongerOrEqualRun() {
        let md = "````markdown\n```swift\nlet x = 1\n```\n````\nafter"
        let got = blocks(md)
        XCTAssertEqual(got.first, .code(language: "markdown",
                                        text: "```swift\nlet x = 1\n```"))
        XCTAssertEqual(text(got.last), "after")
    }

    func testAFenceClosesOnlyOnABareRun() {
        let got = blocks("```\n``` not a close\n```")
        XCTAssertEqual(got, [.code(language: nil, text: "``` not a close")])
    }

    func testIndentedCodeIsCodeWhateverItStartsWith() {
        XCTAssertEqual(blocks("    # not a heading\n    ---"),
                       [.code(language: nil, text: "# not a heading\n---")])
    }

    func testAnIndentedLineContinuesAParagraph() {
        let got = blocks("words\n    # still words")
        XCTAssertEqual(got.count, 1)
        XCTAssertEqual(text(got.first), "words # still words")
    }

    func testNotesAndFootnotesAreNotLinkDefinitions() {
        let got = blocks("[Note]: do not run this.\n\n[^1]: a footnote")
        XCTAssertEqual(got.count, 2)
        XCTAssertEqual(text(got.first), "[Note]: do not run this.")
        XCTAssertEqual(text(got.last), "[^1]: a footnote")
    }

    func testALinkDefinitionMayCarryATitle() {
        let got = blocks("[home]: https://example.com \"Home\"\n\nSee [home].")
        XCTAssertEqual(got.count, 1)
        var linked = false
        if case .paragraph(let a)? = got.first {
            linked = a.runs.contains { run in run.link != nil }
        }
        XCTAssertTrue(linked)
    }

    func testANumberMidParagraphIsNotAList() {
        let got = blocks("In the year\n1984. Things changed.")
        XCTAssertEqual(got.count, 1)
        XCTAssertEqual(text(got.first), "In the year 1984. Things changed.")
        if case .list? = blocks("Steps:\n1. first\n2. second").last {
        } else {
            XCTFail("a list numbered 1 must interrupt a paragraph")
        }
    }

    func testSetextHeadingsAndClosingHashes() {
        XCTAssertEqual(text(blocks("Title\n===").first), "Title")
        if case .heading(let level, _)? = blocks("Sub\n---").first {
            XCTAssertEqual(level, 2)
        } else {
            XCTFail("a dashed underline makes a second-level heading")
        }
        XCTAssertEqual(text(blocks("## Head ##").first), "Head")
        XCTAssertEqual(text(blocks("# C#").first), "C#")
        XCTAssertEqual(blocks("> quote\n===").count, 2)
    }

    func testALoneDashUnderAParagraphIsNotAHeading() {
        if case .heading? = blocks("Para\n-").first {
            XCTFail("mid-stream a lone dash is the start of an item")
        }
    }

    func testTheLastLineKeepsNoHardBreak() {
        XCTAssertEqual(text(blocks("line one  \nline two  ").first),
                       "line one" + Markdown.lineBreak + "line two")
        XCTAssertEqual(text(blocks("a\\\nb").first),
                       "a" + Markdown.lineBreak + "b")
    }

    func testTableCellsResolveReferenceLinks() {
        let md = "[d]: https://example.com/docs\n\n| a |\n|---|\n| [d] |"
        if case .table(_, let rows, _)? = blocks(md).first {
            XCTAssertEqual(rows, [["[d](https://example.com/docs)"]])
        } else {
            XCTFail("expected a table")
        }
    }

    func testAnEscapedAngleIsText() {
        XCTAssertEqual(text(blocks("a \\<sub>b</sub> and &lt;u>").first),
                       "a <sub>b</sub> and <u>")
    }

    func testAnEscapedAngleInsideMathsIsAnAngle() {
        for md in ["a $x &lt; y$ b", "a $x \\< y$ b"] {
            XCTAssertEqual(text(blocks(md).first), "a x < y b", md)
        }
    }

    func testATagSpelledWithEntitiesStillApplies() {
        var underlined = ""
        if case .paragraph(let a)? = blocks("&#60;u>x&#60;/u> y").first {
            for run in a.runs where run.underlineStyle != nil {
                underlined += String(a[run.range].characters)
            }
        }
        XCTAssertEqual(underlined, "x")
    }

    func testMathIsNotSplitInsideALinkTarget() {
        let got = blocks("[a](https://x.org/$a$b) costs $5")
        var link: URL? = nil
        if case .paragraph(let a)? = got.first {
            link = a.runs.compactMap { run in run.link }.first
        }
        XCTAssertEqual(link?.absoluteString, "https://x.org/$a$b")
    }

    func testDeepNestingIsCapped() {
        for marker in ["> ", "- "] {
            let deep = String(repeating: marker, count: 100_000) + "deep"
            XCTAssertEqual(blocks(deep).count, 1)
        }
    }

    func testCarriageReturnsEndLines() {
        XCTAssertEqual(blocks("# A\rtext").count, 2)
    }
}
