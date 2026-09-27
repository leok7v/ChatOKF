import XCTest
@testable import MD

final class HtmlTests: XCTestCase {

    private func paragraph(_ md: String) -> AttributedString? {
        var out: AttributedString? = nil
        if case .paragraph(let a) = Markdown.parse(md).items.first?.block {
            out = a
        }
        return out
    }

    private func text(_ a: AttributedString) -> String {
        String(a.characters)
    }

    func testABreakIsALineSeparatorInsideItsParagraph() throws {
        let a = try XCTUnwrap(paragraph(
            "First<br>second<br/>third<br />fourth."))
        XCTAssertEqual(text(a),
                       "First\u{2028}second\u{2028}third\u{2028}fourth.")
        let hard = try XCTUnwrap(paragraph("one  \ntwo"))
        XCTAssertEqual(text(hard), "one\u{2028}two")
        XCTAssertEqual(Markdown.parse("First<br>second").items.count, 1)
    }

    func testSmallPrintIsARunAttribute() throws {
        let a = try XCTUnwrap(paragraph(
            "Text with <small>small print</small> after."))
        XCTAssertEqual(text(a), "Text with small print after.")
        let small = a.runs.filter { run in run[SmallAttribute.self] == true }
        XCTAssertEqual(small.count, 1)
        XCTAssertEqual(String(a[small[0].range].characters), "small print")
    }

    func testAnEntityDecodesAndAnUnknownTagStaysLiteral() throws {
        let a = try XCTUnwrap(paragraph(
            "An &nbsp; entity, &amp; and an <unknown attr=\"x\">tag</unknown>."))
        XCTAssertEqual(text(a),
            "An \u{00A0} entity, & and an <unknown attr=\"x\">tag</unknown>.")
    }

    func testSynonymsBecomeTheirMarkdown() throws {
        let a = try XCTUnwrap(paragraph(
            "Bold <b>b</b>, <em>em</em>, <s>s</s>, <kbd>Cmd</kbd> and "
            + "<a href=\"https://example.com/a\">to a</a>."))
        XCTAssertEqual(text(a), "Bold b, em, s, Cmd and to a.")
        var seen: Set<String> = []
        for run in a.runs {
            let intent = run.inlinePresentationIntent ?? []
            let piece = String(a[run.range].characters)
            if intent.contains(.stronglyEmphasized), piece == "b" {
                seen.insert("b")
            }
            if intent.contains(.emphasized), piece == "em" { seen.insert("em") }
            if intent.contains(.strikethrough), piece == "s" { seen.insert("s") }
            if intent.contains(.code), piece == "Cmd" { seen.insert("kbd") }
            if run.link?.absoluteString == "https://example.com/a",
               piece == "to a" {
                seen.insert("a")
            }
        }
        XCTAssertEqual(seen, ["b", "em", "s", "kbd", "a"])
    }

    func testATagInsideCodeStaysAsTyped() throws {
        let a = try XCTUnwrap(paragraph("`<br>` and `<sup>x</sup>` stay."))
        XCTAssertEqual(text(a), "<br> and <sup>x</sup> stay.")
        XCTAssertFalse(a.runs.contains { run in
            run[ScriptAttribute.self] != nil
        })
    }

    func testACommentIsDropped() throws {
        let inline = try XCTUnwrap(paragraph("Before <!-- gone --> after."))
        XCTAssertFalse(text(inline).contains("<!--"))
        XCTAssertTrue(text(inline).hasPrefix("Before"))
        let blocks = Markdown.parse("<!-- a block comment\nthat spans lines "
                                    + "-->\n\nStill here.").items
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks.first.flatMap { item in
            if case .paragraph(let a) = item.block { return text(a) }
            return nil
        }, "Still here.")
    }

    func testAnImageTagIsAnImageBlock() throws {
        let doc = Markdown.parse("<img src=\"https://example.com/logo.png\" "
                                 + "alt=\"Logo\" width=\"128\" height=\"64\">")
        if case .image(let alt, let url, let w, let h) = doc.items.first?.block {
            XCTAssertEqual(alt, "Logo")
            XCTAssertEqual(url.absoluteString, "https://example.com/logo.png")
            XCTAssertEqual(w, 128)
            XCTAssertEqual(h, 64)
        } else {
            XCTFail("an <img> alone on its line is an image block")
        }
    }

    func testAnEscapedPipeIsACellCharacterAndCopiesBackEscaped() throws {
        let doc = Markdown.parse("| a | b |\n|---|:-:|\n| x \\| y | z<br>w |")
        guard case .table(let headers, let rows, let aligns)
            = doc.items.first?.block else {
            return XCTFail("expected a table")
        }
        XCTAssertEqual(headers, ["a", "b"])
        XCTAssertEqual(rows, [["x | y", "z<br>w"]])
        let cell = try XCTUnwrap(paragraph("z<br>w"))
        XCTAssertEqual(text(cell), "z\u{2028}w")
        let copied = TableMetrics.serializeMonospaced(
            headers: headers, rows: rows, alignments: aligns)
        XCTAssertTrue(copied.contains("x \\| y"))
        XCTAssertTrue(copied.contains("| ------ | :----: |"), copied)
        let again = Markdown.parse(copied)
        XCTAssertEqual(again.items.first?.block, doc.items.first?.block)
    }

    func testExportsSpellTheBreakAndTheSmallPrint() {
        let doc = Markdown.parse("one<br>two with <small>small</small> and "
                                 + "<u>under</u>.")
        let html = Markdown.html(doc, title: "t")
        XCTAssertTrue(html.contains("one<br>two"))
        XCTAssertTrue(html.contains("<small>small</small>"))
        XCTAssertTrue(html.contains("<u>under</u>"))
        XCTAssertEqual(Markdown.plainText(doc),
                       "one  \ntwo with small and under.\n")
    }

}
