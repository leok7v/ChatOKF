import XCTest
@testable import MD
import PDFKit

final class ExportAndHighlightTests: XCTestCase {

    private func html(_ md: String, images: [URL: Data] = [:]) -> String {
        Markdown.html(Markdown.parse(md), title: "t", images: images)
    }

    func testHtmlLinksKeepOnlySafeSchemes() {
        let out = html("[a](https://a.b) [m](mailto:a@b) [r](../x.md) "
            + "[f](#top) [bad](javascript:alert(1))")
        for href in ["https://a.b", "mailto:a@b", "../x.md", "#top"] {
            XCTAssertTrue(out.contains("href=\"\(href)\""), href)
        }
        XCTAssertFalse(out.contains("javascript"), out)
    }

    func testHtmlOrderedListKeepsItsStart() {
        XCTAssertTrue(html("3. c\n4. d").contains("<ol start=\"3\""))
        XCTAssertFalse(html("1. a").contains("start="))
    }

    func testTableCellsKeepReferenceLinksAndPictures() throws {
        let md = "[ref]: https://example.com\n\n| a | b |\n|---|---|\n"
            + "| [ref] | ![p](https://example.com/p.png) |"
        let png = Data([0x89, 0x50, 0x4E, 0x47])
        let url = try XCTUnwrap(URL(string: "https://example.com/p.png"))
        let out = html(md, images: [url: png])
        XCTAssertTrue(out.contains("<a href=\"https://example.com\">ref"), out)
        XCTAssertTrue(out.contains("<img alt=\"p\""), out)
    }

    func testImagesAreFoundInsideQuotesListsAndHeaders() {
        let md = "> ![q](https://e.com/q.png)\n\n- ![l](https://e.com/l.png)"
            + "\n\n| ![h](https://e.com/h.png) |\n|---|\n| x |"
        let names = ImagePrefetch.collectURLs(in: Markdown.parse(md))
            .map { url in url.lastPathComponent }.sorted()
        XCTAssertEqual(names, ["h.png", "l.png", "q.png"])
    }

    func testOnlyWebImagesAreFetched() throws {
        XCTAssertTrue(ImagePrefetch.fetchable(
            try XCTUnwrap(URL(string: "https://e.com/a.png"))))
        XCTAssertFalse(ImagePrefetch.fetchable(
            URL(fileURLWithPath: "/dev/zero")))
        XCTAssertFalse(ImagePrefetch.fetchable(
            try XCTUnwrap(URL(string: "images/a.png"))))
    }

    func testPdfTextKeepsItsLinksAndEmphasis() throws {
        let data = try XCTUnwrap(MarkdownPDF.data(
            Markdown.parse("See **bold** and [the site](https://example.com) "
                + "but not [this](javascript:alert(1))."), title: "t"))
        let doc = try XCTUnwrap(PDFDocument(data: data))
        let page = try XCTUnwrap(doc.page(at: 0))
        let links = page.annotations.compactMap { note in
            note.url?.absoluteString
        }
        XCTAssertTrue(links.contains("https://example.com"), "\(links)")
        XCTAssertFalse(links.contains { link in link.hasPrefix("javascript") })
        var media = CGRect(x: 0, y: 0, width: 600, height: 800)
        let buffer = NSMutableData()
        let consumer = try XCTUnwrap(CGDataConsumer(data: buffer))
        let ctx = try XCTUnwrap(CGContext(consumer: consumer,
                                          mediaBox: &media, nil))
        let renderer = PDFRenderer(ctx: ctx, pageSize: media.size,
                                   title: "t", images: [:])
        var bold = false
        if case .paragraph(let attr)? = Markdown.parse("See **bold** here.")
            .items.first?.block {
            let text = renderer.styled(attr, base: CTFontCreateWithName(
                "Helvetica" as CFString, 11, nil), color: renderer.textColor)
            let at = (text.string as NSString).range(of: "bold").location
            if let font = text.attribute(.font, at: at, effectiveRange: nil) {
                bold = CTFontGetSymbolicTraits(font as! CTFont)
                    .contains(.traitBold)
            }
        }
        XCTAssertTrue(bold, "the bold run lost its weight")
    }

    private let mono = PlatformFont.monospacedSystemFont(ofSize: 12,
                                                         weight: .regular)

    private func ink(_ code: String, _ language: String,
                     at needle: String) -> PlatformColor? {
        let text = Highlight.attribute(code, language: language,
                                       baseFont: mono)
        let at = (code as NSString).range(of: needle).location
        return text.attribute(.foregroundColor, at: at,
                              effectiveRange: nil) as? PlatformColor
    }

    func testTheHighlighterReadsCommentsStringsAndKeysInOrder() {
        let js = "let u = \"http://e.com\"; // note"
        XCTAssertEqual(ink(js, "js", at: "//e.com"), ink(js, "js", at: "\""))
        XCTAssertNotEqual(ink(js, "js", at: "note"), ink(js, "js", at: "\""))
        let json = "{\"key\": \"value\"}"
        XCTAssertNotEqual(ink(json, "json", at: "key"),
                          ink(json, "json", at: "value"))
        let ruby = "s = 'it\\'s' + 'x' # note"
        XCTAssertNotEqual(ink(ruby, "ruby", at: "+ '"),
                          ink(ruby, "ruby", at: "it"),
                          "the escaped quote ended the string")
        XCTAssertNotEqual(ink(ruby, "ruby", at: "note"),
                          ink(ruby, "ruby", at: "it"))
        XCTAssertEqual(ink("x = 1", "rs", at: "1"),
                       ink("x = 1", "rust", at: "1"))
        let quoted = "/* it's */ s = 'abc';"
        XCTAssertEqual(ink(quoted, "js", at: "abc"),
                       ink("q = 'z';", "js", at: "z"))
    }

    func testALongDigitRunHighlightsInLinearTime() {
        let digits = String(repeating: "7", count: 20_000) + "x"
        let start = ContinuousClock.now
        _ = Highlight.attribute(digits, language: "c", baseFont: mono)
        XCTAssertLessThan(ContinuousClock.now - start, .seconds(1))
    }

    func testHostileCodeHighlightsWithinItsBudget() {
        let hostile = [("js", String(repeating: "/* ", count: 20_000)),
                       ("rust", String(repeating: "\" x ", count: 20_000))]
        for (language, code) in hostile {
            let start = ContinuousClock.now
            _ = Highlight.attribute(code, language: language, baseFont: mono)
            XCTAssertLessThan(ContinuousClock.now - start, .seconds(1),
                              language)
        }
    }
}
