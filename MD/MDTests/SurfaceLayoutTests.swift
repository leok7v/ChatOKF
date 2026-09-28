import XCTest
@testable import MD

@MainActor
final class SurfaceLayoutTests: XCTestCase {

    private let source = """
        # Title

        First paragraph with **bold** and a [link](https://e.com).

        - [x] a done item
        - [ ] an open item

        ```swift
        let x = 1
        ```

        | a | b |
        |---|---|
        | 1 | 2 |

        $$x^2$$

        ## Last heading

        The last paragraph.
        """

    func testAnEditReplacesOnlyFromTheEditedBlock() throws {
        let edited = source.replacingOccurrences(of: "The last paragraph.",
                                                 with: "The last one, edited.")
        let cache = DocumentText.RenderCache()
        let view = NativeText.ResizingTextView()
        view.applyResolved(DocumentText.attributed(
            from: Markdown.parse(source), style: .default, cache: cache))
        let storage = try XCTUnwrap(view.textStorage)
        let next = DocumentText.attributed(from: Markdown.parse(edited),
                                           style: .default, cache: cache)
        let replaced = incrementalRange(storage, next)
        let last = (storage.string as NSString)
            .range(of: "The last paragraph.").location
        XCTAssertGreaterThanOrEqual(replaced.location, last,
                                    "the splice starts before the edited block")
        _ = applyIncremental(storage, next)
        XCTAssertEqual(storage.string, next.string)
    }

    private func text(_ md: String) -> NSAttributedString {
        DocumentText.attributed(from: Markdown.parse(md), style: .default)
    }

    private func para(_ text: NSAttributedString,
                      at needle: String) -> NSParagraphStyle? {
        let at = (text.string as NSString).range(of: needle).location
        return text.attribute(.paragraphStyle, at: at, effectiveRange: nil)
            as? NSParagraphStyle
    }

    func testANestedListIndentsUnderItsParent() {
        let t = text("- outer\n  - inner\n- next")
        let outer = para(t, at: "outer")?.headIndent ?? 0
        let inner = para(t, at: "inner")?.headIndent ?? 0
        XCTAssertGreaterThan(inner, outer)
        XCTAssertEqual(para(t, at: "next")?.headIndent, outer)
    }

    func testAListInAListShareTheMarkerLine() {
        XCTAssertTrue(text("- - a").string.hasPrefix("\u{2022}\t\u{2022}\ta\n"),
                      text("- - a").string.debugDescription)
    }

    func testAWideOrdinalGetsTheRoomItNeeds() {
        let t = text("100. a\n101. b")
        let width = NSAttributedString(
            string: "100.",
            attributes: [.font: DocumentText.bodyFont(.default)])
            .size().width
        XCTAssertGreaterThan(para(t, at: "a")?.headIndent ?? 0, width)
    }

    func testAnItemOpeningWithCodeKeepsTheCodeIndented() {
        let t = text("- ```swift\n  let x = 1\n  ```\n- after")
        XCTAssertTrue(t.string.hasPrefix("\u{2022}\n"), t.string)
        let marker = para(t, at: "after")?.headIndent ?? 0
        XCTAssertGreaterThanOrEqual(para(t, at: "let x")?.headIndent ?? 0,
                                    marker)
        let kind = t.attribute(atomicKindKey,
                               at: (t.string as NSString)
                                   .range(of: "let x").location,
                               effectiveRange: nil) as? String
        XCTAssertEqual(kind, AtomicKind.code.rawValue)
    }

    func testAListEndingInANestedListKeepsTheBlockSpacing() {
        let t = text("- a\n  - b\n\nAfter.")
        XCTAssertGreaterThanOrEqual(para(t, at: "b")?.paragraphSpacing ?? 0,
                                    MarkdownStyle.default.blockSpacing)
    }

    func testAnArrivingImageRerendersOnlyItsOwnBlock() throws {
        let doc = Markdown.parse("Words.\n\n![p](https://e.com/p.png)")
        let cache = DocumentText.RenderCache()
        _ = DocumentText.attributed(from: doc, style: .default, cache: cache)
        let before = try XCTUnwrap(cache.entries[0]?.text)
        let url = try XCTUnwrap(URL(string: "https://e.com/p.png"))
        _ = DocumentText.attributed(from: doc, style: .default,
                                    images: [url: PlatformImage()],
                                    cache: cache)
        XCTAssertTrue(cache.entries[0]?.text === before,
                      "the paragraph re-rendered for an image it lacks")
        XCTAssertNotNil(cache.entries[1]?.images[url])
    }

    func testACellIsNeverNarrowerThanItsOwnLongestRun() {
        for text in ["`code`", "a `monospaced` word<br>and a second line"] {
            let cell = DocumentText.tableCell(
                text, base: DocumentText.bodyFont(.default), style: .default,
                images: [:])
            XCTAssertGreaterThanOrEqual(
                DocumentText.attributedWidth(cell.text), cell.minimum, text)
        }
    }

    func testADisplayAddsNoSpaceOfItsOwnBefore() {
        let t = text("A.\n\n$$x$$")
        let at = t.length - 2
        let style = t.attribute(.paragraphStyle, at: at, effectiveRange: nil)
            as? NSParagraphStyle
        XCTAssertEqual(style?.paragraphSpacingBefore, 0)
    }
}
