import XCTest
@testable import MD

final class StreamTests: XCTestCase {

    // Feeding a document in any chunking must, after finish(), equal the
    // batch parse block-for-block. This is the core streaming contract.
    func testStreamFinishEqualsParse() {
        for sample in Self.samples {
            let expected = Markdown.parse(sample).items.map { i in i.block }
            for size in [1, 3, 7, 13, 128] {
                let stream = MarkdownStream()
                for chunk in Self.chunked(sample, size: size) {
                    stream.append(chunk)
                }
                let got = stream.finish().items.map { i in i.block }
                XCTAssertEqual(got, expected,
                    "chunk size \(size) diverged for sample:\n\(sample)")
            }
        }
    }

    // A snapshot mid-stream never crashes and always yields ids 0..<count.
    func testSnapshotIdsAreDense() {
        let stream = MarkdownStream()
        let text = Self.samples.joined(separator: "\n\n")
        for chunk in Self.chunked(text, size: 4) {
            stream.append(chunk)
            let ids = stream.snapshot().items.map { i in i.id }
            XCTAssertEqual(ids, Array(0..<ids.count))
        }
    }

    // A sealed block keeps its id and value as more tokens arrive: every
    // snapshot's sealed prefix is a prefix of the final document.
    func testSealedPrefixStable() {
        let stream = MarkdownStream()
        let text = """
        # Title

        First paragraph.

        Second paragraph.

        - a
        - b

        ```swift
        let x = 1
        ```

        Done.
        """
        var snapshots: [[Markdown.Block]] = []
        for chunk in Self.chunked(text, size: 6) {
            stream.append(chunk)
            snapshots.append(stream.snapshot().items.map { i in i.block })
        }
        let final = stream.finish().items.map { i in i.block }
        // Every earlier snapshot's non-last blocks must appear unchanged in
        // the final document (sealed blocks never mutate).
        for snap in snapshots where snap.count >= 2 {
            let sealed = Array(snap.dropLast())
            XCTAssertEqual(Array(final.prefix(sealed.count)), sealed)
        }
    }

    // Table alignment is parsed from the separator row.
    func testTableAlignment() {
        let md = """
        | a | b | c | d |
        |:--|:-:|--:|---|
        | 1 | 2 | 3 | 4 |
        """
        let doc = Markdown.parse(md)
        guard case .table(_, _, let aligns) = doc.items.first?.block else {
            return XCTFail("expected a table")
        }
        XCTAssertEqual(aligns, [.left, .center, .right, .none])
    }

    // A reference definition seen before its use resolves while streaming.
    func testReferenceLinkBackward() {
        let md = """
        [home]: https://example.com

        See [the site][home] here.
        """
        let doc = Markdown.parse(md)
        var linked = false
        for item in doc.items {
            if case .paragraph(let a) = item.block {
                for run in a.runs where run.link != nil { linked = true }
            }
        }
        XCTAssertTrue(linked, "reference link did not resolve")
    }

    // An unterminated fence becomes a code block on finish.
    func testUnterminatedFence() {
        let stream = MarkdownStream()
        stream.append("```swift\nlet x = 1\nlet y = 2\n")
        let blocks = stream.finish().items.map { i in i.block }
        XCTAssertEqual(blocks.count, 1)
        if case .code(let lang, let text) = blocks.first {
            XCTAssertEqual(lang, "swift")
            XCTAssertEqual(text, "let x = 1\nlet y = 2")
        } else {
            XCTFail("expected a code block")
        }
    }

    private static func chunked(_ s: String, size: Int) -> [String] {
        var out: [String] = []
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: size, limitedBy: s.endIndex)
                ?? s.endIndex
            out.append(String(s[i..<j]))
            i = j
        }
        return out
    }

    static let samples: [String] = [
        "# Heading one\n\nA paragraph with **bold**, *italic*, `code`.",
        "## H2\n### H3\ntext under headings\n",
        "Para line one\nsame paragraph line two\n\nnew paragraph",
        "- one\n- two\n- three",
        "1. first\n2. second\n3. third",
        "- [ ] todo\n- [x] done",
        "- loose\n\n- list\n\n- items",
        "- outer\n    - nested\n    - nested two\n- outer two",
        "> a quote\n> second line\n\nafter quote",
        "> outer\n> > nested quote\n",
        "```swift\nlet x = 1\nprint(x)\n```\n",
        "    indented code\n    line two\n",
        "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
        "| left | mid | right |\n|:--|:-:|--:|\n| a | b | c |",
        "---\n\ntext\n\n***\n",
        "rule spam\n---\n---\n---\n---\nafter",
        "![alt](https://example.com/x.png)\n",
        "[ref]: https://example.com\n\nlink [here][ref].",
        "Euler: $e^{i\\pi} + 1 = 0$ inline math.",
        "$$\\sum_{i=0}^{n} i$$\n",
        "Mixed:\n\n# Title\n\n- a\n- b\n\n```\ncode\n```\n\n"
            + "| x | y |\n|-|-|\n| 1 | 2 |\n\n> quote\n\nend.",
    ]
}
