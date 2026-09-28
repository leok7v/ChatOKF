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

    func testLongBlocksFinishEqualsParse() {
        for (name, sample) in Self.longSamples {
            let expected = Markdown.parse(sample).items.map { i in i.block }
            for seed in Self.pieceSeeds {
                let stream = MarkdownStream()
                var fed = 0
                for piece in Self.pieces(sample, seed: seed) {
                    stream.append(piece)
                    fed += 1
                    if fed % 50 == 0 { _ = stream.snapshot() }
                }
                let got = stream.finish().items.map { i in i.block }
                XCTAssertEqual(got, expected,
                    "\(name) with piece seed \(seed) diverged")
            }
        }
    }

    func testLongBlocksSnapshotEqualsParseOfPrefix() {
        for (name, sample) in Self.longSamples {
            let stream = MarkdownStream()
            var fed = ""
            var pieces = 0
            for piece in Self.pieces(sample, seed: 7) {
                stream.append(piece)
                fed += piece
                pieces += 1
                let pending = fed.split(separator: "\n",
                                        omittingEmptySubsequences: false)
                    .last ?? ""
                let halfDefinition =
                    Markdown.parseLinkDefinition(String(pending)) != nil
                if pieces % 5 == 0, !halfDefinition {
                    let got = stream.snapshot().items.map { i in i.block }
                    let seen = fed.hasSuffix("\n") ? String(fed.dropLast())
                                                   : fed
                    let want = Markdown.parse(seen).items.map { i in i.block }
                    XCTAssertEqual(got, want,
                        "\(name) snapshot after \(pieces) pieces diverged")
                }
            }
        }
    }

    func testStreamingCostOfLongBlocks() {
        for (name, sample) in Self.longSamples {
            let start = DispatchTime.now().uptimeNanoseconds
            let stream = MarkdownStream()
            var fed = 0
            for ch in sample {
                stream.append(String(ch))
                fed += 1
                if fed % 50 == 0 { _ = stream.snapshot() }
            }
            let blocks = stream.finish().items.count
            let end = DispatchTime.now().uptimeNanoseconds
            let ms = Double(end - start) / 1_000_000
            print(String(format: "stream-cost %@: %.1f ms, %d chars, "
                         + "%d snapshots, %d blocks", name, ms, fed,
                         fed / 50, blocks))
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
        "First<br>second line<br/>third<br />fourth.",
        "Some <small>small print</small>, a <kbd>Cmd</kbd> key and a "
            + "<!-- dropped --> comment, `<br>` in code.",
        "<!-- a block comment\nthat spans lines -->\n\nStill here.",
        "| a | b |\n|---|:-:|\n| x \\| y | z<br>w |",
        "<img src=\"https://example.com/i.png\" alt=\"Logo\" width=\"32\">",
        "Bold <b>b</b>, <a href=\"https://example.com/a\">to a</a> and an "
            + "<unknown attr=\"x\">unknown tag</unknown> stays.",
        "Displays:\n\n$$a$$ $$b$$ $$c$$\n\nafter them.",
        "$$a$$ is the area\n$$b$$ $$c$$ and more",
        "$$\nx^2\n$$ tail text\n\nnext",
        "<!-- note --> visible after a comment\n\nmore",
        "````md\n```swift\nlet x = 1\n```\n````\n\nafter",
        "Title\n===\n\nSub\n---\n\n## Closed ##\n",
        "In the year\n1984. Things changed.\n- a bullet\n- two",
        "[Note]: not a definition.\n\n[^1]: nor a footnote",
        "    # indented code\n    ---\n\nwords\n    still words",
    ]

    static let pieceSeeds: [UInt64] = [0, 7, 20260926]

    static func pieces(_ s: String, seed: UInt64) -> [String] {
        var out: [String] = []
        var state = seed
        var i = s.startIndex
        while i < s.endIndex {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            let size = seed == 0 ? 1 : Int(z % 17) + 1
            let j = s.index(i, offsetBy: size, limitedBy: s.endIndex)
                ?? s.endIndex
            out.append(String(s[i..<j]))
            i = j
        }
        return out
    }

    static let longSamples: [(String, String)] = [
        ("flat list 200", (1...200).map { n in
            "- item \(n) with **bold** and `code`"
        }.joined(separator: "\n")),
        ("nested list 200", (1...200).map { n in
            "\(n). step \(n)\n    - detail a of \(n)\n    - detail b of \(n)"
        }.joined(separator: "\n") + "\n"),
        ("loose list 200", (1...200).map { n in
            "* point \(n)\n\n  continued \(n)"
        }.joined(separator: "\n\n")),
        ("table 200", "| n | name | cost |\n|--:|:--|:-:|\n"
            + (1...200).map { n in
                "| \(n) | item \(n) | \(n * 3) |"
            }.joined(separator: "\n")),
        ("fence 400", "~~~text\n" + (1...400).map { n in
            let shapes = StreamTests.fenceLines
            return shapes[n % shapes.count] + " \(n)"
        }.joined(separator: "\n") + "\n~~~\n"),
        ("mixed", StreamTests.mixedSample),
    ]

    static let fenceLines: [String] = [
        "- looks like an item", "# looks like a heading", "1. numbered",
        "| a | b |", "|---|---|", "> quoted", "```", "    indented", "",
        "plain line", "$$ x $$", "---",
    ]

    static let mixedSample: String = [
        "# Title",
        "",
        "Intro paragraph with *emphasis* and `code`.",
        "",
        (1...40).map { n in "- bullet \(n)" }.joined(separator: "\n"),
        "",
        "| a | b |\n|---|---|\n"
            + (1...40).map { n in "| \(n) | \(n * n) |" }
                .joined(separator: "\n"),
        "",
        "```swift\n" + (1...60).map { n in "let v\(n) = \(n)" }
            .joined(separator: "\n") + "\n```",
        "",
        (1...30).map { n in "> line \(n) of the quote" }
            .joined(separator: "\n"),
        "",
        (1...30).map { n in "\(n). loose \(n), see [docs][d]" }
            .joined(separator: "\n\n"),
        "",
        "[d]: https://example.com/docs",
        "",
        "$$\\sum_{i=0}^{n} i$$",
        "",
        "Closing paragraph after everything.",
    ].joined(separator: "\n")
}
