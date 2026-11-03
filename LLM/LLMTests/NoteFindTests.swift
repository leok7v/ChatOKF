import XCTest
@testable import LLM

final class NoteFindTests: XCTestCase {

    private func concept(_ id: String, _ title: String,
                         _ body: String) -> Concept {
        Concept(id: id, path: URL(fileURLWithPath: "/" + id + ".md"),
                type: "Document", title: title, description: "", tags: [],
                status: "", staleAfter: nil, generatedBy: "", verifiers: [],
                extraFrontmatter: [], body: body, links: [], hash: 0)
    }

    func testTermsKeepANumberWholeAndStemAPlural() {
        XCTAssertEqual(NoteFind.terms("rule 4.2"),
                       [.word("rule"), .phrase("4.2")])
        XCTAssertEqual(NoteFind.terms("paragraph sentences"),
                       [.word("paragraph"), .word("sentence")])
        XCTAssertEqual(NoteFind.terms("\"make sure\" contract*"),
                       [.phrase("make sure"), .prefix("contract")])
        XCTAssertEqual(NoteFind.terms("[rule 7.3, safety]"),
                       [.word("rule"), .phrase("7.3"), .word("safety")])
        XCTAssertEqual(NoteFind.stem("sentences"), "sentence")
        XCTAssertEqual(NoteFind.stem("boxes"), "box")
        XCTAssertEqual(NoteFind.stem("bodies"), "body")
        XCTAssertEqual(NoteFind.stem("glass"), "glass")
    }

    func testAWindowPrefersOneLineAndAHeadingShowsTheNextLine() {
        let note = concept("docs/sec", "Section 6", """
        > Rule 6.3 Write short sentences.
        >
        > **Paragraphs**

        ### Rule 6.6

        > **Make sure that no paragraph has more than six sentences.**
        """)
        let lines = NoteFind.lines(NoteFind.terms("paragraph sentences"),
                                   asked: "paragraph sentences", limit: 30,
                                   in: [note])
        let first = lines.components(separatedBy: "\n").first ?? ""
        XCTAssertTrue(first.hasPrefix("docs/sec:9  >"), lines)
        XCTAssertTrue(first.contains("six sentences"), lines)
        let rule = NoteFind.lines(NoteFind.terms("rule 6.6"),
                                  asked: "rule 6.6", limit: 30, in: [note])
        XCTAssertTrue(rule.contains("docs/sec:7  ### Rule 6.6"), rule)
        XCTAssertTrue(rule.contains("docs/sec:9  >"), rule)
    }

    func testAStopWordIsDroppedBeforeAndAfterTheStem() {
        let terms = NoteFind.contentTerms("Does this rule allow contractions")
        XCTAssertEqual(terms, [.word("rule"), .word("allow"),
                               .word("contraction")])
    }

    func testTheMoreLineCountsPlacesNotLines() {
        let note = concept("docs/p", "P", "alpha\nbeta\n\nalpha\nbeta\n\n"
                           + "alpha\nbeta\n\nalpha\nbeta")
        let lines = NoteFind.lines(NoteFind.terms("alpha beta"),
                                   asked: "alpha beta", limit: 4, in: [note])
        XCTAssertEqual(lines.components(separatedBy: "docs/p:").count, 5,
                       lines)
        XCTAssertTrue(lines.contains("2 more place(s) not shown"), lines)
    }

    func testTheFindRelaxesAndSaysSo() {
        let note = concept("docs/a", "A",
                           "Use a maximum of 20 words in each sentence.")
        let lines = NoteFind.lines(
            NoteFind.terms("sentence word limit procedure"),
            asked: "sentence word limit procedure", limit: 30, in: [note])
        XCTAssertTrue(lines.hasPrefix("no place holds all 4 words"), lines)
        XCTAssertTrue(lines.contains("docs/a:3  Use a maximum"), lines)
        let none = NoteFind.lines(NoteFind.terms("zebra"), asked: "zebra",
                                  limit: 30, in: [note])
        XCTAssertTrue(none.hasPrefix("no place in the user's notes"), none)
    }

    func testTheReadNumbersLinesAndSaysHowToContinue() {
        let note = concept("docs/b", "B", "one\ntwo\nthree\nfour")
        let read = NoteFind.read(note, from: 3, count: 2, more: "read_note")
        XCTAssertEqual(read.end, 4)
        XCTAssertTrue(read.text.contains("3: one\n4: two\n"), read.text)
        XCTAssertTrue(read.text.contains("read_note docs/b line=5"), read.text)
        let tail = NoteFind.read(note, from: 5, count: 40, more: "read_note")
        XCTAssertEqual(tail.end, 6)
        XCTAssertFalse(tail.text.contains("for more"), tail.text)
    }

    func testTheCutEndsOnAParagraphASentenceOrAWord() {
        let paragraphs = "First paragraph here.\n\nSecond one is long enough "
            + "to pass the limit of the cut by a good margin, really."
        XCTAssertEqual(NoteFind.cut(paragraphs, limit: 60),
                       "First paragraph here.")
        let sentences = "One sentence. Another sentence that runs past the "
            + "limit of the cut and on."
        XCTAssertEqual(NoteFind.cut(sentences, limit: 40), "One sentence.")
        let words = "no punctuation at all in this text that runs on and on"
        XCTAssertEqual(NoteFind.cut(words, limit: 20), "no punctuation at ...")
        XCTAssertEqual(NoteFind.cut("short", limit: 20), "short")
    }

    func testATableRowCarriesItsHeaderWhenTheHeaderIsShort() {
        let table = """
        | Word | Approved | Example |
        | --- | --- | --- |
        | ensure (v) | MAKE SURE (v) | MAKE SURE THAT IT IS ON. |
        """
        let shaped = NoteFind.shaped(table)
        XCTAssertTrue(shaped.contains(
            "ensure (v) | Approved: MAKE SURE (v) | Example: MAKE SURE"),
            shaped)
        let wide = """
        | Before closing the hatch it is mandatory to ensure | x |
        | --- | --- |
        | a | b |
        """
        XCTAssertTrue(NoteFind.shaped(wide).contains("| a | b |"))
    }
}
