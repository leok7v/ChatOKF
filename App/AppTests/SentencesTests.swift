import Chat
import XCTest

final class SentencesTests: XCTestCase {

    func testCutsAtTerminatorsAndNewlines() {
        let parts = Sentences.split("One. Two! Three?\nFour")
        XCTAssertEqual(parts, ["One.", "Two!", "Three?", "Four"])
    }

    func testKeepsAbbreviationsListMarkersAndNumbers() {
        let parts = Sentences.split("Use e.g. 3.14 here. 1. first\n2. second")
        XCTAssertEqual(parts, ["Use e.g. 3.14 here.", "1. first", "2. second"])
    }

    func testTheLastPieceIsThePartialOne() {
        let parts = Sentences.split("Done. Still going")
        XCTAssertEqual(parts, ["Done.", "Still going"])
        let grown = Sentences.split("Done. Still going on.")
        XCTAssertEqual(grown[0], parts[0])
    }

    func testCapsARunawayLineAtWhitespace() {
        let long = String(repeating: "word ", count: 80)
        let parts = Sentences.split(long)
        XCTAssertGreaterThan(parts.count, 1)
        XCTAssertLessThanOrEqual(parts[0].utf8.count, Sentences.maxLength + 4)
    }

    func testClosingQuotesDoNotHideTheMark() {
        XCTAssertEqual(Sentences.split("He said \"go.\" Then left"),
                       ["He said \"go.\"", "Then left"])
    }

}
