import XCTest
@testable import Chat

final class AboutYouTests: XCTestCase {

    func testNoNameIsNoLine() {
        XCTAssertEqual(AboutYou.line(name: " \n"), "")
    }

    func testTheLineAsksForTheNameInTheReasoningToo() {
        let line = AboutYou.line(name: " Leo \n")
        XCTAssertEqual(line, "[You are talking to Leo. In your private "
            + "reasoning, refer to this person as Leo. In your replies, "
            + "speak to Leo directly as you." + AboutYou.byName)
        XCTAssertFalse(line.contains("\""), "no phrase to copy")
    }

    func testTheSmallestModelIsNotTold() {
        XCTAssertTrue(Models.isSimple("Ternary-Bonsai-1.7B"))
        XCTAssertFalse(Models.isSimple("gemma-4-E2B"))
    }

    func testWhitespaceInsideTheNameIsOneSpace() {
        XCTAssertEqual(AboutYou.called("Mary \n\t Ann"), "Mary Ann")
    }

    func testTheNameIsCutAtItsLimit() {
        XCTAssertEqual(
            AboutYou.called(String(repeating: "x", count: 200)).count, 40)
    }
}
