import XCTest
@testable import Chat

final class AboutYouTests: XCTestCase {

    private let opening = "[About the user you are talking to: "

    func testNothingFilledIsNoLine() {
        XCTAssertEqual(AboutYou.line(name: "", gender: " ", age: "\n"), "")
    }

    func testTheLineCarriesOnlyWhatIsFilled() {
        XCTAssertEqual(
            AboutYou.line(name: "Leo", gender: "male", age: "63"),
            opening + "name Leo, gender male, age 63." + AboutYou.kept
            + AboutYou.byName + "]")
        XCTAssertEqual(
            AboutYou.line(name: " Leo \n", gender: "", age: ""),
            opening + "name Leo." + AboutYou.kept
            + AboutYou.byName + "]")
        XCTAssertEqual(AboutYou.line(name: "", gender: "", age: "9"),
                       opening + "age 9." + AboutYou.kept + "]")
    }

    func testTheSmallestModelIsNotTold() {
        XCTAssertTrue(Models.isSimple("Ternary-Bonsai-1.7B"))
        XCTAssertFalse(Models.isSimple("gemma-4-E2B"))
    }

    func testWhitespaceInsideAFieldIsOneSpace() {
        XCTAssertTrue(AboutYou.line(name: "Mary \n\t Ann", gender: "",
                                    age: "")
            .contains("name Mary Ann."))
    }

    func testAFieldIsCutAtItsLimit() {
        let long = String(repeating: "x", count: 200)
        let line = AboutYou.line(name: long, gender: "", age: "")
        XCTAssertFalse(line.contains(String(repeating: "x", count: 41)))
    }
}
