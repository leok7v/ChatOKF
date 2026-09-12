import XCTest
@testable import LLM

final class CommandArgsTests: XCTestCase {

    private func line(_ w: String...) -> CommandArgs {
        CommandArgs(["chatokf-cli"] + w)
    }

    func testAValueIsTakenOutOfTheLeftovers() {
        let a = line("model.ggxf", "--system", "be brief", "hello")
        XCTAssertEqual(a.value("--system"), "be brief")
        XCTAssertEqual(a.rest, ["model.ggxf", "hello"])
        XCTAssertEqual(a.turns, ["hello"])
    }

    // The whole point: what a mode consumed decides the leftovers, so a turn
    // is never a flag's argument read twice.
    func testLeftoversFollowWhatWasActuallyRead() {
        let a = line("model.ggxf", "--image", "cat.jpg", "describe it")
        XCTAssertEqual(a.turns, ["cat.jpg", "describe it"])
        XCTAssertEqual(a.value("--image"), "cat.jpg")
        XCTAssertEqual(a.turns, ["describe it"])
    }

    func testABareFlagIsNeverATurn() {
        let a = line("model.ggxf", "--think", "why is the sky blue")
        XCTAssertTrue(a.flag("--think"))
        XCTAssertFalse(a.flag("--cpu"))
        XCTAssertEqual(a.turns, ["why is the sky blue"])
    }

    func testSeveralValuesForOneFlag() {
        let a = line("--splice", "base", "donor", "leaf", "out", "x")
        XCTAssertEqual(a.values("--splice", 4),
                       ["base", "donor", "leaf", "out"])
        XCTAssertEqual(a.rest, ["x"])
    }

    func testAFlagMissingItsValuesReadsNothing() {
        let a = line("--splice", "base", "donor")
        XCTAssertEqual(a.values("--splice", 4), [])
        XCTAssertEqual(a.value("--absent"), nil)
    }

    // A malformed number must not eat the word after it, or the turn goes
    // missing and the mode runs on a default.
    func testAnUnparseableNumberLeavesItsWordAlone() {
        let a = line("model.ggxf", "-n", "lots", "hello")
        XCTAssertNil(a.int("-n"))
        XCTAssertEqual(a.turns, ["-n", "lots", "hello"])
        let b = line("model.ggxf", "-n", "64", "hello")
        XCTAssertEqual(b.int("-n"), 64)
        XCTAssertEqual(b.turns, ["hello"])
    }

    func testAtPathIsResolvedWhereverItIsRead() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ca-\(UUID().uuidString).txt")
        try "from a file".write(to: url, atomically: true, encoding: .utf8)
        let a = line("model.ggxf", "--system", "@" + url.path)
        XCTAssertEqual(a.text("--system"), "from a file")
        let b = line("model.ggxf", "@" + url.path)
        XCTAssertEqual(b.turns, ["from a file"])
        try? FileManager.default.removeItem(at: url)
    }

    func testAMissingFileIsLeftAsTheWordItWas() {
        let a = line("model.ggxf", "--system", "@/no/such/file")
        XCTAssertEqual(a.text("--system"), "@/no/such/file")
    }

    func testEveryWordAfterAFlag() {
        let a = line("model.ggxf", "--tok", "one", "two", "three")
        XCTAssertEqual(a.rest(after: "--tok"), ["one", "two", "three"])
        XCTAssertEqual(a.rest, ["model.ggxf"])
    }

    func testTheModelPathIsTheFirstLeftoverAndNotATurn() {
        let a = line("model.ggxf", "hello", "again")
        XCTAssertEqual(a.rest.first, "model.ggxf")
        XCTAssertEqual(a.turns, ["hello", "again"])
    }
}
