import XCTest
@testable import Chat

final class ParkingTests: XCTestCase {

    func testRefusalNamesTheRuleThatFailed() {
        let budget = ParkBudget(free: 100 << 30, ram: 24 << 30)
        XCTAssertNil(budget.refusal(for: 3 << 30))
        let tight = ParkBudget(free: 4 << 30, ram: 24 << 30)
        XCTAssertTrue(tight.refusal(for: 2 << 30)!.contains("free on disk"))
        let phone = ParkBudget(free: 100 << 30, ram: 8 << 30)
        XCTAssertNil(phone.refusal(for: 1 << 30))
        XCTAssertTrue(phone.refusal(for: 1300 << 20)!.contains("memory"))
    }

    func testTheKnobDropsOnlyAQuickTextReplay() {
        XCTAssertEqual(Session.parkRefusal(
            committed: 0, attached: true, resumable: true, soft: false,
            replaySeconds: 1, budget: nil), "nothing committed")
        XCTAssertEqual(Session.parkRefusal(
            committed: 900, attached: true, resumable: true, soft: false,
            replaySeconds: 6, budget: nil), "replays in 6s")
        XCTAssertNil(Session.parkRefusal(
            committed: 900, attached: true, resumable: false, soft: false,
            replaySeconds: 6, budget: nil))
        XCTAssertNil(Session.parkRefusal(
            committed: 900, attached: true, resumable: true, soft: true,
            replaySeconds: 6, budget: nil))
        XCTAssertNil(Session.parkRefusal(
            committed: 3000, attached: true, resumable: true, soft: false,
            replaySeconds: 18, budget: nil))
        XCTAssertEqual(Session.parkRefusal(
            committed: 3000, attached: true, resumable: true, soft: false,
            replaySeconds: 18, budget: "over a quarter"), "over a quarter")
    }
}
