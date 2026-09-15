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
}
