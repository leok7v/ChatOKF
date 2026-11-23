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

    func testAParkKeepsItsOwnTargetAsTheListingSpellsIt() throws {
        let fm = FileManager.default
        let top = fm.temporaryDirectory.appendingPathComponent(
            "parking-" + UUID().uuidString, isDirectory: true)
        let real = top.appendingPathComponent("real", isDirectory: true)
        try fm.createDirectory(
            at: real.appendingPathComponent("parked", isDirectory: true),
            withIntermediateDirectories: true)
        let link = top.appendingPathComponent("link")
        try fm.createSymbolicLink(atPath: link.path,
                                  withDestinationPath: real.path)
        let base = link.appendingPathComponent("parked", isDirectory: true)
        let mine = base.appendingPathComponent(
            UUID().uuidString + ".model.rev", isDirectory: true)
        let other = base.appendingPathComponent(
            UUID().uuidString + ".model.rev", isDirectory: true)
        try fm.createDirectory(at: mine, withIntermediateDirectories: true)
        try fm.createDirectory(at: other, withIntermediateDirectories: true)
        let found = try fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: [.isDirectoryKey])
        XCTAssertFalse(found.contains(mine))
        let dropped = Session.strangers(found, keeping: [mine, mine])
        XCTAssertEqual(dropped.map { url in url.lastPathComponent },
                       [other.lastPathComponent])
        XCTAssertEqual(Session.strangers(found, keeping: [mine, nil]).count,
                       1)
        try fm.removeItem(at: top)
    }

    func testAParkKeepsTheConversationBeingOpened() {
        let base = URL(fileURLWithPath: "/tmp/parked", isDirectory: true)
        let leaving = base.appendingPathComponent("A.model.rev")
        let opening = base.appendingPathComponent("B.model.rev")
        let other = base.appendingPathComponent("C.model.rev")
        let found = [leaving, opening, other]
        XCTAssertEqual(Session.parkDrops(found, parking: leaving, live: nil,
                                         opening: opening, resumable: false),
                       [other])
        XCTAssertEqual(Session.parkDrops(found, parking: leaving, live: nil,
                                         opening: nil, resumable: false),
                       [opening, other])
        XCTAssertEqual(Session.parkDrops(found, parking: leaving, live: nil,
                                         opening: opening, resumable: true),
                       [])
    }
}
