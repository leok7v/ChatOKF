import XCTest
@testable import MD

// MarkdownFindController drives one global cursor across per-bubble text
// views; these pin the cross-view contracts a ranges test cannot reach.
@MainActor
final class FindControllerTests: XCTestCase {

    // A stand-in for the platform text views: findAll recomputes its matches,
    // setActive records every call (so the outgoing-only clear is observable).
    final class MockFindView: FindableTextView {
        let text: String
        private(set) var matches: [NSRange] = []
        private(set) var setActiveCalls: [Int?] = []
        private(set) var spoken: String?

        init(_ text: String) { self.text = text }

        func setSpoken(_ text: String?) { spoken = text }

        func findAll(_ query: String, caseSensitive: Bool) -> Int {
            matches = markdownFindRanges(in: text, query: query,
                                         caseSensitive: caseSensitive)
            return matches.count
        }

        func setActive(_ localIndex: Int?) { setActiveCalls.append(localIndex) }
        func clearFind() { matches = [] }
        var liveFindCount: Int { matches.count }
        func activeMatchFraction() -> CGFloat? { nil }

        func clearActiveLog() { setActiveCalls.removeAll() }
    }

    // After find(), the cursor sits on the first match; Previous wraps to the
    // last (standard editor behavior, and the N2 note's open question).
    func testFindPreviousWrapsToLastAfterFind() {
        let c = MarkdownFindController()
        let a = MockFindView("x . x . x")
        let id = UUID()
        c.order = [id]
        c.register(id, a)
        XCTAssertEqual(c.find("x"), 3)
        XCTAssertEqual(c.currentMatch, 1)
        XCTAssertEqual(c.findPrevious(), 3)
        XCTAssertEqual(c.currentMatch, 3)
    }

    // A bubble that registers while a find is active (a new streaming turn) is
    // counted once the next step recounts, not lost until find() re-runs.
    func testLateRegistrationCountedOnStep() {
        let c = MarkdownFindController()
        let a = MockFindView("x . x")
        let ia = UUID()
        let ib = UUID()
        c.order = [ia, ib]
        c.register(ia, a)
        XCTAssertEqual(c.find("x"), 2)
        let b = MockFindView("x")
        c.register(ib, b)
        _ = c.findNext()
        XCTAssertEqual(c.matchCount, 3)
    }

    // Stepping clears the active selection only on the OUTGOING view; an
    // unrelated bubble is never touched, so a manual selection there survives.
    func testStepClearsOnlyOutgoingView() {
        let c = MarkdownFindController()
        let a = MockFindView("x")
        let b = MockFindView("x")
        let d = MockFindView("no match here")
        let ia = UUID()
        let ib = UUID()
        let idd = UUID()
        c.order = [ia, ib, idd]
        c.register(ia, a)
        c.register(ib, b)
        c.register(idd, d)
        _ = c.find("x")
        a.clearActiveLog()
        b.clearActiveLog()
        d.clearActiveLog()
        _ = c.findNext()
        XCTAssertTrue(a.setActiveCalls.contains(where: { $0 == nil }))
        XCTAssertTrue(b.setActiveCalls.contains(where: { $0 == 0 }))
        XCTAssertTrue(d.setActiveCalls.isEmpty)
    }

    // Forward stepping visits every match across views in order and wraps.
    func testFindNextCyclesAcrossViews() {
        let c = MarkdownFindController()
        let a = MockFindView("x")
        let b = MockFindView("x . x")
        let ia = UUID()
        let ib = UUID()
        c.order = [ia, ib]
        c.register(ia, a)
        c.register(ib, b)
        XCTAssertEqual(c.find("x"), 3)
        XCTAssertEqual(c.currentMatch, 1)
        XCTAssertEqual(c.findNext(), 2)
        XCTAssertEqual(c.findNext(), 3)
        XCTAssertEqual(c.findNext(), 1)
    }
}
