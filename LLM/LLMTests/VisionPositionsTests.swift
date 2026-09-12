import XCTest
@testable import LLM

final class VisionPositionsTests: XCTestCase {

    // Text-only positions advance a shared 3D scalar from startScalar; next is
    // the scalar just past the last token (the decode start).
    func testTextPositionsStartAtScalar() {
        let fresh = VisionPositions.visionPositionsMulti(3, [])
        XCTAssertEqual(fresh.next, 3)
        XCTAssertEqual(fresh.pos.map { $0.0 }, [0, 1, 2])

        let plan = VisionPositions.visionPositionsMulti(3, [], startScalar: 5)
        XCTAssertEqual(plan.next, 8)
        XCTAssertEqual(plan.pos.map { $0.0 }, [5, 6, 7])
        XCTAssertEqual(plan.pos.map { $0.1 }, [5, 6, 7])
        XCTAssertEqual(plan.pos.map { $0.2 }, [5, 6, 7])
    }

    // One image span (4x4 grid -> 2x2 = 4 merged tokens) based at startScalar:
    func testImagePositionsBasedAtScalar() {
        let plan = VisionPositions.visionPositionsMulti(
            4, [(start: 0, gh: 4, gw: 4)], startScalar: 10)
        XCTAssertEqual(plan.pos[0].0, 10)
        XCTAssertEqual([plan.pos[0].1, plan.pos[0].2], [10, 10])
        XCTAssertEqual([plan.pos[1].1, plan.pos[1].2], [10, 11])
        XCTAssertEqual([plan.pos[2].1, plan.pos[2].2], [11, 10])
        XCTAssertEqual([plan.pos[3].1, plan.pos[3].2], [11, 11])
        XCTAssertEqual(plan.next, 12)
    }

    // A text token then an image: the image bases at the post-text scalar, and
    // text resumes past the image's grid extent.
    func testTextThenImageChainsScalar() {
        let plan = VisionPositions.visionPositionsMulti(
            5, [(start: 1, gh: 4, gw: 4)], startScalar: 10)
        XCTAssertEqual(plan.pos[0].0, 10)          // text at the seed scalar
        XCTAssertEqual(plan.pos[1].0, 11)          // image bases at 11
        XCTAssertEqual(plan.next, 13)              // 11 + 4/2
    }
}
