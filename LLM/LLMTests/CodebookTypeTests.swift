import XCTest
@testable import LLM

final class CodebookTypeTests: XCTestCase {

    private let iq: [GGUFType] = [.iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s,
                                  .iq3_xxs, .iq3_s, .iq4_xs, .iq4_nl]
    private let kQuants: [GGUFType] = [.q2k, .q3k, .q4k, .q5k, .q6k]
    private let packed: [GGUFType] = [.q4_0, .q8_0, .q2_0, .f16, .f32, .bf16]

    func testEveryGgmlIQTypeIsCodebook() {
        for t in iq { XCTAssertTrue(Blocks.codebook(t), "\(t)") }
    }

    func testNoKQuantOrPackedTypeIsCodebook() {
        for t in kQuants + packed {
            XCTAssertFalse(Blocks.codebook(t), "\(t)")
        }
    }

    func testEveryCodebookTypeDecodesExceptTheOne32BlockOne() {
        for t in iq where t != .iq4_nl {
            XCTAssertTrue(Blocks.superBlocked(t), "\(t)")
        }
        XCTAssertFalse(Blocks.superBlocked(.iq4_nl))
    }
}
