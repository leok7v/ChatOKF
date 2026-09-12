import XCTest
@testable import LLM

final class GrammarTests: XCTestCase {
    private func bytes(_ s: String) -> [UInt8] {
        Array(s.utf8)
    }

    func testToolCallAcceptsWellFormed() {
        let g = Grammar.toolCall()
        let ok = "<tool_call><function=get_time>"
            + "<parameter=city>NYC</parameter></function></tool_call>"
        XCTAssertTrue(g.accepts(bytes(ok)))
        let noArgs = "<tool_call><function=foo></function></tool_call>"
        XCTAssertTrue(g.accepts(bytes(noArgs)))
    }

    func testToolCallRejectsMalformed() {
        let g = Grammar.toolCall()
        XCTAssertFalse(g.accepts(bytes("<tool_call><funktion=x>")))
        XCTAssertFalse(g.accepts(bytes("<toolcall><function=x>")))
    }

    func testToolCallNamedPinsName() {
        let g = Grammar.toolCallNamed(["get_time", "search"])
        let good = "<tool_call><function=get_time>"
            + "</function></tool_call>"
        XCTAssertTrue(g.accepts(bytes(good)))
        let bad = "<tool_call><function=delete>"
            + "</function></tool_call>"
        XCTAssertFalse(g.accepts(bytes(bad)))
    }

    func testTypedToolCallEnforcesScalar() {
        let params = [GrammarParam(name: "enabled", type: .boolean)]
        let g = Grammar.toolCallTyped(params)
        let good = "<tool_call><function=set>"
            + "<parameter=enabled>true</parameter>"
            + "</function></tool_call>"
        XCTAssertTrue(g.accepts(bytes(good)))
        let bad = "<tool_call><function=set>"
            + "<parameter=enabled>maybe</parameter>"
            + "</function></tool_call>"
        XCTAssertFalse(g.accepts(bytes(bad)))
    }

    func testCanStopDeadPermissive() {
        let g = Grammar.toolCall()
        XCTAssertFalse(g.dead)
        XCTAssertFalse(g.canStop)
        XCTAssertFalse(g.permissive)
        g.advance(bytes("<tool_call><function=get_time>"
            + "<parameter=city>"))
        XCTAssertFalse(g.dead)
        XCTAssertFalse(g.canStop)
        XCTAssertTrue(g.permissive)
        g.advance(bytes("NYC</parameter></function></tool_call>"))
        XCTAssertTrue(g.canStop)
        XCTAssertFalse(g.dead)
    }

    private func maskVocab() -> GrammarVocab {
        GrammarVocab([
            bytes("<"), bytes("x"), bytes("<tool_call>"),
            bytes("abc"), bytes(" "),
        ])
    }

    func testMaskForbidsIllegalAtTightState() {
        let g = Grammar.toolCall()
        var logits = [Float](repeating: 0, count: 5)
        g.maskLogits(maskVocab(), &logits)
        XCTAssertEqual(logits[0], 0)                 // "<"
        XCTAssertEqual(logits[1], -Float.infinity)   // "x"
        XCTAssertEqual(logits[2], 0)                 // "<tool_call>"
        XCTAssertEqual(logits[3], -Float.infinity)   // "abc"
        XCTAssertEqual(logits[4], -Float.infinity)   // " "
    }

    func testMaskAllowsThroughAtFreeRun() {
        let g = Grammar.toolCall()
        g.advance(bytes("<tool_call><function=get_time>"
            + "<parameter=city>"))
        XCTAssertTrue(g.permissive)
        var logits = [Float](repeating: 0, count: 5)
        g.maskLogits(maskVocab(), &logits)
        XCTAssertEqual(logits, [Float](repeating: 0, count: 5))
    }

    func testUtf8RangeAcceptsInsideRejectsOutside() {
        let b = GrammarBuilder()
        b.utf8Range(0x00A1, 0x00FF)
        b.match()
        let g = Grammar(b.program)
        XCTAssertTrue(g.accepts(bytes("\u{00E9}")))   // e-acute, in range
        XCTAssertFalse(g.accepts(bytes("A")))          // ASCII, below range
        XCTAssertFalse(g.accepts(bytes("\u{0100}")))   // above range
    }
}
