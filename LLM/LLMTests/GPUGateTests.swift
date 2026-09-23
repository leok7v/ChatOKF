import XCTest
@testable import LLM

final class GPUGateTests: XCTestCase {

    override func tearDown() {
        GPUGate.shared.simulate(thermal: nil, faultRate: 0)
    }

    func testTheChunkCapFollowsTheThermalState() {
        let gate = GPUGate.shared
        gate.simulate(thermal: .nominal, faultRate: 0)
        XCTAssertEqual(gate.chunkCap(512), 512)
        XCTAssertFalse(gate.hot)
        gate.simulate(thermal: .fair, faultRate: 0)
        XCTAssertEqual(gate.chunkCap(512), 512)
        gate.simulate(thermal: .serious, faultRate: 0)
        XCTAssertEqual(gate.chunkCap(512), 128)
        XCTAssertEqual(gate.chunkCap(32), 32)
        XCTAssertTrue(gate.hot)
        gate.simulate(thermal: .critical, faultRate: 0)
        XCTAssertEqual(gate.chunkCap(512), 32)
        XCTAssertTrue(gate.hot)
    }

    func testTheStagedRateFailsThatFractionAndReplaysFromTheSeed() {
        let gate = GPUGate.shared
        gate.simulate(thermal: nil, faultRate: 1)
        XCTAssertFalse(gate.verdict([], "always"))
        gate.simulate(thermal: nil, faultRate: 0)
        XCTAssertTrue(gate.verdict([], "never"))
        func failures(_ seed: UInt64) -> Int {
            gate.simulate(thermal: nil, faultRate: 0.25, seed: seed)
            var out = 0
            for i in 0..<400 where !gate.verdict([], "\(i)") { out += 1 }
            return out
        }
        let first = failures(7)
        XCTAssertEqual(first, failures(7), "the seed replays the run")
        XCTAssertGreaterThan(first, 60)
        XCTAssertLessThan(first, 140)
        gate.simulate(thermal: nil, faultRate: 0)
    }

    func testTheKnobSpellingsParse() {
        XCTAssertEqual(GPUGate.parse("serious"), .serious)
        XCTAssertEqual(GPUGate.parse("critical"), .critical)
        XCTAssertNil(GPUGate.parse(""))
        XCTAssertNil(GPUGate.parse("hot"))
        XCTAssertEqual(GPUGate.label(.serious), "serious")
    }
}
