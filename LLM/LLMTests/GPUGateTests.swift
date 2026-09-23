import XCTest
@testable import LLM

final class GPUGateTests: XCTestCase {

    override func tearDown() {
        GPUGate.shared.simulate(thermal: nil, faultEvery: 0)
    }

    func testTheChunkCapFollowsTheThermalState() {
        let gate = GPUGate.shared
        gate.simulate(thermal: .nominal, faultEvery: 0)
        XCTAssertEqual(gate.chunkCap(512), 512)
        XCTAssertFalse(gate.hot)
        gate.simulate(thermal: .fair, faultEvery: 0)
        XCTAssertEqual(gate.chunkCap(512), 512)
        gate.simulate(thermal: .serious, faultEvery: 0)
        XCTAssertEqual(gate.chunkCap(512), 128)
        XCTAssertEqual(gate.chunkCap(32), 32)
        XCTAssertTrue(gate.hot)
        gate.simulate(thermal: .critical, faultEvery: 0)
        XCTAssertEqual(gate.chunkCap(512), 32)
        XCTAssertTrue(gate.hot)
    }

    func testAStagedFaultFailsEveryNthVerdict() {
        let gate = GPUGate.shared
        gate.simulate(thermal: nil, faultEvery: 3)
        XCTAssertTrue(gate.verdict([], "one"))
        XCTAssertTrue(gate.verdict([], "two"))
        XCTAssertFalse(gate.verdict([], "three"))
        XCTAssertTrue(gate.verdict([], "four"))
        gate.simulate(thermal: nil, faultEvery: 0)
        XCTAssertTrue(gate.verdict([], "five"))
    }

    func testTheKnobSpellingsParse() {
        XCTAssertEqual(GPUGate.parse("serious"), .serious)
        XCTAssertEqual(GPUGate.parse("critical"), .critical)
        XCTAssertNil(GPUGate.parse(""))
        XCTAssertNil(GPUGate.parse("hot"))
        XCTAssertEqual(GPUGate.label(.serious), "serious")
    }
}
