import XCTest
@testable import Chat
@testable import LLM

@MainActor final class MTPTuningTests: XCTestCase {

    private func store() -> (MTPTuning, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-" + UUID().uuidString,
                                    isDirectory: true)
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return (MTPTuning(root: root), root)
    }

    private func sample(_ drafts: Int, _ tokens: Int, _ seconds: Double,
                        _ bucket: ThermalBucket = .cool,
                        gpu: Double = 0, wall: Double = 0,
                        gpuRate: Double = 0) -> MTPSample {
        MTPSample(model: "gemma-4-E4B-MTP", revision: "8b44354d1de72eff",
                  drafts: drafts, bucket: bucket, tokens: tokens,
                  seconds: seconds, accepted: 0.66, gpuSeconds: gpu,
                  wallSeconds: wall, gpuRate: gpuRate)
    }

    func testAShortTurnIsNotKept() {
        let (t, root) = store()
        XCTAssertFalse(t.fold(sample(2, 12, 1)), "12 tokens says nothing")
        XCTAssertEqual(t.entries, 0)
        XCTAssertTrue(t.fold(sample(2, 128, 4)))
        XCTAssertEqual(t.entries, 1)
        try? FileManager.default.removeItem(at: root)
    }

    func testTheEstimateIsTheBestSampleNotTheMean() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4))
        let fast = t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                          drafts: 2)
        XCTAssertEqual(fast, 32, accuracy: 0.01)
        t.fold(sample(2, 128, 16))
        let after = t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                           drafts: 2)
        XCTAssertGreaterThan(after, 30, "a slow turn must not sink the best")
        XCTAssertLessThan(after, 32, "but the best decays")
        try? FileManager.default.removeItem(at: root)
    }

    func testTheBucketAndTheCountKeepSeparateRecords() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4))
        t.fold(sample(1, 128, 8))
        t.fold(sample(2, 128, 8, .hot))
        XCTAssertEqual(t.entries, 3)
        XCTAssertEqual(t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                              drafts: 2), 32, accuracy: 0.01)
        XCTAssertEqual(t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                              drafts: 1), 16, accuracy: 0.01)
        XCTAssertEqual(t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .hot,
                              drafts: 2), 16, accuracy: 0.01)
        XCTAssertEqual(t.rate("gemma-4-E4B-MTP", "other", .cool, drafts: 2), 0,
                       "another build of the file is another record")
        try? FileManager.default.removeItem(at: root)
    }

    func testTheRecordSurvivesAReopen() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4))
        let again = MTPTuning(root: root)
        XCTAssertEqual(again.rate("gemma-4-E4B-MTP", "8b44354d1de72eff",
                                  .cool, drafts: 2), 32, accuracy: 0.01)
        again.forget()
        XCTAssertEqual(MTPTuning(root: root).entries, 0)
        try? FileManager.default.removeItem(at: root)
    }

    func testAContendedTurnIsRecordedButNotDiscounted() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4, gpu: 3.6, wall: 4))
        let alone = t.busy("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                           drafts: 2)
        XCTAssertEqual(alone, 0.9, accuracy: 0.01)
        t.fold(sample(2, 128, 4, gpu: 1, wall: 4))
        let shared = t.busy("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                            drafts: 2)
        XCTAssertLessThan(shared, alone, "a shared turn pulls the share down")
        XCTAssertEqual(t.rate("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                              drafts: 2), 32, accuracy: 0.01,
                       "stage one records the share, it does not act on it")
        try? FileManager.default.removeItem(at: root)
    }

    func testBothRatesAreKeptSideBySide() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4, gpu: 2, wall: 4, gpuRate: 64))
        let key = ("gemma-4-E4B-MTP", "8b44354d1de72eff")
        XCTAssertEqual(t.rate(key.0, key.1, .cool, drafts: 2), 32,
                       accuracy: 0.01)
        XCTAssertEqual(t.gpuRate(key.0, key.1, .cool, drafts: 2), 64,
                       accuracy: 0.01, "the GPU-normalised rate is its own "
                           + "high-water and neither one is derived")
        t.fold(sample(2, 128, 4, gpu: 3.9, wall: 4, gpuRate: 33))
        XCTAssertGreaterThan(t.gpuRate(key.0, key.1, .cool, drafts: 2), 60,
                             "a slower GPU rate does not sink the best")
        try? FileManager.default.removeItem(at: root)
    }

    func testATurnWithNoGPUAccountingReportsNoShare() {
        let (t, root) = store()
        t.fold(sample(2, 128, 4))
        XCTAssertEqual(t.busy("gemma-4-E4B-MTP", "8b44354d1de72eff", .cool,
                              drafts: 2), 0,
                       "a backend that times nothing must not read as idle")
        try? FileManager.default.removeItem(at: root)
    }

    func testNoDraftingFileIsOfferedAndTheMacStartsPlain() throws {
        XCTAssertFalse(Models.every.contains { name in name.hasSuffix("-MTP") })
        try XCTSkipIf(isOS, "the iOS default is the tier rung")
        XCTAssertFalse(Models.start.hasSuffix("-MTP"),
                       "the Mac must never start on a drafting file")
        let gb = installedGB
        let plain = gb >= 16 ? "gemma-4-12B"
            : (gb >= 8 ? "gemma-4-E4B" : Models.fallback)
        XCTAssertEqual(Models.start, plain,
                       "the macOS default is the 12B where it fits, and "
                           + "this Mac reports \(gb) GB")
    }

}
