import XCTest
@testable import Chat
import LLM

final class SoftFileTests: XCTestCase {

    private func temporary() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("soft-" + UUID().uuidString + ".soft")
    }

    private func turn(_ features: [Float]) -> SoftTurn {
        let image = SoftSpan(
            placeholder: 7, ids: [5, 7, 7, 7, 6], features: features,
            grid: (h: 1, w: 3), wrap: (begin: 5, end: 6))
        let audio = SoftSpan(placeholder: 9, ids: [9, 9],
                             features: [0.5, -0.5, 1.5, -1.5])
        return SoftTurn(stamp: "gemma-4-E4B.712cfb35", labelled: true,
                        parts: [.text("Picture 1: "), .image, .audio,
                                .text("what is this?")],
                        spans: [image, audio])
    }

    func testARoundTripKeepsEveryFieldAtHalfPrecision() throws {
        let url = temporary()
        let features: [Float] = [0, 1, -2, 0.333, 1024.5, -65504]
        try SoftFile.write(turn(features), to: url)
        let back = try SoftFile.read(url)
        XCTAssertEqual(back.stamp, "gemma-4-E4B.712cfb35")
        XCTAssertTrue(back.labelled)
        XCTAssertEqual(back.parts.map { p in p.noun },
                       ["text", "image", "audio", "text"])
        if case .text(let s) = back.parts[3] {
            XCTAssertEqual(s, "what is this?")
        } else {
            XCTFail("the trailing text part was lost")
        }
        XCTAssertEqual(back.spans.count, 2)
        XCTAssertEqual(back.spans[0].ids, [5, 7, 7, 7, 6])
        XCTAssertEqual(back.spans[0].grid?.h, 1)
        XCTAssertEqual(back.spans[0].grid?.w, 3)
        XCTAssertEqual(back.spans[0].wrap?.begin, 5)
        XCTAssertEqual(back.spans[0].wrap?.end, 6)
        XCTAssertNil(back.spans[1].grid)
        XCTAssertNil(back.spans[1].wrap)
        XCTAssertEqual(back.rows, 5)
        for (got, want) in zip(back.spans[0].features, features) {
            XCTAssertEqual(got, want, accuracy: abs(want) * 1e-3 + 1e-6)
        }
        XCTAssertEqual(back.spans[1].features, [0.5, -0.5, 1.5, -1.5])
        XCTAssertEqual(SoftFile.stamp(of: url), "gemma-4-E4B.712cfb35")
        try? FileManager.default.removeItem(at: url)
    }

    func testAValueOutsideHalfRangeRefusesTheWrite() {
        let url = temporary()
        XCTAssertThrowsError(try SoftFile.write(turn([0, 70000, 0, 0, 0, 0]),
                                                to: url)) { error in
            XCTAssertEqual(error as? SoftFileError, .range)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertThrowsError(try SoftFile.write(turn([0, .nan, 0, 0, 0, 0]),
                                                to: url))
    }

    func testATornFileIsRefused() throws {
        let url = temporary()
        try SoftFile.write(turn([1, 2, 3, 4, 5, 6]), to: url)
        let data = try Data(contentsOf: url)
        try data.prefix(data.count - 3).write(to: url)
        XCTAssertThrowsError(try SoftFile.read(url)) { error in
            XCTAssertEqual(error as? SoftFileError, .format)
        }
        try Data("not a soft file".utf8).write(to: url)
        XCTAssertThrowsError(try SoftFile.read(url))
        XCTAssertNil(SoftFile.stamp(of: url))
        try? FileManager.default.removeItem(at: url)
    }

}
