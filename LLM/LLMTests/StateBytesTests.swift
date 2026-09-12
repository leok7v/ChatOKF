import XCTest
@testable import LLM

final class StateBytesTests: XCTestCase {

    func testReaderWalksWhatTheWriterWrote() {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, 7)
        StateBytes.putFloats(&out, [1.5, -2.25, 3.0])
        StateBytes.putInt(&out, 9)
        StateBytes.putFloats(&out, [])
        StateBytes.putFloats(&out, [4.5])
        out.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertTrue(r.header())
            XCTAssertEqual(r.int(), 7)
            XCTAssertEqual(r.span().array, [1.5, -2.25, 3.0])
            XCTAssertEqual(r.int(), 9)
            XCTAssertEqual(r.span().count, 0)
            XCTAssertEqual(r.span().array, [4.5])
        }
    }

    func testASpanReadsAtAnyByteOffset() {
        var out = Data()
        out.append(0)
        StateBytes.putFloats(&out, [6.25, 7.5])
        out.withUnsafeBytes { raw in
            let body = UnsafeRawBufferPointer(rebasing: raw[1...])
            var r = StateBytes.Reader(body)
            XCTAssertEqual(r.span().array, [6.25, 7.5])
        }
    }

    func testAFileFromAnotherBuildIsRefused() {
        var out = Data()
        out.append(contentsOf: Array("XXXX".utf8))
        StateBytes.putInt(&out, StateBytes.version)
        StateBytes.putInt(&out, 1)
        out.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertFalse(r.header())
        }
    }

    func testAnEmptyFileIsRefusedRatherThanRead() {
        Data().withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertFalse(r.header())
        }
    }

    func testATruncatedRunIsEmpty() {
        var out = Data()
        StateBytes.putFloats(&out, [1, 2, 3, 4])
        let cut = out.prefix(out.count - 5)
        cut.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertEqual(r.span().count, 0)
        }
    }
}
