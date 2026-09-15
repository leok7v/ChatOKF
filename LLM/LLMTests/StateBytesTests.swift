import XCTest
@testable import LLM

final class StateBytesTests: XCTestCase {

    private func raw(_ v: [Float]) -> Data {
        var out = Data()
        v.withUnsafeBytes { b in
            StateBytes.putRaw(&out, b.baseAddress!, b.count)
        }
        return out
    }

    private func floats(_ b: UnsafeRawBufferPointer) -> [Float] {
        var out = [Float](repeating: 0, count: b.count / 4)
        out.withUnsafeMutableBytes { dst in
            if b.count > 0 { _ = memcpy(dst.baseAddress!, b.baseAddress!, b.count) }
        }
        return out
    }

    func testReaderWalksWhatTheWriterWrote() {
        var out = Data()
        StateBytes.putHeader(&out)
        StateBytes.putInt(&out, 7)
        out.append(raw([1.5, -2.25, 3.0]))
        StateBytes.putInt(&out, 9)
        out.append(raw([]))
        out.append(raw([4.5]))
        out.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertTrue(r.header())
            XCTAssertEqual(r.int(), 7)
            XCTAssertEqual(floats(r.bytes()), [1.5, -2.25, 3.0])
            XCTAssertEqual(r.int(), 9)
            XCTAssertEqual(r.bytes().count, 0)
            XCTAssertEqual(floats(r.bytes()), [4.5])
        }
    }

    func testABlockCopiesIntoABufferOnlyWhole() {
        var out = Data()
        out.append(raw([6.25, 7.5]))
        var dst = [Float](repeating: 0, count: 2)
        out.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            dst.withUnsafeMutableBytes { d in
                r.bytes(into: d.baseAddress!, count: 8)
            }
        }
        XCTAssertEqual(dst, [6.25, 7.5])
        var wrong = [Float](repeating: 0, count: 3)
        out.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            wrong.withUnsafeMutableBytes { d in
                r.bytes(into: d.baseAddress!, count: 12)
            }
        }
        XCTAssertEqual(wrong, [0, 0, 0], "a block of the wrong size must "
                       + "leave the buffer alone")
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
        var old = Data()
        old.append(contentsOf: StateBytes.magic)
        StateBytes.putInt(&old, StateBytes.version - 1)
        old.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertFalse(r.header(), "an older version must be refused")
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
        out.append(raw([1, 2, 3, 4]))
        let cut = out.prefix(out.count - 5)
        cut.withUnsafeBytes { raw in
            var r = StateBytes.Reader(raw)
            XCTAssertEqual(r.bytes().count, 0)
        }
    }
}
