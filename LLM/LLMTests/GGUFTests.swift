import Foundation
import XCTest
@testable import LLM

// A GGUF that fails to parse must still give back its descriptor and its
// mapping.
final class GGUFTests: XCTestCase {

    // Descriptors are handed out lowest-free-first, so a leak is visible as a
    // rising count rather than a rising number.
    private func openDescriptors() -> Int {
        var live = 0
        for fd in 0..<Int32(min(getdtablesize(), 4096)) {
            if fcntl(fd, F_GETFD) != -1 { live += 1 }
        }
        return live
    }

    private func write(_ bytes: [UInt8], _ name: String) throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-" + name)
        try Data(bytes).write(to: url)
        return url.path
    }

    // Little-endian, which is what the reader expects and what llama.cpp
    // writes.
    private func le(_ v: UInt64, _ n: Int) -> [UInt8] {
        (0..<n).map { i in UInt8((v >> (8 * UInt64(i))) & 0xFF) }
    }

    private var magic: [UInt8] { Array("GGUF".utf8) }

    // A header whose magic is wrong. The first thing the parser checks, and
    // the throw that leaked.
    private func badMagic() throws -> String {
        try write(Array("NOPE".utf8) + le(3, 4) + le(0, 8) + le(0, 8),
                  "bad-magic.gguf")
    }

    // Magic and version are fine, one tensor, and its ggml type is a number no
    // enum case covers.
    private func unknownType() throws -> String {
        let name = Array("w".utf8)
        var bytes = magic + le(3, 4) + le(1, 8) + le(0, 8)
        bytes += le(UInt64(name.count), 8) + name
        bytes += le(1, 4)                       // one dimension
        bytes += le(8, 8)                       // of 8 elements
        bytes += le(999, 4)                     // a type that does not exist
        bytes += le(0, 8)                       // offset
        bytes += [UInt8](repeating: 0, count: 64)
        return try write(bytes, "unknown-type.gguf")
    }

    private func tensorEntry(_ name: String, elements: Int,
                             offset: Int) -> [UInt8] {
        let n = Array(name.utf8)
        return le(UInt64(n.count), 8) + n + le(1, 4)
            + le(UInt64(elements), 8) + le(0, 4) + le(UInt64(offset), 8)
    }

    private func fileOf(_ names: [String]) throws -> String {
        let each = 4096 * 4
        var bytes = magic + le(3, 4) + le(UInt64(names.count), 8) + le(0, 8)
        for (i, name) in names.enumerated() {
            bytes += tensorEntry(name, elements: 4096, offset: i * each)
        }
        bytes += [UInt8](repeating: 0, count: (32 - bytes.count % 32) % 32)
        bytes += [UInt8](repeating: 7, count: names.count * each)
        return try write(bytes, "towers.gguf")
    }

    func testOnlyWhatEveryTokenReadsIsWarmed() throws {
        let g = try GGUF(path: try fileOf([
            "token_embd.weight", "output.weight", "blk.0.attn_q.weight",
            "blk.1.nextn.eh_proj.weight", "assist.blk.0.attn_q.weight",
            "per_layer_token_embd.weight", "v.blk.0.attn_q.weight",
            "mm.input_projection.weight", "a.conv1.weight"]))
        let plain = g.readByEveryToken(drafting: false)
            .map { t in t.name }.sorted()
        XCTAssertEqual(plain, ["blk.0.attn_q.weight", "output.weight"])
        let drafting = g.readByEveryToken(drafting: true)
            .map { t in t.name }.sorted()
        XCTAssertEqual(drafting, ["assist.blk.0.attn_q.weight",
                                  "blk.0.attn_q.weight",
                                  "blk.1.nextn.eh_proj.weight",
                                  "output.weight"])
    }

    func testATiedEmbeddingIsReadByEveryToken() throws {
        let g = try GGUF(path: try fileOf(["token_embd.weight",
                                           "blk.0.attn_q.weight"]))
        XCTAssertEqual(g.readByEveryToken(drafting: false).count, 2)
    }

    func testFaultingCoversWholePagesOnceAndStaysInsideTheFile() throws {
        let g = try GGUF(path: try fileOf(["blk.0.attn_q.weight",
                                           "blk.0.attn_k.weight"]))
        let page = Int(getpagesize())
        let covered = g.fault(Array(g.tensors.values))
        XCTAssertGreaterThanOrEqual(covered, 2 * 4096 * 4)
        XCTAssertLessThanOrEqual(covered, g.mapSize + page)
        XCTAssertEqual(g.fault([]), 0)
    }

    func testBadMagicThrows() throws {
        XCTAssertThrowsError(try GGUF(path: try badMagic()))
    }

    func testUnknownTensorTypeThrows() throws {
        XCTAssertThrowsError(try GGUF(path: try unknownType()))
    }

    func testAMissingFileThrows() {
        XCTAssertThrowsError(try GGUF(path: "/nonexistent/nowhere.gguf"))
    }

    // The gate. Every failure path, many times over, and the descriptor table
    // must come back to where it started.
    func testAFailedParseKeepsNoDescriptor() throws {
        let paths = [try badMagic(), try unknownType(),
                     "/nonexistent/nowhere.gguf"]
        // One round first, so any one-time allocation a first parse makes is
        // already accounted for in the baseline.
        for path in paths { _ = try? GGUF(path: path) }
        let before = openDescriptors()
        for _ in 0..<64 {
            for path in paths { _ = try? GGUF(path: path) }
        }
        XCTAssertEqual(openDescriptors(), before,
                       "a failed parse kept a descriptor")
    }

    // A parse that SUCCEEDS still frees on the way out, which is the case
    // the old deinit covered and the one a fix could quietly break.
    func testASuccessfulParseFreesOnRelease() throws {
        let path = try write(magic + le(3, 4) + le(0, 8) + le(0, 8)
                             + [UInt8](repeating: 0, count: 64),
                             "empty.gguf")
        _ = try GGUF(path: path)
        let before = openDescriptors()
        for _ in 0..<64 { _ = try GGUF(path: path) }
        XCTAssertEqual(openDescriptors(), before,
                       "a released GGUF kept a descriptor")
    }
}
