import Foundation
import Testing
@testable import LLM

struct HubFetchTests {
    private func scratch() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("assemble-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("blob.gguf.part")
    }

    private func write(_ url: URL, _ text: String) {
        try? Data(text.utf8).write(to: url)
    }

    @Test func assembleTakesTheContiguousPrefixOnly() throws {
        let part = scratch()
        write(part, "")
        write(HubFetch.piece(part, 0), "hello ")
        write(HubFetch.piece(part, 6), "world")
        write(HubFetch.piece(part, 99), "orphan")
        let have = try HubFetch.assemble(part)
        #expect(have == 11)
        #expect(try String(decoding: Data(contentsOf: part), as: UTF8.self)
                == "hello world")
        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 0).path))
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 6).path))
        #expect(fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        HubFetch.sweep(part)
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        try? fm.removeItem(at: part.deletingLastPathComponent())
    }

    @Test func assembleNeverWritesAHole() throws {
        let part = scratch()
        write(part, "abc")
        write(HubFetch.piece(part, 99), "way past the end")
        let have = try HubFetch.assemble(part)
        #expect(have == 3)
        #expect(try Data(contentsOf: part).count == 3)
        #expect(FileManager.default.fileExists(
            atPath: HubFetch.piece(part, 99).path))
        try? FileManager.default.removeItem(
            at: part.deletingLastPathComponent())
    }
}
