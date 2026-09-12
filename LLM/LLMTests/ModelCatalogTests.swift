import Foundation
import Testing
@testable import LLM

struct ModelCatalogTests {

    private static let name = "gemma-4-E2B"

    private func store() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-" + UUID().uuidString)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func seed(_ dest: URL, _ revision: String,
                      _ file: String) throws -> URL {
        let fm = FileManager.default
        let dir = dest.appendingPathComponent(ModelCatalogTests.name,
                                              isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let at = dir.appendingPathComponent(file)
        try Data("weights".utf8).write(to: at)
        return at
    }

    @Test func everyCatalogedFileIsGgxf() throws {
        for (_, file) in ModelCatalog.ggufFiles {
            #expect(file.hasSuffix(".ggxf"))
        }
        for (_, src) in ModelCatalog.sources {
            for file in src.files ?? [] {
                #expect(file.hasSuffix(".ggxf"))
            }
        }
    }

    @Test func staleRevisionsAreDropped() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        let old = try seed(dest, "0000000000000000000000000000000000000000",
                           "gemma-4-e2b-it-qat.gguf")
        let older = try seed(dest, "1111111111111111111111111111111111111111",
                             "gemma-4-e2b-it-qat.ggxf")
        _ = ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest)
        #expect(!fm.fileExists(atPath: old.path))
        #expect(!fm.fileExists(atPath: older.path))
        #expect(!fm.fileExists(
            atPath: old.deletingLastPathComponent().path))
        #expect(!fm.fileExists(
            atPath: older.deletingLastPathComponent().path))
    }

    @Test func thePinnedRevisionIsKept() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        let set = try #require(
            ModelCatalog.localSet(ModelCatalogTests.name, in: dest))
        let want = try #require(
            ModelCatalog.ggufFiles[ModelCatalogTests.name])
        let mine = try seed(dest, set.lastPathComponent, want)
        _ = try seed(dest, "2222222222222222222222222222222222222222",
                     "gemma-4-e2b-it-qat.ggxf")
        let path = try #require(
            ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest))
        #expect(path == mine.path)
        #expect(fm.fileExists(atPath: mine.path))
        #expect(try Data(contentsOf: mine) == Data("weights".utf8))
    }

    @Test func anEmptyStoreIsLeftAlone() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        let path = try #require(
            ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest))
        #expect(path.hasSuffix(".ggxf"))
        #expect(!fm.fileExists(atPath: path))
    }
}
