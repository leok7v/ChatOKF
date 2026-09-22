import XCTest
@testable import LLM

final class StoreReferenceTests: XCTestCase {

    static let fixture = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-data", isDirectory: true)
    static let reference = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-reference.txt")
    static let sidecar = ".okf-vectors-multilingual-e5-small.bin"

    private func embedder() throws -> BertEmbedder {
        guard let url = BertEmbedder.bundledMultilingual,
              let loaded = BertEmbedder.load(ggufPath: url.path) else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        return loaded
    }

    private func copyOfData(cold: Bool) throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "okf-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: StoreReferenceTests.fixture, to: root)
        if cold {
            try? fm.removeItem(at: root.appendingPathComponent(
                StoreReferenceTests.sidecar))
        }
        return root
    }

    private func emit(_ out: inout String, _ name: String, _ text: String) {
        out += "### " + name + "\n"
        out += text.hasSuffix("\n") ? text : text + "\n"
    }

    private func readOnlyVerbs(_ store: Store, _ e: BertEmbedder,
                               _ out: inout String) {
        emit(&out, "map", StoreText.map(store, ids: false))
        emit(&out, "search1", StoreText.search(
            store, queries: ["why does my basement smell musty in the "
                + "summer"], filter: Filter(), limit: 8))
        emit(&out, "search2", StoreText.search(
            store, queries: ["Deco X55"], filter: Filter(), limit: 8))
        emit(&out, "search3", StoreText.search(
            store, queries: ["herbs"], filter: Filter(area: "garden"),
            limit: 4))
        emit(&out, "search4", StoreText.search(
            store, queries: ["how do I stop the basement smelling",
                                "mold in the cellar"],
            filter: Filter(), limit: 3))
        emit(&out, "read", StoreText.read(
            store, store.concept("house/basement-humidity")!,
            about: "what makes it worse in July", limit: 2048, offset: 0))
        emit(&out, "links", StoreText.links(store.concept("garden/basil")!))
        emit(&out, "grep", StoreText.grep(store.grep("Deco", limit: 3)))
        emit(&out, "dream", StoreText.dream(store, store.dream(limit: 4)))
        emit(&out, "stats", StoreText.stats(store, e))
        emit(&out, "tok", StoreText.tokens(e, "Kyoto in the morning"))
        emit(&out, "sim", StoreText.similarity(
            e, "musty basement in summer", "damp cellar smell"))
    }

    private func mutatingVerbs(_ root: URL, _ e: BertEmbedder,
                               _ out: inout String) throws {
        let store = Store(root: root, embedder: e)
        store.load()
        let id = "kitchen/zz-probe"
        _ = try store.write(id: id, type: "Note", title: "Probe note",
                            description: "A probe for the update log.",
                            tags: ["probe", "test"], body: "A probe body.")
        store.load()
        emit(&out, "create", StoreText.saved(store, id: id, existed: false))
        let referrers = try store.deprecate(id: id)
        store.load()
        emit(&out, "forget", StoreText.retired(store, id: id,
                                               referrers: referrers))
        emit(&out, "hidden", StoreText.search(
            store, queries: ["Probe note"], filter: Filter(), limit: 1))
        emit(&out, "revealed", StoreText.search(
            store, queries: ["Probe note"],
            filter: Filter(deprecated: true), limit: 1))
        let dangling = try store.purge(id: id)
        store.load()
        emit(&out, "purge", StoreText.purged(store, id: id,
                                             referrers: dangling))
        let log = try String(contentsOf: root.appendingPathComponent("log.md"),
                             encoding: .utf8)
        let dated = log.components(separatedBy: "\n").map { line in
            line.hasPrefix("## ") ? "## <date>" : line
        }.joined(separator: "\n")
        emit(&out, "log", dated)
    }

    private func reproduce(cold: Bool) throws -> (text: String,
                                                  embedded: Int) {
        let e = try embedder()
        let root = try copyOfData(cold: cold)
        let store = Store(root: root, embedder: e)
        store.load()
        var out = ""
        readOnlyVerbs(store, e, &out)
        try mutatingVerbs(try copyOfData(cold: false), e, &out)
        try? FileManager.default.removeItem(at: root)
        return (out, store.embeddedCount)
    }

    private func check(_ actual: String) throws {
        let expected = try String(contentsOf: StoreReferenceTests.reference,
                                  encoding: .utf8)
        let want = expected.components(separatedBy: "\n")
        let got = actual.components(separatedBy: "\n")
        var at = 0
        while at < min(want.count, got.count) && want[at] == got[at] {
            at += 1
        }
        if at < max(want.count, got.count) {
            let before = got[max(0, at - 3)..<at].joined(separator: "\n")
            XCTFail("line \(at + 1) differs\nexpected: "
                + (at < want.count ? want[at] : "<end>")
                + "\n  actual: " + (at < got.count ? got[at] : "<end>")
                + "\ncontext:\n" + before)
        }
    }

    func testColdBuildReproducesReference() throws {
        let run = try reproduce(cold: true)
        XCTAssertEqual(run.embedded, 157)
        try check(run.text)
    }

    func testWarmSidecarReproducesReference() throws {
        let run = try reproduce(cold: false)
        XCTAssertEqual(run.embedded, 0, "the fixture's sidecar must load")
        try check(run.text)
    }
}
