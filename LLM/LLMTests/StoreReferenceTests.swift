import XCTest
@testable import LLM

private final class ZeroEmbedder: Embedder {
    let dim = 4
    let name = "zero"
    let queryPrefix = ""
    let passagePrefix = ""
    let relevanceFloor: Float = 0
    func embed(_ text: String) -> [Float] { [1, 0, 0, 0] }
    func states(_ text: String) -> [Float] { [1, 0, 0, 0] }
    func pooled(_ states: [Float]) -> [Float] { [1, 0, 0, 0] }
    func tokens(_ text: String) -> [Int32] { [0, 1] }
}

final class StoreReferenceTests: XCTestCase {

    func testGrepRefusesHostilePatterns() throws {
        for pattern in ["(a+)+$", "(.*)*", "((ab)*)+", "(a|aa)+", "(x*)?",
                        "(?:a+)*"] {
            XCTAssertTrue(Store.hostile(pattern), pattern)
        }
        for pattern in ["Deco", "dec\\w+", "(foo|bar)", "\\d{3}-\\d{4}",
                        "(https?://\\S+)", "(ab)*c", "[+*]+", "\\(a+\\)+",
                        "(?:ab)+", "(cat)?s"] {
            XCTAssertFalse(Store.hostile(pattern), pattern)
        }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "okf-grep-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try ("---\ntype: Note\ntitle: Mesh\n---\n\nThree Deco X55 units.\n"
             + String(repeating: "a", count: 40) + "\n")
            .write(to: root.appendingPathComponent("mesh.md"),
                   atomically: true, encoding: .utf8)
        let store = Store(root: root, embedder: ZeroEmbedder())
        store.load()
        XCTAssertEqual(store.grep("Deco", limit: 3).map { hit in hit.text },
                       ["Three Deco X55 units."])
        XCTAssertEqual(store.grep("(a+)+$", limit: 3).count, 0)
        try? fm.removeItem(at: root)
    }


    static let fixture = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-data", isDirectory: true)
    static let reference = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-reference.txt")
    static let sidecar = ".okf-vectors-multilingual-e5-small.bin"

    private func embedder() throws -> BertEmbedder {
        guard !TestWeights.skipped else {
            throw XCTSkip("CHATOKF_SKIP_WEIGHTS: embedder ladders skipped")
        }
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

    private func timed(_ store: Store, _ queries: [String])
        -> (hits: [(String, Float)], seconds: Double) {
        let began = Date()
        let result = store.search(queries, filter: Filter(), limit: 8)
        return (result.hits.map { hit in (hit.concept.id, hit.relevance) },
                Date().timeIntervalSince(began))
    }

    func testASecondSearchEncodesNoConceptAgain() throws {
        let e = try embedder()
        let root = try copyOfData(cold: false)
        let store = Store(root: root, embedder: e)
        store.load()
        let queries = ["why does my basement smell musty in the summer",
                       "mold in the cellar"]
        let cold = timed(store, queries)
        let encodedOnce = store.encodedCount
        let warm = timed(store, queries)
        XCTAssertEqual(encodedOnce, 8)
        XCTAssertEqual(store.encodedCount, encodedOnce,
                       "a second search re-encodes nothing")
        XCTAssertEqual(warm.hits.map { hit in hit.0 },
                       cold.hits.map { hit in hit.0 })
        XCTAssertEqual(warm.hits.map { hit in hit.1 },
                       cold.hits.map { hit in hit.1 })
        let again = store.search(["Deco X55"], filter: Filter(), limit: 8)
        XCTAssertEqual(again.hits.count, 8)
        XCTAssertLessThanOrEqual(store.encodedCount, encodedOnce + 8)
        print(String(format: "[store] search of 8: cold %.2fs, warm %.2fs, "
                         + "%d concepts encoded, %d KB held",
                     cold.seconds, warm.seconds, store.encodedCount,
                     store.encodedBytes >> 10))
        try? FileManager.default.removeItem(at: root)
    }

    private func assertSameState(_ live: Store, _ full: Store,
                                 _ queries: [String], line: UInt = #line) {
        XCTAssertEqual(live.concepts.map { c in c.id },
                       full.concepts.map { c in c.id }, line: line)
        XCTAssertEqual(live.byId, full.byId, line: line)
        for (a, b) in zip(live.concepts, full.concepts) {
            XCTAssertEqual(a.hash, b.hash, a.id, line: line)
            XCTAssertEqual(a.links, b.links, a.id, line: line)
            XCTAssertEqual(a.backlinks, b.backlinks, a.id, line: line)
            XCTAssertEqual(a.vector, b.vector, a.id, line: line)
            XCTAssertEqual(a.bodyVector, b.bodyVector, a.id, line: line)
        }
        for query in queries {
            let mine = live.literalRanking(query).map { e in
                "\(e.index) \(e.weight) \(e.terms)"
            }
            let theirs = full.literalRanking(query).map { e in
                "\(e.index) \(e.weight) \(e.terms)"
            }
            XCTAssertEqual(mine, theirs, query, line: line)
            XCTAssertEqual(
                StoreText.search(live, queries: [query], filter: Filter(),
                                 limit: 8),
                StoreText.search(full, queries: [query], filter: Filter(),
                                 limit: 8), query, line: line)
        }
    }

    private func fullyLoaded(_ root: URL, _ e: BertEmbedder)
        -> (store: Store, seconds: Double) {
        let began = Date()
        let store = Store(root: root, embedder: e)
        store.load()
        return (store, Date().timeIntervalSince(began))
    }

    func testReloadOfOneFileMatchesAFullLoad() throws {
        let e = try embedder()
        let root = try copyOfData(cold: false)
        let live = Store(root: root, embedder: e)
        live.load()
        let probe = "garden/aa-probe"
        let queries = ["basil compost probe", "Probe note",
                       "why does my basement smell musty in the summer"]
        _ = try live.write(
            id: probe, type: "Note", title: "Probe note",
            description: "A probe that sorts first in its area.",
            tags: ["probe"],
            body: "A probe body about [basil](/garden/basil.md) and compost.")
        let began = Date()
        live.reload(id: probe)
        let reloadSeconds = Date().timeIntervalSince(began)
        XCTAssertEqual(live.embeddedCount, 1)
        XCTAssertEqual(live.concept(probe)?.links, ["garden/basil"])
        let again = Date()
        live.reload(id: probe)
        let unchangedSeconds = Date().timeIntervalSince(again)
        XCTAssertEqual(live.embeddedCount, 0)
        let created = fullyLoaded(root, e)
        assertSameState(live, created.store, queries)
        _ = try live.deprecate(id: probe)
        live.reload(id: probe)
        XCTAssertEqual(live.embeddedCount, 1)
        assertSameState(live, fullyLoaded(root, e).store, queries)
        _ = try live.purge(id: probe)
        live.reload(id: probe)
        XCTAssertEqual(live.embeddedCount, 0)
        XCTAssertNil(live.concept(probe))
        assertSameState(live, fullyLoaded(root, e).store, queries)
        print(String(format: "[store] one note: reload %.3fs embedding, "
                         + "%.3fs unchanged, full load %.3fs over %d concepts",
                     reloadSeconds, unchangedSeconds, created.seconds,
                     live.concepts.count))
        try? FileManager.default.removeItem(at: root)
    }

    func testPrefixTokensLeadTheWholeSequence() throws {
        let e = try embedder()
        let root = try copyOfData(cold: false)
        let store = Store(root: root, embedder: e)
        store.load()
        var texts = ["why does my basement smell musty in the summer",
                     "Deco X55", "herbs", "mold in the cellar"]
            .map { text in (e.queryPrefix, text) }
        for concept in store.concepts {
            texts.append((e.passagePrefix, concept.passage))
            texts.append((e.passagePrefix,
                          String(concept.body.prefix(Store.relevanceChars))))
        }
        var drifted: [String] = []
        for (prefix, text) in texts {
            let lead = Array(e.tokens(prefix).dropLast())
            let whole = e.tokens(prefix + text)
            if Array(whole.prefix(lead.count)) != lead {
                drifted.append(prefix + text.prefix(40))
            }
        }
        print("[store] prefix drift in \(drifted.count) of \(texts.count): "
              + drifted.prefix(5).joined(separator: " | "))
        XCTAssertEqual(drifted.count, 0)
        try? FileManager.default.removeItem(at: root)
    }

    func testASecondReadEmbedsNoParagraphAgain() throws {
        let e = try embedder()
        let root = try copyOfData(cold: false)
        let store = Store(root: root, embedder: e)
        store.load()
        let concept = try XCTUnwrap(store.concept("house/basement-humidity"))
        let about = "what makes it worse in July"
        let began = Date()
        let first = store.window(concept, about: about, limit: 400)
        let cold = Date().timeIntervalSince(began)
        let again = Date()
        let second = store.window(concept, about: about, limit: 400)
        let warm = Date().timeIntervalSince(again)
        XCTAssertEqual(store.windowedCount, 1)
        XCTAssertEqual(first.from, second.from)
        XCTAssertEqual(first.to, second.to)
        _ = store.window(concept, about: "dehumidifier", limit: 400)
        XCTAssertEqual(store.windowedCount, 1,
                       "a second question re-embeds no paragraph")
        print(String(format: "[store] window: cold %.3fs, warm %.3fs",
                     cold, warm))
        try? FileManager.default.removeItem(at: root)
    }
}
