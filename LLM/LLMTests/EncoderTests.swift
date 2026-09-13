import XCTest
@testable import LLM

final class EncoderTests: XCTestCase {

    private func e5Path() throws -> String {
        guard let url = BertEmbedder.bundledMultilingual else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        return url.path
    }

    private func e5() throws -> MiniLM {
        MiniLM(gguf: try GGUF(path: try e5Path()))
    }

    private func slugs() throws -> WikiSlugs {
        guard let url = WikiSlugs.bundledModel,
              let w = WikiSlugs(ggufPath: url.path) else {
            throw XCTSkip("no bundled minilm.gguf index")
        }
        return w
    }

    func testUnigramMatchesReference() throws {
        let model = try e5()
        XCTAssertTrue(model.multilingual)
        XCTAssertEqual(model.tokenize("Kyoto in the morning"),
                       [0, 217267, 23, 70, 42141, 2])
    }

    func testSimilarityMatchesReference() throws {
        let model = try e5()
        let left = model.embed("query: musty basement in summer")
        let right = model.embed("passage: damp cellar smell")
        var total: Float = 0
        for i in 0 ..< min(left.count, right.count) {
            total += left[i] * right[i]
        }
        XCTAssertEqual(String(format: "%.4f", total), "0.7765")
    }

    func testWordPieceMatchesSlugsOracle() throws {
        let w = try slugs()
        XCTAssertFalse(w.embedder.multilingual)
        let hits = w.query("What is dark matter?", topK: 2)
        XCTAssertEqual(hits.map { h in h.id }, ["34685", "963995"])
        XCTAssertEqual(hits.map { h in h.distance }, [75, 79])
    }

    private struct Resident {
        let footprint: Int
        let dirty: Int
        let fileBacked: Int

        static func sample() -> Resident {
            var info = task_vm_info_data_t()
            let words = MemoryLayout<task_vm_info_data_t>.size
                / MemoryLayout<natural_t>.size
            var count = mach_msg_type_number_t(words)
            _ = withUnsafeMutablePointer(to: &info) { p in
                p.withMemoryRebound(to: integer_t.self, capacity: words) {
                    q in task_info(mach_task_self_,
                                   task_flavor_t(TASK_VM_INFO), q, &count)
                }
            }
            return Resident(footprint: Int(info.phys_footprint),
                            dirty: Int(info.internal) + Int(info.compressed),
                            fileBacked: Int(info.external))
        }

        func minus(_ base: Resident) -> String {
            func mb(_ v: Int) -> String {
                String(format: "%+.1f MB", Double(v) / 1_048_576)
            }
            return "footprint \(mb(footprint - base.footprint)) dirty "
                + "\(mb(dirty - base.dirty)) file-backed "
                + "\(mb(fileBacked - base.fileBacked))"
        }
    }

    func testResidentCostOfE5() throws {
        let path = try e5Path()
        let before = Resident.sample()
        let gguf = try GGUF(path: path)
        let parsed = Resident.sample()
        let model = MiniLM(gguf: gguf)
        let loaded = Resident.sample()
        let vector = model.embed("passage: " + String(
            repeating: "The basement smells musty every July. ", count: 30))
        let embedded = Resident.sample()
        XCTAssertEqual(vector.count, 384)
        print("[e5] gguf parsed:  " + parsed.minus(before))
        print("[e5] model loaded: " + loaded.minus(parsed))
        print("[e5] one embed:    " + embedded.minus(loaded))
        print("[e5] total:        " + embedded.minus(before))
    }
}
