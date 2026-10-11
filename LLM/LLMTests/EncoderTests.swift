import XCTest
@testable import LLM

final class EncoderTests: XCTestCase {

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        var total: Float = 0
        for i in 0 ..< min(a.count, b.count) { total += a[i] * b[i] }
        return total
    }

    private func fnv(_ x: [Float]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in x {
            h = (h ^ UInt64(v.bitPattern)) &* 0x0000_0100_0000_01b3
        }
        return h
    }

    private func fnv(_ ids: [Int32]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for id in ids {
            h = (h ^ UInt64(UInt32(bitPattern: id))) &* 0x0000_0100_0000_01b3
        }
        return h
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

    static let a25mRepo = "leok7v/bekko-embedding-v1-a25m"
    static let a25mFile = "bekko-embedding-v1-a25m-Q8_0.gguf"
    static let halfRepo = "hotchpotch/bekko-embedding-v1-a25m-GGUF"
    static let a25mHalf = "bekko-embedding-v1-a25m-F16.gguf"

    static func cloned(_ repo: String, _ file: String) throws -> String {
        guard let path = TestWeights.clone(repo, file) else {
            throw XCTSkip("no \(file): clone \(repo) under ~/huggingface.co "
                          + "or ~/Models")
        }
        return path
    }

    private func a25mPath() throws -> String {
        try EncoderTests.cloned(EncoderTests.a25mRepo, EncoderTests.a25mFile)
    }

    private func halfPath() throws -> String {
        try EncoderTests.cloned(EncoderTests.halfRepo, EncoderTests.a25mHalf)
    }

    private func a25m() throws -> ModernBERT {
        let gguf = try GGUF(path: try a25mPath())
        XCTAssertTrue(ModernBERT.isModernBert(gguf))
        return ModernBERT(gguf: gguf)
    }

    private struct A25mReference {
        let tokens: [String: [Int32]]
        let vectors: [String: [Float]]
        let longIds: [Int32]
        let longVector: [Float]
        let corpus: [String: (count: Int, hash: String)]

        static func load() throws -> A25mReference {
            let url = URL(fileURLWithPath: TestWeights.repoRoot)
                .appendingPathComponent(
                    "LLM/LLMTests/fixtures/a25m-reference.json")
            let root = try JSONSerialization.jsonObject(
                with: Data(contentsOf: url)) as! [String: Any]
            let long = root["long512"] as! [String: Any]
            var corpus: [String: (count: Int, hash: String)] = [:]
            for (file, entry) in root["corpus"] as! [String: [Any]] {
                corpus[file] = (entry[0] as! Int, entry[1] as! String)
            }
            return A25mReference(
                tokens: (root["tokens"] as! [String: [Int]]).mapValues {
                    ids in ids.map(Int32.init)
                },
                vectors: (root["vectors"] as! [String: [Double]]).mapValues {
                    floats in floats.map(Float.init)
                },
                longIds: (long["ids"] as! [Int]).map(Int32.init),
                longVector: (long["vector"] as! [Double]).map(Float.init),
                corpus: corpus)
        }
    }

    func testA25mTokensMatchTheHubTokenizer() throws {
        let model = try a25m()
        let reference = try A25mReference.load()
        for (text, ids) in reference.tokens.sorted(by: { a, b in
            a.key < b.key
        }) {
            XCTAssertEqual(model.tokenize(text), ids, text.debugDescription)
        }
    }

    func testA25mTokenizesTheSampleCorpusLikeTheHub() throws {
        let model = try a25m()
        let reference = try A25mReference.load()
        let root = StoreReferenceTests.fixture
        var checked = 0
        var drifted: [String] = []
        for (file, expected) in reference.corpus {
            let data = try Data(contentsOf: root.appendingPathComponent(file))
            let ids = model.tokenize(String(decoding: data, as: UTF8.self))
            let hash = String(format: "%016llx", fnv(ids))
            if ids.count != expected.count || hash != expected.hash {
                drifted.append("\(file) \(ids.count)/\(expected.count)")
            }
            checked += 1
        }
        XCTAssertEqual(checked, reference.corpus.count)
        XCTAssertEqual(drifted, [])
    }

    private func closeness(_ ours: [Float], _ theirs: [Float])
        -> (cosine: Float, worst: Float) {
        var worst: Float = 0
        for i in 0 ..< min(ours.count, theirs.count) {
            worst = max(worst, abs(ours[i] - theirs[i]))
        }
        return (dot(ours, theirs), worst)
    }

    func testA25mVectorsMatchThePythonModel() throws {
        let reference = try A25mReference.load()
        let files = [(EncoderTests.a25mRepo, EncoderTests.a25mFile),
                     (EncoderTests.halfRepo, EncoderTests.a25mHalf)]
        for (repo, file) in files {
            if let path = TestWeights.clone(repo, file) {
                let model = ModernBERT(gguf: try GGUF(path: path))
                var lowest: Float = 1
                var worst: Float = 0
                for (text, theirs) in reference.vectors {
                    let near = closeness(model.embed(text), theirs)
                    lowest = min(lowest, near.cosine)
                    worst = max(worst, near.worst)
                    XCTAssertGreaterThan(near.cosine, 0.999,
                                         text.debugDescription)
                }
                print(String(format: "[a25m] %@: %d texts, lowest cosine "
                                 + "%.6f, worst |d| %.2e against the f32 "
                                 + "Python model", file,
                             reference.vectors.count, lowest, worst))
            }
        }
        _ = try a25m()
    }

    func testA25mSimilaritiesFollowThePythonModel() throws {
        let model = try a25m()
        let reference = try A25mReference.load()
        let pairs = [("musty basement in summer", "damp cellar smell"),
                     ("The basement smells musty every July.",
                      "Подвал летом пахнет плесенью."),
                     ("The basement smells musty every July.",
                      "夏の地下室はかび臭い。"),
                     ("The basement smells musty every July.",
                      "Kyoto in the morning")]
        for (left, right) in pairs {
            let ours = dot(model.embed(left), model.embed(right))
            let theirs = dot(reference.vectors[left]!,
                             reference.vectors[right]!)
            XCTAssertEqual(ours, theirs, accuracy: 0.005, left + " / " + right)
            print(String(format: "[a25m] sim %.4f (python %.4f)  %@ / %@",
                         ours, theirs, left, right))
        }
    }

    func testA25mLongPassageMatchesAndTimes() throws {
        let model = try a25m()
        let reference = try A25mReference.load()
        let text = String(repeating: "The basement smells musty every July. ",
                          count: 80)
        let whole = model.tokenize(text)
        XCTAssertEqual(Array(whole.prefix(511)) + [model.eosId],
                       reference.longIds)
        let ids = reference.longIds
        _ = model.encode(ids: ids)
        let began = Date()
        let states = model.encode(ids: ids)
        let seconds = Date().timeIntervalSince(began)
        var pooled = [Float](repeating: 0, count: model.dim)
        EncoderKernels.pooled(states, width: model.dim, into: &pooled)
        let near = closeness(pooled, reference.longVector)
        XCTAssertGreaterThan(near.cosine, 0.999)
        print(String(format: "[a25m] %d tokens encoded in %.3f s, states "
                         + "%016llx, cosine %.6f against python",
                     ids.count, seconds, fnv(states), near.cosine))
    }

    func testResidentCostOfA25m() throws {
        let path = try a25mPath()
        let before = Resident.sample()
        let gguf = try GGUF(path: path)
        let parsed = Resident.sample()
        let model = ModernBERT(gguf: gguf)
        let loaded = Resident.sample()
        let vector = model.embed(String(
            repeating: "The basement smells musty every July. ", count: 30))
        let embedded = Resident.sample()
        XCTAssertEqual(vector.count, 384)
        print("[a25m] gguf parsed:  " + parsed.minus(before))
        print("[a25m] model loaded: " + loaded.minus(parsed))
        print("[a25m] one embed:    " + embedded.minus(loaded))
        print("[a25m] total:        " + embedded.minus(before))
    }

    func testA25mQueryTime() throws {
        let bert = try a25m()
        let query = "why does my basement smell musty in the summer"
        _ = bert.embed(query)
        var seconds = 0.0
        for _ in 0..<5 {
            let began = Date()
            _ = bert.embed(query)
            seconds += Date().timeIntervalSince(began)
        }
        print(String(format: "[a25m] query embed %.1f ms over %d tokens",
                     seconds * 200, bert.tokenize(query).count))
    }

    private func candidates() -> [String] {
        var out: [String] = []
        for root in TestWeights.clones {
            let dir = root + "/" + EncoderTests.a25mRepo
            let names = (try? FileManager.default.contentsOfDirectory(
                atPath: dir)) ?? []
            out += names.filter { name in name.hasSuffix(".gguf") }
                .sorted().map { name in dir + "/" + name }
        }
        return out
    }

    private func corpusVectors(_ embedder: Embedder) throws
        -> (vectors: [[Float]], ids: [String], seconds: Double) {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "okf-cand-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: StoreReferenceTests.fixture, to: root)
        try? fm.removeItem(at: root.appendingPathComponent(
            StoreReferenceTests.sidecar))
        let began = Date()
        let store = Store(root: root, embedder: embedder)
        store.load()
        let seconds = Date().timeIntervalSince(began)
        try? fm.removeItem(at: root)
        return (store.concepts.map { concept in concept.vector },
                store.concepts.map { concept in concept.id }, seconds)
    }

    private func nearest(_ vectors: [[Float]], _ i: Int) -> Int {
        var best = -1
        var score: Float = -2
        for j in vectors.indices where j != i {
            let d = dot(vectors[i], vectors[j])
            if d > score {
                score = d
                best = j
            }
        }
        return best
    }

    private func agreement(_ ours: [[Float]], _ theirs: [[Float]])
        -> (mean: Float, lowest: Float, sameNeighbour: Int) {
        var total: Float = 0
        var lowest: Float = 1
        var same = 0
        for i in ours.indices {
            let cosine = dot(ours[i], theirs[i])
            total += cosine
            lowest = min(lowest, cosine)
            if nearest(ours, i) == nearest(theirs, i) { same += 1 }
        }
        return (total / Float(max(1, ours.count)), lowest, same)
    }

    func testA25mQuantCandidatesAgainstTheHalfPrecisionFile() throws {
        let files = candidates()
        if files.isEmpty { throw XCTSkip("no candidate GGUFs under "
                                         + EncoderTests.a25mRepo) }
        let reference = try A25mReference.load()
        let halfPath = try halfPath()
        guard let half = ModernBertEmbedder.load(ggufPath: halfPath) else {
            throw XCTSkip("no F16 a25m")
        }
        let truth = try corpusVectors(half)
        let loadable = (files + [halfPath]).compactMap { path in
            ModernBertEmbedder.load(ggufPath: path).map { e in (path, e) }
        }
        print("[quant] file                                    MB   python"
              + "   long   corpus mean  lowest  same-nn  cold s")
        for (path, embedder) in loadable {
            let model = ModernBERT(gguf: try GGUF(path: path))
            var lowest: Float = 1
            for (text, theirs) in reference.vectors {
                lowest = min(lowest, dot(embedder.embed(text), theirs))
            }
            var pooled = [Float](repeating: 0, count: model.dim)
            EncoderKernels.pooled(model.encode(ids: reference.longIds),
                                  width: model.dim, into: &pooled)
            let long = dot(pooled, reference.longVector)
            let mine = try corpusVectors(embedder)
            let near = agreement(mine.vectors, truth.vectors)
            let bytes = (try? FileManager.default.attributesOfItem(
                atPath: path)[.size] as? Int) ?? 0
            print(String(format: "[quant] %-38@ %5.1f  %.5f  %.5f  %.5f  "
                             + "%.5f  %3d/%d  %.1f",
                         (path as NSString).lastPathComponent,
                         Double(bytes) / 1e6, lowest, long, near.mean,
                         near.lowest, near.sameNeighbour, mine.ids.count,
                         mine.seconds))
        }
    }

    func testA25mSearchesTheSampleCorpus() throws {
        guard let bert = ModernBertEmbedder.load(ggufPath: try a25mPath())
        else { throw XCTSkip("no a25m") }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "okf-a25m-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: StoreReferenceTests.fixture, to: root)
        let expected = [
            ("why does my basement smell musty in the summer",
             "house/basement-humidity"),
            ("Deco X55", "tech/wifi-mesh"),
            ("herbs", ""),
            ("mold in the cellar", ""),
            ("how do I keep basil alive over the winter",
             "garden/basil-window-box"),
            ("what did the radon test say", "house/radon-test"),
        ]
        let began = Date()
        let store = Store(root: root, embedder: bert)
        store.load()
        print(String(format: "[a25m] %d concepts loaded in %.1f s, "
                         + "%d embedded", store.concepts.count,
                     Date().timeIntervalSince(began), store.embeddedCount))
        for (query, top) in expected {
            let hits = store.search([query], filter: Filter(), limit: 3).hits
            print("[a25m]   " + query.padding(toLength: 48, withPad: " ",
                                              startingAt: 0)
                + hits.map { hit in
                    String(format: "%@ %.3f", hit.concept.id, hit.relevance)
                }.joined(separator: "  "))
            if !top.isEmpty {
                XCTAssertEqual(hits.first?.concept.id, top, query)
            }
        }
        try? fm.removeItem(at: root)
    }
}
