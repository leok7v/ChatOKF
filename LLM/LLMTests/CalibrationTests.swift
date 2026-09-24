import XCTest
@testable import LLM

final class CalibrationTests: XCTestCase {

    static let sample = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("LLM/LLMTests/okf-data", isDirectory: true)
    static let bundle = URL(fileURLWithPath: TestWeights.repoRoot)
        .appendingPathComponent("okf", isDirectory: true)

    static let offTopic = [
        "what is the airspeed of an unladen swallow",
        "how do interest rate swaps work",
        "quantum entanglement explained simply",
        "best sushi restaurant in Tokyo",
        "zzxqy qwwrgh vvbnk",
        "translate good morning into French",
        "who won the football world cup in 2018",
        "how do I compile Swift on Linux",
    ]

    static let sampleQuestions = [
        "why does my basement smell musty in the summer",
        "how do I keep basil alive over the winter",
        "which wifi mesh units do we have",
        "when is the last frost here",
        "what streaming subscriptions are we paying for",
        "the back door sticks, what do I do",
        "how old is the water heater",
        "what did the radon test say",
    ]

    static let bundleQuestions = [
        "why must a turn never re-prefill the conversation",
        "how are the tokens per file measured after a compaction",
        "what does the precook stamp hash",
        "why did loading the e5 tokenizer cost forty megabytes",
        "how does the Wikipedia sign-bit index match a query",
        "why is the iOS device floor the deployment target",
        "how is search confidence measured",
        "what does the comment lint fail on",
    ]

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

    private func copy(_ from: URL, keeping ids: Set<String>? = nil) throws
        -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(
            "calib-" + UUID().uuidString, isDirectory: true)
        try fm.copyItem(at: from, to: root)
        if let ids {
            let base = root.resolvingSymlinksInPath().path
            let walker = fm.enumerator(at: root,
                                       includingPropertiesForKeys: nil)!
            for case let url as URL in walker where url.pathExtension == "md" {
                let path = url.resolvingSymlinksInPath().path
                let id = String(path.dropFirst(base.count + 1).dropLast(3))
                if !ids.contains(id), id != "index", id != "log" {
                    try? fm.removeItem(at: url)
                }
            }
        }
        return root
    }

    private func row(_ store: Store, _ e: BertEmbedder,
                     _ question: String) -> String {
        let result = store.search([question], filter: Filter(), limit: 1)
        let top = result.hits.first
        return String(format: "%5.2f  %.3f %.3f  %-36@ %@%@", result.standout,
                      top?.score ?? 0, top?.relevance ?? 0,
                      String(question.prefix(36)), top?.concept.id ?? "-",
                      top?.terms.isEmpty == false
                          ? " [" + top!.terms.joined(separator: " ") + "]"
                          : "")
    }

    private func table(_ name: String, _ root: URL, _ e: BertEmbedder,
                       _ questions: [String]) {
        let store = Store(root: root, embedder: e)
        store.load()
        print("[calib] \(name): \(store.concepts.count) concepts, "
              + "re-embedded \(store.embeddedCount), floor "
              + "\(e.relevanceFloor)")
        print("[calib] standout cosine relev  question                      "
              + "       top hit")
        for question in questions {
            print("[calib] " + row(store, e, question))
        }
        print("[calib] -- off topic")
        for question in CalibrationTests.offTopic {
            print("[calib] " + row(store, e, question))
        }
    }

    func testStandoutOnBothCorpora() throws {
        let e = try embedder()
        guard FileManager.default.fileExists(
            atPath: CalibrationTests.bundle.path) else {
            throw XCTSkip("no okf/ bundle beside this checkout")
        }
        let sample = try copy(CalibrationTests.sample)
        table("sample data/", sample, e, CalibrationTests.sampleQuestions)
        let bundle = try copy(CalibrationTests.bundle)
        table("this repo's okf/", bundle, e, CalibrationTests.bundleQuestions)
        try? FileManager.default.removeItem(at: sample)
        try? FileManager.default.removeItem(at: bundle)
    }

    static let restated: [(id: String, title: String, description: String)] = [
        ("house/basement-humidity", "Damp cellar",
         "The cellar sits at about sixty-five percent humidity in summer."),
        ("garden/basil", "User's herbs",
         "User grows basil and it dies when it gets cold or flowers."),
        ("tech/wifi-mesh", "Home network",
         "Three Deco mesh units, one per floor, cabled to the router."),
        ("person/leo", "User's interests",
         "User enjoys ice hockey, coaching kids, and swimming."),
    ]

    func testDuplicateThreshold() throws {
        let e = try embedder()
        let root = try copy(CalibrationTests.sample)
        let store = Store(root: root, embedder: e)
        store.load()
        _ = try store.write(id: "person/leo", type: "Note", title: "Hobbies",
                            description: "Leo enjoys ice hockey, coaching "
                                + "kids, and swimming.", tags: [],
                            body: "Leo enjoys ice hockey, coaching children, "
                                + "and swimming.")
        store.load()
        var tops: [Float] = []
        print("[calib] existing concepts a rule would call duplicates:")
        for concept in store.concepts {
            let near = store.nearest(title: concept.title,
                                     description: concept.description,
                                     tags: concept.tags, type: concept.type,
                                     limit: 2)
            let words = store.literalRanking(concept.title + " "
                                             + concept.description)
                .filter { entry in
                    store.concepts[entry.index].id != concept.id
                }.first
            let other = words.map { entry in store.concepts[entry.index].id }
            if near.count == 2 {
                tops.append(near[1].score)
                if near[1].score >= 0.88 {
                    print(String(format: "[calib]   %.3f %@ -> %@, words %d %@",
                                 near[1].score, concept.id, near[1].id,
                                 words?.weight ?? 0, other ?? "-"))
                }
            }
        }
        tops.sort()
        print(String(format: "[calib] nearest other concept over %d: min "
                         + "%.3f median %.3f p90 %.3f max %.3f, %d at or "
                         + "over %.2f",
                     tops.count, tops.first ?? 0, tops[tops.count / 2],
                     tops[tops.count * 9 / 10], tops.last ?? 0,
                     tops.filter { top in top >= Store.mergeAt }.count,
                     Store.mergeAt))
        print("[calib] restated notes against the store:")
        for note in CalibrationTests.restated {
            let near = store.nearest(title: note.title,
                                     description: note.description, tags: [],
                                     type: "Note", limit: 2)
            let words = store.literalRanking(note.title + " "
                                             + note.description).first
            print(String(format: "[calib]   %.3f %@ (then %.3f %@) words %d "
                             + "%@ for %@",
                         near[0].score, near[0].id, near[1].score,
                         near[1].id, words?.weight ?? 0,
                         words.map { entry in
                             store.concepts[entry.index].id
                         } ?? "-", note.id))
        }
        try? FileManager.default.removeItem(at: root)
    }

    static let planted: [(id: String, title: String, description: String,
                          body: String)] = [
        ("person/pet-name", "The user's pet name",
         "The user's pet name is the name of their pet.",
         "The user calls their dog Biscuit."),
        ("car/subaru", "The car", "A 2019 Subaru Outback, green, 60k miles.",
         "Serviced at the dealer every 10k."),
        ("kitchen/pancakes", "Sunday pancakes",
         "Buttermilk pancakes with a pinch of cardamom every Sunday.",
         "Two eggs, a cup of buttermilk, rest the batter ten minutes."),
        ("tech/wifi-mesh", "The mesh wifi setup",
         "Three TP-Link Deco X55 units wired back to the router.",
         "One per floor."),
    ]

    static let leakQuestions = [
        "Which block gives the most fruit per tree, and what makes that "
            + "surprising?",
        "explain dark matter and dark energy",
        "what is 17 times 23",
        "what is my dog called",
        "which car do I drive",
        "how do I make the Sunday pancakes",
        "Deco X55",
        "what is the name of the tallest mountain",
        "does a green apple have more sugar than a red one",
        "who was the user of the first telephone",
    ]

    private func leakRow(_ store: Store, _ question: String) -> String {
        let result = store.search([question], filter: Filter(), limit: 2)
        let top = result.hits.first
        let second = result.hits.dropFirst().first
        return String(format: "%5.2f  %.3f %.3f  %.3f %.3f  %-40@ %@%@",
                      result.standout, top?.score ?? 0, second?.score ?? 0,
                      top?.relevance ?? 0, second?.relevance ?? 0,
                      String(question.prefix(40)), top?.concept.id ?? "-",
                      top?.terms.isEmpty == false
                          ? " [" + top!.terms.joined(separator: " ") + "]"
                          : "")
    }

    func testPlantedNotesAgainstForeignQuestions() throws {
        let e = try embedder()
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("leak-" + UUID().uuidString,
                                    isDirectory: true)
        let store = Store(root: root, embedder: e)
        for note in CalibrationTests.planted {
            _ = try store.write(id: note.id, type: "Note", title: note.title,
                                description: note.description, tags: [],
                                body: note.body)
        }
        store.load()
        print("[leak] \(store.concepts.count) planted notes, floor "
              + "\(e.relevanceFloor)")
        print("[leak] standout cosine top second relev top second question"
              + "                    top hit")
        for question in CalibrationTests.leakQuestions {
            print("[leak] " + leakRow(store, question))
        }
        try? FileManager.default.removeItem(at: root)
    }

    private struct Band {
        var standouts: [Float] = []
        var scores: [Float] = []

        mutating func add(_ result: SearchResult) {
            standouts.append(result.standout)
            scores.append(result.hits.first?.score ?? 0)
        }

        var line: String {
            String(format: "standout %5.2f %5.2f %5.2f  cosine %.3f %.3f %.3f",
                   standouts.reduce(0, +) / Float(max(1, standouts.count)),
                   standouts.min() ?? 0, standouts.max() ?? 0,
                   scores.reduce(0, +) / Float(max(1, scores.count)),
                   scores.min() ?? 0, scores.max() ?? 0)
        }
    }

    func testOffTopicLadder() throws {
        let e = try embedder()
        let full = Store(root: try copy(CalibrationTests.sample), embedder: e)
        full.load()
        let answers = CalibrationTests.sampleQuestions.map { question in
            full.search([question], filter: Filter(), limit: 1)
                .hits.first?.concept.id ?? ""
        }
        let others = full.concepts.map { concept in concept.id }
            .filter { id in !answers.contains(id) }
        print("[calib] ladder: on topic against off topic, mean min max")
        for n in [3, 5, 8, 10, 15, 20, 40, 80, 157] {
            let kept = Set(answers.prefix(min(n, answers.count))
                           + others.prefix(max(0, n - answers.count)))
            let root = try copy(CalibrationTests.sample, keeping: kept)
            let store = Store(root: root, embedder: e)
            store.load()
            var on = Band()
            var off = Band()
            for (i, question) in CalibrationTests.sampleQuestions.enumerated()
            where kept.contains(answers[i]) {
                on.add(store.search([question], filter: Filter(), limit: 1))
            }
            for question in CalibrationTests.offTopic {
                off.add(store.search([question], filter: Filter(), limit: 1))
            }
            print(String(format: "[calib]   %3d on   %@", store.concepts.count,
                         on.line))
            print(String(format: "[calib]   %3d off  %@", store.concepts.count,
                         off.line))
            try? FileManager.default.removeItem(at: root)
        }
        try? FileManager.default.removeItem(at: full.root)
    }

    func testSmallStoreLadder() throws {
        let e = try embedder()
        let full = Store(root: try copy(CalibrationTests.sample), embedder: e)
        full.load()
        let answers = CalibrationTests.sampleQuestions.map { question in
            full.search([question], filter: Filter(), limit: 1)
                .hits.first?.concept.id ?? ""
        }
        let others = full.concepts.map { concept in concept.id }
            .filter { id in !answers.contains(id) }
        print("[calib] ladder: standout of the correct answer, sample data/")
        print("[calib]     n   mean   min   max  found")
        for n in [3, 5, 8, 10, 15, 20, 40, 80, 157] {
            let kept = Set(answers.prefix(min(n, answers.count))
                           + others.prefix(max(0, n - answers.count)))
            let root = try copy(CalibrationTests.sample, keeping: kept)
            let store = Store(root: root, embedder: e)
            store.load()
            var standouts: [Float] = []
            var found = 0
            for (i, question) in CalibrationTests.sampleQuestions.enumerated()
            where kept.contains(answers[i]) {
                let result = store.search([question], filter: Filter(),
                                          limit: 1)
                standouts.append(result.standout)
                if result.hits.first?.concept.id == answers[i] { found += 1 }
            }
            let mean = standouts.reduce(0, +) / Float(max(1, standouts.count))
            print(String(format: "[calib]   %3d  %5.2f %5.2f %5.2f  %d/%d",
                         store.concepts.count, mean, standouts.min() ?? 0,
                         standouts.max() ?? 0, found, standouts.count))
            try? FileManager.default.removeItem(at: root)
        }
        try? FileManager.default.removeItem(at: full.root)
    }
}
