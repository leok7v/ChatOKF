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

    private func encoders() throws -> [(label: String, embedder: Embedder)] {
        guard !TestWeights.skipped else {
            throw XCTSkip("CHATOKF_SKIP_WEIGHTS: embedder ladders skipped")
        }
        guard let a25m = ModernBertEmbedder.load(ggufPath: try EncoderTests
            .cloned(EncoderTests.a25mRepo, EncoderTests.a25mFile)) else {
            throw XCTSkip("not a ModernBERT GGUF: " + EncoderTests.a25mFile)
        }
        return [("a25m", a25m)]
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

    private func row(_ store: Store, _ question: String) -> String {
        let result = store.search([question], filter: Filter(), limit: 1)
        let top = result.hits.first
        return String(format: "%5.2f  %.3f %.3f  %-36@ %@%@", result.standout,
                      top?.score ?? 0, top?.relevance ?? 0,
                      String(question.prefix(36)), top?.concept.id ?? "-",
                      top?.terms.isEmpty == false
                          ? " [" + top!.terms.joined(separator: " ") + "]"
                          : "")
    }

    private func table(_ name: String, _ root: URL, _ label: String,
                       _ e: Embedder, _ questions: [String]) {
        let store = Store(root: root, embedder: e)
        store.load()
        print("[calib] \(label) \(name): \(store.concepts.count) concepts, "
              + "re-embedded \(store.embeddedCount), floor "
              + "\(e.relevanceFloor)")
        print("[calib] standout cosine relev  question                      "
              + "       top hit")
        for question in questions {
            print("[calib] " + row(store, question))
        }
        print("[calib] -- off topic")
        for question in CalibrationTests.offTopic {
            print("[calib] " + row(store, question))
        }
    }

    func testStandoutOnBothCorpora() throws {
        guard FileManager.default.fileExists(
            atPath: CalibrationTests.bundle.path) else {
            throw XCTSkip("no okf/ bundle beside this checkout")
        }
        for (label, e) in try encoders() {
            let sample = try copy(CalibrationTests.sample)
            table("sample data/", sample, label, e,
                  CalibrationTests.sampleQuestions)
            let bundle = try copy(CalibrationTests.bundle)
            table("this repo's okf/", bundle, label, e,
                  CalibrationTests.bundleQuestions)
            try? FileManager.default.removeItem(at: sample)
            try? FileManager.default.removeItem(at: bundle)
        }
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

    private func duplicates(_ label: String, _ e: Embedder) throws {
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
        var lines: [(score: Float, text: String)] = []
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
                lines.append((near[1].score, String(
                    format: "[calib]   %.3f %@ -> %@, words %d %@",
                    near[1].score, concept.id, near[1].id,
                    words?.weight ?? 0, other ?? "-")))
            }
        }
        print("[calib] \(label) the twelve closest pairs of existing "
              + "concepts:")
        for line in lines.sorted(by: { a, b in a.score > b.score })
            .prefix(12) {
            print(line.text)
        }
        tops.sort()
        print(String(format: "[calib] %@ nearest other concept over %d: min "
                         + "%.3f median %.3f p90 %.3f max %.3f, %d at or "
                         + "over %.2f", label,
                     tops.count, tops.first ?? 0, tops[tops.count / 2],
                     tops[tops.count * 9 / 10], tops.last ?? 0,
                     tops.filter { top in top >= e.mergeAt }.count,
                     e.mergeAt))
        print("[calib] \(label) restated notes against the store:")
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

    func testDuplicateThreshold() throws {
        for (label, e) in try encoders() { try duplicates(label, e) }
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
        for (label, e) in try encoders() {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("leak-" + UUID().uuidString,
                                        isDirectory: true)
            let store = Store(root: root, embedder: e)
            for note in CalibrationTests.planted {
                _ = try store.write(id: note.id, type: "Note",
                                    title: note.title,
                                    description: note.description, tags: [],
                                    body: note.body)
            }
            store.load()
            print("[leak] \(label) \(store.concepts.count) planted notes, "
                  + "floor \(e.relevanceFloor)")
            print("[leak] standout cosine top second relev top second "
                  + "question                    top hit")
            for question in CalibrationTests.leakQuestions {
                print("[leak] " + leakRow(store, question))
            }
            try? FileManager.default.removeItem(at: root)
        }
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

    private func answers(_ full: Store) -> (answers: [String],
                                            others: [String]) {
        let answers = CalibrationTests.sampleQuestions.map { question in
            full.search([question], filter: Filter(), limit: 1)
                .hits.first?.concept.id ?? ""
        }
        let others = full.concepts.map { concept in concept.id }
            .filter { id in !answers.contains(id) }
        return (answers, others)
    }

    func testOffTopicLadder() throws {
        for (label, e) in try encoders() {
            let full = Store(root: try copy(CalibrationTests.sample),
                             embedder: e)
            full.load()
            let key = answers(full)
            print("[calib] \(label) ladder: on topic against off topic, "
                  + "mean min max")
            for n in [3, 5, 8, 10, 15, 20, 40, 80, 157] {
                let kept = Set(key.answers.prefix(min(n, key.answers.count))
                               + key.others.prefix(
                                   max(0, n - key.answers.count)))
                let root = try copy(CalibrationTests.sample, keeping: kept)
                let store = Store(root: root, embedder: e)
                store.load()
                var on = Band()
                var off = Band()
                for (i, question) in CalibrationTests.sampleQuestions
                    .enumerated() where kept.contains(key.answers[i]) {
                    on.add(store.search([question], filter: Filter(),
                                        limit: 1))
                }
                for question in CalibrationTests.offTopic {
                    off.add(store.search([question], filter: Filter(),
                                         limit: 1))
                }
                print(String(format: "[calib]   %3d on   %@",
                             store.concepts.count, on.line))
                print(String(format: "[calib]   %3d off  %@",
                             store.concepts.count, off.line))
                try? FileManager.default.removeItem(at: root)
            }
            try? FileManager.default.removeItem(at: full.root)
        }
    }

    func testSmallStoreLadder() throws {
        for (label, e) in try encoders() {
            let full = Store(root: try copy(CalibrationTests.sample),
                             embedder: e)
            full.load()
            let key = answers(full)
            print("[calib] \(label) ladder: standout of the correct answer, "
                  + "sample data/")
            print("[calib]     n   mean   min   max  found")
            for n in [3, 5, 8, 10, 15, 20, 40, 80, 157] {
                let kept = Set(key.answers.prefix(min(n, key.answers.count))
                               + key.others.prefix(
                                   max(0, n - key.answers.count)))
                let root = try copy(CalibrationTests.sample, keeping: kept)
                let store = Store(root: root, embedder: e)
                store.load()
                var standouts: [Float] = []
                var found = 0
                for (i, question) in CalibrationTests.sampleQuestions
                    .enumerated() where kept.contains(key.answers[i]) {
                    let result = store.search([question], filter: Filter(),
                                              limit: 1)
                    standouts.append(result.standout)
                    if result.hits.first?.concept.id == key.answers[i] {
                        found += 1
                    }
                }
                let mean = standouts.reduce(0, +)
                    / Float(max(1, standouts.count))
                print(String(format: "[calib]   %3d  %5.2f %5.2f %5.2f  %d/%d",
                             store.concepts.count, mean,
                             standouts.min() ?? 0, standouts.max() ?? 0,
                             found, standouts.count))
                try? FileManager.default.removeItem(at: root)
            }
            try? FileManager.default.removeItem(at: full.root)
        }
    }

    static let sampleAnswers = [
        "house/basement-humidity", "garden/basil-window-box",
        "tech/wifi-mesh", "garden/last-frost", "watch/subscriptions",
        "house/door-sticking", "house/water-heater", "house/radon-test",
    ]

    static let translated: [(language: String, questions: [String])] = [
        ("French", [
            "pourquoi ma cave sent le moisi en ete",
            "comment garder mon basilic en vie pendant l'hiver",
            "quels boitiers wifi mesh avons-nous",
            "quand a lieu la derniere gelee ici",
            "quels abonnements de streaming payons-nous",
            "la porte de derriere coince, que faire",
            "quel age a le chauffe-eau",
            "qu'a dit le test de radon"]),
        ("Italian", [
            "perche il mio seminterrato puzza di muffa d'estate",
            "come faccio a tenere in vita il basilico durante l'inverno",
            "quali unita wifi mesh abbiamo",
            "quando e l'ultima gelata qui",
            "quali abbonamenti di streaming stiamo pagando",
            "la porta sul retro si incastra, cosa faccio",
            "quanti anni ha lo scaldabagno",
            "cosa diceva il test del radon"]),
        ("German", [
            "warum riecht mein Keller im Sommer muffig",
            "wie halte ich Basilikum ueber den Winter am Leben",
            "welche WLAN-Mesh-Geraete haben wir",
            "wann ist hier der letzte Frost",
            "welche Streaming-Abos bezahlen wir",
            "die Hintertuer klemmt, was soll ich tun",
            "wie alt ist der Warmwasserbereiter",
            "was hat der Radontest ergeben"]),
        ("Spanish", [
            "por que mi sotano huele a humedad en verano",
            "como mantengo viva la albahaca durante el invierno",
            "que unidades wifi mesh tenemos",
            "cuando es la ultima helada aqui",
            "que suscripciones de streaming estamos pagando",
            "la puerta trasera se atasca, que hago",
            "que antiguedad tiene el calentador de agua",
            "que dijo la prueba de radon"]),
        ("Portuguese", [
            "por que o meu porao cheira a mofo no verao",
            "como mantenho o manjericao vivo durante o inverno",
            "quais unidades de wifi mesh nos temos",
            "quando e a ultima geada aqui",
            "quais assinaturas de streaming estamos pagando",
            "a porta dos fundos emperra, o que eu faco",
            "quantos anos tem o aquecedor de agua",
            "o que disse o teste de radonio"]),
        ("Chinese", [
            "\u{4E3A}\u{4EC0}\u{4E48}\u{6211}\u{7684}\u{5730}\u{4E0B}\u{5BA4}"
                + "\u{590F}\u{5929}\u{6709}\u{9709}\u{5473}",
            "\u{51AC}\u{5929}\u{600E}\u{4E48}\u{8BA9}\u{7F57}\u{52D2}"
                + "\u{6D3B}\u{4E0B}\u{6765}",
            "\u{6211}\u{4EEC}\u{5BB6}\u{6709}\u{54EA}\u{4E9B}wifi mesh"
                + "\u{8DEF}\u{7531}\u{5668}",
            "\u{8FD9}\u{91CC}\u{6700}\u{540E}\u{4E00}\u{6B21}\u{971C}"
                + "\u{51BB}\u{662F}\u{4EC0}\u{4E48}\u{65F6}\u{5019}",
            "\u{6211}\u{4EEC}\u{5728}\u{4ED8}\u{54EA}\u{4E9B}\u{6D41}"
                + "\u{5A92}\u{4F53}\u{8BA2}\u{9605}",
            "\u{540E}\u{95E8}\u{5361}\u{4F4F}\u{4E86}\u{FF0C}\u{6211}"
                + "\u{8BE5}\u{600E}\u{4E48}\u{529E}",
            "\u{70ED}\u{6C34}\u{5668}\u{7528}\u{4E86}\u{591A}\u{5C11}"
                + "\u{5E74}\u{4E86}",
            "\u{6C21}\u{6C14}\u{68C0}\u{6D4B}\u{7ED3}\u{679C}\u{662F}"
                + "\u{4EC0}\u{4E48}"]),
        ("Japanese", [
            "\u{590F}\u{306B}\u{5730}\u{4E0B}\u{5BA4}\u{304C}\u{304B}\u{3073}"
                + "\u{81ED}\u{3044}\u{306E}\u{306F}\u{306A}\u{305C}",
            "\u{51AC}\u{306E}\u{9593}\u{30D0}\u{30B8}\u{30EB}\u{3092}\u{67AF}"
                + "\u{3089}\u{3055}\u{306A}\u{3044}\u{306B}\u{306F}\u{3069}"
                + "\u{3046}\u{3059}\u{308C}\u{3070}\u{3044}\u{3044}",
            "\u{3046}\u{3061}\u{306E}wifi\u{30E1}\u{30C3}\u{30B7}\u{30E5}"
                + "\u{6A5F}\u{5668}\u{306F}\u{3069}\u{308C}",
            "\u{3053}\u{3053}\u{306E}\u{6700}\u{5F8C}\u{306E}\u{971C}\u{306F}"
                + "\u{3044}\u{3064}",
            "\u{3069}\u{306E}\u{30B9}\u{30C8}\u{30EA}\u{30FC}\u{30DF}\u{30F3}"
                + "\u{30B0}\u{306E}\u{30B5}\u{30D6}\u{30B9}\u{30AF}\u{3092}"
                + "\u{6255}\u{3063}\u{3066}\u{3044}\u{308B}",
            "\u{88CF}\u{53E3}\u{306E}\u{30C9}\u{30A2}\u{304C}\u{5F15}\u{3063}"
                + "\u{304B}\u{304B}\u{308B}\u{3001}\u{3069}\u{3046}\u{3059}"
                + "\u{308C}\u{3070}\u{3044}\u{3044}",
            "\u{7D66}\u{6E6F}\u{5668}\u{306F}\u{4F55}\u{5E74}\u{4F7F}\u{3063}"
                + "\u{3066}\u{3044}\u{308B}",
            "\u{30E9}\u{30C9}\u{30F3}\u{691C}\u{67FB}\u{306E}\u{7D50}\u{679C}"
                + "\u{306F}\u{4F55}\u{3060}\u{3063}\u{305F}"]),
        ("Korean", [
            "\u{C5EC}\u{B984}\u{C5D0} \u{C9C0}\u{D558}\u{C2E4}\u{C5D0}\u{C11C}"
                + " \u{C65C} \u{ACF0}\u{D321}\u{C774} \u{B0C4}\u{C0C8}\u{AC00}"
                + " \u{B098}\u{C9C0}",
            "\u{ACA8}\u{C6B8} \u{B3D9}\u{C548} \u{BC14}\u{C9C8}\u{C744} "
                + "\u{C5B4}\u{B5BB}\u{AC8C} \u{C0B4}\u{B824} \u{B450}\u{C9C0}",
            "\u{C6B0}\u{B9AC} \u{C9D1} \u{C640}\u{C774}\u{D30C}\u{C774} "
                + "\u{BA54}\u{C2DC} \u{AE30}\u{AE30}\u{B294} \u{C5B4}\u{B5A4}"
                + " \u{AC83}\u{B4E4}\u{C774}\u{C9C0}",
            "\u{C5EC}\u{AE30} \u{B9C8}\u{C9C0}\u{B9C9} \u{C11C}\u{B9AC}\u{B294}"
                + " \u{C5B8}\u{C81C}\u{C9C0}",
            "\u{C6B0}\u{B9AC}\u{AC00} \u{ACB0}\u{C81C}\u{D558}\u{B294} "
                + "\u{C2A4}\u{D2B8}\u{B9AC}\u{BC0D} \u{AD6C}\u{B3C5}\u{C740}"
                + " \u{BB50}\u{C9C0}",
            "\u{B4B7}\u{BB38}\u{C774} \u{AC78}\u{B824}\u{C11C} \u{C548} "
                + "\u{C5F4}\u{B824}, \u{C5B4}\u{B5BB}\u{AC8C} \u{D558}\u{C9C0}",
            "\u{C628}\u{C218}\u{AE30}\u{B294} \u{BA87} \u{B144} \u{B410}"
                + "\u{C9C0}",
            "\u{B77C}\u{B3C8} \u{AC80}\u{C0AC} \u{ACB0}\u{ACFC}\u{AC00} "
                + "\u{BB50}\u{C600}\u{C9C0}"]),
        ("Arabic", [
            "\u{644}\u{645}\u{627}\u{630}\u{627} \u{62A}\u{641}\u{648}\u{62D}"
                + " \u{631}\u{627}\u{626}\u{62D}\u{629} \u{627}\u{644}\u{639}"
                + "\u{641}\u{646} \u{645}\u{646} \u{642}\u{628}\u{648} "
                + "\u{645}\u{646}\u{632}\u{644}\u{64A} \u{641}\u{64A} "
                + "\u{627}\u{644}\u{635}\u{64A}\u{641}",
            "\u{643}\u{64A}\u{641} \u{623}\u{628}\u{642}\u{64A} \u{627}\u{644}"
                + "\u{631}\u{64A}\u{62D}\u{627}\u{646} \u{62D}\u{64A}\u{627}"
                + "\u{64B} \u{62E}\u{644}\u{627}\u{644} \u{627}\u{644}\u{634}"
                + "\u{62A}\u{627}\u{621}",
            "\u{645}\u{627} \u{648}\u{62D}\u{62F}\u{627}\u{62A} \u{627}\u{644}"
                + "\u{648}\u{627}\u{64A} \u{641}\u{627}\u{64A} \u{627}\u{644}"
                + "\u{634}\u{628}\u{643}\u{64A} \u{627}\u{644}\u{62A}\u{64A}"
                + " \u{644}\u{62F}\u{64A}\u{646}\u{627}",
            "\u{645}\u{62A}\u{649} \u{64A}\u{643}\u{648}\u{646} \u{622}\u{62E}"
                + "\u{631} \u{635}\u{642}\u{64A}\u{639} \u{647}\u{646}\u{627}",
            "\u{645}\u{627} \u{627}\u{634}\u{62A}\u{631}\u{627}\u{643}\u{627}"
                + "\u{62A} \u{627}\u{644}\u{628}\u{62B} \u{627}\u{644}\u{62A}"
                + "\u{64A} \u{646}\u{62F}\u{641}\u{639} \u{62B}\u{645}\u{646}"
                + "\u{647}\u{627}",
            "\u{627}\u{644}\u{628}\u{627}\u{628} \u{627}\u{644}\u{62E}\u{644}"
                + "\u{641}\u{64A} \u{64A}\u{639}\u{644}\u{642}\u{60C} "
                + "\u{645}\u{627}\u{630}\u{627} \u{623}\u{641}\u{639}\u{644}",
            "\u{643}\u{645} \u{639}\u{645}\u{631} \u{633}\u{62E}\u{627}\u{646}"
                + " \u{627}\u{644}\u{645}\u{627}\u{621}",
            "\u{645}\u{627}\u{630}\u{627} \u{642}\u{627}\u{644} \u{627}\u{62E}"
                + "\u{62A}\u{628}\u{627}\u{631} \u{627}\u{644}\u{631}\u{627}"
                + "\u{62F}\u{648}\u{646}"]),
        ("Ukrainian", [
            "\u{447}\u{43E}\u{43C}\u{443} \u{43C}\u{456}\u{439} \u{43F}\u{456}"
                + "\u{434}\u{432}\u{430}\u{43B} \u{432}\u{43B}\u{456}\u{442}"
                + "\u{43A}\u{443} \u{43F}\u{430}\u{445}\u{43D}\u{435} "
                + "\u{446}\u{432}\u{456}\u{43B}\u{43B}\u{44E}",
            "\u{44F}\u{43A} \u{437}\u{431}\u{435}\u{440}\u{435}\u{433}\u{442}"
                + "\u{438} \u{431}\u{430}\u{437}\u{438}\u{43B}\u{456}\u{43A}"
                + " \u{436}\u{438}\u{432}\u{438}\u{43C} \u{443}\u{437}\u{438}"
                + "\u{43C}\u{43A}\u{443}",
            "\u{44F}\u{43A}\u{456} \u{443} \u{43D}\u{430}\u{441} wifi mesh "
                + "\u{43F}\u{440}\u{438}\u{441}\u{442}\u{440}\u{43E}\u{457}",
            "\u{43A}\u{43E}\u{43B}\u{438} \u{442}\u{443}\u{442} \u{43E}\u{441}"
                + "\u{442}\u{430}\u{43D}\u{43D}\u{456}\u{439} \u{437}\u{430}"
                + "\u{43C}\u{43E}\u{440}\u{43E}\u{437}\u{43E}\u{43A}",
            "\u{437}\u{430} \u{44F}\u{43A}\u{456} \u{441}\u{442}\u{440}\u{456}"
                + "\u{43C}\u{456}\u{43D}\u{433}\u{43E}\u{432}\u{456} \u{43F}"
                + "\u{456}\u{434}\u{43F}\u{438}\u{441}\u{43A}\u{438} \u{43C}"
                + "\u{438} \u{43F}\u{43B}\u{430}\u{442}\u{438}\u{43C}\u{43E}",
            "\u{437}\u{430}\u{434}\u{43D}\u{456} \u{434}\u{432}\u{435}\u{440}"
                + "\u{456} \u{437}\u{430}\u{43A}\u{43B}\u{438}\u{43D}\u{44E}"
                + "\u{454}, \u{449}\u{43E} \u{440}\u{43E}\u{431}\u{438}"
                + "\u{442}\u{438}",
            "\u{441}\u{43A}\u{456}\u{43B}\u{44C}\u{43A}\u{438} \u{440}\u{43E}"
                + "\u{43A}\u{456}\u{432} \u{432}\u{43E}\u{434}\u{43E}\u{43D}"
                + "\u{430}\u{433}\u{440}\u{456}\u{432}\u{430}\u{447}\u{443}",
            "\u{449}\u{43E} \u{43F}\u{43E}\u{43A}\u{430}\u{437}\u{430}\u{432}"
                + " \u{442}\u{435}\u{441}\u{442} \u{43D}\u{430} \u{440}\u{430}"
                + "\u{434}\u{43E}\u{43D}"]),
        ("Russian", [
            "\u{43F}\u{43E}\u{447}\u{435}\u{43C}\u{443} \u{43C}\u{43E}\u{439}"
                + " \u{43F}\u{43E}\u{434}\u{432}\u{430}\u{43B} \u{43B}\u{435}"
                + "\u{442}\u{43E}\u{43C} \u{43F}\u{430}\u{445}\u{43D}\u{435}"
                + "\u{442} \u{43F}\u{43B}\u{435}\u{441}\u{435}\u{43D}\u{44C}"
                + "\u{44E}",
            "\u{43A}\u{430}\u{43A} \u{441}\u{43E}\u{445}\u{440}\u{430}\u{43D}"
                + "\u{438}\u{442}\u{44C} \u{431}\u{430}\u{437}\u{438}\u{43B}"
                + "\u{438}\u{43A} \u{436}\u{438}\u{432}\u{44B}\u{43C} "
                + "\u{437}\u{438}\u{43C}\u{43E}\u{439}",
            "\u{43A}\u{430}\u{43A}\u{438}\u{435} \u{443} \u{43D}\u{430}\u{441}"
                + " wifi mesh \u{443}\u{441}\u{442}\u{440}\u{43E}\u{439}"
                + "\u{441}\u{442}\u{432}\u{430}",
            "\u{43A}\u{43E}\u{433}\u{434}\u{430} \u{437}\u{434}\u{435}\u{441}"
                + "\u{44C} \u{43F}\u{43E}\u{441}\u{43B}\u{435}\u{434}\u{43D}"
                + "\u{438}\u{435} \u{437}\u{430}\u{43C}\u{43E}\u{440}\u{43E}"
                + "\u{437}\u{43A}\u{438}",
            "\u{437}\u{430} \u{43A}\u{430}\u{43A}\u{438}\u{435} \u{441}\u{442}"
                + "\u{440}\u{438}\u{43C}\u{438}\u{43D}\u{433}\u{43E}\u{432}"
                + "\u{44B}\u{435} \u{43F}\u{43E}\u{434}\u{43F}\u{438}\u{441}"
                + "\u{43A}\u{438} \u{43C}\u{44B} \u{43F}\u{43B}\u{430}\u{442}"
                + "\u{438}\u{43C}",
            "\u{437}\u{430}\u{434}\u{43D}\u{44F}\u{44F} \u{434}\u{432}\u{435}"
                + "\u{440}\u{44C} \u{437}\u{430}\u{435}\u{434}\u{430}\u{435}"
                + "\u{442}, \u{447}\u{442}\u{43E} \u{434}\u{435}\u{43B}"
                + "\u{430}\u{442}\u{44C}",
            "\u{441}\u{43A}\u{43E}\u{43B}\u{44C}\u{43A}\u{43E} \u{43B}\u{435}"
                + "\u{442} \u{432}\u{43E}\u{434}\u{43E}\u{43D}\u{430}\u{433}"
                + "\u{440}\u{435}\u{432}\u{430}\u{442}\u{435}\u{43B}\u{44E}",
            "\u{447}\u{442}\u{43E} \u{43F}\u{43E}\u{43A}\u{430}\u{437}\u{430}"
                + "\u{43B} \u{442}\u{435}\u{441}\u{442} \u{43D}\u{430} "
                + "\u{440}\u{430}\u{434}\u{43E}\u{43D}"]),
    ]

    func testTranslatedQuestionsFindTheSameNotes() throws {
        var found = 0
        var passed = 0
        for (label, e) in try encoders() {
            let root = try copy(CalibrationTests.sample)
            let store = Store(root: root, embedder: e)
            store.load()
            print("[multi] \(label): language, found at rank one of 8, right "
                  + "answers over the floor \(e.relevanceFloor), relevance "
                  + "range of the right answers, the misses")
            for (language, questions) in CalibrationTests.translated {
                var hits = 0
                var passing = 0
                var low: Float = 1
                var high: Float = 0
                var misses: [String] = []
                for (i, question) in questions.enumerated() {
                    let result = store.search([question], filter: Filter(),
                                              limit: 3)
                    let top = result.hits.first
                    if top?.concept.id == CalibrationTests.sampleAnswers[i] {
                        hits += 1
                        low = min(low, top!.relevance)
                        high = max(high, top!.relevance)
                        if store.relevant(top!) { passing += 1 }
                    } else {
                        misses.append(String(i + 1) + ":"
                                      + (top?.concept.id ?? "-"))
                    }
                }
                found += hits
                passed += passing
                XCTAssertGreaterThanOrEqual(hits, 4, label + " " + language)
                print(String(format: "[multi]   %-10@ %d/8  pass %d  "
                                 + "%.3f..%.3f  %@", language, hits, passing,
                             hits > 0 ? low : 0, high,
                             misses.joined(separator: " ")))
            }
            try? FileManager.default.removeItem(at: root)
        }
        print("[multi] found in total: \(found), over the floor: \(passed)")
        XCTAssertGreaterThanOrEqual(found, 60)
        XCTAssertGreaterThanOrEqual(passed, 50)
    }
}
