import XCTest
@testable import LLM

@MainActor final class RecallTests: XCTestCase {

    struct Planted {
        let id: String
        let title: String
        let description: String
        let body: String
        var at = "2026-11-07T00:00:00Z"
    }

    static let basilParagraphs = [
        "The box is a terracotta trough on the kitchen sill, forty "
            + "centimetres long, with three plants bought as seedlings at the "
            + "market in May.",
        "Watering is every second morning in summer and once a week once "
            + "the heating comes on, always from below so the stems stay dry.",
        "The leaves go into pesto on Sundays and onto tomatoes most other "
            + "days, picked from the top so the plant branches instead of "
            + "bolting.",
        "Aphids showed up once in August and a rinse under the tap took "
            + "care of them, with no soap and no spray.",
        "The pot sits on a saucer of gravel so the roots never stand in "
            + "water, and the saucer gets emptied after every watering.",
        "To keep it alive over the winter, cut the flower spikes as soon as "
            + "they show, move the box off the cold sill at night, and give "
            + "it a lamp for four hours on dark days.",
    ]

    static let neighbours = [
        Planted(id: "house/radon", title: "The radon test",
                description: "The basement test read 2.1 picocuries per "
                    + "litre in March.",
                body: "A charcoal kit from the hardware store, left on the "
                    + "workbench for four days with the windows shut."),
        Planted(id: "tech/mesh", title: "The mesh wifi",
                description: "Three Deco X55 units, one per floor, wired "
                    + "back to the router.",
                body: "The attic one drops out when the microwave runs."),
    ]

    static let basil = Planted(
        id: "garden/basil", title: "Basil in the window box",
        description: "The basil on the kitchen sill, and what it needs "
            + "through the year.",
        body: basilParagraphs.joined(separator: "\n\n"))

    static let twins = [
        Planted(id: "garden/basil-a", title: "Basil over the winter",
                description: "What the basil on the sill needs to live "
                    + "through the cold months.",
                body: "A lamp for four hours on dark days.",
                at: "2026-01-10T09:00:00Z"),
        Planted(id: "garden/basil-b", title: "Basil over the winter",
                description: "What the basil on the sill needs to live "
                    + "through the cold months.",
                body: "A lamp for four hours on dark days.",
                at: "2026-11-01T09:00:00Z"),
    ]

    private func plant(_ notes: [Planted]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("recall-" + UUID().uuidString,
                                    isDirectory: true)
        for note in notes {
            let url = root.appendingPathComponent(note.id + ".md")
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let text = "---\ntype: Note\ntitle: " + note.title
                + "\ndescription: " + note.description
                + "\ntags: [" + Store.area(of: note.id) + "]\n"
                + "generated: { by: test, at: " + note.at + " }\n---\n\n"
                + note.body + "\n"
            try text.write(to: url, atomically: true, encoding: .utf8)
        }
        return root
    }

    private func opened(_ notes: [Planted]) async throws
        -> (memories: Memories, root: URL) {
        let weights = try EncoderTests.cloned(EncoderTests.a25mRepo,
                                              EncoderTests.a25mFile)
        let root = try plant(notes)
        let memories = Memories(root: root)
        memories.open(encoder: weights)
        await memories.awaitOpen()
        XCTAssertTrue(memories.isOpen)
        return (memories, root)
    }

    func testRecallCarriesTheParagraphThatAnswers() async throws {
        let store = try await opened([RecallTests.basil]
                                     + RecallTests.neighbours)
        let recall = await store.memories.recall(
            "how do I keep basil alive over the winter", also: [],
            pp: 100, excluding: [])
        let block = recall?.block ?? ""
        XCTAssertEqual(recall?.ids.first, "garden/basil")
        XCTAssertTrue(block.contains("cut the flower spikes"), block)
        XCTAssertFalse(block.contains("terracotta trough"), block)
        XCTAssertTrue(block.contains("\n  ... "), block)
        print("[recall] window block:\n" + block)
        try? FileManager.default.removeItem(at: store.root)
    }

    func testLineWindowsTheBodyOnUTF8Boundaries() {
        let body = String(repeating: "\u{65E5}", count: 10)
        let concept = Concept(
            id: "a/b", path: URL(fileURLWithPath: "/a/b.md"), type: "Note",
            title: "T", description: "", tags: [], status: "",
            staleAfter: nil, generatedBy: "", verifiers: [],
            extraFrontmatter: [], body: body, links: [], hash: 0)
        let cut = Memories.line(concept, window: (from: 3, to: 10))
        XCTAssertEqual(cut, "- T\n  ... \u{65E5}\u{65E5} ...\n")
        let whole = Memories.line(concept, window: nil)
        XCTAssertEqual(whole, "- T\n  " + body + "\n")
    }

    func testTheNewerOfTwoRestatedNotesRidesFirst() async throws {
        let store = try await opened(RecallTests.twins
                                     + RecallTests.neighbours)
        let recall = await store.memories.recall(
            "how do I keep basil alive over the winter", also: [],
            pp: 100, excluding: [])
        XCTAssertEqual(Array((recall?.ids ?? []).prefix(2)),
                       ["garden/basil-b", "garden/basil-a"])
        print("[recall] order: " + (recall?.ids ?? []).joined(separator: " "))
        try? FileManager.default.removeItem(at: store.root)
    }

    private func hit(_ id: String, _ relevance: Float, _ at: String) -> Hit {
        Hit(concept: Concept(
                id: id, path: URL(fileURLWithPath: "/" + id + ".md"),
                type: "Note", title: id, description: "", tags: [],
                status: "", staleAfter: nil, generatedBy: "", verifiers: [],
                generatedAt: Store.iso.date(from: at), extraFrontmatter: [],
                body: "", links: [], hash: 0),
            score: relevance, relevance: relevance, terms: [])
    }

    func testTheMarginBoundsTheRecencyPass() {
        let hits = [hit("a", 0.70, "2026-01-01T00:00:00Z"),
                    hit("b", 0.69, "2026-06-01T00:00:00Z"),
                    hit("c", 0.60, "2026-10-01T00:00:00Z")]
        XCTAssertEqual(Memories.freshestFirst(hits).map { h in h.concept.id },
                       ["b", "a", "c"])
        let undated = [hit("a", 0.70, ""), hit("b", 0.69, "")]
        XCTAssertEqual(Memories.freshestFirst(undated)
                           .map { h in h.concept.id }, ["a", "b"])
        XCTAssertEqual(Memories.freshestFirst([]).count, 0)
    }
}
