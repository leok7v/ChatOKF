import XCTest
@testable import Chat
@testable import ChatOKF
@testable import LLM

@MainActor final class MemoriesTests: XCTestCase {

    private func temporaryRoot() -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memories-" + UUID().uuidString,
                                    isDirectory: true)
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    private func seeded() throws -> Memories {
        guard let url = BertEmbedder.bundledMultilingual,
              let embedder = BertEmbedder.load(ggufPath: url.path) else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = temporaryRoot()
        let store = Store(root: root, embedder: embedder)
        _ = try store.write(
            id: "house/basement-humidity", type: "Problem",
            title: "Basement humidity and the musty smell",
            description: "Sixty-five percent relative humidity in summer, "
                + "above the mold threshold, and the grading is part of it.",
            tags: ["house"], body: "The hygrometer reads 62 to 68 percent "
                + "from June through September.")
        _ = try store.write(
            id: "garden/basil", type: "Note", title: "Growing basil",
            description: "Basil hates cold, bolts the moment it flowers, "
                + "and wants pinching every week.",
            tags: ["garden"], body: "Pinch above a leaf pair.")
        _ = try store.write(
            id: "tech/wifi-mesh", type: "Note", title: "The mesh wifi setup",
            description: "Three TP-Link Deco X55 units wired back to the "
                + "router.", tags: ["tech"], body: "One per floor.")
        _ = try store.write(
            id: "person/pet-name", type: "Note", title: "The user's pet name",
            description: "The user's dog is called Biscuit.",
            tags: ["person"], body: "A beagle, three years old.")
        let memories = Memories(root: root)
        memories.enabled = true
        return memories
    }

    static let foreign = [
        "Which block gives the most fruit per tree, and what makes that "
            + "surprising?",
        "explain dark matter and dark energy",
        "what is the name of the tallest mountain",
        "who was the user of the first telephone",
    ]

    func testAForeignQuestionRecallsNothingOnASmallStore() async throws {
        let memories = try seeded()
        await memories.awaitOpen()
        for question in MemoriesTests.foreign {
            let got = await memories.recall(question, also: [], pp: 400,
                                      excluding: [])
            XCTAssertNil(got, question + " -> " + (got?.block ?? ""))
        }
        let dog = await memories.recall("what is my dog called", also: [],
                                  pp: 400, excluding: [])
        XCTAssertEqual(dog?.ids, ["person/pet-name"])
        XCTAssertTrue(dog?.block.contains("Biscuit") == true, dog?.block ?? "")
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testRecallFindsTheNoteAndSkipsWhatWasRead() async throws {
        let memories = try seeded()
        let cold = await memories.recall("basil", also: [], pp: 400,
                                         excluding: [])
        XCTAssertNil(cold)
        await memories.awaitOpen()
        XCTAssertTrue(memories.isOpen)
        let first = await memories.recall("why does my basement smell musty",
                                    also: [], pp: 400, excluding: [])
        XCTAssertEqual(first?.ids.first, "house/basement-humidity")
        XCTAssertTrue(first?.block.contains("musty smell") == true)
        XCTAssertGreaterThan(first?.tokens ?? 0, 20)
        let again = await memories.recall("why does my basement smell musty",
                                    also: [], pp: 400,
                                    excluding: ["house/basement-humidity",
                                                "garden/basil",
                                                "tech/wifi-mesh"])
        XCTAssertNil(again)
        let map = await memories.map()
        XCTAssertTrue(map.contains("house/"))
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testAVerbatimModelNumberRecallsOnASmallStore() async throws {
        let memories = try seeded()
        await memories.awaitOpen()
        let hit = await memories.recall("Deco X55", also: [], pp: 400,
                                  excluding: [])
        XCTAssertEqual(hit?.ids, ["tech/wifi-mesh"])
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testSwitchOffRecallsNothing() async throws {
        let memories = try seeded()
        memories.enabled = false
        await memories.awaitOpen()
        XCTAssertFalse(memories.isOpen)
        let off = await memories.recall("basil", also: [], pp: 400,
                                        excluding: [])
        XCTAssertNil(off)
        let map = await memories.map()
        XCTAssertEqual(map, "")
        memories.enabled = true
        try? FileManager.default.removeItem(at: memories.root)
    }
}

@MainActor final class MemoryToolsTests: XCTestCase {

    private func opened() async throws -> Memories {
        guard let url = BertEmbedder.bundledMultilingual,
              let embedder = BertEmbedder.load(ggufPath: url.path) else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memtools-" + UUID().uuidString,
                                    isDirectory: true)
        let store = Store(root: root, embedder: embedder)
        _ = try store.write(id: "tech/wifi-mesh", type: "Note",
                            title: "The mesh wifi setup",
                            description: "Three TP-Link Deco X55 units wired "
                                + "back to the router.",
                            tags: ["tech"], body: "One per floor.")
        let memories = Memories(root: root)
        memories.enabled = true
        await memories.awaitOpen()
        return memories
    }

    func testSearchReadCreateUpdateForget() async throws {
        let m = try await opened()
        let runner = MemoryToolRunner(inner: SafeToolRunner(), memories: m)
        XCTAssertTrue(runner.tools.contains { spec in
            spec.name == "memory_search"
        })
        let found = await runner.execute(
            "memory_search", [ToolArg(name: "query", value: "Deco X55")])
        XCTAssertTrue(found.contains("tech/wifi-mesh"), found)
        XCTAssertFalse(found.contains("[unrelated]"), found)
        XCTAssertFalse(found.contains("none is about"), found)
        let foreign = await runner.execute("memory_search", [
            ToolArg(name: "query",
                    value: "explain dark matter and dark energy")])
        XCTAssertTrue(foreign.hasPrefix("none of the user's notes"), foreign)
        XCTAssertFalse(foreign.contains("wifi"), foreign)
        let read = await runner.execute(
            "memory_read", [ToolArg(name: "id", value: "tech/wifi-mesh")])
        XCTAssertTrue(read.contains("One per floor."), read)
        let dup = await runner.execute("memory_create", [
            ToolArg(name: "id", value: "tech/wifi-mesh"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "x"),
            ToolArg(name: "description", value: "y"),
            ToolArg(name: "body", value: "z")])
        XCTAssertTrue(dup.contains("already exists"), dup)
        let made = await runner.execute("memory_create", [
            ToolArg(name: "id", value: "garden/basil"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "Growing basil"),
            ToolArg(name: "description", value: "Basil hates cold."),
            ToolArg(name: "tags", value: "garden, private"),
            ToolArg(name: "body", value: "Pinch weekly.")])
        XCTAssertTrue(made.contains("created garden/basil"), made)
        XCTAssertFalse(made.contains("draft"), made)
        let text = try String(contentsOf: m.root.appendingPathComponent(
            "garden/basil.md"), encoding: .utf8)
        XCTAssertFalse(text.contains("status: draft"), text)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        XCTAssertTrue(text.contains("tags: [garden, private]"), text)
        XCTAssertEqual(m.takeNoted().map { note in note.id },
                       ["garden/basil"])
        XCTAssertTrue(m.takeNoted().isEmpty)
        let restated = await runner.execute("memory_create", [
            ToolArg(name: "id", value: "tech/mesh-network"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "Home mesh network"),
            ToolArg(name: "description", value: "Three TP-Link Deco X55 "
                    + "units wired back to the router."),
            ToolArg(name: "body", value: "One per floor.")])
        XCTAssertTrue(restated.contains("restates tech/wifi-mesh"), restated)
        XCTAssertNil(m.note("tech/mesh-network"))
        let gone = await runner.execute(
            "memory_forget", [ToolArg(name: "id", value: "garden/basil")])
        XCTAssertTrue(gone.contains("retired garden/basil"), gone)
        let missing = await runner.execute("memory_update", [
            ToolArg(name: "id", value: "nope/never"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "x"),
            ToolArg(name: "description", value: "y"),
            ToolArg(name: "body", value: "z")])
        XCTAssertTrue(missing.contains("no such note"), missing)
        try? FileManager.default.removeItem(at: m.root)
    }

    func testNoToolThatWritesIsOfferedUnlessAsked() async throws {
        let m = try await opened()
        func memory(_ runner: MemoryToolRunner) -> [String] {
            runner.tools.map { spec in spec.name }
                .filter { name in name.hasPrefix("memory_") }
        }
        let reads = MemoryToolRunner(inner: SafeToolRunner(), memories: m,
                                     writes: false)
        XCTAssertEqual(memory(reads), ["memory_search", "memory_read"])
        let all = MemoryToolRunner(inner: SafeToolRunner(), memories: m)
        XCTAssertEqual(memory(all), MemoryTools.names)
        XCTAssertFalse(Models.offersNoteTools)
        try? FileManager.default.removeItem(at: m.root)
    }

    func testUpdateReplacesTheNoteInPlace() async throws {
        let m = try await opened()
        let path = m.root.appendingPathComponent("tech/wifi-mesh.md")
        let runner = MemoryToolRunner(inner: SafeToolRunner(), memories: m)
        let updated = await runner.execute("memory_update", [
            ToolArg(name: "id", value: "tech/wifi-mesh"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "The mesh wifi setup"),
            ToolArg(name: "description", value: "Four Deco X55 units."),
            ToolArg(name: "body", value: "One per floor and one in the "
                    + "garage.")])
        XCTAssertTrue(updated.contains("updated tech/wifi-mesh"), updated)
        let text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertFalse(text.contains("status: draft"), text)
        XCTAssertFalse(text.contains("verified:"), text)
        XCTAssertEqual(text.components(separatedBy: "generated:").count, 2)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        XCTAssertTrue(text.contains("Four Deco X55 units."), text)
        XCTAssertEqual(m.takeNoted().map { note in note.id },
                       ["tech/wifi-mesh"])
        XCTAssertNotNil(m.note("tech/wifi-mesh"))
        try? FileManager.default.removeItem(at: m.root)
    }
}

@MainActor final class MemoryTrashTests: XCTestCase {

    private let source = UUID()

    private func opened() async throws -> Memories {
        guard let url = BertEmbedder.bundledMultilingual,
              let embedder = BertEmbedder.load(ggufPath: url.path) else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memtrash-" + UUID().uuidString,
                                    isDirectory: true)
        let store = Store(root: root, embedder: embedder)
        _ = try store.write(
            id: "tech/wifi-mesh", type: "Note", title: "The mesh wifi setup",
            description: "Three TP-Link Deco X55 units.", tags: ["tech"],
            body: "One per floor.",
            adding: ["sources:",
                     "  - resource: chatokf://conversation/"
                         + source.uuidString])
        _ = try store.write(
            id: "house/wiring", type: "Note", title: "The house wiring",
            description: "Cat6 to every floor.", tags: ["house", "private"],
            body: "The run feeds [the mesh setup](/tech/wifi-mesh.md) "
                + "upstairs.")
        let memories = Memories(root: root)
        memories.enabled = true
        await memories.awaitOpen()
        return memories
    }

    func testTrashCollapsesTheLinkAndRestorePutsItBack() async throws {
        let m = try await opened()
        let wiring = m.root.appendingPathComponent("house/wiring.md")
        XCTAssertEqual(m.list.count, 2)
        XCTAssertEqual(m.note(detail: "house/wiring")?.links,
                       ["tech/wifi-mesh"])
        await m.trash("tech/wifi-mesh")
        XCTAssertEqual(m.list.map { row in row.id }, ["house/wiring"])
        XCTAssertEqual(m.trashed.map { row in row.id }, ["tech/wifi-mesh"])
        XCTAssertNil(m.note("tech/wifi-mesh"))
        let gone = await m.recall("Deco X55", also: [], pp: 400,
                                  excluding: [])
        XCTAssertFalse(gone?.ids.contains("tech/wifi-mesh") ?? false)
        var text = try String(contentsOf: wiring, encoding: .utf8)
        XCTAssertTrue(text.contains("feeds the mesh setup upstairs."), text)
        XCTAssertFalse(text.contains("wifi-mesh.md"), text)
        XCTAssertEqual(m.note(detail: "house/wiring")?.links, [])
        await m.restore("tech/wifi-mesh")
        XCTAssertEqual(m.trashed.count, 0)
        XCTAssertNotNil(m.note("tech/wifi-mesh"))
        text = try String(contentsOf: wiring, encoding: .utf8)
        XCTAssertTrue(
            text.contains("[the mesh setup](/tech/wifi-mesh.md)"), text)
        XCTAssertEqual(m.note(detail: "house/wiring")?.links,
                       ["tech/wifi-mesh"])
        try? FileManager.default.removeItem(at: m.root)
    }

    func testDeleteForeverAndEmptyTrashLeaveNoFile() async throws {
        let m = try await opened()
        await m.trash("tech/wifi-mesh")
        let kept = m.root.appendingPathComponent(
            Memories.trashFolder + "/tech/wifi-mesh.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: kept.path))
        await m.deleteForever("tech/wifi-mesh")
        XCTAssertFalse(FileManager.default.fileExists(atPath: kept.path))
        XCTAssertEqual(m.trashed.count, 0)
        await m.trash("house/wiring")
        XCTAssertEqual(m.trashed.count, 1)
        await m.emptyTrash()
        XCTAssertEqual(m.trashed.count, 0)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: m.root.appendingPathComponent(
                Memories.trashFolder).path))
        XCTAssertTrue(m.list.isEmpty)
        try? FileManager.default.removeItem(at: m.root)
    }

    func testRowsCarryTheAreaThePrivateTagAndTheSourceChat() async throws {
        let m = try await opened()
        let wiring = m.list.first { row in row.id == "house/wiring" }
        let mesh = m.list.first { row in row.id == "tech/wifi-mesh" }
        XCTAssertEqual(wiring?.area, "house")
        XCTAssertEqual(wiring?.isPrivate, true)
        XCTAssertEqual(mesh?.isPrivate, false)
        XCTAssertEqual(mesh?.source, source)
        XCTAssertNil(wiring?.source)
        XCTAssertEqual(m.notes(from: source).map { row in row.id },
                       ["tech/wifi-mesh"])
        XCTAssertEqual(m.notes(from: UUID()).count, 0)
        try? FileManager.default.removeItem(at: m.root)
    }

    func testTrashSurvivesAReopenAndAreasGroupEveryNote() async throws {
        let m = try await opened()
        await m.trash("tech/wifi-mesh")
        m.reopen()
        await m.awaitOpen()
        XCTAssertEqual(m.trashed.map { row in row.id }, ["tech/wifi-mesh"])
        XCTAssertEqual(m.list.map { row in row.id }, ["house/wiring"])
        let areas = Sidebar.byArea(m.list).map { group in group.area }
        XCTAssertEqual(areas, ["house"])
        XCTAssertEqual(Sidebar.areaTitle("house"), "House")
        XCTAssertEqual(Sidebar.areaTitle("."), "Loose")
        try? FileManager.default.removeItem(at: m.root)
    }

    func testSearchRanksTheTitleOverTheDescription() async throws {
        let m = try await opened()
        XCTAssertTrue(MemorySearch.active("mesh"))
        XCTAssertFalse(MemorySearch.active("m"))
        XCTAssertEqual(MemorySearch.rank(m.list, "mesh").map { r in r.id },
                       ["tech/wifi-mesh"])
        XCTAssertEqual(MemorySearch.rank(m.list, "cat6").map { r in r.id },
                       ["house/wiring"])
        XCTAssertEqual(MemorySearch.rank(m.list, "private").map { r in r.id },
                       ["house/wiring"])
        XCTAssertTrue(MemorySearch.rank(m.list, "zzzz").isEmpty)
        try? FileManager.default.removeItem(at: m.root)
    }
}

@MainActor final class ExtractionTests: XCTestCase {

    func testParseDraftsKeepsSentencesAboutTheUser() {
        let raw = """
        ```text
        - The user's cat is called Marmalade and is 7 years old. She naps.
        2. The user works as a marine biologist in Lisbon.
        FACT
        title:
        I work as a marine biologist.
        the user is allergic to shellfish
        The user likes tea.
        ```
        """
        let drafts = Memories.parseDrafts(raw)
        XCTAssertEqual(drafts.map { d in d.title },
                       ["Cat is called Marmalade",
                        "Works as a marine biologist",
                        "Is allergic to shellfish"], "three at most")
        XCTAssertEqual(drafts[0].description,
                       "The user's cat is called Marmalade and is 7 years "
                       + "old.")
        XCTAssertEqual(Memories.parseDrafts("NONE\n```").count, 0)
        XCTAssertEqual(Memories.parseDrafts("The user.").count, 0)
        XCTAssertFalse(Memories.extractionInstruction.contains("/"),
                       "the form shows no id to copy")
    }

    func testOnlyAMessageAboutTheUserIsWorthAnExtraction() {
        XCTAssertTrue(Memories.speaksOfSelf(
            "Remember that my dog is called Biscuit."))
        XCTAssertTrue(Memories.speaksOfSelf("I'm allergic to nuts"))
        XCTAssertTrue(Memories.speaksOfSelf(
            "\u{041C}\u{043E}\u{044E} \u{043A}\u{043E}\u{0448}\u{043A}"
            + "\u{0443} \u{0437}\u{043E}\u{0432}\u{0443}\u{0442} "
            + "\u{041C}\u{0443}\u{0440}\u{043A}\u{0430}, "
            + "\u{0437}\u{0430}\u{043F}\u{043E}\u{043C}\u{043D}\u{0438}"))
        XCTAssertFalse(Memories.speaksOfSelf("What is 17 times 23?"))
        XCTAssertFalse(Memories.speaksOfSelf(
            "If I put 1000 dollars in savings, how much do I have later?"))
        XCTAssertTrue(Memories.speaksOfSelf(
            "I moved to Lisbon last year. What should I see there?"))
        XCTAssertFalse(Memories.speaksOfSelf(
            "Explain dark matter and dark energy in a few sentences."))
    }

    func testATitleTheUserNeverSaidIsNotGrounded() {
        let asked = "If I put 1000 dollars in a savings account at 5 percent "
            + "compound interest, how much is it after 22 years?"
        XCTAssertFalse(Memories.grounded(Memories.Draft(
            title: "Black Holes",
            description: "The user asked about compound interest."),
            in: asked))
        XCTAssertTrue(Memories.grounded(Memories.Draft(
            title: "Savings account",
            description: "Has 1000 dollars in savings."), in: asked))
        XCTAssertTrue(Memories.grounded(Memories.Draft(
            title: "Tea", description: "Drinks green tea."),
            in: "I drink green tea in the afternoon."))
        XCTAssertTrue(Memories.grounded(Memories.Draft(
            title: "Health", description: "Has a shellfish allergy."),
            in: "I work in Lisbon and I am allergic to shellfish."))
        XCTAssertFalse(Memories.grounded(Memories.Draft(
            title: "User's name", description: "Leo"),
            in: "How much was that savings balance?"))
        XCTAssertEqual(MemoryTools.slug("User's cat", words: 6), "users-cat")
        let cat = "Remember that my cat is called Marmalade and she is 7 "
            + "years old."
        XCTAssertTrue(Memories.grounded(Memories.Draft(
            title: "Cat is 7 years old",
            description: "The user's cat is 7 years old."), in: cat))
        XCTAssertFalse(Memories.grounded(Memories.Draft(
            title: "Is 63 years old",
            description: "The user is 63 years old."), in: cat))
        XCTAssertFalse(Memories.grounded(Memories.Draft(
            title: "Wants to know",
            description: "The user wants to know about the cat called "
                + "Marmalade."), in: cat))
        XCTAssertEqual(
            Memories.clauses("What is my cat called, and what do I do for "
                             + "a living?"),
            ["What is my cat called", "what do I do for a living"])
        XCTAssertEqual(Memories.clauses("What is my cat called?"), [])
    }

    func testRememberWritesNotesWithProvenance() async throws {
        guard let url = BertEmbedder.bundledMultilingual,
              BertEmbedder.load(ggufPath: url.path) != nil else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("extract-" + UUID().uuidString,
                                    isDirectory: true)
        let memories = Memories(root: root)
        memories.enabled = true
        await memories.awaitOpen()
        let source = UUID()
        let said = "User: I drink two espressos of coffee before nine, never "
            + "after lunch, and green tea in the afternoon."
        let kept = await memories.remember([Memories.Draft(
            area: "person", title: "Coffee",
            description: "Drinks two espressos before nine.")],
            said: said, source: source, excluding: [])
        XCTAssertEqual(kept.map { note in note.id }, ["person/coffee"])
        let seen = await memories.remember([Memories.Draft(
            area: "person", title: "Tea", description: "Drinks tea.")],
            said: said, source: source, excluding: ["person/tea"])
        XCTAssertTrue(seen.isEmpty, "a note this chat already saw")
        let path = root.appendingPathComponent("person/coffee.md")
        var text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(text.contains("chatokf://conversation/"
                                    + source.uuidString), text)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        XCTAssertTrue(text.contains("tags: [person]"), text)
        let again = await memories.remember([Memories.Draft(
            area: "person", title: "Coffee",
            description: "Drinks three espressos before nine.",
            isPrivate: true)], said: said, source: source, excluding: [])
        XCTAssertEqual(again.map { note in note.id }, ["person/coffee"],
                       "a later extraction updates the note in place")
        text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(text.contains("three espressos"), text)
        XCTAssertTrue(text.contains("tags: [person, private]"), text)
        let allergy = await memories.remember([Memories.Draft(
            area: "health", title: "Coffee allergy",
            description: "The user gets a rash from coffee.")],
            said: said + " I get a rash from coffee, an allergy.",
            source: source, excluding: [])
        XCTAssertEqual(allergy.map { note in note.id },
                       ["health/coffee-allergy"])
        XCTAssertTrue(try String(contentsOf: root.appendingPathComponent(
            "health/coffee-allergy.md"), encoding: .utf8)
            .contains("tags: [health, private]"))
        let filed = await memories.remember([Memories.Draft(
            title: "Green tea",
            description: "Drinks green tea in the afternoon.")],
            said: said, source: source, excluding: [])
        XCTAssertEqual(filed.count, 1)
        XCTAssertTrue(filed.first?.id.hasSuffix("/green-tea") == true,
                      filed.first?.id ?? "")
        let known = Memories.extractionInstruction(
            known: [("person/coffee", "Coffee")])
        XCTAssertTrue(known.contains("Already on file: Coffee."), known)
        try? FileManager.default.removeItem(at: root)
    }

    func testAnEchoOfTheNotesOnFileIsNotRemembered() async throws {
        guard let url = BertEmbedder.bundledMultilingual,
              BertEmbedder.load(ggufPath: url.path) != nil else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("echo-" + UUID().uuidString,
                                    isDirectory: true)
        let memories = Memories(root: root)
        memories.enabled = true
        await memories.awaitOpen()
        let owned = await memories.remember(
            [Memories.Draft(area: "home", title: "Subaru Outback",
                            description: "A 2019 Subaru Outback, green.")],
            said: "I drive a 2019 Subaru Outback, green, 60k miles.",
            source: nil, excluding: [])
        XCTAssertEqual(owned.map { note in note.id }, ["home/subaru-outback"])
        let asked = "Explain dark matter and dark energy in a few sentences."
        let echoed = await memories.remember(
            [Memories.Draft(area: "home", title: "Subaru",
                            description: "The user owns a Subaru.")],
            said: asked, source: nil, excluding: [])
        XCTAssertTrue(echoed.isEmpty)
        XCTAssertEqual(
            memories.note("home/subaru-outback")?.text.contains("2019"), true)
        try? FileManager.default.removeItem(at: root)
    }
}

@MainActor final class ConversationNoteTests: XCTestCase {

    func testOnlyShortRealPromptsAreKept() {
        let long = String(repeating: "word ", count: 60)
        let sample = "Tell me a story about this picture"
        let asked = ConversationNote.asked(
            ["Hi there.", "What should  I cook\ntonight?", long, sample,
             "How long do I roast a chicken?"], samples: [sample])
        XCTAssertEqual(asked, ["What should I cook tonight?",
                               "How long do I roast a chicken?"])
        let many = (1...20).map { n in "question number \(n) please" }
        XCTAssertEqual(ConversationNote.asked(many, samples: []).count, 12)
        XCTAssertEqual(ConversationNote.asked(many, samples: []).last,
                       "question number 20 please")
    }

    func testAConclusionIsKeptOnlyWhenTheExchangeBearsItOut() {
        let exchange = "How much is 1000 at 5 percent after 22 years?\n"
            + "After 22 years the balance is $2,925.26. Canberra is far."
        XCTAssertEqual(
            ConversationNote.concluded(
                "```\nThe balance after 22 years is $2,925.26.\n```",
                in: exchange),
            "The balance after 22 years is $2,925.26.")
        XCTAssertEqual(ConversationNote.concluded(
            "The balance after 22 years is $3,100.", in: exchange), "")
        XCTAssertEqual(ConversationNote.concluded(
            "The answer came from Sydney.", in: exchange), "")
        XCTAssertEqual(ConversationNote.concluded(
            "Shall we go on?", in: exchange), "")
        XCTAssertEqual(ConversationNote.concluded("", in: exchange), "")
        XCTAssertEqual(ConversationNote.concluded(
            "I have noted that the balance is $2,925.26.", in: exchange), "")
        XCTAssertEqual(ConversationNote.concluded("391", in: "391"), "")
        XCTAssertEqual(ConversationNote.concluded(
            "How much is 1000 at 5 percent after 22 years?", in: exchange,
            asked: ["How much is 1000 at 5 percent after 22 years?"]), "")
        XCTAssertEqual(ConversationNote.concluded(
            "Canberra is far away", in: exchange,
            asked: ["Canberra is far away, is it not?"]), "")
    }

    func testTheNoteIsRewrittenInPlaceAndStaysOutOfItsOwnChat() async throws {
        guard let url = BertEmbedder.bundledMultilingual,
              BertEmbedder.load(ggufPath: url.path) != nil else {
            throw XCTSkip("no bundled e5-small.gguf")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("convnote-" + UUID().uuidString,
                                    isDirectory: true)
        let memories = Memories(root: root)
        memories.enabled = true
        await memories.awaitOpen()
        let chat = UUID()
        let id = ConversationNote.id(chat)
        let first = ConversationNote(
            conversation: chat, title: "Roast chicken",
            asked: ["How long do I roast a chicken?"], exchange: "",
            concludes: false)
        await memories.keep(first, title: "Roast chicken", concluded: "")
        XCTAssertEqual(memories.list.map { row in row.id }, [id])
        XCTAssertEqual(memories.list.first?.source, chat)
        let second = ConversationNote(
            conversation: chat, title: "Roast chicken",
            asked: ["How long do I roast a chicken?",
                    "And at what temperature?"], exchange: "",
            concludes: true)
        await memories.keep(second, title: "Roasting a chicken",
                            concluded: "About 90 minutes at 200 degrees.")
        XCTAssertEqual(memories.list.count, 1, "one note per conversation")
        let text = try String(
            contentsOf: root.appendingPathComponent(id + ".md"),
            encoding: .utf8)
        XCTAssertTrue(text.contains("title: Roasting a chicken"), text)
        XCTAssertTrue(text.contains("- And at what temperature?"), text)
        XCTAssertTrue(text.contains("Concluded:\n- About 90 minutes"), text)
        XCTAssertTrue(text.contains("Asked: How long do I roast"), text)
        XCTAssertTrue(text.contains("temperature? Concluded: About 90"), text)
        await memories.keep(second, title: "Roasting a chicken",
                            concluded: "Rest it for ten minutes.")
        let later = try String(
            contentsOf: root.appendingPathComponent(id + ".md"),
            encoding: .utf8)
        XCTAssertTrue(later.contains("- About 90 minutes at 200 degrees.\n"
                                     + "- Rest it for ten minutes."), later)
        XCTAssertEqual(text.components(separatedBy: "generated:").count, 2)
        let elsewhere = await memories.recall(
            "how long to roast a chicken", also: [], pp: 400, excluding: [])
        XCTAssertEqual(elsewhere?.ids, [id])
        let within = await memories.recall(
            "how long to roast a chicken", also: [], pp: 400, excluding: [id])
        XCTAssertNil(within)
        let covered = await memories.coverage("How long do I roast a chicken?")
        XCTAssertFalse(covered.covered,
                       "a conversation note never stands in for a fact")
        await memories.trash(id)
        XCTAssertTrue(memories.list.isEmpty)
        try? FileManager.default.removeItem(at: root)
    }
}
