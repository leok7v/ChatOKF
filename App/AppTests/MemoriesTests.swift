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
            let got = memories.recall(question, also: [], pp: 400,
                                      excluding: [])
            XCTAssertNil(got, question + " -> " + (got?.block ?? ""))
        }
        let dog = memories.recall("what is my dog called", also: [],
                                  pp: 400, excluding: [])
        XCTAssertEqual(dog?.ids, ["person/pet-name"])
        XCTAssertTrue(dog?.block.contains("Biscuit") == true, dog?.block ?? "")
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testRecallFindsTheNoteAndSkipsWhatWasRead() async throws {
        let memories = try seeded()
        XCTAssertNil(memories.recall("basil", also: [], pp: 400,
                                     excluding: []))
        await memories.awaitOpen()
        XCTAssertTrue(memories.isOpen)
        let first = memories.recall("why does my basement smell musty",
                                    also: [], pp: 400, excluding: [])
        XCTAssertEqual(first?.ids.first, "house/basement-humidity")
        XCTAssertTrue(first?.block.contains("musty smell") == true)
        XCTAssertGreaterThan(first?.tokens ?? 0, 20)
        let again = memories.recall("why does my basement smell musty",
                                    also: [], pp: 400,
                                    excluding: ["house/basement-humidity",
                                                "garden/basil",
                                                "tech/wifi-mesh"])
        XCTAssertNil(again)
        XCTAssertTrue(memories.map.contains("house/"))
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testAVerbatimModelNumberRecallsOnASmallStore() async throws {
        let memories = try seeded()
        await memories.awaitOpen()
        let hit = memories.recall("Deco X55", also: [], pp: 400,
                                  excluding: [])
        XCTAssertEqual(hit?.ids, ["tech/wifi-mesh"])
        try? FileManager.default.removeItem(at: memories.root)
    }

    func testSwitchOffRecallsNothing() async throws {
        let memories = try seeded()
        memories.enabled = false
        await memories.awaitOpen()
        XCTAssertFalse(memories.isOpen)
        XCTAssertNil(memories.recall("basil", also: [], pp: 400,
                                     excluding: []))
        XCTAssertEqual(memories.map, "")
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
        XCTAssertFalse(found.contains("[weak"), found)
        XCTAssertFalse(found.contains("[unrelated]"), found)
        let foreign = await runner.execute("memory_search", [
            ToolArg(name: "query",
                    value: "explain dark matter and dark energy")])
        XCTAssertTrue(foreign.hasPrefix("[weak"), foreign)
        XCTAssertTrue(foreign.contains("[unrelated]"), foreign)
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
        XCTAssertEqual(m.store?.concept("house/wiring")?.links,
                       ["tech/wifi-mesh"])
        m.trash("tech/wifi-mesh")
        XCTAssertEqual(m.list.map { row in row.id }, ["house/wiring"])
        XCTAssertEqual(m.trashed.map { row in row.id }, ["tech/wifi-mesh"])
        XCTAssertNil(m.note("tech/wifi-mesh"))
        XCTAssertFalse(m.recall("Deco X55", also: [], pp: 400,
                                excluding: [])?.ids
            .contains("tech/wifi-mesh") ?? false)
        var text = try String(contentsOf: wiring, encoding: .utf8)
        XCTAssertTrue(text.contains("feeds the mesh setup upstairs."), text)
        XCTAssertFalse(text.contains("wifi-mesh.md"), text)
        XCTAssertEqual(m.store?.concept("house/wiring")?.links, [])
        m.restore("tech/wifi-mesh")
        XCTAssertEqual(m.trashed.count, 0)
        XCTAssertNotNil(m.note("tech/wifi-mesh"))
        text = try String(contentsOf: wiring, encoding: .utf8)
        XCTAssertTrue(
            text.contains("[the mesh setup](/tech/wifi-mesh.md)"), text)
        XCTAssertEqual(m.store?.concept("house/wiring")?.links,
                       ["tech/wifi-mesh"])
        try? FileManager.default.removeItem(at: m.root)
    }

    func testDeleteForeverAndEmptyTrashLeaveNoFile() async throws {
        let m = try await opened()
        m.trash("tech/wifi-mesh")
        let kept = m.root.appendingPathComponent(
            Memories.trashFolder + "/tech/wifi-mesh.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: kept.path))
        m.deleteForever("tech/wifi-mesh")
        XCTAssertFalse(FileManager.default.fileExists(atPath: kept.path))
        XCTAssertEqual(m.trashed.count, 0)
        m.trash("house/wiring")
        XCTAssertEqual(m.trashed.count, 1)
        m.emptyTrash()
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
        m.trash("tech/wifi-mesh")
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

    func testParseDraftsAcceptsTheFormAndRejectsTheRest() {
        let raw = """
        ```text
        ### house/basement-humidity
        type: Problem
        title: Basement humidity
        description: The basement runs at sixty-five percent in summer.
        tags: house, private
        body:
        A dehumidifier drains to the sump.
        Regrade the north side in spring.
        ### Bad Id Here
        title: nope
        description: nope
        body:
        x
        ### person/coffee
        type: Note
        title: Coffee
        description: Drinks two espressos before nine.
        body:
        Never after lunch.
        ```
        """
        let drafts = Memories.parseDrafts(raw)
        XCTAssertEqual(drafts.map { d in d.id },
                       ["house/basement-humidity", "person/coffee"])
        XCTAssertEqual(drafts[0].tags, ["house", "private"])
        XCTAssertTrue(drafts[0].body.hasSuffix("in spring."))
        XCTAssertEqual(drafts[1].type, "Note")
        XCTAssertEqual(Memories.parseDrafts("NONE\n```").count, 0)
        let bare = "### cosmology\ntype: Note\ntitle: Dark Matter and Dark "
            + "Energy\ndescription: What was asked.\nbody:\nAsked.\n"
            + "### travel plans\ntitle: Kyoto Trip\ndescription: A plan.\n"
            + "body:\nLate November.\n"
            + "### interest/dark energy\ntitle: Dark energy\n"
            + "description: What was asked.\nbody:\nAsked.\n```"
        XCTAssertEqual(Memories.parseDrafts(bare).map { d in d.id },
                       ["cosmology/dark-matter-and-dark-energy",
                        "travel-plans/kyoto-trip", "interest/dark-energy"])
        let unmarked = "person/joe\ntype: Note\ntitle: Indoor preference\n"
            + "description: Joe prefers being indoors.\ntags: \n"
            + "body: Joe is generally more of an indoor person.\n```"
        let loose = Memories.parseDrafts(unmarked)
        XCTAssertEqual(loose.map { d in d.id }, ["person/joe"])
        XCTAssertEqual(loose.first?.body,
                       "Joe is generally more of an indoor person.")
        XCTAssertEqual(loose.first?.tags, [])
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
        let kept = memories.remember([Memories.Draft(
            id: "person/coffee", type: "Note", title: "Coffee",
            description: "Drinks two espressos before nine.", tags: [],
            body: "Never after lunch.")], said: said, source: source,
            excluding: [])
        XCTAssertEqual(kept.map { note in note.id }, ["person/coffee"])
        let seen = memories.remember([Memories.Draft(
            id: "person/tea", type: "Note", title: "Tea",
            description: "Drinks tea.", tags: [], body: "Green.")],
            said: said, source: source, excluding: ["person/tea"])
        XCTAssertTrue(seen.isEmpty, "a note this chat already saw")
        let path = root.appendingPathComponent("person/coffee.md")
        var text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertFalse(text.contains("status: draft"), text)
        XCTAssertTrue(text.contains("chatokf://conversation/"
                                    + source.uuidString), text)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        let again = memories.remember([Memories.Draft(
            id: "person/coffee", type: "Note", title: "Coffee again",
            description: "Restated.", tags: [], body: "Restated.")],
            said: said, source: source, excluding: [])
        XCTAssertEqual(again.map { note in note.id }, ["person/coffee"],
                       "a later extraction updates the note in place")
        text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(text.contains("title: Coffee again\n"), text)
        let restated = memories.remember([Memories.Draft(
            id: "person/morning-coffee", type: "Note",
            title: "Morning coffee",
            description: "Restated.",
            tags: [], body: "Restated.")], said: said, source: source,
            excluding: [])
        XCTAssertEqual(restated.map { note in note.id }, ["person/coffee"],
                       "a restated note lands under the id on file")
        XCTAssertNil(memories.note("person/morning-coffee"))
        let tea = memories.remember([Memories.Draft(
            id: "person/tea", type: "Note", title: "Tea",
            description: "Drinks tea.", tags: [], body: "Green.")],
            said: said, source: source, excluding: [])
        XCTAssertEqual(tea.map { note in note.id }, ["person/tea"])
        let moreTea = memories.remember([Memories.Draft(
            id: "person/afternoon-tea", type: "Note", title: "Tea",
            description: "Drinks green tea.", tags: [], body: "Green.")],
            said: said, source: source, excluding: [])
        XCTAssertEqual(moreTea.map { note in note.id }, ["person/tea"],
                       "a restated draft is updated under its own id")
        XCTAssertNil(memories.note("person/afternoon-tea"))
        let known = Memories.extractionInstruction(
            known: [("person/coffee", "Coffee")])
        XCTAssertTrue(known.contains("person/coffee (Coffee)"), known)
        XCTAssertEqual(memories.coverage("Restated").known.first?.id,
                       "person/coffee")
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
        let car = Memories.Draft(
            id: "car/subaru", type: "Note", title: "The car",
            description: "A 2019 Subaru Outback, green, 60k miles.", tags: [],
            body: "Serviced at the dealer every 10k.")
        let owned = memories.remember(
            [car], said: "I drive a 2019 Subaru Outback, green, 60k "
                + "miles, serviced at the dealer every 10k.",
            source: nil, excluding: [])
        XCTAssertEqual(owned.map { note in note.id }, ["car/subaru"])
        let asked = "Explain dark matter and dark energy in a few sentences."
        let cosmos = "User: " + asked + "\n\nAssistant: Dark matter is "
            + "inferred from gravity; dark energy drives the accelerating "
            + "expansion."
        let echo = Memories.Draft(
            id: "car/subaru", type: "Note", title: "Subaru",
            description: "The user owns a Subaru.", tags: ["private"],
            body: "This is a known possession.")
        XCTAssertFalse(Memories.grounded(echo, in: asked))
        XCTAssertTrue(memories.remember([echo], said: asked, source: nil,
                                        excluding: []).isEmpty)
        XCTAssertEqual(memories.note("car/subaru")?.text.contains("Outback"),
                       true)
        let interest = Memories.Draft(
            id: "interest/dark-matter", type: "Note", title: "Dark matter",
            description: "The user asked how dark matter differs from dark "
                + "energy.", tags: [], body: "Asked on 2026-09-21.")
        XCTAssertTrue(Memories.grounded(interest, in: asked))
        XCTAssertTrue(memories.coverage(cosmos).known.isEmpty,
                      "an unrelated note is not offered as already on file")
        let listed = Memories.Draft(
            id: "interest/garden", type: "Note", title: "Joe's garden",
            description: "Joe is interested in herbs, tomatoes and basil.",
            tags: ["garden"], body: "Listed from the map.")
        let listing = "List everything we ever talked about that you "
            + "remember with short titles"
        XCTAssertFalse(Memories.grounded(listed, in: listing),
                       "a draft drawn from the answer, not the user, drops")
        XCTAssertTrue(memories.remember([listed], said: listing, source: nil,
                                        excluding: []).isEmpty)
        XCTAssertNil(memories.note("interest/garden"))
        try? FileManager.default.removeItem(at: root)
    }
}
