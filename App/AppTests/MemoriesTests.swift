import XCTest
@testable import Chat
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
        let memories = Memories(root: root)
        memories.enabled = true
        return memories
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

    func testSmallStoreRecallsWithoutTheFloor() async throws {
        let memories = try seeded()
        await memories.awaitOpen()
        let hit = memories.recall("Deco X55", also: [], pp: 400,
                                  excluding: [])
        XCTAssertEqual(hit?.ids.first, "tech/wifi-mesh")
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
        XCTAssertTrue(made.contains("draft"), made)
        let text = try String(contentsOf: m.root.appendingPathComponent(
            "garden/basil.md"), encoding: .utf8)
        XCTAssertTrue(text.contains("status: draft"), text)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        XCTAssertTrue(text.contains("tags: [garden, private]"), text)
        XCTAssertEqual(m.takeDrafts().map { note in note.id },
                       ["garden/basil"])
        XCTAssertTrue(m.takeDrafts().isEmpty)
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

    func testUpdateOfAKeptNoteIsADraftAndForgetRestoresIt() async throws {
        let m = try await opened()
        let path = m.root.appendingPathComponent("tech/wifi-mesh.md")
        m.confirm("tech/wifi-mesh")
        let kept = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(kept.contains("verified: { by: human:user"), kept)
        let runner = MemoryToolRunner(inner: SafeToolRunner(), memories: m)
        let updated = await runner.execute("memory_update", [
            ToolArg(name: "id", value: "tech/wifi-mesh"),
            ToolArg(name: "type", value: "Note"),
            ToolArg(name: "title", value: "The mesh wifi setup"),
            ToolArg(name: "description", value: "Four Deco X55 units."),
            ToolArg(name: "body", value: "One per floor and one in the "
                    + "garage.")])
        XCTAssertTrue(updated.contains("updated tech/wifi-mesh"), updated)
        var text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(text.contains("status: draft"), text)
        XCTAssertFalse(text.contains("verified:"), text)
        XCTAssertEqual(text.components(separatedBy: "generated:").count, 2)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        XCTAssertEqual(m.takeDrafts().map { note in note.id },
                       ["tech/wifi-mesh"])
        m.discard("tech/wifi-mesh")
        text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertEqual(text, kept)
        XCTAssertNotNil(m.note("tech/wifi-mesh"))
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
        let bare = "person/joe\ntype: Note\ntitle: Indoor preference\n"
            + "description: Joe prefers being indoors.\ntags: \n"
            + "body: Joe is generally more of an indoor person.\n```"
        let loose = Memories.parseDrafts(bare)
        XCTAssertEqual(loose.map { d in d.id }, ["person/joe"])
        XCTAssertEqual(loose.first?.body,
                       "Joe is generally more of an indoor person.")
        XCTAssertEqual(loose.first?.tags, [])
    }

    func testRememberWritesDraftsWithProvenanceAndKeepPromotes() async throws {
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
        let kept = memories.remember([Memories.Draft(
            id: "person/coffee", type: "Note", title: "Coffee",
            description: "Drinks two espressos before nine.", tags: [],
            body: "Never after lunch.")], source: source, excluding: [])
        XCTAssertEqual(kept.map { note in note.id }, ["person/coffee"])
        let seen = memories.remember([Memories.Draft(
            id: "person/tea", type: "Note", title: "Tea",
            description: "Drinks tea.", tags: [], body: "Green.")],
            source: source, excluding: ["person/tea"])
        XCTAssertTrue(seen.isEmpty, "a note this chat already saw")
        let path = root.appendingPathComponent("person/coffee.md")
        var text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertTrue(text.contains("status: draft"), text)
        XCTAssertTrue(text.contains("chatokf://conversation/"
                                    + source.uuidString), text)
        memories.confirm("person/coffee")
        text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertFalse(text.contains("status: draft"), text)
        XCTAssertTrue(text.contains("verified: { by: human:user"), text)
        XCTAssertTrue(text.contains("generated: { by: model"), text)
        let again = memories.remember([Memories.Draft(
            id: "person/coffee", type: "Note", title: "Coffee again",
            description: "Restated.", tags: [], body: "Restated.")],
            source: source, excluding: [])
        XCTAssertTrue(again.isEmpty, "a kept note is never re-drafted")
        memories.confirm("person/coffee")
        text = try String(contentsOf: path, encoding: .utf8)
        XCTAssertEqual(text.components(separatedBy: "verified:").count, 2)
        XCTAssertTrue(text.contains("title: Coffee\n"), text)
        let restated = memories.remember([Memories.Draft(
            id: "person/morning-coffee", type: "Note",
            title: "Morning coffee",
            description: "Drinks two espressos before nine each morning.",
            tags: [], body: "Never after lunch.")], source: source,
            excluding: [])
        XCTAssertTrue(restated.isEmpty, "a restated kept note is not new")
        XCTAssertNil(memories.note("person/morning-coffee"))
        let tea = memories.remember([Memories.Draft(
            id: "person/tea", type: "Note", title: "Tea",
            description: "Drinks tea.", tags: [], body: "Green.")],
            source: source, excluding: [])
        XCTAssertEqual(tea.map { note in note.id }, ["person/tea"])
        let moreTea = memories.remember([Memories.Draft(
            id: "person/afternoon-tea", type: "Note", title: "Tea",
            description: "Drinks green tea.", tags: [], body: "Green.")],
            source: source, excluding: [])
        XCTAssertEqual(moreTea.map { note in note.id }, ["person/tea"],
                       "a restated draft is updated under its own id")
        XCTAssertNil(memories.note("person/afternoon-tea"))
        let known = Memories.extractionInstruction(
            known: [("person/coffee", "Coffee")])
        XCTAssertTrue(known.contains("person/coffee (Coffee)"), known)
        XCTAssertEqual(memories.coverage("espresso before nine").known.first?
                           .id, "person/coffee")
        memories.discard("person/coffee")
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
        try? FileManager.default.removeItem(at: root)
    }
}
