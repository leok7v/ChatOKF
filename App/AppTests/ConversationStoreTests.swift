import CoreGraphics
import XCTest
@testable import Chat
import LLM

@MainActor final class ConversationStoreTests: XCTestCase {

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-" + UUID().uuidString,
                                    isDirectory: true)
    }

    private func square() throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        return try XCTUnwrap(ctx.makeImage())
    }

    private func exchange(_ image: CGImage?) -> [Message] {
        var asked = Message(fromUser: true, text: "look at this red square")
        if let image { asked.images = [image] }
        var answer = Message(fromUser: false, text: "A small red **square**.")
        answer.reasoning = "It is red."
        return [asked, answer]
    }

    private func files(_ root: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: root.path))
            ?? []).filter { name in name.hasSuffix(".json") }.sorted()
    }

    func testCommitOpenTrashRestoreAndForever() async throws {
        let root = temporaryRoot()
        let store = ConversationStore(root: root)
        await store.loaded()
        XCTAssertTrue(store.list.isEmpty)
        let id = UUID()
        store.commit(id: id, title: nil, fallbackTitle: "Red square",
                     messages: exchange(try square()), trace: [],
                     extracted: nil, followup: "How big is it?")
        await store.settled()
        XCTAssertEqual(store.list.map { c in c.id }, [id])
        XCTAssertEqual(store.list.first?.title, "Red square")
        XCTAssertEqual(store.list.first?.followup, "How big is it?")
        XCTAssertEqual(store.words[id]?["square"], 2 + 5)
        XCTAssertEqual(files(root), [id.uuidString + ".json"])
        let restored = await store.open(id)
        let opened = try XCTUnwrap(restored)
        XCTAssertEqual(opened.followup, "How big is it?")
        XCTAssertEqual(opened.messages.count, 2)
        XCTAssertEqual(opened.messages[0].images.count, 1)
        XCTAssertEqual(opened.messages[0].images[0].width, 8)
        XCTAssertFalse(opened.messages[1].answerDoc.items.isEmpty)
        XCTAssertFalse(opened.messages[1].reasoningDoc.items.isEmpty)
        let created = try XCTUnwrap(store.list.first?.created)
        store.commit(id: id, title: "Renamed", fallbackTitle: "x",
                     messages: opened.messages, trace: [], extracted: nil)
        await store.settled()
        XCTAssertEqual(store.list.first?.title, "Renamed")
        XCTAssertEqual(store.list.first?.created, created)
        XCTAssertEqual(store.list.count, 1)
        await store.trash(id)
        XCTAssertTrue(store.list.isEmpty)
        XCTAssertEqual(store.trashed.map { c in c.id }, [id])
        XCTAssertNotNil(store.trashed.first?.trashedAt)
        XCTAssertNil(store.words[id])
        XCTAssertTrue(files(root).isEmpty)
        let live = await store.load(id)
        XCTAssertNil(live)
        let binned = await store.loadTrashed(id)
        XCTAssertNotNil(binned)
        let reopened = await store.open(id)
        XCTAssertNotNil(reopened, "a trashed chat still opens")
        await store.restore(id)
        XCTAssertEqual(store.list.map { c in c.id }, [id])
        XCTAssertNil(store.list.first?.trashedAt)
        XCTAssertTrue(store.trashed.isEmpty)
        XCTAssertEqual(store.words[id]?["square"], 2)
        await store.trash(id)
        await store.deleteForever(id)
        XCTAssertTrue(store.trashed.isEmpty)
        XCTAssertTrue(files(root.appendingPathComponent("trash")).isEmpty)
        try? FileManager.default.removeItem(at: root)
    }

    func testTrashAllEmptyTrashAndAReloadFromDisk() async throws {
        let root = temporaryRoot()
        let store = ConversationStore(root: root)
        await store.loaded()
        let a = UUID()
        let b = UUID()
        store.commit(id: a, title: "A", fallbackTitle: "",
                     messages: exchange(nil), trace: [], extracted: nil)
        store.commit(id: b, title: "B", fallbackTitle: "",
                     messages: exchange(nil), trace: [], extracted: nil)
        await store.settled()
        XCTAssertEqual(store.list.count, 2)
        let again = ConversationStore(root: root)
        await again.loaded()
        XCTAssertEqual(Set(again.list.map { c in c.id }), [a, b])
        XCTAssertEqual(again.words[a]?["square"], 2)
        await store.trashAll()
        XCTAssertTrue(store.list.isEmpty)
        XCTAssertEqual(store.trashed.count, 2)
        let third = ConversationStore(root: root)
        await third.loaded()
        XCTAssertEqual(third.trashed.count, 2)
        await store.emptyTrash()
        XCTAssertTrue(store.trashed.isEmpty)
        XCTAssertTrue(files(root.appendingPathComponent("trash")).isEmpty)
        try? FileManager.default.removeItem(at: root)
    }

    func testAnOlderOpenIsSupersededByANewerOne() async throws {
        let root = temporaryRoot()
        let store = ConversationStore(root: root)
        await store.loaded()
        let id = UUID()
        store.commit(id: id, title: "A", fallbackTitle: "",
                     messages: exchange(nil), trace: [], extracted: nil)
        await store.settled()
        let older = store.files.opens.take()
        let newer = store.files.opens.take()
        let skipped = await store.files.open(id, ticket: older)
        XCTAssertNil(skipped)
        let served = await store.files.open(id, ticket: newer)
        XCTAssertEqual(served?.messages.count, 2)
        try? FileManager.default.removeItem(at: root)
    }

}
