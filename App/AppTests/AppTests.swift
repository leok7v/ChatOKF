import Chat
import XCTest
@testable import ChatOKF

final class ModelNamingTests: XCTestCase {

    private let both = ["gemma-4-12B", "gemma-4-12B-MTP"]
    private let one = ["gemma-4-12B-MTP", "Ternary-Bonsai-1.7B"]

    func testMtpIsHiddenWhenNothingSharesTheFamily() {
        XCTAssertEqual(Models.display("gemma-4-12B-MTP", among: one),
                       "Gemma 12B")
    }

    func testMtpAppearsWhenBothBuildsAreVisible() {
        XCTAssertEqual(Models.display("gemma-4-12B", among: both),
                       "Gemma 12B")
        XCTAssertEqual(Models.display("gemma-4-12B-MTP", among: both),
                       "Gemma 12B MTP")
    }

    func testQuantSuffixOnlyWhenRungsCollide() {
        let rungs = ["Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ4_XS"]
        XCTAssertEqual(Models.display("Qwen3.8-27B-IQ1_S", among: rungs),
                       "Qwen3.8 27B 1-bit")
        XCTAssertEqual(Models.display("Qwen3.8-27B-IQ1_S",
                                      among: ["Qwen3.8-27B-IQ1_S"]),
                       "Qwen3.8 27B")
    }

    func testProseNameNeverCarriesATag() {
        XCTAssertEqual(Models.display("gemma-4-E2B-MTP"), "Gemma E2B")
        XCTAssertEqual(Models.display("gemma-4-E4B-MTP"), "Gemma E4B")
    }

}

final class AttachmentRefsTests: XCTestCase {

    func testInsertThenStripLeavesThePlainText() {
        let r = AttachmentRefs.insert("a.png", into: "look", at: 4)
        XCTAssertTrue(r.text.contains("a.png"))
        XCTAssertEqual(AttachmentRefs.names(in: r.text), ["a.png"])
        XCTAssertEqual(
            AttachmentRefs.stripped(r.text)
                .trimmingCharacters(in: .whitespaces),
            "look @a.png")
    }

    func testScrubRemovesTheWholeToken() {
        let r = AttachmentRefs.insert("a.png", into: "", at: 0)
        XCTAssertEqual(
            AttachmentRefs.scrub("a.png", from: r.text)
                .trimmingCharacters(in: .whitespaces), "")
    }

    func testSubstituteReplacesByName() {
        let r = AttachmentRefs.insert("doc.md", into: "read", at: 4)
        let out = AttachmentRefs.substitute(r.text) { name in
            "<\(name)>"
        }
        XCTAssertTrue(out.contains("<doc.md>"))
    }

}

@MainActor final class ZoomTests: XCTestCase {

    func testZoomClampsToTheDeclaredLimit() {
        XCTAssertEqual(ChatModel.clampZoom(9), ChatModel.zoomLimit)
        XCTAssertEqual(ChatModel.clampZoom(-9), -ChatModel.zoomLimit)
    }

    func testNotchRoundTripsThroughScale() {
        for notch in -ChatModel.zoomLimit ... ChatModel.zoomLimit {
            let scale = ChatModel.zoomScale(notch)
            XCTAssertEqual(ChatModel.zoomNotch(nearest: scale), notch)
        }
    }

}

@MainActor final class StoredMessageTests: XCTestCase {

    private func msg(posters: [Data]?) -> ConversationStore.Msg {
        ConversationStore.Msg(
            fromUser: true, text: "watch this", reasoning: "",
            rounds: [], images: [], loopStopped: false, clips: nil,
            docs: nil, posters: posters)
    }

    func testAPosterSurvivesTheRoundTrip() throws {
        let frame = Data([0xFF, 0xD8, 0xFF])
        let data = try JSONEncoder().encode(msg(posters: [frame]))
        let back = try JSONDecoder().decode(ConversationStore.Msg.self,
                                            from: data)
        XCTAssertEqual(back.posters, [frame])
    }

    func testARecordWrittenBeforePostersStillDecodes() throws {
        let json = "{\"fromUser\":true,\"text\":\"watch this\","
            + "\"reasoning\":\"\",\"rounds\":[],\"images\":[],"
            + "\"loopStopped\":false}"
        let back = try JSONDecoder().decode(ConversationStore.Msg.self,
                                            from: Data(json.utf8))
        XCTAssertNil(back.posters)
        XCTAssertEqual(back.text, "watch this")
    }

}

@MainActor final class ConversationSearchTests: XCTestCase {

    func testShortQueryIsNotActive() {
        XCTAssertFalse(ConversationSearch.active("a"))
        XCTAssertTrue(ConversationSearch.active("ab"))
    }

    func testRankingRequiresEveryWordToLand() {
        let id = UUID()
        let convo = ConversationStore.Convo(
            id: id, title: "Harvest report", created: Date(),
            updated: Date(), messages: [])
        let index = [id: ["harvest": 5, "report": 5]]
        XCTAssertEqual(
            ConversationSearch.rank([convo], index, "harvest").count, 1)
        XCTAssertEqual(
            ConversationSearch.rank([convo], index, "harvest wombat").count,
            0)
    }

}
