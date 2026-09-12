import XCTest
@testable import LLM

final class StopIdsTests: XCTestCase {

    private func write(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gc-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testAListIsReadWhole() throws {
        let url = try write("{\"eos_token_id\": [248046, 248044]}")
        XCTAssertEqual(Tokenizer.stopIds(generationConfig: url),
                       [248046, 248044])
    }

    func testAScalarStillWorks() throws {
        let url = try write("{\"eos_token_id\": 248046}")
        XCTAssertEqual(Tokenizer.stopIds(generationConfig: url), [248046])
    }

    func testAbsentOrUnreadableIsEmpty() throws {
        let url = try write("{\"temperature\": 0.7}")
        XCTAssertEqual(Tokenizer.stopIds(generationConfig: url), [])
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-\(UUID().uuidString).json")
        XCTAssertEqual(Tokenizer.stopIds(generationConfig: missing), [])
    }

    func testTheShippedQwen35CardListsBothStops() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "scripts/convert/qwen35/generation_config.4b.json")
        let ids = Tokenizer.stopIds(generationConfig: url)
        if FileManager.default.fileExists(atPath: url.path) {
            XCTAssertTrue(ids.contains(248046), "im_end missing: \(ids)")
            XCTAssertTrue(ids.contains(248044), "endoftext missing: \(ids)")
        } else {
            throw XCTSkip("the emit card lives under scripts/, which the "
                + "public tree does not ship")
        }
    }

    func testTheEmbeddedTextReadsTheSameAsTheFile() throws {
        let json = "{\"eos_token_id\": [248046, 248044]}"
        let url = try write(json)
        XCTAssertEqual(Tokenizer.stopIds(generationConfigText: json),
                       Tokenizer.stopIds(generationConfig: url))
    }

    func testAbsentEmbeddedTextIsEmptyNotACrash() throws {
        XCTAssertEqual(Tokenizer.stopIds(generationConfigText: nil), [])
        XCTAssertEqual(Tokenizer.stopIds(generationConfigText: "not json"), [])
    }

    func testAddStopsUnionsAndIgnoresNegatives() throws {
        let url = try write("{\"eos_token_id\": [7, 9]}")
        var got = Set<Int32>([7])
        for id in Tokenizer.stopIds(generationConfig: url) where id >= 0 {
            got.insert(id)
        }
        XCTAssertEqual(got, [7, 9])
    }
}
