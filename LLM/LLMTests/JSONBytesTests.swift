import XCTest
@testable import LLM

final class JSONBytesTests: XCTestCase {

    private struct Meta: Codable, Equatable {
        let stamp: String
        let committed: [Int32]
        let attachments: [String: Int]
    }

    private static let sample = Meta(
        stamp: "abc", committed: [1, 2, 3],
        attachments: ["Picture": 2, "Audio": 1, "Video": 3, "Doc": 4,
                      "Clip": 5, "Frame": 6])

    func testTheSameValueGivesTheSameBytesEveryTime() throws {
        var seen: Set<Data> = []
        for _ in 0..<8 {
            seen.insert(try JSONBytes.reproducible(JSONBytesTests.sample))
        }
        XCTAssertEqual(seen.count, 1)
    }

    // A cache written before the keys were sorted must still load, or every
    // precooked prefix on a device silently re-cooks.
    func testAnUnsortedBlobStillDecodes() throws {
        let loose = try JSONEncoder().encode(JSONBytesTests.sample)
        let sorted = try JSONBytes.reproducible(JSONBytesTests.sample)
        let a = try JSONDecoder().decode(Meta.self, from: loose)
        let b = try JSONDecoder().decode(Meta.self, from: sorted)
        XCTAssertEqual(a, JSONBytesTests.sample)
        XCTAssertEqual(b, JSONBytesTests.sample)
    }
}
