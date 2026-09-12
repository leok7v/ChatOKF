import XCTest
@testable import LLM

final class TokenizerTests: XCTestCase {
    // The set sits either directly under models/Qwen3.5-0.8B (a dev copy) or
    // in a {sha} subdir (the HubFetch staging layout).
    private func modelsDir() -> URL? {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: URL? = nil
        if fm.fileExists(
            atPath: base.appendingPathComponent("tokenizer.json").path) {
            found = base
        } else if let subs = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) {
            for sub in subs where fm.fileExists(
                atPath: sub.appendingPathComponent("tokenizer.json").path) {
                found = sub
            }
        }
        return found
    }

    func testParityWithReference() throws {
        guard let dir = modelsDir() else {
            throw XCTSkip("models/ not present")
        }
        let tok = try Tokenizer(modelsDir: dir)
        let text = "The ISDA Master Agreement is the most widely "
            + "used, don't you think? <|im_start|>user"
        let expected: [Int32] = [
            760, 3314, 6151, 10507, 21800, 369, 279, 1379, 13177, 1429,
            11, 1459, 914, 488, 1683, 30, 220, 248045, 846,
        ]
        XCTAssertEqual(tok.encode(text, addSpecial: true), expected)
        XCTAssertEqual(tok.decode(expected), text)
    }
}
