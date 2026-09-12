import Foundation
import Testing
@testable import LLM

// The Swift byte-level BPE + special-token path (Tokenizer.encode) is the one
private func firstDivergence(_ a: [Int32], _ b: [Int32]) -> Int? {
    let n = min(a.count, b.count)
    var i = 0
    while i < n && a[i] == b[i] {
        i += 1
    }
    var result: Int? = nil
    if i < n {
        result = i
    } else if a.count != b.count {
        result = n
    }
    return result
}

struct TokenizerGateTests {
    static let system = "You are a helpful assistant."
    static let user = "Hello, 世界! Don't 🌍 test123."
    // Captured from capture_chatml_ids.py against models/ (HF tokenizer,
    // add_special_tokens=False, SYSTEM/USER above).
    static let referenceIDs: [Int32] = [
        248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
        248045, 846, 198, 9419, 11, 220, 96748, 0, 4179, 914, 10838, 234, 235,
        1228, 16, 17, 18, 13, 248046, 198, 248045, 74455, 198, 248068, 271,
        248069, 271,
    ]

    // The set holding tokenizer.json: CHATOKF_MODELS_DIR when exported, else
    // the first sha under the repo's own models/Qwen3.5-0.8B.
    static let modelsDir: URL? = {
        let fm = FileManager.default
        var dir: URL? = ProcessInfo.processInfo
            .environment["CHATOKF_MODELS_DIR"]
            .map { path in URL(fileURLWithPath: path, isDirectory: true) }
        if dir == nil {
            // Flat (a hand-made dev copy, or `hf download` of the two files)
            let base = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("models/Qwen3.5-0.8B")
            let holdsTokenizer = { (url: URL) in
                fm.fileExists(
                    atPath: url.appendingPathComponent("tokenizer.json").path)
            }
            dir = holdsTokenizer(base) ? base
                : ((try? fm.contentsOfDirectory(atPath: base.path)) ?? [])
                    .map { sha in base.appendingPathComponent(sha) }
                    .first(where: holdsTokenizer)
        }
        let present = dir.map { url in
            fm.fileExists(
                atPath: url.appendingPathComponent("tokenizer.json").path)
        }
        return present == true ? dir : nil
    }()

    static var haveTokenizer: Bool { modelsDir != nil }

    static func tokenizer() throws -> Tokenizer {
        try Tokenizer(modelsDir: modelsDir ?? URL(fileURLWithPath: "."))
    }

    // Byte-level BPE is a bijection on valid UTF-8, so decode(encode(x)) == x
    // is a hard invariant.
    @Test(.enabled(if: TokenizerGateTests.haveTokenizer))
    func plainTextRoundTrip() throws {
        let tok = try Self.tokenizer()
        let samples = [
            "Hello, world!",
            "café naïve résumé",
            "日本語のテスト 你好",
            "emoji 🌍 👩‍👩‍👧‍👦 🇯🇵",
            "digits 001234567890 don't can't",
            "tabs\tnewlines\n\n  runs   of spaces",
            "mixed <think> inside </think> text",
        ]
        for s in samples {
            let ids = tok.encode(s, addSpecial: true)
            let back = tok.decode(ids)
            let msg = "drift " + s.debugDescription + " -> "
                + back.debugDescription
            #expect(back == s, Comment(rawValue: msg))
        }
    }

    @Test(.enabled(if: TokenizerGateTests.haveTokenizer))
    func controlTokensRoundTrip() throws {
        let tok = try Self.tokenizer()
        let controls = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        for text in controls {
            let ids = tok.encode(text, addSpecial: true)
            let back = tok.decode(ids)
            let msg = "lost " + text + " -> \(ids) -> " + back
            #expect(back == text, Comment(rawValue: msg))
        }
    }

    // The structural specials are non-negotiably atomic (added_tokens), and
    @Test(.enabled(if: TokenizerGateTests.haveTokenizer))
    func structuralSpecialsAreAtomic() throws {
        let tok = try Self.tokenizer()
        let imStart = tok.encode("<|im_start|>", addSpecial: true)
        let imEnd = tok.encode("<|im_end|>", addSpecial: true)
        #expect(imStart.count == 1, Comment(rawValue: "im_start \(imStart)"))
        #expect(imEnd.count == 1, Comment(rawValue: "im_end \(imEnd)"))
        #expect(imEnd.first == tok.eosId,
                Comment(rawValue: "eos \(imEnd) vs \(tok.eosId)"))
    }

    // Streaming must reconstruct the text EXACTLY, including multi-scalar emoji
    // (variation selector, skin-tone, ZWJ family).
    @Test(.enabled(if: TokenizerGateTests.haveTokenizer))
    func streamingPreservesMultiScalarEmoji() throws {
        let tok = try Self.tokenizer()
        let samples = [
            "ok 👍🏻 done",
            "love ❤️ it",
            "family 👨‍👩‍👧‍👦 here",
            "flag 🇯🇵 end",
            "plain ascii only",
        ]
        for s in samples {
            let ids = tok.encode(s, addSpecial: false)
            var dec = StreamDecoder()
            var acc = ""
            for k in 1 ... ids.count {
                acc += dec.step(Array(ids[0 ..< k]), tok)
            }
            let whole = tok.decode(ids)
            let msg = "stream drift " + acc.debugDescription + " != "
                + whole.debugDescription
            #expect(acc == whole, Comment(rawValue: msg))
        }
    }

    @Test(.enabled(if: TokenizerGateTests.haveTokenizer))
    func chatMLRoundTripMatchesReference() throws {
        let ref = Self.referenceIDs
        if ref.isEmpty {
            Issue.record("reference not captured — run capture_chatml_ids.py")
        } else {
            let tok = try Self.tokenizer()
            let prompt = ChatML.prompt(system: Self.system, user: Self.user)
            let ids = tok.encode(prompt, addSpecial: true)
            let div = firstDivergence(ids, ref)
            if let div {
                let lo = max(0, div - 3)
                let a = Array(ids[lo ..< min(ids.count, div + 4)])
                let b = Array(ref[lo ..< min(ref.count, div + 4)])
                Issue.record(Comment(rawValue:
                    "diverge @\(div) swift=\(a) py=\(b) "
                    + "n=\(ids.count)/\(ref.count)"))
            }
            #expect(div == nil)
        }
    }
}
