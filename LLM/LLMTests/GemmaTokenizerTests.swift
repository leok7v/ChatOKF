import Foundation
import Testing
@testable import LLM

struct GemmaTokenizerTests {

    static let sample = "The bat and ball cost 1.10 dollars.\n"
        + "\u{65E5}\u{672C}\u{8A9E}\u{306E}\u{30C6}\u{30AD}\u{30B9}\u{30C8}\n"
        + ":) tab\there  and   spaces\n"

    static let sampleIds: [Int32] = [
        818, 9537, 532, 4299, 2157, 236743, 236770, 236761, 236770, 236771,
        11092, 236761, 107, 94951, 236945, 95830, 107, 42951, 6937, 255968,
        8472, 138, 624, 139, 35220, 107,
    ]

    @Test(needsGemmaWeights) func sampleEncodesToTheRecordedIds() throws {
        let path = try #require(gemmaGgufPath)
        let chat = try GemmaChat(ggufPath: path)
        let got = chat.encode(GemmaTokenizerTests.sample)
        #expect(got == GemmaTokenizerTests.sampleIds,
                "gemma ids moved: \(got)")
    }

    @Test(needsGemmaWeights) func hardTextsRoundTrip() throws {
        let path = try #require(gemmaGgufPath)
        let chat = try GemmaChat(ggufPath: path)
        for text in [GemmaTokenizerTests.sample,
                     "plain words only",
                     "\u{1F642}\u{1F1EF}\u{1F1F5} caf\u{E9} na\u{EF}ve",
                     "no\nspaces\nat\nall\njust\nnewlines",
                     "   leading and trailing   ",
                     "a", ""] {
            let back = chat.decode(chat.encode(text))
            #expect(back == text, "round trip lost \(text.debugDescription)")
        }
    }

    @Test(needsGemmaWeights) func lengthDoesNotChangeTheIdsOfAPart() throws {
        let path = try #require(gemmaGgufPath)
        let chat = try GemmaChat(ggufPath: path)
        let unit = "The quarterly logistics review covered warehouse "
            + "throughput and pallet rotation.\n"
        let once = chat.encode(unit)
        for repeats in [2, 8, 64] {
            let many = chat.encode(String(repeating: unit, count: repeats))
            let msg = "\(repeats) copies gave \(many.count) ids, not "
                + "\(once.count * repeats)"
            #expect(many.count == once.count * repeats,
                    Comment(rawValue: msg))
            #expect(Array(many.prefix(once.count)) == once,
                    "the first copy tokenized differently in a longer text")
        }
    }
}
