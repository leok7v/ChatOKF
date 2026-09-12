import Foundation
import Testing
@testable import LLM

// What a title turn appends to the generation prompt, per template. No weights
// and no engine here.
struct TitleSeedTests {
    private static func fixture(_ name: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/" + name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // The generation prompt as ChatSession sees it: render the same turn with
    // and without it and take the suffix.
    private static func genSuffix(_ template: String) throws -> String {
        let msgs = [AgentMessage(role: "user", content: "x")]
        let closed = try renderPrompt(
            template: template, messages: msgs, tools: [],
            addGenerationPrompt: false, enableThinking: true, bosToken: "")
        let full = try renderPrompt(
            template: template, messages: msgs, tools: [],
            addGenerationPrompt: true, enableThinking: true, bosToken: "")
        return full.hasPrefix(closed)
            ? String(full.dropFirst(closed.count)) : ""
    }

    @Test func gemmaGetsOnlyTheTextBlock() throws {
        let template = try TitleSeedTests.fixture("gemma4-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        #expect(!wire.opensReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(!wire.closesReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(ChatSession.titleSeed(gen, wire) == ChatSession.metaBlock)
    }

    // Qwen3.5 OPENS the marker and leaves it open, so the seed closes it and
    // must not open a second one. This is the branch that shipped unproven.
    @Test func qwenGetsOnlyTheClosingMarker() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        #expect(wire.opensReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(!wire.closesReasoning(gen), "gen: \(gen.debugDescription)")
        #expect(ChatSession.titleSeed(gen, wire)
                == wire.reasoningClose + ChatSession.metaBlock)
    }

    // A template that already bakes the closed block wants nothing, and would
    // be corrupted by a second one.
    @Test func anAlreadyClosedPromptIsLeftAlone() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let baked = wire.reasoningOpen + wire.reasoningClose
        #expect(ChatSession.titleSeed(baked, wire) == ChatSession.metaBlock)
    }

    // The seed is idempotent: applying it and asking again yields nothing.
    // A turn that seeded twice would lay two blocks into the KV.
    @Test func seedingTwiceClosesNothingTwice() throws {
        let template = try TitleSeedTests.fixture("qwen35-chat-template.jinja")
        let wire = ChatWire.derive(template)
        let gen = try TitleSeedTests.genSuffix(template)
        let once = gen + ChatSession.titleSeed(gen, wire)
        #expect(ChatSession.titleSeed(once, wire) == ChatSession.metaBlock)
    }

    static let toolCallRaw = "<tool_call>\n<function=calculator>\n"
        + "<parameter=expression>\n1.10 - 1.00\n</parameter>\n</function>\n"
        + "</tool_call>"

    @Test func aToolCallNeverBecomesATitle() {
        #expect(ChatSession.cleanTitle(TitleSeedTests.toolCallRaw) == "")
    }

    @Test func aToolCallNeverBecomesAFollowup() {
        #expect(ChatSession.cleanFollowup(TitleSeedTests.toolCallRaw) == "")
    }

    @Test func reasoningMarkersAreStripped() {
        #expect(ChatSession.cleanTitle("<think>hm</think>Bat and ball")
                == "hmBat and ball")
        #expect(ChatSession.cleanTitle("<|im_end|>") == "")
    }

    @Test func aTitleWithoutLettersIsRejected() {
        #expect(ChatSession.cleanTitle("1.10 - 1.00") == "")
        #expect(ChatSession.cleanTitle("42") == "")
    }

    @Test func ordinaryTitlesSurvive() {
        #expect(ChatSession.cleanTitle("Bat and ball puzzle")
                == "Bat and ball puzzle")
        #expect(ChatSession.cleanTitle("Title: Riddle Solution")
                == "Riddle Solution")
    }

    @Test func anUnmatchedAngleIsNotATag() {
        #expect(ChatSession.withoutMarkup("5 < 6 and more") == "5 < 6 and more")
    }

    @Test func aRealFollowupSurvives() {
        let q = "How did you get 0.05 from the equation?"
        #expect(ChatSession.cleanFollowup(q) == q)
    }

    static let narratedTitle = "The\nThe user wants a short title for the "
        + "conversation.\nConstraints:\n1.  "

    static let narratedOneLine = "präzise: The user wants a short title for "
        + "the conversation, summarizing its topic."

    @Test func deliberationNeverBecomesATitle() {
        #expect(ChatSession.cleanTitle(TitleSeedTests.narratedTitle) == "")
        #expect(ChatSession.cleanTitle(TitleSeedTests.narratedOneLine) == "")
        #expect(ChatSession.cleanTitle("prä") == "")
    }

    @Test func aOneWordTitleIsRejected() {
        #expect(ChatSession.cleanTitle("Animals") == "")
        #expect(ChatSession.cleanTitle("The") == "")
    }

    @Test func realTitlesFromTheStoreSurvive() {
        for title in ["Savings Account Interest Calculation",
                      "Compound Interest Calculation",
                      "Dark Matter Dark Energy",
                      "Weather forecast Los Angeles",
                      "Bat and Ball Cost Puzzle",
                      "Harvest Report Analysis",
                      "Riddle Solution"] {
            #expect(ChatSession.cleanTitle(title) == title)
        }
    }

    @Test func topicTitleRanksByFrequencyAndReadsInOrder() {
        let texts = ["Tell me about dark matter and dark energy.",
                     "Dark matter is matter that does not emit light. "
                     + "Dark energy drives expansion."]
        #expect(TopicTitle.from(texts) == "Dark Matter Energy")
    }

    @Test func topicTitleKeepsASingleSubject() {
        #expect(TopicTitle.from(["Tell me about animals",
                                 "Animals are wonderful"]) == "Animals")
    }

    @Test func topicTitleIsEmptyWithoutSubstance() {
        #expect(TopicTitle.from([]) == "")
        #expect(TopicTitle.from(["Hi", "Hello", "How are you?"]) == "")
    }

    @Test func topicTitleDropsStopWordsAndShortTokens() {
        #expect(TopicTitle.from(["The the the savings savings"])
                == "Savings")
    }

    @Test func topicTitleIgnoresWordsSeenOnce() {
        #expect(TopicTitle.from(["a unique unrepeated sentence"]) == "")
    }
}
