import XCTest
@testable import LLM

// The wire a template speaks must come OUT OF THAT TEMPLATE.
final class ChatWireTests: XCTestCase {

    private func gemmaTemplate() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent(
                "fixtures/gemma4-chat-template.jinja"),
            encoding: .utf8)
    }

    // A ChatML template that renders BOTH a reasoning channel and a tool call,
    // i.e. one that actually speaks so the derivation has something to read.
    private let qwenish = "{% for m in messages %}"
        + "<|im_start|>{{ m.role }}\n"
        + "{% if m.reasoning_content %}"
        + "<think>\n{{ m.reasoning_content }}\n</think>\n\n"
        + "{% endif %}"
        + "{{ m.content }}"
        + "{% for tc in m.tool_calls %}"
        + "<tool_call>\n<function={{ tc.name }}>"
        + "</function>\n</tool_call>"
        + "{% endfor %}"
        + "<|im_end|>\n"
        + "{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    // The silence case: no reasoning, no tool markup rendered anywhere.
    private let silent = "{% for m in messages %}"
        + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        + "{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    func testDerivesQwenWire() {
        let w = ChatWire.derive(qwenish)
        XCTAssertEqual(w.reasoningOpen, "<think>")
        XCTAssertEqual(w.reasoningClose, "</think>")
        XCTAssertEqual(w.toolCallOpen, "<tool_call>")
        XCTAssertEqual(w.toolCallClose, "</tool_call>")
        XCTAssertEqual(w.turnOpen, "<|im_start|>user\n")
        XCTAssertEqual(w.turnClose, "<|im_end|>\n")
        XCTAssertTrue(w.derivedReasoning)
    }

    func testDerivesGemmaWire() throws {
        let w = ChatWire.derive(try gemmaTemplate())
        XCTAssertEqual(w.reasoningOpen, "<|channel>thought")
        XCTAssertEqual(w.reasoningClose, "<channel|>")
        XCTAssertEqual(w.toolCallOpen, "<|tool_call>")
        XCTAssertEqual(w.toolCallClose, "<tool_call|>")
        XCTAssertTrue(w.derivedReasoning)
    }

    // A silent template keeps the Qwen wire, so every session built on one
    // behaves exactly as it did before the wire was derived at all.
    func testSilentTemplateKeepsTheDefault() {
        let w = ChatWire.derive(silent)
        XCTAssertEqual(w.reasoningOpen, ChatWire.defaultReasoningOpen)
        XCTAssertEqual(w.reasoningClose, ChatWire.defaultReasoningClose)
        XCTAssertEqual(w.toolCallOpen, ChatWire.defaultToolCallOpen)
        XCTAssertEqual(w.toolCallClose, ChatWire.defaultToolCallClose)
        XCTAssertFalse(w.derivedReasoning)
    }

    // Whether decoding starts inside the reasoning region is the template's
    // call.
    func testStartsInReasoningAcrossShapes() throws {
        let gemma = ChatWire.derive(try gemmaTemplate())
        XCTAssertFalse(gemma.startsInReasoning(genPrompt: "<|turn>model\n",
                                               enabled: true),
                       "gemma opens its own channel; decoding starts in "
                       + "content")
        XCTAssertTrue(gemma.startsInReasoning(
            genPrompt: "<|channel>thought\n", enabled: true),
            "after a tool response gemma DOES open the channel")

        let qwen = ChatWire.derive(qwenish)
        XCTAssertTrue(qwen.startsInReasoning(
            genPrompt: "<|im_start|>assistant\n<think>\n", enabled: true))
        XCTAssertFalse(qwen.startsInReasoning(
            genPrompt: "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            enabled: true))

        // Silence on the DEFAULT wire keeps the historical permissive
        // reading, which the reduced test templates depend on.
        let quiet = ChatWire.derive(silent)
        XCTAssertTrue(quiet.startsInReasoning(
            genPrompt: "<|im_start|>assistant\n", enabled: true))
        XCTAssertFalse(quiet.startsInReasoning(
            genPrompt: "<|im_start|>assistant\n", enabled: false))
    }

    // The bug F15 predicted and Phase 3b measured: gemma reasons, but the old
    func testGemmaReportsThinkingSupported() throws {
        let t = try gemmaTemplate()
        XCTAssertTrue(ChatWire.derive(t).reasons(template: t))
        XCTAssertTrue(templateSupportsThinking(t),
                      "gemma's reasoning capability is still invisible")
    }

    // The dense-Qwen3 shape (a baked CLOSED empty block, enable_thinking
    // never mentioned) must still report NO reasoning.
    func testBakedClosedThinkStillReportsUnsupported() {
        let direct = "{% for m in messages %}"
            + "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            + "{% endfor %}{% if add_generation_prompt %}"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
        XCTAssertFalse(templateSupportsThinking(direct))
    }
}
