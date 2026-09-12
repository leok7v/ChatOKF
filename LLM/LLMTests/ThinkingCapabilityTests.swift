import XCTest
@testable import LLM

// Whether a model reasons is asked of its own chat template, never of its name
// (see feedback-no-hardcoded-model-constants).
final class ThinkingCapabilityTests: XCTestCase {

    // The Qwen3.5 shape, reduced to the branch that matters.
    private let reasoning = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\\n' }}
        {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        {%- endif %}
        """

    // The dense Qwen3 shape: the think block is always closed, and
    // enable_thinking never appears.
    private let direct = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
        """

    // The Qwen3.8 acceptance set and raise VERBATIM: a stand-in may reduce
    // the shipped template, never forgive an input it rejects.
    private let effortful = """
        {%- if enable_thinking is undefined or enable_thinking is true %}
        {%- set e = reasoning_effort|default('xhigh') %}
        {%- if e not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ e) }}
        {%- endif %}
        {%- if e != 'medium' %}{{- 'effort=' ~ e }}{%- endif %}
        {%- endif %}
        {% for message in messages %}{{ message.content }}{% endfor %}
        """

    private let probe = [AgentMessage(role: "user", content: "x")]

    private func render(_ level: String?) throws -> String {
        try renderPrompt(template: effortful, messages: probe, tools: [],
                         addGenerationPrompt: true, enableThinking: true,
                         reasoningEffort: level)
    }

    func testEffortTemplateReportsSupported() {
        XCTAssertTrue(templateTakesReasoningEffort(effortful))
    }

    func testTemplateIgnoringEffortReportsUnsupported() {
        XCTAssertFalse(templateTakesReasoningEffort(reasoning))
    }

    func testAbsentEffortTakesTheTemplateDefault() throws {
        XCTAssertTrue(try render(nil).contains("effort=xhigh"))
    }

    func testEachLevelReachesTheTemplate() throws {
        XCTAssertTrue(try render("low").contains("effort=low"))
        XCTAssertFalse(try render("medium").contains("effort="))
        XCTAssertTrue(try render("xhigh").contains("effort=xhigh"))
    }

    func testLevelsAreTheOnesTheTemplateAccepts() {
        XCTAssertThrowsError(try render("high"),
            "the shipped template REFUSES `high`; a gate that forgives it "
            + "passes while the app renders an empty prompt")
        XCTAssertEqual(templateEffortLevels(effortful),
                       ["low", "medium", "xhigh"])
        XCTAssertEqual(templateEffortLevels(reasoning).count, 1,
            "a template ignoring the variable offers no choice")
    }

    func testUiLevelsMapOntoAcceptedSpellings() throws {
        let levels = templateEffortLevels(effortful)
        let wires = ["low", "medium", "high"].enumerated().map { slot, name in
            effortSpelling(name, slot: slot, in: levels)
        }
        XCTAssertEqual(wires, ["low", "medium", "xhigh"])
        for wire in wires {
            XCTAssertFalse(try render(wire).isEmpty,
                "\(wire) rendered nothing")
        }
    }

    func testSpellingKeepsAnAcceptedName() {
        XCTAssertEqual(
            effortSpelling("high", slot: 2, in: ["low", "high"]), "high")
        XCTAssertEqual(effortSpelling("high", slot: 2, in: []), "high")
    }

    func testReasoningTemplateReportsThinkingSupported() {
        XCTAssertTrue(templateSupportsThinking(reasoning))
    }

    func testBakedClosedThinkTemplateReportsUnsupported() {
        XCTAssertFalse(templateSupportsThinking(direct))
    }

    // A template with no generation prompt at all cannot reason either, and
    // must not crash the probe.
    func testTemplateWithoutGenerationPromptIsUnsupported() {
        let bare = "{% for message in messages %}{{ message.content }}"
            + "{% endfor %}"
        XCTAssertFalse(templateSupportsThinking(bare))
    }

    // The SHIPPED 0.8B template, when present, must classify as reasoning --
    // the inline shapes above are a reduction, this is the real bytes.
    func testShippedQwen35TemplateIsReasoning() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: String? = nil
        let flat = base.appendingPathComponent("chat_template.jinja")
        if fm.fileExists(atPath: flat.path) {
            found = try? String(contentsOf: flat, encoding: .utf8)
        } else {
            for sha in (try? fm.contentsOfDirectory(atPath: base.path)) ?? [] {
                let url = base.appendingPathComponent(sha)
                    .appendingPathComponent("chat_template.jinja")
                if found == nil, fm.fileExists(atPath: url.path) {
                    found = try? String(contentsOf: url, encoding: .utf8)
                }
            }
        }
        try XCTSkipIf(found == nil, "no Qwen3.5-0.8B chat template on disk")
        XCTAssertTrue(templateSupportsThinking(found!))
    }
}
