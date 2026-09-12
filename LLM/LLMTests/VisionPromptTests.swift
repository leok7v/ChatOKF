import XCTest
@testable import LLM

// The chat template's render_content macro dispatches a message on whether its
final class VisionPromptTests: XCTestCase {

    // A minimal stand-in for the shipped render_content macro's dispatch shape.
    private static let template =
        "{% for m in messages %}"
        + "{% if m.content is string %}S:{{ m.content }}"
        + "{% elif m.content is iterable and m.content is not mapping %}"
        + "{% for item in m.content %}"
        + "{% if item.type == 'image' %}<V>"
        + "{% elif 'text' in item %}{{ item.text }}"
        + "{% endif %}{% endfor %}{% endif %}{% endfor %}"

    func testContentPartsRenderVisionPlaceholderInOrder() throws {
        let msg = AgentMessage(role: "user", content: "unused",
                               contentParts: [.text("a"), .image, .image,
                                              .text("b")])
        let out = try renderPrompt(
            template: VisionPromptTests.template, messages: [msg],
            tools: [], addGenerationPrompt: false, enableThinking: false)
        XCTAssertEqual(out, "a<V><V>b")
    }

    func testPlainContentStillRendersAsString() throws {
        let msg = AgentMessage(role: "user", content: "hello")
        let out = try renderPrompt(
            template: VisionPromptTests.template, messages: [msg],
            tools: [], addGenerationPrompt: false, enableThinking: false)
        XCTAssertEqual(out, "S:hello")
    }
}
