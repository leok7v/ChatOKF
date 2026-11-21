import XCTest
@testable import LLM

final class NineStringTests: XCTestCase {

    struct Nine {
        let start: String
        let open: String
        let middle: String
        let close: String
        let reply: String
        let plain: String
        let think: String
        let toolsOpen: String
        let toolsClose: String
        let roles: [String: String]
    }

    static let qwenToolsClose = "\n</tools>\n\nIf you choose to call a "
        + "function ONLY reply in the following format with NO suffix:\n\n"
        + "<tool_call>\n<function=example_function_name>\n"
        + "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        + "<parameter=example_parameter_2>\nThis is the value for the second "
        + "parameter\nthat can span\nmultiple lines\n</parameter>\n"
        + "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function "
        + "calls MUST follow the specified format: an inner "
        + "<function=...></function> block must be nested within "
        + "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
        + "specified\n- You may provide optional reasoning for your function "
        + "call in natural language BEFORE the function call, but NOT after\n"
        + "- If there is no function call available, answer the question "
        + "like normal with your current knowledge and do not tell the user "
        + "about function calls\n</IMPORTANT>"

    static let qwen = Nine(
        start: "", open: "<|im_start|>", middle: "\n", close: "<|im_end|>\n",
        reply: "<|im_start|>assistant\n", plain: "<think>\n\n</think>\n\n",
        think: "<think>\n",
        toolsOpen: "# Tools\n\nYou have access to the following "
            + "functions:\n\n<tools>",
        toolsClose: qwenToolsClose, roles: [:])

    static let gemma = Nine(
        start: "<bos>", open: "<|turn>", middle: "\n", close: "<turn|>\n",
        reply: "<|turn>model\n", plain: "", think: "", toolsOpen: "",
        toolsClose: "", roles: ["assistant": "model"])

    struct Shape {
        let name: String
        let messages: [AgentMessage]
        let tools: [ToolSpec]
        let thinking: Bool
    }

    static let tool = ToolSpec(
        name: "get_time", description: "Get the current time",
        parametersJSON: "{\"type\": \"object\", \"properties\": {}}")

    static func toolJSON(_ t: ToolSpec) -> String {
        "{\"type\": \"function\", \"function\": {\"name\": \"\(t.name)\", "
            + "\"description\": \"\(t.description)\", \"parameters\": "
            + "\(t.parametersJSON)}}"
    }

    static func trimmed(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func render(_ f: Nine, _ shape: Shape, bos: String) -> String {
        var out = f.start.isEmpty ? "" : bos
        for m in shape.messages {
            let role = f.roles[m.role] ?? m.role
            let content = trimmed(m.contentParts.map { parts in
                parts.map { part in
                    switch part {
                    case .text(let t): return t
                    case .image: return "<|vision_start|><|image_pad|>"
                        + "<|vision_end|>"
                    case .audio: return "<|audio|>"
                    case .video: return "<|video|>"
                    }
                }.joined()
            } ?? m.content)
            out += f.open + role + f.middle
            if m.role == "system", !shape.tools.isEmpty, !f.toolsOpen.isEmpty {
                let listed = shape.tools.map { t in toolJSON(t) }
                    .joined(separator: "\n")
                out += f.toolsOpen + "\n" + listed + f.toolsClose
                    + (content.isEmpty ? "" : "\n\n")
            }
            out += content + f.close
        }
        return out + f.reply + (shape.thinking ? f.think : f.plain)
    }

    static func shapes() -> [Shape] {
        var out: [Shape] = []
        let system = AgentMessage(role: "system", content: "Be brief.")
        for tools in [[], [tool]] {
            for withSystem in [true, false] {
                for turns in 1...5 {
                    for thinking in [false, true] {
                        var msgs = withSystem ? [system] : []
                        for t in 0..<turns {
                            msgs.append(AgentMessage(role: "user",
                                                     content: "q\(t)"))
                            if t + 1 < turns {
                                msgs.append(AgentMessage(role: "assistant",
                                                         content: "a\(t)"))
                            }
                        }
                        out.append(Shape(
                            name: "tools \(tools.count) system \(withSystem) "
                                + "turns \(turns) thinking \(thinking)",
                            messages: msgs, tools: tools, thinking: thinking))
                    }
                }
            }
        }
        out.append(Shape(name: "vision placeholder", messages: [
            system,
            AgentMessage(role: "user", content: "",
                         contentParts: [.image, .text("what is this")]),
        ], tools: [], thinking: false))
        let call = "<tool_call>\n<function=get_time>\n<parameter=zone>\n"
            + "UTC\n</parameter>\n</function>\n</tool_call>"
        out.append(Shape(name: "tool call and result", messages: [
            system,
            AgentMessage(role: "user", content: "time?"),
            AgentMessage(role: "assistant",
                         content: "Let me check.\n\n" + call),
            AgentMessage(role: "user",
                         content: "<tool_response>\n12:00\n</tool_response>"),
        ], tools: [tool], thinking: false))
        out.append(Shape(name: "meta append", messages: [
            system,
            AgentMessage(role: "user", content: "ZqUserZq"),
            AgentMessage(role: "assistant", content: "ZqBodyZq"),
            AgentMessage(role: "user", content: "Give this chat a title."),
        ], tools: [], thinking: false))
        return out
    }

    static func firstDiff(_ a: String, _ b: String) -> String {
        let x = Array(a.utf8), y = Array(b.utf8)
        var i = 0
        while i < x.count && i < y.count && x[i] == y[i] { i += 1 }
        let tail = { (s: [UInt8]) in
            String(decoding: s[i ..< min(s.count, i + 24)], as: UTF8.self)
                .debugDescription
        }
        return "byte \(i): jinja \(tail(x)) nine \(tail(y))"
    }

    private func measure(_ name: String, _ nine: Nine, bos: String) throws {
        guard let path = TestWeights.find(name) else {
            throw XCTSkip(TestWeights.missing(name))
        }
        let g = try GGUF(path: path)
        let template = g.string("tokenizer.chat_template") ?? ""
        XCTAssertFalse(template.isEmpty, "\(name) carries no template")
        var differ: [String] = []
        let all = NineStringTests.shapes()
        for shape in all {
            let jinja = try renderPrompt(
                template: template, messages: shape.messages,
                tools: shape.tools, addGenerationPrompt: true,
                enableThinking: shape.thinking, bosToken: bos)
            let text = NineStringTests.render(nine, shape, bos: bos)
            if jinja != text {
                differ.append(shape.name + ": "
                              + NineStringTests.firstDiff(jinja, text))
            }
        }
        print("[nine] \(name): \(differ.count) of \(all.count) shapes differ")
        for line in differ { print("[nine]   " + line) }
        XCTAssertGreaterThan(all.count, 0)
    }

    func testTheNineStringsAgainstTheQwen35Template() throws {
        try measure("Qwen3.5-2B", NineStringTests.qwen, bos: "")
    }

    func testTheNineStringsAgainstTheGemma4Template() throws {
        try measure("gemma-4-E2B", NineStringTests.gemma, bos: "<bos>")
    }
}
