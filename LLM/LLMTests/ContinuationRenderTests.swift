import XCTest
@testable import LLM

// Proves the append-only continuation render: rendering only each turn's DELTA
final class ContinuationRenderTests: XCTestCase {

    // The 0.8B chat template ships under a sha subdir of models/Qwen3.5-0.8B.
    private func template() -> String? {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: String? = nil
        let direct = base.appendingPathComponent("chat_template.jinja")
        if let s = try? String(contentsOf: direct, encoding: .utf8) {
            found = s
        } else if let subs = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) {
            for sub in subs {
                let t = sub.appendingPathComponent("chat_template.jinja")
                if let s = try? String(contentsOf: t, encoding: .utf8) {
                    found = s
                }
            }
        }
        return found
    }

    private func user(_ s: String) -> AgentMessage {
        AgentMessage(role: "user", content: s)
    }

    private func assistant(_ s: String) -> AgentMessage {
        AgentMessage(role: "assistant", content: s)
    }

    private func render(_ tmpl: String, _ msgs: [AgentMessage],
                        thinking: Bool, vid: Bool = false) throws -> String {
        try renderPrompt(template: tmpl, messages: msgs, tools: [],
                         addGenerationPrompt: true, enableThinking: thinking,
                         addVisionId: vid)
    }

    // Everything before the LAST "<|im_start|>assistant\n": the mark boundary
    // (before the generation prompt), matching ChatSession.seedDelta.
    private func closed(_ render: String) -> String {
        var result = render
        if let r = render.range(of: "<|im_start|>assistant\n",
                                options: .backwards) {
            result = String(render[..<r.lowerBound])
        }
        return result
    }

    // The core gate: a 3-turn text conversation.
    private func checkText(thinking: Bool) throws {
        guard let tmpl = template() else {
            throw XCTSkip("models/Qwen3.5-0.8B chat template not present")
        }
        let sys = AgentMessage(role: "system", content: "You are helpful.")
        let pairs: [(String, String?)] = [
            ("What is 2+2?", "It is 4."),
            ("And 3+3?", "It is 6."),
            ("Thanks, and 4+4?", nil),   // current turn, no answer yet
        ]
        var msgs = [sys]
        var reconstructed = ""
        for (i, pair) in pairs.enumerated() {
            msgs.append(user(pair.0))
            let fresh = i == 0
            let delta = fresh
                ? msgs
                : [msgs[msgs.count - 2], msgs[msgs.count - 1]]
            let r = try render(tmpl, delta, thinking: thinking)
            reconstructed += pair.1 == nil ? r : closed(r)
            if let a = pair.1 { msgs.append(assistant(a)) }
        }
        let whole = try render(tmpl, msgs, thinking: thinking)
        XCTAssertEqual(reconstructed, whole,
            "delta-render concatenation diverged from the whole render")
    }

    func testDeltaEqualsWholeReasoningOff() throws { try checkText(thinking: false) }
    func testDeltaEqualsWholeReasoningOn() throws { try checkText(thinking: true) }

    // A later turn carrying an image: its delta render [prev stripped answer,
    func testVisionDeltaEqualsWholeTail() throws {
        guard let tmpl = template() else {
            throw XCTSkip("models/Qwen3.5-0.8B chat template not present")
        }
        let sys = AgentMessage(role: "system", content: "You are helpful.")
        let u1 = user("Hello.")
        let a1 = assistant("Hi there.")
        let imgTurn = AgentMessage(role: "user", content: "Describe it.",
            contentParts: [.image, .text("Describe it.")])
        let whole = try render(tmpl, [sys, u1, a1, imgTurn], thinking: false)
        let delta = try render(tmpl, [a1, imgTurn], thinking: false)
        // The delta must be the exact suffix of the whole render from a1 on.
        XCTAssertTrue(whole.hasSuffix(delta),
            "vision delta render is not the whole-render tail")
        // And it must carry exactly one image placeholder (this turn's).
        let pads = delta.components(separatedBy: "<|image_pad|>").count - 1
        XCTAssertEqual(pads, 1, "expected one image placeholder in the delta")
    }

    // Numbering is LOCAL under delta-render: add_vision_id on a multi-image
    func testNumberingIsPerTurnLocal() throws {
        guard let tmpl = template() else {
            throw XCTSkip("models/Qwen3.5-0.8B chat template not present")
        }
        let a1 = assistant("Earlier answer.")
        let twoImages = AgentMessage(role: "user", content: "Compare.",
            contentParts: [.image, .image, .text("Compare.")])
        let delta = try render(tmpl, [a1, twoImages], thinking: false, vid: true)
        XCTAssertTrue(delta.contains("Picture 1:"), "missing Picture 1 label")
        XCTAssertTrue(delta.contains("Picture 2:"), "missing Picture 2 label")
        // The prior assistant block is untouched by the flag.
        XCTAssertTrue(delta.contains("Earlier answer."), "prior block changed")
    }
}
