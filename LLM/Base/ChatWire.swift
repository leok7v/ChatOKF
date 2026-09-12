import Foundation

struct ChatWire: Sendable {
    let reasoningOpen: String
    let reasoningClose: String
    let toolCallOpen: String
    let toolCallClose: String
    let turnOpen: String
    let turnClose: String
    let derivedReasoning: Bool

    static let defaultReasoningOpen = "<think>"
    static let defaultReasoningClose = "</think>"
    static let defaultToolCallOpen = "<tool_call>"
    static let defaultToolCallClose = "</tool_call>"

    // Sentinels are ASCII and shaped to survive a `| trim` and any
    // strip-thinking macro, while never occurring in a real template.
    private static let sBody = "ZqBodyZq"
    private static let sReason = "ZqReasonZq"
    private static let sTool = "ZqToolZq"
    private static let sArgKey = "ZqArgKZq"
    private static let sArgVal = "ZqArgVZq"
    private static let sUser = "ZqUserZq"

    static func derive(_ template: String) -> ChatWire {
        let reason = deriveReasoning(template)
        let call = deriveToolCall(template)
        let turn = deriveTurn(template)
        return ChatWire(
            reasoningOpen: reason?.open ?? defaultReasoningOpen,
            reasoningClose: reason?.close ?? defaultReasoningClose,
            toolCallOpen: call?.open ?? defaultToolCallOpen,
            toolCallClose: call?.close ?? defaultToolCallClose,
            turnOpen: turn?.open ?? "",
            turnClose: turn?.close ?? "",
            derivedReasoning: reason != nil)
    }

    private static func deriveReasoning(
        _ t: String
    ) -> (open: String, close: String)? {
        let user = AgentMessage(role: "user", content: sUser)
        let bare = AgentMessage(role: "assistant", content: sBody)
        let with = AgentMessage(role: "assistant", content: sBody,
                                reasoning: sReason)
        let a = render(t, [user, with], think: true)
        let b = render(t, [user, bare], think: true)
        return span(a, b, sReason)
            .map { pair in
                (trimWhitespace(pair.open), trimWhitespace(pair.close))
            }
            .flatMap { pair in
                pair.open.isEmpty || pair.close.isEmpty ? nil : pair
            }
    }

    private static func deriveToolCall(
        _ t: String
    ) -> (open: String, close: String)? {
        let user = AgentMessage(role: "user", content: sUser)
        let bare = AgentMessage(role: "assistant", content: sBody)
        let call = AgentToolCall(name: sTool,
                                 arguments: [ToolArg(name: sArgKey,
                                                     value: sArgVal)])
        let with = AgentMessage(role: "assistant", content: sBody,
                                toolCalls: [call])
        let a = render(t, [user, with], think: false)
        let b = render(t, [user, bare], think: false)
        return span(a, b, sTool).map { pair in
            (headTag(pair.open), tailTag(pair.close))
        }
    }

    private static func deriveTurn(
        _ t: String
    ) -> (open: String, close: String)? {
        let a = render(t, [AgentMessage(role: "user", content: sUser)],
                       think: false)
        var result: (open: String, close: String)? = nil
        if let r = a.range(of: sUser) {
            result = (String(a[..<r.lowerBound]),
                      String(a[r.upperBound...]))
        }
        return result
    }

    private static func render(_ t: String, _ msgs: [AgentMessage],
                               think: Bool) -> String {
        (try? renderPrompt(template: t, messages: msgs, tools: [],
                           addGenerationPrompt: false,
                           enableThinking: think)) ?? ""
    }

    private static func span(_ a: String, _ b: String,
                             _ sentinel: String) -> (open: String,
                                                     close: String)? {
        let x = Array(a.utf8)
        let y = Array(b.utf8)
        let s = Array(sentinel.utf8)
        var pre = 0
        while pre < x.count && pre < y.count && x[pre] == y[pre] { pre += 1 }
        var suf = 0
        while suf < x.count - pre && suf < y.count - pre
            && x[x.count - 1 - suf] == y[y.count - 1 - suf] { suf += 1 }
        // Rotate to the LEFTMOST diff: a byte shared by the prefix end and the
        // insertion start would otherwise decapitate a marker opening with '<'.
        let len = x.count - suf - pre
        while pre > 0 && len > 0 && x[pre - 1] == x[pre - 1 + len] {
            pre -= 1
            suf += 1
        }
        let mid = Array(x[pre ..< (x.count - suf)])
        var result: (open: String, close: String)? = nil
        if let at = index(mid, s) {
            result = (String(decoding: mid[..<at], as: UTF8.self),
                      String(decoding: mid[(at + s.count)...], as: UTF8.self))
        }
        return result
    }

    private static func index(_ b: [UInt8], _ pat: [UInt8]) -> Int? {
        var result: Int? = nil
        var i = 0
        let last = b.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private static func trimWhitespace(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func headTag(_ s: String) -> String {
        let t = trimWhitespace(s)
        var result = t
        if let gt = t.firstIndex(of: ">") {
            result = String(t[...gt])
        }
        return result
    }

    private static func tailTag(_ s: String) -> String {
        let t = trimWhitespace(s)
        var result = t
        if let lt = t.lastIndex(of: "<") {
            result = String(t[lt...])
        }
        return result
    }

    func reasons(template: String) -> Bool {
        var result = derivedReasoning
        if !result {
            let probe = [AgentMessage(role: "user", content: "x")]
            let closed = (try? renderPrompt(
                template: template, messages: probe, tools: [],
                addGenerationPrompt: false, enableThinking: true)) ?? ""
            let full = (try? renderPrompt(
                template: template, messages: probe, tools: [],
                addGenerationPrompt: true, enableThinking: true)) ?? ""
            if !closed.isEmpty, full.hasPrefix(closed) {
                let gen = String(full.dropFirst(closed.count))
                result = opensReasoning(gen)
            }
        }
        return result
    }

    func opensReasoning(_ gen: String) -> Bool {
        var result = false
        if let at = gen.range(of: reasoningOpen, options: .backwards) {
            result = gen.range(of: reasoningClose,
                               range: at.upperBound ..< gen.endIndex) == nil
        }
        return result
    }

    func closesReasoning(_ gen: String) -> Bool {
        gen.contains(reasoningClose)
    }

    func startsInReasoning(genPrompt: String, enabled: Bool) -> Bool {
        var result = false
        if enabled {
            if closesReasoning(genPrompt) {
                result = false
            } else if opensReasoning(genPrompt) {
                result = true
            } else {
                result = !derivedReasoning
            }
        }
        return result
    }

    var penaltyExemptCandidates: [String] {
        var out: [String] = []
        for marker in [reasoningOpen, reasoningClose, toolCallOpen,
                       toolCallClose, turnOpen, turnClose]
        where !marker.isEmpty {
            out.append(marker)
            out.append(contentsOf: ChatWire.tags(marker))
        }
        return out
    }

    private static func tags(_ s: String) -> [String] {
        var out: [String] = []
        var rest = Substring(s)
        while let lt = rest.firstIndex(of: "<") {
            let tail = rest[lt...]
            if let gt = tail.firstIndex(of: ">") {
                out.append(String(tail[...gt]))
                rest = tail[tail.index(after: gt)...]
            } else {
                rest = rest[rest.endIndex...]
            }
        }
        return out
    }
}
