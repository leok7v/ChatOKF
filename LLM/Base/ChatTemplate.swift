import Foundation

private struct NodeArena {
    var isObj: [Bool] = []
    var objs: [[(String, JinjaValue)]] = []
    var arrs: [[JinjaValue]] = []
    var json: [String] = []

    mutating func obj(_ entries: [(String, JinjaValue)],
                      _ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(true)
        objs.append(entries)
        arrs.append([])
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }

    mutating func arr(_ items: [JinjaValue],
                      _ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(false)
        objs.append([])
        arrs.append(items)
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }

    mutating func raw(_ jsonText: String) -> JinjaValue {
        let h = isObj.count
        isObj.append(true)
        objs.append([])
        arrs.append([])
        json.append(jsonText)
        return .node(handle: h, tag: AgentJinjaHost.tag)
    }
}

private func jsonEscape(_ s: String) -> String {
    var out = ""
    for ch in s.unicodeScalars {
        switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.unicodeScalars.append(ch)
        }
    }
    return out
}

private func jsonScalar(_ v: JinjaValue, _ arena: NodeArena) -> String {
    switch v {
    case .str(let s): return "\"" + jsonEscape(s) + "\""
    case .int(let n): return String(n)
    case .bool(let b): return b ? "true" : "false"
    case .node(let h, _): return arena.json[h]
    default: return "null"
    }
}

private func buildObj(_ pairs: [(String, JinjaValue)],
                      _ arena: inout NodeArena) -> JinjaValue {
    var parts: [String] = []
    for (k, v) in pairs {
        parts.append("\"" + jsonEscape(k) + "\": " + jsonScalar(v, arena))
    }
    return arena.obj(pairs, "{" + parts.joined(separator: ", ") + "}")
}

private func buildArr(_ items: [JinjaValue],
                      _ arena: inout NodeArena) -> JinjaValue {
    var parts: [String] = []
    for v in items {
        parts.append(jsonScalar(v, arena))
    }
    return arena.arr(items, "[" + parts.joined(separator: ", ") + "]")
}

// Qwen serializes the schema whole (`| tojson`), so the original text stays
// verbatim (a re-serialization would shift the precook stamp); gemma WALKS it.
private func buildParams(_ json: String,
                         _ arena: inout NodeArena) -> JinjaValue {
    var result = arena.raw(json)
    if let obj = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
       let dict = obj as? [String: Any] {
        var pairs: [(String, JinjaValue)] = []
        for (k, v) in dict {
            pairs.append((k, buildJSON(v, &arena)))
        }
        result = arena.obj(pairs, json)
    }
    return result
}

private func buildJSON(_ v: Any, _ arena: inout NodeArena) -> JinjaValue {
    var result = JinjaValue.none
    if let s = v as? String {
        result = .str(s)
    } else if let n = v as? NSNumber {
        result = CFGetTypeID(n) == CFBooleanGetTypeID()
            ? .bool(n.boolValue) : .int(n.intValue)
    } else if let d = v as? [String: Any] {
        var pairs: [(String, JinjaValue)] = []
        for (k, item) in d {
            pairs.append((k, buildJSON(item, &arena)))
        }
        result = buildObj(pairs, &arena)
    } else if let a = v as? [Any] {
        var items: [JinjaValue] = []
        for item in a {
            items.append(buildJSON(item, &arena))
        }
        result = buildArr(items, &arena)
    }
    return result
}

private func buildTool(_ t: ToolSpec,
                       _ arena: inout NodeArena) -> JinjaValue {
    let params = buildParams(t.parametersJSON, &arena)
    let fn = buildObj([
        ("name", .str(t.name)),
        ("description", .str(t.description)),
        ("parameters", params),
    ], &arena)
    return buildObj([
        ("type", .str("function")),
        ("function", fn),
    ], &arena)
}

// Both shapes at once: Qwen reads the flat name/arguments, gemma reads the
// OpenAI nesting and RAISES when it is absent. The key sets do not collide.
private func buildToolCall(_ c: AgentToolCall,
                           _ arena: inout NodeArena) -> JinjaValue {
    var args: [(String, JinjaValue)] = []
    for a in c.arguments {
        args.append((a.name, .str(a.value)))
    }
    let argObj = buildObj(args, &arena)
    let fn = buildObj([("name", .str(c.name)),
                       ("arguments", argObj)], &arena)
    return buildObj([("name", .str(c.name)),
                     ("arguments", argObj),
                     ("type", .str("function")),
                     ("function", fn)], &arena)
}

private func buildContentParts(_ parts: [ContentPart],
                               _ arena: inout NodeArena) -> JinjaValue {
    var items: [JinjaValue] = []
    for part in parts {
        var item = JinjaValue.undefined
        switch part {
        case .image:
            item = buildObj([("type", .str("image"))], &arena)
        case .audio:
            item = buildObj([("type", .str("audio"))], &arena)
        case .video:
            item = buildObj([("type", .str("video"))], &arena)
        case .text(let text):
            item = buildObj([("type", .str("text")),
                             ("text", .str(text))], &arena)
        }
        items.append(item)
    }
    return buildArr(items, &arena)
}

private func buildMessage(_ m: AgentMessage,
                          _ arena: inout NodeArena) -> JinjaValue {
    var contentVal = JinjaValue.str(m.content)
    if let parts = m.contentParts {
        contentVal = buildContentParts(parts, &arena)
    }
    var pairs: [(String, JinjaValue)] = [
        ("role", .str(m.role)),
        ("content", contentVal),
    ]
    if let reasoning = m.reasoning {
        pairs.append(("reasoning_content", .str(reasoning)))
        pairs.append(("reasoning", .str(reasoning)))
    }
    if let name = m.name {
        pairs.append(("name", .str(name)))
    }
    if !m.toolCalls.isEmpty {
        var calls: [JinjaValue] = []
        for c in m.toolCalls {
            calls.append(buildToolCall(c, &arena))
        }
        pairs.append(("tool_calls", buildArr(calls, &arena)))
    }
    return buildObj(pairs, &arena)
}

final class AgentJinjaHost: JinjaHost {
    static let tag = 0x5157      // reserved handle tag ('QW'); != dict/builtin
    private let arena: NodeArena
    let messagesValue: JinjaValue
    let toolsValue: JinjaValue

    init(messages: [AgentMessage], tools: [ToolSpec]) {
        var a = NodeArena()
        var msgs: [JinjaValue] = []
        for m in messages {
            msgs.append(buildMessage(m, &a))
        }
        var tls: [JinjaValue] = []
        for t in tools {
            tls.append(buildTool(t, &a))
        }
        self.messagesValue = buildArr(msgs, &a)
        self.toolsValue = buildArr(tls, &a)
        self.arena = a
    }

    private func handle(_ v: JinjaValue) -> Int? {
        var result: Int? = nil
        if case .node(let h, let t) = v, t == AgentJinjaHost.tag {
            result = h
        }
        return result
    }

    func get(_ obj: JinjaValue, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj), arena.isObj[h] {
            for entry in arena.objs[h] where entry.0 == name {
                result = entry.1
            }
        }
        return result
    }

    func index(_ obj: JinjaValue, _ i: Int) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj) {
            result = arena.isObj[h]
                ? .str(arena.objs[h][i].0) : arena.arrs[h][i]
        }
        return result
    }

    func len(_ obj: JinjaValue) -> Int {
        var result = 0
        if let h = handle(obj) {
            result = arena.isObj[h]
                ? arena.objs[h].count : arena.arrs[h].count
        }
        return result
    }

    func truthy(_ v: JinjaValue) -> Bool {
        var result = false
        if let h = handle(v) {
            result = arena.isObj[h]
                ? !arena.objs[h].isEmpty : !arena.arrs[h].isEmpty
        }
        return result
    }

    func test(_ v: JinjaValue, _ name: String) -> Bool {
        var result = false
        if let h = handle(v) {
            if name == "mapping" {
                result = arena.isObj[h]
            } else if name == "sequence" || name == "iterable" {
                result = !arena.isObj[h]
            }
        }
        return result
    }

    func method(_ obj: JinjaValue, _ name: String,
                _ arg: JinjaValue) -> JinjaValue {
        var result = JinjaValue.undefined
        if name == "tojson", let h = handle(obj) {
            result = .str(arena.json[h])
        }
        return result
    }
}

public func renderPrompt(template: String, messages: [AgentMessage],
                         tools: [ToolSpec], addGenerationPrompt: Bool,
                         enableThinking: Bool,
                         reasoningEffort: String? = nil,
                         addVisionId: Bool = false,
                         bosToken: String = "") throws -> String {
    let host = AgentJinjaHost(messages: messages, tools: tools)
    return try jinjaRender(template, host: host, vars: [
        ("messages", host.messagesValue),
        ("tools", host.toolsValue),
        ("add_generation_prompt", .bool(addGenerationPrompt)),
        ("enable_thinking", .bool(enableThinking)),
        ("reasoning_effort",
         reasoningEffort.map { level in JinjaValue.str(level) } ?? .undefined),
        ("add_vision_id", .bool(addVisionId)),
        ("bos_token", .str(bosToken)),
    ])
}

// Ascending. Qwen3.8 RAISES on any other word, which callers swallow as "".
public func templateEffortLevels(_ template: String) -> [String] {
    let probe = [AgentMessage(role: "user", content: "x")]
    var levels: [String] = []
    var renders: [String] = []
    for level in ["low", "medium", "high", "xhigh"] {
        let text = (try? renderPrompt(
            template: template, messages: probe, tools: [],
            addGenerationPrompt: true, enableThinking: true,
            reasoningEffort: level)) ?? ""
        if !text.isEmpty, !renders.contains(text) {
            levels.append(level)
            renders.append(text)
        }
    }
    return levels
}

public func templateTakesReasoningEffort(_ template: String) -> Bool {
    templateEffortLevels(template).count > 1
}

public func effortSpelling(_ name: String, slot: Int,
                           in levels: [String]) -> String {
    var out = name
    if !levels.isEmpty, !levels.contains(name) {
        var index = levels.count / 2
        if slot <= 0 { index = 0 }
        if slot >= 2 { index = levels.count - 1 }
        out = levels[index]
    }
    return out
}

public func templateSupportsThinking(_ template: String) -> Bool {
    ChatWire.derive(template).reasons(template: template)
}
