import Foundation
import LLM

public struct MemoryToolRunner: ToolRunner {
    let inner: SafeToolRunner
    let memories: Memories

    public init(inner: SafeToolRunner, memories: Memories) {
        self.inner = inner
        self.memories = memories
    }

    public var tools: [ToolSpec] { inner.tools + MemoryTools.specs }

    public func execute(_ name: String, _ args: [ToolArg]) async -> String {
        var out = ""
        if name.hasPrefix("memory_") {
            out = await MainActor.run { memories.tool(name, args) }
        } else {
            out = await inner.execute(name, args)
        }
        return out
    }

    public func beginTurn() {
        inner.beginTurn()
    }
}

public enum MemoryTools {

    public static let names = ["memory_search", "memory_read", "memory_create",
                               "memory_update", "memory_forget"]

    static let specs: [ToolSpec] = [
        ToolSpec(
            name: "memory_search",
            description: "Search the user's own notes by meaning AND exact "
                + "wording; returns id, title and description of every note "
                + "that is about the question, and nothing else. Ask a full "
                + "question, not a keyword. Use what comes back only when it "
                + "answers the user's latest message; when it says no note "
                + "is about this, answer from your own knowledge and say the "
                + "notes do not cover it. Put rephrasings in `also` rather "
                + "than searching again; narrow with `area` only when the "
                + "map shows the answer is there.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"query\":{\"type\":\"string\"},"
                + "\"also\":{\"type\":\"string\",\"description\":\"Other "
                + "phrasings, separated by ;\"},"
                + "\"area\":{\"type\":\"string\"},"
                + "\"limit\":{\"type\":\"integer\"}},"
                + "\"required\":[\"query\"]}"),
        ToolSpec(
            name: "memory_read",
            description: "Read one of the user's notes by id, with its links. "
                + "Always pass `about`: a long note then returns the part "
                + "answering it rather than its opening. Capped to `limit` "
                + "bytes (default 2048); `offset` pages the rest.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"},"
                + "\"about\":{\"type\":\"string\"},"
                + "\"limit\":{\"type\":\"integer\"},"
                + "\"offset\":{\"type\":\"integer\"}},"
                + "\"required\":[\"id\"]}"),
        ToolSpec(
            name: "memory_create",
            description: "Record something NEW and durable about the user, "
                + "only when they ask you to remember it; having read a "
                + "document or a page is not such an ask. Fails if the id "
                + "exists or the note restates one on file, so memory_search "
                + "first and use memory_update when the subject is already "
                + "covered. Ids are area/name, e.g. garden/tomatoes. Tag "
                + "medical, financial or address facts `private`.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"},"
                + "\"type\":{\"type\":\"string\"},"
                + "\"title\":{\"type\":\"string\"},"
                + "\"description\":{\"type\":\"string\",\"description\":"
                + "\"One sentence; this is what search matches on.\"},"
                + "\"tags\":{\"type\":\"string\"},"
                + "\"body\":{\"type\":\"string\"}},"
                + "\"required\":[\"id\",\"type\",\"title\",\"description\","
                + "\"body\"]}"),
        ToolSpec(
            name: "memory_update",
            description: "REPLACE one of the user's notes whole, only when "
                + "they ask. Fails if the id does not exist. memory_read it "
                + "first and carry forward everything worth keeping, "
                + "including its links, because nothing you omit survives.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"},"
                + "\"type\":{\"type\":\"string\"},"
                + "\"title\":{\"type\":\"string\"},"
                + "\"description\":{\"type\":\"string\"},"
                + "\"tags\":{\"type\":\"string\"},"
                + "\"body\":{\"type\":\"string\"}},"
                + "\"required\":[\"id\",\"type\",\"title\",\"description\","
                + "\"body\"]}"),
        ToolSpec(
            name: "memory_forget",
            description: "Retire one of the user's notes: it leaves search "
                + "but its links still resolve. Only when the user says it "
                + "is obsolete, never to tidy up.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"}},\"required\":[\"id\"]}"),
    ]

    static let glyphs: [String: (label: String, symbol: String)] = [
        "memory_search": ("Memory Search", "brain"),
        "memory_read": ("Memory Read", "text.book.closed"),
        "memory_create": ("Remember", "square.and.pencil"),
        "memory_update": ("Update Memory", "square.and.pencil"),
        "memory_forget": ("Forget", "eraser"),
    ]

    static func arg(_ args: [ToolArg], _ name: String) -> String? {
        args.first { arg in arg.name == name }?.value
    }

    static func list(_ text: String?, _ separator: Character) -> [String] {
        (text ?? "").split(separator: separator)
            .map { part in part.trimmingCharacters(in: .whitespaces) }
            .filter { part in !part.isEmpty }
    }

    static func slug(_ text: String, words: Int) -> String {
        text.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .prefix(words).joined(separator: "-")
    }

    static func repaired(_ heading: String, _ title: String) -> String {
        let parts = heading.split(separator: "/",
                                  omittingEmptySubsequences: false)
        var out = heading
        if parts.count == 2 {
            out = slug(String(parts[0]), words: 2) + "/"
                + slug(String(parts[1]), words: 6)
        } else if parts.count == 1, heading.split(separator: " ").count <= 2 {
            out = slug(heading, words: 2) + "/" + slug(title, words: 6)
        }
        return out
    }

    static func validId(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { part in
            !part.isEmpty && part.allSatisfy { c in
                c.isLetter || c.isNumber || c == "-" || c == "_"
            }
        }
    }
}
