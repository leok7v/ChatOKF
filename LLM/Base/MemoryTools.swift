import Foundation

public struct MemoryToolRunner: ToolRunner {
    let inner: SafeToolRunner
    let memories: Memories
    let writes: Bool

    public init(inner: SafeToolRunner, memories: Memories,
                writes: Bool = true) {
        self.inner = inner
        self.memories = memories
        self.writes = writes
    }

    public var tools: [ToolSpec] {
        inner.tools + MemoryTools.specs.filter { spec in
            writes || !MemoryTools.writeNames.contains(spec.name)
        }
    }

    public func execute(_ name: String, _ args: [ToolArg]) async -> String {
        var out = ""
        if MemoryTools.names.contains(name) {
            out = await memories.tool(name, args)
        } else if MemoryTools.pageTools.contains(name),
                  let kept = await memories.page(name, args) {
            out = kept
        } else {
            out = await inner.execute(name, args)
            if MemoryTools.pageTools.contains(name) {
                out = await memories.keepPage(name, args, text: out)
            }
        }
        return out
    }

    public func beginTurn() {
        inner.beginTurn()
        memories.beginTurn()
    }
}

public enum MemoryTools {

    public static let search = "search_notes"
    public static let grep = "grep_notes"
    public static let read = "read_note"
    public static let create = "create_note"
    public static let update = "update_note"
    public static let forget = "forget_note"

    public static let names = [search, grep, read, create, update, forget]

    static let writeNames: Set<String> = [create, update, forget]

    static let pageTools: Set<String> = ["fetch_url", "wikipedia_query"]

    static let specs: [ToolSpec] = [
        ToolSpec(
            name: search,
            description: "Search the user's notes by meaning; returns id, "
                + "title and description of every note that is about the "
                + "question, then the lines anywhere in the notes that hold "
                + "the words of the question. Ask a full question, not a "
                + "keyword. For an exact word, number or phrase use "
                + grep + " instead. Use what comes back only when it "
                + "answers the user's latest message; when it says no note "
                + "is about this, answer from your own knowledge and say the "
                + "notes do not cover it. Put rephrasings in `also` rather "
                + "than searching again; narrow with `area` only when the "
                + "user names one.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"query\":{\"type\":\"string\"},"
                + "\"also\":{\"type\":\"string\",\"description\":\"Other "
                + "phrasings, separated by ;\"},"
                + "\"area\":{\"type\":\"string\"},"
                + "\"limit\":{\"type\":\"integer\"}},"
                + "\"required\":[\"query\"]}"),
        ToolSpec(
            name: grep,
            description: "Find the places in the user's notes where ALL the "
                + "given words occur close together, in any order, any "
                + "letter case. Put a phrase in double quotes to match it "
                + "exactly. End a word with * to match every word that "
                + "starts so (contract* finds contraction and contractions). "
                + "Not a regex. Returns note id, line number and the line; "
                + "then " + read + " that note from that line to see the "
                + "context.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"words\":{\"type\":\"string\",\"description\":\"The words "
                + "every place must hold, e.g. rule contractions\"},"
                + "\"limit\":{\"type\":\"integer\"}},"
                + "\"required\":[\"words\"]}"),
        ToolSpec(
            name: read,
            description: "Read one of the user's notes by id, 40 numbered "
                + "lines at a time. Pass `line` to start at a line number "
                + "from a " + grep + " hit (start a little before it), or "
                + "`about` to start where the note answers a question. "
                + "`lines` changes how many lines come back; the same read "
                + "again continues where the last one ended.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"},"
                + "\"line\":{\"type\":\"integer\"},"
                + "\"lines\":{\"type\":\"integer\"},"
                + "\"about\":{\"type\":\"string\"}},"
                + "\"required\":[\"id\"]}"),
        ToolSpec(
            name: create,
            description: "Record something NEW and durable about the user, "
                + "only when they ask you to remember it; having read a "
                + "document or a page is not such an ask. Fails if the id "
                + "exists or the note restates one on file, so " + search
                + " first and use " + update + " when the subject is "
                + "already covered. Ids are area/name, e.g. garden/tomatoes. "
                + "Tag medical, financial or address facts `private`.",
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
            name: update,
            description: "REPLACE one of the user's notes whole, only when "
                + "they ask. Fails if the id does not exist. " + read
                + " it first and carry forward everything worth keeping, "
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
            name: forget,
            description: "Retire one of the user's notes: it leaves search "
                + "but its links still resolve. Only when the user says it "
                + "is obsolete, never to tidy up.",
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"id\":{\"type\":\"string\"}},\"required\":[\"id\"]}"),
    ]

    public static let glyphs: [String: (label: String, symbol: String)] = [
        search: ("Note Search", "brain"),
        grep: ("Note Grep", "text.magnifyingglass"),
        read: ("Note Read", "text.book.closed"),
        create: ("Remember", "square.and.pencil"),
        update: ("Update Note", "square.and.pencil"),
        forget: ("Forget", "eraser"),
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
        text.lowercased().filter { c in c != "'" && c != "\u{2019}" }
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .prefix(words).joined(separator: "-")
    }

    static func validId(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { part in
            !part.isEmpty && part.allSatisfy { c in
                c.isLetter || c.isNumber || c == "-" || c == "_"
            }
        }
    }

    public static func pageId(_ name: String, _ args: [ToolArg]) -> String? {
        var out: String? = nil
        if name == "wikipedia_query", let q = arg(args, "query") {
            out = "pages/wiki-" + NoteFind.slug(q)
        } else if let url = arg(args, "url") {
            let bare = url.replacingOccurrences(of: "https://", with: "")
                .replacingOccurrences(of: "http://", with: "")
                .replacingOccurrences(of: "www.", with: "")
            out = "pages/" + NoteFind.slug(bare) + "-"
                + NoteFind.stamp(Data(url.utf8))
        }
        let offset = Int(arg(args, "offset") ?? "0") ?? 0
        return offset == 0 && out != nil && out != "pages/" ? out : nil
    }

    static func pageFailed(_ text: String) -> Bool {
        text.hasPrefix("error") || text.hasPrefix("No ")
            || text.hasPrefix("Article ") || text.hasPrefix("This exact")
    }

    static func pageTitle(_ name: String, _ args: [ToolArg],
                          _ text: String) -> String {
        var title = arg(args, "url") ?? arg(args, "query") ?? ""
        if let first = text.components(separatedBy: "\n").first,
           let open = first.firstIndex(of: "\""),
           let close = first.lastIndex(of: "\""), open < close {
            title = String(first[first.index(after: open)..<close])
        }
        return title
    }
}
