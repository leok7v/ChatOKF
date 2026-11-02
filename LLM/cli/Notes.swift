import Foundation
import LLM

final class NotesToolRunner: ToolRunner, @unchecked Sendable {
    let inner: SafeToolRunner
    let store: Store
    private let lock = NSLock()

    static var defaultEncoder: String {
        EncoderPack.path ?? "(no encoder; pass --encoder-folder)"
    }

    struct Names {
        let search: String
        let find: String
        let read: String
    }

    static let nameSets: [String: Names] = [
        "memory": Names(search: "memory_search", find: "memory_find",
                        read: "memory_read"),
        "notes": Names(search: "search_notes", find: "find_in_notes",
                       read: "read_note"),
        "grep": Names(search: "search_notes", find: "grep_notes",
                      read: "read_note"),
    ]

    let names: Names

    init?(root: String, encoder: String, ingest: String?, network: Bool,
          names: String) {
        guard let chosen = NotesToolRunner.nameSets[names] else {
            err("[notes] no name set \(names); one of "
                + NotesToolRunner.nameSets.keys.sorted().joined(
                    separator: ", ") + "\n")
            return nil
        }
        self.names = chosen
        guard let embedder = ModernBertEmbedder.load(ggufPath: encoder) else {
            err("[notes] no encoder at \(encoder)\n")
            return nil
        }
        inner = SafeToolRunner(network: network)
        let url = URL(fileURLWithPath: root, isDirectory: true)
        try? FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        store = Store(root: url, embedder: embedder)
        if let ingest { NotesToolRunner.ingest(store, from: ingest) }
        let t0 = Date()
        store.load()
        err(String(format: "[notes] %d notes at %@, embedded %d, %.2fs\n",
                   store.concepts.count, root, store.embeddedCount,
                   Date().timeIntervalSince(t0)))
    }

    static func ingest(_ store: Store, from dir: String) {
        let fm = FileManager.default
        let names = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { name in name.hasSuffix(".md") }.sorted()
        var count = 0
        for name in names {
            let path = dir + "/" + name
            if let text = try? String(contentsOfFile: path, encoding: .utf8) {
                let id = "docs/" + String(name.dropLast(3))
                let lines = text.components(separatedBy: "\n")
                let title = lines.first { line in line.hasPrefix("# ") }
                    .map { line in String(line.dropFirst(2)) } ?? id
                let description = lines.first { line in
                    let t = line.trimmingCharacters(in: .whitespaces)
                    return !t.isEmpty && !t.hasPrefix("#")
                        && !t.hasPrefix("<!--") && !t.hasPrefix(">")
                        && !t.hasPrefix("*") && !t.hasPrefix("|")
                }.map { line in NotesToolRunner.cut(line, limit: 200) }
                    ?? ""
                _ = try? store.write(
                    id: id, type: "Document", title: title,
                    description: description, tags: ["docs"],
                    body: NotesToolRunner.shaped(lines)
                        .joined(separator: "\n"),
                    adding: ["sources:", "  - resource: file://" + path])
                count += 1
            }
        }
        err("[notes] ingested \(count) files from \(dir)\n")
    }

    static func cut(_ text: String, limit: Int) -> String {
        let whole = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var out = whole
        if whole.count > limit {
            let head = String(whole.prefix(limit))
            var end: String.Index? = nil
            if let p = head.range(of: "\n\n", options: .backwards) {
                end = p.lowerBound
            } else {
                var i = head.startIndex
                while i < head.endIndex {
                    let next = head.index(after: i)
                    if ".!?".contains(head[i])
                        && (next == head.endIndex || head[next].isWhitespace) {
                        end = next
                    }
                    i = next
                }
            }
            if let end {
                out = String(head[..<end])
            } else if let space = head.lastIndex(of: " ") {
                out = String(head[..<space]) + " ..."
            } else {
                out = head + " ..."
            }
            out = out.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ")
        }
        return out
    }

    static func cells(_ line: String) -> [String] {
        var out = line.trimmingCharacters(in: .whitespaces)
            .split(separator: "|", omittingEmptySubsequences: false)
            .map { cell in cell.trimmingCharacters(in: .whitespaces) }
        if out.first == "" { out.removeFirst() }
        if out.last == "" { out.removeLast() }
        return out
    }

    static func isRule(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        return t.hasPrefix("|") && t.allSatisfy { c in
            c == "|" || c == "-" || c == ":" || c == " "
        }
    }

    static func shaped(_ lines: [String]) -> [String] {
        var out: [String] = []
        var header: [String] = []
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let table = line.trimmingCharacters(in: .whitespaces)
                .hasPrefix("|")
            if table && i + 1 < lines.count && isRule(lines[i + 1]) {
                let found = cells(line)
                let short = found.allSatisfy { cell in
                    cell.split(separator: " ").count <= 5
                }
                header = short ? found : []
                out.append(line)
                i += 1
            } else if table && !header.isEmpty && !isRule(line) {
                let row = cells(line)
                var parts: [String] = []
                for (k, cell) in row.enumerated() where !cell.isEmpty {
                    let name = k < header.count ? header[k] : ""
                    parts.append(name.isEmpty || k == 0
                                 ? cell : name + ": " + cell)
                }
                out.append(parts.joined(separator: " | "))
            } else {
                if !table { header = [] }
                out.append(line)
            }
            i += 1
        }
        return out
    }

    var searchSpec: ToolSpec { ToolSpec(
        name: names.search,
        description: "Search the user's notes by meaning; returns id, "
            + "title and description of every note that is about the "
            + "question. Ask a full question, not a keyword. For an exact "
            + "word, number or phrase use \(names.find) instead.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"query\":{\"type\":\"string\"},"
            + "\"limit\":{\"type\":\"integer\"}},"
            + "\"required\":[\"query\"]}") }

    var findSpec: ToolSpec { ToolSpec(
        name: names.find,
        description: "Find the lines in the user's notes that contain ALL "
            + "the given words, in any order, any letter case. Put a "
            + "phrase in double quotes to match it exactly. End a word "
            + "with * to match every word that starts so (contract* "
            + "finds contraction and contractions). Not a regex. Returns "
            + "note id, line number and the line; then \(names.read) that "
            + "note from that line to see the context.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"words\":{\"type\":\"string\",\"description\":\"The words "
            + "every matching line must contain, e.g. rule contractions\"},"
            + "\"limit\":{\"type\":\"integer\"}},"
            + "\"required\":[\"words\"]}") }

    var readSpec: ToolSpec { ToolSpec(
        name: names.read,
        description: "Read one of the user's notes by id, 40 numbered "
            + "lines at a time. Pass `line` to start at a line number "
            + "from a \(names.find) hit (start a little before it), or "
            + "`about` to start where the note answers a question. "
            + "`lines` changes how many lines come back.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"id\":{\"type\":\"string\"},"
            + "\"line\":{\"type\":\"integer\"},"
            + "\"lines\":{\"type\":\"integer\"},"
            + "\"about\":{\"type\":\"string\"}},"
            + "\"required\":[\"id\"]}") }

    var tools: [ToolSpec] {
        inner.tools + [searchSpec, findSpec, readSpec]
    }

    private var readEnds: [String: Int] = [:]

    func beginTurn() {
        inner.beginTurn()
        locked { readEnds = [:] }
    }

    func execute(_ name: String, _ args: [ToolArg]) async -> String {
        var out = ""
        switch name {
        case names.search: out = locked { search(args) }
        case names.find: out = locked { find(args) }
        case names.read: out = locked { read(args) }
        case "fetch_url", "wikipedia_query":
            if let kept = locked({ fresh(name, args) }) {
                out = kept
            } else {
                out = await inner.execute(name, args)
                out = locked { keep(name, args, out) }
            }
        default: out = await inner.execute(name, args)
        }
        return out
    }

    static let captionInstruction = "Describe the picture in this "
        + "conversation in one sentence of about 25 words: what it shows, "
        + "any text in it, and what it seems to be for. Write only that "
        + "sentence."

    func keepPicture(path: String, caption: String, answer: String) {
        let name = (path as NSString).lastPathComponent
        let id = "pictures/" + NotesToolRunner.slug(name)
        let description = NotesToolRunner.cut(caption, limit: 200)
        let body = "Picture " + name + "\n\n" + caption + "\n\nWhat was "
            + "said about it:\n" + answer
        let adding = [
            NotesToolRunner.fetchedKey + " " + Store.iso.string(from: Date()),
            "sources:", "  - resource: file://" + path,
        ]
        let wrote = locked {
            (try? store.write(id: id, type: "Picture", title: name,
                              description: description, tags: ["pictures"],
                              body: body, adding: adding)) != nil
        }
        if wrote {
            locked { store.reload(id: id) }
            err("[notes] saved \(id): \(description)\n")
        }
    }

    static let freshDays = 7.0
    static let fetchedKey = "fetched:"

    static func slug(_ text: String) -> String {
        var out = ""
        var dash = false
        for c in text.lowercased() where out.count < 60 {
            if c.isLetter || c.isNumber {
                out.append(c)
                dash = false
            } else if !dash && !out.isEmpty {
                out.append("-")
                dash = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return out
    }

    static func pageId(_ name: String, _ args: [ToolArg]) -> String? {
        var out: String? = nil
        if name == "wikipedia_query", let q = arg(args, "query") {
            out = "pages/wiki-" + slug(q)
        } else if let url = arg(args, "url") {
            let bare = url.replacingOccurrences(of: "https://", with: "")
                .replacingOccurrences(of: "http://", with: "")
                .replacingOccurrences(of: "www.", with: "")
            out = "pages/" + slug(bare)
        }
        let offset = Int(arg(args, "offset") ?? "0") ?? 0
        return offset == 0 && out != nil && out != "pages/" ? out : nil
    }

    static func fetchedAt(_ concept: Concept) -> Date? {
        var out: Date? = nil
        for line in concept.extraFrontmatter
        where line.hasPrefix(NotesToolRunner.fetchedKey) {
            out = Store.iso.date(from: String(line.dropFirst(
                NotesToolRunner.fetchedKey.count))
                .trimmingCharacters(in: .whitespaces))
        }
        return out
    }

    private func fresh(_ name: String, _ args: [ToolArg]) -> String? {
        var out: String? = nil
        if let id = NotesToolRunner.pageId(name, args),
           let concept = store.concept(id),
           let at = NotesToolRunner.fetchedAt(concept),
           Date().timeIntervalSince(at)
               < NotesToolRunner.freshDays * 86400 {
            let day = String(Store.iso.string(from: at).prefix(10))
            out = "From the user's notes, fetched on \(day) (note \(id)):"
                + "\n" + concept.body + "\n"
        }
        return out
    }

    private func keep(_ name: String, _ args: [ToolArg],
                      _ text: String) -> String {
        var out = text
        let failed = text.hasPrefix("error") || text.hasPrefix("No ")
            || text.hasPrefix("Article ") || text.hasPrefix("This exact")
        if !failed, let id = NotesToolRunner.pageId(name, args) {
            let lines = text.components(separatedBy: "\n")
            var title = NotesToolRunner.arg(args, "url")
                ?? NotesToolRunner.arg(args, "query") ?? id
            if let first = lines.first, let open = first.firstIndex(of: "\""),
               let close = first.lastIndex(of: "\""), open < close {
                title = String(first[first.index(after: open)..<close])
            }
            let description = NotesToolRunner.cut(
                lines.dropFirst().joined(separator: "\n"), limit: 200)
            let source = NotesToolRunner.arg(args, "url")
                ?? "wikipedia:" + (NotesToolRunner.arg(args, "query") ?? "")
            let adding = [
                NotesToolRunner.fetchedKey + " "
                    + Store.iso.string(from: Date()),
                "sources:", "  - resource: " + source,
            ]
            if (try? store.write(id: id, type: "Page", title: title,
                                 description: description, tags: ["pages"],
                                 body: text, adding: adding)) != nil {
                store.reload(id: id)
                out += "\n[saved as note \(id)]"
            }
        }
        return out
    }

    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    static func arg(_ args: [ToolArg], _ name: String) -> String? {
        args.first { arg in arg.name == name }?.value
    }

    private func search(_ args: [ToolArg]) -> String {
        let query = NotesToolRunner.arg(args, "query") ?? ""
        let limit = Int(NotesToolRunner.arg(args, "limit") ?? "") ?? 8
        var out = "error: \(names.search) needs a query"
        if !query.isEmpty {
            let found = store.search([query], filter: Filter(),
                                     limit: max(1, limit))
            let shown = found.hits.filter { hit in store.relevant(hit) }
            let content = NotesToolRunner.terms(query).filter { term in
                if case .word(let w) = term {
                    return !NotesToolRunner.stops.contains(w)
                }
                return true
            }
            if shown.isEmpty {
                out = "no note is about this as a whole\n"
            } else {
                out = "notes about this, best first:\n"
                for hit in shown { out += StoreText.line(store, hit) }
            }
            if !content.isEmpty {
                out += "lines anywhere in the notes that contain the "
                    + "words of the question:\n"
                    + lines(content, asked: query, limit: 10)
            }
        }
        return out
    }

    static func haystack(_ concept: Concept) -> [String] {
        (concept.title + "\n" + concept.description + "\n" + concept.body)
            .components(separatedBy: "\n")
    }

    enum Term {
        case phrase(String)
        case word(String)
        case prefix(String)
    }

    static let separators: Set<Character> = ["[", "]", ",", ";"]

    static func terms(_ text: String) -> [Term] {
        var out: [Term] = []
        var current = ""
        var quoted = false
        func flush() {
            let lower = current.lowercased()
            let plain = lower.allSatisfy { c in c.isLetter || c.isNumber }
            if lower.isEmpty {
            } else if quoted || (!plain && !lower.hasSuffix("*")) {
                out.append(.phrase(lower))
            } else if lower.hasSuffix("*") {
                let stem = String(lower.dropLast())
                if !stem.isEmpty { out.append(.prefix(stem)) }
            } else {
                out.append(.word(NotesToolRunner.stem(lower)))
            }
            current = ""
        }
        for c in text {
            if c == "\"" {
                flush()
                quoted.toggle()
            } else if (c.isWhitespace || NotesToolRunner.separators
                        .contains(c)) && !quoted {
                flush()
            } else {
                current.append(c)
            }
        }
        quoted = false
        flush()
        return out
    }

    static let sibilants = ["sses", "shes", "ches", "xes", "zes"]

    static func stem(_ word: String) -> String {
        var out = word
        if word.count > 4 && word.hasSuffix("ies") {
            out = String(word.dropLast(3)) + "y"
        } else if word.count > 4 && NotesToolRunner.sibilants.contains(
                    where: { end in word.hasSuffix(end) }) {
            out = String(word.dropLast(2))
        } else if word.count > 3 && word.hasSuffix("s")
                    && !word.hasSuffix("ss") {
            out = String(word.dropLast())
        }
        return out
    }

    struct Token {
        let text: String
        let line: Int
    }

    static func joins(_ chars: [Character], _ i: Int) -> Bool {
        let c = chars[i]
        var out = c.isLetter || c.isNumber
        if !out && i > 0 && i + 1 < chars.count {
            let before = chars[i - 1]
            let after = chars[i + 1]
            out = (c == "." && before.isNumber && after.isNumber)
                || ((c == "'" || c == "\u{2019}") && before.isLetter
                    && after.isLetter)
        }
        return out
    }

    static func tokens(_ lines: [String]) -> [Token] {
        var out: [Token] = []
        for (n, line) in lines.enumerated() {
            let chars = Array(line.lowercased())
            var current = ""
            for i in chars.indices {
                if NotesToolRunner.joins(chars, i) {
                    current.append(chars[i])
                } else if !current.isEmpty {
                    out.append(Token(text: current.first!.isNumber
                                     ? current : stem(current), line: n))
                    current = ""
                }
            }
            if !current.isEmpty {
                out.append(Token(text: current.first!.isNumber
                                 ? current : stem(current), line: n))
            }
        }
        return out
    }

    static func positions(_ term: Term, in tokens: [Token]) -> [Int] {
        var out: [Int] = []
        switch term {
        case .word(let w):
            for (i, t) in tokens.enumerated() where t.text == w {
                out.append(i)
            }
        case .prefix(let p):
            for (i, t) in tokens.enumerated() where t.text.hasPrefix(p) {
                out.append(i)
            }
        case .phrase(let text):
            let seq = NotesToolRunner.tokens([text]).map { t in t.text }
            if !seq.isEmpty && tokens.count >= seq.count {
                for i in 0...(tokens.count - seq.count) {
                    var same = true
                    for k in seq.indices where same {
                        same = tokens[i + k].text == seq[k]
                    }
                    if same { out.append(i) }
                }
            }
        }
        return out
    }

    static let maxSpan = 32
    static let lineCost = 8
    static let windowsPerNote = 16
    static let maxTerms = 8

    static func span(_ a: Int, _ b: Int, _ tokens: [Token]) -> Int {
        b - a + NotesToolRunner.lineCost * (tokens[b].line - tokens[a].line)
    }

    static func window(_ lists: [[Int]], used: Set<Int>, _ tokens: [Token])
        -> (start: Int, end: Int, hits: Int)? {
        var merged: [(pos: Int, list: Int)] = []
        for (k, list) in lists.enumerated() {
            for pos in list where !used.contains(pos) {
                merged.append((pos, k))
            }
        }
        merged.sort { a, b in a.pos < b.pos }
        var counts = [Int](repeating: 0, count: lists.count)
        var covered = 0
        var best: (start: Int, end: Int, hits: Int)? = nil
        var lo = 0
        for hi in merged.indices {
            counts[merged[hi].list] += 1
            if counts[merged[hi].list] == 1 { covered += 1 }
            while covered == lists.count {
                let span = NotesToolRunner.span(merged[lo].pos,
                                                merged[hi].pos, tokens)
                if span <= NotesToolRunner.maxSpan
                    && (best == nil || span < NotesToolRunner.span(
                        best!.start, best!.end, tokens)) {
                    best = (merged[lo].pos, merged[hi].pos, hi - lo + 1)
                }
                counts[merged[lo].list] -= 1
                if counts[merged[lo].list] == 0 { covered -= 1 }
                lo += 1
            }
        }
        return best
    }

    struct Found {
        let id: String
        let lines: [String]
        let line: Int
        let endLine: Int
        let matched: Int
        let span: Int
        let hits: Int
        let lineHits: Int
    }


    func found(_ terms: [Term], atLeast need: Int) -> [Found] {
        var out: [Found] = []
        let n = terms.count
        for concept in store.concepts {
            let lines = NotesToolRunner.haystack(concept)
            let tokens = NotesToolRunner.tokens(lines)
            let lists = terms.map { term in
                NotesToolRunner.positions(term, in: tokens)
            }
            var used = Set<Int>()
            var taken = 0
            var more = true
            while more && taken < NotesToolRunner.windowsPerNote {
                var best: (start: Int, end: Int, hits: Int)? = nil
                for mask in 1..<(1 << n) where mask.nonzeroBitCount == need {
                    let subset = (0..<n).filter { k in mask & (1 << k) != 0 }
                        .map { k in lists[k] }
                    if let w = NotesToolRunner.window(subset, used: used,
                                                      tokens) {
                        let wide = NotesToolRunner.span(w.start, w.end, tokens)
                        let known = best.map { b in
                            NotesToolRunner.span(b.start, b.end, tokens)
                        } ?? Int.max
                        if wide < known || (wide == known
                                            && w.hits > best!.hits) {
                            best = w
                        }
                    }
                }
                if let w = best {
                    let first = tokens[w.start].line
                    let last = tokens[w.end].line
                    for p in lists.joined()
                    where tokens[p].line >= first && tokens[p].line <= last {
                        used.insert(p)
                    }
                    let onLine = lists.joined().filter { p in
                        tokens[p].line == first
                    }.count
                    out.append(Found(
                        id: concept.id, lines: lines, line: first + 1,
                        endLine: last + 1, matched: need,
                        span: NotesToolRunner.span(w.start, w.end, tokens),
                        hits: w.hits, lineHits: onLine))
                    taken += 1
                } else {
                    more = false
                }
            }
        }
        return out.sorted { a, b in
            (-a.matched, a.span, -a.lineHits, -a.hits, a.id, a.line)
                < (-b.matched, b.span, -b.lineHits, -b.hits, b.id, b.line)
        }
    }

    static let stops: Set<String> = [
        "the", "a", "an", "in", "of", "for", "to", "is", "are", "what",
        "how", "many", "may", "does", "do", "which", "about", "i", "can",
        "say", "it", "my", "me", "that", "this", "with", "on", "and", "or",
        "be", "was", "were", "have", "has", "should", "would", "could",
        "there", "any", "some", "tell", "please", "know", "specification",
    ]

    private func lines(_ terms: [Term], asked: String, limit: Int)
        -> String {
        let terms = Array(terms.prefix(NotesToolRunner.maxTerms))
        var out = "no place in the user's notes holds " + asked
            + " close together; try other words\n"
        var need = terms.count
        var hits = found(terms, atLeast: need)
        let floor = max(1, (terms.count + 1) / 2)
        while hits.isEmpty && need > floor {
            need -= 1
            hits = found(terms, atLeast: need)
        }
        if !hits.isEmpty {
            out = need < terms.count
                ? "no place holds all \(terms.count) words close together; "
                    + "these hold \(need) of them:\n" : ""
            var shown = 0
            for hit in NotesToolRunner.interleaved(hits) where shown < limit {
                var last = min(hit.endLine, hit.line + 2)
                let heading = hit.lines[last - 1]
                    .trimmingCharacters(in: .whitespaces).hasPrefix("#")
                if heading {
                    var next = last + 1
                    while next <= hit.lines.count && hit.lines[next - 1]
                        .trimmingCharacters(in: .whitespaces).isEmpty {
                        next += 1
                    }
                    if next <= hit.lines.count { last = next }
                }
                for n in hit.line...last where shown < limit {
                    let text = hit.lines[n - 1]
                        .trimmingCharacters(in: .whitespaces)
                    let blank = text.isEmpty || text == ">"
                    if !blank {
                        out += "\(hit.id):\(n)  " + String(text.prefix(160))
                            + "\n"
                        shown += 1
                    }
                }
            }
            if hits.count > shown {
                out += "\(hits.count - shown) more place(s) not shown; "
                    + "add a word to narrow\n"
            }
        }
        return out
    }

    static func interleaved(_ hits: [Found]) -> [Found] {
        var byNote: [[Found]] = []
        var at: [String: Int] = [:]
        for hit in hits {
            if let k = at[hit.id] {
                byNote[k].append(hit)
            } else {
                at[hit.id] = byNote.count
                byNote.append([hit])
            }
        }
        var out: [Found] = []
        var round = 0
        var more = true
        while more {
            more = false
            for list in byNote where round < list.count {
                out.append(list[round])
                more = true
            }
            round += 1
        }
        return out
    }

    private func find(_ args: [ToolArg]) -> String {
        let asked = NotesToolRunner.arg(args, "words") ?? ""
        let limit = Int(NotesToolRunner.arg(args, "limit") ?? "") ?? 30
        let terms = NotesToolRunner.terms(asked)
        var out = "error: \(names.find) needs one or more words"
        if !terms.isEmpty { out = lines(terms, asked: asked, limit: limit) }
        return out
    }

    static func debugFind(_ runner: NotesToolRunner, _ words: String) {
        print(runner.find([ToolArg(name: "words", value: words)]))
        let terms = NotesToolRunner.terms(words)
        for f in runner.found(terms, atLeast: terms.count).prefix(12) {
            err("\(f.id):\(f.line)-\(f.endLine) span \(f.span) "
                + "lineHits \(f.lineHits) hits \(f.hits)\n")
        }
    }

    private func read(_ args: [ToolArg]) -> String {
        var id = NotesToolRunner.arg(args, "id") ?? ""
        var glued: Int? = nil
        if store.concept(id) == nil, let colon = id.lastIndex(of: ":"),
           let n = Int(id[id.index(after: colon)...]) {
            glued = n
            id = String(id[..<colon])
        }
        if store.concept(id) == nil {
            let leaf = (id as NSString).lastPathComponent.lowercased()
            let same = store.concepts.filter { concept in
                (concept.id as NSString).lastPathComponent.lowercased()
                    == leaf
            }
            if same.count == 1 { id = same[0].id }
        }
        var out = "error: no such note: " + id
        if store.concept(id) == nil {
            out += "; ids are exactly as \(names.find) and "
                + "\(names.search) print them"
        }
        if let concept = store.concept(id) {
            let lines = NotesToolRunner.haystack(concept)
            let count = max(1, Int(NotesToolRunner.arg(args, "lines") ?? "")
                            ?? 40)
            var start = max(1, glued
                            ?? Int(NotesToolRunner.arg(args, "line") ?? "")
                            ?? 1)
            let about = NotesToolRunner.arg(args, "about") ?? ""
            if !about.isEmpty, NotesToolRunner.arg(args, "line") == nil {
                let span = store.window(concept, about: about, limit: 2048)
                let before = String(decoding: Array(concept.body.utf8)
                                        .prefix(span.from), as: UTF8.self)
                start = 3 + before.filter { c in c == "\n" }.count
            }
            let key = id + "#" + about + "#" + String(start)
            if let last = readEnds[key], last < lines.count {
                start = last + 1
            }
            start = min(start, lines.count)
            let end = min(lines.count, start + count - 1)
            readEnds[key] = end
            out = "# \(concept.title) (\(id), \(lines.count) lines)\n"
            for i in start...end {
                out += "\(i): " + lines[i - 1] + "\n"
            }
            if end < lines.count {
                out += "[lines \(start)-\(end) of \(lines.count); "
                    + "\(names.read) \(id) line=\(end + 1) for more]\n"
            }
        }
        return out
    }
}
