import Foundation

public enum NoteFind {

    public enum Term: Equatable {
        case phrase(String)
        case word(String)
        case prefix(String)
    }

    static let separators: Set<Character> = ["[", "]", ",", ";"]
    static let sibilants = ["sses", "shes", "ches", "xes", "zes"]
    static let maxSpan = 32
    static let lineCost = 8
    static let windowsPerNote = 16
    static let maxTerms = 8
    public static let lineLimit = 30
    public static let lineWidth = 160
    public static let readLines = 40

    public static func terms(_ text: String) -> [Term] {
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
                out.append(.word(NoteFind.stem(lower)))
            }
            current = ""
        }
        for c in text {
            if c == "\"" {
                flush()
                quoted.toggle()
            } else if (c.isWhitespace || NoteFind.separators.contains(c))
                        && !quoted {
                flush()
            } else {
                current.append(c)
            }
        }
        quoted = false
        flush()
        return Array(out.prefix(NoteFind.maxTerms))
    }

    public static func stem(_ word: String) -> String {
        var out = word
        if word.count > 4 && word.hasSuffix("ies") {
            out = String(word.dropLast(3)) + "y"
        } else if word.count > 4 && NoteFind.sibilants.contains(
                    where: { end in word.hasSuffix(end) }) {
            out = String(word.dropLast(2))
        } else if word.count > 3 && word.hasSuffix("s")
                    && !word.hasSuffix("ss") {
            out = String(word.dropLast())
        }
        return out
    }

    public struct Token {
        public let text: String
        public let line: Int
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

    public static func tokens(_ lines: [String]) -> [Token] {
        var out: [Token] = []
        for (n, line) in lines.enumerated() {
            let chars = Array(line.lowercased())
            var current = ""
            for i in chars.indices {
                if NoteFind.joins(chars, i) {
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
            let seq = NoteFind.tokens([text]).map { t in t.text }
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

    static func span(_ a: Int, _ b: Int, _ tokens: [Token]) -> Int {
        b - a + NoteFind.lineCost * (tokens[b].line - tokens[a].line)
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
                let wide = NoteFind.span(merged[lo].pos, merged[hi].pos,
                                         tokens)
                if wide <= NoteFind.maxSpan
                    && (best == nil || wide < NoteFind.span(
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

    public struct Found {
        public let id: String
        public let lines: [String]
        public let line: Int
        public let endLine: Int
        public let matched: Int
        public let span: Int
        public let hits: Int
        public let lineHits: Int
    }

    public static func haystack(_ concept: Concept) -> [String] {
        (concept.title + "\n" + concept.description + "\n" + concept.body)
            .components(separatedBy: "\n")
    }

    static func found(_ terms: [Term], atLeast need: Int,
                      in concepts: [Concept]) -> [Found] {
        var out: [Found] = []
        let n = terms.count
        for concept in concepts {
            let lines = NoteFind.haystack(concept)
            let tokens = NoteFind.tokens(lines)
            let lists = terms.map { term in
                NoteFind.positions(term, in: tokens)
            }
            var used = Set<Int>()
            var taken = 0
            var more = true
            while more && taken < NoteFind.windowsPerNote {
                var best: (start: Int, end: Int, hits: Int)? = nil
                for mask in 1..<(1 << n) where mask.nonzeroBitCount == need {
                    let subset = (0..<n).filter { k in mask & (1 << k) != 0 }
                        .map { k in lists[k] }
                    if let w = NoteFind.window(subset, used: used, tokens) {
                        let wide = NoteFind.span(w.start, w.end, tokens)
                        let known = best.map { b in
                            NoteFind.span(b.start, b.end, tokens)
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
                        span: NoteFind.span(w.start, w.end, tokens),
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

    public static let stops: Set<String> = [
        "the", "a", "an", "in", "of", "for", "to", "is", "are", "what",
        "how", "many", "may", "does", "do", "which", "about", "i", "can",
        "say", "it", "my", "me", "that", "this", "with", "on", "and", "or",
        "be", "was", "were", "have", "has", "should", "would", "could",
        "there", "any", "some", "tell", "please", "know",
    ]

    static let stemmedStops = Set(NoteFind.stops.map { w in stem(w) })

    public static func contentTerms(_ query: String) -> [Term] {
        NoteFind.terms(query).filter { term in
            if case .word(let w) = term {
                return !NoteFind.stemmedStops.contains(w)
            }
            return true
        }
    }

    public static func stamp(_ data: Data) -> String {
        String(format: "%06llx", fnv1a(data) & 0xFFFFFF)
    }

    public static func lines(_ terms: [Term], asked: String, limit: Int,
                             in concepts: [Concept]) -> String {
        var out = "no place in the user's notes holds " + asked
            + " close together; try other words\n"
        var need = terms.count
        var hits = found(terms, atLeast: need, in: concepts)
        let floor = max(1, (terms.count + 1) / 2)
        while hits.isEmpty && need > floor {
            need -= 1
            hits = found(terms, atLeast: need, in: concepts)
        }
        if !hits.isEmpty {
            out = need < terms.count
                ? "no place holds all \(terms.count) words close together; "
                    + "these hold \(need) of them:\n" : ""
            var shown = 0
            var places = 0
            for hit in NoteFind.interleaved(hits) where shown < limit {
                places += 1
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
                        out += "\(hit.id):\(n)  "
                            + String(text.prefix(NoteFind.lineWidth)) + "\n"
                        shown += 1
                    }
                }
            }
            if hits.count > places {
                out += "\(hits.count - places) more place(s) not shown; "
                    + "add a word to narrow\n"
            }
        }
        return out
    }

    public static func lineOf(_ concept: Concept, byteOffset: Int) -> Int {
        let before = String(decoding: Array(concept.body.utf8)
                                .prefix(byteOffset), as: UTF8.self)
        return 3 + before.filter { c in c == "\n" }.count
    }

    public static func read(_ concept: Concept, from start: Int, count: Int,
                            more readName: String) -> (text: String,
                                                       end: Int) {
        let lines = NoteFind.haystack(concept)
        let first = min(max(1, start), lines.count)
        let end = min(lines.count, first + max(1, count) - 1)
        var out = "# \(concept.title) (\(concept.id), \(lines.count) lines)\n"
        for i in first...end {
            out += "\(i): " + lines[i - 1] + "\n"
        }
        if end < lines.count {
            out += "[lines \(first)-\(end) of \(lines.count); "
                + "\(readName) \(concept.id) line=\(end + 1) for more]\n"
        }
        return (out, end)
    }

    public static func cut(_ text: String, limit: Int) -> String {
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
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
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

    static let headerWords = 5

    public static func shaped(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")
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
                    cell.split(separator: " ").count <= NoteFind.headerWords
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
        return out.joined(separator: "\n")
    }

    public static func slug(_ text: String) -> String {
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
}
