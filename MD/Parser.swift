import Foundation

// <sub> and <sup> survive Apple's inline parser as literal text, so they
// are consumed here and the level left on the run for each renderer.
enum ScriptAttribute: AttributedStringKey {
    typealias Value = Int
    static let name = "md.script"
}

enum SmallAttribute: AttributedStringKey {
    typealias Value = Bool
    static let name = "md.small"
}

// Batch parser, adapted from md.too. `blocks` is the seam the streaming
// parser reuses so streaming and batch results cannot diverge.
extension Markdown {

    @TaskLocal static var currentRefs: [String: URL] = [:]
    // Parse-time math switch. `$inline$` / `$$display$$` are rendered to
    // Unicode only when true.
    @TaskLocal static var mathEnabled: Bool = true
    @TaskLocal private static var nesting = 0
    static let maxNesting = 32

    public static func parse(_ source: String,
                             math: Bool = true) -> Document {
        let raw = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let (lines, refs) = stripLinkDefinitions(raw)
        let bs = Markdown.$mathEnabled.withValue(math) {
            Markdown.$currentRefs.withValue(refs) {
                blockSpans(lines).map { pair in pair.block }
            }
        }
        var items: [Document.Item] = []
        for (i, b) in bs.enumerated() {
            items.append(Document.Item(id: i, block: b))
        }
        return Document(items: items)
    }

    static let cellLock = NSLock()
    nonisolated(unsafe) private static var cells: [String: Document] = [:]
    private static let cellLimit = 4096

    static func parseCell(_ cell: String) -> Document {
        cellLock.lock()
        let hit = cells[cell]
        cellLock.unlock()
        let result: Document
        if let hit {
            result = hit
        } else {
            result = parse(cell)
            cellLock.lock()
            if cells.count >= cellLimit { cells.removeAll() }
            cells[cell] = result
            cellLock.unlock()
        }
        return result
    }

    // The shared block engine. Reads `currentRefs` for reference links.
    static func blocks(_ lines: [String]) -> [Block] {
        var result: [Block] = []
        if nesting >= maxNesting {
            let raw = lines.joined(separator: "\n").trimmedOuter()
            if !raw.isEmpty { result = [.paragraph(AttributedString(raw))] }
        } else {
            result = Markdown.$nesting.withValue(nesting + 1) {
                blockSpans(lines).map { pair in pair.block }
            }
        }
        return result
    }

    enum OpenBlock {
        case code(language: String?, fence: String, indent: Int,
                  body: [String])
        case table(headers: [String], alignments: [Alignment],
                   rows: [[String]])
        case list(items: [ListItem], tight: Bool)
    }

    struct Grown {
        let block: Block
        let open: OpenBlock?
        let cut: Int
    }

    // Same engine, recording the START line of each block.
    static func blockSpans(_ lines: [String])
        -> [(start: Int, block: Block)] {
        openSpans(lines, resuming: nil).spans
    }

    struct Rewrite {
        let line: Int
        let by: Int
        let text: String
    }

    static func openSpans(_ source: [String], resuming carried: OpenBlock?)
        -> (spans: [(start: Int, block: Block)], open: OpenBlock?,
            cut: Int, rewrites: [Rewrite]) {
        var lines = source
        var rewrites: [Rewrite] = []
        func kept(tailOf start: Int, at line: Int) {
            if line < lines.count, lines[line] != source[line] {
                rewrites.append(Rewrite(line: line, by: start,
                                        text: lines[line]))
            }
        }
        var out: [(start: Int, block: Block)] = []
        var i = 0
        var grown: Grown? = nil
        var grownAt = -1
        func grow(_ g: Grown, at start: Int) {
            out.append((start, g.block))
            grown = g
            grownAt = out.count - 1
        }
        if let carried { grow(resume(carried, lines, &i), at: 0) }
        while i < lines.count {
            let start = i
            let line = lines[i]
            if isIndentedCode(line) {
                out.append((start, consumeIndentedCode(lines, &i)))
            } else if isFence(line) {
                grow(consumeFenced(lines, &i), at: start)
            } else if isMathFence(line) {
                for math in consumeMath(&lines, &i) {
                    out.append((start, math))
                }
                kept(tailOf: start, at: i)
            } else if isHeading(line) {
                out.append((start, consumeHeading(lines, &i)))
            } else if isHR(line) {
                let lastRule: Bool
                if let lb = out.last?.block, case .rule = lb {
                    lastRule = true
                } else {
                    lastRule = false
                }
                if !lastRule { out.append((start, .rule)) }
                i += 1
            } else if isTableStart(lines, i) {
                grow(consumeTable(lines, &i), at: start)
            } else if isQuoteStart(line) {
                out.append((start, consumeQuote(lines, &i)))
            } else if isListStart(line) {
                grow(consumeList(lines, &i), at: start)
            } else if line.trimmedOuter().isEmpty {
                i += 1
            } else if isCommentStart(line) {
                skipComment(&lines, &i)
                kept(tailOf: start, at: i)
            } else if let img = imageBlock(line) {
                out.append((start, img))
                i += 1
            } else {
                out.append((start, consumeParagraph(lines, &i)))
            }
        }
        let settled = grownAt == out.count - 1 ? grown : nil
        return (out, settled?.open, settled?.cut ?? 0, rewrites)
    }

    static func resume(_ open: OpenBlock, _ lines: [String],
                       _ i: inout Int) -> Grown {
        switch open {
            case .code(let language, let fence, let indent, let body):
                return growFenced(language: language, fence: fence,
                                  indent: indent, body: body, lines, &i)
            case .table(let headers, let alignments, let rows):
                return growTable(headers: headers, alignments: alignments,
                                 rows: rows, lines, &i)
            case .list(let items, let tight):
                return growList(items: items, tight: tight, lines, &i)
        }
    }

    static func stripLinkDefinitions(_ raw: [String])
        -> (lines: [String], refs: [String: URL]) {
        var refs: [String: URL] = [:]
        var out: [String] = []
        var fence: String? = nil
        for line in raw {
            if let open = fence {
                if closesFence(line, open) { fence = nil }
                out.append(line)
            } else if let run = openingFence(line) {
                fence = run
                out.append(line)
            } else if let parsed = parseLinkDefinition(line) {
                refs[parsed.label] = parsed.url
            } else {
                out.append(line)
            }
        }
        return (out, refs)
    }

    static func parseLinkDefinition(_ line: String)
        -> (label: String, url: URL)? {
        var result: (String, URL)? = nil
        let t = line.trimmedLeading()
        if t.hasPrefix("["), leadingSpaces(line) < 4,
           let close = t.dropFirst().firstIndex(of: "]") {
            let rest = t.dropFirst()
            let label = String(rest[..<close]).trimmedOuter()
            let after = rest[rest.index(after: close)...]
            if !label.isEmpty, !label.hasPrefix("^"), after.hasPrefix(":") {
                var rhs = String(after.dropFirst()).trimmedOuter()
                var title = ""
                if let space = rhs.firstIndex(of: " ") {
                    title = String(rhs[space...]).trimmedOuter()
                    rhs = String(rhs[..<space])
                }
                if rhs.hasPrefix("<"), rhs.hasSuffix(">") {
                    rhs = String(rhs.dropFirst().dropLast())
                }
                if !rhs.isEmpty, isLinkTitle(title),
                   let url = URL(string: rhs) {
                    result = (refKey(label), url)
                }
            }
        }
        return result
    }

    static func isLinkTitle(_ s: String) -> Bool {
        var result = s.isEmpty
        if let open = s.first, let close = s.last, s.count >= 2 {
            result = (open == "\"" && close == "\"") ||
                     (open == "'" && close == "'") ||
                     (open == "(" && close == ")")
        }
        return result
    }

    static func refKey(_ label: String) -> String {
        label.lowercased().split(whereSeparator: { c in
            c == " " || c == "\t" || c == "\n"
        }).joined(separator: " ")
    }

    static let lineBreak = "\u{2028}"

    static func inline(_ raw: String) -> AttributedString {
        var stitched = ""
        var maths: [AttributedString] = []
        for segment in codeSpanSegments(raw) {
            if segment.code {
                stitched += segment.text
            } else {
                let literal = escapedAngles(segment.text)
                let withRefs = substituteRefs(htmlInline(literal))
                let pieces = mathEnabled ? mathsOutsideLinks(withRefs)
                                         : [.text(withRefs)]
                for piece in pieces {
                    switch piece {
                        case .text(let s): stitched += s
                        case .math(let s, let display):
                            stitched += mathMark + String(maths.count)
                                + mathMark
                            maths.append(TeX.render(s, display: display))
                    }
                }
            }
        }
        var out = parseInlineMarkdown(normalizeBreaks(stitched))
        for (i, math) in maths.enumerated() {
            let token = mathMark + String(i) + mathMark
            if let r = out.range(of: token) {
                var m = math
                let picked = out[r].inlinePresentationIntent ?? []
                let own = m.inlinePresentationIntent ?? []
                m.inlinePresentationIntent = picked.union(own)
                out.replaceSubrange(r, with: m)
            }
        }
        let angled = raw.unicodeScalars.contains { s in
            s == "<" || s == "&" || s == angle
        }
        if angled {
            applyTag(&out, "u") { sub in sub.underlineStyle = .single }
            applyTag(&out, "sup") { sub in sub[ScriptAttribute.self] = 1 }
            applyTag(&out, "sub") { sub in sub[ScriptAttribute.self] = -1 }
            applyTag(&out, "small") { sub in
                sub[SmallAttribute.self] = true
            }
            restoreAngles(&out)
        }
        return out
    }

    private static let angle = Unicode.Scalar(0x10FFFD) ?? "<"

    private static func escapedAngles(_ s: String) -> String {
        var result = s
        if s.contains("<") || s.contains("&") {
            let mark = String(angle)
            result = s.replacingOccurrences(
                    of: #"(?<!\\)((?:\\\\)*)\\<"#, with: "$1" + mark,
                    options: .regularExpression)
                .replacingOccurrences(of: "&lt;", with: mark,
                                      options: .caseInsensitive)
        }
        return result
    }

    private static func restoreAngles(_ a: inout AttributedString) {
        let mark = String(angle)
        if a.characters.contains(Character(angle)) {
            var rebuilt = AttributedString()
            for run in a.runs {
                let text = String(a[run.range].characters)
                    .replacingOccurrences(of: mark, with: "<")
                rebuilt.append(AttributedString(text,
                                                attributes: run.attributes))
            }
            a = rebuilt
        }
    }

    private static let linkTarget = try? NSRegularExpression(
        pattern: #"\]\([^)\s]*"#)

    private static func mathsOutsideLinks(_ s: String) -> [TeX.Segment] {
        var out: [TeX.Segment] = []
        let ns = s as NSString
        var last = 0
        let full = NSRange(location: 0, length: ns.length)
        let targets = s.contains("](")
            ? linkTarget?.matches(in: s, range: full) ?? [] : []
        for m in targets {
            out += TeX.split(ns.substring(with: NSRange(
                location: last, length: m.range.location - last)))
            out.append(.text(ns.substring(with: m.range)))
            last = NSMaxRange(m.range)
        }
        out += TeX.split(ns.substring(from: last))
        return out
    }

    private static let mathMark = "\u{F8FF}"

    static func codeSpanSegments(_ line: String)
        -> [(text: String, code: Bool)] {
        var out: [(text: String, code: Bool)] = []
        let chars = Array(line)
        var text = ""
        var i = 0
        while i < chars.count {
            if chars[i] == "`" {
                var n = 0
                while i + n < chars.count, chars[i + n] == "`" { n += 1 }
                if let close = closingRun(chars, from: i + n, length: n) {
                    if !text.isEmpty { out.append((text, false)) }
                    text = ""
                    out.append((String(chars[i..<(close + n)]), true))
                    i = close + n
                } else {
                    text += String(chars[i..<(i + n)])
                    i += n
                }
            } else {
                text.append(chars[i])
                i += 1
            }
        }
        if !text.isEmpty { out.append((text, false)) }
        return out
    }

    private static func closingRun(_ chars: [Character], from start: Int,
                                   length: Int) -> Int? {
        var result: Int? = nil
        var i = start
        while i < chars.count, result == nil {
            if chars[i] == "`" {
                var n = 0
                while i + n < chars.count, chars[i + n] == "`" { n += 1 }
                if n == length { result = i }
                i += n
            } else {
                i += 1
            }
        }
        return result
    }

    static func isCommentStart(_ line: String) -> Bool {
        line.trimmedLeading().hasPrefix("<!--")
    }

    static func skipComment(_ lines: inout [String], _ i: inout Int) {
        var closed = false
        while i < lines.count, !closed {
            if let end = lines[i].range(of: "-->") {
                var from = end.upperBound
                var chaining = true
                while chaining {
                    let rest = lines[i][from...].drop { c in
                        c == " " || c == "\t"
                    }
                    if rest.hasPrefix("<!--"),
                       let next = rest.range(of: "-->") {
                        from = next.upperBound
                    } else {
                        chaining = false
                    }
                }
                let tail = String(lines[i][from...]).trimmedLeading()
                closed = true
                if tail.trimmedOuter().isEmpty {
                    i += 1
                } else {
                    lines[i] = tail
                }
            } else {
                i += 1
            }
        }
    }

    static func htmlLine(_ line: String) -> String {
        var out = ""
        for segment in codeSpanSegments(line) {
            out += segment.code ? segment.text : htmlInline(segment.text)
        }
        return out
    }

    private struct TagRule {
        let re: NSRegularExpression?
        let template: String
        let closers: [String]
    }

    private static func tagRule(_ pattern: String, _ template: String,
                                closers: [String] = []) -> TagRule {
        TagRule(re: try? NSRegularExpression(pattern: pattern,
                                             options: .caseInsensitive),
                template: template, closers: closers)
    }

    private static let imgRule =
        tagRule(#"<img\b[^>]*?\bsrc\s*=\s*"([^"]*)"[^>]*>"#, "![]($1)")

    private static let tagRules: [TagRule] = [
        tagRule(#"<!--(?:(?!<!--).)*?-->"#, "", closers: ["-->"]),
        tagRule(#"<br\s*/?>"#, lineBreak),
        imgRule,
        tagRule(#"<img\b[^>]*?\bsrc\s*=\s*'([^']*)'[^>]*>"#, "![]($1)"),
        tagRule(#"<a\b[^>]*?\bhref\s*=\s*"([^"]*)"[^>]*>"# +
                #"((?:(?!<a\b).)*?)</a>"#,
                "[$2]($1)", closers: ["</a>"]),
        tagRule(#"<a\b[^>]*?\bhref\s*=\s*'([^']*)'[^>]*>"# +
                #"((?:(?!<a\b).)*?)</a>"#,
                "[$2]($1)", closers: ["</a>"]),
        tagRule(#"<(b|strong)>((?:(?!<\1>).)*?)</\1>"#, "**$2**",
                closers: ["</b>", "</strong>"]),
        tagRule(#"<(i|em)>((?:(?!<\1>).)*?)</\1>"#, "*$2*",
                closers: ["</i>", "</em>"]),
        tagRule(#"<(s|del|strike)>((?:(?!<\1>).)*?)</\1>"#, "~~$2~~",
                closers: ["</s>", "</del>", "</strike>"]),
        tagRule(#"<(code|kbd)>((?:(?!<\1>).)*?)</\1>"#, "`$2`",
                closers: ["</code>", "</kbd>"]),
    ]

    private static let imgAltRE = try? NSRegularExpression(
        pattern: #"<img\b[^>]*?\balt\s*=\s*"([^"]*)""#,
        options: .caseInsensitive)

    private static let imgSizeRE = try? NSRegularExpression(
        pattern: #"\b(width|height)\s*=\s*"?(\d+)"?"#,
        options: .caseInsensitive)

    static func htmlInline(_ text: String) -> String {
        var result = text
        if text.contains("<") {
            result = imagesWithAttributes(result)
            for rule in tagRules {
                let lower = rule.closers.isEmpty ? "" : result.lowercased()
                let closed = rule.closers.isEmpty ||
                             rule.closers.contains { c in lower.contains(c) }
                if closed, let re = rule.re {
                    let ns = result as NSString
                    result = re.stringByReplacingMatches(
                        in: result,
                        range: NSRange(location: 0, length: ns.length),
                        withTemplate: rule.template)
                }
            }
        }
        return result
    }

    private static func imagesWithAttributes(_ text: String) -> String {
        var result = text
        if let altRE = imgAltRE, let sizeRE = imgSizeRE {
            let ns = text as NSString
            let full = NSRange(location: 0, length: ns.length)
            let tags = imgRule.re?.matches(in: text, range: full) ?? []
            let mutable = NSMutableString(string: text)
            for m in tags.reversed() {
                let tag = ns.substring(with: m.range)
                let src = ns.substring(with: m.range(at: 1))
                let tagNS = tag as NSString
                let tagRange = NSRange(location: 0, length: tagNS.length)
                var alt = ""
                if let a = altRE.firstMatch(in: tag, range: tagRange) {
                    alt = tagNS.substring(with: a.range(at: 1))
                }
                var dims: [String] = []
                for d in sizeRE.matches(in: tag, range: tagRange) {
                    let key = tagNS.substring(with: d.range(at: 1))
                    let value = tagNS.substring(with: d.range(at: 2))
                    dims.append(key.lowercased() + "=" + value)
                }
                let suffix = dims.isEmpty
                    ? "" : "{" + dims.joined(separator: " ") + "}"
                mutable.replaceCharacters(
                    in: m.range, with: "![\(alt)](\(src))" + suffix)
            }
            result = mutable as String
        }
        return result
    }

    static func parseInlineMarkdown(_ s: String) -> AttributedString {
        let opts = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        var result = AttributedString(s)
        if let parsed = try? AttributedString(markdown: s, options: opts) {
            result = parsed
        }
        return result
    }

    static func normalizeBreaks(_ s: String) -> String {
        let lines = s
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var out: [String] = []
        for (idx, line) in lines.enumerated() {
            let last = idx == lines.count - 1
            let slashes = line.reversed().prefix { ch in ch == "\\" }.count
            let spaced = line.hasSuffix("  ")
            let hardBreak = (spaced || slashes % 2 == 1) && !last
            let trimmed = hardBreak ? String(line.dropLast(spaced ? 2 : 1))
                                    : line
            if hardBreak {
                out.append(trimmed + lineBreak)
            } else if last {
                var tail = trimmed
                while tail.hasSuffix(" ") { tail.removeLast() }
                out.append(tail)
            } else {
                out.append(trimmed + " ")
            }
        }
        return out.joined().replacingOccurrences(of: lineBreak + " ",
                                                 with: lineBreak)
    }

    static func applyTag(_ a: inout AttributedString, _ tag: String,
                         style: (inout AttributedSubstring) -> Void) {
        var pairing = true
        while pairing {
            pairing = tagPass(&a, tag, style: style)
        }
    }

    private static func tagPass(_ a: inout AttributedString, _ tag: String,
                                style: (inout AttributedSubstring) -> Void)
        -> Bool {
        let open = "<\(tag)>"
        let close = "</\(tag)>"
        var out = AttributedString()
        var from = a.startIndex
        var paired = false
        var closerAhead = true
        while let o = a[from...].range(of: open, options: .caseInsensitive) {
            let intent = a.runs[o.lowerBound].inlinePresentationIntent
            let c = closerAhead && intent?.contains(.code) != true
                ? a[o.upperBound...].range(of: close,
                                           options: .caseInsensitive)
                : nil
            if intent?.contains(.code) == true {
                out.append(a[from..<o.upperBound])
                from = o.upperBound
            } else if let c {
                out.append(a[from..<o.lowerBound])
                var sub = a[o.upperBound..<c.lowerBound]
                style(&sub)
                out.append(sub)
                from = c.upperBound
                paired = true
            } else {
                out.append(a[from..<o.lowerBound])
                from = o.upperBound
                closerAhead = false
            }
        }
        out.append(a[from...])
        a = out
        return paired
    }

    private static let fullRefRE = try? NSRegularExpression(
        pattern: "(!?)\\[([^\\]\\n]+)\\]\\[([^\\]\\n]*)\\]")

    private static let shortRefRE = try? NSRegularExpression(
        pattern: "(!?)\\[([^\\]\\n]+)\\](?![\\[\\(:])")

    static func substituteRefs(_ s: String) -> String {
        let refs = Markdown.currentRefs
        var result = s
        if !refs.isEmpty {
            result = applyRefPattern(result, fullRefRE,
                                     hasLabelGroup: true, refs: refs)
            result = applyRefPattern(result, shortRefRE,
                                     hasLabelGroup: false, refs: refs)
        }
        return result
    }

    static func applyRefPattern(_ s: String, _ re: NSRegularExpression?,
                                hasLabelGroup: Bool,
                                refs: [String: URL]) -> String {
        var result = s
        if let re {
            let ns = s as NSString
            let matches = re.matches(
                in: s, range: NSRange(location: 0, length: ns.length))
            if !matches.isEmpty {
                let mutable = NSMutableString(string: s)
                for m in matches.reversed() {
                    let bang = ns.substring(with: m.range(at: 1))
                    let text = ns.substring(with: m.range(at: 2))
                    var labelSrc = text
                    if hasLabelGroup, m.numberOfRanges > 3,
                       m.range(at: 3).location != NSNotFound {
                        let g3 = ns.substring(with: m.range(at: 3))
                        if !g3.isEmpty { labelSrc = g3 }
                    }
                    if let url = refs[refKey(labelSrc)] {
                        let rep = "\(bang)[\(text)](\(url.absoluteString))"
                        mutable.replaceCharacters(in: m.range, with: rep)
                    }
                }
                result = mutable as String
            }
        }
        return result
    }

    static func isHeading(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        let n = t.prefix { c in c == "#" }.count
        if n >= 1 && n <= 6 {
            let rest = t.dropFirst(n)
            result = rest.hasPrefix(" ") || rest.hasPrefix("\t") ||
                     rest.isEmpty
        }
        return result
    }

    static func consumeHeading(_ lines: [String], _ i: inout Int) -> Block {
        let t = lines[i].trimmedOuter()
        let n = t.prefix { c in c == "#" }.count
        var body = String(t.dropFirst(n)).trimmedOuter()
        let closing = body.reversed().prefix { c in c == "#" }.count
        if closing == body.count {
            body = ""
        } else if closing > 0,
                  body.dropLast(closing).last?.isWhitespace == true {
            body = String(body.dropLast(closing)).trimmedOuter()
        }
        i += 1
        return .heading(level: n, text: inline(body))
    }

    static func isHR(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        if t.count >= 3, let c = t.first, c == "-" || c == "*" || c == "_" {
            result = t.allSatisfy { ch in ch == c || ch == " " || ch == "\t" }
        }
        return result
    }

    static func isFence(_ s: String) -> Bool {
        fenceRun(s) != nil
    }

    static func fenceRun(_ s: String) -> String? {
        var result: String? = nil
        let t = s.trimmedLeading()
        if let c = t.first, c == "`" || c == "~" {
            let run = t.prefix { ch in ch == c }
            if run.count >= 3 { result = String(run) }
        }
        return result
    }

    static func openingFence(_ s: String) -> String? {
        leadingSpaces(s) < 4 ? fenceRun(s) : nil
    }

    static func closesFence(_ s: String, _ fence: String) -> Bool {
        var result = false
        if let run = fenceRun(s), let mark = fence.first,
           run.first == mark, run.count >= fence.count {
            result = s.trimmedOuter().allSatisfy { ch in ch == mark }
        }
        return result
    }

    static func consumeFenced(_ lines: [String], _ i: inout Int) -> Grown {
        let raw = lines[i]
        let t = raw.trimmedLeading()
        let fence = fenceRun(raw) ?? "```"
        let lang = String(t.dropFirst(fence.count)).trimmedOuter()
        i += 1
        return growFenced(language: lang.isEmpty ? nil : lang,
                          fence: fence,
                          indent: raw.count - t.count, body: [], lines, &i)
    }

    static func growFenced(language: String?, fence: String, indent: Int,
                           body: [String], _ lines: [String],
                           _ i: inout Int) -> Grown {
        let pad = String(repeating: " ", count: indent)
        var body = body
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            if closesFence(line, fence) {
                done = true
            } else if indent > 0, line.hasPrefix(pad) {
                body.append(String(line.dropFirst(indent)))
            } else {
                body.append(line)
            }
            i += 1
        }
        let open: OpenBlock? = done ? nil
            : .code(language: language, fence: fence, indent: indent,
                    body: body)
        return Grown(block: .code(language: language,
                                  text: body.joined(separator: "\n")),
                     open: open, cut: i)
    }

    // Only a line that OPENS with $$ starts a display.
    static func isMathFence(_ s: String) -> Bool {
        mathEnabled && s.trimmedOuter().hasPrefix("$$")
    }

    // Accepts both spellings authors use: the whole thing on one line, and an
    // opening $$ with the formula on the lines below.
    private static func consumeMath(_ lines: inout [String],
                                    _ i: inout Int) -> [Block] {
        let line = lines[i].trimmedOuter()
        var blocks: [Block] = []
        var from = line.startIndex
        var chaining = true
        while chaining {
            let rest = line[from...].drop { c in c == " " || c == "\t" }
            let inner = rest.dropFirst(2)
            if rest.hasPrefix("$$"), let end = inner.range(of: "$$") {
                blocks.append(.math(String(inner[..<end.lowerBound])
                    .trimmedOuter()))
                from = end.upperBound
            } else {
                chaining = false
            }
        }
        var result = blocks
        if blocks.isEmpty {
            result = [consumeOpenMath(&lines, &i)]
        } else {
            let tail = String(line[from...]).trimmedOuter()
            if tail.allSatisfy({ ch in ch == "$" }) {
                i += 1
            } else {
                lines[i] = tail
            }
        }
        return result
    }

    private static func consumeOpenMath(_ lines: inout [String],
                                        _ i: inout Int) -> Block {
        var body: [String] = []
        var line = String(lines[i].trimmedOuter().dropFirst(2))
        var closed = false
        var reading = true
        while reading {
            if let end = line.range(of: "$$") {
                let head = String(line[..<end.lowerBound])
                let tail = String(line[end.upperBound...]).trimmedLeading()
                if !head.trimmedOuter().isEmpty { body.append(head) }
                closed = true
                if tail.trimmedOuter().allSatisfy({ ch in ch == "$" }) {
                    i += 1
                } else {
                    lines[i] = tail
                }
            } else {
                if !line.trimmedOuter().isEmpty || !body.isEmpty {
                    body.append(line)
                }
                i += 1
            }
            reading = !closed && i < lines.count
            if reading { line = lines[i] }
        }
        return .math(body.joined(separator: "\n").trimmedOuter())
    }

    static func isIndentedCode(_ s: String) -> Bool {
        var result = false
        if !s.trimmedOuter().isEmpty {
            result = s.hasPrefix("    ") || s.hasPrefix("\t")
        }
        return result
    }

    static func consumeIndentedCode(_ lines: [String],
                                    _ i: inout Int) -> Block {
        var body: [String] = []
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            if line.trimmedOuter().isEmpty {
                body.append("")
                i += 1
            } else if line.hasPrefix("    ") {
                body.append(String(line.dropFirst(4)))
                i += 1
            } else if line.hasPrefix("\t") {
                body.append(String(line.dropFirst(1)))
                i += 1
            } else {
                done = true
            }
        }
        while let last = body.last, last.isEmpty { body.removeLast() }
        return .code(language: nil, text: body.joined(separator: "\n"))
    }

    static func isQuoteStart(_ s: String) -> Bool {
        leadingSpaces(s) <= 3 && s.trimmedLeading().hasPrefix(">")
    }

    static func consumeQuote(_ lines: [String], _ i: inout Int) -> Block {
        var inner: [String] = []
        var collecting = true
        while i < lines.count, collecting {
            let line = lines[i]
            if isQuoteStart(line) {
                var t = line.trimmedLeading()
                t = String(t.dropFirst())
                if t.hasPrefix(" ") || t.hasPrefix("\t") {
                    t = String(t.dropFirst())
                }
                inner.append(t)
                i += 1
            } else if !line.trimmedOuter().isEmpty,
                      isLazyContinuation(line) {
                inner.append(line.trimmedLeading())
                i += 1
            } else {
                collecting = false
            }
        }
        return .quote(blocks(inner))
    }

    static func isListStart(_ s: String) -> Bool { listMarker(s) != nil }

    static func listInterrupts(_ s: String) -> Bool {
        var result = false
        if let m = listMarker(s), !m.rest.trimmedOuter().isEmpty {
            result = m.label == "\u{2022}" || m.label == "1."
        }
        return result
    }

    static func listMarker(_ line: String)
        -> (label: String, sig: Character, offset: Int, rest: String)? {
        var result: (String, Character, Int, String)? = nil
        let leading = line.prefix { c in c == " " }.count
        if leading <= 3 {
            let afterIndent = line.dropFirst(leading)
            if let first = afterIndent.first,
               first == "-" || first == "*" || first == "+" {
                result = afterMarker(
                    afterIndent.dropFirst(), leading: leading,
                    markerWidth: 1, label: "\u{2022}", sig: first)
            } else {
                let digits = afterIndent.prefix { c in c.isNumber }
                let afterDigits = afterIndent.dropFirst(digits.count)
                if !digits.isEmpty, digits.count <= 9,
                   let delim = afterDigits.first,
                   delim == "." || delim == ")" {
                    result = afterMarker(
                        afterDigits.dropFirst(), leading: leading,
                        markerWidth: digits.count + 1,
                        label: String(digits) + ".", sig: delim)
                }
            }
        }
        return result
    }

    static func afterMarker(_ tail: Substring, leading: Int,
                            markerWidth: Int, label: String,
                            sig: Character)
        -> (label: String, sig: Character, offset: Int, rest: String)? {
        var result: (String, Character, Int, String)? = nil
        let spaces = tail.prefix { c in c == " " }.count
        let blankRest = tail.allSatisfy { c in c == " " || c == "\t" }
        let column = leading + markerWidth
        if blankRest {
            result = (label, sig, column + 1, "")
        } else if tail.hasPrefix("\t") {
            result = (label, sig, column + 4 - column % 4,
                      String(tail.dropFirst()))
        } else if spaces >= 1 {
            let n = spaces >= 5 ? 1 : spaces
            result = (label, sig, column + n, String(tail.dropFirst(n)))
        }
        return result
    }

    static func consumeList(_ lines: [String], _ i: inout Int) -> Grown {
        growList(items: [], tight: true, lines, &i)
    }

    static func growList(items: [ListItem], tight: Bool, _ lines: [String],
                         _ i: inout Int) -> Grown {
        var items = items
        var tight = tight
        var sig: Character? = nil
        var done = false
        var lastStart = i
        while i < lines.count, !done {
            if let m = listMarker(lines[i]), sig == nil || m.sig == sig {
                sig = m.sig
                lastStart = i
                var body: [String] = []
                let (checked, rest) = stripTaskMarker(m.rest)
                body.append(rest)
                i += 1
                if collectItemBody(lines, &i, m.offset, &body) {
                    tight = false
                }
                items.append(ListItem(marker: m.label, checked: checked,
                                      blocks: blocks(body)))
                let gap = interItemGap(lines, &i, sig: m.sig)
                if gap.loose { tight = false }
                if gap.ended { done = true }
            } else {
                done = true
            }
        }
        let open: OpenBlock? = items.count >= 2
            ? .list(items: Array(items.dropLast()), tight: tight) : nil
        return Grown(block: .list(items: items, tight: tight), open: open,
                     cut: lastStart)
    }

    static func stripTaskMarker(_ s: String)
        -> (checked: Bool?, rest: String) {
        var result: (Bool?, String) = (nil, s)
        let boxes: [(String, Bool)] = [("[ ]", false), ("[x]", true),
                                       ("[X]", true)]
        for (box, checked) in boxes where result.0 == nil {
            if s == box {
                result = (checked, "")
            } else if s.hasPrefix(box + " ") || s.hasPrefix(box + "\t") {
                result = (checked, String(s.dropFirst(box.count + 1)))
            }
        }
        return result
    }

    static func collectItemBody(_ lines: [String], _ i: inout Int,
                                _ offset: Int,
                                _ body: inout [String]) -> Bool {
        var loose = false
        var lastWasBlank = false
        var collecting = true
        while i < lines.count, collecting {
            let line = lines[i]
            if line.trimmedOuter().isEmpty {
                var j = i
                while j < lines.count, lines[j].trimmedOuter().isEmpty {
                    j += 1
                }
                if j < lines.count, leadingSpaces(lines[j]) >= offset {
                    var k = i
                    while k < j { body.append(""); k += 1 }
                    i = j
                    loose = true
                    lastWasBlank = true
                } else {
                    collecting = false
                }
            } else if leadingSpaces(line) >= offset {
                body.append(dropIndent(line, offset))
                i += 1
                lastWasBlank = false
            } else if !lastWasBlank, isLazyContinuation(line) {
                body.append(line.trimmedLeading())
                i += 1
                lastWasBlank = false
            } else {
                collecting = false
            }
        }
        return loose
    }

    static func interItemGap(_ lines: [String], _ i: inout Int,
                             sig: Character) -> (loose: Bool, ended: Bool) {
        var result: (loose: Bool, ended: Bool) = (false, false)
        let before = i
        while i < lines.count, lines[i].trimmedOuter().isEmpty { i += 1 }
        if i > before {
            if i < lines.count, let n = listMarker(lines[i]), n.sig == sig {
                result = (true, false)
            } else {
                i = before
                result = (false, true)
            }
        }
        return result
    }

    static func isLazyContinuation(_ line: String) -> Bool {
        !(isHeading(line) || isHR(line) || isFence(line) ||
          isMathFence(line) || isQuoteStart(line) || isListStart(line) ||
          setextLevel(line) > 0)
    }

    static func leadingSpaces(_ s: String) -> Int {
        var n = 0
        var done = false
        for c in s {
            if !done {
                if c == " " { n += 1 }
                else if c == "\t" { n += 4 - (n % 4) }
                else { done = true }
            }
        }
        return n
    }

    static func dropIndent(_ s: String, _ n: Int) -> String {
        var dropped = 0
        var idx = s.startIndex
        var done = false
        while idx < s.endIndex, !done {
            let c = s[idx]
            if c == " ", dropped < n {
                dropped += 1
                idx = s.index(after: idx)
            } else if c == "\t", dropped < n {
                dropped += 4 - (dropped % 4)
                idx = s.index(after: idx)
            } else {
                done = true
            }
        }
        return String(s[idx...])
    }

    static func isTableRow(_ s: String) -> Bool {
        let t = s.trimmedOuter()
        return t.contains("|") && !t.isEmpty
    }

    // A well-formed delimiter cell: dashes, optionally colon-anchored.
    static func isAlignmentCell(_ cell: String) -> Bool {
        let t = cell.trimmingCharacters(in: .whitespaces)
        return t.contains("-") && t.allSatisfy { ch in "-: ".contains(ch) }
    }

    // A cell that is malformed but still clearly punctuation rather than
    // content.
    static func isJunkCell(_ cell: String) -> Bool {
        cell.allSatisfy { ch in !ch.isLetter && !ch.isNumber }
    }

    // The delimiter row, read tolerantly: one good cell and no cell carrying
    // content is enough, or one stray character costs the whole table.
    static func isTableSeparator(_ s: String) -> Bool {
        var result = false
        let t = s.trimmedOuter()
        if t.contains("|"), t.contains("-") {
            let cells = parseRow(t)
            var good = 0
            var usable = true
            for cell in cells {
                if isAlignmentCell(cell) {
                    good += 1
                } else if !isJunkCell(cell) {
                    usable = false
                }
            }
            result = usable && good > 0
        }
        return result
    }

    static func isTableStart(_ lines: [String], _ i: Int) -> Bool {
        var result = false
        if i + 1 < lines.count {
            result = isTableRow(lines[i]) && isTableSeparator(lines[i + 1])
        }
        return result
    }

    static func consumeTable(_ lines: [String], _ i: inout Int) -> Grown {
        var headers: [String] = []
        var alignments: [Alignment] = []
        if i < lines.count, isTableRow(lines[i]) {
            headers = parseRow(lines[i])
            i += 1
        }
        if i < lines.count, isTableSeparator(lines[i]) {
            alignments = parseAlignments(lines[i])
            i += 1
        }
        return growTable(headers: headers, alignments: alignments, rows: [],
                         lines, &i)
    }

    static func growTable(headers: [String], alignments: [Alignment],
                          rows: [[String]], _ lines: [String],
                          _ i: inout Int) -> Grown {
        var rows = rows
        while i < lines.count, isTableRow(lines[i]) {
            rows.append(parseRow(lines[i]))
            i += 1
        }
        return Grown(block: .table(headers: headers.map { c in
                                       cellWithRefs(c)
                                   },
                                   rows: rows.map { r in
                                       r.map { c in cellWithRefs(c) }
                                   },
                                   alignments: alignments),
                     open: .table(headers: headers, alignments: alignments,
                                  rows: rows),
                     cut: i)
    }

    static func cellWithRefs(_ cell: String) -> String {
        var out = ""
        if Markdown.currentRefs.isEmpty || !cell.contains("[") {
            out = cell
        } else {
            for segment in codeSpanSegments(cell) {
                out += segment.code ? segment.text
                                    : substituteRefs(segment.text)
            }
        }
        return out
    }

    static func parseRow(_ s: String) -> [String] {
        let t = s.trimmedOuter()
        var cells: [String] = []
        var cell = ""
        var escaping = false
        for ch in t {
            if escaping {
                if ch != "|" { cell.append("\\") }
                cell.append(ch)
                escaping = false
            } else if ch == "\\" {
                escaping = true
            } else if ch == "|" {
                cells.append(cell)
                cell = ""
            } else {
                cell.append(ch)
            }
        }
        if escaping { cell.append("\\") }
        cells.append(cell)
        if t.hasPrefix("|"), !cells.isEmpty { cells.removeFirst() }
        if t.hasSuffix("|"), !t.hasSuffix("\\|"), !cells.isEmpty {
            cells.removeLast()
        }
        return cells.map { p in p.trimmingCharacters(in: .whitespaces) }
    }

    static func parseAlignments(_ s: String) -> [Alignment] {
        parseRow(s).map { cell in
            let t = cell.trimmingCharacters(in: .whitespaces)
            let left = t.hasPrefix(":")
            let right = t.hasSuffix(":")
            let a: Alignment
            if left && right { a = .center }
            else if right { a = .right }
            else if left { a = .left }
            else { a = .none }
            return a
        }
    }

    static let imagePattern =
        #"^!\[([^\]]*)\]\(([^\s\)]+)(?:\s+"[^"]*")?\)"#
        + #"\s*(?:\{([^}]*)\})?\s*$"#

    static let imageLineRegex: NSRegularExpression? =
        try? NSRegularExpression(pattern: imagePattern)

    static func imageBlock(_ line: String) -> Block? {
        var result: Block? = nil
        if let re = imageLineRegex {
            let trimmed = htmlLine(line).trimmedOuter()
            let ns = trimmed as NSString
            let range = NSRange(location: 0, length: ns.length)
            if let m = re.firstMatch(in: trimmed, options: [],
                                     range: range) {
                let alt = ns.substring(with: m.range(at: 1))
                let raw = ns.substring(with: m.range(at: 2))
                if let url = URL(string: raw) {
                    var width: CGFloat?
                    var height: CGFloat?
                    if m.numberOfRanges >= 4,
                       m.range(at: 3).location != NSNotFound {
                        let attrs = ns.substring(with: m.range(at: 3))
                        (width, height) = parseDimensions(attrs)
                    }
                    result = .image(alt: alt, url: url,
                                    width: width, height: height)
                }
            }
        }
        return result
    }

    static func parseDimensions(_ attrs: String) -> (CGFloat?, CGFloat?) {
        var width: CGFloat?
        var height: CGFloat?
        let pat = #"(width|height)\s*=\s*(\d+(?:\.\d+)?)(?:px)?"#
        if let re = try? NSRegularExpression(pattern: pat,
                                             options: .caseInsensitive) {
            let ns = attrs as NSString
            let full = NSRange(location: 0, length: ns.length)
            re.enumerateMatches(in: attrs, options: [],
                                range: full) { m, _, _ in
                if let m, m.numberOfRanges == 3 {
                    let key = ns.substring(with: m.range(at: 1)).lowercased()
                    let val = ns.substring(with: m.range(at: 2))
                    if let n = Double(val) {
                        if key == "width" { width = CGFloat(n) }
                        else if key == "height" { height = CGFloat(n) }
                    }
                }
            }
        }
        return (width, height)
    }

    static func consumeParagraph(_ lines: [String],
                                 _ i: inout Int) -> Block {
        var body: [String] = []
        var level = 0
        var done = false
        while i < lines.count, !done {
            let line = lines[i]
            level = body.isEmpty ? 0 : setextLevel(line)
            let blank = line.trimmedOuter().isEmpty
            let other = leadingSpaces(line) < 4 &&
                        (isHeading(line) || isHR(line) || isFence(line) ||
                         isMathFence(line) ||
                         isTableStart(lines, i) || isQuoteStart(line) ||
                         listInterrupts(line) || imageBlock(line) != nil ||
                         isCommentStart(line))
            if level > 0 {
                i += 1
                done = true
            } else if blank || other {
                done = true
            } else {
                body.append(line.trimmedLeading())
                i += 1
            }
        }
        let raw = body.joined(separator: "\n")
        return level > 0
            ? .heading(level: level, text: inline(raw.trimmedOuter()))
            : bareMath(raw) ?? .paragraph(inline(raw))
    }

    static func setextLevel(_ line: String) -> Int {
        var result = 0
        let t = line.trimmedOuter()
        if leadingSpaces(line) < 4, let c = t.first,
           c == "=" || (c == "-" && t.count >= 2),
           t.allSatisfy({ ch in ch == c }) {
            result = c == "=" ? 1 : 2
        }
        return result
    }

    // A converter lifting an equation out of a PDF writes the TeX with nothing
    // around it, and the paragraph then reads as a wall of backslashes.
    private static func bareMath(_ raw: String) -> Block? {
        var result: Block? = nil
        if mathEnabled, opensWithControlWord(raw), !hasProseWord(raw),
           TeX.parses(raw) {
            result = .math(raw)
        }
        return result
    }

    // Parsing cleanly is not enough: neighbouring letters multiply, so a
    // run of three letters outside braces is prose wearing a backslash.
    private static func hasProseWord(_ raw: String) -> Bool {
        var depth = 0
        var run = 0
        var found = false
        var inCommand = false
        for ch in raw {
            let namesCommand = inCommand && ch.isLetter
            if ch == "\\" {
                inCommand = true
                run = 0
            } else if !namesCommand {
                inCommand = false
                if ch == "{" {
                    depth += 1
                    run = 0
                } else if ch == "}" {
                    depth = max(depth - 1, 0)
                    run = 0
                } else if ch.isLetter, ch.isASCII, depth == 0 {
                    run += 1
                    if run >= 3 { found = true }
                } else {
                    run = 0
                }
            }
        }
        return found
    }

    private static func opensWithControlWord(_ raw: String) -> Bool {
        var result = false
        let t = raw.trimmedLeading()
        if t.hasPrefix("\\"), let after = t.dropFirst().first {
            result = after.isLetter
        }
        return result
    }
}

extension String {

    func trimmedOuter() -> String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func trimmedLeading() -> String {
        var i = startIndex
        while i < endIndex, self[i] == " " || self[i] == "\t" {
            i = index(after: i)
        }
        return String(self[i...])
    }
}
