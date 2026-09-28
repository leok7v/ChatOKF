import Foundation

enum Highlight {

    static func attribute(_ code: String, language: String?,
                          baseFont: PlatformFont) -> NSAttributedString {
        let ns = NSMutableAttributedString(string: code)
        let full = NSRange(location: 0, length: (code as NSString).length)
        ns.addAttribute(.font, value: baseFont, range: full)
        ns.addAttribute(.foregroundColor, value: platformDefaultTextColor,
                        range: full)
        if let language {
            let key = data.aliases[language.lowercased()]
                ?? language.lowercased()
            if let spec = data.languages[key] {
                colorize(spec, code: code, full: full, into: ns)
            }
        }
        return ns
    }

    private typealias Span = (NSRange, PlatformColor)

    static let budget: Duration = .milliseconds(250)

    private static func colorize(_ spec: Spec, code: String, full: NSRange,
                                 into ns: NSMutableAttributedString) {
        var mask = [Bool](repeating: false, count: full.length)
        var spans: [Span] = []
        let deadline = ContinuousClock.now + budget
        tokenize([(spec.blockComment, data.comment),
                  (spec.lineComment, data.comment),
                  (spec.attr, data.attr),
                  (spec.string, data.string)],
                 code, full, deadline, &spans, &mask)
        apply(spec.meta, code, full, data.builtin, deadline, &spans, &mask)
        apply(spec.tag, code, full, data.variable, deadline, &spans, &mask)
        apply(spec.type, code, full, data.type, deadline, &spans, &mask)
        apply(spec.builtin, code, full, data.builtin, deadline, &spans,
              &mask)
        apply(spec.number, code, full, data.number, deadline, &spans, &mask)
        applyKeywords(spec.keywords, code, full, data.keyword, deadline,
                      &spans, mask)
        spans.sort { a, b in a.0.location < b.0.location }
        for (range, color) in spans {
            ns.addAttribute(.foregroundColor, value: color, range: range)
        }
    }

    private static func tokenize(
        _ classes: [(NSRegularExpression?, PlatformColor)], _ code: String,
        _ full: NSRange, _ deadline: ContinuousClock.Instant,
        _ spans: inout [Span], _ mask: inout [Bool]) {
        var next = classes.map { entry in
            firstMatch(entry.0, code, from: 0, full, deadline)
        }
        var position = 0
        var pending = true
        while pending {
            for k in next.indices {
                if let r = next[k], r.location < position {
                    next[k] = firstMatch(classes[k].0, code, from: position,
                                         full, deadline)
                }
            }
            let best = next.indices.compactMap { k in
                next[k].map { r in (range: r, rank: k) }
            }.min { a, b in
                a.range.location != b.range.location
                    ? a.range.location < b.range.location
                    : a.rank < b.rank
            }
            if let best, NSMaxRange(best.range) <= mask.count {
                fill(best.range, &mask)
                spans.append((best.range, classes[best.rank].1))
                position = NSMaxRange(best.range)
            } else {
                pending = false
            }
        }
    }

    private static func firstMatch(_ re: NSRegularExpression?,
                                   _ code: String, from start: Int,
                                   _ full: NSRange,
                                   _ deadline: ContinuousClock.Instant)
        -> NSRange? {
        var result: NSRange? = nil
        re?.enumerateMatches(
            in: code, options: [.reportProgress, .withTransparentBounds,
                                .withoutAnchoringBounds],
            range: NSRange(location: start, length: full.length - start)) {
            m, _, stop in
            if let m, m.range.length > 0 {
                result = m.range
                stop.pointee = true
            } else if ContinuousClock.now > deadline {
                stop.pointee = true
            }
        }
        return result
    }

    private static func apply(_ re: NSRegularExpression?, _ code: String,
                              _ full: NSRange, _ color: PlatformColor,
                              _ deadline: ContinuousClock.Instant,
                              _ spans: inout [Span], _ mask: inout [Bool]) {
        if let re {
            re.enumerateMatches(in: code, options: .reportProgress,
                                range: full) { m, _, stop in
                if ContinuousClock.now > deadline { stop.pointee = true }
                if let m, canColor(m.range, mask) {
                    fill(m.range, &mask)
                    spans.append((m.range, color))
                }
            }
        }
    }

    private static func applyKeywords(_ re: NSRegularExpression?,
                                      _ code: String, _ full: NSRange,
                                      _ color: PlatformColor,
                                      _ deadline: ContinuousClock.Instant,
                                      _ spans: inout [Span],
                                      _ mask: [Bool]) {
        if let re {
            re.enumerateMatches(in: code, options: .reportProgress,
                                range: full) { m, _, stop in
                if ContinuousClock.now > deadline { stop.pointee = true }
                if let m, canColor(m.range, mask) {
                    spans.append((m.range, color))
                }
            }
        }
    }

    private static func compiled(_ pattern: String?) -> NSRegularExpression? {
        let opts: NSRegularExpression.Options =
            [.dotMatchesLineSeparators, .anchorsMatchLines]
        return pattern.flatMap { text in
            try? NSRegularExpression(pattern: text, options: opts)
        }
    }

    private static func keywordPattern(_ words: [String])
        -> NSRegularExpression? {
        var out: NSRegularExpression? = nil
        if !words.isEmpty {
            let escaped = words
                .map { w in NSRegularExpression.escapedPattern(for: w) }
                .joined(separator: "|")
            out = try? NSRegularExpression(
                pattern: "(?<![\\w@])(" + escaped + ")(?![\\w])")
        }
        return out
    }

    private static func canColor(_ r: NSRange, _ mask: [Bool]) -> Bool {
        let lo = r.location
        let hi = r.location + r.length
        let inside = lo >= 0 && hi <= mask.count
        let free = inside && !(lo..<hi).contains { i in mask[i] }
        return free
    }

    private static func fill(_ r: NSRange, _ mask: inout [Bool]) {
        var i = r.location
        while i < r.location + r.length { mask[i] = true; i += 1 }
    }

    private struct Spec {
        let keywords: NSRegularExpression?
        let lineComment: NSRegularExpression?
        let blockComment: NSRegularExpression?
        let string: NSRegularExpression?
        let number: NSRegularExpression?
        let tag: NSRegularExpression?
        let attr: NSRegularExpression?
        let meta: NSRegularExpression?
        let type: NSRegularExpression?
        let builtin: NSRegularExpression?
    }

    private struct Loaded {
        let languages: [String: Spec]
        let aliases: [String: String]
        let keyword: PlatformColor
        let string: PlatformColor
        let number: PlatformColor
        let comment: PlatformColor
        let type: PlatformColor
        let builtin: PlatformColor
        let variable: PlatformColor
        let attr: PlatformColor

        static let empty = Loaded(
            languages: [:], aliases: [:],
            keyword: .gray, string: .gray, number: .gray, comment: .gray,
            type: .gray, builtin: .gray, variable: .gray, attr: .gray)
    }

    private static let data: Loaded = load()

    private static func load() -> Loaded {
        var result: Loaded = .empty
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Resources")
        let url = Res.url("highlights", "ini", dev: dir)
        if let url, let src = try? String(contentsOf: url, encoding: .utf8) {
            result = build(from: parseINI(src))
        }
        return result
    }

    private static func parseINI(_ source: String) -> [String: String] {
        var result: [String: String] = [:]
        var pending = ""
        let lines = source.split(separator: "\n",
                                 omittingEmptySubsequences: false)
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasSuffix("\\") {
                pending += line.dropLast()
            } else {
                let merged = pending + line
                pending = ""
                let comment = merged.hasPrefix("#") || merged.hasPrefix(";")
                if !merged.isEmpty, !comment,
                   let eq = merged.firstIndex(of: "=") {
                    let key = merged[..<eq]
                        .trimmingCharacters(in: .whitespaces)
                    let value = merged[merged.index(after: eq)...]
                        .trimmingCharacters(in: .whitespaces)
                    result[String(key)] = String(value)
                }
            }
        }
        return result
    }

    private static func build(from dict: [String: String]) -> Loaded {
        var families: [String: [String: String]] = [:]
        var langs: [String: [String: String]] = [:]
        var themes: [String: [String: String]] = [:]
        for (k, v) in dict {
            let parts = k.split(separator: ".", maxSplits: 2,
                                omittingEmptySubsequences: false)
            if parts.count == 3 {
                let domain = String(parts[0])
                let id = String(parts[1])
                let prop = String(parts[2])
                switch domain {
                    case "family": families[id, default: [:]][prop] = v
                    case "lang": langs[id, default: [:]][prop] = v
                    case "theme": themes[id, default: [:]][prop] = v
                    default: break
                }
            }
        }
        let (languages, aliases) = buildLanguages(langs, families)
        let dark = themes["dark"] ?? [:]
        let light = themes["light"] ?? [:]
        func color(_ key: String) -> PlatformColor {
            platformAdaptiveColor(light: hex(light[key]), dark: hex(dark[key]))
        }
        return Loaded(
            languages: languages, aliases: aliases,
            keyword: color("keyword"), string: color("string"),
            number: color("number"), comment: color("comment"),
            type: color("type"), builtin: color("builtin"),
            variable: color("variable"), attr: color("attr"))
    }

    private static func buildLanguages(
        _ langs: [String: [String: String]],
        _ families: [String: [String: String]])
        -> (languages: [String: Spec], aliases: [String: String]) {
        var languages: [String: Spec] = [:]
        var aliases: [String: String] = [:]
        for (id, fields) in langs {
            let family = families[fields["family"] ?? ""] ?? [:]
            func pick(_ k: String) -> NSRegularExpression? {
                compiled(fields[k] ?? family[k])
            }
            let keywords = (fields["keywords"] ?? "")
                .split(separator: ",")
                .map { s in s.trimmingCharacters(in: .whitespaces) }
                .filter { s in !s.isEmpty }
            languages[id] = Spec(
                keywords: keywordPattern(keywords),
                lineComment: pick("lineComment"),
                blockComment: pick("blockComment"),
                string: pick("string"), number: pick("number"),
                tag: pick("tag"), attr: pick("attr"),
                meta: pick("meta"), type: pick("type"),
                builtin: pick("builtin"))
            aliases[id] = id
            for a in (fields["aliases"] ?? "").split(separator: ",") {
                let key = a.trimmingCharacters(in: .whitespaces).lowercased()
                if !key.isEmpty { aliases[key] = id }
            }
        }
        return (languages, aliases)
    }

    private static func hex(_ s: String?) -> PlatformColor {
        var result: PlatformColor = .gray
        if var v = s, !v.isEmpty {
            if v.hasPrefix("#") { v.removeFirst() }
            if v.count == 6, let n = UInt32(v, radix: 16) {
                let r = CGFloat((n >> 16) & 0xff) / 255.0
                let g = CGFloat((n >> 8) & 0xff) / 255.0
                let b = CGFloat(n & 0xff) / 255.0
                result = PlatformColor(red: r, green: g, blue: b, alpha: 1.0)
            }
        }
        return result
    }
}
