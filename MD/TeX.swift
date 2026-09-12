import SwiftUI

// Inline math: spans are split out and LaTeX tokens mapped to Unicode.
// Not a layout engine; complex layouts degrade to readable text.
enum TeX {

    // KaTeX typesets a display where there is a context to draw into; nil
    // means it refused and the caller falls back to render(_:display:).
    static func layout(_ tex: String, size: CGFloat) -> MathLayout? {
        var settings = MathSettings()
        settings.displayMode = true
        settings.fontSize = size
        return try? KaTeX.layout(tex, settings: settings)
    }

    // Display maths is set larger than the prose around it, the way a TeX
    // document does.
    static func displaySize(body: CGFloat) -> CGFloat { body * 4 / 3 }

    // Whether the WHOLE string is TeX the parser recognises: every token known,
    // nothing left over.
    static func parses(_ tex: String) -> Bool {
        (try? Parser.parse(tex)) != nil
    }

    enum Segment {
        case text(String)
        case math(String, display: Bool)
    }

    static func split(_ s: String) -> [Segment] {
        var out: [Segment] = []
        var buf = ""
        var i = s.startIndex
        while i < s.endIndex {
            let c = s[i]
            var consumed = false
            if c == "\\",
               let next = s.index(i, offsetBy: 1, limitedBy: s.endIndex),
               next < s.endIndex {
                if s[next] == "$" {
                    buf.append("$")
                    i = s.index(after: next)
                    consumed = true
                } else if s[next] == "(" || s[next] == "[" {
                    let taken = takeBracketMath(s, from: i, open: s[next],
                                                into: &out, buf: &buf)
                    if let end = taken { i = end; consumed = true }
                }
            }
            if !consumed, c == "$" {
                let taken = takeMath(s, from: i, into: &out, buf: &buf)
                if let end = taken { i = end; consumed = true }
            }
            if !consumed {
                buf.append(c)
                i = s.index(after: i)
            }
        }
        if !buf.isEmpty { out.append(.text(buf)) }
        return out
    }

    private static func takeMath(_ s: String, from i: String.Index,
                                 into out: inout [Segment],
                                 buf: inout String) -> String.Index? {
        var result: String.Index? = nil
        var isDisplay = false
        if let nx = s.index(i, offsetBy: 1, limitedBy: s.endIndex) {
            isDisplay = nx < s.endIndex && s[nx] == "$"
        }
        let off = isDisplay ? 2 : 1
        let searchStart = s.index(i, offsetBy: off)
        if isDisplay, searchStart <= s.endIndex,
           let endRange = s.range(of: "$$",
                                  range: searchStart..<s.endIndex) {
            if !buf.isEmpty { out.append(.text(buf)); buf.removeAll() }
            let body = String(s[searchStart..<endRange.lowerBound])
            out.append(.math(body, display: true))
            result = endRange.upperBound
        } else if !isDisplay, searchStart <= s.endIndex,
                  let body = inlineBody(s, from: searchStart) {
            if !buf.isEmpty { out.append(.text(buf)); buf.removeAll() }
            out.append(.math(String(s[body]), display: false))
            result = s.index(after: body.upperBound)
        }
        return result
    }

    private static func inlineBody(_ s: String, from start: String.Index)
        -> Range<String.Index>? {
        var result: Range<String.Index>? = nil
        if start < s.endIndex, !s[start].isWhitespace,
           let close = s[start...].firstIndex(of: "$"), close > start {
            let after = s.index(after: close)
            let digit = after < s.endIndex && s[after].isNumber
            if !s[s.index(before: close)].isWhitespace, !digit {
                result = start..<close
            }
        }
        return result
    }

    // A \( ... \) inline or \[ ... \] display span; an unclosed opener stays
    // literal so a streaming prefix does not flicker into math.
    private static func takeBracketMath(_ s: String, from i: String.Index,
                                        open: Character,
                                        into out: inout [Segment],
                                        buf: inout String) -> String.Index? {
        var result: String.Index? = nil
        let display = open == "["
        let closer = display ? "\\]" : "\\)"
        let searchStart = s.index(i, offsetBy: 2)
        if searchStart <= s.endIndex,
           let endRange = s.range(of: closer,
                                  range: searchStart..<s.endIndex) {
            if !buf.isEmpty { out.append(.text(buf)); buf.removeAll() }
            let body = String(s[searchStart..<endRange.lowerBound])
            out.append(.math(body, display: display))
            result = endRange.upperBound
        }
        return result
    }

    // The math run carries the emphasized INTENT, not a SwiftUI font, which
    // the NSAttributedString bridge would drop.
    static func render(_ src: String, display: Bool) -> AttributedString {
        var a = AttributedString(renderToString(src))
        a.inlinePresentationIntent = .emphasized
        return a
    }

    private static func renderToString(_ src: String) -> String {
        var s = expandText(src)
        s = stripEnvironments(s)
        s = expandFractions(s)
        s = replaceTokens(s)
        s = expandScript(s, prefix: "^", map: superscriptMap)
        s = expandScript(s, prefix: "_", map: subscriptMap)
        s = stripCommands(s)
        s = s.replacingOccurrences(of: "{", with: "")
             .replacingOccurrences(of: "}", with: "")
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func expandText(_ s: String) -> String {
        var out = s
        let pattern = #"\\text\s*\{([^{}]*)\}"#
        while let r = out.range(of: pattern, options: .regularExpression) {
            let replaced = out[r].replacingOccurrences(
                of: #"^\\text\s*\{([^{}]*)\}$"#,
                with: "{$1}", options: .regularExpression)
            out.replaceSubrange(r, with: replaced)
        }
        return out
    }

    // Environments vanish and row breaks become newlines HERE, ahead of the
    // token map whose "\ " entry would race the break's second backslash.
    private static func stripEnvironments(_ s: String) -> String {
        var out = s
        let pattern = #"\\(begin|end)\s*\{[^}]*\}"#
        while let r = out.range(of: pattern, options: .regularExpression) {
            out.replaceSubrange(r, with: "")
        }
        return out.replacingOccurrences(of: "\\\\", with: "\n")
            .replacingOccurrences(of: "&", with: "")
    }

    // Remaining unknown commands degrade to their bare word ("\operatorname"
    // -> "operatorname"), never a leaked backslash.
    private static func stripCommands(_ s: String) -> String {
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            let c = s[i]
            let next = s.index(after: i)
            if c == "\\", next < s.endIndex, s[next].isLetter {
                i = next
            } else {
                out.append(c)
                i = next
            }
        }
        return out
    }

    private static func expandFractions(_ s: String) -> String {
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            var consumed = false
            if let frac = parseFracAt(s, from: i) {
                out.append(frac.a)
                out.append("\u{2044}")
                out.append(frac.b)
                i = frac.end
                consumed = true
            }
            if !consumed {
                out.append(s[i])
                i = s.index(after: i)
            }
        }
        return out
    }

    private static func parseFracAt(_ s: String, from: String.Index)
        -> (a: String, b: String, end: String.Index)? {
        var result: (String, String, String.Index)? = nil
        if let afterCmd = s.index(from, offsetBy: 5, limitedBy: s.endIndex),
           s[from..<afterCmd] == "\\frac" {
            var j = afterCmd
            while j < s.endIndex, s[j].isWhitespace { j = s.index(after: j) }
            if j < s.endIndex, s[j] == "{", let endA = matchBrace(s, from: j) {
                var k = s.index(after: endA)
                while k < s.endIndex, s[k].isWhitespace {
                    k = s.index(after: k)
                }
                if k < s.endIndex, s[k] == "{",
                   let endB = matchBrace(s, from: k) {
                    let a = String(s[s.index(after: j)..<endA])
                    let b = String(s[s.index(after: k)..<endB])
                    result = (a, b, s.index(after: endB))
                }
            }
        }
        return result
    }

    private static func matchBrace(_ s: String,
                                   from: String.Index) -> String.Index? {
        var result: String.Index? = nil
        if from < s.endIndex, s[from] == "{" {
            var depth = 1
            var i = s.index(after: from)
            while i < s.endIndex, result == nil {
                if s[i] == "{" { depth += 1 }
                else if s[i] == "}" {
                    depth -= 1
                    if depth == 0 { result = i }
                }
                i = s.index(after: i)
            }
        }
        return result
    }

    private static func expandScript(_ s: String, prefix: Character,
                                     map: [Character: Character]) -> String {
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            let c = s[i]
            var consumed = false
            if c == prefix,
               let next = s.index(i, offsetBy: 1, limitedBy: s.endIndex),
               next < s.endIndex {
                let taken = takeScript(s, next: next, map: map,
                                       prefix: prefix, into: &out)
                if let end = taken { i = end; consumed = true }
            }
            if !consumed {
                out.append(c)
                i = s.index(after: i)
            }
        }
        return out
    }

    private static func takeScript(_ s: String, next: String.Index,
                                   map: [Character: Character],
                                   prefix: Character,
                                   into out: inout String) -> String.Index? {
        var result: String.Index? = nil
        let after = s[next]
        if after == "{" {
            let tail = s[s.index(after: next)...]
            if let close = tail.firstIndex(of: "}") {
                let start = s.index(after: next)
                out.append(mapScript(String(s[start..<close]), map: map,
                                     prefix: prefix))
                result = s.index(after: close)
            }
        } else {
            out.append(mapScript(String(after), map: map, prefix: prefix))
            result = s.index(after: next)
        }
        return result
    }

    // A script whose every character has a Unicode form maps whole; anything
    // else keeps its operator so the meaning survives.
    private static func mapScript(_ s: String,
                                  map: [Character: Character],
                                  prefix: Character) -> String {
        var result = ""
        if !s.isEmpty {
            let mapped = s.compactMap { c in map[c] }
            if mapped.count == s.count {
                result = String(mapped)
            } else if s.count == 1 {
                result = "\(prefix)\(s)"
            } else {
                result = "\(prefix)(\(s))"
            }
        }
        return result
    }

    // The plain-text answer to <sub>/<sup>: every character or none, so one
    // unrepresentable letter sends the whole run to parentheses.
    static func unicodeScript(_ s: String, superscript sup: Bool) -> String {
        let map = sup ? superscriptMap : subscriptMap
        let mapped = s.compactMap { c in map[c] }
        var result = "(" + s + ")"
        if s.isEmpty {
            result = ""
        } else if mapped.count == s.count {
            result = String(mapped)
        }
        return result
    }

    // A body with no '<' is an INNERMOST pair, which is what makes one pass
    // safe against nested tags.
    private static let scriptTagRE: NSRegularExpression? =
        try? NSRegularExpression(pattern: #"<(sub|sup)>([^<]*)</\1>"#,
                                 options: .caseInsensitive)

    // Rewrites the tags in a RAW cell for the table measurers, repeated
    // until it stops changing so nesting unwinds inside out.
    static func scriptsToUnicode(_ s: String) -> String {
        var result = s
        var unwinding = s.contains("<")
        while unwinding {
            let next = innermostScripts(result)
            unwinding = next != result
            result = next
        }
        return result
    }

    private static func innermostScripts(_ s: String) -> String {
        var result = s
        if let re = scriptTagRE {
            let ns = s as NSString
            let full = NSRange(location: 0, length: ns.length)
            let m = NSMutableString(string: s)
            for match in re.matches(in: s, range: full).reversed() {
                let tag = ns.substring(with: match.range(at: 1))
                let body = ns.substring(with: match.range(at: 2))
                let sup = tag.lowercased() == "sup"
                m.replaceCharacters(in: match.range,
                                    with: unicodeScript(body, superscript: sup))
            }
            result = m as String
        }
        return result
    }

    // Longest-first, then lexicographic, so equal-length keys replace in a
    // deterministic order.
    private static func replaceTokens(_ s: String) -> String {
        var out = s
        let pairs = tokenMap.sorted { a, b in
            a.key.count > b.key.count
                || (a.key.count == b.key.count && a.key > b.key)
        }
        for (k, v) in pairs { out = replaceToken(out, k, v) }
        return out
    }

    // A control word ends where a non-letter begins, or \ne fires inside
    // \newcommand. Keys not ending in a letter have no boundary.
    private static func replaceToken(_ s: String, _ key: String,
                                     _ value: String) -> String {
        var result = s
        if let last = key.last, last.isLetter {
            let pattern = NSRegularExpression.escapedPattern(for: key)
                        + "(?![A-Za-z])"
            // Case-sensitive, as TeX is, except where the expansion is
            // empty and dropping a mis-cased command loses nothing.
            let folding: NSRegularExpression.Options =
                value.isEmpty ? [.caseInsensitive] : []
            if let re = try? NSRegularExpression(pattern: pattern,
                                                 options: folding) {
                let ns = s as NSString
                result = re.stringByReplacingMatches(
                    in: s, range: NSRange(location: 0, length: ns.length),
                    withTemplate:
                        NSRegularExpression.escapedTemplate(for: value))
            }
        } else {
            result = s.replacingOccurrences(of: key, with: value)
        }
        return result
    }

    private static let tokenMap: [String: String] = [
        "\\alpha": "\u{03B1}", "\\beta": "\u{03B2}", "\\gamma": "\u{03B3}",
        "\\delta": "\u{03B4}", "\\epsilon": "\u{03B5}",
        "\\varepsilon": "\u{03B5}", "\\zeta": "\u{03B6}",
        "\\eta": "\u{03B7}", "\\theta": "\u{03B8}",
        "\\vartheta": "\u{03D1}", "\\iota": "\u{03B9}",
        "\\kappa": "\u{03BA}", "\\lambda": "\u{03BB}", "\\mu": "\u{03BC}",
        "\\nu": "\u{03BD}", "\\xi": "\u{03BE}", "\\pi": "\u{03C0}",
        "\\varpi": "\u{03D6}", "\\rho": "\u{03C1}", "\\varrho": "\u{03F1}",
        "\\sigma": "\u{03C3}", "\\varsigma": "\u{03C2}", "\\tau": "\u{03C4}",
        "\\upsilon": "\u{03C5}", "\\phi": "\u{03C6}", "\\varphi": "\u{03D5}",
        "\\chi": "\u{03C7}", "\\psi": "\u{03C8}", "\\omega": "\u{03C9}",
        "\\Gamma": "\u{0393}", "\\Delta": "\u{0394}", "\\Theta": "\u{0398}",
        "\\Lambda": "\u{039B}", "\\Xi": "\u{039E}", "\\Pi": "\u{03A0}",
        "\\Sigma": "\u{03A3}", "\\Upsilon": "\u{03A5}", "\\Phi": "\u{03A6}",
        "\\Psi": "\u{03A8}", "\\Omega": "\u{03A9}",
        "\\times": "\u{00D7}", "\\cdot": "\u{00B7}", "\\div": "\u{00F7}",
        "\\pm": "\u{00B1}", "\\mp": "\u{2213}",
        "\\le": "\u{2264}", "\\leq": "\u{2264}", "\\ge": "\u{2265}",
        "\\geq": "\u{2265}", "\\neq": "\u{2260}", "\\ne": "\u{2260}",
        "\\approx": "\u{2248}", "\\equiv": "\u{2261}", "\\sim": "\u{223C}",
        "\\propto": "\u{221D}", "\\to": "\u{2192}",
        "\\rightarrow": "\u{2192}", "\\leftarrow": "\u{2190}",
        "\\Rightarrow": "\u{21D2}", "\\Leftarrow": "\u{21D0}",
        "\\leftrightarrow": "\u{2194}", "\\Leftrightarrow": "\u{21D4}",
        "\\sum": "\u{2211}", "\\prod": "\u{220F}", "\\int": "\u{222B}",
        "\\oint": "\u{222E}", "\\infty": "\u{221E}",
        "\\partial": "\u{2202}", "\\nabla": "\u{2207}",
        "\\forall": "\u{2200}", "\\exists": "\u{2203}",
        "\\nexists": "\u{2204}", "\\in": "\u{2208}", "\\notin": "\u{2209}",
        "\\subset": "\u{2282}", "\\supset": "\u{2283}",
        "\\subseteq": "\u{2286}", "\\supseteq": "\u{2287}",
        "\\cup": "\u{222A}", "\\cap": "\u{2229}",
        "\\emptyset": "\u{2205}", "\\varnothing": "\u{2205}",
        "\\sqrt": "\u{221A}", "\\angle": "\u{2220}", "\\perp": "\u{22A5}",
        "\\parallel": "\u{2225}", "\\land": "\u{2227}", "\\lor": "\u{2228}",
        "\\lnot": "\u{00AC}", "\\neg": "\u{00AC}", "\\dots": "\u{2026}",
        "\\ldots": "\u{2026}", "\\cdots": "\u{22EF}", "\\vdots": "\u{22EE}",
        "\\hbar": "\u{210F}", "\\ell": "\u{2113}", "\\Re": "\u{211C}",
        "\\Im": "\u{2111}", "\\mathbb{R}": "\u{211D}",
        "\\mathbb{N}": "\u{2115}", "\\mathbb{Z}": "\u{2124}",
        "\\mathbb{Q}": "\u{211A}", "\\mathbb{C}": "\u{2102}",
        // Operator names render as their plain words; the wrappers vanish
        // (their brace payload survives the later brace strip).
        "\\arcsin": "arcsin", "\\arccos": "arccos", "\\arctan": "arctan",
        "\\sinh": "sinh", "\\cosh": "cosh", "\\tanh": "tanh",
        "\\sin": "sin", "\\cos": "cos", "\\tan": "tan",
        "\\sec": "sec", "\\csc": "csc", "\\cot": "cot",
        "\\ln": "ln", "\\log": "log", "\\exp": "exp", "\\lim": "lim",
        "\\min": "min", "\\max": "max", "\\arg": "arg", "\\det": "det",
        "\\gcd": "gcd", "\\deg": "deg", "\\dim": "dim", "\\bmod": "mod",
        "\\mod": "mod",
        "\\mathbf": "", "\\mathrm": "", "\\mathit": "", "\\mathsf": "",
        "\\mathcal": "", "\\boldsymbol": "", "\\operatorname": "",
        "\\displaystyle": "", "\\textstyle": "",
        // \circ is the ring operator inline; superscripted (57.3^\circ) the
        // script map turns it into the degree sign.
        "\\circ": "\u{2218}", "\\degree": "\u{00B0}",
        "\\left": "", "\\right": "", "\\,": " ", "\\;": " ", "\\ ": " ",
        "\\quad": " ", "\\qquad": "  ", "\\!": "", "\\:": " ",
    ]

    private static let superscriptMap: [Character: Character] = [
        "0": "\u{2070}", "1": "\u{00B9}", "2": "\u{00B2}", "3": "\u{00B3}",
        "4": "\u{2074}", "5": "\u{2075}", "6": "\u{2076}", "7": "\u{2077}",
        "8": "\u{2078}", "9": "\u{2079}", "+": "\u{207A}", "-": "\u{207B}",
        "\u{2212}": "\u{207B}", "=": "\u{207C}", "(": "\u{207D}",
        ")": "\u{207E}", "a": "\u{1D43}",
        "b": "\u{1D47}", "c": "\u{1D9C}", "d": "\u{1D48}", "e": "\u{1D49}",
        "f": "\u{1DA0}", "g": "\u{1D4D}", "h": "\u{02B0}", "i": "\u{2071}",
        "j": "\u{02B2}", "k": "\u{1D4F}", "l": "\u{02E1}", "m": "\u{1D50}",
        "n": "\u{207F}", "o": "\u{1D52}", "p": "\u{1D56}", "r": "\u{02B3}",
        "s": "\u{02E2}", "t": "\u{1D57}", "u": "\u{1D58}", "v": "\u{1D5B}",
        "w": "\u{02B7}", "x": "\u{02E3}", "y": "\u{02B8}", "z": "\u{1DBB}",
        "\u{2218}": "\u{00B0}", "\u{00B0}": "\u{00B0}",
    ]

    private static let subscriptMap: [Character: Character] = [
        "0": "\u{2080}", "1": "\u{2081}", "2": "\u{2082}", "3": "\u{2083}",
        "4": "\u{2084}", "5": "\u{2085}", "6": "\u{2086}", "7": "\u{2087}",
        "8": "\u{2088}", "9": "\u{2089}", "+": "\u{208A}", "-": "\u{208B}",
        "\u{2212}": "\u{208B}", "=": "\u{208C}", "(": "\u{208D}",
        ")": "\u{208E}", "a": "\u{2090}",
        "e": "\u{2091}", "h": "\u{2095}", "i": "\u{1D62}", "j": "\u{2C7C}",
        "k": "\u{2096}", "l": "\u{2097}", "m": "\u{2098}", "n": "\u{2099}",
        "o": "\u{2092}", "p": "\u{209A}", "r": "\u{1D63}", "s": "\u{209B}",
        "t": "\u{209C}", "u": "\u{1D64}", "v": "\u{1D65}", "x": "\u{2093}",
    ]
}
