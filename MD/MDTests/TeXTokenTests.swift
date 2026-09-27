import Foundation
import XCTest
@testable import MD

@MainActor
final class TeXTokenTests: XCTestCase {

    private enum Rule {
        case bounded(NSRegularExpression, String)
        case literal(String, String)
    }

    private static let rules: [Rule] = TeX.tokenMap
        .sorted { a, b in
            a.key.count > b.key.count
                || (a.key.count == b.key.count && a.key > b.key)
        }
        .compactMap { pair in rule(pair.key, pair.value) }

    private static func rule(_ key: String, _ value: String) -> Rule? {
        var result: Rule? = nil
        if let last = key.last, last.isLetter {
            let pattern = NSRegularExpression.escapedPattern(for: key)
                        + "(?![A-Za-z])"
            let folding: NSRegularExpression.Options =
                value.isEmpty ? [.caseInsensitive] : []
            if let re = try? NSRegularExpression(pattern: pattern,
                                                 options: folding) {
                result = .bounded(
                    re, NSRegularExpression.escapedTemplate(for: value))
            }
        } else {
            result = .literal(key, value)
        }
        return result
    }

    private static func sequential(_ s: String) -> String {
        var out = s
        for rule in rules {
            switch rule {
                case .bounded(let re, let template):
                    let ns = out as NSString
                    out = re.stringByReplacingMatches(
                        in: out,
                        range: NSRange(location: 0, length: ns.length),
                        withTemplate: template)
                case .literal(let key, let value):
                    out = out.replacingOccurrences(of: key, with: value)
            }
        }
        return out
    }

    private static let tricky: [String] = [
        "\\alpha\\beta\\gamma", "\\ne\\alpha", "\\alpha\\ne",
        "\\mathbb{R}\\mathbb{X}\\mathbb {R} \\mathbb{N}\\mathbb{Z}",
        "\\LEFT(\\sum_i a_i\\RIGHT)", "\\Left(\\sum_i a_i\\Right)",
        "\\ALPHA + \\beta", "\\newcommand\\ne1", "\\ne 1 \\neq 2",
        "\\,\\;\\ \\quad\\qquad\\!\\:", "a\\,b\\;c\\quad d\\qquad e",
        "\\sin\\theta + \\cos\\phi", "\\frac{\\partial u}{\\partial t}",
        "\\alpha\u{0301}", "\\ne\u{03B1}", "x\\to\\infty",
        "a\\leq b\\geq c\\le d\\ge e", "\\{x\\}", "\\\\", "trailing\\",
        "\\ ", "\\", "", "no commands at all",
        "\\mathbfx \\mathbf{x}\\mathrm{d}x", "\\sqrt{2}\\int_0^\\infty",
        "\\operatorname{sinc}\\lim_{x\\to0}",
        "\\varepsilon\\epsilon\\vartheta\\theta\\varphi\\phi",
        "\\Rightarrow\\rightarrow\\to\\leftrightarrow\\Leftrightarrow",
        "\\cdots\\dots\\ldots\\vdots", "\\foo\\alpha\\bar{x}\\baz",
        "\\mod \\bmod \\degree\\circ", "57.3^\\circ",
        "\\displaystyle\\frac{a}{b}\\textstyle\\frac{c}{d}",
        "\\Mathbb{R}", "\\MATHBB{R}", "\\mathbb{RR}", "\\mathbb{",
        "\\alphabet\\betamax", "\\pi r^2 \\mu\\nu",
        "\\$ 5 \\% \\& \\#", "\\left( \\frac{a}{b} \\right)",
        "\\mathcal{L}\\boldsymbol{\\alpha}", "\\int\\oint\\sum\\prod",
        "\\in\\notin\\subset\\supseteq\\cup\\cap\\emptyset\\varnothing",
        "\\lnot\\neg\\land\\lor\\forall\\exists\\nexists",
        "\\hbar\\ell\\Re\\Im\\nabla\\partial\\perp\\parallel\\angle",
        "\\arcsin x \\arccos y \\arctan z \\sinh a \\cosh b \\tanh c",
        "\\sec \\csc \\cot \\ln \\log \\exp \\min \\max \\arg \\det",
        "\\gcd \\deg \\dim \\lim", "\\times\\cdot\\div\\pm\\mp",
        "\\approx\\equiv\\sim\\propto", "\\Gamma\\Delta\\Theta\\Omega",
        "\\alpha\n\\beta\t\\gamma", "e^{i\\pi} + 1 = 0",
        "\\vec{v} \\cdot \\vec{w} = |v||w|\\cos\\theta",
    ]

    private static var spans: [String] {
        KaTeXGoldenTests.corpus.map { entry in entry.tex } + tricky
    }

    func testOneScanMatchesTheSequentialPasses() {
        for span in Self.spans {
            XCTAssertEqual(TeX.replaceTokens(span), Self.sequential(span),
                           span.debugDescription)
        }
    }

    func testAnOperatorNameDoesNotBlockTheWordBeforeIt() {
        XCTAssertEqual(TeX.replaceTokens("a\\ne\\sin b"), "a\u{2260}sin b")
        XCTAssertEqual(Self.sequential("a\\ne\\sin b"), "a\\nesin b")
        XCTAssertEqual(TeX.replaceTokens("\\le\\ln x"), "\u{2264}ln x")
        XCTAssertEqual(TeX.replaceTokens("2\\pi\\cos t"), "2\u{03C0}cos t")
    }

    func testOneScanIsCheaperThanTheSequentialPasses() {
        let spans = Self.spans
        let rounds = 20
        let t0 = Date()
        for _ in 0..<rounds { for s in spans { _ = Self.sequential(s) } }
        let old = Date().timeIntervalSince(t0)
        let t1 = Date()
        for _ in 0..<rounds { for s in spans { _ = TeX.replaceTokens(s) } }
        let new = Date().timeIntervalSince(t1)
        let n = Double(rounds * spans.count)
        print(String(format: "TEXTOKEN sequential %.1f us/span, one scan"
                     + " %.1f us/span, %.0fx over %d spans",
                     old / n * 1e6, new / n * 1e6, old / new, spans.count))
        XCTAssertLessThan(new, old)
    }
}
