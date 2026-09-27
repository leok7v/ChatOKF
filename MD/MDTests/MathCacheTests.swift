import Foundation
import XCTest
@testable import MD

@MainActor
final class MathCacheTests: XCTestCase {

    private static let formula =
        "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} + \\sum_{i=1}^{n} "
        + "\\int_0^\\infty e^{-x^2} \\, dx"

    func testTheSameFormulaIsLaidOutOnce() throws {
        TeX.forgetLayouts()
        let first = try XCTUnwrap(TeX.layout(Self.formula, size: 20))
        let second = try XCTUnwrap(TeX.layout(Self.formula, size: 20))
        XCTAssertTrue(first.box === second.box)
        let other = try XCTUnwrap(TeX.layout(Self.formula, size: 21))
        XCTAssertFalse(first.box === other.box)
        XCTAssertEqual(TeX.cachedLayoutCount, 2)
    }

    func testARefusalIsRememberedToo() {
        TeX.forgetLayouts()
        XCTAssertNil(TeX.layout("\\notarealmacro{x}", size: 20))
        XCTAssertNil(TeX.layout("\\notarealmacro{x}", size: 20))
        XCTAssertEqual(TeX.cachedLayoutCount, 1)
    }

    func testTheLayoutCacheIsBounded() {
        TeX.forgetLayouts()
        for i in 0..<600 { _ = TeX.layout("x_{\(i)}", size: 20) }
        XCTAssertLessThanOrEqual(TeX.cachedLayoutCount, 256)
        XCTAssertGreaterThan(TeX.cachedLayoutCount, 0)
    }

    func testACachedLayoutCostsLessThanAColdOne() {
        let rounds = 50
        var cold: TimeInterval = 0
        var cached: TimeInterval = 0
        for _ in 0..<rounds {
            TeX.forgetLayouts()
            let t0 = Date()
            _ = TeX.layout(Self.formula, size: 20)
            cold += Date().timeIntervalSince(t0)
            let t1 = Date()
            _ = TeX.layout(Self.formula, size: 20)
            cached += Date().timeIntervalSince(t1)
        }
        let perCold = cold / Double(rounds) * 1000
        let perCached = cached / Double(rounds) * 1000
        print(String(format: "MATHCACHE layout cold %.3f ms, cached %.4f ms,"
                     + " %.0fx", perCold, perCached, perCold / perCached))
        XCTAssertLessThan(cached, cold)
    }

    func testFittedPdfSizesLeaveTheFontCacheBounded() throws {
        let font = try MathFontFile.shared()
        let before = font.cachedFontCount
        var formulas: [String] = []
        for n in 0..<80 {
            let terms = (0...(24 + n)).map { k in "x_{\(k)}" }
            formulas.append("$$" + terms.joined(separator: " + ") + "$$")
        }
        let doc = Markdown.parse(formulas.joined(separator: "\n\n"))
        XCTAssertEqual(doc.items.count, 80)
        XCTAssertNotNil(MarkdownPDF.data(doc, title: "t"))
        let grown = font.cachedFontCount - before
        print("MATHCACHE fonts grown by \(grown) over 80 fitted formulas")
        XCTAssertLessThanOrEqual(grown, 3 * 31)
    }

    #if os(iOS)
    func testARasterIsReusedForTheSameLayout() throws {
        let layout = try XCTUnwrap(TeX.layout("\\frac{a}{b}", size: 20))
        let first = DocumentText.mathAttachment(layout)
        let second = DocumentText.mathAttachment(layout)
        XCTAssertNotNil(first.image)
        XCTAssertTrue(first.image === second.image)
    }
    #endif
}
