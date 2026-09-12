import Foundation
import XCTest
@testable import MD

// A display formula has no line breaks to give, so a narrower surface
// cuts it off; boundingRect clamps, so the rule is asked directly.
@MainActor
final class MathFitTests: XCTestCase {

    private static let wide =
        "$$ \\sum_{i=1}^{n} \\frac{\\alpha_i + \\beta_i}{\\gamma_i} = "
        + "\\int_0^\\infty e^{-x^2} \\, dx + \\sqrt{\\lambda + \\mu} $$"

    private let natural = CGSize(width: 300, height: 60)

    func testANarrowSurfaceScalesTheFormulaToIt() {
        let fitted = DocumentText.mathFit(natural: natural, available: 180)
        XCTAssertEqual(fitted.width, 180, accuracy: 0.01)
        // Uniform, or the formula is distorted rather than scaled.
        XCTAssertEqual(fitted.height, 60 * 180 / 300, accuracy: 0.01)
    }

    // A ceiling, not a resize: given room, the formula keeps its own size
    // rather than stretching to fill the bubble.
    func testAFormulaThatFitsIsLeftAlone() {
        for available in [300.0, 600.0, 5000.0] as [CGFloat] {
            let fitted = DocumentText.mathFit(natural: natural,
                                              available: available)
            XCTAssertEqual(fitted.width, 300, accuracy: 0.01)
            XCTAssertEqual(fitted.height, 60, accuracy: 0.01)
        }
    }

    // Never zero or negative however absurd the offer: a width of zero
    // arrives during first layout and must leave the formula alone.
    func testDegenerateWidthsLeaveItAlone() {
        for available in [0.0, -1.0] as [CGFloat] {
            let fitted = DocumentText.mathFit(natural: natural,
                                              available: available)
            XCTAssertEqual(fitted.width, 300, accuracy: 0.01)
        }
    }

    #if os(macOS)
    // The rule is only worth anything if TextKit asks and obeys.
    func testTextKitDrawsTheFittedWidth() throws {
        let ns = DocumentText.attributed(
            from: Markdown.parse(Self.wide), style: .default)
        let storage = NSTextStorage(attributedString: ns)
        let manager = NSLayoutManager()
        let box = NSTextContainer(size: CGSize(width: 900, height: 1e6))
        box.lineFragmentPadding = 0
        storage.addLayoutManager(manager)
        manager.addTextContainer(box)
        manager.ensureLayout(for: box)
        var natural: CGFloat = 0
        var drawn: CGFloat = 0
        let full = NSRange(location: 0, length: storage.length)
        storage.enumerateAttribute(.attachment, in: full,
                                   options: []) { value, range, _ in
            if let cell = (value as? NSTextAttachment)?
                .attachmentCell as? NSTextAttachmentCell {
                natural = cell.cellSize().width
                let glyphs = manager.glyphRange(forCharacterRange: range,
                                                actualCharacterRange: nil)
                drawn = manager.boundingRect(forGlyphRange: glyphs,
                                             in: box).width
            }
        }
        XCTAssertGreaterThan(natural, 0, "no math cell was laid out")
        XCTAssertEqual(drawn, natural, accuracy: 0.5,
                       "TextKit did not draw the size the cell asked for")
    }
    #endif
}
