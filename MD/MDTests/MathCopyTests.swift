import Foundation
import XCTest
@testable import MD
#if os(macOS)
import AppKit

// What a copied formula carries: a display is a layout, not characters,
// so the object-replacement character alone pastes as a gap.
@MainActor
final class MathCopyTests: XCTestCase {

    private func mathCell() throws -> MathAttachmentCell {
        let ns = DocumentText.attributed(
            from: Markdown.parse("$$\\frac{a}{b}$$"), style: .default)
        var found: MathAttachmentCell?
        ns.enumerateAttribute(.attachment,
                              in: NSRange(location: 0, length: ns.length),
                              options: []) { value, _, _ in
            if let cell = (value as? NSTextAttachment)?.attachmentCell
                as? MathAttachmentCell {
                found = cell
            }
        }
        return try XCTUnwrap(found, "no math attachment was built")
    }

    func testAFormulaRendersToAPDF() throws {
        let pdf = try XCTUnwrap(mathCell().pdf(dark: false), "no PDF produced")
        XCTAssertGreaterThan(pdf.count, 0)
        XCTAssertEqual(pdf.prefix(4), Data("%PDF".utf8), "not a PDF")
    }

    // Vector, so it stays crisp and prints.
    func testThePDFIsVectorAndSized() throws {
        let pdf = try XCTUnwrap(mathCell().pdf(dark: false))
        let image = try XCTUnwrap(NSImage(data: pdf), "NSImage refused it")
        XCTAssertGreaterThan(image.size.width, 0)
        XCTAssertGreaterThan(image.size.height, 0)
        XCTAssertTrue(image.representations.contains {
            $0 is NSPDFImageRep
        }, "the pasteboard image is not vector")
    }

    // MEASURED: AppKit's plain-RTF writer embeds NO picture for an image
    // attachment, so the rich flavour is RTFD and RTF gets the TeX.
    func testPlainRTFCarriesNoPictureButRTFDDoes() throws {
        let pdf = try XCTUnwrap(mathCell().pdf(dark: false))
        let image = try XCTUnwrap(NSImage(data: pdf))
        let shown = NSTextAttachment()
        shown.image = image
        let m = NSMutableAttributedString(string: "before ")
        m.append(NSAttributedString(attachment: shown))
        m.append(NSAttributedString(string: " after"))
        let full = NSRange(location: 0, length: m.length)
        let rtf = try XCTUnwrap(m.rtf(from: full, documentAttributes: [:]))
        let text = String(decoding: rtf, as: UTF8.self)
        print("RTFPROBE \(rtf.count) bytes, pict=\(text.contains("\\pict"))")
        XCTAssertFalse(text.contains("\\pict"),
                       "plain RTF grew a picture; the rich flavour could be "
                       + "RTF after all, so simplify writeSelection")
        let rtfd = try XCTUnwrap(m.rtfd(from: full, documentAttributes: [:]))
        XCTAssertGreaterThan(rtfd.count, rtf.count * 4,
                             "RTFD is too small to hold the formula")
    }

    // Drives a REAL copy through a real text view onto a real pasteboard.
    func testCopyPutsThePictureAndTheTeXOnThePasteboard() throws {
        let ns = DocumentText.attributed(
            from: Markdown.parse("$$\\frac{a}{b}$$"), style: .default)
        let view = NativeText.ResizingTextView(frame: .zero)
        view.textStorage?.setAttributedString(ns)
        view.setSelectedRange(NSRange(location: 0, length: ns.length))
        let board = NSPasteboard(name: .init("md.test.copy"))
        board.clearContents()
        // copy(_:) writes to the general pasteboard, so the assertions read
        // that; the named board only proves nothing stale is read.
        NSPasteboard.general.clearContents()
        view.copy(nil)
        let types = NSPasteboard.general.types ?? []
        XCTAssertTrue(types.contains(.rtfd), "no rtfd flavour: \(types)")
        let rtfd = try XCTUnwrap(NSPasteboard.general.data(forType: .rtfd))
        XCTAssertGreaterThan(rtfd.count, 2000,
                             "rtfd is \(rtfd.count) bytes, too small to "
                             + "hold the formula -- the picture is missing")
        let plain = NSPasteboard.general.string(forType: .string) ?? ""
        XCTAssertTrue(plain.contains("\\frac"),
                      "plain flavour is \(plain.debugDescription), not TeX")
    }

    // The PDF must hold the WHOLE formula: an origin computed from the
    // descent put the baseline below the media box, which shipped.
    func testThePDFHoldsTheWholeFormula() throws {
        let ns = DocumentText.attributed(
            from: Markdown.parse("$$\\frac{a}{bbb}$$"), style: .default)
        var cell: MathAttachmentCell?
        ns.enumerateAttribute(.attachment,
                              in: NSRange(location: 0, length: ns.length),
                              options: []) { value, _, _ in
            if let c = (value as? NSTextAttachment)?.attachmentCell
                as? MathAttachmentCell { cell = c }
        }
        let pdf = try XCTUnwrap(try XCTUnwrap(cell).pdf(dark: false))
        let image = try XCTUnwrap(NSImage(data: pdf))
        var box = CGRect(origin: .zero, size: image.size)
        let cg = try XCTUnwrap(image.cgImage(forProposedRect: &box,
                                             context: nil, hints: nil))
        let w = cg.width
        let h = cg.height
        var pixels = [UInt8](repeating: 0, count: w * h)
        let ctx = try XCTUnwrap(CGContext(
            data: &pixels, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue))
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Ink in the top third is the numerator, ink in the bottom third is
        // the denominator. A clipped formula has the first and not the second.
        var top = 0
        var bottom = 0
        for y in 0..<h {
            for x in 0..<w where pixels[y * w + x] < 128 {
                if y < h / 3 { top += 1 } else if y > 2 * h / 3 { bottom += 1 }
            }
        }
        XCTAssertGreaterThan(top, 0, "no ink at all; the PDF is blank")
        XCTAssertGreaterThan(bottom, 0,
                             "ink in the top third but none in the bottom: "
                             + "the formula is cut off")
    }

    // Grey level at a point of the rendered PDF, over a mid grey ground so
    // both a light and a dark patch are visible against it.
    private func sample(_ pdf: Data, at spot: (CGFloat, CGFloat)) throws
        -> CGFloat {
        let image = try XCTUnwrap(NSImage(data: pdf))
        var box = CGRect(origin: .zero, size: image.size)
        let cg = try XCTUnwrap(image.cgImage(forProposedRect: &box,
                                             context: nil, hints: nil))
        let w = cg.width, h = cg.height
        var pixels = [UInt8](repeating: 0, count: w * h)
        let ctx = try XCTUnwrap(CGContext(
            data: &pixels, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue))
        ctx.setFillColor(CGColor(gray: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let x = min(w - 1, max(0, Int(CGFloat(w) * spot.0)))
        let y = min(h - 1, max(0, Int(CGFloat(h) * spot.1)))
        return CGFloat(pixels[y * w + x]) / 255
    }

    // The theme comes from the caller, not the process: the system
    // appearance gave a dark patch for every copy on a dark Mac.
    func testTheCallerDecidesTheTheme() throws {
        let cell = try mathCell()
        let light = try XCTUnwrap(cell.pdf(dark: false))
        let dark = try XCTUnwrap(cell.pdf(dark: true))
        XCTAssertNotEqual(light, dark, "the theme flag changed nothing")
        // The SAME point in both, so the comparison needs no knowledge of
        // where the glyphs fall: light paper must read lighter than dark.
        let lightPaper = try sample(light, at: (0.5, 0.2))
        let darkPaper = try sample(dark, at: (0.5, 0.2))
        XCTAssertGreaterThan(lightPaper - darkPaper, 0.2,
                             "light \(lightPaper) vs dark \(darkPaper): the "
                             + "theme barely moved the paper")
    }

    // Edgeless on ALL four sides: a radial fade leaves the mid-edges of a
    // wide box opaque, so the mid-edge is what to sample.
    func testThePatchFadesAtEveryEdge() throws {
        let pdf = try XCTUnwrap(try mathCell().pdf(dark: false))
        let ground: CGFloat = 0.5
        let edges: [(String, (CGFloat, CGFloat))] =
            [("top", (0.5, 0.005)), ("bottom", (0.5, 0.995)),
             ("left", (0.004, 0.5)), ("right", (0.996, 0.5))]
        for (name, spot) in edges {
            let edge = try sample(pdf, at: spot)
            XCTAssertLessThan(abs(edge - ground), 0.12,
                              "the \(name) edge is opaque, not faded "
                              + "(\(edge) against a \(ground) ground)")
        }
        // Inside the patch there IS paper: lighter than the ground, at a
        // point the formula's own ink does not reach.
        let inside = try sample(pdf, at: (0.5, 0.2))
        XCTAssertGreaterThan(inside, 0.6,
                             "no paper inside the patch (\(inside))")
    }

    func testThePDFIsRenderedOnceAndKept() throws {
        let cell = try mathCell()
        let first = try XCTUnwrap(cell.pdf(dark: false))
        let second = try XCTUnwrap(cell.pdf(dark: false))
        XCTAssertEqual(first, second)
    }
}
#endif
