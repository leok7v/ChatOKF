import SwiftUI
import XCTest
@testable import MD
#if os(macOS)
import AppKit

@MainActor
final class BridgeUpdateTests: XCTestCase {

    @MainActor @Observable final class Feed {
        var text = ""
        var document = Markdown.parse("hello there")
    }

    struct Harness: View {
        let feed: Feed
        var body: some View {
            VStack {
                MarkdownTextView(feed.document, style: .default,
                                 scrolls: false)
                Text(feed.text)
            }
        }
    }

    static func longSource(blocks: Int) -> String {
        var out = ""
        for i in 0..<blocks {
            switch i % 10 {
                case 0:
                    out += "## Section \(i)\n\n"
                case 3:
                    out += "```swift\nlet v\(i) = \(i) * 2\nprint(v\(i))\n"
                        + "```\n\n"
                case 6:
                    out += "- first point \(i)\n- second point \(i)\n\n"
                default:
                    out += "Paragraph \(i) with **bold** and *italic* words, "
                        + "a `span` of code and a [link](https://x.y/\(i)).\n\n"
            }
            if i == 50 {
                out += "| Name | Value | Note |\n|---|---:|---|\n"
                for r in 0..<10 {
                    out += "| row \(r) | \(r * 7) | note for row \(r) |\n"
                }
                out += "\n"
            }
            if i == 100 || i == 200 {
                out += "![pic \(i)](https://example.invalid/pic\(i).png)\n\n"
            }
            if i == 150 { out += "$$\\frac{a}{b} + c^2$$\n\n" }
        }
        return out
    }

    static func images(for doc: Markdown.Document) -> [URL: NSImage] {
        var result: [URL: NSImage] = [:]
        for url in ImagePrefetch.collectURLs(in: doc) {
            let img = NSImage(size: NSSize(width: 120, height: 80))
            img.lockFocus()
            NSColor.systemBlue.setFill()
            NSRect(x: 0, y: 0, width: 120, height: 80).fill()
            img.unlockFocus()
            result[url] = img
        }
        return result
    }

    private func surface(_ ns: NSAttributedString) -> NativeText {
        NativeText(attributed: nil, ns: ns,
                   font: FontRole.body(15).platformFont,
                   nowrap: false, selectable: true, bold: false,
                   secondary: false, scrolls: false, find: nil,
                   findId: nil, speaking: nil)
    }

    private func textView(width: CGFloat) -> NativeText.ResizingTextView {
        let v = NativeText.ResizingTextView(
            frame: NSRect(x: 0, y: 0, width: width, height: 100))
        v.textContainerInset = .zero
        v.textContainer?.lineFragmentPadding = 0
        v.isVerticallyResizable = true
        v.textContainer?.containerSize = NSSize(
            width: width, height: CGFloat.greatestFiniteMagnitude)
        return v
    }

    private func ms(_ body: () -> Void) -> Double {
        let t0 = DispatchTime.now().uptimeNanoseconds
        body()
        return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    }

    private func pump() {
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.15))
    }

    func testTheBuiltStringIsFullyAttributed() {
        let doc = Markdown.parse(Self.longSource(blocks: 60))
        let ns = DocumentText.attributed(from: doc, style: .default,
                                         images: Self.images(for: doc),
                                         width: 600)
        let full = NSRange(location: 0, length: ns.length)
        var bareFont = 0
        var bareColor = 0
        ns.enumerateAttribute(.font, in: full, options: []) { v, r, _ in
            if v == nil { bareFont += r.length }
        }
        ns.enumerateAttribute(.foregroundColor, in: full,
                              options: []) { v, r, _ in
            if v == nil { bareColor += r.length }
        }
        XCTAssertEqual(bareFont, 0, "characters with no font")
        XCTAssertEqual(bareColor, 0, "characters with no colour")
        XCTAssertTrue(surface(ns).resolved() === ns,
                      "an ns: source must reach TextKit as it is")
    }

    func testHeadingAndCodeSizesReachTheSurface() {
        let style = MarkdownStyle(bodySize: 14, codeSize: 12,
                                  headingSizes: [22, 18, 16, 14, 14, 13])
        let doc = Markdown.parse(
            "# Title\n\nbody\n\n```swift\nlet x = 1\n```\n")
        let ns = surface(DocumentText.attributed(from: doc, style: style))
            .resolved()
        let title = ns.attribute(.font, at: 0, effectiveRange: nil) as? NSFont
        let code = (ns.string as NSString).range(of: "let x")
        let mono = ns.attribute(.font, at: code.location,
                                effectiveRange: nil) as? NSFont
        XCTAssertEqual(title?.pointSize, 22)
        XCTAssertEqual(mono?.pointSize, 12)
    }

    func testASpokenSentenceIsLocatedWhenItArrives() {
        let v = textView(width: 400)
        let style = MarkdownStyle.default
        let first = DocumentText.attributed(
            from: Markdown.parse("The opener.\n"), style: style)
        v.applyResolved(first)
        v.setSpoken("A late sentence.")
        let second = DocumentText.attributed(
            from: Markdown.parse("The opener. A late sentence.\n"),
            style: style)
        v.applyResolved(second)
        let at = (v.string as NSString).range(of: "A late sentence.").location
        let tint = v.layoutManager?.temporaryAttribute(
            .backgroundColor, atCharacterIndex: at, effectiveRange: nil)
        XCTAssertNotNil(tint, "the spoken sentence was not tinted")
        let before = v.layoutManager?.temporaryAttribute(
            .backgroundColor, atCharacterIndex: 0, effectiveRange: nil)
        XCTAssertNil(before, "the opener must not be tinted")
    }

    func testCopyButtonsFollowTheContent() {
        let v = textView(width: 400)
        let cache = DocumentText.RenderCache()
        let one = DocumentText.attributed(
            from: Markdown.parse("```\na\n```\n\ntext\n"), style: .default,
            cache: cache)
        v.applyResolved(one)
        v.layout()
        XCTAssertEqual(v.subviews.count, 1, "one code block, one button")
        let two = DocumentText.attributed(
            from: Markdown.parse("```\na\n```\n\ntext\n\n```\nb\n```\n"),
            style: .default, cache: cache)
        v.applyResolved(two)
        v.layout()
        XCTAssertEqual(v.subviews.count, 2, "two code blocks, two buttons")
        v.layout()
        XCTAssertEqual(v.subviews.count, 2)
    }

    func testATextOnlyChangeDoesNotRunTheMarkdownBody() {
        let feed = Feed()
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
            styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = NSHostingView(rootView: Harness(feed: feed))
        window.orderFront(nil)
        pump()
        feed.text = "warm"
        pump()
        let settled = MarkdownTextView.bodyEvaluations
        XCTAssertGreaterThan(settled, 0, "the harness never rendered")
        feed.text = "typed more"
        pump()
        XCTAssertEqual(MarkdownTextView.bodyEvaluations, settled,
                       "a text-only change re-ran the markdown body")
        feed.document = Markdown.parse("hello there friend")
        pump()
        XCTAssertGreaterThan(MarkdownTextView.bodyEvaluations, settled,
                             "a document change did not re-run the body")
        window.orderOut(nil)
    }

    func testUpdateAndLayoutCost() {
        let src = Self.longSource(blocks: 300)
        let doc = Markdown.parse(src)
        let images = Self.images(for: doc)
        let cache = DocumentText.RenderCache()
        let ns = DocumentText.attributed(from: doc, style: .default,
                                         images: images, width: 600,
                                         cache: cache)
        let v = textView(width: 600)
        let nt = surface(ns)
        func update(_ n: NativeText) {
            v.applyResolved(n.resolved())
            v.setSpoken(n.speaking)
        }
        let first = ms { update(nt) }
        let lay0 = ms { v.layout() }
        let doc2 = Markdown.parse(src + "One more sentence at the end.\n")
        let ns2 = DocumentText.attributed(from: doc2, style: .default,
                                          images: images, width: 600,
                                          cache: cache)
        let nt2 = surface(ns2)
        var upd = [Double]()
        var lay = [Double]()
        var same = [Double]()
        var idle = [Double]()
        for _ in 0..<5 {
            upd.append(ms { update(nt2) })
            lay.append(ms { v.layout() })
            idle.append(ms { v.layout() })
            same.append(ms { update(nt2) })
            _ = ms { update(nt) }
        }
        func list(_ xs: [Double]) -> String {
            xs.map { x in String(format: "%.2f", x) }.joined(separator: " ")
        }
        print(String(format: "COST chars=%d first=%.2f layout0=%.2f "
                     + "update(append)=%@ layout(after)=%@ layout(idle)=%@ "
                     + "update(same)=%@",
                     ns.length, first, lay0, list(upd), list(lay),
                     list(idle), list(same)))
    }
}
#endif
