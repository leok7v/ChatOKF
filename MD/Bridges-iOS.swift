#if os(iOS)
import SwiftUI
import UIKit

// The background color a run carried BEFORE a find tint replaced it, stashed in
// the text storage so clearing a find restores code / table tints.
private let findBaseBgKey = NSAttributedString.Key("MD.find.baseBg")

// iOS backing for NativeText. Self-sizing, non-scrolling, selectable.
extension NativeText: UIViewRepresentable {

    func makeUIView(context: Context) -> ResizingTextView {
        let v = ResizingTextView.textKit1()
        v.isEditable = false
        v.isSelectable = selectable
        // A whole-document surface scrolls internally so Find can
        // scrollRangeToVisible; a per-block chat surface self-sizes.
        v.scrolls = scrolls
        v.isScrollEnabled = scrolls
        v.backgroundColor = .clear
        v.textContainerInset = .zero
        v.textContainer.lineFragmentPadding = 0
        v.adjustsFontForContentSizeCategory = true
        v.linkTextAttributes = [
            .foregroundColor: UIColor.link,
            .underlineStyle: NSUnderlineStyle.single.rawValue,
        ]
        v.setContentCompressionResistancePriority(.defaultLow,
                                                  for: .horizontal)
        v.nowrap = nowrap
        if nowrap {
            v.textContainer.widthTracksTextView = false
            v.textContainer.size = CGSize(
                width: CGFloat.greatestFiniteMagnitude,
                height: CGFloat.greatestFiniteMagnitude)
        }
        // The OS find navigator (hardware Cmd+F) rides alongside the
        // controller-driven find; either presents and selects matches.
        if find != nil { v.isFindInteractionEnabled = true }
        v.findId = findId
        v.findController = find
        if let find, let findId { find.register(findId, v) }
        return v
    }

    static func dismantleUIView(_ v: ResizingTextView, coordinator: ()) {
        if let id = v.findId { v.findController?.unregister(id, v) }
    }

    func updateUIView(_ v: ResizingTextView, context: Context) {
        v.nowrap = nowrap
        v.isSelectable = selectable
        let next = MarkdownDiag.timed("resolve") { resolved() }
        MarkdownDiag.timed("setText len=\(next.length)") {
            v.applyResolved(next)
        }
        v.setSpoken(speaking)
    }

    // Height computed for the PROPOSED width: inside a LazyVStack the
    // intrinsic-size invalidation is ignored and every block stays one line.
    func sizeThatFits(_ proposal: ProposedViewSize,
                      uiView v: ResizingTextView,
                      context: Context) -> CGSize? {
        var result: CGSize? = nil
        // A scrolling surface accepts the proposed size (fills its frame); only
        // the self-sizing surface computes a height.
        if !nowrap, !scrolls, let w = proposal.width, w > 0, w.isFinite {
            let fit = MarkdownDiag.timed("size len=\(v.textStorage.length)") {
                v.sizeThatFits(
                    CGSize(width: w, height: .greatestFiniteMagnitude))
            }
            result = CGSize(width: w, height: ceil(fit.height))
        }
        return result
    }

    final class ResizingTextView: UITextView, FindableTextView {

        // The designated initializer: usingTextLayoutManager skips a Swift
        // subclass's stored-property initializers (MEASURED on iOS 18).
        static func textKit1() -> ResizingTextView {
            let storage = NSTextStorage()
            let layout = NSLayoutManager()
            let container = NSTextContainer(
                size: CGSize(width: 0,
                             height: CGFloat.greatestFiniteMagnitude))
            storage.addLayoutManager(layout)
            layout.addTextContainer(container)
            return ResizingTextView(frame: .zero, textContainer: container)
        }

        var nowrap: Bool = false
        var scrolls: Bool = false
        var findId: UUID?
        weak var findController: MarkdownFindController?
        private var lastWidth: CGFloat = 0
        private var findMatches: [NSRange] = []
        private var activeIndex: Int? = nil
        private var findQuery = ""
        private var findCaseSensitive = false
        private var spokenText: String?
        private var spokenRange: NSRange?
        private var copyButtons: [String: CopyRunButton] = [:]

        var liveFindCount: Int { findMatches.count }

        // The active match's vertical center as a fraction of the laid-out
        // height, so the transcript can scroll the exact line into view.
        func activeMatchFraction() -> CGFloat? {
            var result: CGFloat? = nil
            if let i = activeIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= textStorage.length {
                let gr = layoutManager.glyphRange(
                    forCharacterRange: findMatches[i],
                    actualCharacterRange: nil)
                let rect = layoutManager.boundingRect(forGlyphRange: gr,
                                                      in: textContainer)
                let h = layoutManager.usedRect(for: textContainer).height
                if h > 0 { result = min(1, max(0, rect.midY / h)) }
            }
            return result
        }

        // Find and spoken tints are real backgrounds here: stripped before
        // the diff and re-tinted after, or the splice reads them as edits.
        func applyResolved(_ next: NSAttributedString) {
            let active = !findQuery.isEmpty || spokenRange != nil
            if active { clearHighlights() }
            if !textStorage.isEqual(to: next) {
                textStorage.beginEditing()
                applyIncremental(textStorage, next)
                textStorage.endEditing()
                invalidateIntrinsicContentSize()
            }
            if active { reapplyFind() }
        }

        // Concrete (never a dynamic system color, which resolves to nil off a
        // trait environment and aborts the attribute set).
        private var findTint: UIColor {
            UIColor(red: 1.0, green: 0.84, blue: 0.2, alpha: 0.35)
        }
        private var activeTint: UIColor {
            UIColor(red: 1.0, green: 0.6, blue: 0.0, alpha: 0.6)
        }
        // Cool against find's warm pair, so the two never read as the same
        // thing when a search happens to be open while a reply is spoken.
        private var spokenTint: UIColor {
            UIColor(red: 0.25, green: 0.55, blue: 1.0, alpha: 0.28)
        }

        // Highlight EVERY match; the controller activates one later. No
        // selection here, so other bubbles do not each grab a selection.
        func findAll(_ query: String, caseSensitive: Bool) -> Int {
            findQuery = query
            findCaseSensitive = caseSensitive
            findMatches = markdownFindRanges(in: text, query: query,
                                             caseSensitive: caseSensitive)
            activeIndex = nil
            highlightAll()
            return findMatches.count
        }

        func setActive(_ localIndex: Int?) {
            activeIndex = localIndex
            highlightAll()
            if let i = localIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= textStorage.length {
                selectedRange = findMatches[i]
                scrollRangeToVisible(findMatches[i])
            } else {
                selectedRange = NSRange(location: 0, length: 0)
            }
        }

        func clearFind() {
            findQuery = ""
            findMatches = []
            activeIndex = nil
            highlightAll()
        }

        func reapplyFind() {
            if !findQuery.isEmpty {
                findMatches = markdownFindRanges(
                    in: text, query: findQuery,
                    caseSensitive: findCaseSensitive)
                if let a = activeIndex, a >= findMatches.count {
                    activeIndex = nil
                }
            }
            locateSpoken()
            highlightAll()
        }

        func setSpoken(_ text: String?) {
            if text != spokenText {
                spokenText = text
                locateSpoken()
                highlightAll()
            }
        }

        // A literal search, so a sentence the speech layer reshaped simply
        // does not tint rather than tinting the wrong one.
        private func locateSpoken() {
            let ns = text as NSString
            var found: NSRange? = nil
            if let want = spokenText, !want.isEmpty {
                let r = ns.range(of: want)
                if r.location != NSNotFound { found = r }
            }
            spokenRange = found
        }

        // UIKit's NSLayoutManager has no temporary attributes, so find tints
        // are real .backgroundColor.
        private func clearHighlights() {
            let full = NSRange(location: 0, length: textStorage.length)
            var restores: [(range: NSRange, base: Any)] = []
            textStorage.enumerateAttribute(findBaseBgKey, in: full,
                                           options: []) { val, r, _ in
                if let val { restores.append((r, val)) }
            }
            for e in restores {
                if let color = e.base as? UIColor {
                    textStorage.addAttribute(.backgroundColor, value: color,
                                             range: e.range)
                } else {
                    textStorage.removeAttribute(.backgroundColor,
                                                range: e.range)
                }
                textStorage.removeAttribute(findBaseBgKey, range: e.range)
            }
        }

        // The ONE writer of tinted backgrounds. Tints are computed from local
        // copies first: a write can re-enter here and reassign findMatches.
        private func highlightAll() {
            let matches = findMatches
            let active = activeIndex
            let spoken = spokenRange
            textStorage.beginEditing()
            clearHighlights()
            let len = textStorage.length
            var tinted: [(range: NSRange, tint: UIColor)] = []
            if let r = spoken, NSMaxRange(r) <= len {
                tinted.append((r, spokenTint))
            }
            for (i, r) in matches.enumerated() where NSMaxRange(r) <= len {
                tinted.append((r, i == active ? activeTint : findTint))
            }
            for entry in tinted {
                var bases: [(range: NSRange, base: Any)] = []
                textStorage.enumerateAttribute(.backgroundColor,
                                               in: entry.range,
                                               options: []) { val, sub, _ in
                    bases.append((sub, val ?? NSNull()))
                }
                for b in bases where
                    textStorage.attribute(findBaseBgKey, at: b.range.location,
                                          effectiveRange: nil) == nil {
                    textStorage.addAttribute(findBaseBgKey, value: b.base,
                                             range: b.range)
                }
                textStorage.addAttribute(.backgroundColor, value: entry.tint,
                                         range: entry.range)
            }
            textStorage.endEditing()
        }

        override var intrinsicContentSize: CGSize {
            // A scrolling surface fills whatever frame it is given; only the
            // self-sizing per-block surface reports its content height.
            if scrolls {
                return CGSize(width: UIView.noIntrinsicMetric,
                              height: UIView.noIntrinsicMetric)
            }
            layoutManager.ensureLayout(for: textContainer)
            let r = layoutManager.usedRect(for: textContainer)
            let h = r.height + textContainerInset.top +
                    textContainerInset.bottom
            let w = nowrap
                ? r.width + textContainerInset.left + textContainerInset.right
                : UIView.noIntrinsicMetric
            return CGSize(width: w, height: h)
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            if bounds.size.width != lastWidth {
                lastWidth = bounds.size.width
                invalidateIntrinsicContentSize()
            }
            rebuildCopyOverlays()
        }

        // A REAL control, unlike the macOS twin's decorative one: a touch
        // selection begins with a long press, which a button does not steal.
        private func rebuildCopyOverlays() {
            var live: Set<String> = []
            layoutManager.ensureLayout(for: textContainer)
            let inset = textContainerInset
            let full = NSRange(location: 0, length: textStorage.length)
            textStorage.enumerateAttribute(atomicIdKey, in: full,
                                           options: []) { value, range, _ in
                let id = value as? String
                let source = id == nil ? nil : textStorage.attribute(
                    atomicCopyKey, at: range.location,
                    effectiveRange: nil) as? String
                let kind = id == nil ? nil : textStorage.attribute(
                    atomicKindKey, at: range.location,
                    effectiveRange: nil) as? String
                if let id, source != nil {
                    let gr = layoutManager.glyphRange(
                        forCharacterRange: range, actualCharacterRange: nil)
                    let block = layoutManager.boundingRect(
                        forGlyphRange: gr, in: textContainer)
                    // Centred on the FIRST line fragment, not the block top,
                    // or it reads as sitting on the baseline.
                    let line = layoutManager.lineFragmentUsedRect(
                        forGlyphAt: gr.location, effectiveRange: nil)
                    // A display is CENTRED, so its button anchors to the
                    // full-width line fragment, not to the formula.
                    let right = kind == AtomicKind.math.rawValue
                        ? layoutManager.lineFragmentRect(
                            forGlyphAt: gr.location,
                            effectiveRange: nil).maxX
                        : block.maxX
                    let side = CopyRunButton.side
                    let btn = copyButtons[id] ?? makeCopyButton(id)
                    btn.frame = CGRect(
                        x: right + inset.left - side - 4,
                        y: line.minY + inset.top + (line.height - side) / 2,
                        width: side, height: side)
                    copyButtons[id] = btn
                    live.insert(id)
                }
            }
            for (id, btn) in copyButtons where !live.contains(id) {
                btn.removeFromSuperview()
                copyButtons[id] = nil
            }
        }

        private func makeCopyButton(_ id: String) -> CopyRunButton {
            let btn = CopyRunButton()
            btn.atomicId = id
            btn.addAction(UIAction { [weak self] _ in
                self?.copyBlock(id)
            }, for: .touchUpInside)
            addSubview(btn)
            return btn
        }

        // The block's CURRENT source by atomic id: ranges shift as tokens
        // stream, the id is stable.
        private func copyBlock(_ id: String) {
            var copy: String? = nil
            let full = NSRange(location: 0, length: textStorage.length)
            textStorage.enumerateAttribute(atomicIdKey, in: full,
                                           options: []) { value, range, stop in
                if value as? String == id {
                    copy = textStorage.attribute(
                        atomicCopyKey, at: range.location,
                        effectiveRange: nil) as? String
                    stop.pointee = true
                }
            }
            if let copy {
                platformSetClipboardString(copy)
                copyButtons[id]?.flashDone()
            }
        }
    }
}

private final class CopyRunButton: UIButton {

    static let side: CGFloat = 26

    var atomicId = ""

    init() {
        super.init(frame: .zero)
        tintColor = .secondaryLabel
        accessibilityLabel = "Copy"
        setSymbol("doc.on.doc")
    }

    required init?(coder: NSCoder) { nil }

    private func setSymbol(_ name: String) {
        setImage(UIImage(systemName: name), for: .normal)
    }

    func flashDone() {
        setSymbol("checkmark")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            [weak self] in self?.setSymbol("doc.on.doc")
        }
    }
}
#endif
