#if os(macOS)
import SwiftUI
import AppKit

// macOS backing for NativeText. Selection expands to whole atomic units
// (code / table / image) so a drag never cuts a highlighted block in half.
extension NativeText: NSViewRepresentable {

    final class Coordinator: NSObject, NSTextViewDelegate {

        private var anchorScope: NSRange? = nil
        var findId: UUID?
        weak var findController: MarkdownFindController?

        func textView(_ tv: NSTextView, clickedOnLink link: Any,
                      at: Int) -> Bool {
            var url: URL? = nil
            if let u = link as? URL { url = u }
            else if let s = link as? String { url = URL(string: s) }
            var handled = false
            if let url {
                NSWorkspace.shared.open(url)
                handled = true
            }
            return handled
        }

        func textView(_ textView: NSTextView,
                      willChangeSelectionFromCharacterRange old: NSRange,
                      toCharacterRange new: NSRange) -> NSRange {
            var result = new
            if let storage = textView.textStorage {
                if new.length == 0 {
                    anchorScope = atomicScope(at: new.location, in: storage)
                } else if let scope = anchorScope {
                    result = extend(new, scope: scope, in: storage)
                } else {
                    result = expand(new, in: storage)
                }
            }
            return result
        }

        private func extend(_ new: NSRange, scope: NSRange,
                            in storage: NSTextStorage) -> NSRange {
            let endLo = new.location
            let endHi = new.location + new.length
            let scopeLo = scope.location
            let scopeHi = scope.location + scope.length
            let inside = endLo >= scopeLo && endHi <= scopeHi
            var result = new
            if !inside {
                let lo = min(endLo, scopeLo)
                let hi = max(endHi, scopeHi)
                result = expand(NSRange(location: lo, length: hi - lo),
                                in: storage)
            }
            return result
        }

        private func atomicScope(at pos: Int,
                                 in storage: NSTextStorage) -> NSRange? {
            var result: NSRange? = nil
            if pos >= 0, pos < storage.length {
                var effective = NSRange(location: 0, length: 0)
                let value = storage.attribute(atomicKindKey, at: pos,
                                              effectiveRange: &effective)
                if value != nil { result = effective }
            }
            return result
        }

        private func expand(_ range: NSRange,
                            in storage: NSTextStorage) -> NSRange {
            var lo = range.location
            var hi = range.location + range.length
            storage.enumerateAttribute(atomicKindKey, in: range,
                                       options: []) { value, r, _ in
                if value != nil {
                    if r.location < lo { lo = r.location }
                    let end = r.location + r.length
                    if end > hi { hi = end }
                }
            }
            return NSRange(location: lo, length: hi - lo)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> ResizingTextView {
        let v = ResizingTextView()
        v.delegate = context.coordinator
        v.isEditable = false
        v.isSelectable = selectable
        v.drawsBackground = false
        v.backgroundColor = .clear
        v.textContainerInset = .zero
        v.textContainer?.lineFragmentPadding = 0
        v.textContainer?.widthTracksTextView = !nowrap
        if nowrap {
            v.textContainer?.containerSize = NSSize(
                width: CGFloat.greatestFiniteMagnitude,
                height: CGFloat.greatestFiniteMagnitude)
        }
        v.isVerticallyResizable = true
        v.isHorizontallyResizable = nowrap
        v.setContentCompressionResistancePriority(.defaultLow,
                                                  for: .horizontal)
        v.linkTextAttributes = [
            .foregroundColor: NSColor.linkColor,
            .underlineStyle: NSUnderlineStyle.single.rawValue,
            .cursor: NSCursor.pointingHand,
        ]
        context.coordinator.findId = findId
        context.coordinator.findController = find
        if let find, let findId { find.register(findId, v) }
        return v
    }

    static func dismantleNSView(_ v: ResizingTextView,
                                coordinator: Coordinator) {
        if let id = coordinator.findId {
            coordinator.findController?.unregister(id, v)
        }
    }

    func updateNSView(_ v: ResizingTextView, context: Context) {
        v.nowrap = nowrap
        v.isSelectable = selectable
        let next = resolved()
        if let ts = v.textStorage, !ts.isEqual(to: next) {
            ts.beginEditing()
            applyIncremental(ts, next)
            ts.endEditing()
            v.invalidateIntrinsicContentSize()
            v.reapplyFind()
        }
        v.setSpoken(speaking)
    }

    final class ResizingTextView: NSTextView, FindableTextView {

        var nowrap: Bool = false
        private var lastBounds: NSSize = .zero
        private var copyButtons: [String: CopyRunButton] = [:]
        private var copyRects: [(id: String, rect: NSRect)] = []
        private var findMatches: [NSRange] = []
        private var activeIndex: Int? = nil
        private var findQuery = ""
        private var findCaseSensitive = false
        private var spokenText: String?
        private var spokenRange: NSRange?

        var liveFindCount: Int { findMatches.count }

        // The active match's vertical center as a fraction of the laid-out
        // height, so the transcript can scroll the exact line into view.
        func activeMatchFraction() -> CGFloat? {
            var result: CGFloat? = nil
            if let i = activeIndex, i >= 0, i < findMatches.count,
               let lm = layoutManager, let tc = textContainer,
               NSMaxRange(findMatches[i]) <= (textStorage?.length ?? 0) {
                let gr = lm.glyphRange(forCharacterRange: findMatches[i],
                                       actualCharacterRange: nil)
                let rect = lm.boundingRect(forGlyphRange: gr, in: tc)
                let h = lm.usedRect(for: tc).height
                if h > 0 { result = min(1, max(0, rect.midY / h)) }
            }
            return result
        }
        private var findTint: NSColor {
            NSColor(srgbRed: 1.0, green: 0.84, blue: 0.2, alpha: 0.35)
        }
        private var activeTint: NSColor {
            NSColor(srgbRed: 1.0, green: 0.6, blue: 0.0, alpha: 0.6)
        }
        // Cool against find's warm pair, so the two never read as the same
        // thing when a search happens to be open while a reply is spoken.
        private var spokenTint: NSColor {
            NSColor(srgbRed: 0.25, green: 0.55, blue: 1.0, alpha: 0.28)
        }

        // Highlight EVERY match; the controller activates one later. No
        // selection here, so the other bubbles do not each grab a selection.
        func findAll(_ query: String, caseSensitive: Bool) -> Int {
            findQuery = query
            findCaseSensitive = caseSensitive
            findMatches = markdownFindRanges(in: string, query: query,
                                             caseSensitive: caseSensitive)
            activeIndex = nil
            highlightAll()
            return findMatches.count
        }

        func setActive(_ localIndex: Int?) {
            activeIndex = localIndex
            highlightAll()
            let len = textStorage?.length ?? 0
            if let i = localIndex, i >= 0, i < findMatches.count,
               NSMaxRange(findMatches[i]) <= len {
                setSelectedRange(findMatches[i])
                scrollRangeToVisible(findMatches[i])
            } else {
                setSelectedRange(NSRange(location: 0, length: 0))
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
                    in: string, query: findQuery,
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
            let ns = string as NSString
            var found: NSRange? = nil
            if let text = spokenText, !text.isEmpty {
                let r = ns.range(of: text)
                if r.location != NSNotFound { found = r }
            }
            spokenRange = found
        }

        // Temporary attributes rather than real backgrounds, so code and table
        // tints survive a find and applyIncremental's diff is undisturbed.
        private func highlightAll() {
            let matches = findMatches
            let active = activeIndex
            let spoken = spokenRange
            if let lm = layoutManager, let ts = textStorage {
                let full = NSRange(location: 0, length: ts.length)
                lm.removeTemporaryAttribute(.backgroundColor,
                                            forCharacterRange: full)
                if let r = spoken, NSMaxRange(r) <= ts.length {
                    lm.setTemporaryAttributes([.backgroundColor: spokenTint],
                                              forCharacterRange: r)
                }
                for (i, r) in matches.enumerated()
                where NSMaxRange(r) <= ts.length {
                    lm.setTemporaryAttributes(
                        [.backgroundColor: i == active
                            ? activeTint : findTint],
                        forCharacterRange: r)
                }
            }
        }

        override var intrinsicContentSize: NSSize {
            var result = super.intrinsicContentSize
            if let lm = layoutManager, let tc = textContainer {
                lm.ensureLayout(for: tc)
                let r = lm.usedRect(for: tc)
                let inset = textContainerInset
                let w = nowrap ? r.width + inset.width * 2
                               : NSView.noIntrinsicMetric
                result = NSSize(width: w, height: r.height + inset.height * 2)
            }
            return result
        }

        // A display is a layout, not characters: a rich target gets it as a
        // vector PDF, a plain one gets the TeX it was written from.
        override var writablePasteboardTypes: [NSPasteboard.PasteboardType] {
            var types: [NSPasteboard.PasteboardType] = [.rtfd, .rtf, .string]
            for one in super.writablePasteboardTypes where
                !types.contains(one) {
                types.append(one)
            }
            return types
        }

        // MEASURED: NSTextView does not route copy: through writeSelection, so
        // the flavours are written here, where the command lands.
        override func copy(_ sender: Any?) {
            let picked = selectedRange()
            let store = picked.length > 0 ? textStorage : nil
            if let store {
                let slice = store.attributedSubstring(from: picked)
                let board = NSPasteboard.general
                board.clearContents()
                board.declareTypes([.rtfd, .rtf, .string], owner: nil)
                let rich = Self.illustrated(slice, dark: self.isDark)
                if let data = rich.rtfd(
                    from: NSRange(location: 0, length: rich.length),
                    documentAttributes: [:]) {
                    board.setData(data, forType: .rtfd)
                }
                let text = Self.plain(slice)
                board.setString(text, forType: .string)
                let flat = NSAttributedString(string: text)
                if let data = flat.rtf(
                    from: NSRange(location: 0, length: flat.length),
                    documentAttributes: [:]) {
                    board.setData(data, forType: .rtf)
                }
            } else {
                super.copy(sender)
            }
        }

        override func writeSelection(to pboard: NSPasteboard,
                                     type: NSPasteboard.PasteboardType)
            -> Bool {
            let picked = selectedRange()
            var written = false
            if picked.length > 0, let ts = textStorage {
                let slice = ts.attributedSubstring(from: picked)
                if type == .string {
                    written = pboard.setString(Self.plain(slice), forType: type)
                } else if type == .rtfd {
                    let rich = Self.illustrated(slice, dark: self.isDark)
                    let full = NSRange(location: 0, length: rich.length)
                    if let data = rich.rtfd(from: full,
                                            documentAttributes: [:]) {
                        written = pboard.setData(data, forType: type)
                    }
                } else if type == .rtf {
                    // MEASURED: AppKit's plain-RTF writer embeds no picture for
                    // an image attachment -- 324 bytes and no \pict.
                    let text = NSAttributedString(string: Self.plain(slice))
                    let full = NSRange(location: 0, length: text.length)
                    if let data = text.rtf(from: full,
                                           documentAttributes: [:]) {
                        written = pboard.setData(data, forType: type)
                    }
                }
            }
            if !written {
                written = super.writeSelection(to: pboard, type: type)
            }
            return written
        }

        // The TeX a display was written from, in place of the object
        // replacement character that stands for it.
        private static func plain(_ slice: NSAttributedString) -> String {
            let m = NSMutableAttributedString(attributedString: slice)
            let full = NSRange(location: 0, length: m.length)
            for range in Self.attachments(in: m, full).reversed() {
                let tex = m.attribute(atomicCopyKey, at: range.location,
                                      effectiveRange: nil) as? String
                m.replaceCharacters(in: range, with: tex ?? "")
            }
            return m.string
        }

        // The same selection with every formula swapped for a picture of
        // itself, which is the only form a foreign document can render.
        private var isDark: Bool {
            effectiveAppearance
                .bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        }

        // The VIEW's appearance decides the theme, never the process: a theme
        // forced in the app is invisible to the system appearance.
        private static func illustrated(_ slice: NSAttributedString,
                                        dark: Bool)
            -> NSAttributedString {
            let m = NSMutableAttributedString(attributedString: slice)
            let full = NSRange(location: 0, length: m.length)
            for range in Self.attachments(in: m, full).reversed() {
                let cell = (m.attribute(.attachment, at: range.location,
                                        effectiveRange: nil)
                            as? NSTextAttachment)?.attachmentCell
                if let math = cell as? MathAttachmentCell,
                   let pdf = math.pdf(dark: dark),
                   let image = NSImage(data: pdf) {
                    let shown = NSTextAttachment()
                    shown.image = image
                    m.replaceCharacters(
                        in: range,
                        with: NSAttributedString(attachment: shown))
                }
            }
            return m
        }

        private static func attachments(in m: NSAttributedString,
                                        _ full: NSRange) -> [NSRange] {
            var found: [NSRange] = []
            m.enumerateAttribute(.attachment, in: full,
                                 options: []) { value, range, _ in
                if value != nil { found.append(range) }
            }
            return found
        }

        override func layout() {
            super.layout()
            if bounds.size != lastBounds {
                lastBounds = bounds.size
                invalidateIntrinsicContentSize()
            }
            rebuildCopyOverlays()
        }

        // A corner Copy button per atomic block, reused across layouts by
        // atomic id so a copy checkmark survives a streaming reflow.
        func rebuildCopyOverlays() {
            copyRects = []
            var live: Set<String> = []
            if let lm = layoutManager, let tc = textContainer,
               let ts = textStorage {
                lm.ensureLayout(for: tc)
                let origin = textContainerOrigin
                ts.enumerateAttribute(
                    atomicIdKey,
                    in: NSRange(location: 0, length: ts.length),
                    options: []) { value, range, _ in
                    let id = value as? String
                    let hasCopy = id != nil && ts.attribute(
                        atomicCopyKey, at: range.location,
                        effectiveRange: nil) != nil
                    let kind = id == nil ? nil : ts.attribute(
                        atomicKindKey, at: range.location,
                        effectiveRange: nil) as? String
                    if let id, hasCopy {
                        let gr = lm.glyphRange(forCharacterRange: range,
                                               actualCharacterRange: nil)
                        let block = lm.boundingRect(forGlyphRange: gr, in: tc)
                        // Centred on the FIRST line fragment, not the block
                        // top, or it reads as sitting on the baseline.
                        let line = lm.lineFragmentUsedRect(
                            forGlyphAt: gr.location, effectiveRange: nil)
                        // A display is CENTRED, so its button goes out to
                        // the margin rather than onto the formula.
                        let right = kind == AtomicKind.math.rawValue
                            ? lm.lineFragmentRect(forGlyphAt: gr.location,
                                                  effectiveRange: nil).maxX
                            : block.maxX
                        let x = right + origin.x - copyButtonGutter
                        let y = line.minY + origin.y + (line.height - 22) / 2
                        let frame = NSRect(x: x, y: y, width: 22, height: 22)
                        let btn = copyButtons[id] ?? makeCopyButton(id)
                        btn.frame = frame
                        copyButtons[id] = btn
                        live.insert(id)
                        copyRects.append((id, frame))
                    }
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
            btn.autoresizingMask = [.minXMargin]
            addSubview(btn)
            return btn
        }

        // A click landing in a block's corner button copies that block; a drag
        // (selection) leaves a non-empty selection, so it is never a copy.
        override func mouseDown(with event: NSEvent) {
            let p = convert(event.locationInWindow, from: nil)
            let hit = copyRects.first { rc in NSPointInRect(p, rc.rect) }?.id
            super.mouseDown(with: event)
            if let hit, selectedRange().length == 0 { copyBlock(hit) }
        }

        // The block's CURRENT source by atomic id: ranges shift as tokens
        // stream, the id is stable.
        private func copyBlock(_ id: String) {
            var copy: String? = nil
            if let ts = textStorage {
                ts.enumerateAttribute(
                    atomicIdKey,
                    in: NSRange(location: 0, length: ts.length),
                    options: []) { value, range, stop in
                    if value as? String == id {
                        copy = ts.attribute(atomicCopyKey, at: range.location,
                                            effectiveRange: nil) as? String
                        stop.pointee = true
                    }
                }
            }
            if let copy {
                platformSetClipboardString(copy)
                copyButtons[id]?.flashDone()
            }
        }
    }
}

// A small Copy button overlaid at a code / table block's corner.
private final class CopyRunButton: NSButton {
    var atomicId = ""

    init() {
        super.init(frame: .zero)
        isBordered = false
        bezelStyle = .regularSquare
        imagePosition = .imageOnly
        imageScaling = .scaleProportionallyDown
        contentTintColor = .secondaryLabelColor
        toolTip = "Copy"
        setSymbol("doc.on.doc")
    }

    required init?(coder: NSCoder) { nil }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    private func setSymbol(_ name: String) {
        image = NSImage(systemSymbolName: name,
                        accessibilityDescription: "Copy")
    }

    func flashDone() {
        setSymbol("checkmark")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            [weak self] in self?.setSymbol("doc.on.doc")
        }
    }
}
#endif
