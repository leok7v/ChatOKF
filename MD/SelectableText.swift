import SwiftUI

// Atomic units mark spans (a code block, a table, an image) that selection
// should snap around instead of cutting through, in the single-surface view.
enum AtomicKind: String {
    case code, table, image, math
}

// Where the copy overlay sets a button, measured in from the right edge of the
// block it belongs to.
let copyButtonGutter: CGFloat = 26

let atomicKindKey = NSAttributedString.Key("MD.atomic.kind")
let atomicIdKey = NSAttributedString.Key("MD.atomic.id")
// The block's SOURCE text for the in-block copy button, so Copy yields
// the original markdown rather than the flattened render.
let atomicCopyKey = NSAttributedString.Key("MD.atomic.copy")

// A read-only, self-sizing, selectable native text view.
struct SelectableText: View {

    let attributed: AttributedString?
    let ns: NSAttributedString?
    let font: PlatformFont
    let nowrap: Bool
    let selectable: Bool
    let bold: Bool
    let secondary: Bool
    // scrolls: the whole-document surface fills its frame and scrolls; a
    // per-block chat surface self-sizes so the transcript owns scroll.
    let scrolls: Bool
    let find: MarkdownFindController?
    // The message id this surface registers under, so Find spans every bubble
    // of the transcript in order. nil = not a Find target.
    let findId: UUID?
    // The sentence being read aloud right now, tinted where it appears.
    let speaking: String?

    init(_ attributed: AttributedString, font: PlatformFont,
         nowrap: Bool = false, selectable: Bool = true,
         bold: Bool = false, secondary: Bool = false,
         scrolls: Bool = false, find: MarkdownFindController? = nil,
         findId: UUID? = nil, speaking: String? = nil) {
        self.attributed = attributed
        self.ns = nil
        self.font = font
        self.nowrap = nowrap
        self.selectable = selectable
        self.bold = bold
        self.secondary = secondary
        self.scrolls = scrolls
        self.find = find
        self.findId = findId
        self.speaking = speaking
    }

    init(ns: NSAttributedString, font: PlatformFont,
         nowrap: Bool = false, selectable: Bool = true,
         bold: Bool = false, secondary: Bool = false,
         scrolls: Bool = false, find: MarkdownFindController? = nil,
         findId: UUID? = nil, speaking: String? = nil) {
        self.attributed = nil
        self.ns = ns
        self.font = font
        self.nowrap = nowrap
        self.selectable = selectable
        self.bold = bold
        self.secondary = secondary
        self.scrolls = scrolls
        self.find = find
        self.findId = findId
        self.speaking = speaking
    }

    var body: some View {
        NativeText(attributed: attributed, ns: ns, font: font,
                   nowrap: nowrap, selectable: selectable,
                   bold: bold, secondary: secondary,
                   scrolls: scrolls, find: find, findId: findId,
                   speaking: speaking)
            .fixedSize(horizontal: nowrap, vertical: !scrolls)
    }
}

// resolved() lowers whichever source was given into a fully-attributed
// string: run fonts merged onto the base, absent colours defaulted.
struct NativeText {

    let attributed: AttributedString?
    let ns: NSAttributedString?
    let font: PlatformFont
    let nowrap: Bool
    let selectable: Bool
    let bold: Bool
    let secondary: Bool
    let scrolls: Bool
    let find: MarkdownFindController?
    let findId: UUID?
    let speaking: String?

    func resolved() -> NSAttributedString {
        let m: NSMutableAttributedString
        if let ns {
            m = NSMutableAttributedString(attributedString: ns)
        } else if let attributed {
            m = NSMutableAttributedString(
                attributedString: NSAttributedString(attributed))
        } else {
            m = NSMutableAttributedString(string: "")
        }
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(.font, in: full, options: []) { value, r, _ in
            let merged: PlatformFont
            if let f = value as? PlatformFont {
                merged = platformMergeFontTraits(of: f, into: font,
                                                 additionalBold: bold)
            } else {
                merged = bold ? boldFont(of: font) : font
            }
            m.addAttribute(.font, value: merged, range: r)
        }
        let fallback = secondary ? platformSecondaryColor
                                 : platformDefaultTextColor
        m.enumerateAttribute(.foregroundColor, in: full,
                             options: []) { value, r, _ in
            if value == nil {
                m.addAttribute(.foregroundColor, value: fallback, range: r)
            }
        }
        return m
    }
}

// Replaces ONLY the span that changed, so a streaming re-render re-lays
// out O(delta) and a selection outside the edit survives.
func applyIncremental(_ storage: NSMutableAttributedString,
                      _ next: NSAttributedString) {
    let curLen = storage.length
    let nextLen = next.length
    let p = sharedAttributedPrefix(storage, next)
    let s = sharedAttributedSuffix(storage, next, after: p)
    storage.replaceCharacters(
        in: NSRange(location: p, length: curLen - p - s),
        with: next.attributedSubstring(
            from: NSRange(location: p, length: nextLen - p - s)))
}

// Steps by attribute run so the dictionary compare is per-run; `scanning`
// is the loop's termination condition, not a status flag.
private func sharedAttributedPrefix(_ a: NSAttributedString,
                                    _ b: NSAttributedString) -> Int {
    let sa = a.string as NSString
    let sb = b.string as NSString
    let n = min(a.length, b.length)
    var i = 0
    var scanning = n > 0
    while scanning {
        if sa.character(at: i) != sb.character(at: i) {
            scanning = false
        } else {
            var ra = NSRange(location: 0, length: 0)
            var rb = NSRange(location: 0, length: 0)
            let da = a.attributes(at: i, effectiveRange: &ra) as NSDictionary
            let db = b.attributes(at: i, effectiveRange: &rb) as NSDictionary
            if !da.isEqual(db) {
                scanning = false
            } else {
                let end = min(NSMaxRange(ra), NSMaxRange(rb), n)
                var j = i + 1
                while j < end && sa.character(at: j) == sb.character(at: j) {
                    j += 1
                }
                i = j
                scanning = j >= end && i < n
            }
        }
    }
    return i
}

private func sharedAttributedSuffix(_ a: NSAttributedString,
                                    _ b: NSAttributedString,
                                    after prefix: Int) -> Int {
    let sa = a.string as NSString
    let sb = b.string as NSString
    let cap = min(a.length, b.length) - prefix
    var s = 0
    var scanning = cap > 0
    while scanning {
        let ia = a.length - 1 - s
        let ib = b.length - 1 - s
        if sa.character(at: ia) != sb.character(at: ib) {
            scanning = false
        } else {
            let da = a.attributes(at: ia, effectiveRange: nil) as NSDictionary
            let db = b.attributes(at: ib, effectiveRange: nil) as NSDictionary
            if da.isEqual(db) {
                s += 1
                scanning = s < cap
            } else {
                scanning = false
            }
        }
    }
    return s
}
