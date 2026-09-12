import Foundation

// Find / Find Next across the transcript's PER-MESSAGE single-surface text
// views.
@MainActor
public final class MarkdownFindController {

    public struct Options: Sendable {
        public var caseSensitive: Bool
        public init(caseSensitive: Bool = false) {
            self.caseSensitive = caseSensitive
        }
    }

    private var targets: [UUID: any FindableTextView] = [:]
    public var order: [UUID] = []
    // Scroll the transcript to a match: the message id plus the match's
    // vertical position as a fraction of that message; nil centres.
    public var scrollTo: ((UUID, CGFloat?) -> Void)?

    public private(set) var matchCount = 0
    public private(set) var currentMatch = 0   // 1-based; 0 when no match

    private var slots: [(id: UUID, view: any FindableTextView, count: Int)] = []
    private var cursor = -1
    private var query = ""
    private var caseSensitive = false
    // The view holding the active-match selection, so stepping clears ONLY
    // its selection and a manual selection elsewhere survives.
    private weak var activeView: (any FindableTextView)?

    public init() {}

    // A registration during an active find re-highlights that view, so a
    // bubble scrolled into existence mid-search shows its matches.
    func register(_ id: UUID, _ view: any FindableTextView) {
        targets[id] = view
        if !query.isEmpty {
            _ = view.findAll(query, caseSensitive: caseSensitive)
        }
    }

    func unregister(_ id: UUID, _ view: any FindableTextView) {
        if targets[id] === view { targets[id] = nil }
    }

    @discardableResult
    public func find(_ q: String,
                     options: Options = Options()) -> Int {
        query = q
        caseSensitive = options.caseSensitive
        rebuildSlots()
        cursor = -1
        if matchCount > 0 { _ = step(forward: true) }
        return matchCount
    }

    @discardableResult
    public func findNext() -> Int { step(forward: true) }

    @discardableResult
    public func findPrevious() -> Int { step(forward: false) }

    public func clear() {
        for (_, v) in targets { v.clearFind() }
        query = ""
        slots = []
        cursor = -1
        matchCount = 0
        currentMatch = 0
        activeView = nil
    }

    // Re-highlight every view from scratch and total the counts. Only find()
    // uses this (it re-tints); navigation uses recountSlots (no re-tint).
    private func rebuildSlots() {
        slots = order.compactMap { id in
            targets[id].map { v in
                (id, v, v.findAll(query, caseSensitive: caseSensitive))
            }
        }
        matchCount = slots.reduce(0) { sum, s in sum + s.count }
    }

    // Recount from each view's CURRENT matches without re-tinting, so
    // navigation sees bubbles that streamed or registered after find().
    private func recountSlots() {
        slots = order.compactMap { id in
            targets[id].map { v in (id, v, v.liveFindCount) }
        }
        matchCount = slots.reduce(0) { sum, s in sum + s.count }
        if matchCount == 0 {
            cursor = -1
        } else if cursor >= matchCount {
            cursor = matchCount - 1
        }
    }

    private func step(forward: Bool) -> Int {
        recountSlots()
        var result = 0
        activeView?.setActive(nil)
        activeView = nil
        if matchCount > 0 {
            cursor = ((cursor + (forward ? 1 : -1)) % matchCount
                      + matchCount) % matchCount
            if let hit = locate(cursor) {
                hit.view.setActive(hit.local)
                activeView = hit.view
                scrollTo?(hit.id, hit.view.activeMatchFraction())
                currentMatch = cursor + 1
                result = currentMatch
            }
        } else {
            currentMatch = 0
        }
        return result
    }

    private func locate(_ global: Int)
        -> (id: UUID, view: any FindableTextView, local: Int)? {
        var g = global
        var result: (id: UUID, view: any FindableTextView, local: Int)? = nil
        var i = 0
        while result == nil && i < slots.count {
            if g < slots[i].count {
                result = (slots[i].id, slots[i].view, g)
            } else {
                g -= slots[i].count
            }
            i += 1
        }
        return result
    }
}

// Implemented by the platform text views (Bridges-iOS / Bridges-macOS).
@MainActor
protocol FindableTextView: AnyObject {
    func findAll(_ query: String, caseSensitive: Bool) -> Int
    func setActive(_ localIndex: Int?)
    func clearFind()
    // The sentence being READ ALOUD right now; matched literally, so a
    // reshaped segment does not tint rather than tinting the wrong one.
    func setSpoken(_ text: String?)
    // Current match count without recomputing, so the controller can re-total
    // after a view's matches change under streaming (reapplyFind).
    var liveFindCount: Int { get }
    // Vertical centre of the active match as a fraction of the view's
    // height, so the transcript scrolls the exact line into view.
    func activeMatchFraction() -> CGFloat?
}

// All (overlapping-free) ranges of `query` in `text`. Non-advancing matches are
// guarded so an empty or degenerate query terminates.
func markdownFindRanges(in text: String, query: String,
                        caseSensitive: Bool) -> [NSRange] {
    var result: [NSRange] = []
    if !query.isEmpty {
        let ns = text as NSString
        let opts: NSString.CompareOptions =
            caseSensitive ? [] : .caseInsensitive
        var start = 0
        var searching = true
        while searching {
            let scope = NSRange(location: start, length: ns.length - start)
            let r = ns.range(of: query, options: opts, range: scope)
            if r.location == NSNotFound {
                searching = false
            } else {
                result.append(r)
                start = r.location + max(r.length, 1)
                if start >= ns.length { searching = false }
            }
        }
    }
    return result
}
