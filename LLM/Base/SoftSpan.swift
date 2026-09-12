import Foundation

// `ids` is a whole BLOCK, not a count: a video repeats the bracket per frame
// with a stamp, and N follows the aspect ratio, known only after the tower.
public struct SoftSpan: Sendable {
    public let placeholder: Int32
    public let ids: [Int32]
    public let features: [Float]
    public let grid: (h: Int, w: Int)?
    public let wrap: (begin: Int32, end: Int32)?

    public init(placeholder: Int32, ids: [Int32], features: [Float],
                grid: (h: Int, w: Int)? = nil,
                wrap: (begin: Int32, end: Int32)? = nil) {
        self.placeholder = placeholder
        self.ids = ids
        self.features = features
        self.grid = grid
        self.wrap = wrap
    }

    public var rows: Int {
        ids.filter { id in id == placeholder }.count
    }

    public static func bracket(begin: Int32, placeholder: Int32,
                               end: Int32, count: Int) -> [Int32] {
        var out = [begin]
        out.append(contentsOf: [Int32](repeating: placeholder, count: count))
        out.append(end)
        return out
    }

    public static func bracketed(begin: Int32, placeholder: Int32,
                                 end: Int32, count: Int,
                                 features: [Float]) -> SoftSpan {
        SoftSpan(placeholder: placeholder,
                 ids: bracket(begin: begin, placeholder: placeholder,
                              end: end, count: count),
                 features: features)
    }
}

public final class SoftFeed {
    private var rows: [Int32: [Float]] = [:]
    private var width: [Int32: Int] = [:]
    private var taken: [Int32: Int] = [:]

    public init(_ spans: [SoftSpan]) {
        for span in spans {
            let n = span.rows
            precondition(n > 0 && span.features.count % n == 0,
                         "span carries \(span.features.count) floats for "
                         + "\(n) placeholder positions")
            rows[span.placeholder, default: []]
                .append(contentsOf: span.features)
            width[span.placeholder] = span.features.count / n
        }
    }

    public func row(_ id: Int32) -> [Float]? {
        var out: [Float]? = nil
        if let e = width[id], let all = rows[id] {
            let at = (taken[id] ?? 0) * e
            if at + e <= all.count {
                out = Array(all[at ..< (at + e)])
                taken[id] = (taken[id] ?? 0) + 1
            }
        }
        return out
    }
}
