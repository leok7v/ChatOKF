import Foundation

// Shared column-width math for every table renderer. Widths use a
// sqrt-damped character count so a wide column does not starve narrow ones.
enum TableMetrics {

    static func columnCount(headers: [String], rows: [[String]]) -> Int {
        var n = headers.count
        for row in rows where row.count > n { n = row.count }
        return n
    }

    static func charWidths(headers: [String], rows: [[String]]) -> [Int] {
        let n = columnCount(headers: headers, rows: rows)
        var widths = [Int](repeating: 1, count: n)
        var all = rows
        all.insert(headers, at: 0)
        for cells in all {
            for (i, cell) in cells.enumerated() where i < n {
                if cell.count > widths[i] { widths[i] = cell.count }
            }
        }
        return widths
    }

    static func pointWidths(headers: [String], rows: [[String]],
                            available: CGFloat,
                            minimums: [CGFloat]? = nil) -> [CGFloat] {
        let n = columnCount(headers: headers, rows: rows)
        var result = [CGFloat](repeating: 0, count: n)
        let chars = charWidths(headers: headers, rows: rows)
        let weights = chars.map { c in sqrt(CGFloat(c)) }
        let sum = weights.reduce(0, +)
        if available > 0, n > 0 {
            if let mins = minimums, mins.count == n {
                result = distribute(available: available, weights: weights,
                                    sum: sum, minimums: mins)
            } else if sum > 0 {
                result = weights.map { wt in available * wt / sum }
            }
        }
        return result
    }

    static func columnLayout(headers: [String], rows: [[String]],
                             natural: [CGFloat], minimums: [CGFloat],
                             available: CGFloat)
        -> (widths: [CGFloat], wrap: Bool) {
        var result = (widths: natural, wrap: false)
        if natural.reduce(0, +) > available {
            let shared = pointWidths(headers: headers, rows: rows,
                                     available: available, minimums: minimums)
            result = (capped(shared, natural), true)
        }
        return result
    }

    static func scrollingLayout(headers: [String], rows: [[String]],
                                natural: [CGFloat], minimums: [CGFloat],
                                available: CGFloat)
        -> (widths: [CGFloat], wrap: Bool, scrolls: Bool) {
        var result: (widths: [CGFloat], wrap: Bool, scrolls: Bool)
        if minimums.reduce(0, +) > available {
            result = (minimums, true, true)
        } else {
            let fit = columnLayout(headers: headers, rows: rows,
                                   natural: natural, minimums: minimums,
                                   available: available)
            result = (fit.widths, fit.wrap, false)
        }
        return result
    }

    private static func capped(_ widths: [CGFloat],
                               _ natural: [CGFloat]) -> [CGFloat] {
        var out = widths
        let want = (0 ..< out.count).map { c in max(natural[c] - out[c], 0) }
        let short = want.reduce(0, +)
        var slack: CGFloat = 0
        for c in 0 ..< out.count where out[c] > natural[c] {
            slack += out[c] - natural[c]
            out[c] = natural[c]
        }
        let give = min(slack, short)
        for c in 0 ..< out.count where short > 0 {
            out[c] += give * want[c] / short
        }
        return out
    }

    private static func distribute(available: CGFloat, weights: [CGFloat],
                                   sum: CGFloat,
                                   minimums: [CGFloat]) -> [CGFloat] {
        let minSum = minimums.reduce(0, +)
        let result: [CGFloat]
        if minSum >= available, minSum > 0 {
            result = fairWidths(minimums: minimums, weights: weights,
                                available: available)
        } else if sum > 0 {
            let remainder = available - minSum
            result = (0..<minimums.count).map { i in
                minimums[i] + remainder * weights[i] / sum
            }
        } else {
            result = minimums
        }
        return result
    }

    private static func fairWidths(minimums: [CGFloat], weights: [CGFloat],
                                   available: CGFloat) -> [CGFloat] {
        var result = [CGFloat](repeating: 0, count: minimums.count)
        var open = Array(0 ..< minimums.count)
        var remaining = available
        var granted: [Int] = []
        repeat {
            let pending = open.reduce(0) { sum, c in sum + weights[c] }
            let room = remaining
            granted = open.filter { c in
                pending > 0 && minimums[c] <= room * weights[c] / pending
            }
            for c in granted {
                result[c] = minimums[c]
                remaining -= minimums[c]
            }
            open.removeAll { c in granted.contains(c) }
        } while !granted.isEmpty
        let short = open.reduce(0) { sum, c in sum + weights[c] }
        for c in open {
            result[c] = short > 0 ? remaining * weights[c] / short
                                  : remaining / CGFloat(open.count)
        }
        return result
    }

    static func normalize(_ s: String) -> String {
        let trimmed = s.trimmingCharacters(in: .whitespaces)
        var out = ""
        var inSpace = false
        for ch in trimmed {
            if ch.isWhitespace {
                if !inSpace { out.append(" ") }
                inSpace = true
            } else {
                out.append(ch)
                inSpace = false
            }
        }
        return out
    }

    static func longestWord(headers: [String], rows: [[String]],
                            col: Int) -> String {
        var best = ""
        var column: [String] = []
        if col < headers.count { column.append(headers[col]) }
        for row in rows where col < row.count { column.append(row[col]) }
        for cell in column {
            for word in cell.split(separator: " ",
                                   omittingEmptySubsequences: true) {
                if word.count > best.count { best = String(word) }
            }
        }
        return best
    }

    static func serializeMonospaced(headers: [String],
                                    rows: [[String]]) -> String {
        let n = columnCount(headers: headers, rows: rows)
        let widths = charWidths(headers: headers, rows: rows)
        var lines: [String] = []
        if !headers.isEmpty {
            lines.append(monoRow(headers, n: n, widths: widths))
            let dashes = (0..<n).map { i in
                String(repeating: "-", count: widths[i])
            }
            lines.append("| " + dashes.joined(separator: " | ") + " |")
        }
        for row in rows { lines.append(monoRow(row, n: n, widths: widths)) }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func monoRow(_ cells: [String], n: Int,
                                widths: [Int]) -> String {
        var parts: [String] = []
        for i in 0..<n {
            let cell = i < cells.count ? cells[i] : ""
            let fill = max(0, widths[i] - cell.count)
            parts.append(cell + String(repeating: " ", count: fill))
        }
        return "| " + parts.joined(separator: " | ") + " |"
    }
}
