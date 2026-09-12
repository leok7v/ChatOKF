import Foundation
import CoreText
import CoreGraphics

// Draws one Markdown table into the PDF renderer with per-column widths and
// alignment, header shade, zebra rows, and a bottom rule per row.
final class PDFTable {

    let r: PDFRenderer
    let headers: [String]
    let rows: [[String]]
    let aligns: [Markdown.Alignment]
    let cols: Int
    let rowPad: CGFloat = 4

    init(renderer: PDFRenderer, headers: [String], rows: [[String]],
         aligns: [Markdown.Alignment], cols: Int) {
        self.r = renderer
        self.headers = headers
        self.rows = rows
        self.aligns = aligns
        self.cols = cols
    }

    func draw() {
        let widths = columnWidths()
        if !headers.isEmpty {
            drawRow(headers, bold: true, shade: r.headerShadeColor,
                    widths: widths)
        }
        for (idx, row) in rows.enumerated() {
            drawRow(row, bold: false,
                    shade: idx % 2 == 1 ? r.rowShadeColor : nil,
                    widths: widths)
        }
    }

    private func columnWidths() -> [CGFloat] {
        let mins: [CGFloat] = (0..<cols).map { c in
            var widest: CGFloat = 0
            if c < headers.count {
                let w = renderedWidth(headers[c], bold: true)
                if w > widest { widest = w }
            }
            for row in rows where c < row.count {
                let w = renderedWidth(row[c], bold: false)
                if w > widest { widest = w }
            }
            return widest + 2 * rowPad
        }
        var widths = TableMetrics.pointWidths(headers: headers, rows: rows,
                                              available: r.contentWidth,
                                              minimums: mins)
        let total = widths.reduce(0, +)
        if total > r.contentWidth, total > 0 {
            let scale = r.contentWidth / total
            widths = widths.map { v in v * scale }
        }
        return widths
    }

    private func drawRow(_ cells: [String], bold: Bool, shade: CGColor?,
                         widths: [CGFloat]) {
        var rowH: CGFloat = r.bodySize * 1.3
        for c in 0..<cols {
            let txt = c < cells.count ? cells[c] : ""
            let h = cellHeight(txt, bold: bold, width: widths[c] - 2 * rowPad)
            if h > rowH { rowH = h }
        }
        r.ensureSpace(rowH + rowPad * 2)
        let savedY = r.y
        if let shade {
            r.ctx.setFillColor(shade)
            r.ctx.fill(CGRect(x: r.contentLeft, y: savedY - rowH - rowPad,
                              width: r.contentWidth,
                              height: rowH + 2 * rowPad))
        }
        var maxUsed: CGFloat = 0
        var x = r.contentLeft
        for c in 0..<cols {
            let txt = c < cells.count ? cells[c] : ""
            let used = drawCell(txt, bold: bold, x: x + rowPad, topY: savedY,
                                width: widths[c] - 2 * rowPad, col: c)
            if used > maxUsed { maxUsed = used }
            x += widths[c]
        }
        r.y = savedY - maxUsed - rowPad
        r.ctx.setStrokeColor(r.secondaryColor)
        r.ctx.setLineWidth(0.5)
        r.ctx.move(to: CGPoint(x: r.contentLeft, y: r.y))
        r.ctx.addLine(to: CGPoint(x: r.contentRight, y: r.y))
        r.ctx.strokePath()
        r.y -= rowPad
    }

    private func drawCell(_ txt: String, bold: Bool, x: CGFloat, topY: CGFloat,
                          width: CGFloat, col: Int) -> CGFloat {
        let inner = cellAttributed(txt, bold: bold, col: col)
        let fs = CTFramesetterCreateWithAttributedString(inner)
        let rect = CGRect(x: x, y: r.contentBottom, width: width,
                          height: topY - r.contentBottom)
        let path = CGPath(rect: rect, transform: nil)
        let frame = CTFramesetterCreateFrame(
            fs, CFRange(location: 0, length: 0), path, nil)
        let used = r.lineHeightUsed(frame: frame, in: rect)
        CTFrameDraw(frame, r.ctx)
        return used
    }

    private func renderedWidth(_ text: String, bold: Bool) -> CGFloat {
        let line = CTLineCreateWithAttributedString(
            cellAttributed(text, bold: bold, col: 0))
        return CTLineGetBoundsWithOptions(line, []).width
    }

    private func cellHeight(_ txt: String, bold: Bool,
                            width: CGFloat) -> CGFloat {
        let inner = cellAttributed(txt, bold: bold, col: 0)
        let fs = CTFramesetterCreateWithAttributedString(inner)
        let rect = CGRect(x: 0, y: 0, width: width, height: r.pageSize.height)
        let path = CGPath(rect: rect, transform: nil)
        let frame = CTFramesetterCreateFrame(
            fs, CFRange(location: 0, length: 0), path, nil)
        return r.lineHeightUsed(frame: frame, in: rect)
    }

    private func cellAttributed(_ text: String, bold: Bool,
                                col: Int) -> NSAttributedString {
        let parsed = Markdown.parse(text)
        var attr = AttributedString(text)
        if let first = parsed.items.first,
           case .paragraph(let a) = first.block {
            attr = a
        }
        let base = bold ? r.bodyFontBold() : r.bodyFont()
        let baseSize = CTFontGetSize(base)
        let para = NSMutableParagraphStyle()
        para.alignment = alignment(col)
        let m = NSMutableAttributedString()
        for run in attr.runs {
            let intent = run.inlinePresentationIntent ?? []
            let font = runFont(intent: intent, size: baseSize, bold: bold)
            var attrs: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: r.textColor,
                .paragraphStyle: para,
            ]
            if intent.contains(.code) {
                attrs[.backgroundColor] = r.codeBgColor
            }
            if intent.contains(.strikethrough) {
                attrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
                attrs[.strikethroughColor] = r.textColor
            }
            let segment = protectNumerics(String(attr[run.range].characters))
            m.append(NSAttributedString(string: segment, attributes: attrs))
        }
        return m
    }

    private func runFont(intent: InlinePresentationIntent, size: CGFloat,
                         bold: Bool) -> CTFont {
        let result: CTFont
        if intent.contains(.code) {
            result = CTFontCreateWithName("Menlo" as CFString, size, nil)
        } else {
            let wantBold = bold || intent.contains(.stronglyEmphasized)
            let wantItalic = intent.contains(.emphasized)
            var traits: CTFontSymbolicTraits = []
            if wantBold { traits.insert(.traitBold) }
            if wantItalic { traits.insert(.traitItalic) }
            let plain = CTFontCreateUIFontForLanguage(.system, size, nil)
                ?? CTFontCreateWithName("Helvetica" as CFString, size, nil)
            result = traits.isEmpty ? plain
                : (CTFontCreateCopyWithSymbolicTraits(plain, size, nil,
                    traits, traits) ?? plain)
        }
        return result
    }

    private func alignment(_ col: Int) -> NSTextAlignment {
        let a = col < aligns.count ? aligns[col] : .none
        let result: NSTextAlignment
        switch a {
            case .center: result = .center
            case .right: result = .right
            default: result = .left
        }
        return result
    }

    // Word-join '.' and ',' between digits (U+2060) so CoreText cannot split
    // a number like "70.1" or "1,234.56" across lines in a narrow cell.
    private func protectNumerics(_ s: String) -> String {
        var result = s
        if let re = try? NSRegularExpression(pattern: #"\d[\d.,]*\d"#) {
            let ns = s as NSString
            let full = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: s, range: full)
            if !matches.isEmpty {
                let m = NSMutableString(string: s)
                for match in matches.reversed() {
                    let token = ns.substring(with: match.range)
                    let joined = token
                        .replacingOccurrences(of: ".",
                            with: "\u{2060}.\u{2060}")
                        .replacingOccurrences(of: ",",
                            with: "\u{2060},\u{2060}")
                    m.replaceCharacters(in: match.range, with: joined)
                }
                result = m as String
            }
        }
        return result
    }
}
