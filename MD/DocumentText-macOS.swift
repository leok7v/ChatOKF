#if os(macOS)
import AppKit

// A cell rather than an image, so the formula stays vector and picks up the
// text colour at DRAW time.
final class MathAttachmentCell: NSTextAttachmentCell {

    // Held apart from the layout because the sizing overrides are nonisolated
    // and the laid-out formula holds the shared math font.
    private nonisolated let extent: CGSize
    private nonisolated let descent: CGFloat
    private let layout: MathLayout
    private nonisolated static let inset: CGFloat = 4

    init(layout: MathLayout) {
        self.layout = layout
        self.extent = CGSize(width: layout.width + Self.inset * 2,
                             height: layout.height)
        self.descent = layout.descent
        super.init()
    }

    // Never archived: the attachment is built fresh from the markdown every
    // time the document is laid out.
    required init(coder: NSCoder) {
        fatalError("MathAttachmentCell is not decodable")
    }

    // Rendered on demand and kept, so a copy pays once and building a
    // document pays nothing.
    private var pdfData: Data?
    private var pdfDark = false

    // Testing hook: drop the cached page so the other theme can be asked
    // for without building a second document.
    func forget() { pdfData = nil }

    func pdf(dark: Bool) -> Data? {
        if pdfData == nil || pdfDark != dark {
            pdfData = mathPDF(layout, dark: dark)
            pdfDark = dark
        }
        return pdfData
    }

    override func cellSize() -> NSSize { extent }

    // The formula sits on the text baseline like a very tall glyph, so its
    // descent is what hangs below.
    override func cellBaselineOffset() -> NSPoint {
        NSPoint(x: 0, y: -descent)
    }

    // A formula has no line breaks to give: TextKit clips it, so it is scaled
    // to the width offered here.

    // Asked more than once per layout, so it answers from the width offered
    // and remembers nothing between calls.
    override func cellFrame(for textContainer: NSTextContainer,
                            proposedLineFragment lineFrag: NSRect,
                            glyphPosition position: NSPoint,
                            characterIndex charIndex: Int) -> NSRect {
        let fitted = DocumentText.mathFit(natural: extent,
                                          available: lineFrag.width)
        return NSRect(origin: .zero, size: fitted)
    }

    override func draw(withFrame cellFrame: NSRect, in controlView: NSView?) {
        if let ctx = NSGraphicsContext.current?.cgContext {
            // Scale read back off the frame cellFrame(for:) asked for, since
            // draw is never told the width.
            let scale = extent.width > 0 ? cellFrame.width / extent.width : 1
            ctx.saveGState()
            ctx.translateBy(x: cellFrame.minX, y: cellFrame.minY)
            ctx.scaleBy(x: scale, y: scale)
            layout.draw(in: ctx, at: CGPoint(x: Self.inset, y: 0),
                        color: NSColor.textColor.cgColor,
                        flipped: controlView?.isFlipped ?? true)
            ctx.restoreGState()
        }
    }
}

// An alpha mask that is opaque in the middle and fades to nothing within
// `feather` of every edge.
private func featherMask(_ size: CGSize, feather: CGFloat) -> CGImage? {
    let w = Int(size.width.rounded(.up))
    let h = Int(size.height.rounded(.up))
    var result: CGImage? = nil
    if w > 0, h > 0,
       let ctx = CGContext(data: nil, width: w, height: h,
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue) {
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        let steps = 24
        for step in 0...steps {
            let part = CGFloat(step) / CGFloat(steps)
            // Smaller AND brighter each step: the rim stays black, the inside
            // reaches white.
            let inset = feather * part
            let rect = CGRect(x: inset, y: inset,
                              width: size.width - inset * 2,
                              height: size.height - inset * 2)
            if rect.width > 0, rect.height > 0 {
                ctx.setFillColor(CGColor(gray: part, alpha: 1))
                ctx.fill(rect)
            }
        }
        result = ctx.makeImage()
    }
    return result
}

// A formula as a PDF page, for the pasteboard. Vector rather than a raster so
// it stays crisp wherever it lands and prints properly.
func mathPDF(_ layout: MathLayout, dark: Bool,
             padding: CGFloat = 14) -> Data? {
    let data = NSMutableData()
    var box = CGRect(x: 0, y: 0, width: layout.width + padding * 2,
                     height: layout.height + padding * 2)
    var result: Data? = nil
    // 0.07 rather than a lighter grey: a dark document is nearer black, and
    // a patch lighter than the page reads as a card.
    let paper: CGFloat = dark ? 0.07 : 0.98
    let ink: CGFloat = dark ? 0.95 : 0.05
    if let consumer = CGDataConsumer(data: data),
       let ctx = CGContext(consumer: consumer, mediaBox: &box, nil) {
        ctx.beginPDFPage(nil)
        // Paper under the ink, fading within the padding band: legible pasted
        // into the opposite theme, invisible pasted into the same one.
        if let mask = featherMask(box.size, feather: padding) {
            ctx.saveGState()
            ctx.clip(to: box, mask: mask)
            ctx.setFillColor(CGColor(gray: paper, alpha: 1))
            ctx.fill(box)
            ctx.restoreGState()
        }
        // `at` is the TOP-LEFT of the bounding box and draw subtracts the
        // ascent, so the top edge is padding + height in this y-up page.
        layout.draw(in: ctx, at: CGPoint(x: padding, y: padding + layout.height),
                    color: CGColor(gray: ink, alpha: 1))
        ctx.endPDFPage()
        ctx.closePDF()
        result = data as Data
    }
    return result
}

extension DocumentText {

    static func mathAttachment(_ layout: MathLayout) -> NSTextAttachment {
        let attachment = NSTextAttachment()
        attachment.attachmentCell = MathAttachmentCell(layout: layout)
        return attachment
    }

    // Horizontal cell padding, as a percentage of the table width so it can
    // be subtracted from the column-share budget in the same unit.
    private static var cellPad: CGFloat { 0.6 }

    private static func contentBudget(cols: Int) -> CGFloat {
        max(100 - CGFloat(cols) * cellPad * 2, 50)
    }

    // Solved against the SHARES the table will actually be built with, not
    // against the bare sum of the minimums.
    static func tableMinimumWidth(headers: [String], rows: [[String]],
                                  style: MarkdownStyle) -> CGFloat {
        var result: CGFloat = 0
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let mins = columnMinimums(headers: headers, rows: rows,
                                      cols: cols, style: style)
            let fractions = TableMetrics.pointWidths(
                headers: headers, rows: rows,
                available: contentBudget(cols: cols) / 100)
            for c in 0..<cols where c < fractions.count && fractions[c] > 0 {
                let need = mins[c] / fractions[c]
                if need > result { result = need }
            }
            result = ceil(result)
        }
        return result
    }

    static func table(headers: [String], rows: [[String]],
                      alignments: [Markdown.Alignment], style: MarkdownStyle,
                      images: [URL: PlatformImage],
                      width: CGFloat) -> NSAttributedString {
        let m = NSMutableAttributedString()
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let atomicId = UUID().uuidString
            let textTable = NSTextTable()
            textTable.numberOfColumns = cols
            // Automatic layout: a fixed cell that outgrows its width spills
            // over the next column instead of widening.
            textTable.layoutAlgorithm = .automaticLayoutAlgorithm
            // The shares leave room for the padding, or the table wants 100%
            // plus padding and the last columns are squeezed off.
            let shares = TableMetrics.pointWidths(
                headers: headers, rows: rows,
                available: contentBudget(cols: cols))
            var rowIdx = 0
            if !headers.isEmpty {
                m.append(tableRow(headers, table: textTable, rowIdx: rowIdx,
                                  cols: cols, shares: shares,
                                  alignments: alignments, bold: true,
                                  tint: platformWhite(0.5, alpha: 0.14),
                                  atomicId: atomicId, style: style,
                                  images: images))
                rowIdx += 1
            }
            for (idx, row) in rows.enumerated() {
                let tint: PlatformColor = idx % 2 == 1
                    ? platformWhite(0.5, alpha: 0.07) : platformClearColor
                m.append(tableRow(row, table: textTable, rowIdx: rowIdx,
                                  cols: cols, shares: shares,
                                  alignments: alignments, bold: false,
                                  tint: tint, atomicId: atomicId,
                                  style: style, images: images))
                rowIdx += 1
            }
            // One CONTIGUOUS atomic id and kind over the whole table, stamped
            // before the trailing newline, so the copy overlay sees one block.
            let content = NSRange(location: 0, length: m.length)
            m.addAttribute(atomicIdKey, value: atomicId, range: content)
            m.addAttribute(atomicKindKey,
                           value: AtomicKind.table.rawValue, range: content)
            m.addAttribute(atomicCopyKey,
                           value: TableMetrics.serializeMonospaced(
                               headers: headers, rows: rows),
                           range: content)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func tableRow(_ cells: [String], table: NSTextTable,
                                 rowIdx: Int, cols: Int, shares: [CGFloat],
                                 alignments: [Markdown.Alignment], bold: Bool,
                                 tint: PlatformColor, atomicId: String,
                                 style: MarkdownStyle,
                                 images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        let base = bold ? boldFont(of: bodyFont(style)) : bodyFont(style)
        for col in 0..<cols {
            let cellText = col < cells.count ? cells[col] : ""
            let block = NSTextTableBlock(table: table, startingRow: rowIdx,
                                         rowSpan: 1, startingColumn: col,
                                         columnSpan: 1)
            if col < shares.count {
                block.setValue(shares[col], type: .percentageValueType,
                               for: .width)
            }
            block.setWidth(cellPad, type: .percentageValueType,
                           for: .padding, edge: .minX)
            block.setWidth(cellPad, type: .percentageValueType,
                           for: .padding, edge: .maxX)
            block.setWidth(3, type: .absoluteValueType,
                           for: .padding, edge: .minY)
            block.setWidth(3, type: .absoluteValueType,
                           for: .padding, edge: .maxY)
            block.backgroundColor = tint
            let para = NSMutableParagraphStyle()
            // Only safe because the surface floors at tableMinimumWidth:
            // NSTextTable overlaps the cells of a row it cannot fit.
            para.lineBreakMode = .byWordWrapping
            para.textBlocks = [block]
            para.alignment = nsAlignment(col, alignments)
            let cellAttr = NSMutableAttributedString(
                attributedString: tableCell(cellText, base: base,
                                            style: style, images: images))
            if cellAttr.length == 0 {
                cellAttr.append(NSAttributedString(
                    string: "\u{00A0}",
                    attributes: [.font: base,
                                 .foregroundColor: platformDefaultTextColor]))
            }
            let full = NSRange(location: 0, length: cellAttr.length)
            cellAttr.addAttribute(.paragraphStyle, value: para, range: full)
            cellAttr.addAttribute(atomicKindKey,
                                  value: AtomicKind.table.rawValue,
                                  range: full)
            cellAttr.addAttribute(atomicIdKey, value: atomicId, range: full)
            m.append(cellAttr)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func nsAlignment(_ col: Int,
                                    _ aligns: [Markdown.Alignment])
        -> NSTextAlignment {
        let a = col < aligns.count ? aligns[col] : .none
        let result: NSTextAlignment
        switch a {
            case .center: result = .center
            case .right: result = .right
            default: result = .left
        }
        return result
    }
}
#endif
