#if os(iOS)
import UIKit

// A formula has no line breaks to give, so it is scaled to the width
// TextKit offers here rather than cut off at the right.
final class MathAttachment: NSTextAttachment {

    var natural: CGRect = .zero

    override func attachmentBounds(for textContainer: NSTextContainer?,
                                   proposedLineFragment lineFrag: CGRect,
                                   glyphPosition position: CGPoint,
                                   characterIndex charIndex: Int) -> CGRect {
        let fitted = DocumentText.mathFit(natural: natural.size,
                                         available: lineFrag.width)
        let scale = natural.width > 0 ? fitted.width / natural.width : 1
        return CGRect(origin: CGPoint(x: 0, y: natural.origin.y * scale),
                      size: fitted)
    }
}

extension DocumentText {

    private struct RasterKey: Hashable {
        let box: ObjectIdentifier
        let scale: CGFloat
    }

    private struct Raster {
        let layout: MathLayout
        let ink: CGColor
        let image: UIImage
    }

    private static var rasters: [RasterKey: Raster] = [:]
    private static let rasterCapacity = 32

    // UIKit has no attachment cell to draw through, so the formula is
    // rasterized with the ink current when the document was built.
    static func mathAttachment(_ layout: MathLayout) -> NSTextAttachment {
        let attachment = MathAttachment()
        attachment.image = raster(layout, scale: UIScreen.main.scale,
                                  ink: platformDefaultTextColor.cgColor)
        attachment.natural = CGRect(x: 0, y: -layout.descent,
                                    width: layout.width + 8,
                                    height: layout.height)
        attachment.bounds = attachment.natural
        return attachment
    }

    private static func raster(_ layout: MathLayout, scale: CGFloat,
                               ink: CGColor) -> UIImage? {
        let key = RasterKey(box: ObjectIdentifier(layout.box), scale: scale)
        var result: UIImage? = nil
        if let hit = rasters[key], hit.layout.box === layout.box,
           CFEqual(hit.ink, ink) {
            result = hit.image
        } else if let cg = layout.cgImage(scale: scale, padding: 4,
                                          background: nil, color: ink) {
            let image = UIImage(cgImage: cg, scale: scale, orientation: .up)
            if rasters.count >= rasterCapacity { rasters.removeAll() }
            rasters[key] = Raster(layout: layout, ink: ink, image: image)
            result = image
        }
        return result
    }

    private static var columnGap: CGFloat { 10 }

    static var tableUsesNaturals: Bool { true }

    static func tableMinimumWidth(_ cells: TableCells) -> CGFloat {
        var result: CGFloat = 0
        if cells.cols > 0 {
            result = ceil(cells.minimums.reduce(0, +))
                + CGFloat(cells.cols) * columnGap
        }
        return result
    }

    static func table(_ cells: TableCells,
                      alignments: [Markdown.Alignment], id: String,
                      style: MarkdownStyle,
                      width: CGFloat) -> NSAttributedString {
        let m = NSMutableAttributedString()
        if cells.cols > 0 {
            let atomicId = id
            let natural = cells.naturals.map { w in w + columnGap }
            let minimums = cells.minimums.map { w in w + columnGap }
            let room = width > 0 ? width : natural.reduce(0, +)
            let widths = TableMetrics.scrollingLayout(
                headers: cells.headers, rows: cells.rows, natural: natural,
                minimums: minimums, available: room).widths
            let texts = widths.map { w in max(w - columnGap, 1) }
            let stops = tabStops(widths: widths, texts: texts,
                                 alignments: alignments)
            if !cells.header.isEmpty {
                m.append(tableRow(cells.header, stops: stops, texts: texts,
                                  bold: true,
                                  tint: platformWhite(0.5, alpha: 0.14),
                                  atomicId: atomicId, style: style))
            }
            for (idx, row) in cells.body.enumerated() {
                let tint: PlatformColor = idx % 2 == 1
                    ? platformWhite(0.5, alpha: 0.07) : platformClearColor
                m.append(tableRow(row, stops: stops, texts: texts,
                                  bold: false, tint: tint,
                                  atomicId: atomicId, style: style))
            }
            // One CONTIGUOUS atomic id over the whole table content so the copy
            // grouping is ONE block.
            let content = NSRange(location: 0, length: m.length)
            m.addAttribute(atomicIdKey, value: atomicId, range: content)
            m.addAttribute(atomicCopyKey,
                           value: TableMetrics.serializeMonospaced(
                               headers: cells.headers, rows: cells.rows,
                               alignments: alignments),
                           range: content)
            m.append(NSAttributedString(string: "\n"))
        }
        return m
    }

    private static func tabStops(widths: [CGFloat], texts: [CGFloat],
                                 alignments: [Markdown.Alignment])
        -> [NSTextTab] {
        var out: [NSTextTab] = []
        var left: CGFloat = 0
        for (col, w) in widths.enumerated() {
            if col > 0 {
                let align = tabAlignment(col, alignments)
                var at = left
                if align == .right { at = left + texts[col] }
                if align == .center { at = left + texts[col] / 2 }
                out.append(NSTextTab(textAlignment: align, location: at))
            }
            left += w
        }
        return out
    }

    private static func tableRow(_ cells: [TableCell], stops: [NSTextTab],
                                 texts: [CGFloat], bold: Bool,
                                 tint: PlatformColor,
                                 atomicId: String, style: MarkdownStyle)
        -> NSAttributedString {
        let para = NSMutableParagraphStyle()
        para.tabStops = stops
        para.lineBreakMode = .byWordWrapping
        let base = bold ? boldFont(of: bodyFont(style)) : bodyFont(style)
        let columns = cells.enumerated().map { pair in
            wrapCell(pair.element.text,
                     width: pair.offset < texts.count ? texts[pair.offset] : 1)
        }
        let m = NSMutableAttributedString()
        let height = max(columns.map { lines in lines.count }.max() ?? 0, 1)
        for line in 0 ..< height {
            for (i, lines) in columns.enumerated() {
                if i > 0 {
                    m.append(NSAttributedString(string: "\t",
                                                attributes: [.font: base]))
                }
                if line < lines.count { m.append(lines[line]) }
            }
            m.append(NSAttributedString(string: "\n",
                                        attributes: [.font: base]))
        }
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(.paragraphStyle, value: para, range: full)
        m.addAttribute(.backgroundColor, value: tint, range: full)
        m.addAttribute(atomicKindKey, value: AtomicKind.table.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: atomicId, range: full)
        return m
    }

    private static func tabAlignment(_ col: Int,
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
