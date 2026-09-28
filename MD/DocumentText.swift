import CoreText
import Foundation

// Flattens a document into one NSAttributedString for the single selectable
// surface. Adapted from md.too `src/DocumentText.swift`.
@MainActor enum DocumentText {

    @MainActor final class RenderCache {
        struct Entry {
            let block: Markdown.Block
            let style: MarkdownStyle
            let width: CGFloat
            let images: [URL: ObjectIdentifier]
            let text: NSAttributedString
        }

        struct Minimum {
            let block: Markdown.Block
            let style: MarkdownStyle
            let formulas: Bool
            let width: CGFloat
        }

        struct Table {
            let block: Markdown.Block
            let style: MarkdownStyle
            let images: [URL: ObjectIdentifier]
            let cells: TableCells
        }

        var entries: [Int: Entry] = [:]
        var minimums: [Int: Minimum] = [:]
        var tables: [Int: Table] = [:]

        nonisolated init() {}
    }

    static func attributed(from document: Markdown.Document,
                           style: MarkdownStyle,
                           images: [URL: PlatformImage] = [:],
                           width: CGFloat = 0, wide: Bool = false,
                           cache: RenderCache? = nil)
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        let seen = images.mapValues { image in ObjectIdentifier(image) }
        var live: [Int: RenderCache.Entry] = [:]
        for item in document.items {
            var entry = cache?.entries[item.id]
            let own = ownImages(item.block, seen)
            let stale = entry?.block != item.block || entry?.style != style
                || entry?.width != width || entry?.images != own
            if stale {
                entry = RenderCache.Entry(
                    block: item.block, style: style, width: width,
                    images: own,
                    text: completed(
                        render(item, style: style, images: images,
                               seen: seen, width: width, cache: cache),
                        style: style))
            }
            if let entry {
                live[item.id] = entry
                m.append(entry.text)
            }
        }
        if let cache {
            cache.entries = live
            cache.tables = cache.tables.filter { pair in
                live[pair.key] != nil
            }
        }
        if wide, width > 0 { wrapProse(m, at: width) }
        return m
    }

    private static func ownImages(_ block: Markdown.Block,
                                  _ seen: [URL: ObjectIdentifier])
        -> [URL: ObjectIdentifier] {
        var result: [URL: ObjectIdentifier] = [:]
        if !seen.isEmpty {
            let alone = Markdown.Document(
                items: [Markdown.Document.Item(id: 0, block: block)])
            for url in ImagePrefetch.collectURLs(in: alone) {
                result[url] = seen[url]
            }
        }
        return result
    }

    private static func completed(_ text: NSAttributedString,
                                  style: MarkdownStyle)
        -> NSAttributedString {
        let m = NSMutableAttributedString(attributedString: text)
        let full = NSRange(location: 0, length: m.length)
        let base = bodyFont(style)
        m.enumerateAttribute(.font, in: full, options: []) { value, r, _ in
            if value == nil { m.addAttribute(.font, value: base, range: r) }
        }
        m.enumerateAttribute(.foregroundColor, in: full,
                             options: []) { value, r, _ in
            if value == nil {
                m.addAttribute(.foregroundColor,
                               value: platformDefaultTextColor, range: r)
            }
        }
        m.fixAttributes(in: full)
        return m
    }

    private static func render(_ item: Markdown.Document.Item,
                               style: MarkdownStyle,
                               images: [URL: PlatformImage],
                               seen: [URL: ObjectIdentifier],
                               width: CGFloat,
                               cache: RenderCache?) -> NSAttributedString {
        let result: NSAttributedString
        if case .table(_, _, let aligns) = item.block,
           let cells = tableCells(of: item, style: style, images: images,
                                  seen: seen, cache: cache) {
            result = table(cells, alignments: aligns, id: String(item.id),
                           style: style, width: width)
        } else {
            result = render(item.block, id: String(item.id), style: style,
                            images: images, width: width)
        }
        return result
    }

    private static func tableCells(of item: Markdown.Document.Item,
                                   style: MarkdownStyle,
                                   images: [URL: PlatformImage],
                                   seen: [URL: ObjectIdentifier],
                                   cache: RenderCache?) -> TableCells? {
        var result: TableCells? = nil
        if case .table(let headers, let rows, _) = item.block {
            var known = cache?.tables[item.id]
            let own = ownImages(item.block, seen)
            let stale = known?.block != item.block || known?.style != style
                || known?.images != own
            if stale {
                known = RenderCache.Table(
                    block: item.block, style: style, images: own,
                    cells: tableCells(headers: headers, rows: rows,
                                      style: style, images: images))
                cache?.tables[item.id] = known
            }
            result = known?.cells
        }
        return result
    }

    private static func wrapProse(_ m: NSMutableAttributedString,
                                  at visible: CGFloat) {
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(atomicKindKey, in: full,
                             options: []) { kind, span, _ in
            if kind as? String != AtomicKind.table.rawValue {
                endLines(of: m, in: span, at: visible)
            }
        }
    }

    private static func endLines(of m: NSMutableAttributedString,
                                 in span: NSRange, at visible: CGFloat) {
        m.enumerateAttribute(.paragraphStyle, in: span,
                             options: []) { value, range, _ in
            let para = NSMutableParagraphStyle()
            if let existing = value as? NSParagraphStyle {
                para.setParagraphStyle(existing)
            }
            para.tailIndent = visible
            m.addAttribute(.paragraphStyle, value: para, range: range)
        }
    }

    static func render(_ block: Markdown.Block, id: String,
                       style: MarkdownStyle, images: [URL: PlatformImage],
                       width: CGFloat = 0) -> NSAttributedString {
        let result: NSAttributedString
        switch block {
            case .paragraph(let attr):
                result = paragraph(attr, style: style)
            case .heading(let level, let attr):
                result = heading(level: level, text: attr, style: style)
            case .code(let lang, let text):
                result = code(language: lang, text: text, id: id,
                              style: style)
            case .quote(let inner):
                result = quote(inner, id: id, style: style, images: images,
                               width: width - 18)
            case .list(let items, let tight):
                result = list(items: items, tight: tight, depth: 0, base: 0,
                              id: id, style: style, images: images,
                              width: width)
            case .table(let headers, let rows, let aligns):
                result = table(headers: headers, rows: rows,
                               alignments: aligns, id: id, style: style,
                               images: images, width: width)
            case .math(let tex):
                result = math(tex, id: id, style: style)
            case .rule:
                result = rule(style: style)
            case .image(let alt, let url, let w, let h):
                result = image(alt: alt, url: url, width: w, height: h,
                               id: id, style: style, images: images)
        }
        return result
    }

    static func bodyFont(_ style: MarkdownStyle) -> PlatformFont {
        FontRole.body(style.bodySize).platformFont
    }

    static func blockParagraph(_ style: MarkdownStyle)
        -> NSMutableParagraphStyle {
        let para = NSMutableParagraphStyle()
        para.paragraphSpacing = style.blockSpacing
        return para
    }

    // The narrowest this document can be drawn before a table or a formula is
    // asked for less room than its content can occupy.
    static func minimumWidth(of document: Markdown.Document,
                             style: MarkdownStyle,
                             formulas: Bool = true,
                             images: [URL: PlatformImage] = [:],
                             cache: RenderCache? = nil) -> CGFloat {
        var widest: CGFloat = 0
        let seen = images.mapValues { image in ObjectIdentifier(image) }
        var live: [Int: RenderCache.Minimum] = [:]
        for item in document.items {
            var known = cache?.minimums[item.id]
            let stale = known?.block != item.block || known?.style != style
                || known?.formulas != formulas
            if stale {
                known = RenderCache.Minimum(
                    block: item.block, style: style, formulas: formulas,
                    width: minimumWidth(of: item, style: style,
                                        formulas: formulas, images: images,
                                        seen: seen, cache: cache))
            }
            if let known {
                live[item.id] = known
                if known.width > widest { widest = known.width }
            }
        }
        cache?.minimums = live
        return widest
    }

    static func minimumWidth(of blocks: [Markdown.Block],
                             style: MarkdownStyle,
                             formulas: Bool = true) -> CGFloat {
        var widest: CGFloat = 0
        for block in blocks {
            let w = minimumWidth(ofBlock: block, style: style,
                                 formulas: formulas)
            if w > widest { widest = w }
        }
        return widest
    }

    private static func minimumWidth(of item: Markdown.Document.Item,
                                     style: MarkdownStyle, formulas: Bool,
                                     images: [URL: PlatformImage],
                                     seen: [URL: ObjectIdentifier],
                                     cache: RenderCache?) -> CGFloat {
        let result: CGFloat
        if let cells = tableCells(of: item, style: style, images: images,
                                  seen: seen, cache: cache) {
            result = tableMinimumWidth(cells)
        } else {
            result = minimumWidth(ofBlock: item.block, style: style,
                                  formulas: formulas)
        }
        return result
    }

    private static func minimumWidth(ofBlock block: Markdown.Block,
                                     style: MarkdownStyle,
                                     formulas: Bool) -> CGFloat {
        var result: CGFloat = 0
        switch block {
            case .table(let headers, let rows, _):
                result = tableMinimumWidth(headers: headers, rows: rows,
                                           style: style)
            case .math(let tex):
                result = formulas ? mathMinimumWidth(tex, style: style) : 0
            case .quote(let inner):
                result = indented(minimumWidth(of: inner, style: style,
                                               formulas: formulas), by: 18)
            case .list(let items, _):
                let step = markerStep(items, style: style)
                for item in items {
                    let w = indented(minimumWidth(of: item.blocks,
                                                  style: style,
                                                  formulas: formulas),
                                     by: step)
                    if w > result { result = w }
                }
            default:
                result = 0
        }
        return result
    }

    // A formula scaled to the width on offer, or left alone when it already
    // fits.

    // nonisolated because the sizing overrides that ask it are: pure
    // arithmetic over two values, touching nothing shared.
    nonisolated static func mathFit(natural: CGSize,
                                    available: CGFloat) -> CGSize {
        var scale: CGFloat = 1
        if available > 0, natural.width > available {
            scale = available / natural.width
        }
        return CGSize(width: natural.width * scale,
                      height: natural.height * scale)
    }

    // A formula has no line breaks to give, so the surface must be wide
    // enough to hold it whole, plus the copy button's gutter on both sides.
    private static func mathMinimumWidth(_ tex: String,
                                         style: MarkdownStyle) -> CGFloat {
        let size = TeX.displaySize(body: bodyFont(style).pointSize)
        var result: CGFloat = 0
        if let layout = TeX.layout(tex, size: size) {
            result = ceil(layout.width) + 8 + copyButtonGutter * 2
        }
        return result
    }

    // An indent only widens a document that had something to widen it; a
    // quote full of prose still asks for nothing.
    private static func indented(_ inner: CGFloat,
                                 by amount: CGFloat) -> CGFloat {
        inner > 0 ? inner + amount : 0
    }

    static func tableMinimumWidth(headers: [String], rows: [[String]],
                                  style: MarkdownStyle) -> CGFloat {
        tableMinimumWidth(tableCells(headers: headers, rows: rows,
                                     style: style, images: [:]))
    }

    static func table(headers: [String], rows: [[String]],
                      alignments: [Markdown.Alignment], id: String,
                      style: MarkdownStyle, images: [URL: PlatformImage],
                      width: CGFloat) -> NSAttributedString {
        table(tableCells(headers: headers, rows: rows, style: style,
                         images: images),
              alignments: alignments, id: id, style: style, width: width)
    }

    struct TableCell {
        let text: NSAttributedString
        let minimum: CGFloat
        let natural: CGFloat
    }

    struct TableCells {
        let headers: [String]
        let rows: [[String]]
        let cols: Int
        let header: [TableCell]
        let body: [[TableCell]]
        let minimums: [CGFloat]
        let naturals: [CGFloat]
    }

    static func tableCells(headers: [String], rows: [[String]],
                           style: MarkdownStyle,
                           images: [URL: PlatformImage]) -> TableCells {
        let body = bodyFont(style)
        let bold = boldFont(of: body)
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        let header = headers.map { cell in
            tableCell(cell, base: bold, style: style, images: images)
        }
        let built = rows.map { row in
            row.map { cell in
                tableCell(cell, base: body, style: style, images: images)
            }
        }
        var minimums = [CGFloat](repeating: 0, count: cols)
        var naturals = [CGFloat](repeating: 0, count: cols)
        for row in [header] + built {
            for (c, cell) in row.enumerated() where c < cols {
                if cell.minimum > minimums[c] { minimums[c] = cell.minimum }
                if cell.natural > naturals[c] { naturals[c] = cell.natural }
            }
        }
        return TableCells(headers: headers, rows: rows, cols: cols,
                          header: header, body: built,
                          minimums: minimums.map { w in ceil(w) },
                          naturals: naturals.map { w in ceil(w) })
    }

    static func tableCell(_ text: String, base: PlatformFont,
                          style: MarkdownStyle,
                          images: [URL: PlatformImage]) -> TableCell {
        let block = Markdown.parseCell(text).items.first?.block
        let m = NSMutableAttributedString()
        if let block {
            fillCell(block, text: text, base: base, style: style,
                     images: images, into: m)
        }
        var minimum: CGFloat = 0
        var natural: CGFloat = 0
        switch block {
            case .image?:
                break
            case .paragraph?:
                minimum = longestRunWidth(m)
                natural = tableUsesNaturals ? attributedWidth(m) : 0
            default:
                minimum = longestRunWidth(m)
                natural = naturalWidth(TeX.scriptsToUnicode(text),
                                       font: base)
        }
        return TableCell(text: m, minimum: minimum, natural: natural)
    }

    private static func naturalWidth(_ text: String,
                                     font: PlatformFont) -> CGFloat {
        var result: CGFloat = 0
        if tableUsesNaturals {
            result = (text as NSString)
                .size(withAttributes: [.font: font]).width
        }
        return result
    }

    static func attributedWidth(_ drawn: NSAttributedString) -> CGFloat {
        var widest: CGFloat = 0
        let ns = drawn.string as NSString
        var at = 0
        while at < ns.length {
            let end = ns.rangeOfCharacter(
                from: CharacterSet(charactersIn: "\n\u{2028}"),
                range: NSRange(location: at, length: ns.length - at))
            let stop = end.location == NSNotFound ? ns.length : end.location
            let line = CTLineCreateWithAttributedString(
                drawn.attributedSubstring(
                    from: NSRange(location: at, length: stop - at)))
            let w = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            if w > widest { widest = w }
            at = stop + 1
        }
        return widest
    }

    private static func longestRunWidth(_ drawn: NSAttributedString)
        -> CGFloat {
        var widest: CGFloat = 0
        for run in unbreakableRuns(drawn.string as NSString) {
            let line = CTLineCreateWithAttributedString(
                drawn.attributedSubstring(from: run))
            let w = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            if w > widest { widest = w }
        }
        return widest
    }

    static func unbreakableRuns(_ text: NSString) -> [NSRange] {
        var out: [NSRange] = []
        var start = 0
        for i in 0 ..< text.length {
            let c = text.character(at: i)
            let next = i + 1 < text.length ? text.character(at: i + 1) : 0
            let space = c == 0x20 || c == 0x09 || c == 0x0A || c == 0x2028
            let soft = [0x2D, 0x2F, 0x2013, 0x2014].contains(c)
                && !(0x30 ... 0x39).contains(next)
            if space || soft {
                let end = space ? i : i + 1
                if end > start {
                    out.append(NSRange(location: start, length: end - start))
                }
                start = i + 1
            }
        }
        if text.length > start {
            out.append(NSRange(location: start, length: text.length - start))
        }
        return out
    }

    static func wrapCell(_ cell: NSAttributedString,
                         width: CGFloat) -> [NSAttributedString] {
        var out: [NSAttributedString] = []
        let setter = CTTypesetterCreateWithAttributedString(cell)
        var at = 0
        while at < cell.length {
            let suggested = CTTypesetterSuggestLineBreak(
                setter, at, Double(max(width, 1)))
            let take = min(max(suggested, 1), cell.length - at)
            out.append(cell.attributedSubstring(
                from: NSRange(location: at, length: take)))
            at += take
        }
        return out
    }

    private static func fillCell(_ block: Markdown.Block, text: String,
                                 base: PlatformFont, style: MarkdownStyle,
                                 images: [URL: PlatformImage],
                                 into m: NSMutableAttributedString) {
        switch block {
            case .image(let alt, let url, let w, let h):
                appendImage(alt: alt, url: url, width: w, height: h,
                            base: base, images: images, into: m)
            case .paragraph(let attr):
                translateInline(attr, base: base, style: style, into: m)
            default:
                m.append(NSAttributedString(
                    string: text,
                    attributes: [.font: base,
                                 .foregroundColor: platformDefaultTextColor]))
        }
    }

    private static func appendImage(alt: String, url: URL, width: CGFloat?,
                                    height: CGFloat?, base: PlatformFont,
                                    images: [URL: PlatformImage],
                                    into m: NSMutableAttributedString) {
        if let img = images[url] {
            let attachment = NSTextAttachment()
            attachment.image = img
            attachment.bounds = imageBounds(img, width: width, height: height)
            m.append(NSAttributedString(attachment: attachment))
        } else {
            let label = alt.isEmpty ? url.absoluteString : alt
            m.append(NSAttributedString(
                string: "[Image: \(label)]",
                attributes: [.font: base,
                             .foregroundColor: platformSecondaryColor]))
        }
    }

    private static func quote(_ blocks: [Markdown.Block], id: String,
                              style: MarkdownStyle,
                              images: [URL: PlatformImage],
                              width: CGFloat)
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        for (i, inner) in blocks.enumerated() {
            m.append(render(inner, id: id + "." + String(i), style: style,
                            images: images, width: width))
        }
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(.paragraphStyle, in: full,
                             options: []) { value, range, _ in
            let merged = NSMutableParagraphStyle()
            if let existing = value as? NSParagraphStyle {
                merged.setParagraphStyle(existing)
            }
            merged.headIndent += 18
            merged.firstLineHeadIndent += 18
            m.addAttribute(.paragraphStyle, value: merged, range: range)
        }
        m.addAttribute(.backgroundColor,
                       value: platformWhite(0.5, alpha: 0.06), range: full)
        return m
    }

    private static func list(items: [Markdown.ListItem], tight: Bool,
                             depth: Int, base: CGFloat, id: String,
                             style: MarkdownStyle,
                             images: [URL: PlatformImage],
                             width: CGFloat)
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        let indent = base + markerStep(items, style: style)
        for (idx, item) in items.enumerated() {
            let para = NSMutableParagraphStyle()
            para.headIndent = indent
            para.firstLineHeadIndent = base
            para.tabStops = [NSTextTab(textAlignment: .left,
                                       location: indent)]
            para.paragraphSpacing = tight ? 2 : 8
            para.paragraphSpacingBefore = tight ? 2 : 4
            m.append(listItem(item, para: para, depth: depth,
                              id: id + "." + String(idx), style: style,
                              images: images, width: width))
        }
        if depth == 0 { spaceAfter(m, style.blockSpacing) }
        return m
    }

    private static func markerText(_ item: Markdown.ListItem) -> String {
        item.checked.map { c in c ? "\u{2611}" : "\u{2610}" } ?? item.marker
    }

    static func markerStep(_ items: [Markdown.ListItem],
                           style: MarkdownStyle) -> CGFloat {
        let font = bodyFont(style)
        let widest = items.map { item in
            NSAttributedString(string: markerText(item),
                               attributes: [.font: font]).size().width
        }.max() ?? 0
        return max(20, ceil(widest + style.bodySize * 0.5))
    }

    private static func spaceAfter(_ m: NSMutableAttributedString,
                                   _ spacing: CGFloat) {
        if m.length > 0 {
            let last = (m.string as NSString).paragraphRange(
                for: NSRange(location: m.length - 1, length: 0))
            m.enumerateAttribute(.paragraphStyle, in: last,
                                 options: []) { value, r, _ in
                let para = NSMutableParagraphStyle()
                if let v = value as? NSParagraphStyle {
                    para.setParagraphStyle(v)
                }
                para.paragraphSpacing = max(para.paragraphSpacing, spacing)
                m.addAttribute(.paragraphStyle, value: para, range: r)
            }
        }
    }

    private static func listItem(_ item: Markdown.ListItem,
                                 para: NSParagraphStyle, depth: Int,
                                 id: String, style: MarkdownStyle,
                                 images: [URL: PlatformImage],
                                 width: CGFloat)
        -> NSAttributedString {
        let body = NSMutableAttributedString()
        for (k, block) in item.blocks.enumerated() {
            let blockId = id + "." + String(k)
            if case .list(let inner, let innerTight) = block {
                body.append(list(items: inner, tight: innerTight,
                                 depth: depth + 1, base: para.headIndent,
                                 id: blockId, style: style, images: images,
                                 width: width))
            } else if k == 0, case .paragraph(let attr) = block {
                let line = NSMutableAttributedString()
                translateInline(attr, base: bodyFont(style), style: style,
                                into: line)
                line.append(NSAttributedString(string: "\n"))
                line.addAttribute(.paragraphStyle, value: para,
                                  range: NSRange(location: 0,
                                                 length: line.length))
                body.append(line)
            } else {
                let rendered = NSMutableAttributedString(
                    attributedString: render(
                        block, id: blockId, style: style, images: images,
                        width: width - para.headIndent))
                shift(rendered, by: para.headIndent)
                body.append(rendered)
            }
        }
        return marked(body, marker: markerText(item), para: para,
                      first: item.blocks.first, style: style)
    }

    private static func shift(_ m: NSMutableAttributedString,
                              by amount: CGFloat) {
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(.paragraphStyle, in: full,
                             options: []) { value, range, _ in
            let kind = m.attribute(atomicKindKey, at: range.location,
                                   effectiveRange: nil) as? String
            if kind != AtomicKind.table.rawValue {
                let para = NSMutableParagraphStyle()
                if let v = value as? NSParagraphStyle {
                    para.setParagraphStyle(v)
                }
                para.headIndent += amount
                para.firstLineHeadIndent += amount
                para.tabStops = para.tabStops.map { stop in
                    NSTextTab(textAlignment: stop.alignment,
                              location: stop.location + amount,
                              options: stop.options)
                }
                m.addAttribute(.paragraphStyle, value: para, range: range)
            }
        }
    }

    private static func marked(_ body: NSMutableAttributedString,
                               marker: String, para: NSParagraphStyle,
                               first: Markdown.Block?, style: MarkdownStyle)
        -> NSAttributedString {
        var attrs: [NSAttributedString.Key: Any] = [
            .font: bodyFont(style),
            .foregroundColor: platformSecondaryColor,
            .paragraphStyle: para,
        ]
        var joins = false
        switch first {
            case .paragraph?, .list?, .heading?: joins = true
            default: joins = false
        }
        let head = body.length > 0
            ? body.attribute(.paragraphStyle, at: 0, effectiveRange: nil)
                as? NSParagraphStyle
            : nil
        if joins, let head {
            let joined = NSMutableParagraphStyle()
            joined.setParagraphStyle(head)
            joined.firstLineHeadIndent = para.firstLineHeadIndent
            joined.tabStops = [NSTextTab(textAlignment: .left,
                                         location: head.firstLineHeadIndent)]
                + head.tabStops.filter { stop in
                    stop.location > head.firstLineHeadIndent
                }
            let line = (body.string as NSString)
                .paragraphRange(for: NSRange(location: 0, length: 0))
            body.addAttribute(.paragraphStyle, value: joined, range: line)
            attrs[.paragraphStyle] = joined
            body.insert(NSAttributedString(string: marker + "\t",
                                           attributes: attrs), at: 0)
        } else {
            body.insert(NSAttributedString(string: marker + "\n",
                                           attributes: attrs), at: 0)
        }
        return body
    }

    private static func image(alt: String, url: URL, width: CGFloat?,
                              height: CGFloat?, id: String,
                              style: MarkdownStyle,
                              images: [URL: PlatformImage])
        -> NSAttributedString {
        let m = NSMutableAttributedString()
        appendImage(alt: alt, url: url, width: width, height: height,
                    base: bodyFont(style), images: images, into: m)
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(atomicKindKey, value: AtomicKind.image.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: id, range: full)
        m.addAttribute(.paragraphStyle, value: blockParagraph(style),
                       range: full)
        m.append(NSAttributedString(string: "\n"))
        return m
    }

    private static func imageBounds(_ img: PlatformImage, width: CGFloat?,
                                    height: CGFloat?) -> CGRect {
        let fit = aspectFit(intrinsicWidth: img.size.width,
                            intrinsicHeight: img.size.height,
                            explicitWidth: width, explicitHeight: height,
                            maxWidth: 320)
        return CGRect(x: 0, y: 0, width: fit.width, height: fit.height)
    }

    private static func code(language: String?, text: String, id: String,
                             style: MarkdownStyle) -> NSAttributedString {
        let font = FontRole.mono(style.codeSize).platformFont
        let highlighted = style.highlightCode
            ? Highlight.attribute(text, language: language, baseFont: font)
            : NSAttributedString(string: text, attributes: [.font: font])
        let m = NSMutableAttributedString(attributedString: highlighted)
        // A trailing newline INSIDE the tinted run, or NSTextView paints no
        // background for the last code line.
        if !text.hasSuffix("\n") {
            m.append(NSAttributedString(string: "\n", attributes: [.font: font]))
        }
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(.backgroundColor,
                       value: platformWhite(0.5, alpha: 0.10), range: full)
        m.addAttribute(atomicKindKey, value: AtomicKind.code.rawValue,
                       range: full)
        m.addAttribute(atomicIdKey, value: id, range: full)
        m.addAttribute(atomicCopyKey, value: text, range: full)
        m.addAttribute(.paragraphStyle, value: blockParagraph(style),
                       range: full)
        m.append(NSAttributedString(string: "\n"))
        return m
    }

    // A display carries its TeX on atomicCopyKey so Copy yields the
    // formula, not the object-replacement character.
    private static func math(_ tex: String, id: String,
                             style: MarkdownStyle) -> NSAttributedString {
        let base = bodyFont(style)
        let m = NSMutableAttributedString()
        if let layout = TeX.layout(tex,
                                   size: TeX.displaySize(body: base.pointSize)) {
            m.append(NSAttributedString(attachment: mathAttachment(layout)))
        } else {
            translateInline(TeX.render(tex, display: true), base: base,
                            style: style, into: m)
        }
        let content = NSRange(location: 0, length: m.length)
        m.addAttribute(atomicKindKey, value: AtomicKind.math.rawValue,
                       range: content)
        m.addAttribute(atomicIdKey, value: id, range: content)
        m.addAttribute(atomicCopyKey, value: tex, range: content)
        m.append(NSAttributedString(string: "\n"))
        let para = NSMutableParagraphStyle()
        para.alignment = .center
        para.paragraphSpacing = style.blockSpacing
        m.addAttribute(.paragraphStyle, value: para,
                       range: NSRange(location: 0, length: m.length))
        return m
    }

    private static func paragraph(_ attr: AttributedString,
                                  style: MarkdownStyle) -> NSAttributedString {
        let m = NSMutableAttributedString()
        translateInline(attr, base: bodyFont(style), style: style, into: m)
        m.addAttribute(.paragraphStyle, value: blockParagraph(style),
                       range: NSRange(location: 0, length: m.length))
        m.append(NSAttributedString(string: "\n"))
        return m
    }

    private static func heading(level: Int, text: AttributedString,
                                style: MarkdownStyle) -> NSAttributedString {
        let font = FontRole.heading(
            level: level, size: style.headingSize(level)).platformFont
        let m = NSMutableAttributedString()
        translateInline(text, base: font, style: style, into: m)
        // A tight list ends with no blank line, so a heading right after it
        // needs its own spacing-before.
        let para = blockParagraph(style)
        para.paragraphSpacingBefore = style.blockSpacing
        m.addAttribute(.paragraphStyle, value: para,
                       range: NSRange(location: 0, length: m.length))
        m.append(NSAttributedString(string: "\n"))
        return m
    }

    private static func rule(style: MarkdownStyle) -> NSAttributedString {
        NSAttributedString(
            string: String(repeating: "\u{2500}", count: 8) + "\n",
            attributes: [.font: bodyFont(style),
                         .foregroundColor: platformSecondaryColor,
                         .paragraphStyle: blockParagraph(style)])
    }

    static func translateInline(_ attr: AttributedString, base: PlatformFont,
                                style: MarkdownStyle,
                                into m: NSMutableAttributedString) {
        for run in attr.runs {
            let segment = String(attr[run.range].characters)
            let intent = run.inlinePresentationIntent ?? []
            var runFont = styledRunFont(intent: intent, base: base)
            var attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: platformDefaultTextColor,
            ]
            if run[SmallAttribute.self] == true {
                runFont = smallRunFont(base: runFont)
            }
            if let level = run[ScriptAttribute.self] {
                let script = scriptRunFont(level, base: runFont)
                runFont = script.font
                attrs[.baselineOffset] = script.offset
            }
            attrs[.font] = runFont
            if intent.contains(.strikethrough) {
                attrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
            }
            if run.underlineStyle != nil {
                attrs[.underlineStyle] = NSUnderlineStyle.single.rawValue
            }
            if let url = run.link { attrs[.link] = url }
            m.append(NSAttributedString(string: segment, attributes: attrs))
        }
    }
}
