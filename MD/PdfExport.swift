import Foundation
import CoreText
import CoreGraphics

// Paginated PDF via CoreText. Adapted from md.too `src/PDFRenderer.swift` and
// `src/PDFExport.swift`, on the block model, with per-column table alignment.
public enum MarkdownPDF {

    public static func data(_ document: Markdown.Document, title: String,
                            images: [URL: CGImage] = [:]) -> Data? {
        var result: Data? = nil
        let body = {
            result = PdfWriter.data(
                blocks: document.items.map { i in i.block },
                title: title, images: images)
        }
        platformPerformLightAppearance(body)
        return result
    }

    public static func export(_ document: Markdown.Document,
                              title: String) async -> Data? {
        let images = await ImagePrefetch.fetchAndDecode(
            in: document, decode: { d in platformDecodeCGImage(d) })
        return data(document, title: title, images: images)
    }
}

enum PdfWriter {

    static func data(blocks: [Markdown.Block], title: String,
                     images: [URL: CGImage]) -> Data? {
        var result: Data? = nil
        let buffer = NSMutableData()
        let pageSize = paperSize()
        var media = CGRect(origin: .zero, size: pageSize)
        if let consumer = CGDataConsumer(data: buffer),
           let ctx = CGContext(consumer: consumer, mediaBox: &media, nil) {
            let r = PDFRenderer(ctx: ctx, pageSize: pageSize, title: title,
                                images: images)
            r.startPage()
            for block in blocks { r.draw(block) }
            r.endPage()
            ctx.closePDF()
            result = buffer as Data
        }
        return result
    }

    static func paperSize() -> CGSize {
        let a4 = CGSize(width: 595, height: 842)
        let letter = CGSize(width: 612, height: 792)
        let letterRegions: Set<String> = [
            "US", "CA", "MX", "PH", "PR", "CL", "CO", "CR", "PA", "PE", "VE",
        ]
        let region = Locale.current.region?.identifier ?? ""
        return letterRegions.contains(region) ? letter : a4
    }
}

final class PDFRenderer {

    let ctx: CGContext
    let pageSize: CGSize
    let title: String
    let images: [URL: CGImage]
    let margin: CGFloat = 54
    let headerH: CGFloat = 28
    let footerH: CGFloat = 28
    let blockGap: CGFloat = 10
    let bodySize: CGFloat = 11
    let monoSize: CGFloat = 10
    var pageNumber = 0
    var y: CGFloat = 0
    var listIndent: CGFloat = 0

    init(ctx: CGContext, pageSize: CGSize, title: String,
         images: [URL: CGImage]) {
        self.ctx = ctx
        self.pageSize = pageSize
        self.title = title
        self.images = images
    }

    var contentLeft: CGFloat { margin + listIndent }
    var contentRight: CGFloat { pageSize.width - margin }
    var contentWidth: CGFloat { contentRight - contentLeft }
    var contentTop: CGFloat { pageSize.height - margin - headerH }
    var contentBottom: CGFloat { margin + footerH }
    var remaining: CGFloat { y - contentBottom }

    func startPage() {
        ctx.beginPDFPage(nil)
        pageNumber += 1
        y = contentTop
        drawHeader()
        drawFooter()
    }

    func endPage() { ctx.endPDFPage() }

    func newPage() {
        endPage()
        startPage()
    }

    func ensureSpace(_ minHeight: CGFloat) {
        if remaining < minHeight { newPage() }
    }

    func draw(_ block: Markdown.Block) {
        switch block {
            case .heading(let level, let text):
                drawHeading(level: level, text: text)
            case .paragraph(let attr):
                drawText(attr, font: bodyFont(), color: textColor)
            case .code(let language, let text):
                drawCode(text, language: language)
            case .quote(let blocks):
                drawQuote(blocks)
            case .list(let items, _):
                drawList(items)
            case .table(let headers, let rows, let aligns):
                drawTable(headers: headers, rows: rows, aligns: aligns)
            case .math(let tex):
                drawMath(tex)
            case .rule:
                drawRule()
            case .image(let alt, let url, let width, _):
                if let cg = images[url] {
                    drawImage(cg, alt: alt, explicitWidth: width)
                } else {
                    drawImagePlaceholder(alt: alt, url: url)
                }
        }
        y -= blockGap
    }

    private func drawHeading(level: Int, text: AttributedString) {
        let sizes: [Int: CGFloat] = [1: 22, 2: 18, 3: 16, 4: 14, 5: 12, 6: 11]
        let size = sizes[level] ?? 11
        let font = CTFontCreateUIFontForLanguage(.system, size, nil)
            ?? CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
        let bold = CTFontCreateCopyWithSymbolicTraits(font, size, nil,
            .traitBold, .traitBold) ?? font
        drawText(text, font: bold, color: textColor)
    }

    // Takes the AttributedString because the script level rides a custom
    // key that NSAttributedString(_:) drops.
    private func drawText(_ attr: AttributedString, font: CTFont,
                          color: CGColor) {
        let m = NSMutableAttributedString(
            attributedString: NSAttributedString(attr))
        let full = NSRange(location: 0, length: m.length)
        m.enumerateAttribute(.font, in: full, options: []) { v, range, _ in
            if v == nil {
                m.addAttribute(.font, value: font, range: range)
            } else if let existing = v as? PlatformFont {
                let sized = platformResizedFont(existing,
                                                to: CTFontGetSize(font))
                m.addAttribute(.font, value: sized, range: range)
            }
        }
        m.enumerateAttribute(.foregroundColor, in: full,
                             options: []) { v, range, _ in
            if v == nil {
                m.addAttribute(.foregroundColor, value: color, range: range)
            }
        }
        applyScriptRuns(m, from: attr)
        flow(m)
    }

    // Centred while it fits, shrunk to the column rather than clipped when
    // it does not: a page cannot scroll sideways.
    private func drawMath(_ tex: String) {
        if let layout = fittedMath(tex) {
            ensureSpace(layout.height + bodySize)
            let slack = contentWidth - layout.width
            let x = contentLeft + max(slack / 2, 0)
            layout.draw(in: ctx, at: CGPoint(x: x, y: y), color: textColor)
            y -= layout.height
        } else {
            drawText(TeX.render(tex, display: true),
                     font: bodyFontItalic(), color: textColor)
        }
    }

    private func fittedMath(_ tex: String) -> MathLayout? {
        let wanted = TeX.displaySize(body: bodySize)
        var result = TeX.layout(tex, size: wanted)
        if let first = result, first.width > contentWidth, first.width > 0 {
            let fitted = wanted * contentWidth / first.width
            result = TeX.layout(tex, size: max(fitted, wanted * 0.5))
        }
        return result
    }

    private func flow(_ attr: NSAttributedString) {
        if attr.length > 0 {
            let fs = CTFramesetterCreateWithAttributedString(attr)
            var consumed = 0
            while consumed < attr.length {
                ensureSpace(20)
                let rem = CFRange(location: consumed,
                                  length: attr.length - consumed)
                let rect = CGRect(x: contentLeft, y: contentBottom,
                                  width: contentWidth, height: remaining)
                let path = CGPath(rect: rect, transform: nil)
                let frame = CTFramesetterCreateFrame(fs, rem, path, nil)
                let visible = CTFrameGetVisibleStringRange(frame)
                if visible.length == 0 {
                    newPage()
                } else {
                    let used = lineHeightUsed(frame: frame, in: rect)
                    CTFrameDraw(frame, ctx)
                    y -= used
                    consumed = visible.location + visible.length
                    if consumed < attr.length { newPage() }
                }
            }
        }
    }

    func lineHeightUsed(frame: CTFrame, in rect: CGRect) -> CGFloat {
        var result: CGFloat = 0
        let lines = CTFrameGetLines(frame) as? [CTLine] ?? []
        if !lines.isEmpty {
            var origins = [CGPoint](repeating: .zero, count: lines.count)
            CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 0),
                                  &origins)
            var ascent: CGFloat = 0
            var descent: CGFloat = 0
            var leading: CGFloat = 0
            _ = CTLineGetTypographicBounds(lines[0], &ascent, &descent,
                                           &leading)
            let topPadding = rect.height - origins[0].y - ascent
            let lastIdx = lines.count - 1
            _ = CTLineGetTypographicBounds(lines[lastIdx], &ascent, &descent,
                                           &leading)
            let used = rect.height - origins[lastIdx].y + descent - topPadding
            result = max(used, 0)
        }
        return result
    }

    private func drawCode(_ text: String, language: String?) {
        let ns = Highlight.attribute(text, language: language,
                                     baseFont: monoFont(at: monoSize))
        let m = NSMutableAttributedString(attributedString: ns)
        let full = NSRange(location: 0, length: m.length)
        m.addAttribute(.font, value: monoCTFont(), range: full)
        m.enumerateAttribute(.foregroundColor, in: full,
                             options: []) { value, range, _ in
            if let c = value as? PlatformColor {
                m.addAttribute(.foregroundColor, value: c.cgColor,
                               range: range)
            }
        }
        let fs = CTFramesetterCreateWithAttributedString(m)
        var consumed = 0
        while consumed < m.length {
            ensureSpace(20)
            let inset: CGFloat = 6
            let rem = CFRange(location: consumed, length: m.length - consumed)
            let rect = CGRect(x: contentLeft + inset, y: contentBottom,
                              width: contentWidth - 2 * inset,
                              height: remaining - 2 * inset)
            let path = CGPath(rect: rect, transform: nil)
            let frame = CTFramesetterCreateFrame(fs, rem, path, nil)
            let visible = CTFrameGetVisibleStringRange(frame)
            if visible.length == 0 {
                newPage()
            } else {
                let used = lineHeightUsed(frame: frame, in: rect)
                let bg = CGRect(x: contentLeft, y: y - used - 2 * inset,
                                width: contentWidth, height: used + 2 * inset)
                ctx.setFillColor(codeBgColor)
                ctx.fill(bg)
                CTFrameDraw(frame, ctx)
                y -= used + 2 * inset
                consumed = visible.location + visible.length
                if consumed < m.length { newPage() }
            }
        }
    }

    private func drawQuote(_ blocks: [Markdown.Block]) {
        let saved = listIndent
        let barX = margin + saved
        var startY = y
        var startPage = pageNumber
        listIndent = saved + 16
        for b in blocks {
            draw(b)
            if pageNumber != startPage {
                startPage = pageNumber
                startY = contentTop
            }
        }
        listIndent = saved
        if startY > y {
            ctx.setFillColor(secondaryColor)
            ctx.fill(CGRect(x: barX, y: y, width: 2, height: startY - y))
        }
    }

    private func drawList(_ items: [Markdown.ListItem]) {
        let saved = listIndent
        let gutter: CGFloat = 20
        for item in items {
            ensureSpace(bodySize * 2)
            let glyph = item.checked == nil ? item.marker
                : (item.checked == true ? "\u{2611}" : "\u{2610}")
            let markerAttr = NSAttributedString(string: glyph, attributes: [
                .font: bodyFont(), .foregroundColor: textColor])
            let line = CTLineCreateWithAttributedString(markerAttr)
            ctx.textPosition = CGPoint(x: margin + saved, y: y - bodySize)
            CTLineDraw(line, ctx)
            listIndent = saved + gutter
            if item.blocks.isEmpty {
                y -= bodySize * 1.4
            } else {
                for b in item.blocks { draw(b) }
            }
            listIndent = saved
        }
    }

    private func drawTable(headers: [String], rows: [[String]],
                           aligns: [Markdown.Alignment]) {
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        if cols > 0 {
            let table = PDFTable(renderer: self, headers: headers, rows: rows,
                                 aligns: aligns, cols: cols)
            table.draw()
        }
    }

    private func drawRule() {
        ensureSpace(8)
        ctx.setStrokeColor(secondaryColor)
        ctx.setLineWidth(0.5)
        ctx.move(to: CGPoint(x: contentLeft, y: y - 4))
        ctx.addLine(to: CGPoint(x: contentRight, y: y - 4))
        ctx.strokePath()
        y -= 8
    }

    private func drawImage(_ cg: CGImage, alt: String,
                           explicitWidth: CGFloat?) {
        let imgW = CGFloat(cg.width)
        let imgH = CGFloat(cg.height)
        if imgW > 0, imgH > 0 {
            let maxH = pageSize.height - margin * 2 - headerH - footerH
                - bodySize * 2
            let fit = aspectFit(intrinsicWidth: imgW, intrinsicHeight: imgH,
                                explicitWidth: explicitWidth,
                                defaultScale: 0.5, maxWidth: contentWidth)
            var drawW = fit.width
            var drawH = fit.height
            if drawH > maxH {
                drawH = maxH
                drawW = imgW > 0 ? drawH * (imgW / imgH) : drawW
            }
            ensureSpace(drawH + bodySize * 1.6)
            let originX = contentLeft + (contentWidth - drawW) / 2
            ctx.draw(cg, in: CGRect(x: originX, y: y - drawH,
                                    width: drawW, height: drawH))
            y -= drawH
            if !alt.isEmpty { drawCaption(alt) }
        }
    }

    private func drawCaption(_ alt: String) {
        let cap = NSAttributedString(string: alt, attributes: [
            .font: bodyFontItalic(), .foregroundColor: secondaryColor])
        let line = CTLineCreateWithAttributedString(cap)
        let bounds = CTLineGetBoundsWithOptions(line, [])
        let cx = contentLeft + (contentWidth - bounds.width) / 2
        ctx.textPosition = CGPoint(x: cx, y: y - bodySize - 4)
        CTLineDraw(line, ctx)
        y -= bodySize * 1.6
    }

    private func drawImagePlaceholder(alt: String, url: URL) {
        let label = alt.isEmpty ? url.absoluteString : alt
        let attr = NSAttributedString(string: "[image] \(label)", attributes: [
            .font: bodyFontItalic(), .foregroundColor: secondaryColor])
        ensureSpace(bodySize * 2)
        let inset: CGFloat = 8
        let line = CTLineCreateWithAttributedString(attr)
        let bounds = CTLineGetBoundsWithOptions(line, [])
        let h = bounds.height + 2 * inset
        ctx.setFillColor(codeBgColor)
        ctx.fill(CGRect(x: contentLeft, y: y - h, width: contentWidth,
                        height: h))
        let baselineY = y - inset - bounds.size.height - bounds.minY
        ctx.textPosition = CGPoint(x: contentLeft + inset, y: baselineY)
        CTLineDraw(line, ctx)
        y -= h
    }

    private func drawHeader() {
        let attr = NSAttributedString(string: title, attributes: [
            .font: smallFont(), .foregroundColor: secondaryColor])
        let line = CTLineCreateWithAttributedString(attr)
        ctx.textPosition = CGPoint(x: margin, y: pageSize.height - margin - 14)
        CTLineDraw(line, ctx)
        ctx.setStrokeColor(secondaryColor)
        ctx.setLineWidth(0.3)
        let lineY = pageSize.height - margin - 18
        ctx.move(to: CGPoint(x: margin, y: lineY))
        ctx.addLine(to: CGPoint(x: pageSize.width - margin, y: lineY))
        ctx.strokePath()
    }

    private func drawFooter() {
        let attr = NSAttributedString(string: "\(pageNumber)", attributes: [
            .font: smallFont(), .foregroundColor: secondaryColor])
        let line = CTLineCreateWithAttributedString(attr)
        let bounds = CTLineGetBoundsWithOptions(line, [])
        ctx.textPosition = CGPoint(x: (pageSize.width - bounds.width) / 2,
                                   y: margin + 6)
        CTLineDraw(line, ctx)
    }

    func bodyFont() -> CTFont {
        CTFontCreateUIFontForLanguage(.system, bodySize, nil)
            ?? CTFontCreateWithName("Helvetica" as CFString, bodySize, nil)
    }

    func bodyFontBold() -> CTFont {
        CTFontCreateCopyWithSymbolicTraits(bodyFont(), bodySize, nil,
            .traitBold, .traitBold) ?? bodyFont()
    }

    func bodyFontItalic() -> CTFont {
        CTFontCreateCopyWithSymbolicTraits(bodyFont(), bodySize, nil,
            .traitItalic, .traitItalic) ?? bodyFont()
    }

    func smallFont() -> CTFont {
        CTFontCreateUIFontForLanguage(.system, 9, nil)
            ?? CTFontCreateWithName("Helvetica" as CFString, 9, nil)
    }

    func monoCTFont() -> CTFont {
        CTFontCreateWithName("Menlo" as CFString, monoSize, nil)
    }

    var textColor: CGColor {
        CGColor(srgbRed: 0.10, green: 0.10, blue: 0.12, alpha: 1.0)
    }
    var secondaryColor: CGColor {
        CGColor(srgbRed: 0.40, green: 0.40, blue: 0.43, alpha: 1.0)
    }
    var codeBgColor: CGColor {
        CGColor(srgbRed: 0.95, green: 0.95, blue: 0.95, alpha: 1.0)
    }
    var rowShadeColor: CGColor {
        CGColor(srgbRed: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
    }
    var headerShadeColor: CGColor {
        CGColor(srgbRed: 0.93, green: 0.93, blue: 0.94, alpha: 1.0)
    }
}
