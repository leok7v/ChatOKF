import Foundation

// Self-contained HTML export. Adapted from md.too `src/HtmlExport.swift`, with
// per-column table alignment.
extension Markdown {

    public static func html(_ document: Document, title: String,
                            images: [URL: Data] = [:]) -> String {
        HtmlExport.render(document.items.map { i in i.block },
                          title: title, images: images)
    }

    public static func htmlPrefetching(_ document: Document,
                                       title: String) async -> String {
        let images = await MarkdownImages.fetch(document)
        return html(document, title: title, images: images)
    }
}

enum HtmlExport {

    static func render(_ blocks: [Markdown.Block], title: String,
                       images: [URL: Data]) -> String {
        var head = "<!DOCTYPE html>\n<html>\n<head>\n"
        head += "<meta charset=\"utf-8\">\n"
        head += "<title>\(esc(title))</title>\n</head>\n<body>\n"
        var body = ""
        for block in blocks { body += renderBlock(block, images: images) }
        return head + body + "</body>\n</html>\n"
    }

    private static func renderBlock(_ block: Markdown.Block,
                                    images: [URL: Data]) -> String {
        let result: String
        switch block {
            case .heading(let level, let text):
                result = "<h\(level)>\(renderInline(text))</h\(level)>\n"
            case .paragraph(let text):
                result = "<p>\(renderInline(text))</p>\n"
            case .code(let lang, let text):
                result = renderCode(lang: lang, text: text)
            case .quote(let inner):
                result = renderQuote(inner, images: images)
            case .list(let items, let tight):
                result = renderList(items, tight: tight, images: images)
            case .table(let headers, let rows, let aligns):
                result = renderTable(headers: headers, rows: rows,
                                     aligns: aligns)
            case .math(let tex):
                // Unicode, not a picture: a raster bakes in one theme's ink
                // and stops scaling with the page.
                result = "<p style=\"\(mathStyle)\">"
                    + "\(renderInline(TeX.render(tex, display: true)))</p>\n"
            case .rule:
                result = "<hr style=\"\(ruleStyle)\">\n"
            case .image(let alt, let url, let w, let h):
                result = renderImage(alt: alt, url: url, w: w, h: h,
                                     images: images)
        }
        return result
    }

    private static func renderInline(_ attr: AttributedString) -> String {
        var out = ""
        for run in attr.runs {
            let segment = esc(String(attr[run.range].characters))
            let intent = run.inlinePresentationIntent ?? []
            var open: [String] = []
            var close: [String] = []
            if let url = run.link {
                open.append("<a href=\"\(escAttr(url.absoluteString))\">")
                close.insert("</a>", at: 0)
            }
            if intent.contains(.code) {
                open.append("<code style=\"\(inlineCodeStyle)\">")
                close.insert("</code>", at: 0)
            }
            if intent.contains(.stronglyEmphasized) {
                open.append("<strong>")
                close.insert("</strong>", at: 0)
            }
            if intent.contains(.emphasized) {
                open.append("<em>")
                close.insert("</em>", at: 0)
            }
            if intent.contains(.strikethrough) {
                open.append("<del>")
                close.insert("</del>", at: 0)
            }
            // Back to the tag the source wrote.
            if let level = run[ScriptAttribute.self] {
                let tag = level > 0 ? "sup" : "sub"
                open.append("<\(tag)>")
                close.insert("</\(tag)>", at: 0)
            }
            out += open.joined() + segment + close.joined()
        }
        return out
    }

    private static func renderCode(lang: String?, text: String) -> String {
        let cls = lang.map { l in " class=\"language-\(escAttr(l))\"" } ?? ""
        return "<pre style=\"\(codeBlockStyle)\"><code\(cls)>"
            + "\(esc(text))</code></pre>\n"
    }

    private static func renderQuote(_ inner: [Markdown.Block],
                                    images: [URL: Data]) -> String {
        var s = "<blockquote style=\"\(quoteStyle)\">\n"
        for b in inner { s += renderBlock(b, images: images) }
        return s + "</blockquote>\n"
    }

    private static func renderList(_ items: [Markdown.ListItem], tight: Bool,
                                   images: [URL: Data]) -> String {
        var ordered = false
        if let first = items.first, let c = first.marker.first {
            ordered = c.isNumber
        }
        let tag = ordered ? "ol" : "ul"
        let style = tight ? listStyleTight : listStyleLoose
        var out = "<\(tag) style=\"\(style)\">\n"
        for item in items {
            var mark = ""
            if let c = item.checked {
                mark = "<span style=\"margin-right:0.4em\">"
                    + "\(c ? "&#9745;" : "&#9744;")</span>"
            }
            out += "<li>\(mark)"
            for b in item.blocks { out += renderBlock(b, images: images) }
            out += "</li>\n"
        }
        return out + "</\(tag)>\n"
    }

    private static func renderTable(headers: [String], rows: [[String]],
                                    aligns: [Markdown.Alignment]) -> String {
        let n = TableMetrics.columnCount(headers: headers, rows: rows)
        var out = "<table style=\"\(tableStyle)\">\n"
        if !headers.isEmpty {
            out += "<thead><tr style=\"\(rowHeaderStyle)\">\n"
            for i in 0..<n {
                let cell = i < headers.count ? headers[i] : ""
                let style = cellStyle(i, of: n, aligns, header: true)
                out += "<th style=\"\(style)\">"
                    + "\(inlineFromCell(cell))</th>\n"
            }
            out += "</tr></thead>\n"
        }
        out += "<tbody>\n"
        for (idx, row) in rows.enumerated() {
            let open = idx % 2 == 1
                ? "<tr style=\"\(rowShadeStyle)\">" : "<tr>"
            out += open + "\n"
            for i in 0..<n {
                let cell = i < row.count ? row[i] : ""
                let style = cellStyle(i, of: n, aligns, header: false)
                out += "<td style=\"\(style)\">"
                    + "\(inlineFromCell(cell))</td>\n"
            }
            out += "</tr>\n"
        }
        return out + "</tbody>\n</table>\n"
    }

    // The last column carries no divider, or the table gains an outer right
    // border no other edge has.
    private static func cellStyle(_ col: Int, of count: Int,
                                  _ aligns: [Markdown.Alignment],
                                  header: Bool) -> String {
        let base = (header ? thStyle : tdStyle)
            + (col < count - 1 ? colDividerStyle : "")
        let a = col < aligns.count ? aligns[col] : .none
        let text: String
        switch a {
            case .center: text = "text-align:center;"
            case .right: text = "text-align:right;"
            case .left: text = "text-align:left;"
            case .none: text = ""
        }
        return base + text
    }

    private static func inlineFromCell(_ raw: String) -> String {
        let parsed = Markdown.parse(raw)
        var attr = AttributedString(raw)
        if let first = parsed.items.first,
           case .paragraph(let a) = first.block {
            attr = a
        }
        return renderInline(attr)
    }

    private static func renderImage(alt: String, url: URL,
                                    w: CGFloat?, h: CGFloat?,
                                    images: [URL: Data]) -> String {
        var style = "max-width:100%;"
        if let w { style += "width:\(Int(w))px;" }
        if let h { style += "height:\(Int(h))px;" }
        let result: String
        if let data = images[url] {
            result = "<p><img alt=\"\(escAttr(alt))\" "
                + "src=\"\(dataURI(data))\" style=\"\(style)\"></p>\n"
        } else {
            let label = alt.isEmpty ? url.absoluteString : alt
            result = "<p style=\"\(imagePlaceholderStyle)\">"
                + "[\(esc(label))]</p>\n"
        }
        return result
    }

    private static func dataURI(_ data: Data) -> String {
        "data:\(detectMime(data));base64,\(data.base64EncodedString())"
    }

    private static func detectMime(_ data: Data) -> String {
        var result = "application/octet-stream"
        let b = [UInt8](data.prefix(4))
        if b.count >= 3, b[0] == 0xFF, b[1] == 0xD8, b[2] == 0xFF {
            result = "image/jpeg"
        } else if b.count >= 4, b[0] == 0x89, b[1] == 0x50,
                                b[2] == 0x4E, b[3] == 0x47 {
            result = "image/png"
        } else if b.count >= 3, b[0] == 0x47, b[1] == 0x49, b[2] == 0x46 {
            result = "image/gif"
        } else if b.count >= 4, b[0] == 0x52, b[1] == 0x49,
                                b[2] == 0x46, b[3] == 0x46 {
            result = "image/webp"
        }
        return result
    }

    private static func esc(_ s: String) -> String {
        var out = ""
        for c in s {
            switch c {
                case "&": out += "&amp;"
                case "<": out += "&lt;"
                case ">": out += "&gt;"
                default: out.append(c)
            }
        }
        return out
    }

    private static func escAttr(_ s: String) -> String {
        var out = ""
        for c in s {
            switch c {
                case "&": out += "&amp;"
                case "<": out += "&lt;"
                case ">": out += "&gt;"
                case "\"": out += "&quot;"
                case "'": out += "&#39;"
                default: out.append(c)
            }
        }
        return out
    }

    private static let codeBlockStyle =
        "background:rgba(128,128,128,0.10);padding:8px 12px;"
        + "border-radius:4px;font-family:ui-monospace,Menlo,monospace;"
        + "font-size:0.92em;overflow-x:auto;white-space:pre;"
    private static let inlineCodeStyle =
        "background:rgba(128,128,128,0.14);padding:1px 4px;"
        + "border-radius:3px;font-family:ui-monospace,Menlo,monospace;"
        + "font-size:0.92em;"
    private static let quoteStyle =
        "border-left:3px solid rgba(128,128,128,0.5);padding-left:12px;"
        + "margin-left:0;opacity:0.85;"
    private static let ruleStyle =
        "border:none;border-top:1px solid rgba(128,128,128,0.3);margin:1em 0;"
    private static let tableStyle =
        "border-collapse:collapse;margin:0.5em 0;"
    // 14px, not 10px: half an average character on each side, so the gutter
    // between two columns grows by a full character.
    private static let thStyle =
        "padding:6px 14px;border-bottom:1px solid rgba(128,128,128,0.3);"
    private static let tdStyle =
        "padding:6px 14px;border-bottom:1px solid rgba(128,128,128,0.12);"
    // Lighter than the row rules so the grid reads as columns first;
    // border-collapse on the table merges it with border-bottom.
    private static let colDividerStyle =
        "border-right:1px solid rgba(128,128,128,0.20);"
    private static let mathStyle =
        "text-align:center;font-style:italic;margin:0.8em 0;"
    private static let rowHeaderStyle = "background:rgba(128,128,128,0.14);"
    private static let rowShadeStyle = "background:rgba(128,128,128,0.07);"
    private static let listStyleTight = "margin:0.2em 0;padding-left:1.5em;"
    private static let listStyleLoose = "margin:0.5em 0;padding-left:1.5em;"
    private static let imagePlaceholderStyle =
        "color:rgba(128,128,128,0.7);font-style:italic;"
}
