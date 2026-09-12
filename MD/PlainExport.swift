import Foundation

// Round-trips a document back to Markdown-ish plain text. Adapted from
// md.too `src/PlainExport.swift`, operating on the block model.
extension Markdown {
    public static func plainText(_ document: Document) -> String {
        PlainExport.render(document.items.map { item in item.block })
    }
}

enum PlainExport {

    static func render(_ blocks: [Markdown.Block]) -> String {
        var out = ""
        for (i, b) in blocks.enumerated() {
            out += renderBlock(b)
            if i < blocks.count - 1 { out += "\n" }
        }
        return out
    }

    private static func renderBlock(_ block: Markdown.Block) -> String {
        let result: String
        switch block {
            case .heading(let level, let text):
                result = String(repeating: "#", count: level)
                    + " \(plain(text))\n"
            case .paragraph(let text):
                result = "\(plain(text))\n"
            case .code(_, let text):
                result = text + "\n"
            case .quote(let inner):
                result = quote(inner)
            case .list(let items, _):
                result = renderList(items)
            case .table(let h, let rows, _):
                result = TableMetrics.serializeMonospaced(headers: h,
                                                          rows: rows)
            case .math(let tex):
                // The source, not the rendering, so a display pastes back
                // as the markdown it was written in.
                result = "$$\n\(tex)\n$$\n"
            case .rule:
                result = "---\n"
            case .image(let alt, let url, _, _):
                result = "![\(alt)](\(url.absoluteString))\n"
        }
        return result
    }

    private static func quote(_ inner: [Markdown.Block]) -> String {
        let body = render(inner)
        let lines = body.split(separator: "\n",
                               omittingEmptySubsequences: false)
        return lines.map { line in "> \(line)" }.joined(separator: "\n")
            + "\n"
    }

    private static func renderList(_ items: [Markdown.ListItem]) -> String {
        var out = ""
        for item in items {
            var mark = item.marker
            if let c = item.checked { mark = c ? "[x]" : "[ ]" }
            let inner = render(item.blocks)
            let lines = inner.split(separator: "\n",
                                    omittingEmptySubsequences: false)
                .map(String.init)
            if let first = lines.first {
                out += "\(mark) \(first)\n"
                for rest in lines.dropFirst() where !rest.isEmpty {
                    out += "  \(rest)\n"
                }
            }
        }
        return out
    }

    // Plain text has no baseline to offset, so a script run spends the
    // Unicode the TeX renderer already keeps tables of.
    private static func plain(_ a: AttributedString) -> String {
        var out = ""
        for run in a.runs {
            let segment = String(a[run.range].characters)
            if let level = run[ScriptAttribute.self] {
                out += TeX.unicodeScript(segment, superscript: level > 0)
            } else {
                out += segment
            }
        }
        return out
    }
}
