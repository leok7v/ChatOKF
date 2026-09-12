import SwiftUI

// GFM table: natural widths when they fit, else redistributed and the
// cells WRAP; sideways scrolling only before the width is known.
struct TableBlock: View {

    let headers: [String]
    let rows: [[String]]
    let alignments: [Markdown.Alignment]
    let style: MarkdownStyle
    @State private var available: CGFloat = 0

    var body: some View {
        let h = headers.map { s in TableMetrics.normalize(s) }
        let r = rows.map { row in row.map { s in TableMetrics.normalize(s) } }
        let n = TableMetrics.columnCount(headers: h, rows: r)
        let layout = columnLayout(n, headers: h, rows: r)
        let fitWidth: CGFloat? = layout.wrap ? max(0, available - 16) : nil
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                if !h.isEmpty {
                    rowView(h, bold: true, shade: style.tableHeaderShade,
                            n: n, widths: layout.widths, wrap: layout.wrap)
                    Divider()
                }
                ForEach(Array(r.enumerated()), id: \.offset) { pair in
                    rowView(pair.element, bold: false,
                            shade: pair.offset % 2 == 1
                                ? style.tableRowShade : Color.clear,
                            n: n, widths: layout.widths, wrap: layout.wrap)
                }
            }
            .padding(8)
            .frame(width: fitWidth, alignment: .leading)
        }
        .background(RoundedRectangle(cornerRadius: style.cornerRadius)
            .fill(style.codeBackground))
        .onGeometryChange(for: CGFloat.self, of: { proxy in
            proxy.size.width
        }, action: { w in
            if w > 0, w != available { available = w }
        })
        .overlay(alignment: .topTrailing) {
            CopyButton(text: TableMetrics.serializeMonospaced(
                headers: h, rows: r), style: style).padding(6)
        }
    }

    // Natural widths when they fit; else fit-to-available floored at each
    // column's longest word. nil until the first geometry pass lands.
    private func columnLayout(_ n: Int, headers h: [String],
                              rows r: [[String]])
        -> (widths: [CGFloat]?, wrap: Bool) {
        var result: (widths: [CGFloat]?, wrap: Bool) = (nil, false)
        let usable = available - CGFloat(max(n - 1, 0)) * 12 - 16
        if available > 0, n > 0 {
            let fit = TableMetrics.columnLayout(
                headers: h, rows: r,
                natural: naturalWidths(n, headers: h, rows: r),
                minimums: minimumWidths(n, headers: h, rows: r),
                available: usable)
            if !fit.widths.isEmpty { result = (fit.widths, fit.wrap) }
        }
        return result
    }

    private func rowView(_ cells: [String], bold: Bool, shade: Color,
                         n: Int, widths: [CGFloat]?,
                         wrap: Bool) -> some View {
        let fill: CGFloat? = wrap ? .infinity : nil
        return HStack(alignment: .top, spacing: 12) {
            ForEach(0..<n, id: \.self) { i in
                cell(i < cells.count ? cells[i] : "", bold: bold,
                     width: widths?[i], wrap: wrap, align: alignment(i))
            }
        }
        .frame(maxWidth: fill, alignment: .leading)
        .padding(.vertical, 4)
        .background(shade)
    }

    @ViewBuilder
    private func cell(_ text: String, bold: Bool, width: CGFloat?,
                      wrap: Bool, align: Alignment) -> some View {
        let parsed = Markdown.parse(text)
        if let first = parsed.items.first,
           case .image(let alt, let url, let w, let h) = first.block {
            ImageBlockCell(alt: alt, url: url, width: w, height: h,
                           style: style)
                .frame(width: width, alignment: align)
        } else if let width {
            SelectableText(cellText(text, parsed: parsed),
                           font: FontRole.body(style.bodySize).platformFont,
                           nowrap: !wrap,
                           selectable: style.selectable, bold: bold)
                .frame(width: width, alignment: align)
        } else {
            SelectableText(cellText(text, parsed: parsed),
                           font: FontRole.body(style.bodySize).platformFont,
                           nowrap: true,
                           selectable: style.selectable, bold: bold)
                .fixedSize(horizontal: true, vertical: false)
        }
    }

    private func cellText(_ raw: String,
                          parsed: Markdown.Document) -> AttributedString {
        var result = AttributedString(raw)
        if let first = parsed.items.first,
           case .paragraph(let a) = first.block {
            result = a
        }
        return result
    }

    private func alignment(_ col: Int) -> Alignment {
        let a = col < alignments.count ? alignments[col] : .none
        let result: Alignment
        switch a {
            case .center: result = .center
            case .right: result = .trailing
            default: result = .leading
        }
        return result
    }

    private func naturalWidths(_ n: Int, headers h: [String],
                               rows r: [[String]]) -> [CGFloat] {
        let body: [NSAttributedString.Key: Any] = [
            .font: FontRole.body(style.bodySize).platformFont]
        let bold: [NSAttributedString.Key: Any] = [
            .font: boldFont(of: FontRole.body(style.bodySize).platformFont)]
        var widths = [CGFloat](repeating: 0, count: n)
        for c in 0..<n {
            var maxW: CGFloat = 0
            if c < h.count {
                let s = (h[c] as NSString).size(withAttributes: bold).width
                if s > maxW { maxW = s }
            }
            for row in r where c < row.count {
                let s = (row[c] as NSString).size(withAttributes: body).width
                if s > maxW { maxW = s }
            }
            widths[c] = ceil(maxW) + 6
        }
        return widths
    }

    private func minimumWidths(_ n: Int, headers h: [String],
                               rows r: [[String]]) -> [CGFloat] {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: FontRole.body(style.bodySize).platformFont]
        var widths = [CGFloat](repeating: 0, count: n)
        for c in 0..<n {
            let word = TableMetrics.longestWord(headers: h, rows: r, col: c)
            widths[c] = ceil((word as NSString)
                .size(withAttributes: attrs).width) + 6
        }
        return widths
    }
}

// A table-cell image reuses the block image loader at a bounded size.
private struct ImageBlockCell: View {
    let alt: String
    let url: URL
    let width: CGFloat?
    let height: CGFloat?
    let style: MarkdownStyle
    @State private var image: Image?

    var body: some View {
        Group {
            if let image {
                image.resizable().scaledToFit()
                    .frame(maxWidth: width ?? 120,
                           maxHeight: height ?? 120, alignment: .leading)
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(style.secondaryColor)
            }
        }
        .task(id: url) { await load() }
    }

    private func load() async {
        var req = URLRequest(url: url)
        req.setValue(ImagePrefetch.userAgent, forHTTPHeaderField: "User-Agent")
        let data = try? await URLSession.shared.data(for: req).0
        if let data, let img = platformDecodeImage(data) { image = img }
    }
}
