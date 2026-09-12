import Foundation
import CoreGraphics

// Explicit width / height win; otherwise the intrinsic size is scaled by
// `defaultScale` and clamped to `maxWidth`, preserving aspect.
func aspectFit(intrinsicWidth iw: CGFloat, intrinsicHeight ih: CGFloat,
               explicitWidth: CGFloat? = nil,
               explicitHeight: CGFloat? = nil,
               defaultScale: CGFloat = 1.0,
               maxWidth: CGFloat = .greatestFiniteMagnitude)
    -> (width: CGFloat, height: CGFloat) {
    var result: (width: CGFloat, height: CGFloat) = (0, 0)
    if iw > 0, ih > 0 {
        let aspect = iw / ih
        var w: CGFloat
        var h: CGFloat
        if let ew = explicitWidth, let eh = explicitHeight {
            w = ew
            h = eh
        } else if let ew = explicitWidth {
            w = ew
            h = ew / aspect
        } else if let eh = explicitHeight {
            h = eh
            w = eh * aspect
        } else {
            w = min(maxWidth, iw * defaultScale)
            h = w / aspect
        }
        if w > maxWidth {
            w = maxWidth
            h = w / aspect
        }
        result = (w, h)
    }
    return result
}

enum ImagePrefetch {

    static let userAgent = "MD/1.0 (https://github.com/leok7v/ChatOKF)"

    static func collectURLs(in document: Markdown.Document) -> Set<URL> {
        var urls: Set<URL> = []
        for item in document.items { collect(item.block, into: &urls) }
        return urls
    }

    private static func collect(_ block: Markdown.Block,
                                into urls: inout Set<URL>) {
        switch block {
            case .image(_, let u, _, _):
                urls.insert(u)
            case .quote(let inner):
                for b in inner { collect(b, into: &urls) }
            case .list(let items, _):
                for item in items {
                    for b in item.blocks { collect(b, into: &urls) }
                }
            case .table(_, let rows, _):
                for row in rows {
                    for cell in row {
                        if let info = imageInCell(cell) { urls.insert(info.0) }
                    }
                }
            default:
                break
        }
    }

    // A table cell that is a lone image, decomposed to (url, width, height).
    static func imageInCell(_ cell: String) -> (URL, CGFloat?, CGFloat?)? {
        var result: (URL, CGFloat?, CGFloat?)? = nil
        let parsed = Markdown.parse(cell)
        if let first = parsed.items.first,
           case .image(_, let url, let w, let h) = first.block {
            result = (url, w, h)
        }
        return result
    }

    static func fetch(_ document: Markdown.Document) async -> [URL: Data] {
        await fetch(collectURLs(in: document))
    }

    static func fetch(_ urls: Set<URL>) async -> [URL: Data] {
        await withTaskGroup(of: (URL, Data?).self) { group in
            for u in urls {
                group.addTask {
                    var req = URLRequest(url: u)
                    req.setValue(userAgent,
                                 forHTTPHeaderField: "User-Agent")
                    let data = try? await URLSession.shared.data(for: req).0
                    return (u, data)
                }
            }
            var result: [URL: Data] = [:]
            for await (u, d) in group where d != nil { result[u] = d }
            return result
        }
    }

    static func fetchAndDecode<T>(in document: Markdown.Document,
                                  decode: @Sendable (Data) -> T?)
        async -> [URL: T] {
        await fetch(document).compactMapValues(decode)
    }
}

public enum MarkdownImages {
    public static func fetch(_ document: Markdown.Document)
        async -> [URL: Data] {
        await ImagePrefetch.fetch(document)
    }
}
