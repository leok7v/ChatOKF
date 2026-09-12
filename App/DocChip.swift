import Chat
import PDFKit
import QuickLook
import SwiftUI

struct DocChip: View {

    let doc: DocRef
    @State private var page: CGImage?
    @State private var previewing: URL?

    private var name: String { doc.url.lastPathComponent }

    private var present: Bool {
        FileManager.default.fileExists(atPath: doc.url.path)
    }

    var body: some View {
        Button {
            if present { previewing = doc.url }
        } label: {
            HStack(spacing: 10) {
                cover
                VStack(alignment: .leading, spacing: 1) {
                    Text(name)
                        .appFont(.callout)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(present ? size : "no longer on this device")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(8)
            .frame(maxWidth: 320)
            .background(.regularMaterial,
                        in: RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(.separator, lineWidth: 0.5)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!present)
        .quickLookPreview($previewing)
        .task { page = DocChip.firstPage(doc.url) }
    }

    private var size: String {
        let kb = Double(doc.bytes) / 1024
        let shown = kb < 1 ? String(format: "%.0f bytes", Double(doc.bytes))
                           : String(format: "%.1f KB", kb)
        return shown + " of text" + (doc.short ? ", truncated" : "")
    }

    @ViewBuilder
    private var cover: some View {
        if let page {
            Image(decorative: page, scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 30, height: 38)
                .background(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 3))
                .overlay {
                    RoundedRectangle(cornerRadius: 3)
                        .stroke(.separator, lineWidth: 0.5)
                }
        } else {
            Image(systemName: "doc.richtext")
                .appFont(.title3)
                .frame(width: 30, height: 38)
                .foregroundStyle(Color.accentColor)
        }
    }

    private static func firstPage(_ url: URL) -> CGImage? {
        var out: CGImage? = nil
        // `file` must stay bound: a page whose PDFDocument has been released
        // draws nothing, and says so at runtime rather than at compile time.
        if url.pathExtension.lowercased() == "pdf",
           let file = PDFDocument(url: url), let page = file.page(at: 0) {
            let box = page.bounds(for: .mediaBox)
            let scale = box.height > 0 ? 76 / box.height : 1
            let wide = max(Int((box.width * scale).rounded()), 1)
            let high = max(Int((box.height * scale).rounded()), 1)
            let ctx = CGContext(
                data: nil, width: wide, height: high, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            if let ctx {
                ctx.setFillColor(gray: 1, alpha: 1)
                ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(wide),
                                height: CGFloat(high)))
                ctx.scaleBy(x: scale, y: scale)
                ctx.translateBy(x: -box.origin.x, y: -box.origin.y)
                page.draw(with: .mediaBox, to: ctx)
                out = ctx.makeImage()
            }
        }
        return out
    }

}
