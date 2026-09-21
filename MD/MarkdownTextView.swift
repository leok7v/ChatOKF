import SwiftUI

// The whole document as one selectable native text surface. Drag-selection
// snaps around atomic code / table / image units (see Bridges-macOS).
public struct MarkdownTextView: View {

    let document: Markdown.Document
    let style: MarkdownStyle
    let find: MarkdownFindController?
    // The message id this surface registers under, so Find spans every bubble.
    let findId: UUID?
    // scrolls: fill the frame and scroll internally, or self-size inside a
    // chat bubble the transcript scrolls.
    let scrolls: Bool
    // The sentence being read aloud right now, tinted where it appears.
    let speaking: String?
    @State private var images: [URL: PlatformImage] = [:]
    @State private var available: CGFloat = 0

    public init(_ document: Markdown.Document,
                style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil,
                findId: UUID? = nil,
                scrolls: Bool = true,
                speaking: String? = nil) {
        self.document = document
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public init(_ source: String, style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil,
                findId: UUID? = nil,
                scrolls: Bool = true,
                speaking: String? = nil) {
        self.document = Markdown.parse(source, math: style.renderMath)
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public var body: some View {
        let need = DocumentText.minimumWidth(of: document, style: style,
                                             formulas: false)
        return surface(width: max(available, need),
                       sideways: available > 0 && need > available)
            .onGeometryChange(for: CGFloat.self, of: { proxy in
                proxy.size.width
            }, action: { w in
                if w > 0, w != available { available = w }
            })
            // Keyed on the image URLs: keying on the streaming document would
            // restart the fetch on every token.
            .task(id: ImagePrefetch.collectURLs(in: document)) {
                images = await ImagePrefetch.fetchAndDecode(
                    in: document, decode: { data in
                        platformDocumentImage(data)
                    })
            }
    }

    @ViewBuilder
    private func surface(width: CGFloat, sideways: Bool) -> some View {
        if sideways {
            ScrollView(.horizontal) {
                text(wide: true).frame(width: width, alignment: .leading)
            }
        } else {
            text(wide: false)
        }
    }

    private func text(wide: Bool) -> some View {
        SelectableText(
            ns: DocumentText.attributed(from: document, style: style,
                                        images: images, width: available,
                                        wide: wide),
            font: FontRole.body(style.bodySize).platformFont,
            selectable: style.selectable, scrolls: scrolls, find: find,
            findId: findId, speaking: speaking)
    }

    // The narrowest this document can be drawn before a table is asked for less
    // room than its widest token needs.
    @MainActor
    public static func minimumWidth(of document: Markdown.Document,
                                    style: MarkdownStyle = .default)
        -> CGFloat {
        DocumentText.minimumWidth(of: document, style: style)
    }
}

// The raw text on the SAME findable / selectable single surface as
// MarkdownTextView, but WITHOUT parsing -- markers like ** stay literal.
public struct PlainTextView: View {

    let text: String
    let style: MarkdownStyle
    let find: MarkdownFindController?
    let findId: UUID?
    let scrolls: Bool
    let speaking: String?

    public init(_ text: String, style: MarkdownStyle = .default,
                find: MarkdownFindController? = nil, findId: UUID? = nil,
                scrolls: Bool = false, speaking: String? = nil) {
        self.text = text
        self.style = style
        self.find = find
        self.findId = findId
        self.scrolls = scrolls
        self.speaking = speaking
    }

    public var body: some View {
        let font = FontRole.body(style.bodySize).platformFont
        return SelectableText(
            ns: NSAttributedString(string: text,
                attributes: [.font: font,
                             .foregroundColor: platformDefaultTextColor]),
            font: font, selectable: style.selectable,
            scrolls: scrolls, find: find, findId: findId,
            speaking: speaking)
    }
}
