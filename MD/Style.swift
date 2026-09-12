import SwiftUI

// The App-facing look. Fonts, colors, spacing, and the code / math toggles, so
// a host can match its own surface.
public struct MarkdownStyle: Sendable {

    public var bodySize: CGFloat
    public var codeSize: CGFloat
    public var headingSizes: [CGFloat]      // index 0 -> h1 ... 5 -> h6

    public var blockSpacing: CGFloat
    public var paragraphSpacing: CGFloat
    public var listIndent: CGFloat
    public var codePadding: CGFloat
    public var cornerRadius: CGFloat

    public var textColor: Color
    public var secondaryColor: Color
    public var linkColor: Color
    public var codeBackground: Color
    public var quoteBar: Color
    public var ruleColor: Color
    public var tableHeaderShade: Color
    public var tableRowShade: Color

    public var highlightCode: Bool
    public var renderMath: Bool
    public var selectable: Bool

    public init(
        bodySize: CGFloat = 15,
        codeSize: CGFloat = 14,
        headingSizes: [CGFloat] = [28, 22, 18, 16, 14, 13],
        blockSpacing: CGFloat = 10,
        paragraphSpacing: CGFloat = 4,
        listIndent: CGFloat = 20,
        codePadding: CGFloat = 10,
        cornerRadius: CGFloat = 6,
        textColor: Color = .primary,
        secondaryColor: Color = .secondary,
        linkColor: Color = .accentColor,
        codeBackground: Color = Color.secondary.opacity(0.12),
        quoteBar: Color = Color.secondary.opacity(0.5),
        ruleColor: Color = Color.secondary.opacity(0.4),
        tableHeaderShade: Color = Color.primary.opacity(0.07),
        tableRowShade: Color = Color.primary.opacity(0.04),
        highlightCode: Bool = true,
        renderMath: Bool = true,
        selectable: Bool = true
    ) {
        self.bodySize = bodySize
        self.codeSize = codeSize
        self.headingSizes = headingSizes
        self.blockSpacing = blockSpacing
        self.paragraphSpacing = paragraphSpacing
        self.listIndent = listIndent
        self.codePadding = codePadding
        self.cornerRadius = cornerRadius
        self.textColor = textColor
        self.secondaryColor = secondaryColor
        self.linkColor = linkColor
        self.codeBackground = codeBackground
        self.quoteBar = quoteBar
        self.ruleColor = ruleColor
        self.tableHeaderShade = tableHeaderShade
        self.tableRowShade = tableRowShade
        self.highlightCode = highlightCode
        self.renderMath = renderMath
        self.selectable = selectable
    }

    public static let `default` = MarkdownStyle()

    // Clamp to the six heading levels; anything past h6 keeps the h6 size.
    public func headingSize(_ level: Int) -> CGFloat {
        let i = min(max(level, 1), 6) - 1
        return headingSizes[i]
    }
}
