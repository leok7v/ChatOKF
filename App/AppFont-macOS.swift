import AppKit
import SwiftUI

func appTextFont(_ style: Font.TextStyle, _ scale: CGFloat) -> Font {
    let base = NSFont.preferredFont(forTextStyle: appTextStyle(style))
    let sized = NSFont(descriptor: base.fontDescriptor,
                       size: base.pointSize * scale)
    return Font(sized ?? base)
}

func appTextPoints(_ style: Font.TextStyle) -> CGFloat {
    NSFont.preferredFont(forTextStyle: appTextStyle(style)).pointSize
}

private func appTextStyle(_ style: Font.TextStyle) -> NSFont.TextStyle {
    let result: NSFont.TextStyle
    switch style {
        case .largeTitle:  result = .largeTitle
        case .title:       result = .title1
        case .title2:      result = .title2
        case .title3:      result = .title3
        case .headline:    result = .headline
        case .subheadline: result = .subheadline
        case .body:        result = .body
        case .callout:     result = .callout
        case .footnote:    result = .footnote
        case .caption:     result = .caption1
        case .caption2:    result = .caption2
        default:           result = .body
    }
    return result
}
