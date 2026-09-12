import SwiftUI
import UIKit

// Scale multiplies on top of this device's Dynamic Type; never add it twice.

func appTextFont(_ style: Font.TextStyle, _ scale: CGFloat) -> Font {
    let base = UIFont.preferredFont(forTextStyle: appTextStyle(style))
    return Font(base.withSize(base.pointSize * scale))
}

func appTextPoints(_ style: Font.TextStyle) -> CGFloat {
    UIFont.preferredFont(forTextStyle: appTextStyle(style)).pointSize
}

private func appTextStyle(_ style: Font.TextStyle) -> UIFont.TextStyle {
    let result: UIFont.TextStyle
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
