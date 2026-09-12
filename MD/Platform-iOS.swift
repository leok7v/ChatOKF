#if os(iOS)
import SwiftUI
@_exported import UIKit

typealias PlatformFont = UIFont
typealias PlatformColor = UIColor
typealias PlatformImage = UIImage

func monoFont(at size: CGFloat) -> PlatformFont {
    UIFont.monospacedSystemFont(ofSize: size, weight: .regular)
}

func boldFont(of f: PlatformFont) -> PlatformFont {
    var traits = f.fontDescriptor.symbolicTraits
    traits.insert(.traitBold)
    var result = f
    if let d = f.fontDescriptor.withSymbolicTraits(traits) {
        result = UIFont(descriptor: d, size: f.pointSize)
    }
    return result
}

func platformResizedFont(_ f: PlatformFont, to size: CGFloat)
    -> PlatformFont {
    UIFont(descriptor: f.fontDescriptor, size: size)
}

func platformBoldItalicFont(of f: PlatformFont, bold: Bool,
                            italic: Bool) -> PlatformFont {
    var traits = f.fontDescriptor.symbolicTraits
    if bold { traits.insert(.traitBold) }
    if italic { traits.insert(.traitItalic) }
    var result = f
    if let d = f.fontDescriptor.withSymbolicTraits(traits) {
        result = UIFont(descriptor: d, size: f.pointSize)
    }
    return result
}

func platformMergeFontTraits(of source: PlatformFont,
                             into base: PlatformFont,
                             additionalBold: Bool) -> PlatformFont {
    var traits = source.fontDescriptor.symbolicTraits
    traits.formUnion(base.fontDescriptor.symbolicTraits)
    if additionalBold { traits.insert(.traitBold) }
    var result = base
    if let d = base.fontDescriptor.withSymbolicTraits(traits) {
        result = UIFont(descriptor: d, size: base.pointSize)
    }
    return result
}

let platformDefaultTextColor: PlatformColor = UIColor.label
let platformSecondaryColor: PlatformColor = UIColor.secondaryLabel
let platformClearColor: PlatformColor = UIColor.clear

func platformWhite(_ white: CGFloat, alpha: CGFloat) -> PlatformColor {
    UIColor(white: white, alpha: alpha)
}

func platformAdaptiveColor(light: PlatformColor,
                           dark: PlatformColor) -> PlatformColor {
    UIColor { traits in
        traits.userInterfaceStyle == .dark ? dark : light
    }
}

func platformDecodeImage(_ data: Data) -> Image? {
    let result: Image?
    if let ui = UIImage(data: data) { result = Image(uiImage: ui) }
    else { result = nil }
    return result
}

func platformDocumentImage(_ data: Data) -> PlatformImage? {
    UIImage(data: data)
}

func platformDecodeCGImage(_ data: Data) -> CGImage? {
    UIImage(data: data)?.cgImage
}

func platformSetClipboardString(_ s: String) {
    UIPasteboard.general.string = s
}

func platformPerformLightAppearance(_ body: () -> Void) {
    let light = UITraitCollection(userInterfaceStyle: .light)
    light.performAsCurrent(body)
}
#endif
