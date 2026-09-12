#if os(macOS)
import SwiftUI
@_exported import AppKit

typealias PlatformFont = NSFont
typealias PlatformColor = NSColor
typealias PlatformImage = NSImage

func monoFont(at size: CGFloat) -> PlatformFont {
    NSFont.monospacedSystemFont(ofSize: size, weight: .regular)
}

func boldFont(of f: PlatformFont) -> PlatformFont {
    var traits = f.fontDescriptor.symbolicTraits
    traits.insert(.bold)
    let d = f.fontDescriptor.withSymbolicTraits(traits)
    return NSFont(descriptor: d, size: f.pointSize) ?? f
}

func platformResizedFont(_ f: PlatformFont, to size: CGFloat)
    -> PlatformFont {
    NSFont(descriptor: f.fontDescriptor, size: size) ?? f
}

func platformBoldItalicFont(of f: PlatformFont, bold: Bool,
                            italic: Bool) -> PlatformFont {
    var traits = f.fontDescriptor.symbolicTraits
    if bold { traits.insert(.bold) }
    if italic { traits.insert(.italic) }
    let d = f.fontDescriptor.withSymbolicTraits(traits)
    return NSFont(descriptor: d, size: f.pointSize) ?? f
}

func platformMergeFontTraits(of source: PlatformFont,
                             into base: PlatformFont,
                             additionalBold: Bool) -> PlatformFont {
    var traits = source.fontDescriptor.symbolicTraits
    traits.formUnion(base.fontDescriptor.symbolicTraits)
    if additionalBold { traits.insert(.bold) }
    let d = base.fontDescriptor.withSymbolicTraits(traits)
    return NSFont(descriptor: d, size: base.pointSize) ?? base
}

let platformDefaultTextColor: PlatformColor = NSColor.textColor
let platformSecondaryColor: PlatformColor = NSColor.secondaryLabelColor
let platformClearColor: PlatformColor = NSColor.clear

func platformWhite(_ white: CGFloat, alpha: CGFloat) -> PlatformColor {
    NSColor(white: white, alpha: alpha)
}

// A single dynamic color that resolves per drawing appearance, so a code
// theme baked from hex stays correct under a light / dark switch.
func platformAdaptiveColor(light: PlatformColor,
                           dark: PlatformColor) -> PlatformColor {
    NSColor(name: nil) { appearance in
        let darkNames: [NSAppearance.Name] = [
            .darkAqua, .vibrantDark,
            .accessibilityHighContrastDarkAqua,
            .accessibilityHighContrastVibrantDark,
        ]
        let isDark = appearance.bestMatch(from: darkNames) != nil
        return isDark ? dark : light
    }
}

func platformDecodeImage(_ data: Data) -> Image? {
    let result: Image?
    if let ns = NSImage(data: data) { result = Image(nsImage: ns) }
    else { result = nil }
    return result
}

func platformDocumentImage(_ data: Data) -> PlatformImage? {
    NSImage(data: data)
}

func platformDecodeCGImage(_ data: Data) -> CGImage? {
    NSImage(data: data)?
        .cgImage(forProposedRect: nil, context: nil, hints: nil)
}

func platformSetClipboardString(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

// PDF is rendered in a forced-light appearance so an app in dark mode
// still produces a light document.
func platformPerformLightAppearance(_ body: () -> Void) {
    if let aqua = NSAppearance(named: .aqua) {
        aqua.performAsCurrentDrawingAppearance(body)
    } else {
        body()
    }
}
#endif
