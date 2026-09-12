import SwiftUI

private struct AppTextScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

extension EnvironmentValues {
    var appTextScale: CGFloat {
        get { self[AppTextScaleKey.self] }
        set { self[AppTextScaleKey.self] = newValue }
    }
}

extension View {
    func appFont(_ style: Font.TextStyle) -> some View {
        modifier(AppFontModifier(style: style))
    }
}

struct AppFontModifier: ViewModifier {

    @Environment(\.appTextScale) private var scale
    let style: Font.TextStyle

    func body(content: Content) -> some View {
        content.font(appTextFont(style, scale))
    }
}
