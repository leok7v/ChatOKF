import SwiftUI

struct DoneButton: View {

    let action: () -> Void

    var body: some View {
        Button("Done", action: action)
            .modifier(TintedOnPhone())
            .keyboardShortcut(.defaultAction)
    }

}

// The TOOLBAR draws the item's capsule, so no button style reshapes it;
// only sharedBackgroundVisibility(.hidden) removes it.
private struct TintedOnPhone: ViewModifier {

    @ViewBuilder
    func body(content: Content) -> some View {
        if isOS {
            content.buttonStyle(.plain).foregroundStyle(Color.accentColor)
        } else {
            content
        }
    }

}

struct DoneToolbar: ToolbarContent {

    let action: () -> Void

    var body: some ToolbarContent {
        if #available(iOS 26.0, macOS 26.0, *) {
            ToolbarItem(placement: .confirmationAction) {
                DoneButton(action: action)
            }
            .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(placement: .confirmationAction) {
                DoneButton(action: action)
            }
        }
    }

}

struct EscapeToClose: View {

    let action: () -> Void

    var body: some View {
        Button("", action: action)
            .keyboardShortcut(.cancelAction)
            .hidden()
    }

}
