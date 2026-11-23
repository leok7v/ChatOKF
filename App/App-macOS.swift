import AppKit
import Chat
import SwiftUI

@main struct ChatOKFApp: App {

    @State private var model = ChatModel()

    var body: some Scene {
        Window(Bundle.appName, id: "main") {
            ContentView(model: model).frame(minWidth: 800, minHeight: 600)
        }
        .defaultLaunchBehavior(.presented)
        .commands { ViewCommands(model: model) }
    }

}

struct ViewCommands: Commands {

    @Bindable var model: ChatModel

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About \(Bundle.appName)") { showAbout() }
        }
        CommandGroup(replacing: .appSettings) {
            Button {
                model.openSettings()
            } label: {
                Label("Settings\u{2026}", systemImage: "gearshape")
            }
            .keyboardShortcut(",", modifiers: .command)
        }
        CommandGroup(after: .toolbar) {
            Button { model.resetZoom() } label: {
                Label("Reset Zoom", systemImage: "1.magnifyingglass")
            }
            .keyboardShortcut("0", modifiers: .command)
            .disabled(model.atDefaultZoom)
            Button { model.zoomIn() } label: {
                Label("Zoom In", systemImage: "plus.magnifyingglass")
            }
            .keyboardShortcut("=", modifiers: .command)
            .disabled(!model.canZoomIn)
            Button { model.zoomOut() } label: {
                Label("Zoom Out", systemImage: "minus.magnifyingglass")
            }
            .keyboardShortcut("-", modifiers: .command)
            .disabled(!model.canZoomOut)
            Divider()
            Toggle(isOn: $model.statusLine) {
                Label("Debug",
                      systemImage: "rectangle.bottomthird.inset.filled")
            }
        }
    }

}

@MainActor func showAbout() {
    let line = "\u{00A9} Dmitry \u{201C}Leo\u{201D} Kuznetsov"
    let centred = NSMutableParagraphStyle()
    centred.alignment = .center
    let credits = NSAttributedString(string: line, attributes: [
        .link: URL(string: "https://leok7v.github.io/ChatOKF/") as Any,
        .font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
        .paragraphStyle: centred,
    ])
    NSApp.orderFrontStandardAboutPanel(options: [.credits: credits])
    NSApp.activate()
}

@MainActor func quitApp() { NSApp.terminate(nil) }

func setClipboard(_ plain: String, html: String? = nil) {
    NSPasteboard.general.clearContents()
    if let html { NSPasteboard.general.setString(html, forType: .html) }
    NSPasteboard.general.setString(plain, forType: .string)
}

var isOS: Bool { return false }

@MainActor func applyAppearance(_ scheme: ColorScheme?) {
    let appearance: NSAppearance?
    switch scheme {
    case .dark: appearance = NSAppearance(named: .darkAqua)
    case .light: appearance = NSAppearance(named: .aqua)
    default: appearance = nil
    }
    NSApp.appearance = appearance
    for window in NSApp.windows {
        window.appearance = appearance
        if let root = window.contentView { markDirty(root) }
        window.invalidateShadow()
        window.display()
    }
}

@MainActor private func markDirty(_ view: NSView) {
    view.needsLayout = true
    view.needsDisplay = true
    for child in view.subviews { markDirty(child) }
}
