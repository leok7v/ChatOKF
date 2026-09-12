import AppKit
import Chat
import SwiftUI

@main struct ChatOKFApp: App {

    @State private var model = ChatModel()

    var body: some Scene {
        Window(Bundle.appName, id: "main") {
            ContentView(model: model).frame(minWidth: 800, minHeight: 600)
        }
        .commands { ViewCommands(model: model) }
    }

}

struct ViewCommands: Commands {

    @Bindable var model: ChatModel

    var body: some Commands {
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

@MainActor func quitApp() { NSApp.terminate(nil) }

func setClipboard(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

var isOS: Bool { return false }
