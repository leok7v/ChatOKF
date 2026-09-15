import Chat
import LLM
import SwiftUI
import UIKit
import UniformTypeIdentifiers

final class AppDelegate: NSObject, UIApplicationDelegate {
    nonisolated(unsafe) static var completion: (() -> Void)?

    func application(
        _ app: UIApplication,
        didFinishLaunchingWithOptions opts:
            [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        Relay.idle = {
            DispatchQueue.main.async {
                AppDelegate.completion?()
                AppDelegate.completion = nil
            }
        }
        Relay.shared.start()
        return true
    }

    func application(_ app: UIApplication,
                     handleEventsForBackgroundURLSession id: String,
                     completionHandler: @escaping () -> Void) {
        AppDelegate.completion = completionHandler
        Relay.shared.start()
    }
}

@main struct ChatOKFApp: App {
    @State private var model = ChatModel()
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            // Never wrap this in a NavigationStack or NavigationSplitView:
            // nested navigation containers render black on iPhone.
            ContentView(model: model)
                .onChange(of: scenePhase) { _, phase in
                    BackgroundGate.shared.setBackgrounded(phase != .active)
                    if phase == .background { model.commitCurrent() }
                }
        }
    }
}

@MainActor func quitApp() { exit(0) }

func setClipboard(_ plain: String, html: String? = nil) {
    if let html {
        UIPasteboard.general.items = [[
            UTType.utf8PlainText.identifier: plain,
            UTType.html.identifier: html,
        ]]
    } else {
        UIPasteboard.general.string = plain
    }
}

var isOS: Bool { return true }
