import AppKit
import SwiftUI

struct TitleBarClickMonitor: ViewModifier {

    let onClick: () -> Void
    @State private var monitor: Any?

    func body(content: Content) -> some View {
        content
            .onAppear {
                if monitor == nil {
                    monitor = NSEvent.addLocalMonitorForEvents(
                        matching: .leftMouseDown) { event in
                        MainActor.assumeIsolated {
                            if TitleBarClickMonitor.inTitleBar(event) {
                                onClick()
                            }
                        }
                        return event
                    }
                }
            }
            .onDisappear {
                if let m = monitor { NSEvent.removeMonitor(m); monitor = nil }
            }
    }

    private static func inTitleBar(_ event: NSEvent) -> Bool {
        var out = false
        if let window = event.window {
            out = event.locationInWindow.y > window.contentLayoutRect.maxY
        }
        return out
    }

}
