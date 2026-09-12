import AppKit
import SwiftUI

struct OptionKeyMonitor: ViewModifier {

    @Binding var down: Bool
    @State private var monitor: Any?

    func body(content: Content) -> some View {
        content
            .onAppear {
                if monitor == nil {
                    monitor = NSEvent.addLocalMonitorForEvents(
                        matching: .flagsChanged) { event in
                        MainActor.assumeIsolated {
                            down = event.modifierFlags.contains(.option)
                        }
                        return event
                    }
                }
            }
            .onDisappear {
                if let m = monitor { NSEvent.removeMonitor(m); monitor = nil }
            }
    }

}
