import AppKit
import SwiftUI

struct SharedFile: Identifiable {
    let id = UUID()
    let url: URL
}

extension View {
    func shareSheet(_ file: Binding<SharedFile?>) -> some View {
        background {
            if let shared = file.wrappedValue {
                SharingPicker(url: shared.url) { file.wrappedValue = nil }
            }
        }
    }
}

private struct SharingPicker: NSViewRepresentable {

    let url: URL
    let onDone: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onDone: onDone) }

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let coordinator = context.coordinator
        let shared = url
        Task { @MainActor in
            let picker = NSSharingServicePicker(items: [shared])
            picker.delegate = coordinator
            coordinator.picker = picker
            picker.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
        }
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {}

    final class Coordinator: NSObject, NSSharingServicePickerDelegate {

        let onDone: () -> Void
        var picker: NSSharingServicePicker?

        init(onDone: @escaping () -> Void) { self.onDone = onDone }

        func sharingServicePicker(_ picker: NSSharingServicePicker,
                                  didChoose service: NSSharingService?) {
            onDone()
        }

    }

}
