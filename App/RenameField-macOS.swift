import AppKit
import SwiftUI

struct RenameField: NSViewRepresentable {

    @Binding var text: String
    let onSubmit: () -> Void
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> SelectingTextField {
        let field = SelectingTextField(string: text)
        field.delegate = context.coordinator
        field.isBordered = false
        field.drawsBackground = false
        field.focusRingType = .none
        field.lineBreakMode = .byTruncatingTail
        field.font = NSFont.preferredFont(forTextStyle: .body)
        return field
    }

    func updateNSView(_ field: SelectingTextField, context: Context) {
        context.coordinator.parent = self
        if field.stringValue != text { field.stringValue = text }
    }

    @MainActor final class Coordinator: NSObject, NSTextFieldDelegate {

        var parent: RenameField

        init(_ parent: RenameField) { self.parent = parent }

        func controlTextDidChange(_ notification: Notification) {
            if let field = notification.object as? NSTextField {
                parent.text = field.stringValue
            }
        }

        func control(_ control: NSControl, textView: NSTextView,
                     doCommandBy selector: Selector) -> Bool {
            var handled = true
            if selector == #selector(NSResponder.insertNewline(_:)) {
                parent.onSubmit()
            } else if selector == #selector(NSResponder.cancelOperation(_:)) {
                parent.onCancel()
            } else {
                handled = false
            }
            return handled
        }
    }
}

final class SelectingTextField: NSTextField {

    private var claimed = false

    // A representable whose inputs never change is never updated again, so
    // updateNSView cannot claim focus.
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil, !claimed {
            claimed = true
            Task { @MainActor in
                self.window?.makeFirstResponder(self)
                self.selectText(nil)
            }
        }
    }

}
