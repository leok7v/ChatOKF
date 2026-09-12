import SwiftUI
import UIKit

struct RenameField: UIViewRepresentable {

    @Binding var text: String
    let onSubmit: () -> Void
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> SelectingTextField {
        let field = SelectingTextField()
        field.delegate = context.coordinator
        field.text = text
        field.font = UIFont.preferredFont(forTextStyle: .body)
        field.returnKeyType = .done
        field.clearButtonMode = .whileEditing
        field.autocorrectionType = .no
        field.addTarget(context.coordinator,
                        action: #selector(Coordinator.changed(_:)),
                        for: .editingChanged)
        return field
    }

    func updateUIView(_ field: SelectingTextField, context: Context) {
        context.coordinator.parent = self
        if field.text != text { field.text = text }
    }

    @MainActor final class Coordinator: NSObject, UITextFieldDelegate {

        var parent: RenameField

        init(_ parent: RenameField) { self.parent = parent }

        @objc func changed(_ field: UITextField) {
            parent.text = field.text ?? ""
        }

        func textFieldShouldReturn(_ field: UITextField) -> Bool {
            parent.onSubmit()
            return false
        }
    }
}

final class SelectingTextField: UITextField {

    private var claimed = false

    // A representable whose inputs never change is never updated again, so
    // updateUIView cannot claim focus.
    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil, !claimed {
            claimed = true
            Task { @MainActor in
                self.becomeFirstResponder()
                self.selectAll(nil)
            }
        }
    }

}
