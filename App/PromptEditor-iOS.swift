import SwiftUI
import UIKit

struct PromptEditor: UIViewRepresentable {

    @Binding var text: String
    @Binding var editing: Bool
    @Binding var caret: Int
    var focus: FocusRequest
    var disabled: Bool
    var minLines: Int
    var maxLines: Int
    var scale: CGFloat = 1
    var hasHint = false
    var onSubmit: () -> Void
    var onAcceptHint: () -> Void = { }
    var onBeginEditing: () -> Void = { }
    var onPasteLarge: (String, Int) -> Bool = { _, _ in false }
    var onDropFiles: ([URL]) -> Void = { _ in }

    static func points(_ scale: CGFloat) -> CGFloat {
        UIFont.labelFontSize * scale
    }

    private var font: UIFont {
        UIFont.systemFont(ofSize: PromptEditor.points(scale))
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> PromptTextView {
        let tv = PromptTextView()
        tv.delegate = context.coordinator
        tv.textDropDelegate = context.coordinator
        tv.onLeaveWindow = { [weak coordinator = context.coordinator] in
            coordinator?.report(false)
        }
        tv.font = font
        tv.backgroundColor = .clear
        tv.textContainerInset = UIEdgeInsets(
            top: 2, left: 0, bottom: 2, right: 0)
        tv.textContainer.lineFragmentPadding = 0
        tv.isScrollEnabled = true
        tv.text = text
        let swipe = UISwipeGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.takeHint))
        swipe.direction = .right
        tv.addGestureRecognizer(swipe)
        context.coordinator.textView = tv
        return tv
    }

    func updateUIView(_ tv: PromptTextView, context: Context) {
        context.coordinator.parent = self
        if tv.font != font { tv.font = font }
        if tv.text != text {
            tv.text = text
            let len = (text as NSString).length
            let loc = max(0, min(caret, len))
            if let pos = tv.position(from: tv.beginningOfDocument,
                                     offset: loc) {
                tv.selectedTextRange = tv.textRange(from: pos, to: pos)
            }
        }
        // Editable and first-responder changes re-enter SwiftUI's responder
        // graph, so applying them here cycles the AttributeGraph.
        if tv.isEditable == disabled
            || focus.serial != context.coordinator.served {
            Task { @MainActor [coordinator = context.coordinator] in
                coordinator.serve()
            }
        }
    }

    static func dismantleUIView(_ tv: PromptTextView,
                                coordinator: Coordinator) {
        Task { @MainActor in coordinator.report(false) }
    }

    func sizeThatFits(_ proposal: ProposedViewSize, uiView: PromptTextView,
                      context: Context) -> CGSize? {
        let font = uiView.font ?? self.font
        let width = proposal.width ?? 300
        let inset = uiView.textContainerInset.top +
            uiView.textContainerInset.bottom
        let line = font.lineHeight
        let fit = uiView.sizeThatFits(
            CGSize(width: width, height: .greatestFiniteMagnitude)).height
        let lo = ceil(CGFloat(minLines) * line) + inset
        let hi = ceil(CGFloat(maxLines) * line) + inset
        return CGSize(width: width, height: min(max(fit, lo), hi))
    }

    @MainActor final class Coordinator: NSObject, UITextViewDelegate,
                                        UITextDropDelegate {
        var parent: PromptEditor
        weak var textView: PromptTextView?
        private(set) var served: Int

        init(_ parent: PromptEditor) {
            self.parent = parent
            served = parent.focus.serial
        }

        func serve() {
            if let tv = textView {
                let editable = !parent.disabled
                let fresh = parent.focus.serial != served
                served = parent.focus.serial
                if tv.isEditable != editable { tv.isEditable = editable }
                if !editable || (fresh && !parent.focus.focus) {
                    if tv.isFirstResponder { tv.resignFirstResponder() }
                } else if fresh, parent.focus.focus, !tv.isFirstResponder {
                    tv.becomeFirstResponder()
                }
            }
        }

        func report(_ on: Bool) {
            if parent.editing != on { parent.editing = on }
        }

        func textViewDidChange(_ tv: UITextView) {
            if parent.text != tv.text { parent.text = tv.text }
        }

        func textView(_ tv: UITextView, shouldChangeTextIn range: NSRange,
                      replacementText text: String) -> Bool {
            let spilled = text.utf8.count >= ChatModel.pasteAttachBytes
                && parent.onPasteLarge(text, range.location)
            return !spilled
        }

        func textViewDidChangeSelection(_ tv: UITextView) {
            if let range = tv.selectedTextRange {
                let loc = tv.offset(from: tv.beginningOfDocument,
                                    to: range.start)
                if parent.caret != loc { parent.caret = loc }
            }
        }

        @objc func takeHint() {
            if parent.hasHint { parent.onAcceptHint() }
        }

        func textViewDidBeginEditing(_ tv: UITextView) {
            report(true)
            parent.onBeginEditing()
        }

        func textViewDidEndEditing(_ tv: UITextView) {
            report(false)
        }

        func textView(_ textView: UITextView,
                      willBecomeEditableForDrop drop: UITextDroppable)
            -> UITextDropEditability {
            .no
        }
    }

}

final class PromptTextView: UITextView {

    var onLeaveWindow: (() -> Void)?

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window == nil { onLeaveWindow?() }
    }

}
