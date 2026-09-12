import AppKit
import SwiftUI

struct PromptEditor: NSViewRepresentable {

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
        NSFont.preferredFont(forTextStyle: .body).pointSize * scale
    }

    private var font: NSFont {
        NSFont.systemFont(ofSize: PromptEditor.points(scale))
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = SubmitTextView()
        let coordinator = context.coordinator
        tv.delegate = coordinator
        tv.onSubmit = { coordinator.parent.onSubmit() }
        tv.onAcceptHint = { coordinator.parent.onAcceptHint() }
        tv.onDropFiles = { coordinator.parent.onDropFiles($0) }
        tv.onFocus = { [weak coordinator] on in coordinator?.report(on) }
        tv.onWindow = { [weak coordinator] in
            Task { @MainActor in coordinator?.serve() }
        }
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.font = font
        tv.textContainerInset = NSSize(width: 0, height: 2)
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 0
        tv.string = text

        let scroll = NSScrollView()
        scroll.documentView = tv
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = false
        scroll.autohidesScrollers = true
        coordinator.textView = tv
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        if let tv = scroll.documentView as? SubmitTextView {
            if tv.string != text {
                tv.string = text
                let len = (text as NSString).length
                tv.setSelectedRange(
                    NSRange(location: max(0, min(caret, len)), length: 0))
            }
            if tv.font != font { tv.font = font }
            tv.hasHint = hasHint
            tv.isEditable = !disabled
            tv.isSelectable = !disabled
            if focus.serial != context.coordinator.served {
                Task { @MainActor [coordinator = context.coordinator] in
                    coordinator.serve()
                }
            }
        }
    }

    static func dismantleNSView(_ scroll: NSScrollView,
                                coordinator: Coordinator) {
        Task { @MainActor in coordinator.report(false) }
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSScrollView,
                      context: Context) -> CGSize? {
        let tv = nsView.documentView as? SubmitTextView
        let font = tv?.font ?? self.font
        let width = proposal.width ?? 300
        let inset = (tv?.textContainerInset.height ?? 2) * 2
        let line = ceil(font.ascender - font.descender + font.leading)
        let lo = ceil(CGFloat(minLines) * line) + inset
        let hi = ceil(CGFloat(maxLines) * line) + inset
        let measured = context.coordinator.height(
            of: tv?.string ?? "", width: max(1, width), font: font,
            atMost: hi - inset)
        return CGSize(width: width,
                      height: min(max(ceil(measured) + inset, lo), hi))
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {

        var parent: PromptEditor
        weak var textView: SubmitTextView?
        private(set) var served: Int

        init(_ parent: PromptEditor) {
            self.parent = parent
            served = parent.focus.serial
        }

        func serve() {
            if let tv = textView, let window = tv.window {
                let fresh = parent.focus.serial != served
                served = parent.focus.serial
                if fresh, parent.focus.focus, window.firstResponder !== tv {
                    window.makeFirstResponder(tv)
                } else if fresh, !parent.focus.focus,
                          window.firstResponder === tv {
                    window.makeFirstResponder(nil)
                }
            }
        }

        func report(_ on: Bool) {
            if parent.editing != on { parent.editing = on }
        }

        private struct Measured {
            let text: String
            let width: CGFloat
            let size: CGFloat
            let height: CGFloat
        }

        private var last: Measured?

        func height(of text: String, width: CGFloat, font: NSFont,
                    atMost ceiling: CGFloat) -> CGFloat {
            let size = font.pointSize
            var result = last?.height ?? 0
            if last?.text != text || last?.width != width
                || last?.size != size {
                result = Coordinator.measure(text, width: width, font: font,
                                             atMost: ceiling)
                last = Measured(text: text, width: width, size: size,
                                height: result)
            }
            return result
        }

        private static func measure(_ text: String, width: CGFloat,
                                    font: NSFont,
                                    atMost ceiling: CGFloat) -> CGFloat {
            let probe = String(text.prefix(overflowProbeLength))
            var result = boundingHeight(probe, width: width, font: font)
            if result < ceiling, text.count > overflowProbeLength {
                result = boundingHeight(text, width: width, font: font)
            }
            return result
        }

        // boundingRect drops a trailing empty line; the space keeps it.

        private static func boundingHeight(_ text: String, width: CGFloat,
                                           font: NSFont) -> CGFloat {
            let box = NSSize(width: width, height: .greatestFiniteMagnitude)
            let options: NSString.DrawingOptions = [.usesLineFragmentOrigin,
                                                    .usesFontLeading]
            let attributes: [NSAttributedString.Key: Any] = [.font: font]
            return ((text + " ") as NSString).boundingRect(
                with: box, options: options, attributes: attributes).height
        }

        private static let overflowProbeLength = 2048

        func textDidChange(_ notification: Notification) {
            if let tv = textView, parent.text != tv.string {
                parent.text = tv.string
            }
        }

        func textView(_ tv: NSTextView, shouldChangeTextIn range: NSRange,
                      replacementString text: String?) -> Bool {
            let spilled = (text?.utf8.count ?? 0) >= ChatModel.pasteAttachBytes
                && parent.onPasteLarge(text ?? "", range.location)
            return !spilled
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            if let tv = textView {
                let loc = tv.selectedRange().location
                if parent.caret != loc { parent.caret = loc }
            }
        }

        func textDidBeginEditing(_ notification: Notification) {
            parent.onBeginEditing()
        }

    }

}

final class SubmitTextView: NSTextView {
    var onSubmit: (() -> Void)?
    var onAcceptHint: (() -> Void)?
    var onFocus: ((Bool) -> Void)?
    var onWindow: (() -> Void)?
    var hasHint = false
    // NSTextView takes file drops through its built-in text-drag machinery,
    // which registerForDraggedTypes does not reach.
    var onDropFiles: (([URL]) -> Void)?

    override func becomeFirstResponder() -> Bool {
        let did = super.becomeFirstResponder()
        if did { onFocus?(true) }
        return did
    }

    override func resignFirstResponder() -> Bool {
        let did = super.resignFirstResponder()
        if did { onFocus?(false) }
        return did
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { onWindow?() }
    }

    private func droppedFileURLs(_ sender: NSDraggingInfo) -> [URL] {
        (sender.draggingPasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFileURLs(sender).isEmpty
            ? super.draggingEntered(sender) : .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        droppedFileURLs(sender).isEmpty
            ? super.draggingUpdated(sender) : .copy
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = droppedFileURLs(sender)
        let handled = !urls.isEmpty
        if handled { onDropFiles?(urls) }
        return handled ? true : super.performDragOperation(sender)
    }

    override func doCommand(by selector: Selector) {
        let isReturn = selector == #selector(NSResponder.insertNewline(_:))
        let shift = NSApp.currentEvent?.modifierFlags.contains(.shift) ?? false
        let takesHint = selector == #selector(NSResponder.insertTab(_:))
            || selector == #selector(NSResponder.moveRight(_:))
        var handled = false
        if isReturn && !shift {
            onSubmit?()
            handled = true
        } else if takesHint, hasHint, string.isEmpty {
            onAcceptHint?()
            handled = true
        }
        if !handled { super.doCommand(by: selector) }
    }

}
