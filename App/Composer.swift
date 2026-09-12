import Chat
import SwiftUI

struct FocusRequest: Equatable {
    private(set) var serial = 0
    private(set) var focus = false

    mutating func request(_ on: Bool) {
        serial += 1
        focus = on
    }
}

struct Composer: View {

    @Bindable var model: ChatModel
    @Binding var focus: FocusRequest
    @Binding var editing: Bool
    @ScaledMetric(relativeTo: .body) private var baseControl: CGFloat = 15
    @ScaledMetric(relativeTo: .body) private var baseLabel: CGFloat = 13
    @ScaledMetric(relativeTo: .body) private var baseSlot: CGFloat = 22
    @ScaledMetric(relativeTo: .body) private var baseSend: CGFloat = 24

    @Environment(\.horizontalSizeClass) private var sizeClass
    private var touch: CGFloat { sizeClass == .compact ? 1.35 : 1.0 }

    private var scale: CGFloat { touch * model.textScale }

    private var controlSize: CGFloat { baseControl * scale }
    private var labelSize: CGFloat { baseLabel * scale }
    private var slotSize: CGFloat { baseSlot * scale }
    private var sendSize: CGFloat { baseSend * scale }

    @ScaledMetric(relativeTo: .body) private var editorType: CGFloat = 1
    private var editorScale: CGFloat { editorType * model.textScale }

    private var editorFont: Font {
        .system(size: PromptEditor.points(editorScale))
    }

    private static let maxLines = 10
    private static let caveat = "Chat is AI and can make mistakes."

    var body: some View {
        VStack(spacing: 6) {
            card
            footnote
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 6)
        .onAppear { if !isOS { focus.request(true) } }
    }

    private var card: some View {
        VStack(spacing: 8) {
            ForEach(model.attachedImages) { img in imageChip(img) }
            ForEach(model.attachedClips) { clip in clipChip(clip) }
            ForEach(model.attachedDocs) { doc in docChip(doc) }
            if let warning = model.attachmentWarning { warningBanner(warning) }
            PromptEditor(text: $model.input, editing: $editing,
                         caret: $model.caret, focus: focus,
                         disabled: isOS && model.listening,
                         minLines: 2, maxLines: Composer.maxLines,
                         scale: editorScale, hasHint: hinting,
                         onSubmit: submitReturn,
                         onAcceptHint: acceptHint,
                         onBeginEditing: { model.speech.stopSpeaking() },
                         onPasteLarge: { text, at in
                             model.attachPastedText(text, at: at)
                         },
                         onDropFiles: { model.handleDrop($0, at: model.caret) })
                .overlay(alignment: .topLeading) { ghost }
                .onChange(of: model.input) { _, _ in
                    model.reconcileAttachments()
                }
            controls
        }
        .padding(10)
        .background {
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(.quinary)
                FilmStrip(model: model, quiet: !composing)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(.separator, lineWidth: 0.5)
        }
    }

    private var hinting: Bool {
        model.input.isEmpty && !model.followupHint.isEmpty && !model.busy
    }

    @ViewBuilder
    private var ghost: some View {
        if hinting {
            Text(model.followupHint)
                .font(editorFont)
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .padding(.top, 2)
                .allowsHitTesting(false)
        } else if model.input.isEmpty {
            Text("Write a message…")
                .font(editorFont)
                .foregroundStyle(.tertiary)
                .padding(.top, 2)
                .allowsHitTesting(false)
        }
    }

    private func acceptHint() {
        model.acceptFollowupHint()
        focus.request(true)
    }

    private var composing: Bool {
        isOS ? editing : !model.input.isEmpty
    }

    private func submitReturn() {
        if model.canSend { model.send() }
    }

    private var inVoiceExchange: Bool {
        model.listening || model.speech.engaged || model.voiceReady
    }

    private var controls: some View {
        VStack(spacing: 8) {
            if inVoiceExchange { transport }
            standardControls
        }
    }

    private var showBigMic: Bool { model.listening || model.voiceReady }

    private var transport: some View {
        HStack(spacing: transportGap) {
            Spacer()
            if showBigMic {
                transportButton(
                    model.listening ? "microphone.fill" : "microphone",
                    model.listening ? "Stop and send" : "Speak",
                    micTint,
                    listening: model.listening,
                    action: model.voice)
            }
            if model.speech.paused {
                transportButton("play.fill", "Resume", .accentColor) {
                    model.speech.resume()
                }
                transportButton("stop.fill", "Stop speaking", .red) {
                    model.speech.stopSpeaking()
                }
            } else if model.speech.engaged {
                transportButton("pause.fill", "Pause", .accentColor) {
                    model.speech.pause()
                }
            }
            Spacer()
        }
        .padding(.top, 2)
    }

    private var micTint: Color {
        let tint: Color
        if model.listening {
            tint = .orange
        } else if model.voiceReady {
            tint = .green
        } else {
            tint = .accentColor
        }
        return tint
    }

    private var transportSize: CGFloat { isOS ? slotSize * 2.6 : slotSize * 1.5 }
    private var transportGap: CGFloat { isOS ? 28 : 14 }

    private func transportButton(_ symbol: String, _ label: String,
                                 _ tint: Color, listening: Bool = false,
                                 action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: transportSize * 0.4, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: transportSize, height: transportSize)
                .background(tint, in: Circle())
                .background {
                    if listening {
                        HeardRing(size: transportSize,
                                  level: model.speechLevel,
                                  hearing: model.hearingSpeech)
                    }
                }
        }
        .buttonStyle(.plain)
        .help(label)
        .accessibilityLabel(label)
    }

    private var standardControls: some View {
        HStack(spacing: 8) {
            AttachButton(model: model)
            accessButton
            thinkingButton
            Spacer()
            if model.speech.available { speakerButton }
            micButton
            sendButton
        }
        .font(.system(size: controlSize))
    }

    private var speakerButton: some View {
        let on = model.speech.enabled
        let live = model.speech.speaking
        let tip = on ? "Replies are spoken" : "Speak replies"
        return Button { model.speech.enabled.toggle() } label: {
            Image(systemName: live ? "speaker.wave.2.fill"
                                   : (on ? "speaker.wave.2" : "speaker.slash"))
                .foregroundStyle(on ? Color.accentColor : .secondary)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .help(tip)
    }

    private var micButton: some View {
        Button(action: model.voice) {
            Image(systemName: model.listening
                  ? "microphone.fill" : "microphone")
                .foregroundStyle(model.listening ? Color.orange : .secondary)
                .frame(width: slotSize, height: slotSize)
                .symbolEffect(.pulse, isActive: model.listening)
        }
        .buttonStyle(.plain)
        .disabled(!model.canAttachAudio || (model.busy && !model.listening))
        .help(model.listening ? "Stop and send" : "Speak")
    }

    private var accessButton: some View {
        let icon: String
        let color: Color
        switch model.accessState {
        case .offline: icon = "airplane"; color = .orange
        case .wikipedia: icon = "books.vertical"; color = .teal
        case .full: icon = "globe"; color = .accentColor
        }
        return Button(action: model.cycleAccess) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .help("Web access")
    }

    private var thinkingButton: some View {
        let on = model.thinkingActive
        let tip: String
        if model.modelSupportsThinking {
            tip = on ? "Thinking on" : "Thinking off"
        } else {
            tip = Models.display(model.modelName)
                + " answers directly; it does not think step by step"
        }
        return Button(action: model.toggleThinking) {
            Image(systemName: on ? "lightbulb.fill" : "lightbulb")
                .foregroundStyle(on ? Color.yellow : Color.secondary)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .disabled(!model.modelSupportsThinking)
        .help(tip)
    }

    private func reserved(_ symbol: String,
                          _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .frame(width: slotSize, height: slotSize)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.tertiary)
        .disabled(true)
        .help("Coming soon")
    }

    private func imageChip(_ img: ImageAttachment) -> some View {
        HStack(spacing: 6) {
            thumb(img.thumbnail, or: "photo")
            Text(img.name)
            chipFile(img.file)
            Spacer()
            Button { model.clearImage(img.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .appFont(.caption)
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func thumb(_ cg: CGImage?, or symbol: String) -> some View {
        if let cg {
            Image(decorative: cg, scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 24, height: 24)
                .clipShape(RoundedRectangle(cornerRadius: 4))
        } else {
            Image(systemName: symbol).frame(width: 24, height: 24)
        }
    }

    @ViewBuilder
    private func chipFile(_ file: String) -> some View {
        if !file.isEmpty {
            Text(file)
                .foregroundStyle(.tertiary)
                .lineLimit(1).truncationMode(.middle)
        }
    }

    private func warningBanner(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(text).lineLimit(2)
            Spacer()
        }
        .appFont(.caption)
        .foregroundStyle(.orange)
    }

    private func clipChip(_ clip: ClipAttachment) -> some View {
        HStack(spacing: 6) {
            thumb(clip.thumbnail, or: clip.isVideo ? "film" : "waveform")
            Text(clip.name)
            chipFile(clip.file)
            Spacer()
            Button { model.clearClip(clip.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .appFont(.caption)
        .foregroundStyle(.secondary)
    }

    private func docChip(_ doc: Doc) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text").frame(width: 24, height: 24)
            Text(doc.name).lineLimit(1).truncationMode(.middle)
            Spacer()
            Button { model.clearDoc(doc.id) } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
        }
        .appFont(.caption)
        .foregroundStyle(.secondary)
    }

    // No .keyboardShortcut(.defaultAction): it would register a second
    // Return handler racing PromptEditor's.

    private var sendButton: some View {
        Button(action: fire) {
            Image(systemName: primaryIcon)
                .font(.system(size: labelSize, weight: .semibold))
                .frame(width: sendSize, height: sendSize)
                .foregroundStyle(sendLive ? .white : .secondary)
                .background(sendBackground,
                            in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .disabled(!sendLive)
        .help(primaryHelp)
    }

    private var primaryIcon: String {
        model.busy ? "stop.fill" : "arrow.up"
    }

    private var primaryHelp: String {
        model.busy ? "Stop" : "Send"
    }

    private func fire() {
        if model.busy {
            model.stop()
        } else {
            if isOS { focus.request(false) }
            model.send()
        }
    }

    private var sendLive: Bool {
        model.busy || model.canSend
    }

    private var sendBackground: Color {
        sendLive ? .accentColor : Color.secondary.opacity(0.18)
    }

    private var footnote: some View {
        let quiet = model.listening || model.speech.speaking
        return Text(noteText)
            .appFont(.caption2)
            .foregroundStyle(quiet ? .secondary : .tertiary)
            .frame(maxWidth: .infinity)
            // The colour fades; the STRING must not, or the animation
            // cross-fades the old sentence over the new one.
            .contentTransition(.identity)
            .animation(.easeInOut(duration: 0.2), value: quiet)
    }

    private var noteText: String {
        let text: String
        if model.listening {
            text = listeningNote
        } else if model.speech.paused {
            text = "Paused"
        } else if model.speech.speaking {
            text = "Speaking…"
        } else {
            text = plainFootnote
        }
        return text
    }

    private var listeningNote: String {
        model.heardSeconds > 0.05
            ? String(format: "%@…  heard %.1fs", model.thinkStatus,
                     model.heardSeconds)
            : model.thinkStatus + "…"
    }

    private var plainFootnote: String {
        var text = model.typing || isOS
            ? Composer.caveat
            : "Shift+Return for a new line.  " + Composer.caveat
        if hinting {
            text = isOS ? "Swipe right to ask this"
                        : "Tab or \u{2192} to ask this"
        }
        return text
    }

}
