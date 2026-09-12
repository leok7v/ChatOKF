import Chat
import LLM
import SwiftUI

struct SettingsView: View {

    @Bindable var model: ChatModel
    let onClose: () -> Void
    @State private var deleteName: String?
    @State private var confirmReset = false
    @State private var confirmClear = false
    @ScaledMetric(relativeTo: .body) private var railWidth: CGFloat = 215
    @ScaledMetric(relativeTo: .body) private var railIcon: CGFloat = 20
    @State private var draftZoom: Int?
    @State private var gateRevision = 0
    @State private var searchRevision = 0
    @State private var path: [Category] = []

    private var notch: Int { draftZoom ?? model.textZoom }
    private var draftScale: CGFloat { 1 + CGFloat(notch) / 10 }

    private func dismiss() {
        if let draftZoom { model.textZoom = draftZoom }
        onClose()
    }

    enum Category: String, CaseIterable, Identifiable {
        case systemPrompt = "System Prompt"
        case models = "Models"
        case voice = "Voice"
        case view = "View"
        case intelligence = "Intelligence"
        case privacy = "Privacy"
        case misc = "Misc"
        case diagnostics = "Diagnostics"
        case about = "About"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .systemPrompt: return "text.bubble"
            case .models: return "internaldrive"
            case .voice: return "waveform"
            case .view: return "paintbrush"
            case .intelligence: return "brain"
            case .privacy: return "hand.raised"
            case .misc: return "slider.horizontal.3"
            case .diagnostics: return "stethoscope"
            case .about: return "info.circle"
            }
        }
    }

    var body: some View {
        Group {
            if isOS {
                compactBody
            } else {
                regularBody
            }
        }
        .modifier(OptionKeyMonitor(down: $model.optionDown))
        .alert("Factory reset?", isPresented: $confirmReset) {
            Button("Proceed", role: .destructive) { model.factoryReset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This deletes every downloaded model, every conversation "
               + "including the trash, every attachment kept with them, and "
               + "all settings, then quits. The next launch starts like a "
               + "fresh install.")
        }
        .alert("Clear all conversations?", isPresented: $confirmClear) {
            Button("Clear", role: .destructive) {
                model.clearAllConversations()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Moves every saved conversation to the trash, where it "
               + "stays for 30 days.")
        }
    }

    private var compactBody: some View {
        NavigationStack(path: $path) {
            List {
                ForEach(categories) { item in
                    NavigationLink(value: item) {
                        Label(item.rawValue, systemImage: item.symbol)
                    }
                }
            }
            .navigationDestination(for: Category.self) { item in
                paneScroll(item).navigationTitle(item.rawValue)
            }
            .navigationTitle("Settings")
            .toolbar { DoneToolbar(action: dismiss) }
        }
        .onAppear {
            if model.settingsCategory != .systemPrompt {
                path = [model.settingsCategory]
            }
        }
    }

    private var regularBody: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Settings", systemImage: "gearshape")
                    .appFont(.headline)
                Spacer()
                DoneButton(action: dismiss)
                EscapeToClose(action: dismiss)
            }
            .padding(12)
            Divider()
            HStack(spacing: 0) {
                rail
                Divider()
                paneScroll(category)
            }
        }
    }

    private var categories: [Category] {
        Category.allCases.filter { c in
            (c != .models || Models.all.count > 1)
                && (c != .voice || model.speech.available)
                && (c != .diagnostics || model.statusLine)
        }
    }

    private var rail: some View {
        let current = category
        return VStack(alignment: .leading, spacing: 4) {
            ForEach(categories) { item in railRow(item, current) }
            Spacer()
        }
        .padding(12)
        .frame(width: min(railWidth, 300) * model.textScale)
    }

    private func railRow(_ item: Category,
                         _ current: Category) -> some View {
        let selected = current == item
        return Button {
            model.settingsCategory = item
        } label: {
            HStack(spacing: 8) {
                Image(systemName: item.symbol).frame(width: railIcon)
                Text(item.rawValue)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(selected ? Color.accentColor : .clear,
                        in: RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(selected ? Color.white : Color.primary)
    }

    private func paneScroll(_ item: Category) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                pane(item)
            }
            .padding(20)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func pane(_ item: Category) -> some View {
        switch item {
        case .systemPrompt: systemPromptPane
        case .models: modelsPane
        case .voice: voicePane
        case .view: viewPane
        case .intelligence: intelligencePane
        case .privacy: privacyPane
        case .misc: miscPane
        case .diagnostics: diagnosticsPane
        case .about: aboutPane
        }
    }

    private var cardFill: some ShapeStyle { Color.primary.opacity(0.055) }

    private func card<Content: View>(
        @ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 0) { content() }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(cardFill, in: RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(.separator.opacity(0.6), lineWidth: 0.5)
            }
    }

    private var hairline: some View {
        Divider().padding(.leading, 14)
    }

    private func heading(_ text: String) -> some View {
        Text(text)
            .appFont(.callout)
            .bold()
            .foregroundStyle(.secondary)
            .padding(.leading, 4)
            .padding(.top, 2)
    }

    private func row<Control: View>(
        _ label: String, _ detail: String? = nil,
        @ViewBuilder control: () -> Control) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 12) {
                Text(label)
                Spacer(minLength: 12)
                control()
            }
            if let detail, !detail.isEmpty { explain(detail) }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func wideRow<Control: View>(
        _ label: String?, _ detail: String? = nil,
        @ViewBuilder control: () -> Control) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let label { Text(label) }
            control()
            if let detail, !detail.isEmpty { explain(detail) }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func switchRow(_ label: String, _ detail: String,
                           _ isOn: Binding<Bool>) -> some View {
        row(label, detail) {
            Toggle("", isOn: isOn)
                .labelsHidden()
                .toggleStyle(.switch)
        }
    }

    private func note(_ text: String) -> some View {
        explain(text)
            .padding(.horizontal, 4)
    }

    private var aboutPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("About")
            card { creditRow(Credits.app) }
            note("Built on the work below. Each entry says who made it, "
                + "the terms it arrives under, and what this app changed.")
            card {
                ForEach(Credits.all) { item in
                    if item.id != Credits.all.first?.id { hairline }
                    creditRow(item)
                }
            }
            heading("Licences")
            card {
                licence("Apache License 2.0", Licence.apache2)
                hairline
                licence("GNU General Public License v3", Licence.gpl3)
            }
        }
    }

    private func creditRow(_ item: Credit) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(item.name).bold()
                Text(item.terms)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(item.author).appFont(.callout)
            if let link = item.link {
                Link(item.source, destination: link).appFont(.caption)
            }
            explain(item.changed)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func licence(_ name: String, _ text: String) -> some View {
        DisclosureGroup(name) {
            ScrollView {
                Text(text)
                    .appFont(.caption2)
                    .monospaced()
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
            }
            .frame(maxHeight: 260)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
    }

    private var voicePane: some View {
        @Bindable var speech = model.speech
        return VStack(alignment: .leading, spacing: 18) {
            title("Voice")
            note("Replies are read aloud on this device; nothing is sent "
                + "anywhere. Tap a voice to hear it. Opening the microphone "
                + "stops the reading, so you can interrupt at any time.")
            card {
                wideRow("Speak", speech.mode.detail) {
                    Picker("Speak", selection: $speech.mode) {
                        ForEach(VoiceSession.Mode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                hairline
                row("Speed") {
                    HStack(spacing: 10) {
                        Slider(value: $speech.speed, in: 0.7...1.5, step: 0.05)
                            .frame(maxWidth: 220)
                        Text(String(format: "%.2fx", speech.speed))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            heading("Voices")
            card {
                ForEach(Speech.voices) { v in
                    if v.id != Speech.voices.first?.id { hairline }
                    voiceRow(v)
                }
            }
        }
    }

    private func voiceRow(_ v: SpeechVoice) -> some View {
        let picked = v.name == model.speech.voiceName
        return Button {
            model.speech.voiceName = v.name
            model.speech.preview(v)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(picked ? Color.accentColor : .secondary)
                Text(v.name)
                Spacer()
                Image(systemName: "play.circle")
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var systemPromptPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("System Prompt")
            note("Instructions the assistant reads before every "
                + "conversation, for tone and behaviour. Applies from the "
                + "next new chat.")
            TextEditor(text: $model.systemPrompt)
                .appFont(.body)
                .frame(minHeight: 160)
                .padding(6)
                .background(cardFill, in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(.separator.opacity(0.6), lineWidth: 0.5)
                }
            Button("Reset to default") {
                model.systemPrompt = ChatModel.defaultSystemPrompt
            }
            .disabled(model.systemPrompt == ChatModel.defaultSystemPrompt)
        }
    }

    private var category: Category {
        categories.contains(model.settingsCategory)
            ? model.settingsCategory : .systemPrompt
    }

    private var unlocked: Bool { model.unlocked }

    private var listed: [String] { Models.offered(unlocked: unlocked) }

    private var modelsPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Models")
            note("Each model downloads once and stays on this device. "
                + "Deleting one frees the space, and it can be downloaded "
                + "again. The model in use cannot be deleted. To delete it, "
                + "switch to another model first.")
            card {
                ForEach(listed, id: \.self) { name in
                    if name != listed.first { hairline }
                    modelRow(name)
                }
            }
            .id(model.diskRevision)
            if unlocked {
                note("Every model this Mac can run, shown while Option is "
                    + "held with Debug on. Normally only the ones its memory "
                    + "allows are offered.")
            }
        }
        .alert("Delete \(Models.display(deleteName ?? "", among: listed))?",
               isPresented: Binding(
            get: { deleteName != nil },
            set: { shown in if !shown { deleteName = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let name = deleteName { model.deleteModel(name) }
                deleteName = nil
            }
            Button("Cancel", role: .cancel) { deleteName = nil }
        } message: {
            Text("Frees the space it uses; you can download it again later.")
        }
    }

    private func modelRow(_ name: String) -> some View {
        let downloaded = model.isDownloaded(name)
        let active = name == model.modelName
        let bytes = ModelCatalog.source(name)?.bytes ?? 0
        return HStack(spacing: 8) {
            Text(Models.display(name, among: listed))
            if active {
                Image(systemName: "checkmark.circle.fill")
                    .appFont(.caption)
                    .foregroundStyle(Color.accentColor)
            }
            Spacer()
            Text(ByteCountFormatter.string(fromByteCount: bytes,
                                           countStyle: .file))
                .appFont(.caption)
                .foregroundStyle(.secondary)
            rowButton(name, downloaded: downloaded, active: active)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }

    private func rowButton(_ name: String, downloaded: Bool,
                           active: Bool) -> some View {
        let idle = !model.busy && !model.downloading
        return Group {
            if downloaded {
                Button { deleteName = name } label: {
                    Image(systemName: "trash")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.plain)
                .foregroundStyle(idle && !active
                                 ? Color.red : Color.secondary.opacity(0.4))
                .disabled(active || !idle)
                .help(active ? "Switch to another model first"
                             : "Delete from this device")
            } else {
                Button { model.requestDownload(name) } label: {
                    Image(systemName: "arrow.down.circle")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.plain)
                .foregroundStyle(idle ? Color.accentColor : Color.secondary)
                .disabled(!idle)
                .help("Download and switch to this model")
            }
        }
    }

    private var imageRows: some View {
        wideRow("Images",
                "How closely the assistant looks at a picture: about "
                + "\(model.imageBudget.tokens) tokens per picture, at its "
                + "own shape. Larger sees finer detail and takes longer.") {
            Picker("Image detail", selection: $model.imageBudget) {
                ForEach(ChatModel.ImageBudget.allCases) { size in
                    Text(size.rawValue).tag(size)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    private var documentRows: some View {
        wideRow("Documents",
                "How much of an attached document the assistant reads: up "
                + "to about \(model.docBudget.pages) pages of text. Anything "
                + "past that is left out, and the chat says so.") {
            Picker("Document size", selection: $model.docBudget) {
                ForEach(ChatModel.DocBudget.allCases) { size in
                    Text(size.rawValue).tag(size)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    private var viewPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("View")
            card {
                switchRow("Markdown",
                          "Show replies with headings, lists, code and "
                          + "tables.", $model.renderMarkdown)
                hairline
                switchRow("Always show sample prompts",
                          "Keep the examples on an empty chat after you have "
                          + "tried them all.", $model.alwaysShowSamples)
            }
            heading("Zoom")
            card { textSizeRow }
        }
    }

    private var textSizeRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Text("a").font(.system(size: 11))
                VStack(spacing: 3) {
                    Slider(value: zoomNotch,
                           in: -Double(ChatModel.zoomLimit)
                               ... Double(ChatModel.zoomLimit),
                           step: 1)
                    detents
                }
                .frame(maxWidth: 260)
                Text("A").font(.system(size: 21))
                Spacer()
                Button("Reset Zoom") { draftZoom = 0 }
                    .disabled(notch == 0)
            }
            draftSample
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var zoomNotch: Binding<Double> {
        Binding(get: { Double(notch) },
                set: { value in
                    draftZoom = ChatModel.clampZoom(Int(value))
                })
    }

    private var detents: some View {
        HStack(spacing: 0) {
            ForEach(-ChatModel.zoomLimit ... ChatModel.zoomLimit,
                    id: \.self) { notch in
                if notch > -ChatModel.zoomLimit { Spacer(minLength: 0) }
                Capsule()
                    .fill(notch == 0 ? Color.accentColor : Color.secondary)
                    .frame(width: notch == 0 ? 2 : 1,
                           height: notch == 0 ? 9 : 5)
            }
        }
        .padding(.horizontal, isOS ? 13 : 5)
        .allowsHitTesting(false)
    }

    private var draftSample: some View {
        Text("Text and controls throughout the app, on top of your "
            + "device's own text size.")
            .font(appTextFont(.callout, draftScale))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var intelligencePane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Intelligence")
            if model.modelSupportsThinking {
                card { thinkingRows }
            }
            card {
                if model.canSuggestFollowups {
                    switchRow("Suggest follow-up questions",
                              "Suggest a sensible next question as a hint in "
                              + "the message box. "
                              + (isOS ? "Swipe right to ask it."
                                      : "Tab or the right arrow asks it."),
                              $model.suggestFollowups)
                    hairline
                }
                switchRow("Wikipedia",
                          "Let the assistant look things up in Wikipedia. "
                          + "The lookup runs locally on your device, so what "
                          + "you type stays here. Only the matching article "
                          + "is downloaded.", Binding(
                    get: { model.wikipedia },
                    set: { on in
                        model.setAccess(wikipedia: on, web: model.webAccess)
                    }
                ))
                hairline
                switchRow("Web Access",
                          "Let the assistant search, fetch and read the "
                          + "Internet. Your search terms leave the device, "
                          + "and the weather uses your rough location.",
                          Binding(
                    get: { model.webAccess },
                    set: { on in
                        model.setAccess(wikipedia: model.wikipedia, web: on)
                    }
                ))
            }
            heading("Attachments")
            card {
                if model.canAttachImages {
                    imageRows
                    hairline
                }
                documentRows
            }
        }
    }

    private var thinkingRows: some View {
        VStack(spacing: 0) {
            switchRow("Thinking",
                      "Think before answering. Slower, but better on hard "
                      + "questions. The thinking shows above each answer, "
                      + "and Quick Answer cuts it short.", Binding(
                get: { model.thinkingActive },
                set: { on in
                    if on != model.thinking { model.toggleThinking() }
                }
            ))
            hairline
            wideRow("Budget",
                    "How long to think before answering: about "
                    + "\(Int(model.thinkBudget.seconds)) seconds.") {
                Picker("Thinking budget", selection: $model.thinkBudget) {
                    ForEach(ChatModel.ThinkBudget.allCases) { size in
                        Text(size.rawValue).tag(size)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            if model.modelSupportsReasoningEffort {
                hairline
                reasoningEffortRow
            }
        }
    }

    private var privacyPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Privacy")
            note("What you type is answered on this device. These are "
                + "the only places anything leaves it, and only while the "
                + "tool that needs them is switched on.")
            heading("Web search")
            card {
                ForEach(SearchProvider.allCases) { provider in
                    if provider != SearchProvider.allCases.first { hairline }
                    providerRow(provider)
                }
            }
            .id(searchRevision)
            note("With both off, the assistant is not offered web search "
                + "at all.")
            heading("Also reached")
            card {
                destination("Wikipedia",
                    "The number of the article the model picked. Your "
                        + "question is matched on this device and never "
                        + "sent.",
                    "https://foundation.wikimedia.org/wiki/"
                        + "Policy:Privacy_policy")
                hairline
                destination("ipinfo.io",
                    "Your network address, to place the weather when you do "
                        + "not name a city.",
                    "https://ipinfo.io/privacy-policy")
                hairline
                destination("Open-Meteo and weather.gov",
                    "The coordinates a forecast is for, and nothing else.",
                    "https://open-meteo.com/en/terms")
                hairline
                destination("Pages you ask it to read",
                    "Whatever address you or the model hands to the page "
                        + "reader.", "")
                hairline
                destination("Hugging Face",
                    "Model downloads only. No conversation text.",
                    "https://huggingface.co/privacy")
            }
        }
    }

    private func providerRow(_ provider: SearchProvider) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                Text(provider.label)
                Spacer(minLength: 12)
                Toggle("", isOn: Binding(
                    get: { provider.on },
                    set: { on in
                        model.setSearchProvider(provider, on)
                        searchRevision += 1
                    }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
            explain(provider.detail)
            HStack(spacing: 14) {
                policyLink(provider.home, provider.home)
                policyLink("Privacy policy", provider.privacy)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func destination(_ name: String, _ what: String,
                             _ policy: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(name)
            explain(what)
            policyLink("Privacy policy", policy)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func policyLink(_ label: String, _ address: String) -> some View {
        if let url = URL(string: address), !address.isEmpty {
            Link(label, destination: url).appFont(.caption)
        }
    }

    private var miscPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Misc")
            card {
                switchRow("Confirm before deleting a conversation",
                          "Ask before a conversation is deleted. Off deletes "
                          + "at once.", $model.confirmDeleteConversation)
                hairline
                switchRow("Debug",
                          "Show a status bar under the message box with "
                          + "context, speed and memory, add a session details "
                          + "button to the chat actions, and reveal the "
                          + "Diagnostics pane. Nothing is written to a log "
                          + "file while this is off, and logging starts at "
                          + "the next launch.", $model.statusLine)
            }
            if !ConversationStore.shared.list.isEmpty || unlocked {
                card {
                    if !ConversationStore.shared.list.isEmpty {
                        wideRow(nil,
                                "Move every saved conversation to the "
                                + "trash.") {
                            Button(role: .destructive) {
                                confirmClear = true
                            } label: {
                                Label("Clear all conversations",
                                      systemImage: "trash")
                            }
                            .disabled(model.busy)
                        }
                    }
                    if unlocked {
                        if !ConversationStore.shared.list.isEmpty { hairline }
                        wideRow(nil,
                                "Delete every downloaded model, every "
                                + "conversation and all settings, then "
                                + "quit.") {
                            Button(role: .destructive) {
                                confirmReset = true
                            } label: {
                                Label("Factory Reset", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
    }

    private var reasoningEffortRow: some View {
        wideRow("Reasoning Effort",
                "How hard to think. High thinks longer and more carefully "
                + "but can go in circles. Low answers sooner. Applies from "
                + "the next new chat.") {
            Picker("Reasoning effort", selection: Binding(
                get: { model.reasoningEffort },
                set: { level in model.setReasoningEffort(level) }
            )) {
                ForEach(ChatModel.ReasoningEffort.allCases) { level in
                    Text(level.rawValue).tag(level)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    private var diagnosticsPane: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Diagnostics")
            note("Extra detail for the diagnostics log. A switch that is "
                + "off costs nothing. The session details screen says where "
                + "the log is. A launch argument naming the same category "
                + "wins over these switches for that run.")
            card {
                ForEach(DiagGate.switchable) { gate in
                    if gate != DiagGate.switchable.first { hairline }
                    gateRow(gate)
                }
            }
            .id(gateRevision)
        }
    }

    private func gateRow(_ gate: DiagGate) -> some View {
        switchRow(gate.label, gate.detail, Binding(
            get: { gate.wanted },
            set: { on in
                gate.set(on)
                gateRevision += 1
            }
        ))
    }

    @ViewBuilder
    private func title(_ text: String) -> some View {
        if !isOS {
            Text(text).appFont(.title3).bold()
        }
    }

    private func explain(_ text: String) -> some View {
        Text(text)
            .appFont(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

}
