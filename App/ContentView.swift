import AVKit
import Chat
import LLM
import MD
import SwiftUI

struct ContentView: View {

    @Bindable var model: ChatModel

    static let answerStyle = MarkdownStyle(
        bodySize: isOS ? 17 : 14,
        codeSize: isOS ? 13 : 12,
        headingSizes: isOS ? [25, 21, 19, 17, 17, 16]
                           : [22, 18, 16, 14, 14, 13])
    static let reasoningStyle = MarkdownStyle(
        bodySize: isOS ? 14 : 12,
        codeSize: isOS ? 12 : 10,
        headingSizes: isOS ? [18, 16, 15, 14, 14, 14]
                           : [16, 14, 13, 12, 12, 12],
        textColor: Color(white: 0.53),
        secondaryColor: Color(white: 0.6),
        highlightCode: false)
    @ScaledMetric(relativeTo: .body) private var typeScale: CGFloat = 1

    private var textScale: CGFloat { typeScale * model.textScale }

    static func scaled(_ s: MarkdownStyle, by k: CGFloat) -> MarkdownStyle {
        var out = s
        out.bodySize = s.bodySize * k
        out.codeSize = s.codeSize * k
        out.headingSizes = s.headingSizes.map { size in size * k }
        out.blockSpacing = s.blockSpacing * k
        out.paragraphSpacing = s.paragraphSpacing * k
        out.listIndent = s.listIndent * k
        return out
    }
    @State private var promptFocus = FocusRequest()
    @State private var promptEditing = false
    @State private var chatHeight: CGFloat = 0
    @State private var peek = ToolPeek()
    @State private var calloutSize: CGSize = .zero
    @ScaledMetric(relativeTo: .body) private var popScale: CGFloat = 1
    @State private var follow = true
    @State private var tailInset: CGFloat = 0
    @State private var tailCollapse: Task<Void, Never>?
    @ScaledMetric(relativeTo: .body) private var tailInsetSize: CGFloat = 64
    @State private var showModelInvite = false
    @State private var dropActive = false
    @State private var pinchFrom: Int?
    @State private var pinchShown: Int?

    private var pinchZoom: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let from = pinchFrom ?? model.textZoom
                if pinchFrom == nil { pinchFrom = from }
                let want = ChatModel.zoomScale(from) * value.magnification
                let notch = ChatModel.zoomNotch(nearest: want)
                if notch != pinchShown {
                    pinchShown = notch
                    model.flashZoom(notch)
                }
            }
            .onEnded { value in
                let from = pinchFrom ?? model.textZoom
                let want = ChatModel.zoomScale(from) * value.magnification
                model.textZoom = ChatModel.zoomNotch(nearest: want)
                pinchFrom = nil
                pinchShown = nil
            }
    }
    @State private var downloadingPhrase = Whimsical.current(.downloading)
    @State private var sidebarOpen = false
    @State private var renaming: ConversationStore.Convo?
    @State private var findActive = false
    @State private var findController = MarkdownFindController()
    @State private var findQuery = ""
    @State private var findCount = 0
    @State private var findCurrent = 0
    @FocusState private var findFocused: Bool
    @State private var actionsExpanded = false
    @State private var actionsHovering = false
    @State private var actionsExporting = false
    @State private var shellWidth: CGFloat = 0

    var body: some View {
        content
            .modifier(OptionKeyMonitor(down: $model.optionDown))
            .preferredColorScheme(model.theme.scheme)
            .environment(\.appTextScale, model.textScale)
            .font(appTextFont(.body, model.textScale))
            .onAppear {
                if Models.supported, model.eulaAccepted, model.accepted {
                    model.load(name: model.modelName)
                }
            }
            .alert("Download Failed", isPresented: $model.downloadFailed) {
                Button("OK") { }
            } message: {
                Text(model.downloadFailure ?? "")
            }
    }

    @ViewBuilder
    private var content: some View {
        if model.downloadName != nil || model.downloading
            || model.loadError != nil {
            onboarding
        } else if model.showSettings {
            SettingsView(model: model, onClose: model.closeSettings)
        } else if model.showDebug {
            DebugView(model: model, onClose: { model.showDebug = false })
        } else if model.ready && model.downloadName == nil
            && !model.downloading {
            chatShell
        } else {
            onboarding
        }
    }

    private var chatShell: some View {
        ZStack(alignment: .leading) {
            chat
                .overlay { hudOverlay }
                .animation(.easeInOut(duration: 0.4), value: model.hud)
                .navigationTitle("")
                .disabled(sidebarOpen)
                .modifier(ChatChrome(
                    leading: foldersButton,
                    center: dismissing(centerBar),
                    trailing: dismissing(trailingButtons)))
            drawer
            renameOverlay
        }
        .onGeometryChange(for: CGFloat.self, of: { geo in
            geo.size.width
        }, action: { w in shellWidth = w })
        .background { newChatShortcut }
        .modifier(TitleBarClickMonitor {
            if sidebarOpen { closeSidebar() }
        })
    }

    private func dismissing<V: View>(_ bar: V) -> some View {
        bar
            .opacity(sidebarOpen ? 0.4 : 1)
            .allowsHitTesting(!sidebarOpen)
            .overlay {
                if sidebarOpen {
                    Color.black.opacity(0.001)
                        .contentShape(Rectangle())
                        .onTapGesture(perform: closeSidebar)
                }
            }
    }

    private var newChatShortcut: some View {
        Button("", action: openNewChat)
            .keyboardShortcut("n", modifiers: .command)
            .disabled(model.busy)
            .hidden()
    }

    @Environment(\.horizontalSizeClass) private var sizeClass

    private var barGlyph: Font? {
        sizeClass == .compact ? .system(size: 20 * model.textScale) : nil
    }

    private var foldersButton: some View {
        Button(action: toggleSidebar) {
            Image(systemName: "sidebar.leading")
        }
        .font(barGlyph)
        .help("Menu")
    }

    private var newChatButton: some View {
        Button(action: newChatOrReveal) {
            Image(systemName: "square.and.pencil")
        }
        .help("New chat")
        .disabled(model.busy)
        .simultaneousGesture(revealHidden, including: isOS ? .all : .none)
    }

    private var revealHidden: some Gesture {
        LongPressGesture(minimumDuration: 2).onEnded { _ in
            dismissKeyboard()
            model.revealHidden()
            closeSidebar()
        }
    }

    private var trailingButtons: some View {
        HStack(spacing: 12) {
            if !model.messages.isEmpty {
                if actionsExpanded {
                    Button { actionsExpanded = false } label: {
                        Image(systemName: "chevron.right.2")
                    }
                    .help("Hide actions")
                    TranscriptActions(
                        document: model.transcriptDocument,
                        title: model.transcriptTitle,
                        renderMarkdown: $model.renderMarkdown,
                        exporting: $actionsExporting,
                        onFind: openFind, onDebug: debugAction)
                } else {
                    Button { actionsExpanded = true } label: {
                        Image(systemName: "chevron.left.2")
                    }
                    .help("More actions")
                }
            }
            newChatButton
        }
        .font(barGlyph)
        .onHover { inside in actionsHovering = inside }
        .animation(.easeInOut(duration: 0.2), value: actionsExpanded)
        .task(id: actionsIdle) { await collapseActionsAfterIdle() }
    }

    private var debugAction: (() -> Void)? {
        var out: (() -> Void)? = nil
        if model.statusLine, !model.traceEvents.isEmpty {
            out = openDebug
        }
        return out
    }

    private var actionsIdle: Bool {
        !isOS && actionsExpanded && !actionsHovering && !actionsExporting
    }

    private func collapseActionsAfterIdle() async {
        if actionsIdle {
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { actionsExpanded = false }
        }
    }

    @ViewBuilder
    private var drawer: some View {
        if sidebarOpen {
            Color.black.opacity(0.3)
                .ignoresSafeArea()
                .transition(.opacity)
                .onTapGesture(perform: closeSidebar)
                .zIndex(1)
            Sidebar(model: model, theme: $model.theme,
                    onClose: closeSidebar, onOpen: openConversation,
                    onNewChat: openNewChat,
                    onSettings: openSettings,
                    onRename: { convo in renaming = convo })
                .frame(width: drawerWidth)
                .frame(maxHeight: .infinity)
                .background(.bar)
                .transition(.move(edge: .leading))
                .zIndex(2)
        }
    }

    private var drawerWidth: CGFloat {
        let narrow = 300 * model.textScale
        return isOS && sizeClass == .regular ? max(narrow, shellWidth / 2)
                                             : narrow
    }

    @ViewBuilder
    private var renameOverlay: some View {
        if let convo = renaming {
            RenameDialog(title: convo.title,
                         onCommit: { name in
                             model.renameConversation(convo.id, to: name)
                             renaming = nil
                             refocusPrompt()
                         },
                         onCancel: {
                             renaming = nil
                             refocusPrompt()
                         })
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .zIndex(3)
        }
    }

    private var onboarding: some View {
        Group {
            if !model.eulaAccepted {
                EULAView(onAgree: model.acceptEULA)
            } else if let name = model.downloadName {
                if GemmaTerms.applies(to: name), !model.gemmaTermsAccepted {
                    GemmaTermsView(model: name,
                                   onAgree: model.acceptGemmaTerms,
                                   onCancel: model.cancelDownload)
                } else {
                    downloadConsent(name)
                }
            } else if !model.accepted {
                DisclaimerView(onAgree: model.accept)
            } else if model.downloading {
                downloadProgress
            } else if let err = model.loadError {
                failureView(err)
            } else if model.compiling {
                loadProgress
            } else {
                preparing
            }
        }
        .overlay { hudOverlay }
        .animation(.easeInOut(duration: 0.4), value: model.hud)
        .modifier(OnboardingTitle())
    }

    @ViewBuilder
    private var hudOverlay: some View {
        if let hud = model.hud {
            Text(hud.text)
                .font(hud.prominent ? .title2 : .headline)
                .padding(.horizontal, hud.prominent ? 30 : 22)
                .padding(.vertical, hud.prominent ? 20 : 14)
                .background(.regularMaterial,
                            in: RoundedRectangle(cornerRadius: 14))
                .overlay {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(.separator, lineWidth: 0.5)
                }
                .transition(.opacity)
                .allowsHitTesting(false)
        }
    }

    @ViewBuilder
    private var centerBar: some View {
        if !actionsExpanded {
            if model.inTurn && isOS {
                Text(ContentView.appName)
                    .appFont(.headline)
                    .lineLimit(1)
                    .transition(.opacity)
            } else {
                modelPicker
            }
        }
    }

    private static let appName: String =
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName")
            as? String)
        ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleName")
            as? String) ?? "ChatOKF"

    private var pickable: [String] {
        Models.offered(unlocked: model.unlocked)
    }

    private func pick(_ name: String) {
        if model.isDownloaded(name) {
            model.switchModel(name)
        } else {
            model.requestDownload(name)
        }
    }

    private var modelPicker: some View {
        Group {
            if pickable.count < 2 {
                Button { showModelInvite = true } label: { modelPickerLabel }
                    .buttonStyle(.plain)
            } else {
                Menu {
                    ForEach(pickable, id: \.self) { name in
                        Button(Models.display(name, among: pickable)) {
                            pick(name)
                        }
                    }
                } label: { modelPickerLabel }
                .menuIndicator(.hidden)
                .disabled(model.busy)
            }
        }
        .alert("More models on Mac", isPresented: $showModelInvite) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("This iPhone runs \(Models.display(model.modelName)) to stay "
               + "fast and light. On a Mac, ChatOKF can run our larger 2B, 4B "
               + "and 9B models that are noticeably smarter, "
               + "with much sharper image understanding.")
        }
    }

    private var modelPickerLabel: some View {
        HStack(spacing: 4) {
            Text(Models.display(model.modelName, among: pickable))
                .appFont(.subheadline)
                .fontWeight(.semibold)
                .lineLimit(1)
            if pickable.count >= 2 {
                Image(systemName: "chevron.down").appFont(.caption2)
            }
        }
    }

    private func openConversation(_ id: UUID) {
        dismissKeyboard()
        actionsExpanded = false
        let alreadyLive = id == model.currentConversationId &&
                          !model.readOnly
        if !alreadyLive { model.openConversation(id) }
        closeSidebar()
    }

    private func dismissKeyboard() {
        if isOS { promptFocus.request(false) }
        findFocused = false
    }

    private func refocusPrompt() {
        if !isOS { promptFocus.request(true) }
    }

    private func newChatOrReveal() {
        if !isOS, model.unlocked {
            dismissKeyboard()
            model.revealHidden()
            closeSidebar()
        } else {
            openNewChat()
        }
    }

    private func openNewChat() {
        dismissKeyboard()
        actionsExpanded = false
        model.newChat()
        closeSidebar()
    }

    private func openSettings() {
        dismissKeyboard()
        model.showDebug = false
        model.showSettings = true
        closeSidebar()
    }

    private func openDebug() {
        dismissKeyboard()
        model.showSettings = false
        model.showDebug = true
        closeSidebar()
    }

    private func toggleSidebar() {
        if !sidebarOpen {
            dismissKeyboard()
            model.commitCurrent()
        }
        withAnimation(.snappy) { sidebarOpen.toggle() }
    }

    private func closeSidebar() {
        withAnimation(.snappy) { sidebarOpen = false }
        refocusPrompt()
    }

    private func downloadConsent(_ name: String) -> some View {
        VStack(spacing: 16) {
            VStack(spacing: 4) {
                Text("Download").appFont(.title2).bold()
                Text(Models.display(name, among: pickable))
                    .appFont(.title3)
                    .foregroundStyle(.secondary)
            }
            Text("This model is about \(model.downloadSizeText).")
            if let est = model.estimatedMinutes(name) {
                Text(est.optimize > 0
                     ? "Estimated ~\(est.download) min download\n"
                       + "~\(est.optimize) min optimize"
                     : "Estimated ~\(est.download) min download")
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Text("Stored on this device, excluded from iCloud and backup.\n"
               + "It downloads once.\n"
               + "Later launches need no network.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(spacing: 14) {
                Button("Download") { model.confirmDownload() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                if model.canCancelDownload {
                    Button("Cancel") { model.cancelDownload() }
                        .controlSize(.large)
                        .keyboardShortcut(.cancelAction)
                }
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: 420)
    }

    private var downloadProgress: some View {
        VStack(spacing: 16) {
            ProgressView(value: model.downloadFraction)
                .frame(maxWidth: 320)
            VStack(spacing: 4) {
                Text(model.downloadTotal > 0
                     ? "Downloading" : "Preparing download…")
                    .appFont(.headline)
                if model.downloadTotal > 0 {
                    Text(Models.display(model.modelName))
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                    Text(model.downloadCounter)
                        .appFont(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if model.downloadTotal > 0 {
                stageWhimsical(.downloading, $downloadingPhrase)
            }
            if let eta = model.downloadETA {
                Text(eta).appFont(.caption).foregroundStyle(.tertiary)
            }
            if model.canAbortDownload {
                Button("Cancel") { model.abortDownload() }
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
                    .padding(.top, 8)
            }
        }
        .padding(32)
    }

    private var loadProgress: some View {
        VStack(spacing: 16) {
            ProgressView()
                .frame(maxWidth: 320)
            VStack(spacing: 4) {
                Text("Loading…").appFont(.headline)
                Text(Models.display(model.modelName))
                    .appFont(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 32)
        .padding(.horizontal, 24)
        .frame(maxWidth: 480)
    }

    private func stageWhimsical(_ stage: Whimsical.Stage,
                                _ phrase: Binding<String>) -> some View {
        Text(phrase.wrappedValue)
            .appFont(.subheadline)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.4)
            .frame(maxWidth: .infinity)
            .contentTransition(.opacity)
            .animation(.easeInOut(duration: 0.45),
                       value: phrase.wrappedValue)
            .modifier(Whimsy())
            .task {
                while !Task.isCancelled {
                    phrase.wrappedValue = Whimsical.current(stage)
                    try? await Task.sleep(for: .seconds(1))
                }
            }
    }

    private var preparing: some View {
        ProgressView()
            .padding(32)
            .onAppear {
                if !model.downloading, !model.compiling, !model.ready {
                    model.load(name: model.modelName)
                }
            }
    }

    private func failureView(_ message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .appFont(.largeTitle)
                .foregroundStyle(.secondary)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Try Again") { model.retry() }
                .controlSize(.large)
        }
        .padding(32)
        .frame(maxWidth: 420)
    }

    static let stallProbeOn: Bool = {
        let on = Flags.on("stall-probe")
        Instrument.note(.perf, "[stall] probe \(on ? "ARMED" : "off")")
        return on
    }()

    private var chat: some View {
        VStack(spacing: 0) {
            transcript
                .simultaneousGesture(pinchZoom,
                                     including: isOS ? .all : .none)
                .modifier(Shimmer(active: ContentView.stallProbeOn))
            if !model.readOnly { composerBar }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.handleDrop(urls, at: model.caret)
            return true
        } isTargeted: { dropActive = $0 }
        .overlay {
            if dropActive {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.accentColor, lineWidth: 2)
                    .padding(4)
                    .allowsHitTesting(false)
            }
        }
        .overlay(alignment: .top) { if findActive { findBar } }
        .animation(.easeInOut(duration: 0.2), value: findActive)
    }

    private var composerBar: some View {
        VStack(spacing: 0) {
            Composer(model: model, focus: $promptFocus,
                     editing: $promptEditing)
            if model.statusLine {
                Divider()
                statusLine
            }
        }
        .background(.bar)
    }

    private var findBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Find in transcript", text: $findQuery)
                .textFieldStyle(.plain)
                .focused($findFocused)
                .onSubmit { findStep(findController.findNext) }
                .onChange(of: findQuery) { _, q in
                    findCount = findController.find(q)
                    findCurrent = findController.currentMatch
                }
            if findCount > 0 {
                Text("\(findCurrent)/\(findCount)")
                    .appFont(.caption).monospacedDigit()
                    .foregroundStyle(.secondary)
            } else if !findQuery.isEmpty {
                Text("none").appFont(.caption).foregroundStyle(.secondary)
            }
            Button { findStep(findController.findPrevious) } label: {
                Image(systemName: "chevron.up")
            }
            .disabled(findCount == 0)
            Button { findStep(findController.findNext) } label: {
                Image(systemName: "chevron.down")
            }
            .disabled(findCount == 0)
            Button("Done", action: closeFind)
                .keyboardShortcut(.cancelAction)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
        .transition(.move(edge: .top).combined(with: .opacity))
    }

    private func openFind() {
        findController.order = model.messages.map { m in m.id }
        findActive = true
        if !findQuery.isEmpty {
            findCount = findController.find(findQuery)
            findCurrent = findController.currentMatch
        }
        Task { @MainActor in findFocused = true }
    }

    private func closeFind() {
        findActive = false
        findController.clear()
        findFocused = false
        refocusPrompt()
    }

    private func findStep(_ navigate: () -> Int) {
        findCurrent = navigate()
        if isOS { dismissKeyboard() }
    }

    private var samplePills: some View {
        VStack(spacing: 12) {
            if model.accessState != .offline, model.showSample("research") {
                sampleCard(symbol: "books.vertical",
                           title: "Dark Matter Research",
                           subtitle: "Watch it research Wikipedia "
                               + "and the web",
                           thumb: nil,
                           action: model.runResearchSample)
            }
            if model.showSample("calc") {
                if model.optionDown {
                    sampleCard(symbol: "function",
                               title: "Euler's Formula",
                               subtitle: "Watch it do math with imaginary "
                                   + "numbers",
                               thumb: nil,
                               action: model.runEulerSample)
                } else if model.calcSampleIsInterest {
                    sampleCard(symbol: "percent",
                               title: "Compound Interest",
                               subtitle: "Watch it use the calculator",
                               thumb: nil,
                               action: model.runInterestSample)
                        .help("Try to hold the Option key \u{1F609}")
                } else {
                    sampleCard(symbol: "birthday.cake",
                               title: "Cookie Math",
                               subtitle: "Baker's percentages on the "
                                   + "calculator",
                               thumb: nil,
                               action: model.runCookiesSample)
                        .help("Try to hold the Option key \u{1F609}")
                }
            }
            if model.canAttachImages, model.showSample("picture"),
               ChatModel.samplePictureThumb != nil {
                sampleCard(symbol: "photo",
                           title: "Picture Understanding",
                           subtitle: "It writes a story about this photo",
                           thumb: ChatModel.samplePictureThumb,
                           action: model.runPictureSample)
            }
            if model.canOfferVideoSample, model.showSample("video") {
                sampleCard(symbol: "play.rectangle",
                           title: "Video Understanding",
                           subtitle: "Watch the frames it is looking at",
                           thumb: ChatModel.sampleClipThumb,
                           badge: "play.circle.fill",
                           action: model.runVideoSample)
            }
            if model.canOfferDocumentSample, model.showSample("document") {
                sampleCard(symbol: "doc.richtext",
                           title: "Document Understanding",
                           subtitle: "It reads the table inside a PDF",
                           thumb: nil,
                           action: model.runDocumentSample)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity)
    }

    private func sampleCard(symbol: String, title: String, subtitle: String,
                            thumb: CGImage?, badge: String? = nil,
                            action: @escaping () -> Void)
        -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                if let thumb {
                    Image(decorative: thumb, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay { badgeMark(badge) }
                } else {
                    Image(systemName: symbol)
                        .appFont(.title2)
                        .frame(width: 44, height: 44)
                        .foregroundStyle(Color.accentColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).appFont(.headline)
                    Text(subtitle)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: 440 * textScale)
            .background(.regularMaterial,
                        in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .stroke(.separator, lineWidth: 0.5)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(SampleCardStyle())
    }

    @ViewBuilder
    private func badgeMark(_ symbol: String?) -> some View {
        if let symbol {
            Image(systemName: symbol)
                .font(.system(size: 18))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.55), radius: 2)
        }
    }

    private var statusLine: some View {
        HStack {
            statusText
                .font(.system(size: statusPoints))
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .lineLimit(1)
                .modifier(Shimmer(active: model.prefilling))
            Spacer()
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusText: some View {
        if model.statsLabel.isEmpty {
            ViewThatFits(in: .horizontal) {
                modelSummary(facts: model.modelFacts.count)
                modelSummary(facts: model.modelFacts.count)
                    .font(.system(size: statusPoints * 0.85))
                modelSummary(facts: 0)
            }
        } else {
            Text(model.statsLabel).minimumScaleFactor(0.5)
        }
    }

    // A point under footnote on a phone, where the line has one width: at
    // the old size a zoom step overflowed and rendered smaller, not larger.
    private var statusPoints: CGFloat {
        (appTextPoints(.footnote) + (isOS ? -1 : 1)) * model.textScale
    }

    private func modelSummary(facts: Int) -> Text {
        var out = Text("")
        for tower in model.modelTowers {
            let gap = tower.id == 0 ? "" : " "
            out = out + Text(gap)
                + Text(Image(systemName: tower.symbol))
                + Text(" \(tower.weight)")
        }
        for fact in model.modelFacts.prefix(facts) {
            out = out + Text(" \u{00B7} \(fact)")
        }
        return out
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // A LazyVStack row taller than the viewport blanks on iOS once
                // its top scrolls off-screen; a long answer is one such row.
                VStack(alignment: .leading, spacing: 12) {
                    let _ = Instrument.beat("transcript")
                    ForEach(model.messages) { m in
                        bubble(m).id(m.id)
                    }
                    if model.showSamples { samplePills }
                    Color.clear
                        .frame(height: max(tailInset, 1))
                        .id("chat-tail")
                }
                .padding(12)
                .background(
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { dismissKeyboard() })
            }
            .scrollDismissesKeyboard(.immediately)
            .onChange(of: model.busy) { _, busy in
                tailCollapse?.cancel()
                if busy {
                    withAnimation(.easeOut(duration: 0.25)) {
                        tailInset = tailInsetSize
                    }
                } else {
                    tailCollapse = Task { @MainActor in
                        try? await Task.sleep(for: .seconds(1.2))
                        if !Task.isCancelled {
                            withAnimation(.easeOut(duration: 0.35)) {
                                tailInset = 0
                            }
                        }
                    }
                }
            }
            .onScrollPhaseChange { _, phase, context in
                if phase == .interacting || phase == .decelerating {
                    let g = context.geometry
                    follow = g.contentSize.height - g.visibleRect.maxY < 60
                }
            }
            .onChange(of: model.messages.last?.text) { _, _ in
                if follow { scrollToBottom(proxy) }
            }
            .onChange(of: promptEditing) { _, editing in
                if editing {
                    follow = true
                    withAnimation { scrollToBottom(proxy) }
                }
            }
            .onChange(of: model.messages.count) { _, _ in
                follow = true
                findController.order = model.messages.map { m in m.id }
            }
            .onAppear {
                findController.scrollTo = { id, fraction in
                    withAnimation {
                        proxy.scrollTo(id,
                            anchor: UnitPoint(x: 0.5, y: fraction ?? 0.5))
                    }
                }
                findController.order = model.messages.map { m in m.id }
            }
            .onGeometryChange(for: CGFloat.self, of: { geo in
                geo.size.height
            }, action: { h in
                chatHeight = h
                if follow { scrollToBottom(proxy) }
            })
            .coordinateSpace(.named(transcriptSpace))
            .overlay { toolCallout }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 8) {
                    if isReasoningLive { quickAnswerButton }
                    if !follow { toBottomButton(proxy) }
                }
                .padding(.bottom, 10)
            }
        }
    }

    private var peekedRound: ToolRound? {
        var out: ToolRound? = nil
        if let id = peek.messageId,
           let m = model.messages.first(where: { msg in msg.id == id }) {
            out = m.toolRounds.first { r in r.id == peek.roundId }
        }
        return out
    }

    private var peekScale: CGFloat {
        min(popScale, 1.5) * model.textScale
    }

    private var peekTextSize: CGFloat {
        11 * (1 + (popScale - 1) / 4) * model.textScale
    }

    @ViewBuilder
    private var toolCallout: some View {
        if let round = peekedRound {
            GeometryReader { geo in
                let below = peek.anchor.midY < geo.size.height / 2
                let width = min(440 * peekScale + 24, geo.size.width - 24)
                let shape = Callout(
                    tailAt: peek.anchor.minX + 24 - (geo.size.width - width) / 2,
                    tailOnBottom: !below)
                ZStack {
                    Color.black.opacity(0.001)
                        .contentShape(Rectangle())
                        .onTapGesture { peek.close() }
                    ToolRoundDetail(round: round, size: peekTextSize,
                                    k: peekScale)
                        .padding(below ? .top : .bottom, Callout.tailHeight)
                        .frame(width: width)
                        .background(.regularMaterial, in: shape)
                        .overlay { shape.stroke(.separator, lineWidth: 0.5) }
                        .onGeometryChange(for: CGSize.self, of: { g in
                            g.size
                        }, action: { s in calloutSize = s })
                        .opacity(calloutSize.height > 0 ? 1 : 0)
                        .position(x: geo.size.width / 2,
                                  y: clampedCalloutY(below, geo.size.height))
                }
            }
        }
    }

    private func clampedCalloutY(_ below: Bool,
                                 _ height: CGFloat) -> CGFloat {
        let h = calloutSize.height
        let want = below ? peek.anchor.maxY + 6 + h / 2
                         : peek.anchor.minY - 6 - h / 2
        return min(max(want, h / 2 + 6), max(height - h / 2 - 6, h / 2 + 6))
    }

    private var isReasoningLive: Bool {
        var live = false
        if model.busy, let last = model.messages.last, !last.fromUser {
            live = last.text.isEmpty && !last.reasoning.isEmpty
        }
        return live
    }

    private var quickAnswerButton: some View {
        Button(action: model.quickAnswer) {
            Label("Quick Answer", systemImage: "bolt.fill")
                .appFont(.caption)
                .fontWeight(.semibold)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().stroke(.separator, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        proxy.scrollTo("chat-tail", anchor: .bottom)
    }

    private func toBottomButton(_ proxy: ScrollViewProxy) -> some View {
        Button {
            follow = true
            withAnimation { scrollToBottom(proxy) }
        } label: {
            Image(systemName: "arrow.down.circle.fill")
                .appFont(.title)
                .symbolRenderingMode(.hierarchical)
                .frame(width: 44, height: 44)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
    }

    private func bubble(_ m: Message) -> some View {
        HStack {
            if m.fromUser { Spacer(minLength: 40) }
            VStack(alignment: m.fromUser ? .trailing : .leading,
                   spacing: 4) {
                if !m.images.isEmpty {
                    imageStrip(m.images)
                }
                if m.clips.isEmpty {
                    ForEach(0..<m.posters.count, id: \.self) { i in
                        ClipPoster(image: m.posters[i])
                    }
                } else {
                    ForEach(m.clips, id: \.self) { url in
                        ClipPlayer(url: url)
                    }
                }
                ForEach(m.docs, id: \.self) { doc in
                    DocChip(doc: doc)
                }
                if !m.fromUser, !m.reasoning.isEmpty {
                    ReasoningView(text: m.reasoning,
                                  doc: m.reasoningDoc,
                                  style: ContentView.scaled(
                                      ContentView.reasoningStyle,
                                      by: textScale),
                                  markdown: model.renderMarkdown,
                                  active: isThinking(m),
                                  label: isThinking(m)
                                         ? model.thinkLabel : "Thoughts",
                                  maxHeight: chatHeight / 3,
                                  first: m.id == firstReasoningId)
                }
                if !m.fromUser, !m.toolRounds.isEmpty {
                    ToolCallStrip(messageId: m.id, rounds: m.toolRounds,
                                  peek: peek, live: isLive(m))
                }
                // Answer text on screen outranks the status line: a tool
                // round raises `prefilling` again and would blink it away.
                if let answer = answerText(m) {
                    answerBubble(m, answer)
                    if !m.fromUser, !isLive(m) {
                        AnswerActions(text: answer,
                                      title: model.transcriptTitle)
                    }
                } else if !m.fromUser, isPrefilling(m) {
                    prefillWhimsical
                } else if m.loopStopped {
                    loopStopNote
                } else if isAnswerless(m) {
                    noAnswerNote(m)
                }
            }
            .frame(maxWidth: m.fromUser ? nil : answerColumn,
                   alignment: .leading)
            .frame(maxWidth: m.fromUser ? nil : .infinity,
                   alignment: .leading)
        }
    }

    private var answerColumn: CGFloat { 680 * textScale }

    @ViewBuilder
    private func imageStrip(_ images: [CGImage]) -> some View {
        switch images.count {
        case 1:
            Image(decorative: images[0], scale: 1)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 300, maxHeight: 260)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        case 2, 3:
            HStack(spacing: 6) {
                ForEach(0..<images.count, id: \.self) { i in
                    stripThumb(images[i], 108)
                }
            }
        case 4:
            VStack(spacing: 6) {
                HStack(spacing: 6) {
                    stripThumb(images[0], 132)
                    stripThumb(images[1], 132)
                }
                HStack(spacing: 6) {
                    stripThumb(images[2], 132)
                    stripThumb(images[3], 132)
                }
            }
        default:
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(0..<images.count, id: \.self) { i in
                        stripThumb(images[i], 96)
                    }
                }
            }
        }
    }

    private func stripThumb(_ cg: CGImage, _ side: CGFloat) -> some View {
        Image(decorative: cg, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func spoken(_ m: Message) -> String? {
        m.id == model.messages.last?.id ? model.speech.spokenText : nil
    }

    private func answerBubble(_ m: Message,
                              _ answer: String) -> some View {
        Group {
            if !m.fromUser && model.renderMarkdown
                && !m.answerDoc.items.isEmpty {
                MarkdownTextView(m.answerDoc,
                                 style: ContentView.scaled(
                                     ContentView.answerStyle, by: textScale),
                                 find: findController, findId: m.id,
                                 scrolls: false, speaking: spoken(m))
            } else if !m.fromUser {
                PlainTextView(answer,
                              style: ContentView.scaled(
                                  ContentView.answerStyle, by: textScale),
                              find: findController, findId: m.id,
                              scrolls: false, speaking: spoken(m))
            } else {
                Text(answer)
                    .font(.system(
                        size: ContentView.answerStyle.bodySize * textScale))
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(m.fromUser ? Color.accentColor.opacity(0.2)
                               : Color.gray.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var prefillWhimsical: some View {
        Text(model.thinkStatus)
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .contentTransition(.opacity)
            .animation(.easeInOut(duration: 0.45), value: model.thinkStatus)
            .modifier(Whimsy())
    }

    private func answerText(_ m: Message) -> String? {
        m.text.isEmpty ? nil : m.text
    }

    private func isAnswerless(_ m: Message) -> Bool {
        !m.fromUser && !isLive(m)
    }

    private func isLive(_ m: Message) -> Bool {
        model.busy && m.id == model.messages.last?.id
    }

    private func noAnswerNote(_ m: Message) -> some View {
        Text(m.toolRounds.isEmpty
             ? "The model did not reach an answer. Try rephrasing."
             : "The model made tool calls but gave no final answer.")
            .appFont(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    private var loopStopNote: some View {
        Text("The model began repeating itself and was stopped before it "
           + "reached an answer. Try rephrasing.")
            .appFont(.caption)
            .italic()
            .foregroundStyle(.secondary)
    }

    private func isPrefilling(_ m: Message) -> Bool {
        isLive(m) && model.prefilling
    }

    private func isThinking(_ m: Message) -> Bool {
        isLive(m) && m.text.isEmpty
    }

    private var firstReasoningId: UUID? {
        model.messages.first { m in
            !m.fromUser && !m.reasoning.isEmpty
        }?.id
    }

}

private struct ClipPlayer: View {

    let url: URL
    @State private var player: AVPlayer?
    @State private var playable = true
    @State private var playing = false

    var body: some View {
        Group {
            if playable, let player {
                ClipSurface(player: player)
                    .frame(maxWidth: 300, minHeight: 170, maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay { if !playing { PlayGlyph() } }
                    .contentShape(Rectangle())
                    .onTapGesture { toggle(player) }
                    .onReceive(NotificationCenter.default.publisher(
                        for: AVPlayerItem.didPlayToEndTimeNotification)
                    ) { note in
                        if (note.object as? AVPlayerItem)
                            === player.currentItem {
                            player.seek(to: .zero)
                            playing = false
                        }
                    }
            } else if playable {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quinary)
                    .frame(maxWidth: 300, minHeight: 170, maxHeight: 260)
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "film")
                    Text("Video: \(url.lastPathComponent)")
                        .lineLimit(1).truncationMode(.middle)
                }
                .appFont(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .onAppear {
            playable = FileManager.default.fileExists(atPath: url.path)
            if playable, player == nil { player = AVPlayer(url: url) }
        }
        .onDisappear {
            player?.pause()
            playing = false
        }
    }

    private func toggle(_ player: AVPlayer) {
        if playing { player.pause() } else { player.play() }
        playing.toggle()
    }

}

private struct ClipPoster: View {

    let image: CGImage

    var body: some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .scaledToFit()
            .frame(maxWidth: 300, maxHeight: 260)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(alignment: .bottomLeading) {
                FilmMark().padding(8)
            }
    }

}

private struct FilmMark: View {

    var body: some View {
        Image(systemName: "film.fill")
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.35), radius: 1.5, y: 1)
            .frame(width: 28, height: 28)
            .background(.black.opacity(0.28), in: Circle())
            .background(.ultraThinMaterial, in: Circle())
            .overlay { Circle().stroke(.white.opacity(0.35), lineWidth: 1) }
            .allowsHitTesting(false)
    }

}

private struct PlayGlyph: View {

    var body: some View {
        Image(systemName: "play.fill")
            .font(.system(size: 30, weight: .bold))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.35), radius: 1.5, y: 1)
            .offset(x: 3)
            .frame(width: 72, height: 72)
            .background(.black.opacity(0.28), in: Circle())
            .background(.ultraThinMaterial, in: Circle())
            .overlay { Circle().stroke(.white.opacity(0.35), lineWidth: 1) }
            .shadow(color: .black.opacity(0.3), radius: 6, y: 2)
            .allowsHitTesting(false)
    }

}

private struct SampleCardStyle: ButtonStyle {

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.55 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12),
                       value: configuration.isPressed)
    }

}

private struct ReasoningView: View {

    let text: String
    let doc: Markdown.Document
    let style: MarkdownStyle
    let markdown: Bool
    let active: Bool
    let label: String
    let maxHeight: CGFloat
    @State private var expanded: Bool
    @State private var touched = false
    @State private var innerFollow = true

    init(text: String, doc: Markdown.Document, style: MarkdownStyle,
         markdown: Bool, active: Bool, label: String, maxHeight: CGFloat,
         first: Bool) {
        self.text = text
        self.doc = doc
        self.style = style
        self.markdown = markdown
        self.active = active
        self.label = label
        self.maxHeight = maxHeight
        _expanded = State(initialValue: isOS && first && active)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            header
            if expanded {
                reasoning
            }
        }
        .onChange(of: active) { _, live in
            if !live, !touched { expanded = false }
        }
    }

    private var marqueeing: Bool { active && !expanded }

    private var header: some View {
        Button {
            touched = true
            expanded.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: expanded
                      ? "chevron.down" : "chevron.right")
                    .appFont(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 18, minHeight: 22)
                if marqueeing {
                    ThinkingMarquee(text: text)
                } else {
                    Text(label)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var reasoning: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if markdown && !doc.items.isEmpty {
                        MarkdownView(doc, style: style)
                    } else {
                        Text(text)
                            .appFont(.callout)
                            .foregroundStyle(Color(white: 0.53))
                            .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .id("reason")
            }
            .frame(maxHeight: maxHeight)
            .onScrollPhaseChange { _, phase, context in
                if phase == .interacting || phase == .decelerating {
                    let g = context.geometry
                    let d = g.contentSize.height - g.visibleRect.maxY
                    innerFollow = d < 40
                }
            }
            .onChange(of: text) { _, _ in
                if innerFollow { proxy.scrollTo("reason", anchor: .bottom) }
            }
        }
    }
}

private let transcriptSpace = "transcript"

@MainActor @Observable final class ToolPeek {

    private(set) var messageId: UUID?
    private(set) var roundId = 0
    private(set) var anchor: CGRect = .zero

    func showing(_ message: UUID, _ round: Int) -> Bool {
        messageId == message && roundId == round
    }

    func show(_ message: UUID, _ round: Int, at frame: CGRect) {
        messageId = message
        roundId = round
        anchor = frame
    }

    func toggle(_ message: UUID, _ round: Int, at frame: CGRect) {
        if showing(message, round) {
            close()
        } else {
            show(message, round, at: frame)
        }
    }

    func track(_ message: UUID, _ round: Int, at frame: CGRect) {
        if showing(message, round) { anchor = frame }
    }

    func close() {
        messageId = nil
    }

}

private struct Callout: Shape {

    let tailAt: CGFloat
    let tailOnBottom: Bool
    private static let radius: CGFloat = 12
    private static let tail: CGFloat = 9

    static var tailHeight: CGFloat { tail }

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let body = CGRect(x: rect.minX,
                          y: rect.minY + (tailOnBottom ? 0 : Callout.tail),
                          width: rect.width,
                          height: rect.height - Callout.tail)
        p.addRoundedRect(in: body, cornerSize: CGSize(width: Callout.radius,
                                                      height: Callout.radius))
        let span = Callout.radius + Callout.tail
        let x = min(max(tailAt, body.minX + span), body.maxX - span)
        let edge = tailOnBottom ? body.maxY : body.minY
        let tip = tailOnBottom ? rect.maxY : rect.minY
        p.move(to: CGPoint(x: x - Callout.tail, y: edge))
        p.addLine(to: CGPoint(x: x, y: tip))
        p.addLine(to: CGPoint(x: x + Callout.tail, y: edge))
        p.closeSubpath()
        return p
    }
}

private struct ToolCallStrip: View {
    let messageId: UUID
    let rounds: [ToolRound]
    let peek: ToolPeek
    let live: Bool
    @State private var frames: [Int: CGRect] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                    .appFont(.caption2)
                Text("Using tools")
            }
            .appFont(.caption)
            .foregroundStyle(.secondary)
            ForEach(rounds) { round in row(round) }
        }
    }

    private func row(_ round: ToolRound) -> some View {
        HStack(spacing: 6) {
            Text(round.id == rounds.last?.id ? "└─" : "├─")
                .appFont(.caption).monospaced()
            Image(systemName: round.symbol)
                .appFont(.caption2)
                .foregroundStyle(errored(round) ? Color.orange
                                                : Color.secondary)
            Text(round.label)
                .appFont(.caption)
            if !round.args.isEmpty {
                Text(round.args)
                    .appFont(.caption2)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .foregroundStyle(.secondary)
        .modifier(Shimmer(active: running(round)))
        .contentShape(Rectangle())
        .onGeometryChange(for: CGRect.self, of: { geo in
            geo.frame(in: .named(transcriptSpace))
        }, action: { frame in
            frames[round.id] = frame
            peek.track(messageId, round.id, at: frame)
        })
        .onTapGesture {
            peek.toggle(messageId, round.id, at: frames[round.id] ?? .zero)
        }
    }

    private func errored(_ round: ToolRound) -> Bool {
        round.result?.hasPrefix("error") == true
    }

    private func running(_ round: ToolRound) -> Bool {
        live && round.result == nil
    }

}

private struct ToolRoundDetail: View {

    let round: ToolRound
    let size: CGFloat
    let k: CGFloat
    @State private var contentHeight: CGFloat = 20

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: round.symbol)
                Text(round.emitted).bold()
                if !round.emitted.isEmpty && round.label != round.emitted {
                    Text("-> \(round.label)")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.system(size: size))
            if !round.args.isEmpty {
                Text(round.args)
                    .font(.system(size: size, design: .monospaced))
                    .textSelection(.enabled)
            }
            ScrollView {
                Text(round.result ?? "running")
                    .font(.system(size: size, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self, of: { geo in
                        geo.size.height
                    }, action: { h in contentHeight = h })
            }
            .frame(minWidth: 260 * k, idealWidth: 440 * k, maxWidth: 440 * k)
            .frame(height: min(max(contentHeight, 20), 280 * k))
        }
        .padding(12)
    }

}

private struct Whimsy: ViewModifier {

    @State private var travel: CGFloat = -2
    private static let dim: CGFloat = 0.55
    private static let sweep: TimeInterval = 2.6

    func body(content: Content) -> some View {
        content
            .mask {
                GeometryReader { geo in
                    let w = max(geo.size.width, 1)
                    LinearGradient(stops: [
                        .init(color: .white.opacity(Whimsy.dim),
                              location: 0),
                        .init(color: .white.opacity(Whimsy.dim),
                              location: 0.34),
                        .init(color: .white, location: 0.5),
                        .init(color: .white.opacity(Whimsy.dim),
                              location: 0.66),
                        .init(color: .white.opacity(Whimsy.dim),
                              location: 1),
                    ], startPoint: .leading, endPoint: .trailing)
                        .frame(width: w * 3)
                        .offset(x: travel * w)
                }
            }
            .onAppear {
                withAnimation(.linear(duration: Whimsy.sweep)
                    .repeatForever(autoreverses: false)) { travel = 0 }
            }
    }

}

private struct Shimmer: ViewModifier {

    let active: Bool

    func body(content: Content) -> some View {
        if active {
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { ctx in
                let t = ctx.date.timeIntervalSinceReferenceDate
                content.opacity(0.85 + 0.15 * sin(t * 2 * .pi / 3.6))
            }
        } else {
            content
        }
    }

}
