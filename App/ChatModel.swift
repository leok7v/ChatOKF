import Chat
import Foundation
import ImageIO
import LLM
import MD
import OSLog
import UniformTypeIdentifiers

@MainActor @Observable final class ChatModel {

    enum ImageBudget: String, CaseIterable, Identifiable {
        case XS, S, M, L, XL
        var id: String { rawValue }
        var tokens: Int {
            switch self {
            case .XS: return 70
            case .S: return 140
            case .M: return 280
            case .L: return 560
            case .XL: return 1120
            }
        }
    }

    static let defaultSystemPrompt = "You are a helpful assistant."

    var messages: [Message] = []
    var readOnly = false
    var currentConversationId: UUID?
    var generatedTitle: String?
    var theme: AppTheme = {
        AppTheme(rawValue:
            UserDefaults.standard.string(forKey: "theme") ?? "") ?? .system
    }() {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "theme") }
    }
    var textZoom: Int = ChatModel.clampZoom(
        UserDefaults.standard.integer(forKey: "textZoom")) {
        didSet { UserDefaults.standard.set(textZoom, forKey: "textZoom") }
    }
    static let zoomLimit = 2

    static func clampZoom(_ notches: Int) -> Int {
        min(max(notches, -zoomLimit), zoomLimit)
    }

    var textScale: CGFloat { ChatModel.zoomScale(textZoom) }

    static func zoomScale(_ notch: Int) -> CGFloat {
        1 + CGFloat(notch) / 10
    }

    static func zoomNotch(nearest scale: CGFloat) -> Int {
        clampZoom(Int(((min(max(scale, 0.5), 2) - 1) * 10).rounded()))
    }

    func flashZoom(_ notch: Int) {
        flashHUD("Zoom \(Int(ChatModel.zoomScale(notch) * 100))%")
    }

    var canZoomIn: Bool { textZoom < ChatModel.zoomLimit }
    var canZoomOut: Bool { textZoom > -ChatModel.zoomLimit }
    var atDefaultZoom: Bool { textZoom == 0 }

    func zoomIn() { if canZoomIn { textZoom += 1 } }
    func zoomOut() { if canZoomOut { textZoom -= 1 } }
    func resetZoom() { textZoom = 0 }

    var input = ""
    var caret = 0
    var attachedImages: [ImageAttachment] = []
    var attachedDocs: [Doc] = []
    private(set) var converting = 0
    var attachedClips: [ClipAttachment] = []
    var status = "loading model..."
    var ready: Bool { session.hasSession && !compiling }
    var busy: Bool { genTask != nil || session.metaTaskRunning }

    var lookingAt: VideoPeek?
    var watching = false
    var inTurn: Bool { busy || listening || speech.engaged }

    var voiceReady: Bool {
        lastTurnSpoken && canAttachAudio && !busy && !listening
            && !speech.engaged && input.isEmpty
    }

    private(set) var lastTurnSpoken = false
    var statsLabel = ""

    var modelShape: ModelShape? { session.modelShape }

    struct Tower: Identifiable {
        let id: Int
        let symbol: String
        let weight: String
    }

    var modelTowers: [Tower] {
        var out: [Tower] = []
        if let shape = modelShape {
            for (i, t) in shape.towers.enumerated() {
                out.append(Tower(id: i, symbol: ChatModel.towerSymbol(t.name),
                                 weight: ChatModel.weight(t.bytes)))
            }
        }
        return out
    }

    private static func towerSymbol(_ name: String) -> String {
        var out = "text.alignleft"
        if name == "vision" {
            out = "eye"
        } else if name == "audio" {
            out = "waveform.path"
        }
        return out
    }

    var modelFacts: [String] {
        var out: [String] = []
        if let shape = modelShape {
            if shape.trainedContext > 0 {
                var fact = "ctx " + ChatModel.tokens(shape.trainedContext)
                if shape.embedding > 0 {
                    fact += " \u{00D7} " + shape.embedding.formatted(.number)
                }
                out.append(fact)
            } else if shape.embedding > 0 {
                out.append(shape.embedding.formatted(.number) + " dim")
            }
        }
        return out
    }

    private static func tokens(_ n: Int) -> String {
        n >= 1024 && n % 1024 == 0 ? "\(n / 1024)K" : n.formatted(.number)
    }

    private static func weight(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        return gb >= 1 ? String(format: "%.1f GB", gb)
                       : String(format: "%.0f MB",
                                Double(bytes) / 1_048_576)
    }
    let speech = VoiceSession()
    var listening = false
    var heardSeconds = 0.0
    var hearingSpeech = false
    var speechLevel = 0.0
    var prefilling = false
    var consulting = false
    var thinkStatus = "Thinking"
    var thinkLabel = "Thinking"
    var eulaAccepted = UserDefaults.standard.bool(forKey: "eulaAccepted")
    var accepted = UserDefaults.standard.bool(forKey: "disclaimerAccepted")
    private static func startModel() -> String {
        let saved = UserDefaults.standard.string(forKey: "modelName")
            ?? Models.start
        return Models.all.contains(saved) ? saved : Models.start
    }
    var modelName: String = ChatModel.startModel()
    var downloadName: String? = nil
    var downloading = false
    var downloadDone: Int64 = 0
    var downloadTotal: Int64 = 0
    @ObservationIgnored private var downloadBase: Int64 = -1

    func observeDownload(_ s: HubFetch.Status, set: URL) {
        if downloadBase < 0 {
            downloadBase = s.done
        }
        downloadDone = s.done
        downloadTotal = s.total
    }

    var downloadFraction: Double {
        downloadTotal > 0 ? Double(downloadDone) / Double(downloadTotal) : 0
    }

    var downloadCounter: String {
        String(format: "%.1f of %.1f GB",
               Double(downloadDone) / 1_000_000_000,
               Double(downloadTotal) / 1_000_000_000)
    }

    var canSend: Bool {
        let text = input.trimmingCharacters(
            in: .whitespacesAndNewlines)
        let has = !text.isEmpty || !attachedImages.isEmpty
            || !attachedDocs.isEmpty || !attachedClips.isEmpty
        return ready && !busy && has
    }

    var typing: Bool { !input.isEmpty }

    var attachmentWarning: String? {
        let docTokens = attachedDocs.reduce(0) { sum, doc in
            sum + doc.content.utf8.count / 4
        }
        let total = attachedImages.count * session.perImageTokens + docTokens
        var result: String? = nil
        if total >= Self.warnTokens {
            let secs = lastPP > 0 ? Double(total) / lastPP : 0
            let time = secs >= 2 ? " (~\(Int(secs.rounded()))s)" : ""
            result = "Large attachment\(time); this may take a moment."
        }
        return result
    }

    var canAttachImages: Bool { session.modalities.images }
    var canAttachAudio: Bool { session.modalities.audio }
    var canAttachVideo: Bool {
        session.modalities.video
            && !attachedClips.contains { c in c.isVideo }
    }

    var attachableTypes: [UTType] {
        var out: [UTType] = [.plainText, .pdf]
        out += Docs2md.readable.compactMap { ext in
            UTType(filenameExtension: ext)
        }
        if canAttachImages { out.append(.image) }
        if canAttachAudio { out.append(.audio) }
        if canAttachVideo { out.append(.movie) }
        return out
    }

    var hasAttachments: Bool {
        !attachedImages.isEmpty || !attachedClips.isEmpty
            || !attachedDocs.isEmpty
    }

    var attachGlyph: String {
        var out = "plus"
        if !attachedClips.isEmpty {
            out = attachedClips.contains(where: { c in c.isVideo })
                ? "film.fill" : "waveform"
        } else if !attachedImages.isEmpty {
            out = "photo.fill"
        } else if !attachedDocs.isEmpty {
            out = "doc.text.fill"
        }
        return out
    }

    var attachHelp: String {
        var out = "Attach a document"
        if canAttachAudio {
            out = "Attach an image, sound, video or document"
        } else if canAttachImages {
            out = "Attach an image or document"
        }
        return out
    }

    var modelSupportsThinking: Bool { session.modelSupportsThinking }

    var thinkingActive: Bool { thinking && modelSupportsThinking }

    var downloadSizeText: String {
        let bytes = downloadName.flatMap { name in
            ModelCatalog.source(name)?.bytes
        } ?? 0
        return ByteCountFormatter.string(fromByteCount: bytes,
                                         countStyle: .file)
    }

    var compiling = false

    var loadError: String? = nil

    var wikipedia = true {
        didSet { session.wikipedia = wikipedia }
    }
    var webAccess = true {
        didSet { session.webAccess = webAccess }
    }

    enum Access { case offline, wikipedia, full }
    var accessState: Access {
        webAccess ? .full : (wikipedia ? .wikipedia : .offline)
    }

    func cycleAccess() {
        switch accessState {
        case .offline: setAccess(wikipedia: true, web: false)
        case .wikipedia: setAccess(wikipedia: true, web: true)
        case .full: setAccess(wikipedia: false, web: false)
        }
        switch accessState {
        case .offline: flashHUD("Airplane Mode")
        case .wikipedia: flashHUD("Wikipedia Only")
        case .full: flashHUD("Web Access")
        }
    }

    struct Flash: Equatable {
        let text: String
        let prominent: Bool
    }

    private(set) var hud: Flash?
    @ObservationIgnored private var hudTask: Task<Void, Never>?

    private func flashHUD(_ text: String, prominent: Bool = false,
                          seconds: Double = 3) {
        hudTask?.cancel()
        hud = Flash(text: text, prominent: prominent)
        hudTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            if !Task.isCancelled { hud = nil }
        }
    }

    func setSearchProvider(_ provider: SearchProvider, _ on: Bool) {
        provider.set(on)
        session.pushTools()
    }

    func setAccess(wikipedia wiki: Bool, web: Bool) {
        wikipedia = wiki
        webAccess = web
        session.pushTools()
    }
    var thinking = true
    var systemPrompt: String = UserDefaults.standard
        .string(forKey: "systemPrompt") ?? ChatModel.defaultSystemPrompt {
        didSet {
            UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt")
            session.systemPrompt = systemPrompt
            Session.wipePrecook()
        }
    }
    var imageBudget: ImageBudget = {
        let raw = UserDefaults.standard.string(forKey: "imageBudget") ?? ""
        return ImageBudget(rawValue: raw) ?? .M
    }() {
        didSet {
            UserDefaults.standard.set(imageBudget.rawValue,
                                      forKey: "imageBudget")
        }
    }
    var renderMarkdown: Bool = UserDefaults.standard
        .object(forKey: "renderMarkdown") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(renderMarkdown,
                                      forKey: "renderMarkdown")
        }
    }
    var confirmDeleteConversation: Bool = UserDefaults.standard
        .object(forKey: "confirmDeleteConversation") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(confirmDeleteConversation,
                                      forKey: "confirmDeleteConversation")
        }
    }
    var statusLine: Bool = UserDefaults.standard.bool(forKey: "statusLine") {
        didSet { UserDefaults.standard.set(statusLine, forKey: "statusLine") }
    }
    var showSettings = false
    var showDebug = false
    var settingsCategory: SettingsView.Category = .systemPrompt
    var optionDown = false
    var unlocked: Bool { statusLine && optionDown }
    var traceEvents: [TraceEvent] = []
    var tracePath: String { session.tracePath }
    var diagPath: String { session.diagPath }
    private static let traceCap = 2000
    // A static logit bias on reasoning branch-openers while thinking. Robust
    // across 0.5-4.0. TODO: surface as a setting.

    enum ThinkBudget: String, CaseIterable, Identifiable {
        case XS, S, M, L, XL
        var id: String { rawValue }
        var seconds: Double {
            switch self {
            case .XS: return 10
            case .S: return 20
            case .M: return 45
            case .L: return 90
            case .XL: return 180
            }
        }
    }

    var thinkBudget: ThinkBudget = {
        let raw = UserDefaults.standard.string(forKey: "thinkBudget") ?? ""
        return ThinkBudget(rawValue: raw) ?? .L
    }() {
        didSet {
            UserDefaults.standard.set(thinkBudget.rawValue,
                                      forKey: "thinkBudget")
        }
    }

    enum ReasoningEffort: String, CaseIterable, Identifiable {
        case low = "Low", medium = "Medium", high = "High"
        var id: String { rawValue }
        func wire(_ levels: [String]) -> String {
            effortSpelling(rawValue.lowercased(),
                           slot: Self.allCases.firstIndex(of: self) ?? 0,
                           in: levels)
        }
    }

    private static func effortKey(_ name: String) -> String {
        "reasoningEffort.\(name)"
    }

    private static func effort(for name: String) -> ReasoningEffort {
        let raw = UserDefaults.standard.string(forKey: effortKey(name)) ?? ""
        return ReasoningEffort(rawValue: raw) ?? .medium
    }

    var modelSupportsReasoningEffort: Bool {
        session.modelSupportsReasoningEffort
    }

    private(set) var reasoningEffort: ReasoningEffort =
        ChatModel.effort(for: ChatModel.startModel())

    func setReasoningEffort(_ level: ReasoningEffort) {
        if level != reasoningEffort {
            reasoningEffort = level
            UserDefaults.standard.set(level.rawValue,
                                      forKey: Self.effortKey(modelName))
            Session.wipePrecook()
            session.pushReasoningEffort(level.wire(session.effortLevels))
            flashHUD(messages.isEmpty
                     ? "Reasoning: \(level.rawValue)"
                     : "Reasoning: \(level.rawValue), from the next chat")
        }
    }

    private static func tgKey(_ name: String) -> String { "tg.\(name)" }

    private static func rate(_ v: Double) -> String { Session.rate(v) }

    var thinkTokenCap: Int {
        let raw = thinkBudget.seconds * session.measuredTG
        return max(300, Int((raw / 100).rounded()) * 100)
    }

    let session = Session(
        modelName: ChatModel.startModel(),
        systemPrompt: UserDefaults.standard.string(forKey: "systemPrompt")
            ?? ChatModel.defaultSystemPrompt)
    private var genTask: Task<Void, Never>?
    private static let benchPrompt = Flags.value("bench-prompt") ?? ""
    private static let benchTokens = Flags.int("bench-tokens") ?? 128
    private static let benchCool = Flags.int("bench-cool") ?? 90
    @ObservationIgnored private var benchTask: Task<Void, Never>?
    @ObservationIgnored private var benchPieces = 0
    @ObservationIgnored private var benchMetrics: TurnMetrics?
    // The last NON-ZERO rates: the ticker samples every 400ms from turn start,
    // where tg is still 0, so the raw metric blinks 0.0 at the reader.
    private var lastPP = 0.0
    private var lastTG = 0.0
    @ObservationIgnored private var phaseStart = Date()

    func acceptEULA() {
        eulaAccepted = true
        UserDefaults.standard.set(true, forKey: "eulaAccepted")
        load(name: modelName)
    }

    func accept() {
        accepted = true
        UserDefaults.standard.set(true, forKey: "disclaimerAccepted")
    }

    var gemmaTermsAccepted = GemmaTerms.accepted

    func acceptGemmaTerms() {
        GemmaTerms.accept()
        gemmaTermsAccepted = true
    }

    func load(name: String) {
        if !compiling {
            loadError = nil
            Session.pruneUnavailable()
            let setDir = ModelCatalog.localSet(name, in: Bundle.modelStore())
            if let setDir, ModelCatalog.isComplete(setDir) {
                if let path = ModelCatalog.ggufPath(name,
                                                    in: Bundle.modelStore()) {
                    compiling = true
                    Task { @MainActor in
                        await self.loadReady(name: name, path: path)
                    }
                } else {
                    loadError = Session.prepFailed
                }
            } else if ModelCatalog.source(name) != nil {
                downloadName = name
                status = "download required"
            } else {
                loadError = "Model \(name) is not available."
            }
        }
    }

    private func sessionConfig() -> Session.SessionConfig {
        Session.SessionConfig(
            thinking: thinking,
            reasoningEffortRaw: reasoningEffort.rawValue.lowercased(),
            reasoningEffortSlot:
                ReasoningEffort.allCases.firstIndex(of: reasoningEffort) ?? 0,
            thinkTokenCap: thinkTokenCap)
    }

    private func loadReady(name: String, path: String) async {
        compiling = true
        phaseStart = Date()
        loadError = nil
        let error = await session.buildGguf(name: name, path: path)
        if let error {
            compiling = false
            loadError = error
        } else {
            Instrument.timed("makeSession") {
                session.makeSession(sessionConfig()) { [weak self] e in
                    self?.recordTrace(e)
                }
            }
            Instrument.timed("show chat") {
                compiling = false
                status = ""
            }
            session.primeSession(thinkingActive: thinkingActive)
            if !Self.benchPrompt.isEmpty, benchTask == nil {
                benchTask = Task { @MainActor in await self.runBench() }
            }
        }
    }

    private func runBench() async {
        let text = Self.benchPrompt == "512" ? Self.benchText
                                             : Self.benchPrompt
        await session.awaitPrimed()
        for arm in ["drafted", "plain"] {
            if arm == "plain" {
                newChat()
                await genTask?.value
                await session.awaitPrimed()
                await session.pushSpeculation(false)
                Diag.shared.report("[bench] cooling \(Self.benchCool)s")
                try? await Task.sleep(for: .seconds(Self.benchCool))
            }
            benchPieces = 0
            benchMetrics = nil
            input = text
            caret = input.utf16.count
            send()
            await genTask?.value
            if let m = benchMetrics {
                Diag.shared.report(String(
                    format: "[bench] %@ %@ pp %.1f tg %.1f tokens %d end=%@",
                    arm, modelName, m.pp, m.tg,
                    m.thinkTokens + m.contentTokens, m.endReason))
            }
        }
        Diag.shared.report("[bench] done")
        benchTask = nil
    }

    private func benchPiece() {
        if benchTask != nil {
            benchPieces += 1
            if benchPieces == Self.benchTokens { session.requestStop() }
        }
    }

    func retry() {
        loadError = nil
        load(name: modelName)
    }

    func confirmDownload() {
        if let name = downloadName, let src = ModelCatalog.source(name) {
            commitSwitch(name)
            downloadName = nil
            downloading = true
            downloadDone = 0
            downloadTotal = 0
            downloadBase = -1
            phaseStart = Date()
            status = "downloading \(name)…"
            let dest = Bundle.modelStore().appendingPathComponent(name)
            let setDir = dest.appendingPathComponent(src.revision)
            Task { @MainActor in
                let failure = await session.fetch(name: name) { s in
                    Task { @MainActor in self.observeDownload(s, set: setDir) }
                }
                downloading = false
                if failure == nil {
                    Self.saveSeconds("download", name,
                        Date().timeIntervalSince(self.phaseStart))
                    if let path = ModelCatalog.ggufPath(
                        name, in: Bundle.modelStore()) {
                        await self.loadReady(name: name, path: path)
                    } else {
                        self.status = failure ?? "download failed, check "
                            + "your connection"
                    }
                } else {
                    self.status = failure!
                }
            }
        }
    }

    func switchModel(_ name: String) {
        if name != modelName, !busy, Models.all.contains(name) {
            if Session.isOnDisk(name) {
                commitSwitch(name)
                status = "loading model…"
                load(name: name)
            } else {
                downloadName = name
            }
        }
    }

    func cancelDownload() {
        let fallback = ready ? nil : downloadedFallback()
        downloadName = nil
        if let fallback {
            commitSwitch(fallback)
            status = "loading model…"
            load(name: fallback)
        }
    }

    var canCancelDownload: Bool { ready || downloadedFallback() != nil }

    private func downloadedFallback() -> String? {
        Session.isOnDisk(Models.start)
            ? Models.start : Models.all.first { name in Session.isOnDisk(name) }
    }

    // Named for this: the outgoing conversation is SAVED before the switch
    // clears it, exactly as New Chat saves before starting one.
    private func commitSwitch(_ name: String) {
        commitCurrent()
        genTask?.cancel()
        session.releaseSession()
        for img in attachedImages {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages = []
        attachedClips = []
        attachmentSerials = [:]
        clampCaret()
        downloading = false
        compiling = false
        loadError = nil
        // The identity goes with the messages, or the next conversation
        // commits into this one's id and overwrites it.
        messages = []
        traceEvents = []
        currentConversationId = nil
        readOnly = false
        generatedTitle = nil
        followupHint = ""
        statsLabel = ""
        // Not the outgoing model's rates; pp has no EMA, so it reads "-".
        lastPP = 0
        lastTG = Session.storedTG(name)
        modelName = name
        session.modelName = name
        UserDefaults.standard.set(name, forKey: "modelName")
        usedSamples = ChatModel.usedSamples(for: name)
        reasoningEffort = ChatModel.effort(for: name)
    }

    private(set) var diskRevision = 0

    func isDownloaded(_ name: String) -> Bool { Session.isOnDisk(name) }

    func deleteModel(_ name: String) {
        if name != modelName, !busy, !downloading {
            Session.erase(name)
            diskRevision += 1
        }
    }

    func requestDownload(_ name: String) {
        if !busy, !downloading, !Session.isOnDisk(name),
           ModelCatalog.source(name) != nil {
            downloadName = name
        }
    }

    private func recordTrace(_ e: TraceEvent) {
        traceEvents.append(e)
        if traceEvents.count > Self.traceCap {
            traceEvents.removeFirst(traceEvents.count - Self.traceCap)
        }
    }

    func newChat() {
        Footprint.report(.load, "newChat begin")
        followupHint = ""
        lastTurnSpoken = false
        speech.stopSpeaking()
        commitCurrent()
        readOnly = false
        currentConversationId = nil
        generatedTitle = nil
        messages = []
        traceEvents = []
        statsLabel = ""
        attachmentSerials = [:]
        let running = genTask
        genTask = Task { @MainActor in
            if running != nil {
                session.requestStop()
                running?.cancel()
                _ = await running?.value
            }
            await session.newChatEngine(sessionConfig()) { [weak self] e in
                self?.recordTrace(e)
            }
            genTask = nil
            Footprint.report(.load, "newChat end")
        }
    }

    private func runMetaTurns() {
        let chars = messages.reduce(0) { sum, m in sum + m.text.count }
        let titled = !readOnly && generatedTitle == nil
            && messages.count >= 2 && chars > 200
        if !readOnly, titled || offersFollowupHint {
            session.runMetaTurns(
                titled: titled, wantsFollowup: offersFollowupHint,
                onTitle: { [weak self] t in
                    self?.generatedTitle = t
                    self?.commitCurrent()
                },
                onFollowup: { [weak self] hint in
                    self?.followupHint = hint
                })
        }
    }

    var suggestFollowups: Bool = UserDefaults.standard
        .object(forKey: "suggestFollowups") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(suggestFollowups,
                                      forKey: "suggestFollowups")
            if !suggestFollowups { followupHint = "" }
        }
    }

    var canSuggestFollowups: Bool { modelName != Models.fallback }

    var offersFollowupHint: Bool {
        suggestFollowups && canSuggestFollowups
    }

    var followupHint = ""

    func acceptFollowupHint() {
        if !followupHint.isEmpty {
            input = followupHint
            caret = input.utf16.count
        }
    }

    func toggleSettings() {
        showSettings.toggle()
        if showSettings { showDebug = false }
    }

    func openSettings() {
        showDebug = false
        showSettings = true
    }

    func revealHidden() {
        if statusLine {
            optionDown = true
            settingsCategory = .misc
            openSettings()
        }
    }

    func closeSettings() {
        showSettings = false
        if isOS {
            optionDown = false
            settingsCategory = .systemPrompt
        }
    }

    func toggleDebug() {
        showDebug.toggle()
        if showDebug { showSettings = false }
    }

    func toggleThinking() {
        if modelSupportsThinking {
            thinking.toggle()
            let on = thinking
            let fresh = messages.isEmpty
            let hud = thinkingFlash(on, fresh: fresh)
            Task { @MainActor in
                await session.pushThinking(on, resetIfFresh: fresh)
            }
            flashHUD(hud)
        }
    }

    private func thinkingFlash(_ on: Bool, fresh: Bool) -> String {
        var out = on ? "Thinking: On" : "Thinking: Off"
        if on, !fresh, !session.primedThinking {
            out = "Thinking: On, from the next chat"
        }
        return out
    }

    func quickAnswer() {
        session.requestQuickAnswer()
    }

    func attachImage(_ data: Data, name: String, at offset: Int) {
        let dup = attachedImages.contains { img in img.data == data }
        if canAttachImages, attachedImages.count < Self.maxImages, !dup {
            let unique = uniqueName(serialName("Image"))
            let thumb = VisionPreprocess.thumbnail(data, maxPx: 96)
            attachedImages.append(
                ImageAttachment(name: unique, file: name, data: data,
                                thumbnail: thumb))
            insertRef(unique, at: offset)
        }
    }

    func attachClip(_ url: URL, isVideo: Bool, at offset: Int,
                    file: String? = nil) {
        let allowed = isVideo ? canAttachVideo : canAttachAudio
        let dup = attachedClips.contains { c in c.url == url }
        let seenVideo = isVideo
            && attachedClips.contains { c in c.isVideo }
        if allowed, attachedClips.count < Self.maxClips, !dup, !seenVideo {
            let unique = uniqueName(serialName(isVideo ? "Video" : "Audio"))
            let clip = ClipAttachment(
                name: unique, file: file ?? url.lastPathComponent, url: url,
                isVideo: isVideo, thumbnail: nil)
            attachedClips.append(clip)
            insertRef(unique, at: offset)
            if isVideo { loadPoster(clip.id, url) }
        }
    }

    private func loadPoster(_ id: UUID, _ url: URL) {
        Task { @MainActor in
            let data = await VideoFrames.poster(url: url, maxPx: 640)
            if let data, let cg = VisionPreprocess.image(data) {
                adoptPoster(cg, id: id, url: url)
            }
        }
    }

    private func adoptPoster(_ cg: CGImage, id: UUID, url: URL) {
        if let at = attachedClips.firstIndex(where: { c in c.id == id }) {
            attachedClips[at].thumbnail = cg
        }
        for i in messages.indices
        where messages[i].posters.isEmpty && messages[i].clips.contains(url) {
            messages[i].posters = [cg]
        }
    }

    @ObservationIgnored private var attachmentSerials: [String: Int] = [:]

    private func serialName(_ kind: String) -> String {
        let n = (attachmentSerials[kind] ?? 0) + 1
        attachmentSerials[kind] = n
        return "\(kind)\(n)"
    }

    func attachDoc(_ name: String, _ content: String, at offset: Int,
                   from url: URL? = nil) {
        let capped = Self.capDoc(content, docBudget.bytes)
        if !attachedDocs.contains(where: { d in d.content == capped }) {
            let unique = uniqueName(name)
            attachedDocs.append(
                Doc(name: unique, content: capped, url: url,
                    short: content.utf8.count > docBudget.bytes))
            insertRef(unique, at: offset)
        }
    }

    static let pasteAttachBytes = 2048

    func attachPastedText(_ text: String, at offset: Int) -> Bool {
        let room = attachedDocs.count < Self.maxDocs
        if room {
            let name = uniqueName(serialName("Text"))
            attachDoc(name, text, at: offset,
                      from: ChatModel.keepText(text, name + ".txt"))
        }
        return room
    }

    private static func keepText(_ text: String, _ file: String) -> URL? {
        let fm = FileManager.default
        let dir = Session.attachments.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let to = dir.appendingPathComponent(file)
        let wrote = (try? text.write(to: to, atomically: true,
                                     encoding: .utf8)) != nil
        return wrote ? to : nil
    }

    private func convertDoc(_ from: URL, _ name: String) {
        if let kept = ChatModel.keep(from, name) {
            converting += 1
            flashHUD("Reading \(name)")
            let t0 = Date()
            Task { @MainActor in
                let text = await ChatModel.markdown(of: kept)
                converting -= 1
                ChatModel.read(name, text, t0,
                               docBudget.bytes)
                if let text {
                    attachDoc(name, text, at: caret, from: kept)
                } else {
                    try? FileManager.default.removeItem(at: kept)
                    flashHUD("Cannot read \(name)")
                }
            }
        }
    }

    private static func read(_ name: String, _ text: String?,
                             _ since: Date, _ limit: Int) {
        let bytes = text?.utf8.count ?? 0
        let cut = bytes > limit ? ", TRUNCATED to \(limit)" : ""
        Diag.shared.report(.turn, String(
            format: "attach document %@ -> %d bytes%@ (%.1fs)", name, bytes,
            cut, Date().timeIntervalSince(since)))
    }

    private static func keep(_ from: URL, _ name: String) -> URL? {
        let fm = FileManager.default
        let dir = Session.attachments.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let to = dir.appendingPathComponent(name)
        return (try? fm.copyItem(at: from, to: to)) != nil ? to : nil
    }

    private static func markdown(of url: URL) async -> String? {
        await Task.detached(priority: .userInitiated) {
            url.pathExtension.lowercased() == "pdf"
                ? try? await Pdf2md.markdown(of: url)
                : try? Docs2md.markdown(of: url)
        }.value
    }

    private static func capDoc(_ content: String, _ limit: Int) -> String {
        var out = content
        if content.utf8.count > limit {
            var take = limit
            var head: String? = nil
            while head == nil && take > 0 {
                head = String(content.utf8.prefix(take))
                take -= 1
            }
            out = (head ?? "")
                + "\n[... truncated at \(limit >> 10) KB]"
        }
        return out
    }

    func clearImage(_ id: UUID) {
        if let img = attachedImages.first(where: { i in i.id == id }) {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages.removeAll { i in i.id == id }
        clampCaret()
    }

    func clearClip(_ id: UUID) {
        if let clip = attachedClips.first(where: { c in c.id == id }) {
            input = AttachmentRefs.scrub(clip.name, from: input)
        }
        attachedClips.removeAll { c in c.id == id }
        clampCaret()
    }

    func clearDoc(_ id: UUID) {
        if let doc = attachedDocs.first(where: { d in d.id == id }) {
            input = AttachmentRefs.scrub(doc.name, from: input)
        }
        attachedDocs.removeAll { d in d.id == id }
        clampCaret()
    }

    func reconcileAttachments() {
        let live = Set(AttachmentRefs.names(in: input))
        for doc in attachedDocs where !live.contains(doc.name) {
            input = AttachmentRefs.scrub(doc.name, from: input)
        }
        attachedDocs.removeAll { d in !live.contains(d.name) }
        for img in attachedImages where !live.contains(img.name) {
            input = AttachmentRefs.scrub(img.name, from: input)
        }
        attachedImages.removeAll { i in !live.contains(i.name) }
        for clip in attachedClips where !live.contains(clip.name) {
            input = AttachmentRefs.scrub(clip.name, from: input)
        }
        attachedClips.removeAll { c in !live.contains(c.name) }
        clampCaret()
    }

    private func insertRef(_ name: String, at offset: Int) {
        let r = AttachmentRefs.insert(name, into: input, at: offset)
        input = r.text
        caret = r.caret
    }

    private func clampCaret() {
        caret = max(0, min(caret, input.utf16.count))
    }

    private func uniqueName(_ name: String) -> String {
        var candidate = name
        var n = 2
        while attachedDocs.contains(where: { d in d.name == candidate })
            || attachedImages.contains(where: { i in i.name == candidate })
            || attachedClips.contains(where: { c in c.name == candidate }) {
            candidate = "\(name) (\(n))"
            n += 1
        }
        return candidate
    }

    static let imageExts: Set<String> =
        ["png", "jpg", "jpeg", "heic", "heif", "gif", "webp", "bmp", "tiff"]
    static let docExts: Set<String> = ["txt", "md", "markdown", "text"]
    static let convertExts: Set<String> = Set(["pdf"] + Docs2md.readable)
    static func clipKind(_ url: URL) -> Bool? {
        let type = UTType(filenameExtension:
                            url.pathExtension.lowercased())
        var out: Bool? = nil
        if type?.conforms(to: .movie) == true {
            out = true
        } else if type?.conforms(to: .audio) == true {
            out = false
        }
        return out
    }
    static let maxDocs = 6
    static let maxImages = 4
    static let maxClips = 2
    enum DocBudget: String, CaseIterable, Identifiable {
        case XS, S, M, L, XL
        var id: String { rawValue }
        var bytes: Int {
            switch self {
            case .XS: return 8 << 10
            case .S: return 16 << 10
            case .M: return 32 << 10
            case .L: return 64 << 10
            case .XL: return 128 << 10
            }
        }
        var pages: Int { bytes / 2000 }
    }

    var docBudget: DocBudget = {
        let raw = UserDefaults.standard.string(forKey: "docBudget") ?? ""
        return DocBudget(rawValue: raw) ?? .M
    }() {
        didSet {
            UserDefaults.standard.set(docBudget.rawValue, forKey: "docBudget")
        }
    }
    static let warnTokens = 4000

    func handleDrop(_ urls: [URL], at offset: Int) {
        caret = max(0, min(offset, input.utf16.count))
        var refused: [String] = []
        for url in urls where ready {
            let ext = url.pathExtension.lowercased()
            let scoped = url.startAccessingSecurityScopedResource()
            let before = attachedImages.count + attachedDocs.count
                + attachedClips.count + converting
            if Self.imageExts.contains(ext),
               attachedImages.count < Self.maxImages,
               let data = try? Data(contentsOf: url) {
                attachImage(data, name: url.lastPathComponent, at: caret)
            } else if Self.docExts.contains(ext),
                      attachedDocs.count < Self.maxDocs,
                      let text = try? String(contentsOf: url, encoding: .utf8) {
                attachDoc(url.lastPathComponent, text, at: caret,
                          from: Self.keep(url, url.lastPathComponent))
            } else if Self.convertExts.contains(ext),
                      attachedDocs.count < Self.maxDocs {
                convertDoc(url, url.lastPathComponent)
            } else if let isVideo = Self.clipKind(url) {
                attachClip(url, isVideo: isVideo, at: caret)
            }
            let after = attachedImages.count + attachedDocs.count
                + attachedClips.count + converting
            if after == before { refused.append(ext.isEmpty ? "file" : ext) }
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        if !refused.isEmpty { flashHUD(refusedText(refused)) }
    }

    private func refusedText(_ kinds: [String]) -> String {
        let what = Set(kinds).sorted()
            .map { k in "." + k }.joined(separator: " ")
        var out = "Cannot read \(what)"
        if kinds.contains(where: { k in Self.imageExts.contains(k) }) {
            out = canAttachImages
                ? "Already at \(Self.maxImages) images"
                : "\(Models.display(modelName)) cannot see images"
        } else if kinds.contains(where: { k in Self.clipKind(
            URL(fileURLWithPath: "x." + k)) != nil }) {
            out = canAttachAudio
                ? "Already at \(Self.maxClips) clips"
                : "\(Models.display(modelName)) cannot hear or watch"
        }
        return out
    }

    private func promptFor(_ raw: String) -> String {
        AttachmentRefs.substitute(raw) { name in
            var out = "@\(name)"
            if let doc = self.attachedDocs.first(
                where: { d in d.name == name }) {
                let md = doc.name.hasSuffix(".md")
                    || doc.name.hasSuffix(".markdown")
                out = "\n\n\(doc.name):\n```\(md ? "markdown" : "")\n"
                    + "\(doc.content)\n```\n\n"
            } else if self.attachedImages.contains(
                            where: { img in img.name == name })
                || self.attachedClips.contains(
                            where: { c in c.name == name }) {
                out = ""
            }
            return out
        }
    }

    @ObservationIgnored private var mic: Microphone?
    @ObservationIgnored private var gate: SpeechGate?
    @ObservationIgnored private let heard = HeardSpeech()
    @ObservationIgnored private var rateInUse: Double = 1
    @ObservationIgnored private var endOfTurn: Task<Void, Never>?
    @ObservationIgnored private var micPhrases: Task<Void, Never>?
    private static let endOfTurnSilence = 1.5

    func voice() {
        if listening { endListening() } else { beginListening() }
    }

    private func beginListening() {
        speech.stopSpeaking()
        if ready, !busy, let rate = session.audioSampleRate,
           let maxSeconds = session.maxAudioSeconds {
            rateInUse = rate
            let gate = SpeechGate(rate: rate, maxSeconds: maxSeconds)
            let mic = Microphone(rate: rate)
            heard.clear()
            heardSeconds = 0
            Task { @MainActor in
                if await Microphone.permission() {
                    AudioSession.beginRecording()
                    do {
                        try mic.start { [heard] block in
                            heard.captured(block.count)
                            heard.observe(block)
                            heard.add(gate.push(block))
                        }
                        self.mic = mic
                        self.gate = gate
                        listening = true
                        micPhrases?.cancel()
                        micPhrases = phraseCycler()
                        watchForEndOfTurn()
                        Diag.shared.report(.voice,
                            "[mic] listening at \(Int(rate)) Hz "
                            + AudioSession.describe())
                    } catch {
                        AudioSession.endRecording()
                        Diag.shared.report("[mic] FAILED to start: \(error)")
                        flashHUD("\(error)")
                    }
                } else {
                    flashHUD("Microphone access is off")
                }
            }
        }
    }

    private func watchForEndOfTurn() {
        endOfTurn?.cancel()
        var ticks = 0
        endOfTurn = Task { @MainActor in
            while listening && !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(60))
                heardSeconds = heard.seconds
                hearingSpeech = gate?.hearing ?? false
                speechLevel = Double(gate?.level ?? 0)
                ticks += 1
                if ticks % 16 == 0, let gate, DiagGate.voice.on {
                    Diag.shared.report(.voice, String(
                        format: "[mic] .. %4.1fs open, hearing %@, bar %.5f, "
                            + "loudestFrame %.5f (%.1fx bar), peak %.5f, "
                            + "kept %.1fs, %d utt",
                        Double(heard.samples) / max(rateInUse, 1),
                        gate.hearing ? "YES" : "no ", gate.speechThreshold,
                        gate.loudestFrameEnergy,
                        gate.speechThreshold > 0
                            ? gate.loudestFrameEnergy / gate.speechThreshold
                            : 0,
                        heard.peak, heard.seconds,
                        heard.utteranceCount))
                }
                let quiet = Date().timeIntervalSince(heard.lastAt)
                if listening && heard.hasSpeech
                    && quiet >= ChatModel.endOfTurnSilence {
                    endListening()
                }
            }
        }
    }

    private func endListening() {
        endOfTurn?.cancel()
        endOfTurn = nil
        micPhrases?.cancel()
        micPhrases = nil
        mic?.stop()
        AudioSession.endRecording()
        heard.add(gate?.finish() ?? [])
        let bar = gate?.speechThreshold ?? 0
        mic = nil
        gate = nil
        listening = false
        hearingSpeech = false
        speechLevel = 0
        let said = heard.take()
        let secs = Double(heard.samples) / max(rateInUse, 1)
        Diag.shared.report(.voice, String(
            format: "[mic] stopped: %.1fs captured, %d utterance(s), "
                + "%.1fs of speech, bar %.5f, peak %.6f, rms %.6f, %@",
            secs, said.count,
            said.reduce(0.0) { sum, u in sum + u.seconds }, bar,
            heard.peak, heard.rms, AudioSession.describe()))
        if said.isEmpty {
            flashHUD(secs < 0.5 ? "No audio from the microphone"
                                : "Nothing was said")
        } else {
            flashHUD(Whimsical.current(.heard, hold: 0), prominent: true,
                     seconds: 2)
            sendSpoken(said)
        }
    }

    func stop() {
        speech.stopSpeaking()
        session.stop()
    }

    func factoryReset() {
        let d = UserDefaults.standard
        if let id = Bundle.main.bundleIdentifier {
            d.removePersistentDomain(forName: id)
        }
        ConversationStore.shared.eraseAll()
        let fm = FileManager.default
        try? fm.removeItem(at: Session.attachments)
        try? fm.removeItem(at: Bundle.modelStore())
        Diag.eraseCaches()
        quitApp()
    }

    private var canRunTurn: Bool { session.hasSession && ready && !busy }

    func send() {
        let attached = !attachedImages.isEmpty || !attachedClips.isEmpty
        if attached, canRunTurn {
            sendSoft()
        } else {
            sendText()
        }
    }

    private func sendSoft() {
        let raw = input
        let typed = promptFor(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var scrubbed = raw
        for item in attachedImages {
            scrubbed = AttachmentRefs.scrub(item.name, from: scrubbed)
        }
        for item in attachedClips {
            scrubbed = AttachmentRefs.scrub(item.name, from: scrubbed)
        }
        let display = AttachmentRefs.stripped(scrubbed)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if canRunTurn {
            let images = attachedImages
            let clips = attachedClips
            let docs = Session.refs(attachedDocs)
            input = ""
            caret = 0
            attachedImages = []
            attachedClips = []
            attachedDocs = []
            let cue: SpokenCue.Kind
            if clips.contains(where: { clip in clip.isVideo }) {
                cue = .watching
            } else if !images.isEmpty {
                cue = .looking
            } else {
                cue = .reading
            }
            if let (asked, events) = session.sendSoft(
                typed: typed, display: display, images: images, clips: clips,
                docs: docs, budget: imageBudget.tokens, labelled: true,
                placeholder: false,
                thinkTokenCap: thinkTokenCap, thinkingActive: thinkingActive) {
                beginTurn(asked, spoken: false, cue: cue, stage: .vision,
                         events: events)
            }
        }
    }

    private func sendSpoken(_ said: [SpeechGate.Utterance]) {
        if canRunTurn, let (asked, events) = session.sendSpoken(
            said: said, thinkTokenCap: thinkTokenCap,
            thinkingActive: thinkingActive) {
            beginTurn(asked, spoken: true, cue: .thinking, stage: .vision,
                     events: events)
        }
    }

    private func beginTurn(_ asked: Message, spoken: Bool,
                           cue: SpokenCue.Kind, stage: Whimsical.Stage,
                           events: AsyncStream<TurnEvent>) {
        followupHint = ""
        spokenTurn = spoken
        lastTurnSpoken = spoken
        prefilling = true
        activePrefillStage = stage
        messages.append(asked)
        messages.append(Message(fromUser: false, text: ""))
        let idx = messages.count - 1
        resetLiveBuffers()
        speech.beginTurn(cue: cue)
        let phrases = phraseCycler()
        genTask = Task { @MainActor in
            for await event in events {
                self.apply(event, at: idx)
            }
            phrases.cancel()
            self.genTask = nil
            self.prefilling = false
            self.consulting = false
            self.watching = false
            self.lookingAt = nil
        }
    }

    private func apply(_ event: TurnEvent, at idx: Int) {
        switch event {
        case .reasoning(let piece):
            benchPiece()
            if prefilling { prefilling = false }
            if consulting { consulting = false }
            speech.reasoningArrived(piece)
            if messages.indices.contains(idx) {
                liveReason += piece
                messages[idx].reasoningStream.append(piece)
                flushLive(idx)
            }
        case .answer(let piece):
            benchPiece()
            if prefilling { prefilling = false }
            if consulting { consulting = false }
            speech.answerArrived(piece)
            if messages.indices.contains(idx) {
                liveAnswer += piece
                messages[idx].answerStream.append(piece)
                flushLive(idx)
            }
        case .toolStarting:
            consulting = true
            prefilling = true
        case .toolRound(let round):
            applyToolRound(round, at: idx)
        case .looking(let peek):
            lookingAt = peek
            watching = true
        case .doneLooking:
            watching = false
        case .stats(let metrics):
            if !prefilling { refreshDocs() }
            applyStats(metrics)
        case .finished(let outcome, let metrics):
            speech.endTurn()
            flushLive(idx, force: true)
            finishDocs(idx)
            finishTurn(outcome, metrics, idx)
        case .cancelled:
            if messages.count >= 2 { messages.removeLast(2) }
        case .failed(let message):
            if messages.indices.contains(idx) { messages[idx].text = message }
        }
    }

    private func finishTurn(_ outcome: ChatSession.TurnOutcome,
                            _ metrics: TurnMetrics, _ idx: Int) {
        if benchTask != nil { benchMetrics = metrics }
        if outcome == .stopped {
            if messages.count >= 2 { messages.removeLast(2) }
        } else {
            applyStats(metrics)
            if messages.indices.contains(idx),
               Session.loopStopped(metrics,
                                   textEmpty: messages[idx].text.isEmpty) {
                messages[idx].loopStopped = true
            }
            commitCurrent()
            if benchTask == nil { runMetaTurns() }
        }
    }

    private func applyToolRound(_ event: ToolRoundEvent, at idx: Int) {
        if messages.indices.contains(idx) {
            if event.result == nil {
                speech.toolStarted(event.resolved ?? event.name)
            } else {
                speech.toolFinished()
            }
            let at = messages[idx].toolRounds.firstIndex { row in
                row.id == event.round
            }
            if let at {
                if event.result != nil {
                    messages[idx].toolRounds[at].result = event.result
                }
            } else {
                messages[idx].toolRounds.append(ToolRound(
                    id: event.round, emitted: event.name,
                    label: Session.toolLabel(event),
                    symbol: Session.toolSymbol(event),
                    args: Session.toolArgs(event), result: event.result))
            }
        }
    }

    private func sendText() {
        let raw = input
        let prompt = promptFor(raw)
        let display = AttachmentRefs.stripped(raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           canRunTurn {
            let stage: Whimsical.Stage = attachedDocs.isEmpty
                ? .prefill : .documents
            input = ""
            caret = 0
            let docs = Session.refs(attachedDocs)
            attachedDocs = []
            if let (asked, events) = session.sendText(
                prompt: prompt, display: display, docs: docs,
                thinkTokenCap: thinkTokenCap, thinkingActive: thinkingActive) {
                beginTurn(asked, spoken: false, cue: .thinking, stage: stage,
                         events: events)
            }
        }
    }

    private func applyStats(_ t: TurnMetrics) {
        if t.pp > 0 { lastPP = t.pp }
        if t.tg > 0 { lastTG = t.tg }
        if t.ctx > 0 {
            let tokens = thinkingActive
                ? "🤔 \(t.thinkTokens) 💬 \(t.contentTokens)"
                : "💬 \(t.thinkTokens + t.contentTokens)"
            statsLabel = "⇄ \(t.ctx.formatted(.number))  \(tokens) "
                + String(format: "🐏 %.1fGB", Self.footprintGiB())
                + " t/s: \(Session.rate(lastPP))/\(Session.rate(lastTG))"
        }
    }

    private func refreshDocs() {
        if let idx = messages.indices.last, !messages[idx].fromUser {
            Instrument.timed("refreshDocs") {
                messages[idx].answerDoc = messages[idx].answerStream.snapshot()
                messages[idx].reasoningDoc =
                    messages[idx].reasoningStream.snapshot()
            }
        }
    }

    @ObservationIgnored private var liveAnswer = ""
    @ObservationIgnored private var liveReason = ""
    @ObservationIgnored private var lastFlushNs: UInt64 = 0
    private static let flushIntervalNs: UInt64 = 100_000_000

    private func resetLiveBuffers() {
        liveAnswer = ""
        liveReason = ""
        lastFlushNs = 0
    }

    private func flushLive(_ idx: Int, force: Bool = false) {
        let now = DispatchTime.now().uptimeNanoseconds
        let due = force || now - lastFlushNs >= Self.flushIntervalNs
        if due, messages.indices.contains(idx) {
            lastFlushNs = now
            if messages[idx].text != liveAnswer {
                messages[idx].text = liveAnswer
            }
            if messages[idx].reasoning != liveReason {
                messages[idx].reasoning = liveReason
            }
        }
    }

    private func finishDocs(_ idx: Int) {
        if messages.indices.contains(idx), !messages[idx].fromUser {
            messages[idx].answerDoc = messages[idx].answerStream.finish()
            messages[idx].reasoningDoc =
                messages[idx].reasoningStream.finish()
        }
    }

    private var activePrefillStage: Whimsical.Stage = .prefill
    private var spokenTurn = false
    private var whimsicalStage: Whimsical.Stage {
        let out: Whimsical.Stage
        if listening {
            out = .listening
        } else if consulting {
            out = .consulting
        } else if prefilling {
            out = spokenTurn ? .listening : activePrefillStage
        } else {
            out = spokenTurn ? .mulling : .reasoning
        }
        return out
    }

    private func phraseCycler() -> Task<Void, Never> {
        Task { @MainActor in
            while (busy || listening) && !Task.isCancelled {
                let p = Whimsical.pair(whimsicalStage)
                thinkStatus = p.first
                thinkLabel = p.second
                try? await Task.sleep(for: .seconds(listening ? 2 : 5))
            }
        }
    }

    private static func footprintGiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride
                / MemoryLayout<natural_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self,
                                capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw,
                          &count)
            }
        }
        let gib = 1_073_741_824.0
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / gib : 0
    }

    var downloadETA: String? {
        let had = max(0, downloadBase)
        return Self.eta(phaseStart, downloadDone - had,
                        downloadTotal - had)
    }

    private static func eta(_ start: Date, _ done: Int64,
                            _ total: Int64) -> String? {
        var result: String? = nil
        let fraction = total > 0 ? Double(done) / Double(total) : 0
        if fraction > 0.02 {
            let elapsed = Date().timeIntervalSince(start)
            result = formatETA(elapsed * (1 - fraction) / fraction)
        }
        return result
    }

    static func formatETA(_ seconds: Double) -> String {
        let s = max(1, Int(seconds.rounded()))
        let m = Int((Double(s) / 60).rounded())
        let body = s < 60 ? "~\(s) seconds"
            : "~\(m) minute" + (m == 1 ? "" : "s")
        return "ETA: " + body
    }

    func estimatedMinutes(_ name: String) -> (download: Int, optimize: Int)? {
        var result: (download: Int, optimize: Int)? = nil
        let base = Models.fallback
        let dl0 = Self.savedSeconds("download", base)
        let opt0 = Self.savedSeconds("optimize", base)
        if dl0 > 0, opt0 > 0, let bytes = ModelCatalog.source(name)?.bytes,
           let base0 = ModelCatalog.source(base)?.bytes, base0 > 0 {
            let ratio = Double(bytes) / Double(base0)
            let dl = Self.savedSeconds("download", name)
            let opt = Self.savedSeconds("optimize", name)
            let gpuOnly = ModelCatalog.source(name)?.files != nil
            result = (Self.minutes(dl > 0 ? dl : dl0 * ratio),
                      gpuOnly ? 0
                              : Self.minutes(opt > 0 ? opt
                                                     : opt0 * pow(ratio, 1.3)))
        }
        return result
    }

    private static func minutes(_ seconds: Double) -> Int {
        max(1, Int((seconds / 60).rounded()))
    }

    private static func timeKey(_ phase: String, _ name: String) -> String {
        "seconds.\(phase).\(name)"
    }

    private static func saveSeconds(_ phase: String, _ name: String,
                                    _ seconds: Double) {
        UserDefaults.standard.set(seconds, forKey: timeKey(phase, name))
    }

    private static func savedSeconds(_ phase: String,
                                     _ name: String) -> Double {
        UserDefaults.standard.double(forKey: timeKey(phase, name))
    }

    static let sampleResearch =
        "Using Simple English Wikipedia as your primary source, " +
        "research Dark Matter and Dark Energy. Provide a highly factual, " +
        "structured summary of the current state of the art in the field. " +
        "Breakdown the explanation into: " +
        "1. Core Definitions, 2. Key Differences, and " +
        "3. Current Scientific Evidence. " +
        "Keep the language simple, direct, and free " +
        "of introductory or concluding remarks."
    static let sampleStory = "Describe everything you see in this "
        + "picture, then write a story based on it."
    static let sampleClip = "Describe what happens in this video."
    static let sampleEuler = "Using Euler's formula e^(ix) = cos(x) + "
        + "i*sin(x) and your calculator, explore: e^i (one radian around "
        + "the unit circle), Euler's identity e^(i*pi) + 1 = 0, i^i, "
        + "sqrt(i), and ln(-1). Compute each and explain briefly what it "
        + "means geometrically."
    static let sampleInterest = "A savings account starts with $1,000 and "
        + "earns 5% interest each year, so every year its balance "
        + "multiplies by 1.05. Use the calculator to find the balance "
        + "after 22 years."
    static let sampleCookies = "Bakers weigh every ingredient as a "
        + "percentage of the flour weight. In my cookie recipe, butter "
        + "is 65% of the flour. I have 350 grams of flour. Use the "
        + "calculator to find how many grams of butter I need."

    var calcSampleIsInterest: Bool { modelName != Models.fallback }

    static let benchText = "The town sits where the river slows and widens before it reaches the sea, and for most of its history that bend was the whole reason for the place. Barges could come no farther upstream, so grain, timber and wool from the valleys were unloaded on the stone quay, weighed in the long shed that still stands, and sold on to the coastal ships that waited below the bridge. The bridge itself has been rebuilt three times. The first was wood and burned in a winter fire that also took the mill; the second was stone with five arches, and it carried carts for two centuries until a flood pushed the middle pier off its footing; the third is the iron one that stands today, painted green every seven years by a crew that works from a floating platform and takes most of the summer to finish. Along the north bank the warehouses have become flats, workshops and one small museum, and the cranes that once swung sacks out of the holds are kept as ornaments, oiled but never used. The south bank was always the quieter side. Fishermen kept their boats there, drawn up on the shingle, and the row of low cottages behind the sea wall belonged to their families for as long as anyone kept records. A path runs from the last cottage along the top of the wall to the point, where a squat lighthouse marks the shoals, and from the point on a clear day the hills on the far side of the bay look close enough to walk to, though the ferry takes an hour. The market is held on the square on Wednesdays and Saturdays. Stalls sell vegetables from the terraces above the town, cheese from the two farms that still keep goats, bread from the bakery that opens before dawn, and, in season, the small striped fish that are eaten whole, grilled over charcoal, with nothing but salt and a squeeze of lemon. Visitors come for the fish and stay for the light, which painters have been trying to catch since the railway arrived and made the journey from the city a matter of hours rather than days. The station is at the top of the hill, and the walk down to the quay passes the church, the old school that is now the library, and a terrace of tall houses whose balconies face the water. In the evening the light comes low across the bay and turns the wet sand to copper, the gulls settle on the ridge of the shed, and the last ferry sounds its horn twice before it pulls away from the pier. Describe, in as much detail as you can, a full day in this town from the first bakery light to the last ferry, naming the people who work each part of it and what each of them does hour by hour."

    static let samplePicture: Data? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }

    static let sampleVideo: URL? = Bundle.main
        .url(forResource: "dogs-beach", withExtension: "mp4")
    static let samplePictureThumb: CGImage? = samplePicture.flatMap { data in
        VisionPreprocess.thumbnail(data, maxPx: 128)
    }

    static let sampleReport =
        "Which block gives the most fruit per tree, and what makes that "
        + "surprising?"

    static let samplePdf: URL? = Bundle.main
        .url(forResource: "harvest-report", withExtension: "pdf")

    static let sampleClipThumb: CGImage? = Bundle.main
        .url(forResource: "dogs-beach-poster", withExtension: "jpg")
        .flatMap { url in try? Data(contentsOf: url) }
        .flatMap { data in VisionPreprocess.thumbnail(data, maxPx: 128) }

    var alwaysShowSamples: Bool =
        UserDefaults.standard.bool(forKey: "alwaysShowSamples") {
        didSet {
            UserDefaults.standard.set(alwaysShowSamples,
                                      forKey: "alwaysShowSamples")
        }
    }
    private static func samplesKey(_ name: String) -> String {
        "usedSamples.\(name)"
    }

    private static func usedSamples(for name: String) -> Set<String> {
        Set(UserDefaults.standard.stringArray(forKey: samplesKey(name)) ?? [])
    }

    private var usedSamples: Set<String> =
        ChatModel.usedSamples(for: ChatModel.startModel())

    private var applicableSampleIds: [String] {
        var ids = ["calc"]
        if accessState != .offline { ids.append("research") }
        if canAttachImages, ChatModel.samplePictureThumb != nil {
            ids.append("picture")
        }
        if canOfferVideoSample { ids.append("video") }
        if canOfferDocumentSample { ids.append("document") }
        return ids
    }

    var canOfferDocumentSample: Bool { ChatModel.samplePdf != nil }

    var canOfferVideoSample: Bool {
        installedGB >= 4 && canAttachVideo && ChatModel.sampleVideo != nil
    }

    var showSamples: Bool {
        ready && !busy && messages.isEmpty
            && applicableSampleIds.contains { id in showSample(id) }
    }

    func showSample(_ id: String) -> Bool {
        alwaysShowSamples || !usedSamples.contains(id)
    }

    private func markSampleUsed(_ id: String) {
        usedSamples.insert(id)
        UserDefaults.standard.set(Array(usedSamples),
                                  forKey: Self.samplesKey(modelName))
    }

    func runResearchSample() {
        markSampleUsed("research")
        input = ChatModel.sampleResearch
        caret = input.utf16.count
        send()
    }

    func runEulerSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleEuler
        caret = input.utf16.count
        send()
    }

    func runInterestSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleInterest
        caret = input.utf16.count
        send()
    }

    func runCookiesSample() {
        markSampleUsed("calc")
        input = ChatModel.sampleCookies
        caret = input.utf16.count
        send()
    }

    func runPictureSample() {
        if let data = ChatModel.samplePicture {
            markSampleUsed("picture")
            input = ""
            caret = 0
            attachImage(data, name: "dogs-beach.jpg", at: 0)
            input += ChatModel.sampleStory
            send()
        }
    }

    func runDocumentSample() {
        if let url = ChatModel.samplePdf {
            markSampleUsed("document")
            input = ""
            caret = 0
            flashHUD("Reading the report")
            Task { @MainActor in
                if let text = await ChatModel.markdown(of: url) {
                    attachDoc("harvest-report.pdf", text, at: 0,
                              from: url)
                    input += ChatModel.sampleReport
                    send()
                } else {
                    flashHUD("Cannot read the report")
                }
            }
        }
    }

    func runVideoSample() {
        if let url = ChatModel.sampleVideo {
            markSampleUsed("video")
            input = ""
            caret = 0
            attachClip(url, isVideo: true, at: 0)
            input += ChatModel.sampleClip
            send()
        }
    }

}

final class HeardSpeech: @unchecked Sendable {
    private let lock = NSLock()
    private var said: [SpeechGate.Utterance] = []
    private var count = 0
    private var at = Date()

    var lastAt: Date {
        lock.lock()
        defer { lock.unlock() }
        return at
    }

    var hasSpeech: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !said.isEmpty
    }

    var utteranceCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return said.count
    }

    var seconds: Double {
        lock.lock()
        defer { lock.unlock() }
        return said.reduce(0.0) { sum, u in sum + u.seconds }
    }

    var samples: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func captured(_ n: Int) {
        lock.lock()
        count += n
        lock.unlock()
    }

    private var loudest: Float = 0
    private var square: Double = 0

    func observe(_ block: [Float]) {
        var top: Float = 0
        var sum: Double = 0
        for s in block {
            let a = abs(s)
            if a > top { top = a }
            sum += Double(s) * Double(s)
        }
        lock.lock()
        if top > loudest { loudest = top }
        square += sum
        lock.unlock()
    }

    var peak: Float {
        lock.lock()
        defer { lock.unlock() }
        return loudest
    }

    var rms: Double {
        lock.lock()
        defer { lock.unlock() }
        return count > 0 ? (square / Double(count)).squareRoot() : 0
    }

    func add(_ more: [SpeechGate.Utterance]) {
        if !more.isEmpty {
            lock.lock()
            said.append(contentsOf: more)
            at = Date()
            lock.unlock()
        }
    }

    func clear() {
        lock.lock()
        said = []
        count = 0
        loudest = 0
        square = 0
        at = Date()
        lock.unlock()
    }

    func take() -> [SpeechGate.Utterance] {
        lock.lock()
        defer { lock.unlock() }
        let out = said
        said = []
        return out
    }

}
