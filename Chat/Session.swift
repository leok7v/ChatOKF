import Foundation
import LLM
import Observation

public enum TurnEvent: Sendable {
    case reasoning(String)
    case answer(String)
    case toolStarting
    case toolRound(ToolRoundEvent)
    case looking(VideoPeek)
    case doneLooking
    case soft(URL)
    case stats(TurnMetrics)
    case finished(ChatSession.TurnOutcome, TurnMetrics)
    case cancelled
    case failed(String)
}

@MainActor @Observable public final class Session {

    public struct TurnHooks: Sendable {
        public let onReasoning: @Sendable (String) -> Void
        public let onTool: @Sendable (String) -> Void
        public let onToolRound: @Sendable (ToolRoundEvent) -> Void
        public let onLooking: @Sendable (VideoPeek) -> Void
        public let onDoneLooking: @Sendable () -> Void
        public let onSoft: @Sendable (URL) -> Void
    }

    public var modelName: String
    public var systemPrompt: String
    public var wikipedia = true
    public var webAccess = true
    public let memories = Memories()
    private var recalledIds: Set<String> = []

    var ggufBackend: (any AgentBackend)?
    private var ggufTemplate = ""
    private var ggufVocabCount = 0
    private var activePresets: SamplingPresets?
    var media: (any MediaEncoder)?
    public var audioSampleRate: Double? { media?.audioSampleRate }
    public var maxAudioSeconds: Double? { media?.maxAudioSeconds }
    var session: ChatSession?
    private var metaTask: Task<Void, Never>?
    private var workTask: Task<Void, Never>?
    private var primingTask: Task<Void, Never>?
    private var retiring: Task<Void, Never>?
    private var currentToolRounds: [ToolRoundEvent] = []

    public private(set) var modelShape: ModelShape?
    public var modalities: Modalities { media?.modalities ?? .none }
    public private(set) var modelSupportsThinking = true
    public private(set) var modelSupportsReasoningEffort = false
    public private(set) var effortLevels: [String] = []
    public private(set) var perImageTokens = 256

    public var hasSession: Bool { session != nil }
    public var metaTaskRunning: Bool { metaTask != nil }
    public internal(set) var storage = StorageReport()
    @ObservationIgnored var storageAsked = 0

    private let traceFile: TraceFile? =
        DiagGate.transcript.on ? TraceFile() : nil
    private let instrument = Instrument.install()
    public var tracePath: String { traceFile?.path ?? "" }
    public var diagPath: String { Diag.shared.path }

    public static let overthinkLambda: Float = 1.0

    private var activity: NSObjectProtocol?
    private var working = 0

    private func beginWork() {
        working += 1
        if activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiated, .idleSystemSleepDisabled],
                reason: "answering")
        }
    }

    private func endWork() {
        working -= 1
        if working == 0, let activity {
            ProcessInfo.processInfo.endActivity(activity)
            self.activity = nil
        }
    }

    public init(modelName: String, systemPrompt: String) {
        self.modelName = modelName
        self.systemPrompt = systemPrompt
        memories.open()
    }

    public static let prepFailed = "This model could not be prepared. Try "
        + "Again loads it once more; if it keeps failing, delete the model "
        + "and download it again."

    public static func isOnDisk(_ name: String) -> Bool {
        var result = false
        if let local = ModelCatalog.localSet(name, in: Bundle.modelStore()) {
            result = ModelCatalog.isComplete(local)
        }
        return result
    }

    public static func erase(_ name: String) {
        let fm = FileManager.default
        try? fm.removeItem(
            at: Bundle.modelStore().appendingPathComponent(name))
        let cooked = (try? fm.contentsOfDirectory(
            at: precookDir, includingPropertiesForKeys: nil)) ?? []
        for url in cooked
        where url.lastPathComponent.hasPrefix(name + ".") {
            try? fm.removeItem(at: url)
        }
        if let sha = ModelCatalog.source(name)?.revision {
            let d = UserDefaults.standard
            for k in d.dictionaryRepresentation().keys
            where k.hasPrefix("compiled.\(sha).") {
                d.removeObject(forKey: k)
            }
        }
    }

    public static func pruneUnavailable() {
        let offered = Set(Models.all)
        if !offered.isEmpty {
            let names = (try? FileManager.default.contentsOfDirectory(
                atPath: Bundle.modelStore().path)) ?? []
            for name in names where !offered.contains(name) {
                erase(name)
            }
        }
    }

    public func releaseSession(parking id: UUID? = nil) {
        let outgoing = session
        let meta = metaTask
        let prior = retiring
        let name = modelName
        metaTask = nil
        retiring = Task {
            await prior?.value
            outgoing?.requestStop()
            meta?.cancel()
            await meta?.value
            await outgoing?.endPriming()
            await outgoing?.quiesce()
            if let id, let outgoing {
                await park(outgoing, as: id, model: name)
            }
        }
        ggufBackend = nil
        session = nil
        modelShape = nil
        pendingWarm = nil
        media = nil
        modelSupportsReasoningEffort = false
        effortLevels = []
        forgetRecalled()
    }

    func forgetRecalled() {
        recalledIds = []
        memories.forgetSeen()
    }

    public func requestStop() {
        session?.requestStop()
    }

    public func stop() {
        workTask?.cancel()
        session?.requestStop()
    }

    public func installBackend(_ backend: any AgentBackend, template: String,
                               vocabSize: Int, presets: SamplingPresets) {
        ggufBackend = backend
        ggufTemplate = template
        ggufVocabCount = vocabSize
        activePresets = presets
        turnTables = nil
    }

    public func buildGguf(name: String, path: String) async -> String? {
        memories.modelName = name
        releaseSession()
        await retiring?.value
        retiring = nil
        let loaded = await Session.loadHeavy(name: name, path: path)
        await memories.awaitOpen()
        var failure: String? = Session.prepFailed
        if let built = loaded.built {
            ggufBackend = built.backend
            ggufTemplate = built.template
            ggufVocabCount = built.vocab
            activePresets = built.presets
            modelShape = built.shape
            pendingWarm = built.warm
            turnTables = (built.breakers, built.grammarVocab)
            media = loaded.media
            modelName = name
            failure = nil
        }
        return failure
    }

    private struct HeavyBuild: Sendable {
        let backend: any AgentBackend
        let template: String
        let vocab: Int
        let presets: SamplingPresets
        let shape: ModelShape
        let warm: WeightWarm
        let breakers: Set<Int32>
        let grammarVocab: GrammarVocab?
    }

    private var turnTables: (breakers: Set<Int32>, vocab: GrammarVocab?)?

    nonisolated private static func loadHeavy(name: String, path: String)
        async -> (built: HeavyBuild?, media: (any MediaEncoder)?) {
        Footprint.report(.load, "before \(name)")
        var built: HeavyBuild?
        var loadedMedia: (any MediaEncoder)?
        do {
            if Gemma4Model.isGemma4(path: path) {
                let c = try GemmaChat(ggufPath: path)
                let gpu = try c.metalBackend()
                built = Session.built(
                    gpu, template: c.chatTemplate, vocab: c.vocabCount,
                    presets: c.samplingPresets, shape: c.shape,
                    warm: c.weightWarm(drafting: gpu.draftWidth > 1))
                Session.draftCount = gpu.draftWidth
                if await gpu.supportsSoftTokens() {
                    loadedMedia = c.media(ctx: gpu.ctx)
                }
            } else {
                let c = try QwenMetalChat(ggufPath: path)
                c.engine.loadMTP(drafts: c.mtpDrafts)
                Session.draftCount = c.mtpDrafts
                let backend = c.backend()
                built = Session.built(
                    backend, template: c.chatTemplate,
                    vocab: c.tokenizer.vocabCount,
                    presets: c.samplingPresets, shape: c.shape,
                    warm: c.weightWarm(drafting: c.mtpDrafts > 0))
                loadedMedia = backend.media()
            }
        } catch {
            Diag.shared.report("model prep FAILED \(name): \(error)")
            built = nil
        }
        Footprint.report(.load, "loaded \(name)")
        return (built, loadedMedia)
    }

    nonisolated private static func built(
        _ backend: any AgentBackend, template: String, vocab: Int,
        presets: SamplingPresets, shape: ModelShape,
        warm: WeightWarm) -> HeavyBuild {
        let began = Date()
        let breakers = ChatSession.sequenceBreakers(backend, vocabSize: vocab)
        let grammar = ChatSession.grammarVocab(backend, vocabSize: vocab,
                                               template: template)
        Diag.shared.report(.load, String(
            format: "[chat] %d breakers, grammar %@, over %d tokens in %.0f ms",
            breakers.count, grammar == nil ? "none" : "table", vocab,
            Date().timeIntervalSince(began) * 1000))
        return HeavyBuild(backend: backend, template: template, vocab: vocab,
                          presets: presets, shape: shape, warm: warm,
                          breakers: breakers, grammarVocab: grammar)
    }

    private var pendingWarm: WeightWarm?
    private var warming: Task<Void, Never>?

    private func warmBeside(cooking: Bool) {
        let warm = pendingWarm
        pendingWarm = nil
        if let warm, cooking {
            Diag.shared.report(.load, String(
                format: "not warmed: the cook reads all %.2f GB itself",
                Double(warm.bytes) / 1e9))
        } else if let warm {
            warming = Task.detached { Session.run(warm) }
        }
    }

    nonisolated private static func run(_ warm: WeightWarm) {
        let bytes = warm.bytes
        let room = Int(ProcessInfo.processInfo.physicalMemory / 3 * 2)
        if !Flags.on("warm") {
            Diag.shared.report(.load, "not warmed: --no-warm")
        } else if bytes > 0, bytes < room {
            let t0 = Date()
            let covered = warm.run()
            Diag.shared.report(.load, String(
                format: "warmed %.2f GB in %.1fs", Double(covered) / 1e9,
                Date().timeIntervalSince(t0)))
            Footprint.report(.load, "warmed")
        } else {
            Diag.shared.report(.load, String(
                format: "not warmed: %.2f GB is read by every token, "
                    + "%.2f GB is the most this device warms",
                Double(bytes) / 1e9, Double(room) / 1e9))
        }
    }

    public func fetch(name: String,
                      onProgress: @escaping @Sendable (HubFetch.Status) -> Void
    ) async -> String? {
        var failure: String? = "download failed, check your connection"
        if let src = ModelCatalog.source(name) {
            let dest = Bundle.modelStore().appendingPathComponent(name)
            do {
                _ = try await HubFetch.fetch(
                    repo: src.repo, into: dest, revision: src.revision,
                    files: src.files, excludeFromBackup: true,
                    background: isOS, report: onProgress)
                failure = nil
            } catch HubError.digest {
                failure = "download failed verification, try again"
            } catch {
            }
        }
        return failure
    }

    nonisolated(unsafe) static var draftCount = 0

    nonisolated static func tgKey(_ name: String) -> String { "tg.\(name)" }

    private static func ppKey(_ name: String) -> String { "pp.\(name)" }

    public static func storedTG(_ name: String) -> Double {
        UserDefaults.standard.double(forKey: Session.tgKey(name))
    }

    public var measuredTG: Double {
        let v = UserDefaults.standard.double(forKey: Session.tgKey(modelName))
        return v > 0 ? v : 20
    }

    public var measuredPP: Double {
        let v = UserDefaults.standard.double(forKey: Session.ppKey(modelName))
        return v > 0 ? v : 100
    }

    private func recordRate(_ key: String, _ rate: Double) {
        if rate > 0 {
            let old = UserDefaults.standard.double(forKey: key)
            let ema = old > 0 ? 0.7 * old + 0.3 * rate : rate
            UserDefaults.standard.set(ema, forKey: key)
        }
    }

    private func recordTG(_ tg: Double) {
        recordRate(Session.tgKey(modelName), tg)
    }

    private func recordPP(_ pp: Double) {
        recordRate(Session.ppKey(modelName), pp)
    }

    public func recall(_ question: String, also: [String],
                       within conversation: UUID? = nil) async
        -> Memories.Recall? {
        var out: Memories.Recall? = nil
        if memories.active {
            let own = conversation.map { id in [ConversationNote.id(id)] }
            let found = await memories.recall(
                question, also: also, pp: measuredPP,
                excluding: recalledIds.union(own ?? []))
            if let found {
                Diag.shared.report(.turn, String(
                    format: "[recall] %@ %d note(s), standout %.1f, ~%d tok "
                        + "%.1fs of %.0fs, searched %.2fs: %@",
                    found.silent ? "silent" : "offered", found.ids.count,
                    found.standout, found.tokens, found.seconds,
                    Memories.budgetSeconds, found.searchSeconds,
                    found.ids.joined(separator: " ")))
                if found.silent { recalledIds.formUnion(found.ids) }
                out = found
            } else if Models.judges(modelName), let session {
                out = await judged(question, session,
                                   excluding: recalledIds.union(own ?? []))
            }
            Footprint.report(.memory, "after recall")
        }
        return out
    }

    static let judgeCuts: [Double] = Session.cuts(
        Flags.value("judge-cuts") ?? "")
    static let judgedNotes = 3

    nonisolated static func cuts(_ text: String) -> [Double] {
        let given = text.split(separator: ",")
            .compactMap { part in Double(part) }
        return given.count == 2 ? given : [0.3, 0.6]
    }

    nonisolated static func yes(_ shares: [Double]) -> Double {
        let total = shares.reduce(0, +)
        return total > 0 ? (shares.first ?? 0) / total : 0
    }

    private func judged(_ question: String, _ session: ChatSession,
                        excluding read: Set<String>) async
        -> Memories.Recall? {
        var out: Memories.Recall? = nil
        beginWork()
        defer { endWork() }
        await awaitPrimed()
        await drainMeta()
        let began = Date()
        let about = Session.yes(await session.judge(
            frame: Memories.aboutFrame, items: [question],
            options: ["yes", "no"]).first ?? [])
        Diag.shared.report(.turn, String(
            format: "[judge] about-memory %.2f in %.1fs", about,
            Date().timeIntervalSince(began)))
        if about >= Session.judgeCuts[0] {
            out = memories.listing(pp: measuredPP)
        } else {
            let near = await memories.candidates(
                question, excluding: read, limit: Session.judgedNotes)
            let shares = await session.judge(
                frame: Memories.helpsFrame,
                items: near.map { note in
                    "Note: " + (note.description.isEmpty
                        ? note.title : note.description)
                        + "\nMessage: " + question
                }, options: ["yes", "no"])
            var kept: [Concept] = []
            for (note, share) in zip(near, shares) {
                Diag.shared.report(.turn, String(
                    format: "[judge] note-helps %.2f %@", Session.yes(share),
                    note.id))
                if Session.yes(share) >= Session.judgeCuts[1] {
                    kept.append(note)
                }
            }
            if !kept.isEmpty {
                out = Memories.recall(of: kept, pp: measuredPP)
                out?.judged = true
            }
            Diag.shared.report(.turn, String(
                format: "[judge] %d of %d near notes kept in %.1fs",
                kept.count, near.count, Date().timeIntervalSince(began)))
        }
        if let out { recalledIds.formUnion(out.ids) }
        return out
    }

    public func noteToUse(_ id: String) -> Memories.Note? {
        let note = memories.note(id)
        if note != nil { recalledIds.insert(id) }
        return note
    }

    nonisolated public static func rate(_ v: Double) -> String {
        v > 0 ? String(format: "%.1f", v) : "-"
    }

    public struct SessionConfig: Sendable {
        public var thinking: Bool
        public var reasoningEffortRaw: String
        public var reasoningEffortSlot: Int
        public var thinkTokenCap: Int

        public init(thinking: Bool, reasoningEffortRaw: String,
                    reasoningEffortSlot: Int, thinkTokenCap: Int) {
            self.thinking = thinking
            self.reasoningEffortRaw = reasoningEffortRaw
            self.reasoningEffortSlot = reasoningEffortSlot
            self.thinkTokenCap = thinkTokenCap
        }
    }

    private static let southernRegions: Set<String> = [
        "AU", "NZ", "AR", "CL", "UY", "PY", "BO", "PE", "BR", "ZA", "NA",
        "BW", "ZW", "ZM", "MZ", "MG", "LS", "SZ", "AO", "MW", "PG", "FJ",
        "NC", "WS", "TO", "VU", "SB",
    ]

    private var canAttachAudio: Bool { modalities.audio }
    private var canAttachImages: Bool { modalities.images }

    private var systemStable: String {
        let loc = Locale.current
        let region = loc.region?.identifier ?? "unknown"
        let units = loc.measurementSystem == .metric
            ? "metric (Celsius, kilometers)"
            : "US customary (Fahrenheit, miles)"
        var s = systemPrompt
        s += "\nUser locale: region \(region), timezone "
            + "\(TimeZone.current.identifier), \(units) units. Give "
            + "temperatures and distances in these units unless asked "
            + "otherwise."
        if !webAccess && !wikipedia {
            s += "\nYou are offline: no web search or Wikipedia lookup is "
                + "available. Answer from your own knowledge; do not call any "
                + "search or lookup tool."
        } else if !webAccess {
            s += "\nYou have no web search or page fetching. You may consult "
                + "Wikipedia (wikipedia_query) and today's headlines "
                + "(get_news); for anything else answer from your own "
                + "knowledge."
        } else {
            s += "\nWhen a question names a year, a product, a model or an "
                + "event you may not know, or asks what is current, search "
                + "the web before answering instead of guessing, and say "
                + "when something is past what you know."
        }
        if canAttachAudio {
            s += "\nThe user may speak to you. Their speech reaches you "
                + "already encoded by your own audio tower, so you hear it "
                + "directly: never say you cannot process audio, and never "
                + "treat it as a transcript someone pasted. Spoken words are "
                + "the user talking TO you -- answer them as you would the "
                + "same words typed, and write them out only when asked to."
        }
        if canAttachImages {
            s += "\nThe user may attach images. Each is encoded by your "
                + "vision tower and fully visible to you: describe what you "
                + "actually see, and never claim you cannot view images."
        }
        if memories.active {
            s += "\nThe user keeps their own notes. The ones that fit a "
                + "message arrive with it; memory_search finds the rest. "
                + "What the user asks you to remember is saved after your "
                + "reply, so just confirm it."
        }
        return s
    }

    private static let minilmPath: String? = WikiSlugs.bundledModel?.path

    public var toolRunnerOverride: (any ToolRunner)?

    private var toolRunner: (any ToolRunner)? {
        let safe = SafeToolRunner(
            slugsPath: wikipedia ? Session.minilmPath : nil,
            wikipedia: wikipedia, network: webAccess)
        let runner: any ToolRunner = memories.active
            ? MemoryToolRunner(inner: safe, memories: memories,
                               writes: Models.offersNoteTools)
            : safe
        return toolRunnerOverride ?? runner
    }

    private func recordTrace(_ e: TraceEvent,
                            onEvent: @MainActor (TraceEvent) -> Void) {
        traceFile?.append(e)
        onEvent(e)
    }

    public func hookTrace(thinkingActive: Bool,
                          onEvent: @escaping @MainActor (TraceEvent) -> Void) {
        traceFile?.note("=== \(modelName) thinking=\(thinkingActive) "
            + "wiki=\(wikipedia) web=\(webAccess) \(Date()) app "
            + Instrument.build)
        let s = session
        let sink: @Sendable (TraceEvent) -> Void = { [weak self] e in
            Task { @MainActor in self?.recordTrace(e, onEvent: onEvent) }
        }
        Task { await s?.setTrace(sink) }
        Tools.setDiagSink { [weak self] msg in
            Task { @MainActor in
                self?.recordTrace(TraceEvent(
                    kind: .diag, t0: Date(), t1: Date(), ctx: -1,
                    tokens: 0, summary: String(msg.prefix(80)), text: msg),
                    onEvent: onEvent)
            }
        }
    }

    public func makeSession(_ config: SessionConfig,
                            onEvent: @escaping @MainActor
                                (TraceEvent) -> Void) {
        if let ggufBackend, let activePresets {
            modelSupportsThinking = templateSupportsThinking(ggufTemplate)
            effortLevels = templateEffortLevels(ggufTemplate)
            modelSupportsReasoningEffort = effortLevels.count > 1
            let thinkingActive = config.thinking && modelSupportsThinking
            let wire = effortSpelling(config.reasoningEffortRaw,
                                      slot: config.reasoningEffortSlot,
                                      in: effortLevels)
            session = ChatSession(
                backend: ggufBackend, template: ggufTemplate,
                system: systemStable, systemTail: "",
                vocabSize: ggufVocabCount, presets: activePresets,
                enableThinking: thinkingActive,
                suppressReasoning: !thinkingActive,
                reasoningEffort: modelSupportsReasoningEffort ? wire : nil,
                maxTokens: Flags.int("answer-tokens") ?? .max,
                maxReasoning: config.thinkTokenCap * 2,
                softReasoningCap: config.thinkTokenCap,
                overthink: Session.overthinkLambda,
                seed: Flags.uint64("seed") ?? 0, runner: toolRunner,
                readGuard: Session.memoryGuard,
                breakers: turnTables?.breakers,
                grammarVocab: turnTables?.vocab)
            Diag.shared.report(.load, "[chat] session \(modelName): thinking "
                + (thinkingActive ? "on" : "off, reasoning suppressed")
                + ", asked " + (config.thinking ? "on" : "off")
                + ", model " + (modelSupportsThinking ? "thinks" : "cannot")
                + ", effort " + (modelSupportsReasoningEffort ? wire : "none"))
            hookTrace(thinkingActive: thinkingActive, onEvent: onEvent)
        }
    }

    nonisolated static let readFloor =
        UInt64(Flags.int("read-floor-mb") ?? 300) << 20

    nonisolated static func memoryGuard() -> String? {
        var out: String? = nil
        if let left = Footprint.availableBytes(), left < Session.readFloor {
            out = "\(left >> 20) MB left before the device kills the app, "
                + "floor \(Session.readFloor >> 20) MB"
            Diag.shared.report(.turn, "[read] cut: " + out!)
        }
        return out
    }

    nonisolated private static var precookDir: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent("precook", isDirectory: true)
    }

    private static func precookURL(_ name: String, _ stamp: String) -> URL {
        precookDir.appendingPathComponent(
            "\(name).\(Session.revision8(name)).\(stamp.prefix(16))",
            isDirectory: true)
    }

    // A 27B cooks to hundreds of MB.
    nonisolated private static let precookBudget = 4 << 30

    nonisolated private static func prunePrecook(keeping: URL) {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.contentModificationDateKey]
        let found = (try? fm.contentsOfDirectory(
            at: precookDir, includingPropertiesForKeys: keys)) ?? []
        var newest = found.map { url -> (url: URL, at: Date, size: Int) in
            let v = try? url.resourceValues(forKeys: Set(keys))
            return (url, v?.contentModificationDate ?? .distantPast,
                    ChatSession.allocated(url))
        }
        newest.sort { a, b in a.at > b.at }
        var total = 0
        for file in newest {
            total += file.size
            let stale = !Session.isStampNamed(file.url) || (
                total > precookBudget && file.url != keeping)
            if stale { try? fm.removeItem(at: file.url) }
        }
    }

    nonisolated private static func isStampNamed(_ url: URL) -> Bool {
        let stamp = url.lastPathComponent.split(separator: ".").last ?? ""
        return stamp.count == 16
            && stamp.allSatisfy { c in c.isHexDigit && !c.isUppercase }
    }

    private static func ensurePrecookDir() {
        try? FileManager.default.createDirectory(
            at: precookDir, withIntermediateDirectories: true)
        Platform.protectUntilFirstUnlock(precookDir)
    }

    public static func wipePrecook() {
        try? FileManager.default.removeItem(at: precookDir)
    }

    // Read after every setting has landed, so it cannot omit one the render
    // depends on. Empty means no cookable prefix, so only the reset is owed.
    private func primeOrCookInternal(_ s: ChatSession, reset: Bool) async {
        let stamp = await s.precookStamp
        Diag.shared.report(.load, "[chat] prefix stamp "
            + (stamp.isEmpty ? "none, nothing cookable"
                             : String(stamp.prefix(16))))
        let live = Session.liveDir(modelName)
        try? FileManager.default.removeItem(at: live)
        let attached = (try? await s.attach(live: live)) != nil
        if !attached {
            Diag.shared.report("[park] cannot attach \(live.path); "
                               + "this run keeps no state on disk")
        }
        if stamp.isEmpty || !attached {
            warmBeside(cooking: false)
            if reset { await s.reset() }
        } else {
            Session.ensurePrecookDir()
            let url = Session.precookURL(modelName, stamp)
            warmBeside(cooking: !FileManager.default.fileExists(
                atPath: url.path))
            await s.primeOrCook(at: url, resetFirst: reset)
            await s.awaitPriming()
            Session.recordParkRate(modelName, await s.lastSaved)
            Task.detached { Session.prunePrecook(keeping: url) }
        }
        await judgeIfAsked(s)
    }

    private func judgeIfAsked(_ s: ChatSession) async {
        let file = Flags.value("judge") ?? ""
        if !file.isEmpty, let ggufBackend {
            let url = FileManager.default
                .urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent(file)
            let judge = Judge(backend: ggufBackend, template: ggufTemplate,
                              vocabSize: ggufVocabCount)
            await warming?.value
            do {
                try await judge.run(file: url) { line in
                    Diag.shared.report("[judge] " + line)
                }
            } catch {
                Diag.shared.report("[judge] failed: \(error)")
            }
            await s.reset()
            Diag.shared.report("[judge] done")
        }
    }

    public func primeSession(resetFirst: Bool = false) {
        if let session {
            let prior = primingTask
            primingTask = Task {
                await prior?.value
                await self.primeOrCookInternal(session, reset: resetFirst)
            }
        }
    }

    public func awaitPrimed() async {
        await primingTask?.value
        await warming?.value
    }

    public func awaitMeta() async {
        await metaTask?.value
    }

    public func pushSpeculation(_ on: Bool) async {
        await session?.setSpeculation(on)
    }

    public func drainMeta() async {
        if let running = metaTask {
            metaTask = nil
            session?.requestStop()
            running.cancel()
            await running.value
        }
        await session?.quiesce()
    }

    public func newChatEngine(_ config: SessionConfig,
                              onEvent: @escaping @MainActor (TraceEvent) -> Void
    ) async {
        await drainMeta()
        await session?.endPriming()
        Footprint.report(.load, "newChat outgoing released")
        forgetRecalled()
        await memories.awaitOpen()
        makeSession(config, onEvent: onEvent)
        primeSession(resetFirst: true)
    }

    public func quiesce() async { await session?.quiesce() }

    public func pushThinking(_ on: Bool, resetIfFresh fresh: Bool) async {
        await session?.setThinking(on)
        await session?.setSuppressReasoning(!on)
        if fresh {
            primeSession(resetFirst: true)
            await awaitPrimed()
        }
    }

    public func pushTools() {
        let s = session
        let r = toolRunner
        Task { await s?.setTools(r) }
    }

    public func pushReasoningEffort(_ wire: String) {
        let s = session
        Task { await s?.setReasoningEffort(wire) }
    }

    public func requestQuickAnswer() {
        let s = session
        Task { await s?.requestQuickAnswer() }
    }

    public static func refs(_ docs: [Doc]) -> [DocRef] {
        docs.compactMap { doc in
            doc.url.map { url in
                DocRef(url: url, bytes: doc.content.utf8.count,
                       short: doc.short, total: doc.total)
            }
        }
    }

    private static let toolGlyphs: [String: (label: String, symbol: String)] = [
        "get_current_time": ("Current Time", "clock"),
        "calculator": ("Calculator", "function"),
        "web_search": ("Web Search", "magnifyingglass"),
        "fetch_url": ("URL Fetch", "link"),
        "get_news": ("News", "newspaper"),
        "get_weather": ("Weather", "cloud.sun"),
        "wikipedia_query": ("Wikipedia", "books.vertical"),
    ]

    public static func toolLabel(_ event: ToolRoundEvent) -> String {
        event.resolved.flatMap { name in
            (Session.toolGlyphs[name] ?? MemoryTools.glyphs[name])?.label
        } ?? event.name
    }

    public static func toolSymbol(_ event: ToolRoundEvent) -> String {
        event.resolved.flatMap { name in
            (Session.toolGlyphs[name] ?? MemoryTools.glyphs[name])?.symbol
        } ?? "questionmark.circle"
    }

    public static func toolArgs(_ event: ToolRoundEvent) -> String {
        event.params.map { param in "\(param.name): \"\(param.value)\"" }
            .joined(separator: "  ")
    }

    public static func loopStopped(_ m: TurnMetrics, textEmpty: Bool) -> Bool {
        m.endReason == "loop-breaker" && textEmpty
    }

    private func noteToolRound(_ event: ToolRoundEvent) {
        let at = currentToolRounds.firstIndex { r in r.round == event.round }
        if let at {
            currentToolRounds[at] = event
        } else {
            currentToolRounds.append(event)
        }
    }

    private func toolDigest() -> String {
        var out = "none"
        if !currentToolRounds.isEmpty {
            var counts: [String: Int] = [:]
            for round in currentToolRounds {
                counts[Session.toolLabel(round), default: 0] += 1
            }
            let names = counts.sorted { a, b in a.value > b.value }
                .map { pair in "\(pair.key)x\(pair.value)" }
                .joined(separator: ",")
            let distinct = Set(currentToolRounds.map { r in
                Session.toolLabel(r) + Session.toolArgs(r)
            }).count
            out = "\(names)/\(distinct)distinct"
        }
        return out
    }

    private func specDigest(_ turn: SpecTurn?) -> String {
        var out = ""
        if let turn {
            out = String(
                format: " mtp=%.2ftok/cycle accept=%.0f%% cycles=%d",
                turn.tokensPerCycle, turn.acceptRate * 100, turn.cycles)
        }
        return out
    }

    private func reportTurn(_ m: TurnMetrics,
                           _ outcome: ChatSession.TurnOutcome,
                            _ thinkingActive: Bool, _ cap: Int,
                            _ since: Date) {
        let spec = ggufBackend?.drainSpecTurn()
        MTPTuning.shared.fold(MTPSample(
            model: modelName,
            revision: ModelCatalog.source(modelName)?.revision ?? "",
            drafts: spec == nil ? 0 : Session.draftCount,
            bucket: ThermalBucket.current,
            tokens: m.thinkTokens + m.contentTokens,
            seconds: m.tg > 0
                ? Double(m.thinkTokens + m.contentTokens) / m.tg : 0,
            accepted: spec?.acceptRate ?? 0,
            gpuSeconds: ggufBackend?.drainGPUSeconds() ?? 0,
            wallSeconds: Date().timeIntervalSince(since),
            gpuRate: m.tgGPU))
        Diag.shared.report(.turn, String(
            format: "[turn] %@ thinking=%@ cap=%d outcome=%@ end=%@ tools=%@ "
                + "think=%d content=%d ctx=%d %.1fs%@",
            modelName, thinkingActive ? "on" : "off", cap, outcome.rawValue,
            m.endReason, toolDigest(), m.thinkTokens, m.contentTokens,
            m.ctx, Date().timeIntervalSince(since), specDigest(spec)))
    }

    private func makeHooks(_ cont: AsyncStream<TurnEvent>.Continuation)
        -> TurnHooks {
        TurnHooks(
            onReasoning: { piece in cont.yield(.reasoning(piece)) },
            onTool: { _ in cont.yield(.toolStarting) },
            onToolRound: { (event: ToolRoundEvent) in
                cont.yield(.toolRound(event))
                Task { @MainActor in self.noteToolRound(event) }
            },
            onLooking: { peek in cont.yield(.looking(peek)) },
            onDoneLooking: { cont.yield(.doneLooking) },
            onSoft: { url in cont.yield(.soft(url)) })
    }

    private func statsTicker(_ session: ChatSession,
                             _ cont: AsyncStream<TurnEvent>.Continuation)
        -> Task<Void, Never> {
        Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                if !Task.isCancelled {
                    cont.yield(.stats(await session.lastMetrics))
                }
            }
        }
    }

    private func runTurn(
        thinkTokenCap: Int, thinkingActive: Bool,
        failure: @escaping @Sendable (any Error) -> String,
        _ open: @escaping @MainActor (ChatSession, TurnHooks) async throws
            -> AsyncStream<String>
    ) -> AsyncStream<TurnEvent> {
        AsyncStream { cont in
            let task = Task { @MainActor in
                await self.driveTurn(thinkTokenCap: thinkTokenCap,
                                     thinkingActive: thinkingActive,
                                     failure: failure, open, cont)
            }
            workTask = task
            cont.onTermination = { _ in task.cancel() }
        }
    }

    private func driveTurn(
        thinkTokenCap: Int, thinkingActive: Bool,
        failure: @escaping @Sendable (any Error) -> String,
        _ open: @escaping @MainActor (ChatSession, TurnHooks) async throws
            -> AsyncStream<String>,
        _ cont: AsyncStream<TurnEvent>.Continuation
    ) async {
        if let session {
            beginWork()
            defer { endWork() }
            await awaitPrimed()
            await drainMeta()
            let began = Date()
            _ = ggufBackend?.drainGPUSeconds()
            currentToolRounds = []
            let hooks = makeHooks(cont)
            let ticker = statsTicker(session, cont)
            defer { ticker.cancel() }
            await session.setReasoningCaps(soft: thinkTokenCap,
                                           hard: thinkTokenCap * 2)
            do {
                for await piece in try await open(session, hooks) {
                    cont.yield(.answer(piece))
                }
                if Task.isCancelled { await session.quiesce() }
                let outcome = await session.turnOutcome
                let metrics = await session.lastMetrics
                if outcome != .stopped {
                    recordTG(metrics.tg)
                    recordPP(metrics.pp)
                }
                reportTurn(metrics, outcome, thinkingActive, thinkTokenCap,
                           began)
                cont.yield(.finished(outcome, metrics))
            } catch is CancellationError {
                cont.yield(.cancelled)
            } catch {
                cont.yield(.failed(failure(error)))
            }
        } else {
            cont.yield(.failed("The session ended before this turn "
                + "could start."))
        }
        workTask = nil
        cont.finish()
    }

    private func softTurn(
        _ typed: String, preface: String = "", labelled: Bool,
        thinkTokenCap: Int, thinkingActive: Bool,
        _ encode: @escaping @Sendable (
            @escaping @Sendable (VideoPeek) -> Void
        ) async throws -> (parts: [ContentPart], spans: [SoftSpan],
                           perImage: Int)
    ) -> AsyncStream<TurnEvent> {
        runTurn(thinkTokenCap: thinkTokenCap, thinkingActive: thinkingActive,
               failure: Session.attachmentFailed) { session, hooks in
            await session.awaitPriming()
            let built = try await encode(hooks.onLooking)
            hooks.onDoneLooking()
            try Task.checkCancellation()
            if built.perImage > 0 { self.perImageTokens = built.perImage }
            if Session.resumable, let kept = await Session.keep(SoftTurn(
                stamp: Session.parkStamp(self.modelName), labelled: labelled,
                parts: built.parts, spans: built.spans)) {
                hooks.onSoft(kept)
            }
            let ask = preface + (typed.isEmpty
                ? Session.softDefaultPrompt(built.parts) : typed)
            return session.replySoft(
                ask, parts: built.parts + [.text(ask)], spans: built.spans,
                labelled: labelled,
                onReasoning: hooks.onReasoning,
                onToolRound: hooks.onToolRound)
        }
    }

    public func sendText(prompt: String, display: String, docs: [DocRef],
                         stoppable: String? = nil,
                         thinkTokenCap: Int, thinkingActive: Bool)
        -> (asked: Message, events: AsyncStream<TurnEvent>)? {
        var result: (asked: Message, events: AsyncStream<TurnEvent>)? = nil
        if session != nil {
            var asked = Message(fromUser: true, text: display, prompt: prompt)
            asked.docs = docs
            let events = runTurn(thinkTokenCap: thinkTokenCap,
                                 thinkingActive: thinkingActive,
                                 failure: { _ in "" }) { session, hooks in
                session.reply(prompt, stoppable: stoppable,
                              onReasoning: hooks.onReasoning,
                              onTool: hooks.onTool,
                              onToolRound: hooks.onToolRound)
            }
            result = (asked, events)
        }
        return result
    }

    public func sendSoft(typed: String, display: String,
                         images: [ImageAttachment], clips: [ClipAttachment],
                         docs: [DocRef], budget: Int, labelled: Bool,
                         placeholder: Bool, preface: String = "",
                         thinkTokenCap: Int, thinkingActive: Bool)
        -> (asked: Message, events: AsyncStream<TurnEvent>)? {
        var result: (asked: Message, events: AsyncStream<TurnEvent>)? = nil
        if let media, session != nil {
            let previews = images.compactMap { img in img.preview }
            var asked = Message(
                fromUser: true, text: display, images: previews,
                clips: clips.filter { c in c.isVideo }.map { c in c.url },
                posters: clips.compactMap { c in c.thumbnail },
                prompt: typed.isEmpty ? "" : preface + typed)
            asked.docs = docs
            asked.placeholder = placeholder
            let events = softTurn(typed, preface: preface, labelled: labelled,
                                  thinkTokenCap: thinkTokenCap,
                                  thinkingActive: thinkingActive
            ) { onFrame in
                try await Session.encode(media, images: images, clips: clips,
                                         budget: budget, onFrame: onFrame)
            }
            result = (asked, events)
        }
        return result
    }

    public func sendSpoken(said: [SpeechGate.Utterance], typed: String,
                           images: [ImageAttachment], clips: [ClipAttachment],
                           docs: [DocRef], budget: Int,
                           thinkTokenCap: Int, thinkingActive: Bool)
        -> (asked: Message, events: AsyncStream<TurnEvent>)? {
        var result: (asked: Message, events: AsyncStream<TurnEvent>)? = nil
        if let media, session != nil {
            let secs = said.reduce(0.0) { sum, u in sum + u.seconds }
            let previews = images.compactMap { img in img.preview }
            var asked = Message(
                fromUser: true, text: String(format: "Spoken, %.1fs", secs),
                images: previews,
                clips: clips.filter { c in c.isVideo }.map { c in c.url },
                posters: clips.compactMap { c in c.thumbnail },
                prompt: typed.isEmpty ? Session.spokenPrompt : typed)
            asked.docs = docs
            asked.placeholder = true
            let seen = !images.isEmpty || !clips.isEmpty
            let events = softTurn(typed, labelled: seen,
                                  thinkTokenCap: thinkTokenCap,
                                  thinkingActive: thinkingActive
            ) { onFrame in
                let looked = try await Session.encode(
                    media, images: images, clips: clips, budget: budget,
                    onFrame: onFrame)
                let heard = try await Session.encode(media, speech: said)
                return (looked.parts + heard.parts,
                        looked.spans + heard.spans, looked.perImage)
            }
            result = (asked, events)
        }
        return result
    }

    public struct Extraction: Sendable {
        public let said: String
        public let exchange: String
        public let conversation: UUID?

        public init(said: String, exchange: String, conversation: UUID?) {
            self.said = said
            self.exchange = exchange
            self.conversation = conversation
        }
    }

    public func runMetaTurns(
        titled: Bool, wantsFollowup: Bool, extraction: Extraction?,
        conversation: ConversationNote? = nil,
        onTitle: @escaping @MainActor (String) -> Void,
        onFollowup: @escaping @MainActor (String) -> Void,
        onRemembered: @escaping @MainActor ([Memories.Remembered]) -> Void
    ) {
        if let session, titled || wantsFollowup || extraction != nil
            || conversation != nil {
            let running = metaTask
            beginWork()
            metaTask = Task { @MainActor in
                defer { endWork() }
                await running?.value
                var made = ""
                if titled, !Task.isCancelled {
                    made = await session.makeTitle()
                    if !made.isEmpty, !Task.isCancelled { onTitle(made) }
                }
                if wantsFollowup, !Task.isCancelled {
                    let hint = await session.makeFollowup()
                    if !Task.isCancelled { onFollowup(hint) }
                }
                if let extraction, !Task.isCancelled {
                    let got = await extract(extraction)
                    if !got.isEmpty, !Task.isCancelled { onRemembered(got) }
                }
                if let conversation, !Task.isCancelled {
                    await keep(conversation,
                               title: made.isEmpty ? conversation.title : made)
                }
                self.metaTask = nil
            }
        }
    }

    private func keep(_ note: ConversationNote, title: String) async {
        if let session, memories.active, memories.isOpen,
           !note.asked.isEmpty {
            let began = Date()
            var concluded = ""
            if note.concludes {
                concluded = ConversationNote.concluded(
                    await session.conclude(note.instruction),
                    in: note.exchange, asked: note.asked)
            }
            if !Task.isCancelled {
                await memories.keep(note, title: title, concluded: concluded)
                Diag.shared.report(.turn, String(
                    format: "[conversation] %@ %d asked, concluded %@ in "
                        + "%.1fs", ConversationNote.id(note.conversation),
                    note.asked.count,
                    concluded.isEmpty ? "nothing" : concluded.debugDescription,
                    Date().timeIntervalSince(began)))
            }
        }
    }

    private func extract(_ extraction: Extraction) async
        -> [Memories.Remembered] {
        var out: [Memories.Remembered] = []
        if let session, memories.active, memories.isOpen {
            if !Memories.speaksOfSelf(extraction.said) {
                Diag.shared.report(.turn, "[extract] the user said nothing "
                    + "about themselves, skipped")
            } else {
                out = await extract(extraction, with: session)
            }
        }
        return out
    }

    private func extract(_ extraction: Extraction,
                         with session: ChatSession) async
        -> [Memories.Remembered] {
        var out: [Memories.Remembered] = []
        let coverage = await memories.coverage(extraction.exchange)
        if coverage.covered {
            Diag.shared.report(.turn, String(
                format: "[extract] covered by the store, skipped, "
                    + "searched %.2fs", coverage.seconds))
        } else {
            let began = Date()
            let raw = await session.extractNotes(
                Memories.extractionInstruction(known: coverage.known,
                                               said: extraction.said))
            let drafts = Memories.parseDrafts(
                raw, name: AboutYou.called(AboutYou.shared.name))
            out = await memories.remember(
                drafts, said: extraction.said,
                source: extraction.conversation, excluding: recalledIds)
            Diag.shared.report(.turn, String(
                format: "[extract] %d draft(s) of %d parsed from %d "
                    + "chars in %.1fs, searched %.2fs: %@", out.count,
                drafts.count, raw.count,
                Date().timeIntervalSince(began), coverage.seconds,
                out.map { note in note.id }.joined(separator: " ")))
        }
        return out
    }

    public static let spokenPrompt = "Reply to what I just said."

    nonisolated static var resumable: Bool { Flags.on("resumable") }

    nonisolated static func keep(_ turn: SoftTurn) async -> URL? {
        await Task.detached {
            let url = SoftFile.url(UUID().uuidString + ".soft")
            let t0 = Date()
            var out: URL? = nil
            do {
                try SoftFile.write(turn, to: url)
                out = url
                Diag.shared.report(.turn, String(
                    format: "[soft] kept %d rows, %d KB in %.2fs", turn.rows,
                    Session.treeBytes(url) >> 10,
                    Date().timeIntervalSince(t0)))
            } catch {
                Diag.shared.report("[soft] not kept: \(error)")
            }
            return out
        }.value
    }

    nonisolated static func softDefaultPrompt(_ parts: [ContentPart]) -> String {
        var images = 0, videos = 0, sounds = 0
        for part in parts {
            switch part {
            case .image: images += 1
            case .video: videos += 1
            case .audio: sounds += 1
            case .text: break
            }
        }
        var asks: [String] = []
        if images > 0 {
            asks.append(images == 1 ? "Describe what you see."
                                    : "Describe what you see in each picture.")
        }
        if videos > 0 { asks.append("Describe what you see happening.") }
        if sounds > 0 { asks.append("Write out what you hear.") }
        return asks.joined(separator: " ")
    }

    nonisolated private static func encode(
        _ media: any MediaEncoder, images: [ImageAttachment],
        clips: [ClipAttachment], budget: Int,
        onFrame: (@Sendable (VideoPeek) -> Void)? = nil
    ) async throws -> (parts: [ContentPart], spans: [SoftSpan], perImage: Int) {
        var out = Attached()
        var widest = 0
        for img in images {
            try Task.checkCancellation()
            let t0 = Date()
            let got = try media.image(img.data, budget: budget)
            widest = max(widest, got.spans.map { s in s.rows }.max() ?? 0)
            out.append(got)
            Session.note("image", img.name, got.rows, t0)
        }
        for clip in clips {
            try Task.checkCancellation()
            let t0 = Date()
            let got = try await clip.attached(media, budget: budget,
                                              onFrame: onFrame)
            out.append(got)
            Session.note(clip.isVideo ? "video" : "audio", clip.name,
                         got.rows, t0)
        }
        if !images.isEmpty || !clips.isEmpty {
            Footprint.report(.turn, "encoded \(images.count) img "
                + "\(clips.count) clip")
        }
        return (out.parts, out.spans, widest)
    }

    nonisolated private static func encode(
        _ media: any MediaEncoder, speech: [SpeechGate.Utterance]
    ) async throws -> (parts: [ContentPart], spans: [SoftSpan], perImage: Int) {
        let t0 = Date()
        var parts: [ContentPart] = []
        var spans: [SoftSpan] = []
        for u in speech {
            try Task.checkCancellation()
            for span in try media.audio(u.samples) {
                parts.append(.audio)
                spans.append(span)
            }
        }
        Session.note("speech", "\(speech.count) utterance(s)",
                     spans.reduce(0) { sum, s in sum + s.rows }, t0)
        return (parts, spans, 0)
    }

    nonisolated private static func note(_ kind: String, _ name: String,
                                        _ rows: Int, _ since: Date) {
        Diag.shared.report(.turn, String(
            format: "attach %@ %@ -> %d soft tokens (%.1fs)",
            kind, name, rows, Date().timeIntervalSince(since)))
    }

    nonisolated private static func attachmentFailed(
        _ error: any Error) -> String {
        error is MediaError
            ? "\(error)" : "Could not process the attachment."
    }

}
