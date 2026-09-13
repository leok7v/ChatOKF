import CryptoKit
import Foundation

final class GrammarGate: @unchecked Sendable {
    private let vocab: GrammarVocab
    private let structuralOnly: Bool
    var grammar: Grammar?
    // Armed IS "a live grammar exists", so no parallel flag can drift from it.
    var armed: Bool { grammar != nil }

    init(vocab: GrammarVocab, structuralOnly: Bool) {
        self.vocab = vocab
        self.structuralOnly = structuralOnly
    }

    func disarm() {
        grammar = nil
    }

    func mask(_ logits: inout [Float]) {
        if let g = grammar {
            g.maskLogits(vocab, &logits, structuralOnly: structuralOnly)
        }
    }
}

private final class Collected: @unchecked Sendable {
    private let lock = NSLock()
    private var acc = ""

    func add(_ s: String) -> Int {
        lock.lock()
        acc += s
        let n = acc.count
        lock.unlock()
        return n
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return acc
    }
}

public struct ToolRoundEvent: Sendable {
    public let round: Int
    public let name: String
    public let resolved: String?
    public let params: [ToolArg]
    public let result: String?
}

public actor ChatSession {
    private let backend: any AgentBackend
    private let template: String
    private let presets: SamplingPresets
    private let vocabSize: Int
    private var enableThinking: Bool
    private var suppressReasoning = false
    private var reasoningEffort: String?
    //
    private let maxTokens: Int
    // HARD cap on tokens inside <think>, 0 unbounded: on overflow the loop
    // injects </think>, so a thinking runaway with no EOS cannot hang a turn.
    private var maxReasoning: Int
    // SOFT cap, 0 off: past it the loop ends <think> at the next paragraph
    // break rather than mid-sentence.
    private var softReasoningCap: Int
    private let samplerSeed: UInt64
    private let overthinkTokens: Set<Int32>
    private let overthinkLambda: Float
    private let wireTokens: Set<Int32>
    private var runner: (any ToolRunner)?
    private var toolSpecs: [ToolSpec] { runner?.tools ?? [] }
    enum GrammarMode { case off, full, structural }
    private static let grammarMode: GrammarMode = {
        switch Flags.value("tool-grammar") {
        case "full", "1": return .full
        case "off", "0": return .off
        default: return .structural
        }
    }()
    static let digitsExempt = Flags.on("digit-exempt")
    private var gate: GrammarGate?
    private var grammarVocab: GrammarVocab?
    private let toolDialectXML: Bool
    private let wire: ChatWire
    private var turnSampler: Sampler?
    static let maxToolRounds = 10
    // While a call is open nothing streams and the loop breaker never trips, so
    // a model free-running a huge value is an invisible forever-decode.
    static let maxOpenCallBytes = 8192
    static let toolBudgetNudge =
        "You have reached the tool-call limit for this turn. Do not call any "
        + "more tools; answer the user now using the information you already "
        + "have."
    private var history: [AgentMessage]
    // Append-only mirror of the tokens in the KV up to the turn mark, so park,
    // resume and serialize can round-trip the sequence. Empty is fresh.
    private var committed: [Int32]
    private var visionContext: Bool {
        history.contains { m in m.contentParts != nil }
    }
    private var forceEndThink = false
    private var metaTurn = false
    private var genStartsThink = false
    public enum TurnOutcome: String, Sendable {
        case answered, stopped, answerless
    }

    public private(set) var turnOutcome: TurnOutcome = .answered

    public var turnRolledBack: Bool { turnOutcome != .answered }
    public private(set) var lastMetrics: TurnMetrics
    private let systemStable: String
    private let systemTail: String
    private var traceSink: (@Sendable (TraceEvent) -> Void)?
    private var priming: Task<Void, Never>?

    public init(backend: any AgentBackend, template: String, system: String,
                systemTail: String = "", vocabSize: Int,
                presets: SamplingPresets = .greedy,
                enableThinking: Bool = false,
                reasoningEffort: String? = nil, maxTokens: Int = .max,
                maxReasoning: Int = 0, softReasoningCap: Int = 0,
                overthink: Float = 0, seed: UInt64 = 0,
                runner: (any ToolRunner)? = nil) {
        self.backend = backend
        self.template = template
        let wire = ChatWire.derive(template)
        self.wire = wire
        self.openTag = Array(wire.toolCallOpen.utf8)
        self.closeTag = Array(wire.toolCallClose.utf8)
        self.thinkOpen = Array(wire.reasoningOpen.utf8)
        self.thinkClose = Array(wire.reasoningClose.utf8)
        self.toolDialectXML = template.contains("<function=")
        self.systemStable = system
        self.systemTail = systemTail
        self.runner = runner
        self.presets = presets
        self.vocabSize = max(vocabSize, 1)
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.maxTokens = maxTokens
        self.maxReasoning = maxReasoning
        self.softReasoningCap = softReasoningCap
        self.overthinkLambda = overthink
        self.samplerSeed = seed
        var markers: Set<Int32> = []
        if overthink != 0 {
            for marker in Sampler.overthinkMarkers {
                for variant in [marker, " " + marker] {
                    let ids = backend.encode(variant)
                    if ids.count == 1 { markers.insert(ids[0]) }
                }
            }
        }
        self.overthinkTokens = markers
        var specials: Set<Int32> = []
        for marker in wire.penaltyExemptCandidates {
            let ids = backend.encode(marker)
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        for ch in "0123456789." where ChatSession.digitsExempt {
            let ids = backend.encode(String(ch))
            if ids.count == 1 { specials.insert(ids[0]) }
        }
        self.wireTokens = specials
        self.history = [AgentMessage(role: "system",
                                     content: system + systemTail)]
        self.committed = []
        self.lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0,
                                       contentTokens: 0)
    }

    public func setThinking(_ on: Bool) {
        enableThinking = on
    }

    public func setSuppressReasoning(_ on: Bool) {
        suppressReasoning = on
    }

    public func setReasoningEffort(_ level: String?) {
        reasoningEffort = level
    }

    public func setSpeculation(_ on: Bool) {
        backend.useSpeculation(on)
    }

    public func setTools(_ runner: (any ToolRunner)?) {
        self.runner = runner
    }

    public func setTrace(_ sink: (@Sendable (TraceEvent) -> Void)?) {
        traceSink = sink
    }

    public func supportsSoftTokens() async -> Bool {
        await backend.supportsSoftTokens()
    }

    private func trace(_ kind: TraceEvent.Kind, from t0: Date? = nil,
                       until t1: Date? = nil, ctx: Int = -1, tokens: Int = 0,
                       summary: String, text: String = "",
                       image: Data? = nil) {
        if let sink = traceSink {
            let now = Date()
            sink(TraceEvent(kind: kind, t0: t0 ?? now, t1: t1 ?? now, ctx: ctx,
                            tokens: tokens, summary: summary, text: text,
                            image: image))
        }
    }

    public func requestQuickAnswer() {
        forceEndThink = true
    }

    public func setReasoningCaps(soft: Int, hard: Int) {
        softReasoningCap = soft
        maxReasoning = hard
    }

    public nonisolated func requestStop() {
        backend.requestStop()
    }

    public func reset() async {
        history = Array(history.prefix(1))
        committed = []
        attachmentCounts = [:]
        await backend.reset()
        Diag.memory?("session reset")
        lastMetrics = TurnMetrics(ctx: 0, thinkTokens: 0, contentTokens: 0)
        trace(.reset, ctx: 0, summary: "new chat")
    }

    public struct ChatContext: Sendable {
        let state: any BackendState
        let history: [AgentMessage]
        let committed: [Int32]
        let attachments: [String: Int]
    }

    public func park() async throws -> ChatContext {
        ChatContext(state: try await backend.saveState(),
                    history: history, committed: committed,
                    attachments: attachmentCounts)
    }

    public func resume(_ context: ChatContext) async throws {
        try await backend.loadState(context.state)
        history = context.history
        committed = context.committed
        attachmentCounts = context.attachments
    }

    private struct ContextMeta: Codable {
        let stamp: String
        let committed: [Int32]
        let roles: [String]
        let contents: [String]
        let attachments: [String: Int]
    }

    public func saveContext(to url: URL, stamp: String) async throws {
        let context = try await park()
        let stateData = await backend.serializeState(context.state)
        let meta = ContextMeta(
            stamp: stamp, committed: context.committed,
            roles: context.history.map { $0.role },
            contents: context.history.map { $0.content },
            attachments: context.attachments)
        var out = Data()
        let json = try JSONBytes.reproducible(meta)
        var len = Int64(json.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(stateData)
        try out.write(to: url)
    }

    public func loadContext(from url: URL, stamp: String) async throws -> Bool {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let base = data.startIndex
        var len: Int64 = 0
        withUnsafeMutableBytes(of: &len) { dst in
            _ = data.copyBytes(to: dst, from: base ..< base + 8)
        }
        let jlen = Int(Int64(littleEndian: len))
        let meta = try JSONDecoder().decode(
            ContextMeta.self,
            from: data[base + 8 ..< base + 8 + jlen])
        var loaded = false
        if meta.stamp == stamp {
            let state = try await backend.deserializeState(
                data[(base + 8 + jlen)...])
            try await backend.loadState(state)
            var restored: [AgentMessage] = []
            for i in 0 ..< meta.roles.count {
                restored.append(AgentMessage(role: meta.roles[i],
                                             content: meta.contents[i]))
            }
            history = restored
            committed = meta.committed
            attachmentCounts = meta.attachments
            loaded = true
        }
        return loaded
    }

    private func leadingBlock(thinking: Bool) -> String {
        (try? renderPrompt(
            template: template, messages: [], tools: [],
            addGenerationPrompt: false,
            enableThinking: thinking,
            reasoningEffort: thinking ? reasoningEffort : nil,
            bosToken: backend.bosToken)) ?? ""
    }

    private func systemRenders() -> (full: String, prefix: String) {
        let probe = AgentMessage(role: "user", content: "x")
        let both = (try? renderPrompt(
            template: template, messages: [history[0], probe],
            tools: toolSpecs, addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        // No tools and no reasoning, so these bytes are the user TURN alone; a
        // probe carrying either is not a suffix of `both` on Qwen3.8.
        var probeText = (try? renderPrompt(
            template: template, messages: [probe], tools: [],
            addGenerationPrompt: false, enableThinking: false,
            reasoningEffort: nil,
            bosToken: backend.bosToken)) ?? ""
        let lead = leadingBlock(thinking: false)
        if !lead.isEmpty, probeText.hasPrefix(lead) {
            probeText = String(probeText.dropFirst(lead.count))
        }
        var full = ""
        if !both.isEmpty, !probeText.isEmpty, both.hasSuffix(probeText) {
            full = String(both.dropLast(probeText.count))
        }
        if full.isEmpty {
            Diag.shared.report("no precook prefix under this template")
        }
        var prefix = full
        if !systemTail.isEmpty,
           let at = full.range(of: systemTail, options: .backwards) {
            prefix = String(full[..<at.lowerBound])
        }
        return (full, prefix)
    }

    private static func stamp(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8))
            .map { b in String(format: "%02x", b) }.joined()
    }

    public var precookStamp: String {
        let prefix = systemRenders().prefix
        return prefix.isEmpty ? "" : ChatSession.stamp(prefix)
    }

    public func precook(to url: URL) async throws {
        let r = systemRenders()
        if committed.isEmpty && !r.prefix.isEmpty {
            let t0 = Date()
            let prefixIds = backend.encode(r.prefix)
            let fullIds = backend.encode(r.full)
            await backend.reset()
            _ = try await backend.extend(prefixIds)
            committed = prefixIds
            Diag.memory?("precook before save")
            try await saveContext(to: url, stamp: ChatSession.stamp(r.prefix))
            Diag.memory?("precook after save")
            if fullIds.count >= prefixIds.count,
               Array(fullIds.prefix(prefixIds.count)) == prefixIds {
                let tail = Array(fullIds[prefixIds.count...])
                if !tail.isEmpty { _ = try await backend.extend(tail) }
            } else {
                await backend.reset()
                _ = try await backend.extend(fullIds)
            }
            committed = fullIds
            try await backend.mark()
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: fullIds.count,
                  summary: String(format: "precooked system prefix (%.1fs)",
                                  Date().timeIntervalSince(t0)))
        }
    }

    public func primeOrCook(at url: URL, resetFirst: Bool = false) {
        let running = priming
        priming = Task {
            await running?.value
            if resetFirst { await reset() }
            if await prime(from: url) == false {
                try? await precook(to: url)
            }
        }
    }

    private var engineUsers = 0
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    private func enterEngine() { engineUsers += 1 }

    private func leaveEngine() {
        engineUsers -= 1
        if engineUsers == 0 {
            let waiting = idleWaiters
            idleWaiters = []
            for waiter in waiting { waiter.resume() }
        }
    }

    public func quiesce() async {
        backend.requestStop()
        while engineUsers > 0 {
            await withCheckedContinuation { c in idleWaiters.append(c) }
        }
    }

    public func endPriming() async {
        backend.requestStop()
        priming?.cancel()
        await priming?.value
        priming = nil
    }

    public func awaitPriming() async {
        await priming?.value
    }

    public func prime(from url: URL) async -> Bool {
        let t0 = Date()
        let r = systemRenders()
        var ok = false
        if !r.prefix.isEmpty {
            ok = (try? await loadContext(
                from: url, stamp: ChatSession.stamp(r.prefix))) == true
        }
        if ok {
            let fullIds = backend.encode(r.full)
            let n = committed.count
            // The restored state must hold exactly the committed prefix, which
            // must open today's render; a no-op serialization fails here.
            ok = await backend.position == n && n <= fullIds.count
                && Array(fullIds.prefix(n)) == committed
            if ok, n < fullIds.count {
                ok = (try? await backend.extend(
                    Array(fullIds[n...]))) != nil
            }
            if ok {
                committed = fullIds
                ok = (try? await backend.mark()) != nil
            }
            history = [AgentMessage(role: "system",
                                    content: systemStable + systemTail)]
            if !ok {
                // A torn restore must not leave half a prefix in the KV.
                await backend.reset()
                committed = []
            }
        }
        if ok {
            Diag.memory?("primed \(committed.count) ids from cache")
            trace(.prime, from: t0, ctx: await backend.position,
                  tokens: committed.count,
                  summary: String(format: "primed from cache (%.2fs)",
                                  Date().timeIntervalSince(t0)))
        }
        return ok
    }

    public nonisolated func reply(
        _ user: String,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onTool: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            let task = Task {
                await self.runTurn(user, onReasoning: onReasoning,
                                   onTool: onTool,
                                   onToolRound: onToolRound) { piece in
                    cont.yield(piece)
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    public nonisolated func replySoft(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool = true,
        onReasoning: (@Sendable (String) -> Void)? = nil,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)? = nil
    ) -> AsyncStream<String> {
        AsyncStream { cont in
            let task = Task {
                await self.runSoftTurn(
                    user, parts: parts, spans: spans, labelled: labelled,
                    onReasoning: onReasoning, onToolRound: onToolRound
                ) { piece in cont.yield(piece) }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    static func titleSeed(_ gen: String, _ wire: ChatWire) -> String {
        var out = ""
        if !wire.closesReasoning(gen), wire.opensReasoning(gen) {
            out = wire.reasoningClose
        }
        return out + ChatSession.metaBlock
    }

    static let metaBlock = "```text\n"

    private func oneShot(_ instruction: String, stopAfter: Int,
                         _ label: String) async -> String {
        await priming?.value
        var raw = ""
        if history.count > 1, let save = try? await backend.checkpoint() {
            let began = Date()
            let savedThinking = enableThinking
            let savedRunner = runner
            let savedHistory = history
            let savedCommitted = committed
            let savedMetrics = lastMetrics
            let savedOutcome = turnOutcome
            let savedSink = traceSink
            let savedSuppress = suppressReasoning
            let savedHard = maxReasoning
            let savedSoft = softReasoningCap
            enableThinking = false
            suppressReasoning = true
            maxReasoning = 0
            softReasoningCap = 0
            runner = nil
            traceSink = nil
            metaTurn = true
            let appended = await metaAppend(instruction)
            if let appended {
                raw = await metaDecode(appended, stopAfter: stopAfter)
            } else {
                for await piece in reply(instruction) {
                    raw += piece
                    if raw.count > stopAfter { backend.requestStop() }
                }
            }
            metaTurn = false
            let spent = lastMetrics
            enableThinking = savedThinking
            suppressReasoning = savedSuppress
            maxReasoning = savedHard
            softReasoningCap = savedSoft
            runner = savedRunner
            traceSink = savedSink
            history = savedHistory
            committed = savedCommitted
            lastMetrics = savedMetrics
            turnOutcome = savedOutcome
            try? await backend.rollback(save)
            Diag.shared.report(.turn, String(
                format: "%@ %@ raw=%@ think=%d content=%d in %.1fs", label,
                appended == nil ? "relaid" : "appended",
                raw.debugDescription, spent.thinkTokens, spent.contentTokens,
                Date().timeIntervalSince(began)))
        }
        return raw
    }

    private static let metaUser = "ZqUserZq"
    private static let metaBody = "ZqBodyZq"

    private func metaRender(_ messages: [AgentMessage],
                            generation: Bool) -> String {
        (try? renderPrompt(
            template: template, messages: messages, tools: [],
            addGenerationPrompt: generation, enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
    }

    private func metaAppend(_ instruction: String) async -> [Int32]? {
        var out: [Int32]? = nil
        let end = lastMetrics
        let position = await backend.position
        if position == end.ctx, end.endReason != "tool-call" {
            let probe = AgentMessage(role: "user",
                                     content: ChatSession.metaUser)
            let prior = AgentMessage(role: "assistant",
                                     content: ChatSession.metaBody)
            let ask = AgentMessage(role: "user", content: instruction)
            let asked = metaRender([probe, prior, ask], generation: false)
            let full = metaRender([probe, prior, ask], generation: true)
            if let body = asked.range(of: ChatSession.metaBody),
               full.hasPrefix(asked) {
                var gen = String(full.dropFirst(asked.count))
                gen += ChatSession.titleSeed(gen, wire)
                let laid = backend.encode(String(asked[body.upperBound...]))
                var head: [Int32]? = nil
                if end.overrun == 0 {
                    head = laid
                } else if end.overrun == 1, laid.first == end.stopToken {
                    head = Array(laid.dropFirst())
                }
                genStartsThink = wire.startsInReasoning(genPrompt: gen,
                                                        enabled: true)
                out = head.map { ids in ids + backend.encode(gen) }
            }
        }
        return out
    }

    private func metaDecode(_ ids: [Int32], stopAfter: Int) async -> String {
        enterEngine()
        defer { leaveEngine() }
        await installTurnSampler(vision: visionContext)
        let collected = Collected()
        let brake = backend
        if let seed = try? await backend.extend(ids) {
            _ = await decodeStep(seed: seed, pp: 0, tally: (0, 0),
                                 onReasoning: nil) { piece in
                if collected.add(piece) > stopAfter { brake.requestStop() }
            }
        }
        return collected.text
    }

    public func extractNotes(_ instruction: String) async -> String {
        await oneShot(instruction, stopAfter: 2400, "extract")
    }

    public func makeTitle() async -> String {
        let raw = await oneShot(ChatSession.titleInstruction, stopAfter: 160,
                                "makeTitle")
        return ChatSession.cleanTitle(raw)
    }

    public func makeFollowup() async -> String {
        let raw = await oneShot(ChatSession.followUpInstruction,
                                stopAfter: 280, "makeFollowup")
        let hint = ChatSession.cleanFollowup(raw)
        return alreadyAsked(hint) ? "" : hint
    }

    private func alreadyAsked(_ hint: String) -> Bool {
        let candidate = ChatSession.bagOfWords(hint)
        var repeated = candidate.isEmpty
        for m in history where m.role == "user" && !repeated {
            let prior = ChatSession.bagOfWords(m.content)
            let union = candidate.union(prior).count
            repeated = union > 0
                && Double(candidate.intersection(prior).count)
                    / Double(union) >= ChatSession.echoOverlap
        }
        return repeated
    }

    static let echoOverlap = 0.6

    static func bagOfWords(_ s: String) -> Set<String> {
        Set(s.lowercased()
            .split(whereSeparator: { c in !c.isLetter && !c.isNumber })
            .map(String.init))
    }

    static let titleInstruction: String = {
        let text = Flags.value("title-instruction") ?? ""
        return text.isEmpty ? defaultTitleInstruction : text
    }()

    static let defaultTitleInstruction =
        "Give this conversation a title of two to four words that names "
        + "the subject it is about. Plain words on one line. Reply with "
        + "the title only."

    static func withoutMarkup(_ raw: String) -> String {
        var out = ""
        var rest = Substring(raw)
        while let lt = rest.firstIndex(of: "<") {
            let tail = rest[lt...]
            if let gt = tail.firstIndex(of: ">") {
                out += rest[..<lt]
                rest = tail[tail.index(after: gt)...]
            } else {
                out += rest
                rest = rest[rest.endIndex...]
            }
        }
        return out + rest
    }

    static func cleanTitle(_ raw: String) -> String {
        let line = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .first.map(String.init) ?? ""
        let trimmed = line.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`.*_#"))
        var body = trimmed
        if let r = body.range(of: "title:",
                              options: [.caseInsensitive, .anchored]) {
            body = String(body[r.upperBound...])
                .trimmingCharacters(in: .whitespaces)
        }
        var words: [String] = []
        var chars = 0
        for word in body.split(separator: " ").prefix(6)
        where chars + word.count <= 40 {
            words.append(String(word))
            chars += word.count + 1
        }
        let prose = words.contains { word in
            word.contains(where: { c in c.isLetter })
        }
        let deliberating = ChatSession.deliberation.contains { phrase in
            body.lowercased().contains(phrase)
        }
        let usable = prose && words.count > 1 && !deliberating
        return usable ? words.joined(separator: " ") : ""
    }

    static let deliberation = [
        "the user wants", "the user is asking", "the user would",
        "the conversation is about", "short title for", "thinking process",
    ]

    static let followUpInstruction: String = {
        let text = Flags.value("followup-instruction") ?? ""
        return text.isEmpty ? defaultFollowUpInstruction : text
    }()

    static let defaultFollowUpInstruction =
        "Pick one specific detail from your last answer that the user would "
        + "most want to know more about, and write the question they would "
        + "type to ask about it. One sentence ending in a question mark. "
        + "Never repeat a question already asked. Output only the question."

    static let followupMin = 12
    static let followupMax = 140

    static func cleanFollowup(_ raw: String) -> String {
        let lines = ChatSession.withoutMarkup(raw)
            .split(whereSeparator: \.isNewline)
            .map { line in line.trimmingCharacters(in: .whitespaces) }
            .filter { line in !line.isEmpty }
        let asked = lines.filter { line in line.contains("?") }
        var out = ""
        if asked.count == 1, let line = asked.first {
            out = ChatSession.oneQuestion(line)
        }
        return out
    }

    private static let assistantVoice = [
        "would you like", "do you want", "shall i", "can i help",
        "is there anything", "let me know", "should i", "like me to",
        "your last answer",
    ]

    private static func oneQuestion(_ line: String) -> String {
        var body = line
        if let mark = body.firstIndex(of: "?") {
            body = String(body[...mark])
        }
        if let colon = body.range(of: ": "),
           !body[..<colon.lowerBound].contains("?"),
           body.distance(from: body.startIndex, to: colon.lowerBound) <= 40 {
            body = String(body[colon.upperBound...])
        }
        body = body.trimmingCharacters(
            in: CharacterSet(charactersIn: " \t\"'`*_-#>0123456789."))
        let latex = body.contains(where: { c in "{}\\".contains(c) })
        for tex in ["{", "}", "\\"] + (latex ? ["$"] : []) {
            body = body.replacingOccurrences(of: tex, with: "")
        }
        let lower = body.lowercased()
        let markup = body.contains(where: { c in "{}<>".contains(c) })
            || body.contains("**")
        let aboutTheUser = lower.hasPrefix("the user")
            || lower.hasPrefix("user ")
        let assistant = ChatSession.assistantVoice.contains { phrase in
            lower.contains(phrase)
        }
        let deliberating = ChatSession.deliberation.contains { phrase in
            lower.contains(phrase)
        }
        let usable = !aboutTheUser && !assistant && !markup && !deliberating
            && body.hasSuffix("?")
            && body.count >= ChatSession.followupMin
            && body.count <= ChatSession.followupMax
        return usable ? body : ""
    }

    private func runTurn(_ user: String,
                         onReasoning: (@Sendable (String) -> Void)?,
                         onTool: (@Sendable (String) -> Void)?,
                         onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
                         _ yield: @Sendable (String) -> Void) async {
        await priming?.value
        enterEngine()
        defer { leaveEngine() }
        let saved = await enterTurn()
        trace(.user, ctx: await backend.position,
              summary: String(user.prefix(80)), text: user)
        history.append(AgentMessage(role: "user", content: user))
        await installTurnSampler(vision: visionContext)
        await runSeed(soft: [], saved: saved,
                      onReasoning: onReasoning, onTool: onTool,
                      onToolRound: onToolRound, yield)
    }

    private func runSoftTurn(
        _ user: String, parts: [ContentPart], spans: [SoftSpan],
        labelled: Bool,
        onReasoning: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        await priming?.value
        enterEngine()
        defer { leaveEngine() }
        let saved = await enterTurn()
        let rows = spans.reduce(0) { sum, span in sum + span.rows }
        trace(.user, ctx: await backend.position,
              summary: String(user.prefix(80))
                  + " [\(spans.count) attachment(s), \(rows) soft]",
              text: user)
        history.append(AgentMessage(role: "user", content: user,
                                    contentParts: numbered(parts,
                                                            labelled)))
        await installTurnSampler(vision: true)
        await runSeed(soft: spans,
                      saved: saved, onReasoning: onReasoning, onTool: nil,
                      onToolRound: onToolRound, yield)
    }

    private func runSeed(
        soft: [SoftSpan], saved: SavedTurn,
        onReasoning: (@Sendable (String) -> Void)?,
        onTool: (@Sendable (String) -> Void)?,
        onToolRound: (@Sendable (ToolRoundEvent) -> Void)?,
        _ yield: @Sendable (String) -> Void) async {
        let first = await seedOnce(fresh: committed.isEmpty, soft: soft)
        if first.stopped {
            await rollbackTurn(saved)
        } else {
            var pp = first.pp
            var tally = (think: 0, content: 0)
            var round = 0
            var pagedURL: String? = nil
            var paged = 0
            var spent: [String: String] = [:]
            var step = await decodeStep(seed: first.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
            tally = (lastMetrics.thinkTokens, lastMetrics.contentTokens)
            while let call = step.pending, round < ChatSession.maxToolRounds {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                let url = resolved == "fetch_url"
                    ? call.params.first { p in
                        ["url", "link", "u"].contains(p.name)
                    }?.value
                    : nil
                if url != nil && url == pagedURL && paged < 8 {
                    paged += 1
                } else {
                    round += 1
                    paged = 0
                }
                pagedURL = url
                onTool?(call.functionName)
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: nil))
                trace(.toolCall, summary: call.functionName
                          + (resolved == call.functionName || resolved == nil
                             ? "" : " -> \(resolved!)"),
                      text: wire.toolCallOpen + call.rawBlock
                          + wire.toolCallClose)
                toolLog("tool call \(round): \(call.functionName) "
                    + call.params.map { "\($0.name)=\($0.value)" }
                        .joined(separator: " ")
                    + " | raw="
                    + call.rawBlock.replacingOccurrences(of: "\n", with: "\\n"))
                let toolT0 = Date()
                let signature = ChatSession.callSignature(
                    resolved ?? call.functionName,
                    sanitizedArgs(call, resolved))
                var result = spent[signature].map { prior in
                    ChatSession.repeatedCall(prior)
                } ?? ""
                if result.isEmpty {
                    result = await runTool(call, resolved: resolved)
                    spent[signature] = result
                } else {
                    toolLog("tool call \(round): REFUSED as an exact repeat")
                }
                onToolRound?(ToolRoundEvent(
                    round: round, name: call.functionName,
                    resolved: resolved, params: call.params, result: result))
                trace(.toolResult, from: toolT0,
                      summary: "\(resolved ?? call.functionName) -> "
                          + "\(result.count) chars",
                      text: result)
                toolLog("tool result \(round): \(result.count) chars: "
                    + result.prefix(160))
                let cont = await toolContinuation(
                    call: call, resolved: resolved,
                    preamble: step.preamble, result: result)
                if cont.pp > 0 { pp = cont.pp }
                step = await decodeStep(seed: cont.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
                tally = (lastMetrics.thinkTokens, lastMetrics.contentTokens)
            }
            if let call = step.pending {
                let resolved = ChatSession.resolveTool(
                    call.functionName, toolSpecs.map { spec in spec.name })
                onToolRound?(ToolRoundEvent(
                    round: round + 1, name: call.functionName,
                    resolved: resolved, params: call.params,
                    result: ChatSession.toolBudgetNudge))
                trace(.toolResult, summary: "tool budget spent -> nudge",
                      text: ChatSession.toolBudgetNudge)
                let cont = await toolContinuation(
                    call: call, resolved: resolved, preamble: step.preamble,
                    result: ChatSession.toolBudgetNudge)
                if cont.pp > 0 { pp = cont.pp }
                step = await decodeStep(seed: cont.seed, pp: pp, tally: tally,
                                        onReasoning: onReasoning, yield)
                if step.pending != nil {
                    history.append(AgentMessage(role: "assistant",
                                                content: step.preamble))
                }
            }
            let empty = history.last.map { last in
                last.role == "assistant" && last.content.isEmpty
            } ?? false
            if empty {
                await rollbackTurn(saved, why: ChatSession.userStopped(
                    lastMetrics.endReason) ? .stopped : .answerless)
            }
        }
    }

    private func seedOnce(
        fresh: Bool, soft: [SoftSpan]
    ) async -> (seed: Int32, pp: Double, stopped: Bool) {
        let t0 = Date()
        var seed = backend.eos
        var added = 0
        var stopped = false
        do {
            let r = try await seedDelta(fresh: fresh, soft: soft)
            seed = r.seed
            added = r.added
        } catch EngineError.stopped {
            stopped = true
        } catch {
            seed = backend.eos
        }
        let sec = Date().timeIntervalSince(t0)
        let pp = sec > 0 && added > 0 ? Double(added) / sec : 0
        trace(.prefill, from: t0,
              ctx: await backend.position, tokens: added,
              summary: stopped ? "stopped mid-prefill"
                               : String(format: "%.0f t/s", pp))
        return (seed, pp, stopped)
    }

    private func seedDelta(
        fresh: Bool, soft: [SoftSpan]
    ) async throws -> (seed: Int32, added: Int) {
        // [system, user] over a primed prefix is the first turn: the system
        // block is already in the KV, so the delta is the user turn alone.
        let deltaMsgs = fresh
            ? history
            : (history.count == 2
                ? [history[history.count - 1]]
                : [history[history.count - 2], history[history.count - 1]])
        // Tool specs render into the system message, so only the fresh turn
        // carries them; a continuation delta has no system message.
        let tools = fresh ? toolSpecs : []
        var closedText = (try? renderPrompt(
            template: template, messages: deltaMsgs, tools: tools,
            addGenerationPrompt: false, enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        var fullText = (try? renderPrompt(
            template: template, messages: deltaMsgs, tools: tools,
            addGenerationPrompt: true, enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        // Gemma-4 opens a system turn on `enable_thinking` alone, so every
        // delta would re-lay an empty system turn the fresh turn already laid.
        let lead = fresh ? "" : leadingBlock(thinking: enableThinking)
        if !lead.isEmpty, closedText.hasPrefix(lead), fullText.hasPrefix(lead) {
            closedText = String(closedText.dropFirst(lead.count))
            fullText = String(fullText.dropFirst(lead.count))
        }
        if fullText.isEmpty {
            Diag.shared.report("template rendered nothing for these variables")
        }
        if !fullText.hasPrefix(closedText) {
            Diag.shared.report("generation prompt is not a render suffix")
        }
        var genText = fullText.hasPrefix(closedText)
            ? String(fullText.dropFirst(closedText.count)) : ""
        if metaTurn { genText += ChatSession.titleSeed(genText, wire) }
        genStartsThink = wire.startsInReasoning(genPrompt: genText,
                                                enabled: true)
        let encoded = backend.encode(closedText)
        let closed = soft.isEmpty ? encoded
            : Continuation.expandSpans(encoded, soft)
        let gen = backend.encode(genText)
        if fresh {
            let stale = await backend.position
            if stale > 0 {
                let msgs = history.count
                Diag.shared.report("RESET reprefill (history=\(msgs) msgs)")
                trace(.reset, ctx: 0,
                      summary: "fresh turn over stale state (\(stale) tok)")
            }
            await backend.reset()
            committed = []
        } else {
            try await backend.rewind()
            trace(.rewind, ctx: await backend.position,
                  summary: "rewind to turn mark")
        }
        trace(.render, tokens: closed.count + gen.count,
              summary: fresh ? "fresh (system + tools + user)"
                             : "delta (prev answer + user)",
              text: fullText)
        let afterHead: Int32
        if !soft.isEmpty {
            afterHead = try await backend.extendSoft(closed, spans: soft)
        } else {
            afterHead = try await extendChunked(closed)
        }
        committed += closed
        try await backend.mark()
        let seed = gen.isEmpty ? afterHead : try await backend.extend(gen)
        return (seed, closed.count + gen.count)
    }

    static let prefillChunk = 1024

    private func extendChunked(_ ids: [Int32]) async throws -> Int32 {
        var next = backend.eos
        if ids.count <= ChatSession.prefillChunk {
            next = try await backend.extend(ids)
        } else {
            let t0 = Date()
            let startCtx = await backend.position
            var done = 0
            while done < ids.count {
                if done > 0 && backend.shouldStop() {
                    throw EngineError.stopped
                }
                let end = min(ids.count, done + ChatSession.prefillChunk)
                next = try await backend.extend(Array(ids[done..<end]))
                done = end
                let sec = Date().timeIntervalSince(t0)
                lastMetrics = TurnMetrics(
                    ctx: startCtx + done, thinkTokens: 0, contentTokens: 0,
                    pp: sec > 0 ? Double(done) / sec : 0,
                    prefillDone: done, prefillTotal: ids.count)
            }
        }
        return next
    }

    private var attachmentCounts: [String: Int] = [:]

    private static func noun(_ part: ContentPart) -> String? {
        switch part {
        case .image: "Picture"
        case .audio: "Audio"
        case .video: "Video"
        case .text: nil
        }
    }

    private func numbered(_ parts: [ContentPart],
                          _ labelled: Bool = true) -> [ContentPart] {
        var out: [ContentPart] = []
        for part in parts {
            if let noun = ChatSession.noun(part), labelled {
                let n = (attachmentCounts[noun] ?? 0) + 1
                attachmentCounts[noun] = n
                out.append(.text("\(noun) \(n): "))
            }
            out.append(part)
        }
        return out
    }

    private struct SavedTurn {
        let checkpoint: (any BackendState)?
        let history: [AgentMessage]
        let committed: [Int32]
    }

    static func userStopped(_ endReason: String) -> Bool {
        endReason == "cancelled" || endReason == "stop"
    }

    private func enterTurn() async -> SavedTurn {
        turnOutcome = .answered
        runner?.beginTurn()
        let checkpoint = try? await backend.checkpoint()
        return SavedTurn(checkpoint: checkpoint, history: history,
                         committed: committed)
    }

    private func rollbackTurn(_ saved: SavedTurn,
                              why: TurnOutcome = .stopped) async {
        if let checkpoint = saved.checkpoint {
            try? await backend.rollback(checkpoint)
        }
        history = saved.history
        committed = saved.committed
        turnOutcome = why
        trace(.rewind, ctx: await backend.position,
              summary: "turn rollback (\(why.rawValue))")
    }

    private static let greedyConfig: SamplerConfig = {
        var c = SamplerConfig.default
        c.temperature = 0
        return c
    }()

    private func installTurnSampler(vision: Bool) async {
        let reasons = enableThinking && !suppressReasoning
        var turnConfig = metaTurn
            ? ChatSession.greedyConfig
            : presets.select(thinking: reasons, vision: vision)
        if !turnConfig.setMask.contains(.dryMultiplier) {
            turnConfig.dryMultiplier = 0.8
        }
        if samplerSeed != 0 { turnConfig.seed = samplerSeed }
        var sampler = Sampler(vocabSize: vocabSize, config: turnConfig)
        sampler.penaltyExempt = wireTokens
        if overthinkLambda != 0 && reasons && !overthinkTokens.isEmpty {
            sampler.overthinkTokens = overthinkTokens
            sampler.overthinkLambda = overthinkLambda
        }
        turnSampler = sampler
        if let gate = ensureGate() { gate.disarm() }
        await backend.useSampler(sampler)
    }

    private func installSampler(masked: Bool, verbatim: Bool) async {
        if var sampler = turnSampler {
            if masked, let gate {
                sampler.logitMask = { logits in gate.mask(&logits) }
            }
            sampler.verbatim = verbatim
            await backend.useSampler(sampler)
        }
    }

    private func ensureGate() -> GrammarGate? {
        var result: GrammarGate? = nil
        if ChatSession.grammarMode != .off && toolDialectXML {
            if grammarVocab == nil {
                var toks: [[UInt8]] = []
                toks.reserveCapacity(vocabSize)
                for id in 0 ..< vocabSize {
                    toks.append(backend.tokenBytes(Int32(id)))
                }
                grammarVocab = GrammarVocab(toks)
            }
            if let vocab = grammarVocab {
                let g = gate ?? GrammarGate(
                    vocab: vocab,
                    structuralOnly: ChatSession.grammarMode == .structural)
                gate = g
                result = g
            }
        }
        return result
    }

    private func advanceGrammar(_ gate: GrammarGate, _ token: Int32,
                                _ bytes: [UInt8]) {
        if gate.armed {
            gate.grammar?.advance(backend.tokenBytes(token))
            if gate.grammar?.dead ?? true { gate.grammar = nil }
        } else if let off = stillOpen(bytes) {
            let g = Grammar.toolCall()
            g.advance(Array(bytes[off...]))
            gate.grammar = g
        }
    }

    private let openTag: [UInt8]
    private let closeTag: [UInt8]
    private let thinkOpen: [UInt8]
    private let thinkClose: [UInt8]
    // No trailing '>': a spaced "</function >" is still a complete body. A
    // repair for the Qwen XML dialect specifically, not part of any wire.
    private static let closeFn = Array("</function".utf8)

    enum LeadingThink { case pending, isThink(Int), notThink }

    static func leadingThink(_ b: [UInt8], _ open: [UInt8]) -> LeadingThink {
        var i = 0
        while i < b.count && ChatSession.isBlank(b[i]) { i += 1 }
        var j = 0
        while j < open.count && i < b.count
              && (b[i] == open[j]
                  || (j > 0 && open[j - 1] == 0x3E && ChatSession.isBlank(b[i]))) {
            if b[i] == open[j] { j += 1 }
            i += 1
        }
        let result: LeadingThink
        if j == open.count {
            result = .isThink(i)
        } else if i == b.count {
            result = .pending
        } else {
            result = .notThink
        }
        return result
    }

    private static func isBlank(_ c: UInt8) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D
    }

    private static func partialSuffix(_ b: [UInt8], _ pat: [UInt8],
                                      _ n: Int) -> Int {
        var hold = 0
        var len = min(pat.count - 1, n)
        while hold == 0 && len > 0 {
            var j = 0
            while j < len && b[n - len + j] == pat[j] { j += 1 }
            if j == len { hold = len }
            len -= 1
        }
        return hold
    }
    private func stillOpen(_ b: [UInt8]) -> Int? {
        var result: Int? = nil
        if let open = ChatSession.lastIndex(b, openTag),
           ChatSession.index(b, closeTag, open + openTag.count) == nil {
            result = open
        }
        return result
    }

    private static func index(_ b: [UInt8], _ pat: [UInt8],
                              _ from: Int) -> Int? {
        var result: Int? = nil
        var i = max(from, 0)
        let last = b.count - pat.count
        while result == nil && i <= last {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i += 1
        }
        return result
    }

    private static func lastIndex(_ b: [UInt8], _ pat: [UInt8]) -> Int? {
        var result: Int? = nil
        var i = b.count - pat.count
        while result == nil && i >= 0 {
            var j = 0
            while j < pat.count && b[i + j] == pat[j] { j += 1 }
            if j == pat.count { result = i }
            i -= 1
        }
        return result
    }

    private func decodeStep(
        seed: Int32, pp: Double, tally: (think: Int, content: Int),
        onReasoning: (@Sendable (String) -> Void)?,
        _ yield: @Sendable (String) -> Void)
        async -> (pending: ToolCall?, preamble: String) {
        forceEndThink = false  // a prior turn's Quick Answer must not fire
        let toolsActive = runner != nil && !toolSpecs.isEmpty
        // A completed call parks the grammar in its absorbing match state;
        // armed into the next step it would disable spec decode all turn.
        if let gate, gate.armed {
            gate.disarm()
            await installSampler(masked: false, verbatim: false)
        }
        var pending: ToolCall? = nil
        let startCtx = await backend.position
        var cur = seed
        var ids: [Int32] = []
        var bytes: [UInt8] = []
        var emitted = 0
        var closeAt: Int? = nil
        let startsInThink = enableThinking && genStartsThink
        var inThinkRegion = startsInThink
        var thinkDecided = startsInThink
        var wsDone = !startsInThink
        var thinkSearch = 0
        var toolAt: Int? = nil
        // Outlives toolAt being cleared by the quoted-pair recovery below.
        var sawToolOpen = false
        var toolSearch = 0
        var toolCloseSearch = 0
        var toolReopenSearch = 0
        var think = 0
        var content = 0
        var undecided = 0
        var steps = 0
        var fed = startCtx
        var stop = false
        var thinkRescues = 0
        let g0 = Date()
        lastMetrics = TurnMetrics(ctx: startCtx, thinkTokens: tally.think,
                                  contentTokens: tally.content, pp: pp, tg: 0)
        while !backend.eosIds.contains(cur) && !stop && !Task.isCancelled
              && !backend.shouldStop() {
            ids.append(cur)
            bytes.append(contentsOf: backend.tokenBytes(cur))
            let wasArmed = gate?.armed ?? false
            let wasOpen = toolAt != nil
            if toolsActive, let gate { advanceGrammar(gate, cur, bytes) }
            if !thinkDecided {
                switch ChatSession.leadingThink(bytes, thinkOpen) {
                case .isThink(let past):
                    inThinkRegion = true
                    think += undecided
                    undecided = 0
                    emitted = past
                    wsDone = false          // skip the ws after its </think>
                    thinkDecided = true
                case .notThink:
                    thinkDecided = true
                    content += undecided
                    undecided = 0
                case .pending:
                    break
                }
            }
            if inThinkRegion && closeAt == nil {
                closeAt = ChatSession.index(bytes, thinkClose,
                                            thinkSearch)
                if closeAt == nil {
                    thinkSearch = max(thinkSearch,
                        bytes.count - thinkClose.count + 1)
                }
            }
            let inThink = inThinkRegion && closeAt == nil
            if !thinkDecided {
                undecided += 1
            } else if inThink {
                think += 1
            } else {
                content += 1
            }
            let n = Tokenizer.completeUTF8Count(bytes)
            if toolsActive && toolAt == nil {
                toolAt = ChatSession.index(
                    bytes, openTag, toolSearch)
                sawToolOpen = sawToolOpen || toolAt != nil
                if toolAt == nil {
                    toolSearch = max(toolSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let openHold = toolsActive && toolAt == nil
                ? ChatSession.partialSuffix(bytes, openTag, n)
                : 0
            if inThink {
                let hold = max(openHold, ChatSession.partialSuffix(
                    bytes, thinkClose, n))
                let end = min(toolAt ?? Int.max, n - hold)
                if end > emitted {
                    onReasoning?(String(decoding: bytes[emitted ..< end],
                                        as: UTF8.self))
                    emitted = end
                }
            } else if thinkDecided {
                if let at = closeAt, emitted <= at {
                    let flush = min(at, toolAt ?? Int.max)
                    if flush > emitted {
                        onReasoning?(String(decoding: bytes[emitted ..< flush],
                                            as: UTF8.self))
                    }
                    emitted = at + thinkClose.count
                }
                while !wsDone && emitted < n {
                    let b = bytes[emitted]
                    if b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D {
                        emitted += 1
                    } else {
                        wsDone = true
                    }
                }
                if wsDone {
                    var hold = openHold
                    var markAt = Int.max
                    var markLen = 0
                    // A close with no opener ANYWHERE this round closes
                    // nothing; a quoted pair keeps its close, one preceded it.
                    var marks = [thinkClose, thinkOpen]
                    if toolsActive {
                        if !sawToolOpen { marks.append(closeTag) }
                    } else {
                        marks.append(openTag)
                        marks.append(closeTag)
                    }
                    for pat in marks where !pat.isEmpty {
                        hold = max(hold,
                                   ChatSession.partialSuffix(bytes, pat, n))
                        if let at = ChatSession.index(bytes, pat, emitted),
                           at < markAt {
                            markAt = at
                            markLen = pat.count
                        }
                    }
                    let end = min(toolAt ?? Int.max, n - hold)
                    let upto = min(end, markAt)
                    if upto > emitted {
                        yield(String(decoding: bytes[emitted ..< upto],
                                     as: UTF8.self))
                        emitted = upto
                    }
                    if markAt == emitted, markAt + markLen <= n {
                        if markLen == closeTag.count, !sawToolOpen {
                            Diag.shared.report("suppressed a stray close tag")
                        }
                        emitted = markAt + markLen
                    }
                }
            }
            steps += 1
            if steps % 8 == 0 {
                let tg = Double(steps)
                    / max(Date().timeIntervalSince(g0), 1e-6)
                lastMetrics = TurnMetrics(
                    ctx: startCtx + steps, thinkTokens: tally.think + think,
                    contentTokens: tally.content + content, pp: pp, tg: tg)
            }
            let softOver = softReasoningCap > 0 && think >= softReasoningCap &&
                bytes.last == 0x0A
            let looping = Continuation.isLooping(ids,
                tokenBytes: { [backend] id in backend.tokenBytes(id) })
            let loopRescue = looping && inThinkRegion && closeAt == nil
                && toolAt == nil && thinkRescues == 0
            // toolAt non-nil suppresses injection: a call is mid-flight and
            // </think> must not land inside its markup.
            let overflow = inThink && toolAt == nil && (forceEndThink ||
                suppressReasoning || softOver || loopRescue
                || (maxReasoning > 0 && think >= maxReasoning))
            if toolsActive && pending == nil, let open = toolAt {
                if toolCloseSearch < open + openTag.count {
                    toolCloseSearch = open + openTag.count
                }
                if toolReopenSearch < open + openTag.count {
                    toolReopenSearch = open + openTag.count
                }
                let at = ChatSession.index(bytes, closeTag,
                                           toolCloseSearch)
                let reopen = ChatSession.index(bytes, openTag,
                                               toolReopenSearch)
                if let reopen, at == nil || reopen < at! {
                    pending = Tools.parse(Substring(String(
                        decoding: bytes[
                            (open + openTag.count) ..< reopen],
                        as: UTF8.self)))
                    if pending == nil {
                        toolAt = reopen
                        toolSearch = reopen
                        toolCloseSearch = reopen + openTag.count
                        toolReopenSearch = reopen + openTag.count
                    }
                } else if let at {
                    let end = at + closeTag.count
                    pending = completedCall(String(
                        decoding: bytes[open ..< end], as: UTF8.self))
                    if pending == nil {
                        toolCloseSearch = end
                        toolSearch = end
                        toolReopenSearch = end
                        toolAt = nil
                    }
                } else {
                    toolCloseSearch = max(toolCloseSearch,
                        bytes.count - closeTag.count + 1)
                    toolReopenSearch = max(toolReopenSearch,
                        bytes.count - openTag.count + 1)
                }
            }
            let armed = gate?.armed ?? false
            if armed != wasArmed || (toolAt != nil) != wasOpen {
                await installSampler(masked: armed, verbatim: toolAt != nil)
            }
            let openRunaway = toolAt.map { at in
                bytes.count - at > ChatSession.maxOpenCallBytes
            } == true
            stop = steps >= maxTokens
                || (looping && !loopRescue)
                || pending != nil
                || openRunaway
            if overflow && !stop {
                if await backend.queuedCount() > 0 {
                    // Spec-committed tokens the transcript has not seen yet:
                    // </think> injected now would land after them; drain first.
                    cur = (try? await backend.decode(cur)) ?? backend.eos
                    fed += 1
                } else {
                    forceEndThink = false
                    if loopRescue { thinkRescues = 1 }
                    // The injected close marker is found next pass, and the
                    // whitespace skip eats its newlines, so none of it streams.
                    let close = backend.encode(
                        wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (think \(think))")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                    fed += close.count
                }
            } else if !stop {
                cur = (try? await backend.decode(cur)) ?? backend.eos
                fed += 1
                if backend.eosIds.contains(cur), inThinkRegion,
                   closeAt == nil,
                   toolAt == nil, thinkRescues == 0,
                   await backend.queuedCount() == 0 {
                    thinkRescues = 1
                    let close = backend.encode(
                        wire.reasoningClose + "\n\n")
                    ids.append(contentsOf: close)
                    for id in close {
                        bytes.append(contentsOf: backend.tokenBytes(id))
                    }
                    trace(.inject,
                          summary: "</think> injected (eos inside think)")
                    cur = (try? await backend.extend(close)) ?? backend.eos
                    fed += close.count
                }
            }
        }
        if pending == nil, backend.eosIds.contains(cur), !Task.isCancelled,
           !backend.shouldStop(), let open = toolAt,
           ChatSession.index(bytes, ChatSession.closeFn, open) != nil {
            pending = Tools.parse(Substring(String(
                decoding: bytes[(open + openTag.count)...],
                as: UTF8.self)))
            if pending != nil {
                toolLog("eos-cut call rescued (body complete)")
            }
        }
        let whole = Tokenizer.completeUTF8Count(bytes)
        let readable = min(toolAt ?? whole, whole)
        if pending == nil, emitted < readable, thinkDecided,
           closeAt != nil || !inThinkRegion {
            let tail = String(decoding: bytes[emitted ..< readable],
                              as: UTF8.self)
            if !strippedOfToolBlocks(tail).isEmpty {
                yield(tail)
                emitted = readable
            }
        }

        let reason: String
        if pending != nil {
            reason = "tool-call"
        } else if Task.isCancelled {
            reason = "cancelled"
        } else if backend.shouldStop() {
            reason = "stop"
        } else if backend.eosIds.contains(cur) {
            reason = "eos"
        } else if steps >= maxTokens {
            reason = "max-tokens"
        } else if toolAt.map({ at in
            bytes.count - at > ChatSession.maxOpenCallBytes }) == true {
            reason = "tool-runaway"
        } else {
            reason = "loop-breaker"
        }
        toolLog("decode end: \(reason) after \(steps) tokens "
            + "(think \(think), content \(content))")
        var preamble = ""
        var committedAnswer: String? = nil
        if pending == nil {
            // An opener that never resolved must not enter history as prose; a
            // quoted pair the recovery released has already cleared toolAt.
            let raw = toolAt.map { at in
                String(decoding: bytes[..<at], as: UTF8.self)
            } ?? backend.text(ids)
            var answer = answerText(raw, sawThink: inThinkRegion)
            if strippedOfToolBlocks(answer).isEmpty {
                answer = ""
            } else {
                for pat in [wire.toolCallOpen, wire.toolCallClose]
                where !pat.isEmpty && answer.contains(pat) {
                    Diag.shared.report(
                        "stripped stray tool markup from an answer")
                    answer = answer.replacingOccurrences(of: pat, with: "")
                }
                answer = answer.trimmingCharacters(
                    in: .whitespacesAndNewlines)
            }
            history.append(AgentMessage(role: "assistant", content: answer))
            committedAnswer = answer
        } else {
            let decoded = backend.text(ids)
            var body = inThinkRegion ? "" : decoded
            if inThinkRegion,
               let c = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[c.upperBound...])
            }
            if let tc = body.range(of: wire.toolCallOpen) {
                body = String(body[..<tc.lowerBound])
            }
            preamble = body.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let ctx = await backend.position
        let tgSec = Date().timeIntervalSince(g0)
        let tg = tgSec > 0 ? Double(steps) / tgSec : 0
        lastMetrics = TurnMetrics(ctx: ctx, thinkTokens: tally.think + think,
                                  contentTokens: tally.content + content,
                                  pp: pp, tg: tg, endReason: reason,
                                  overrun: ctx - fed, stopToken: cur)
        trace(.decode, from: g0, ctx: ctx, tokens: steps,
              summary: reason + String(format:
                  " (think %d, content %d, tg %.1f t/s)", think, content, tg),
              text: backend.text(ids))
        if let committedAnswer {
            trace(.answer, tokens: content,
                  summary: "\(committedAnswer.count) chars",
                  text: committedAnswer)
        }
        return (pending, preamble)
    }

    private func completedCall(_ text: String) -> ToolCall? {
        var result: ToolCall? = nil
        if let body = Tools.findToolCall(in: text, from: 0,
                                         open: wire.toolCallOpen,
                                         close: wire.toolCallClose) {
            result = Tools.parse(text[body])
        }
        return result
    }

    private func toolLog(_ s: String) {
        Diag.shared.report(.tools, s)
    }

    private func sanitizedArgs(_ call: ToolCall,
                               _ resolved: String?) -> [ToolArg] {
        let known = resolved
            .flatMap { name in toolSpecs.first { s in s.name == name } }
            .map { spec in Tools.parameterNames(spec.parametersJSON) }
        var out: [ToolArg] = []
        for arg in call.params where !arg.value.isEmpty {
            if let known {
                if let canon = Tools.canonicalArgName(arg.name, known),
                   !out.contains(where: { a in a.name == canon }) {
                    out.append(ToolArg(name: canon, value: arg.value))
                }
            } else {
                out.append(arg)
            }
        }
        return out
    }

    static func callSignature(_ name: String, _ args: [ToolArg]) -> String {
        name + "\u{1}" + args.map { a in a.name + "=" + a.value }
            .sorted().joined(separator: "\u{1}")
    }

    static func repeatedCall(_ prior: String) -> String {
        "You already made this exact call. It returned: \(prior)\n"
            + "Do NOT repeat it. Change the arguments, use a different tool, "
            + "or answer the user now from what you already have."
    }

    private func runTool(_ call: ToolCall,
                         resolved: String?) async -> String {
        var result = "error: no tool runner"
        if let runner {
            if let resolved {
                if let blocked = Tools.sanitize(resolved, call.params) {
                    result = blocked
                } else {
                    result = await runner.execute(resolved, call.params)
                }
            } else {
                let names = toolSpecs.map { spec in spec.name }
                result = "error: no tool named '\(call.functionName)'. "
                    + "Available tools: \(names.joined(separator: ", ")). "
                    + "If none of them fit, answer the user directly from your "
                    + "own knowledge."
            }
        }
        return result
    }

    private static let toolAliases: [String: String] = [
        "tavily_search": "web_search",
        "google_web_search": "web_search",
        "tavily_fetch": "fetch_url",
    ]

    static func resolveTool(_ name: String, _ available: [String]) -> String? {
        var result: String? = nil
        let lower = name.lowercased()
        if available.contains(name) {
            result = name
        } else if let alias = toolAliases[lower], available.contains(alias) {
            result = alias
        } else {
            var best: (name: String, dist: Int)? = nil
            for cand in available {
                let d = editDistance(lower, cand.lowercased())
                if best == nil || d < best!.dist { best = (cand, d) }
            }
            if let best, best.dist <= max(1, name.count / 4) {
                result = best.name
            }
        }
        return result
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let x = Array(a), y = Array(b)
        var result: Int
        if x.isEmpty {
            result = y.count
        } else if y.isEmpty {
            result = x.count
        } else {
            var prev = Array(0 ... y.count)
            var cur = [Int](repeating: 0, count: y.count + 1)
            for i in 1 ... x.count {
                cur[0] = i
                for j in 1 ... y.count {
                    let cost = x[i - 1] == y[j - 1] ? 0 : 1
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                                 prev[j - 1] + cost)
                }
                swap(&prev, &cur)
            }
            result = prev[y.count]
        }
        return result
    }

    private func toolContinuation(call: ToolCall, resolved: String?,
                                  preamble: String, result: String)
        async -> (seed: Int32, pp: Double) {
        let user = history.last { m in m.role == "user" }
            ?? AgentMessage(role: "user", content: "")
        let round = [
            user,
            AgentMessage(role: "assistant", content: preamble,
                         toolCalls: [AgentToolCall(
                             name: call.functionName,
                             arguments: sanitizedArgs(call, resolved))]),
            AgentMessage(role: "tool", content: result,
                         name: resolved ?? call.functionName),
        ]
        let prefix = (try? renderPrompt(
            template: template, messages: [user], tools: [],
            addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let closedText = (try? renderPrompt(
            template: template, messages: round, tools: [],
            addGenerationPrompt: false,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let fullText = (try? renderPrompt(
            template: template, messages: round, tools: [],
            addGenerationPrompt: true,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            bosToken: backend.bosToken)) ?? ""
        let head = closedText.hasPrefix(prefix)
            ? String(closedText.dropFirst(prefix.count)) : closedText
        var genText = fullText.hasPrefix(closedText)
            ? String(fullText.dropFirst(closedText.count)) : ""
        if metaTurn { genText += ChatSession.titleSeed(genText, wire) }
        genStartsThink = wire.startsInReasoning(genPrompt: genText,
                                                enabled: true)
        var seed = backend.eos
        var pp = 0.0
        do {
            let t0 = Date()
            try await backend.rewind()
            trace(.rewind, ctx: await backend.position,
                  summary: "rewind to turn mark")
            let laid = backend.encode(head)
            let gen = backend.encode(genText)
            trace(.render, tokens: laid.count + gen.count,
                  summary: "tool continuation (call + response)",
                  text: head + genText)
            let afterHead = try await backend.extend(laid)
            committed += laid
            try await backend.mark()
            seed = gen.isEmpty ? afterHead : try await backend.extend(gen)
            let sec = Date().timeIntervalSince(t0)
            let total = laid.count + gen.count
            pp = sec > 0 ? Double(total) / sec : 0
            trace(.prefill, from: t0, ctx: await backend.position,
                  tokens: total, summary: String(format: "%.0f t/s", pp))
        } catch {
            seed = backend.eos
        }
        return (seed, pp)
    }

    private func strippedOfToolBlocks(_ s: String) -> String {
        var out = s
        while let open = out.range(of: wire.toolCallOpen),
              let close = out.range(
                  of: wire.toolCallClose,
                  range: open.upperBound ..< out.endIndex) {
            out.removeSubrange(open.lowerBound ..< close.upperBound)
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func answerText(_ decoded: String, sawThink: Bool) -> String {
        var body = ""
        if sawThink {
            if let close = decoded.range(of: wire.reasoningClose) {
                body = String(decoded[close.upperBound...])
            }
        } else {
            body = decoded
        }
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
