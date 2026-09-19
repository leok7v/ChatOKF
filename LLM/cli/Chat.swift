import Foundation
import LLM

// Exactly 512 tokens under the Qwen3.5-0.8B and QwenPaw-2B tokenizers; it
// must equal llama-bench -p 512, so do NOT reflow these paragraphs.
let benchPrompt = """
You are advising the treasury team of a mid-sized manufacturing company. The company has taken on a large amount of floating-rate debt tied to a short-term reference rate, and management is worried that interest rates may rise over the next few years, which would increase their interest expense and squeeze margins. The CFO has heard that interest rate swaps can be used to manage this risk but does not understand how they actually work, what they cost, or what could go wrong. She has asked you to prepare a thorough written explanation that she can share with the board of directors at the next quarterly meeting.

Please write a clear, well-structured explanation that covers all of the following points in detail. First, define what an interest rate swap is and describe the two legs of a plain vanilla fixed-for-floating swap, explaining who pays what to whom and how the net settlement is calculated on each payment date. Second, walk through a concrete numerical example: assume a notional amount of fifty million dollars, a fixed rate of four percent, a floating rate that starts at three percent, and semi-annual payments, and show what happens to the cash flows if the floating rate rises to five percent. Third, explain how entering this swap changes the company's overall interest rate exposure and why it can be described as converting floating-rate debt into synthetic fixed-rate debt. Fourth, describe the main risks the company still faces after entering the swap, including counterparty credit risk, basis risk, and the consequences of wanting to exit the swap early if rates move against them. Finally, summarize the situations in which a swap is a good idea and the situations in which the company might prefer an interest rate cap, a collar, or simply refinancing into fixed-rate debt instead.

Before you begin, note that the board is also concerned about how a swap would appear in the company's financial statements and whether it introduces earnings volatility, so please include a short, non-technical note on how hedge accounting can align the swap's gains and losses with the underlying debt, and what happens if the hedge is later judged to be ineffective. Assume the debt has seven years remaining and cannot be prepaid without a significant penalty.

Please explain everything in plain language suitable for board members who are intelligent but not financial specialists, define any technical terms the first time you use them, and use short worked examples wherever they help make the mechanics concrete. Finally, close with a brief numbered checklist the board can review during the meeting, along with three short questions the directors should ask management before approving any hedging decision.

"""

// onReasoning / onTool fire on the session's task; the content stream here.
final class Acc: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ s: String) { lock.lock(); items.append(s); lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return items.joined() }
    var list: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
}
let traceCounter = Counter()
func traceWrite(_ obj: [String: Any]) {
    if let traceURL,
       let data = try? JSONSerialization.data(withJSONObject: obj),
       let handle = try? FileHandle(forWritingTo: traceURL) {
        handle.seekToEndOfFile()
        handle.write(data)
        handle.write(Data("\n".utf8))
        try? handle.close()
    }
}

@MainActor
func reportHints(_ s: ChatSession) async {
    let title = await s.makeTitle()
    let hint = await s.makeFollowup()
    err("[hint] title    = \(title.debugDescription)\n")
    err("[hint] followup = \(hint.debugDescription)\n")
}

func traceTurn(_ t0: Date, _ user: String, _ reasoning: String,
               _ content: String, _ tools: [String], _ m: TurnMetrics) {
    traceWrite([
        "turn": traceCounter.next(),
        "wall_ms": Int(Date().timeIntervalSince(t0) * 1000),
        "user": user,
        "reasoning": reasoning,
        "content": content,
        "tool_calls": tools,
        "ctx": m.ctx,
        "think_tokens": m.thinkTokens,
        "gen_tokens": m.contentTokens,
        "pp_tps": m.pp.isFinite ? m.pp : 0,
        "tg_tps": m.tg.isFinite ? m.tg : 0,
    ])
}

func cliToolRound(_ event: ToolRoundEvent) -> String {
    let named = event.resolved ?? event.name
    let args = event.params
        .map { arg in "\(arg.name)=\(arg.value)" }
        .joined(separator: " ")
    var out = "[tool \(event.round)] \(named) \(args)\n"
    if let result = event.result {
        out = "[tool \(event.round)] \(named) -> "
            + "\(result.count) chars\n"
    }
    return out
}

@MainActor func cliDrain(_ session: ChatSession,
                         _ stream: AsyncStream<String>,
                         _ each: (String) -> Void) async -> String {
    let cap = stopArmed ? stopAtVal : nil
    var out = ""
    var taken = 0
    var it = stream.makeAsyncIterator()
    var more = true
    while more {
        let piece = await it.next()
        let room = cap.map { n in taken < n } ?? true
        if let piece, room {
            each(piece)
            out += piece
            taken += 1
        } else {
            more = false
        }
    }
    if let cap, taken >= cap {
        stopArmed = false
        session.requestStop()
        await session.quiesce()
        err("\n[stop-at] dropped the stream after \(taken) piece(s)\n")
    }
    return out
}

// Falling through to an engine 100x slower is the bug this fixes, not the
// error handler for it.
@MainActor func metalChatOrExit(_ path: String) -> QwenMetalChat {
    var result: QwenMetalChat? = nil
    do {
        result = try QwenMetalChat(ggufPath: path)
    } catch {
        err("Metal backend unavailable: \(error)\n"
            + "re-run with --cpu for the pure-Swift SIMD engine\n")
        exit(2)
    }
    return result!
}

struct LoadedChat {
    let backend: any AgentBackend
    let template: String
    let vocabCount: Int
    let presets: SamplingPresets
    let attachments: Attachments
    var media: (any MediaEncoder)? = nil
}

struct Attachments {
    var parts: [ContentPart] = []
    var spans: [SoftSpan] = []
    var turn = 0
}

func benchIds(_ encode: (String) -> [Int32]) -> [Int32] {
    var ids = encode(benchPrompt)
    if let benchCtxVal {
        var padded = ids
        while padded.count < benchCtxVal { padded += ids }
        ids = Array(padded.prefix(benchCtxVal))
    }
    return ids
}

@MainActor func primeOrCook(_ session: ChatSession, _ path: String)
    async throws {
    let url = URL(fileURLWithPath: path)
    let t0 = Date()
    if await session.prime(from: url) {
        err(String(format: "[precook] primed in %.2fs\n",
                   Date().timeIntervalSince(t0)))
    } else {
        try await session.precook(to: url)
        err(String(format: "[precook] cooked + saved in %.1fs\n",
                   Date().timeIntervalSince(t0)))
    }
}

@MainActor func loadBonsai(_ path: String) throws -> LoadedChat {
    let out: LoadedChat
    if useGPU {
        let chat = metalChatOrExit(path)
        let backend = chat.backend()
        out = LoadedChat(backend: backend, template: chat.chatTemplate,
                         vocabCount: chat.tokenizer.vocabCount,
                         presets: chat.samplingPresets,
                         attachments: Attachments(), media: backend.media())
    } else {
        let chat = try QwenChat(ggufPath: path)
        out = LoadedChat(backend: chat.backend(), template: chat.chatTemplate,
                         vocabCount: chat.tokenizer.vocabCount,
                         presets: chat.samplingPresets,
                         attachments: Attachments())
    }
    return out
}

@MainActor func runGgufMain() async throws {
    // BEFORE the architecture routing on purpose: the kernels are shared, so
    // a golden capture has to run over BOTH lineages from one entry point.
    if let dir = metalGoldenDir {
        print(try MetalGolden.run(ggufPath: arg1, dir: dir))
        exit(0)
    }
    if rawArgs.contains("--kernel-bench") {
        print(try QwenMetalKernelBench.run(ggufPath: arg1))
        exit(0)
    }
    if rawArgs.contains("--ppl") { try runPerplexity(arg1, args) }
    let loaded: LoadedChat
    if Gemma4Model.isGemma4(path: arg1) {
        loaded = try await loadGemma(arg1, args, capVal)
    } else {
        try await runBonsaiModes()
        err("loading GGUF \(arg1)...\n")
        loaded = try loadBonsai(arg1)
    }
    err("ready (\(useGPU ? "Metal/GPU" : "SIMD/CPU"); "
        + "vocab \(loaded.vocabCount)).\n")
    if rawArgs.contains("--bench") {
        try await runBackendBench(loaded.backend,
                                  useGPU ? "Metal/GPU" : "SIMD/CPU ")
    }
    if rawArgs.contains("--probe") { try await runProbe(loaded) }
    try await runChat(loaded)
}

@MainActor private func runBonsaiModes() async throws {
    // Diff each GPU kernel against the SIMD reference, then exit.
    if rawArgs.contains("--metal-selftest") {
        err("Metal self-test on \(arg1)...\n")
        print(try QwenMetalSelfTest.run(ggufPath: arg1))
        exit(0)
    }
    // minilm.gguf is a BERT embedder + index, not a chat model.
    if rawArgs.contains("--slugs-embed") {
        let text = turnArgs.first ?? ""
        if let w = WikiSlugs(ggufPath: arg1) {
            print(w.embed(text)
                .map { String(format: "%.7f", $0) }.joined(separator: " "))
        } else {
            err("slugs: \(arg1) has no index trailer\n")
        }
        exit(0)
    }
    if rawArgs.contains("--slugs") {
        let text = turnArgs.first ?? ""
        if let w = WikiSlugs(ggufPath: arg1) {
            err("slugs: \(w.articleCount) articles\n")
            for (r, h) in w.query(text, topK: 5).enumerated() {
                let low = h.distance > 82 ? "  [low]" : ""
                print("  \(r + 1). d=\(h.distance)  id=\(h.id)  \(h.title)\(low)")
            }
        } else {
            err("slugs: \(arg1) has no index trailer\n")
        }
        exit(0)
    }
    if rawArgs.contains("--wiki") {
        let text = turnArgs.first ?? ""
        print(await Tools.wikipediaQuery(text, slugsPath: arg1))
        exit(0)
    }
    if args.flag("--tok") {
        let tok = try QwenChat(ggufPath: arg1).tokenizer
        for s in args.rest(after: "--tok") {
            let ids = tok.encode(s, addSpecial: true)
            let pieces = ids.map { tok.decode([$0]) }
            print("\(ids.count) ids  \(ids)\n  pieces \(pieces)")
        }
        exit(0)
    }
    if args.flag("--gdn-dump") {
        let dir = args.value("--gdn-dump") ?? "build/gdn-ref"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let model = try QwenModel(path: arg1)
        let eng = QwenEngine(model)
        let seq = 64
        let ids = (0..<seq).map { Int32(($0 * 6151 + 17) % model.cfg.nVocab) }
        var gin: [Float] = [], gout: [Float] = []
        eng.reset()
        var p = 0
        for id in ids {
            eng.forward(token: Int(id), pos: p) { name, il, v in
                if il == 0 && name == "attn_norm" { gin += v }
                if il == 0 && name == "attn_out" { gout += v }
            }
            p += 1
        }
        func writeBin(_ a: [Float], _ f: String) {
            a.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(fileURLWithPath: dir + "/" + f))
            }
        }
        writeBin(gin, "gdn_in.bin")
        writeBin(gout, "gdn_out.bin")
        writeBin(ids.map { Float($0) }, "ids.bin")
        err("gdn-dump: layer0 seq=\(seq) nEmbd=\(model.cfg.nEmbd) -> \(dir)\n")
        exit(0)
    }
    if args.flag("--hess") {
        let v = args.values("--hess", 4)
        let dir = v[0]
        let lo = Int(v[1])!
        let up = Int(v[2])!
        let corpus = v[3]
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
        let chat = try QwenMetalChat(ggufPath: arg1)
        let ids = chat.tokenizer.encode(text, addSpecial: true)
        chat.engine.reset()
        chat.engine.collectHessians(from: lo, upto: up)
        chat.engine.collectImatrix()
        let t0 = Date()
        for id in ids {
            _ = chat.engine.decode(id)
        }
        for name in chat.engine.hessianNames() {
            try chat.engine.hessianBytes(name)
                .write(to: URL(fileURLWithPath: dir + "/" + name + ".h32"))
        }
        for (name, v) in chat.engine.imatrixSums() {
            v.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: dir + "/" + name + ".bin"))
            }
        }
        err(String(format: "[hess] %d tokens, layers %d..%d, %.0fs -> %@\n",
                   ids.count, lo, up, Date().timeIntervalSince(t0), dir))
        exit(0)
    }
    if args.flag("--imat") {
        let dir = args.value("--imat") ?? "tmp/imat"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let chat = try QwenMetalChat(ggufPath: arg1)
        let ids = benchIds { text in
            chat.tokenizer.encode(text, addSpecial: true)
        }
        chat.engine.reset()
        chat.engine.collectImatrix()
        for id in ids {
            _ = chat.engine.decode(id)
        }
        let sums = chat.engine.imatrixSums()
        for (name, v) in sums {
            v.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: dir + "/" + name + ".bin"))
            }
        }
        err("[imat] \(ids.count) tokens, \(sums.count) sites -> \(dir)\n")
        exit(0)
    }
    if rawArgs.contains("--mtp-verify") || rawArgs.contains("--mtp-bench") {
        try runMetalMTP(arg1, verify: rawArgs.contains("--mtp-verify"))
    }
    if args.flag("--tap") {
        let tap = args.values("--tap", 2)
        let dir = tap.count > 0 ? tap[0] : "tmp/tap"
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        let chat = try QwenMetalChat(ggufPath: arg1)
        let text = tap.count > 1 ? tap[1]
            : (turnArgs.first ?? "What is 2+2?")
        var ids: [Int32] = []
        if text.hasPrefix("@") {
            let raw = (try? String(contentsOfFile: String(text.dropFirst()),
                                   encoding: .utf8)) ?? ""
            ids = raw.split(whereSeparator: { c in c == "," || c.isWhitespace })
                .compactMap { s in Int32(s) }
        } else {
            ids = chat.tokenizer.encode(probeWrap(text), addSpecial: true)
        }
        err("[tap] \(ids.count) ids, first \(ids.prefix(8))\n")
        chat.engine.reset()
        for id in ids.dropLast() {
            _ = chat.engine.decode(id)
        }
        let layers = chat.engine.tapLayers(token: Int(ids.last!))
        for (i, v) in layers.enumerated() {
            v.withUnsafeBytes { raw in
                try? Data(raw).write(to: URL(
                    fileURLWithPath: String(format: "%@/l%02d.bin", dir, i)))
            }
        }
        err("[tap] \(layers.count) vectors of \(layers[0].count) -> \(dir)\n")
        exit(0)
    }
}

@MainActor func runProbe(_ loaded: LoadedChat) async throws {
    let question = turnArgs.first
        ?? "What is 2+2? Reply with just the number."
    let prompt = (try? renderPrompt(
        template: loaded.template,
        messages: [AgentMessage(role: "user", content: question)],
        tools: [], addGenerationPrompt: true, enableThinking: false,
        bosToken: loaded.backend.bosToken)) ?? question
    let backend = loaded.backend
    let ids = backend.encode(prompt)
    let gen = max(1, capVal ?? 32)
    await backend.useSampler(nil)
    await backend.reset()
    let p0 = Date()
    var next = try await backend.extend(ids)
    let ppSec = Date().timeIntervalSince(p0)
    var out: [Int32] = []
    let g0 = Date()
    while out.count < gen && !backend.eosIds.contains(next) {
        out.append(next)
        next = try await backend.decode(next)
    }
    let tgSec = Date().timeIntervalSince(g0)
    err(String(format: "[probe] prefill %d tok in %.2fs (%.1f t/s) | "
        + "gen %d tok in %.2fs (%.1f t/s)\n",
        ids.count, ppSec, ppSec > 0 ? Double(ids.count) / ppSec : 0,
        out.count, tgSec, tgSec > 0 ? Double(out.count) / tgSec : 0))
    print("PROBE ids: \(out)")
    print("PROBE txt: \(backend.text(out))")
    exit(0)
}

@MainActor func runChat(_ loaded: LoadedChat) async throws {
    let session = ChatSession(
        backend: loaded.backend, template: loaded.template,
        system: systemPrompt, vocabSize: loaded.vocabCount,
        presets: greedyDecode ? SamplingPresets.greedy : loaded.presets,
        enableThinking: enableThinking, reasoningEffort: reasoningEffort,
        maxTokens: maxTokens,
        maxReasoning: maxReasoning, softReasoningCap: softReasoning,
        overthink: overthink, seed: seedVal, runner: toolRunner)
    await session.setSuppressReasoning(suppressReasoning)
    let live = FileManager.default.temporaryDirectory
        .appendingPathComponent("chatokf-live-\(UUID().uuidString)")
    if (try? await session.attach(live: live)) == nil {
        err("[state] this backend keeps no state on disk\n")
    }
    if let pkVal { try await primeOrCook(session, pkVal) }
    err("[chat] thinking \(enableThinking), template reasons "
        + "\(templateSupportsThinking(loaded.template)), soft tokens "
        + "\(await session.supportsSoftTokens()), tools "
        + "\(toolRunner?.tools.count ?? 0)\n")
    var index = 0
    if !turnArgs.isEmpty {
        for turn in turnArgs {
            await runTurn(session, loaded, turn, index)
            index += 1
        }
    } else {
        err("enter messages (Ctrl-D to end):\n")
        var line = readLine(strippingNewline: true)
        while let text = line, text != "/quit" {
            if !text.isEmpty {
                await runTurn(session, loaded, text, index)
                index += 1
            }
            line = readLine(strippingNewline: true)
        }
    }
    var ok = true
    if rawArgs.contains("--title") {
        let t1 = await session.makeTitle()
        let t2 = await session.makeTitle()
        err("[title] \"\(t1)\"\n")
        ok = t1 == t2 && !t1.isEmpty
        err(ok ? "[title] PASS: identical across regeneration "
                + "(state preserved)\n"
               : "[title] FAIL: t1=\"\(t1)\" t2=\"\(t2)\"\n")
    }
    if rawArgs.contains("--hint") { await reportHints(session) }
    exit(ok ? 0 : 1)
}

@MainActor private func runTurn(_ session: ChatSession, _ loaded: LoadedChat,
                                _ user: String, _ index: Int) async {
    let t0 = Date()
    let reasoning = Acc()
    let tools = Acc()
    let onReasoning: @Sendable (String) -> Void = { r in
        err(r)
        reasoning.add(r)
    }
    var stream: AsyncStream<String>? = nil
    var shown = user
    if user.hasPrefix("img:") {
        let body = user.dropFirst("img:".count)
        let cut = body.firstIndex(of: " ") ?? body.endIndex
        let text = cut < body.endIndex
            ? String(body[body.index(after: cut)...])
            : VLPrompt.defaultPrompt
        shown = "[image] \(text)"
        stream = imageStream(session, loaded.media, String(body[..<cut]),
                             text, onReasoning)
    } else if index == loaded.attachments.turn,
              !loaded.attachments.spans.isEmpty {
        stream = session.replySoft(
            user,
            parts: Gemma4Media.ordered(loaded.attachments.parts,
                                       around: .text(user)),
            spans: loaded.attachments.spans, onReasoning: onReasoning,
            onToolRound: { event in err(cliToolRound(event)) })
    } else {
        stream = session.reply(
            user, onReasoning: onReasoning,
            onTool: { name in tools.add(name) },
            onToolRound: { event in err(cliToolRound(event)) })
    }
    if let stream {
        print("\nUSER: \(shown)\nASSISTANT: ", terminator: "")
        fflush(stdout)
        let content = await cliDrain(session, stream) { piece in
            print(piece, terminator: "")
            fflush(stdout)
        }
        print()
        let m = await session.lastMetrics
        let outcome = await session.turnOutcome
        err(String(format: "[ctx %d | think %d | gen %d | pp %.1f t/s | "
            + "tg %.1f t/s | %@ %@ %.0fs]\n", m.ctx, m.thinkTokens,
            m.contentTokens, m.pp, m.tg, m.endReason, outcome.rawValue,
            Date().timeIntervalSince(t0)))
        traceTurn(t0, user, reasoning.text, content, tools.list, m)
    }
}

@MainActor private func imageStream(
    _ session: ChatSession, _ media: (any MediaEncoder)?, _ path: String,
    _ text: String, _ onReasoning: @escaping @Sendable (String) -> Void
) -> AsyncStream<String>? {
    let data = try? Data(contentsOf: URL(fileURLWithPath: path))
    var out: AsyncStream<String>? = nil
    if let media, let data,
       let got = try? media.image(data, budget: imageBudget) {
        err("[vl] \(path): \(got.rows) soft tokens\n")
        out = session.replySoft(text, parts: got.parts + [.text(text)],
                                spans: got.spans, onReasoning: onReasoning)
    } else {
        err("no vision: mmproj missing or this backend has no tower\n")
    }
    return out
}

// Apples-to-apples with `llama-bench -p N -n 128`, shared so every lineage
// is measured the same way rather than each growing its own timer.
@MainActor func runBackendBench(_ backend: any AgentBackend,
                                _ label: String) async throws {
    let ids = benchIds { text in backend.encode(text) }
    let gen = capVal ?? 128
    await backend.useSampler(nil)
    await backend.reset()
    var warm = try await backend.extend(ids)
    for _ in 0 ..< 8 { warm = try await backend.decode(warm) }
    await backend.reset()
    let p0 = Date()
    var next = try await backend.extend(ids)
    let ppSec = Date().timeIntervalSince(p0)
    let g0 = Date()
    for _ in 0 ..< gen { next = try await backend.decode(next) }
    let tgSec = Date().timeIntervalSince(g0)
    print(String(format: "%@  pp%d %.1f t/s  |  tg%d %.1f t/s", label,
                 ids.count, Double(ids.count) / ppSec, gen,
                 Double(gen) / tgSec))
    exit(0)
}

// The drafter is blk.<nLayer> of the SAME GGUF; --mtp-verify proves the spec
// stream token-identical to plain greedy, --mtp-bench times it.
@MainActor private func mtpSampler(_ chat: QwenMetalChat) -> Sampler? {
    var out: Sampler? = nil
    if rawArgs.contains("--mtp-sampled") {
        var cfg = chat.samplingPresets.select(thinking: false, vision: false)
        cfg.seed = seedVal != 0 ? seedVal : 1
        out = Sampler(vocabSize: chat.tokenizer.vocabCount, config: cfg)
    }
    return out
}

@MainActor func runMetalMTP(_ path: String, verify: Bool) throws {
    let chat = try QwenMetalChat(ggufPath: path)
    let eng = chat.engine
    let n = specNVal ?? 2
    let sampled = rawArgs.contains("--mtp-sampled")
    eng.loadMTP(drafts: n)
    if !eng.mtpReady {
        err("\(path) carries no nextn drafter\n")
        exit(1)
    }
    let ids = benchIds { text in
        chat.tokenizer.encode(text, addSpecial: true)
    }
    let gen = capVal ?? (verify ? 64 : 128)
    // Both arms warm before either is timed: plain runs first and would
    // otherwise pay the cold cache alone.
    eng.reset()
    eng.sampler = mtpSampler(chat)
    var warm = eng.extend(ids)
    for _ in 0 ..< 8 { warm = eng.decode(warm) }
    eng.reset()
    eng.sampler = mtpSampler(chat)
    var plain: [Int32] = []
    var next = eng.extend(ids)
    let g0 = Date()
    var k = 0
    while k < gen {
        plain.append(next)
        next = eng.decodePlain(next)
        k += 1
    }
    let plainSec = Date().timeIntervalSince(g0)
    eng.reset()
    eng.sampler = mtpSampler(chat)
    var spec: [Int32] = []
    var cur = eng.extend(ids)
    let s0 = Date()
    // Through decode(), the path the app takes: gates the queue too.
    var k2 = 0
    while k2 < gen {
        spec.append(cur)
        cur = eng.decode(cur)
        k2 += 1
    }
    let specSec = Date().timeIntervalSince(s0)
    spec = Array(spec.prefix(gen))
    let cycles = max(eng.specCycles, 1)
    let tpc = Double(eng.specCommitted) / Double(cycles)
    let acc = Double(eng.specAccepted) / Double(max(eng.specDrafted, 1))
    if verify {
        var diff = -1
        var i = 0
        while i < gen && diff < 0 {
            if plain[i] != spec[i] { diff = i }
            i += 1
        }
        print("VERIFY-MTP (metal, n=\(n), gen=\(gen)): "
              + (diff < 0 ? "\(gen)/\(gen) EXACT"
                          : "DIVERGES at \(diff) "
                            + "(plain \(plain[diff]) spec \(spec[diff]))"))
        print("PLAIN ids: " + plain.map { id in String(id) }
            .joined(separator: " "))
    } else {
        print(String(
            format: "Metal/GPU  tg%d %.1f t/s  |  MTP n=%d %.1f t/s  [%@]",
            gen, Double(gen) / plainSec, n, Double(gen) / specSec,
            sampled ? "sampled seed \(seedVal != 0 ? seedVal : 1)" : "greedy"))
    }
    print(String(format: "  %.2f tok/cycle over %d cycles, accept %.0f%%",
                 tpc, cycles, acc * 100))
    exit(0)
}
