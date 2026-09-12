import Foundation
import LLM

// An operator-facing message; the top level prints it and stops.
enum GemmaCLIError: Error, CustomStringConvertible {
    case msg(String)

    var description: String {
        let out: String
        switch self {
        case .msg(let s): out = s
        }
        return out
    }
}

@MainActor func loadGemma(_ path: String, _ args: CommandArgs,
                          _ cap: Int?) async throws -> LoadedChat {
    let t0 = Date()
    let chat = try GemmaChat(ggufPath: path)
    let c = chat.model.cfg
    err("gemma4: \(c.nLayer) layers, \(c.nEmbd) wide, \(c.nVocab) vocab, "
        + "\(c.fullLayers) full / \(c.slidingLayers) sliding, loaded in "
        + String(format: "%.1fs\n", Date().timeIntervalSince(t0)))
    err("tokenizer: \(chat.vocabCount) tokens, "
        + "eos \(chat.eosIds.sorted())\n")

    if let dir = args.value("--gemma-gate") {
        try gemmaGate(chat, dir)
        exit(0)
    }
    if let dir = args.value("--gemma-metal-gate") {
        try gemmaMetalGate(chat, dir)
        exit(0)
    }
    if let dir = args.value("--gemma-vit-gate") {
        try gemmaViTGate(chat, dir)
        exit(0)
    }
    if let dir = args.value("--gemma-audio-gate") {
        try gemmaAudioGate(chat, dir)
        exit(0)
    }
    if args.flag("--gemma-park-gate") {
        try await gemmaParkGate(chat, useGPU)
        exit(0)
    }
    if let dir = args.value("--gemma-image-gate") {
        try gemmaImageGate(chat, dir, args.flag("--ref-embeds"),
                           useGPU)
        exit(0)
    }
    if let dir = args.value("--gemma-unified-gate") {
        try gemmaUnifiedGate(chat, dir,
                             args.value("--image")
                                 ?? "LLM/fixtures/vl/ad-rexall.jpg",
                             args.value("--clip") ?? "")
        exit(0)
    }
    // A release build strips precondition text, so a missing tower would abort
    // with nothing said; only a gate scoring a tower's intermediates needs one.
    let wantsVision = ["--gemma-patch-gate", "--gemma-vit-gate"]
    let wantsAudio = ["--gemma-mel-gate", "--gemma-audio-gate"]
    if !chat.model.hasVisionTower
        && wantsVision.contains(where: { f in args.flag(f) }) {
        err("this checkpoint carries no vision tower: it embeds images "
            + "with no encoder behind them, which is not wired yet\n")
        exit(2)
    }
    if !chat.model.hasAudioTower
        && wantsAudio.contains(where: { f in args.flag(f) }) {
        err("this checkpoint carries no audio tower: it embeds audio with "
            + "no encoder behind it, which is not wired yet\n")
        exit(2)
    }
    if let file = args.value("--gemma-video-say") {
        try await gemmaVideoSay(chat, file, args.turns.first, cap,
                                metal: useGPU,
                                withAudio: args.flag("--with-audio"),
                                dumpFrames: args.value("--dump-frames"))
        exit(0)
    }
    if let file = args.value("--gemma-image-say") {
        try await gemmaImageSay(chat, file, args.turns.first, cap,
                                metal: useGPU,
                                clip: args.value("--with-audio"),
                                seconds: args.value("--seconds")
                                    .flatMap { s in Double(s) })
        exit(0)
    }
    if let dir = args.value("--gemma-patch-gate") {
        try gemmaPatchGate(chat, dir, args.value("--image")
            ?? "LLM/fixtures/vl/ad-rexall.jpg")
        exit(0)
    }
    if let dir = args.value("--gemma-mel-gate") {
        try gemmaMelGate(dir)
        exit(0)
    }
    if let file = args.value("--gemma-audio-say") {
        try await gemmaAudioSay(chat, file, args.turns.first, cap,
                                seconds: args.value("--seconds")
                                    .flatMap { s in Double(s) },
                                offset: args.value("--offset")
                                    .flatMap { s in Double(s) } ?? 0,
                                metal: useGPU,
                                dump: args.value("--dump-proj"),
                                gated: args.flag("--gate"))
        exit(0)
    }
    if args.flag("--gemma-mic") {
        try await gemmaMic(chat, args.turns.first, cap,
                           metal: useGPU,
                           seconds: args.value("--seconds")
                               .flatMap { s in Double(s) })
        exit(0)
    }
    if let dir = args.value("--gemma-batch-gate") {
        try gemmaBatchGate(chat, dir)
        exit(0)
    }
    if args.flag("--gemma-tok") {
        for s in args.turns {
            let ids = chat.encode(s)
            print("\(ids.count) ids  \(ids)")
            print("  round-trip \(chat.decode(ids).debugDescription)")
        }
        exit(0)
    }
    let backend: any AgentBackend
    var ctx: MetalContext? = nil
    if useGPU {
        let gpu = try chat.metalBackend()
        ctx = gpu.ctx
        backend = gpu
    } else {
        backend = chat.backend()
    }
    return LoadedChat(backend: backend, template: chat.chatTemplate,
                      vocabCount: chat.vocabCount,
                      presets: chat.samplingPresets,
                      attachments: try await gemmaAttachments(chat, args, ctx))
}

// In that order, so a turn can carry several modalities at once; the markup
// is the module's, which keeps this probe and the app on one definition.
@MainActor private func gemmaAttachments(_ chat: GemmaChat,
                                         _ args: CommandArgs,
                                         _ ctx: MetalContext?) async throws
    -> Attachments {
    var out = Attachments()
    out.turn = args.value("--attach-turn").flatMap { s in Int(s) } ?? 0
    let media = chat.media(ctx: ctx)
    let clipSeconds = args.value("--seconds").flatMap { s in Double(s) }
    for file in (args.value("--image") ?? "")
        .split(separator: ",").map(String.init) {
        let span = try media.image(
            try Data(contentsOf: URL(fileURLWithPath: file)),
            softTokens: args.value("--image-budget")
                .flatMap { v in Int(v) })
        err("[chat] \(file) -> \(span.rows) image soft tokens\n")
        out.parts.append(.image)
        out.spans.append(span)
    }
    if let file = args.value("--clip") {
        let pcm = try await AudioFile.samples(
            url: URL(fileURLWithPath: file),
            sampleRate: Double(chat.melConfig.sampleRate),
            seconds: clipSeconds)
        for span in try media.audio(pcm) {
            err("[chat] \(file) -> \(span.rows) audio soft tokens\n")
            out.parts.append(.audio)
            out.spans.append(span)
        }
    }
    if let file = args.value("--film") {
        let shot = try await VideoFrames.sample(
            url: URL(fileURLWithPath: file),
            count: try chat.videoWire().frames)
        let span = try media.video(frames: shot.images, seconds: shot.seconds)
        err("[chat] \(file) -> \(shot.images.count) frames, \(span.rows) "
            + "video soft tokens\n")
        out.parts.append(.video)
        out.spans.append(span)
    }
    return out
}

// The SIMD engine is already gated against the Python oracle, so agreeing
// with it token for token proves the GPU port and costs no Python.
@MainActor private func gemmaMetalGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    err("[metal-gate] replaying \(ids.count) reference tokens\n")
    let gpu = try Gemma4MetalEngine(chat.model)
    let cpu = chat.engine
    cpu.reset()
    gpu.reset()
    var worstHidden = 1.0
    var worstLogits = 1.0
    var argmaxHits = 0
    var lastCpu: Int32 = -1
    var lastGpu: Int32 = -1
    for (i, id) in ids.enumerated() {
        let h = cpu.forward(token: Int(id), pos: i)
        gpu.forward(token: Int(id), pos: i)
        worstHidden = min(worstHidden, Vectors.cosine(h, gpu.hidden()))
        let lc = cpu.logits(h)
        let lg = gpu.logits()
        worstLogits = min(worstLogits, Vectors.cosine(lc, lg))
        lastCpu = Int32(Vectors.argmax(lc))
        lastGpu = Int32(Vectors.argmax(lg))
        if lastCpu == lastGpu { argmaxHits += 1 }
        err(String(format: "\r  %d/%d", i + 1, ids.count))
    }
    err("\n")
    // TWO MODES: under SRQ the reference cannot reproduce ITSELF past 0.9585,
    // so the generation point decides; LLM_SRQ=0 must agree to fp rounding.
    let clamped = Flags.on("srq")
    let floor = clamped ? 0.95 : 0.999
    let pass = worstHidden >= floor && worstLogits >= floor
        && lastCpu == lastGpu
        && (clamped || argmaxHits == ids.count)
    err(clamped
        ? "[metal-gate] SRQ ON: accuracy mode, the generation point decides. "
          + "Re-run with --no-srq for the structural assertion.\n"
        : "[metal-gate] SRQ OFF: structural mode, the two engines must agree "
          + "to fp rounding on every token.\n")
    print(String(format: "worst hidden cosine  %.6f", worstHidden))
    print(String(format: "worst logits cosine  %.6f", worstLogits))
    print("argmax agreement     \(argmaxHits)/\(ids.count)")
    print("generation point     "
        + (lastCpu == lastGpu ? "MATCH (\(lastCpu))"
                              : "DIFFER cpu=\(lastCpu) gpu=\(lastGpu)"))
    print(pass ? "GATE PASS" : "GATE FAIL")
}

// Pure DSP, so this gates before any tower weight is touched.
@MainActor private func gemmaMelGate(_ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let wav = try npy(base.appendingPathComponent("audio.waveform.npy"))
    let want = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let mel = Gemma4Mel(.processorDefault)
    let got = mel.features(wav.values)
    let frames = want.shape.first ?? -1
    err("[mel-gate] \(wav.values.count) samples -> \(got.frames) frames, "
        + "reference \(want.shape)\n")
    let c = Vectors.cosine(want.values, got.values)
    print("frames               \(got.frames) "
        + (got.frames == frames ? "MATCH" : "vs reference \(frames)"))
    print(String(format: "features cosine      %.6f", c))
    print(got.frames == frames && c >= 0.999 ? "GATE PASS" : "GATE FAIL")
}

// Fed the dump's own mel features, so the frontend and the tower gate apart.
@MainActor private func gemmaAudioGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let feats = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let want = try npy(base.appendingPathComponent("audio.tower.npy"))
    let wantProj = try npy(base.appendingPathComponent("audio.proj.npy"))
    let frames = feats.shape.first ?? 0
    let bins = feats.shape.count > 1 ? feats.shape[1] : 0
    err("[audio-gate] \(frames) mel frames x \(bins) -> reference "
        + "\(want.shape) tower\n")
    let tower = try Gemma4Audio(chat.model)
    let t0 = Date()
    var taps: [String: [Float]] = [:]
    let got = tower.forward(mel: feats.values, frames: frames, bins: bins) {
        name, v in taps[name] = v
    }
    for name in ["sub", "s_ff1", "s_q", "s_k", "s_v", "s_relk",
                 "s_attn", "s_lconv", "s_ff2"]
        + (0..<12).map({ i in "l\(i)" }) {
        let f32f = base.appendingPathComponent("audio.f32.\(name).npy")
        if let ref32 = try? npy(f32f), let mine = taps[name] {
            print(String(format: "  %@ vs f32  %.6f", name,
                         Vectors.cosine(ref32.values, mine)))
        }
        let f = base.appendingPathComponent("audio.dbg.\(name).npy")
        if let ref = try? npy(f), let mine = taps[name] {
            let c = Vectors.cosine(ref.values, mine)
            print(String(format: "  %@ cosine  %.6f", name, c))
        }
    }
    let secs = Date().timeIntervalSince(t0)
    let cTower = Vectors.cosine(want.values, got.tower)
    let cProj = Vectors.cosine(wantProj.values, got.proj)
    let countOK = got.count == (want.shape.first ?? -1)
    print("soft tokens          \(got.count) "
        + (countOK ? "MATCH" : "vs reference \(want.shape.first ?? -1)"))
    print(String(format: "tower cosine         %.6f", cTower))
    print(String(format: "proj  cosine         %.6f", cProj))
    print(String(format: "forward              %.1fs", secs))
    // Reseeding 10-11 from the reference's layer 9 separates arithmetic from
    // input; the last two layers amplify ~175x, so the f32 dump is the oracle.
    var cTower32 = cTower
    var cProj32 = cProj
    if let t32 = try? npy(base.appendingPathComponent("audio.tower.f32.npy")) {
        cTower32 = Vectors.cosine(t32.values, got.tower)
        print(String(format: "tower vs f32 ref     %.6f", cTower32))
    }
    if let p32 = try? npy(base.appendingPathComponent("audio.proj.f32.npy")) {
        cProj32 = Vectors.cosine(p32.values, got.proj)
        print(String(format: "proj  vs f32 ref     %.6f", cProj32))
    }
    let l9 = try? npy(base.appendingPathComponent("audio.dbg.l9.npy"))
    let l11 = try? npy(base.appendingPathComponent("audio.dbg.l11.npy"))
    if let l9, let l11 {
        let re = tower.runLayers(l9.values, got.count, from: 10, through: 11)
        print(String(format: "reseeded 10-11       %.6f",
                     Vectors.cosine(l11.values, re)))
    }
    // The tower binds through the TEXT engine's context, as in the app; the
    // unread text tensors cost nothing, being mapped rather than resident.
    let gpu = try Gemma4MetalAudio(chat.model,
                                   ctx: Gemma4MetalEngine(chat.model).ctx)
    let g0 = Date()
    let mg = gpu.forward(mel: feats.values, frames: frames, bins: bins)
    let gpuSecs = Date().timeIntervalSince(g0)
    let mTower = Vectors.cosine(got.tower, mg.tower)
    let mProj = Vectors.cosine(got.proj, mg.proj)
    print(String(format: "metal tower vs simd  %.6f", mTower))
    print(String(format: "metal proj  vs simd  %.6f", mProj))
    print(String(format: "forward simd %.1fs | metal %.2fs", secs, gpuSecs))
    // EVERY pass criterion is EXACT: counts, sequence length and the token at
    // the generation point. The cosines are diagnostics, not thresholds.
    let aw = try chat.audioWire()
    func audioSpan(_ r: (tower: [Float], proj: [Float], count: Int))
        -> SoftSpan {
        SoftSpan.bracketed(begin: aw.boa, placeholder: aw.token, end: aw.eoa,
                           count: r.count, features: r.proj)
    }
    let simdSpoke = try gemmaGenerationPoint(
        chat, base, "simd", "audio", .audio, audioSpan(got))
    let metalSpoke = try gemmaGenerationPoint(
        chat, base, "metal", "audio", .audio, audioSpan(mg))
    print(countOK && mg.count == got.count && simdSpoke && metalSpoke
          ? "GATE PASS" : "GATE FAIL")
}

// The turn is rebuilt from the manifest's prompt and the TOWER's own count;
// hitting the dump's n_tokens is the self-check, else it is not scored.
@MainActor private func gemmaGenerationPoint(
    _ chat: GemmaChat, _ base: URL, _ label: String, _ section: String,
    _ part: ContentPart, _ span: SoftSpan) throws -> Bool {
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let turn = man[section] as? [String: Any] ?? [:]
    let prompt = turn["prompt"] as? String ?? ""
    let want = turn["n_tokens"] as? Int ?? -1
    let ref = try npy(base.appendingPathComponent("\(section).logits.npy"))
    let ids = try softTurnIds(chat, prompt, [part],
                              [span.placeholder: span.ids])
    var hits = 0
    var textHits = 0
    var textPositions = 0
    var generationPoint = false
    let vocab = chat.model.cfg.nVocab
    if ids.count == want {
        let eng = chat.engine
        eng.reset()
        let feed = SoftFeed([span])
        for (i, id) in ids.enumerated() {
            let hidden = feed.row(id).map { f in
                eng.forward(embedding: f, pos: i)
            } ?? eng.forward(token: Int(id), pos: i)
            let mine = Vectors.argmax(eng.logits(hidden))
            var best = 0
            var bv = -Float.greatestFiniteMagnitude
            for v in 0..<vocab where ref.values[i * vocab + v] > bv {
                bv = ref.values[i * vocab + v]
                best = v
            }
            let same = mine == best
            if same { hits += 1 }
            // Inside the audio span the model predicts a waveform it was never
            // trained to continue: near-tied and meaningless.
            if id != span.placeholder {
                textPositions += 1
                if same { textHits += 1 }
            }
            if i == ids.count - 1 { generationPoint = same }
        }
    }
    print("\(label) turn          \(ids.count) tokens vs reference \(want)"
        + (ids.count == want ? " MATCH" : " -- NOT SCORED"))
    print("\(label) argmax        \(hits)/\(max(want, 0)) overall, "
        + "\(textHits)/\(textPositions) outside the span")
    print("\(label) gen point     "
        + (generationPoint ? "MATCH" : "DIFFER"))
    return ids.count == want && generationPoint
}

// Spliced where the template's user content begins, found by rendering the
// SAME template with an empty message; no hardcoded marker string.
@MainActor private func gemmaAudioSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      seconds: Double?, offset: Double,
                                      metal: Bool, dump: String?,
                                      gated: Bool)
    async throws {
    let wire = try chat.audioWire()
    let mel = Gemma4Mel(chat.melConfig)
    let url = URL(fileURLWithPath: file)
    let t0 = Date()
    let pcm = try await AudioFile.samples(
        url: url, sampleRate: Double(mel.cfg.sampleRate),
        offset: offset, seconds: seconds)
    let secs = Double(pcm.count) / Double(mel.cfg.sampleRate)
    err(String(format: "[say] %@ -> %.2fs at %d Hz (%.1fs decode)\n",
               url.lastPathComponent, secs, mel.cfg.sampleRate,
               Date().timeIntervalSince(t0)))
    if let path = dump {
        try Data(bytes: pcm, count: pcm.count * 4)
            .write(to: URL(fileURLWithPath: path + ".pcm"))
        err("[say] wrote \(pcm.count) f32 samples to \(path).pcm\n")
    }
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let ears = try gemmaEars(chat, gpu)
    // --gate runs the file through the MIC's detector, which makes the mic path
    // testable without a microphone. Silence is DROPPED on this arm.
    let rate = Double(mel.cfg.sampleRate)
    let chunks: [(range: Range<Int>, samples: [Float], hardCut: Bool)]
    if gated {
        let gate = SpeechGate(rate: rate, maxSeconds: wire.maxSeconds)
        var said = gate.push(pcm)
        said.append(contentsOf: gate.finish())
        err(String(format: "[say] gate bar %.5f\n", gate.speechThreshold))
        chunks = said.map { u in
            let at = Int(u.startSeconds * rate)
            return (at ..< (at + u.samples.count), u.samples, u.hardCut)
        }
    } else {
        chunks = AudioChunks.split(pcm, rate: rate,
                                   maxSeconds: wire.maxSeconds)
            .map { c in (c.range, Array(pcm[c.range]), c.hardCut) }
    }
    var spans: [SoftSpan] = []
    var parts: [ContentPart] = []
    for (i, chunk) in chunks.enumerated() {
        let m0 = Date()
        let got = ears(chunk.samples)
        err(String(format: "[say] %@ %d %.1f-%.1fs -> %d soft tokens "
                   + "(%.1fs)%@\n", gated ? "utterance" : "chunk", i + 1,
                   Double(chunk.range.lowerBound) / rate,
                   Double(chunk.range.upperBound) / rate,
                   got.count, Date().timeIntervalSince(m0),
                   chunk.hardCut ? "  HARD CUT (no pause under the cap)" : ""))
        spans.append(SoftSpan.bracketed(
            begin: wire.boa, placeholder: wire.token, end: wire.eoa,
            count: got.count, features: got.proj))
        parts.append(.audio)
    }
    let question = ask ?? "Transcribe this audio."
    let ids = try softTurnIds(chat, question, parts, spans)
    let soft = spans.reduce(0) { sum, s in sum + s.rows }
    err("[say] ids \(ids.count) = \(ids.count - soft) text + \(soft) soft "
        + "across \(spans.count) chunk(s)\n")
    // Both engines share the entry point, so the decode loop is written once.
    let drive = gemmaDrive(chat, gpu)
    let p0 = Date()
    let feed = SoftFeed(spans)
    var next = drive.prefill(ids) { id in feed.row(id) }
    err(String(format: "[say] prefill %d tokens (%d soft) in %.1fs on %@\n",
               ids.count, soft, Date().timeIntervalSince(p0),
               metal ? "Metal" : "SIMD"))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = drive.step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// The model charges 25 soft tokens per second whatever is in it, so silence
// never becomes tokens; the whole session rides ONE turn.
@MainActor private func gemmaMic(_ chat: GemmaChat, _ ask: String?,
                                 _ cap: Int?, metal: Bool,
                                 seconds: Double?) async throws {
    let wire = try chat.audioWire()
    let mel = Gemma4Mel(chat.melConfig)
    let rate = Double(mel.cfg.sampleRate)
    let allowed = await Microphone.permission()
    if !allowed {
        throw GemmaCLIError.msg("microphone access was refused")
    }
    let gate = SpeechGate(rate: rate, maxSeconds: wire.maxSeconds)
    let heard = Heard()
    let mic = Microphone(rate: rate)
    try mic.start { block in heard.add(gate.push(block)) }
    err("[mic] listening at \(mel.cfg.sampleRate) Hz -- "
        + (seconds.map { s in String(format: "%.0fs\n", s) }
           ?? "press Return to stop\n"))
    if let seconds {
        try await Task.sleep(for: .seconds(seconds))
    } else {
        _ = readLine()
    }
    mic.stop()
    heard.add(gate.finish())
    let said = heard.take()
    let spoken = said.reduce(0.0) { sum, u in
        sum + Double(u.samples.count) / rate
    }
    err(String(format: "[mic] %d utterance(s), %.1fs of speech kept\n",
               said.count, spoken))
    if said.isEmpty {
        throw GemmaCLIError.msg("nothing was said")
    }
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let ears = try gemmaEars(chat, gpu)
    var spans: [SoftSpan] = []
    var parts: [ContentPart] = []
    for (i, u) in said.enumerated() {
        let got = ears(u.samples)
        err(String(format: "[mic] utterance %d at %.1fs, %.1fs -> %d soft\n",
                   i + 1, u.startSeconds,
                   Double(u.samples.count) / rate, got.count))
        spans.append(SoftSpan.bracketed(
            begin: wire.boa, placeholder: wire.token, end: wire.eoa,
            count: got.count, features: got.proj))
        parts.append(.audio)
    }
    let question = ask ?? "Write out what you hear."
    let ids = try softTurnIds(chat, question, parts, spans)
    let drive = gemmaDrive(chat, gpu)
    let feed = SoftFeed(spans)
    var next = drive.prefill(ids) { id in feed.row(id) }
    var out: [Int32] = []
    while out.count < (cap ?? 200) && !chat.eosIds.contains(next) {
        out.append(next)
        next = drive.step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// Utterances arrive on the audio thread and are read on the main one.
private final class Heard: @unchecked Sendable {
    private let lock = NSLock()
    private var said: [SpeechGate.Utterance] = []

    func add(_ more: [SpeechGate.Utterance]) {
        lock.lock()
        said.append(contentsOf: more)
        lock.unlock()
    }

    func take() -> [SpeechGate.Utterance] {
        lock.lock()
        defer { lock.unlock() }
        return said
    }
}

private func gemmaDrive(_ chat: GemmaChat, _ gpu: Gemma4MetalEngine?)
    -> (prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32,
        step: (Int32) -> Int32) {
    let out: (prefill: ([Int32], @escaping (Int32) -> [Float]?) -> Int32,
              step: (Int32) -> Int32)
    if let gpu {
        gpu.reset()
        out = ({ i, fn in gpu.extend(i, softAt: fn) }, { t in gpu.decode(t) })
    } else {
        let cpu = chat.engine
        cpu.reset()
        out = ({ i, fn in cpu.extend(i, softAt: fn) }, { t in cpu.decode(t) })
    }
    return out
}

private func gemmaEars(_ chat: GemmaChat, _ gpu: Gemma4MetalEngine?)
    throws -> ([Float]) -> (proj: [Float], count: Int) {
    let out: ([Float]) -> (proj: [Float], count: Int)
    if !chat.model.hasAudioTower {
        let um = try Gemma4UnifiedMedia(chat.model)
        out = { pcm in
            let rows = um.audio(pcm)
            return (rows.flatMap { r in r }, rows.count)
        }
    } else {
        let mel = Gemma4Mel(chat.melConfig)
        let forward: ([Float], Int, Int)
            -> (tower: [Float], proj: [Float], count: Int)
        if let gpu {
            let gt = try Gemma4MetalAudio(chat.model, ctx: gpu.ctx)
            forward = { m, f, b in gt.forward(mel: m, frames: f, bins: b) }
        } else {
            let ct = try Gemma4Audio(chat.model)
            forward = { m, f, b in ct.forward(mel: m, frames: f, bins: b) }
        }
        out = { pcm in
            let feats = mel.features(pcm)
            let a = forward(feats.values, feats.frames, mel.cfg.bins)
            return (a.proj, a.count)
        }
    }
    return out
}

private func gemmaEyes(_ chat: GemmaChat, _ gpu: Gemma4MetalEngine?)
    throws -> ([Float], [(Int, Int)]) -> (proj: [Float], count: Int) {
    let out: ([Float], [(Int, Int)]) -> (proj: [Float], count: Int)
    if !chat.model.hasVisionTower {
        let media = try Gemma4UnifiedMedia(chat.model)
        out = { pixels, pos in
            let rows = media.image(pixels: pixels, pos: pos)
            return (rows.flatMap { r in r }, rows.count)
        }
    } else if let gpu {
        let tower = try Gemma4MetalViT(chat.model, ctx: gpu.ctx)
        out = { pixels, pos in
            let r = tower.forward(pixels: pixels, pos: pos)
            return (r.proj, r.count)
        }
    } else {
        let tower = try Gemma4ViT(chat.model)
        out = { pixels, pos in
            let r = tower.forward(pixels: pixels, pos: pos)
            return (r.proj, r.count)
        }
    }
    return out
}

// The template writes one placeholder per attachment and the PROCESSOR
// expands it, so jinja decides where a modality sits in the turn.
@MainActor private func softTurnIds(_ chat: GemmaChat, _ question: String,
                                    _ parts: [ContentPart],
                                    _ expand: [Int32: [Int32]])
    throws -> [Int32] {
    let prompt = try renderPrompt(
        template: chat.chatTemplate,
        messages: [AgentMessage(role: "user", content: question,
                                contentParts: Gemma4Media.ordered(
                                    parts, around: .text(question)))],
        tools: [], addGenerationPrompt: true, enableThinking: false,
        bosToken: chat.bosToken)
    var ids: [Int32] = []
    for id in chat.encode(prompt) {
        if let replacement = expand[id] {
            ids.append(contentsOf: replacement)
        } else {
            ids.append(id)
        }
    }
    return ids
}

// A clip cut into pieces emits one <|audio|> per piece; expandSpans is what
// ChatSession uses, so the CLI and the app expand a multi-chunk turn alike.
@MainActor private func softTurnIds(_ chat: GemmaChat, _ question: String,
                                    _ parts: [ContentPart],
                                    _ spans: [SoftSpan]) throws -> [Int32] {
    let prompt = try renderPrompt(
        template: chat.chatTemplate,
        messages: [AgentMessage(role: "user", content: question,
                                contentParts: Gemma4Media.ordered(
                                    parts, around: .text(question)))],
        tools: [], addGenerationPrompt: true, enableThinking: false,
        bosToken: chat.bosToken)
    return Continuation.expandSpans(chat.encode(prompt), spans)
}

// The count comes from the TOWER, not a constant: it follows the aspect ratio.
@MainActor private func gemmaImageSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      metal: Bool, clip: String?,
                                      seconds: Double?) async throws {
    let wire = try chat.visionWire()
    let patch = try Gemma4Patchify(chat.model)
    let data = try Data(contentsOf: URL(fileURLWithPath: file))
    // A unified checkpoint has no tower, so its cut is the MERGED one: the
    // 48-pixel block is the soft token itself.
    let patched = chat.model.hasVisionTower
        ? patch.patches(data)
        : patch.merged(data, softTokens: patch.maxSoftTokens)
    if patched == nil {
        throw GemmaCLIError.msg("could not decode \(file)")
    }
    let cut = patched!
    let real = cut.pos.filter { p in p.0 >= 0 }.count
    err("[see] \(file) -> \(real) patches of \(cut.pos.count)\n")
    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(chat.model) }
    let t0 = Date()
    let got = try gemmaEyes(chat, gpu)(cut.pixels, cut.pos)
    err(String(format: "[see] %d soft tokens in %.1fs on %@\n", got.count,
               Date().timeIntervalSince(t0), metal ? "Metal" : "SIMD"))

    let question = ask ?? "Describe this image."
    var parts: [ContentPart] = [.image]
    var expand = [wire.token: SoftSpan.bracket(
        begin: wire.boi, placeholder: wire.token, end: wire.eoi,
        count: got.count)]
    var heard: [Float] = []
    var heardCount = 0
    if let clip {
        let aw = try chat.audioWire()
        let pcm = try await AudioFile.samples(
            url: URL(fileURLWithPath: clip),
            sampleRate: Double(chat.melConfig.sampleRate), seconds: seconds)
        let a = try gemmaEars(chat, gpu)(pcm)
        heard = a.proj
        heardCount = a.count
        err("[hear] \(clip) -> \(a.count) soft tokens\n")
        parts.append(.audio)
        expand[aw.token] = SoftSpan.bracket(
            begin: aw.boa, placeholder: aw.token, end: aw.eoa, count: a.count)
    }
    let ids = try softTurnIds(chat, question, parts, expand)
    let embd = chat.model.cfg.nEmbd
    let drive = gemmaDrive(chat, gpu)
    let p0 = Date()
    let audioToken = try? chat.audioWire().token
    var seenN = 0
    var heardN = 0
    var next = drive.prefill(ids) { id in
        var out: [Float]? = nil
        if id == wire.token {
            out = Array(got.proj[(seenN * embd)..<((seenN + 1) * embd)])
            seenN += 1
        } else if let audioToken, id == audioToken, heardN < heardCount {
            out = Array(heard[(heardN * embd)..<((heardN + 1) * embd)])
            heardN += 1
        }
        return out
    }
    err(String(format: "[see] prefill %d tokens (%d + %d soft) in %.1fs\n",
               ids.count, got.count, heardCount,
               Date().timeIntervalSince(p0)))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = drive.step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}

// There are no video WEIGHTS: the tower over sampled frames, on the frame
// budget, each frame carrying its own mm:ss TIMESTAMP and begin/end pair.
@MainActor private func gemmaVideoSay(_ chat: GemmaChat, _ file: String,
                                      _ ask: String?, _ cap: Int?,
                                      metal: Bool, withAudio: Bool,
                                      dumpFrames: String?) async throws {
    let g = chat.model
    let iw = try chat.visionWire()
    let film = try chat.videoWire()
    let vw = film.token
    let patch = try Gemma4Patchify(g)
    let budget = patch.patchBudget(film.softTokensPerFrame)
    let url = URL(fileURLWithPath: file)
    let shot = try await VideoFrames.sample(url: url,
                                           count: film.frames)
    err("[film] \(file) -> \(shot.images.count) frames, "
        + "\(budget) patches each\n")

    var gpu: Gemma4MetalEngine? = nil
    if metal { gpu = try Gemma4MetalEngine(g) }
    let vit = try gemmaEyes(chat, gpu)
    // Debug: the frames as the model receives them.
    if let dir = dumpFrames {
        try VideoFrames.write(shot.images, to: dir)
        err("[film] wrote \(shot.images.count) frames to \(dir)\n")
    }
    let t0 = Date()
    var feats: [Float] = []
    var perFrame = 0
    var block: [Int32] = []
    for (i, img) in shot.images.enumerated() {
        let frame = chat.model.hasVisionTower
            ? patch.patches(img, budget: budget)
            : patch.merged(img, softTokens: film.softTokensPerFrame)
        if frame == nil {
            throw GemmaCLIError.msg("frame \(i) would not patchify")
        }
        let cut = frame!
        let out = vit(cut.pixels, cut.pos)
        feats.append(contentsOf: out.proj)
        perFrame = out.count
        block.append(contentsOf:
            chat.encodeRaw(VideoFrames.stamp(shot.seconds[i]) + " "))
        block.append(contentsOf: SoftSpan.bracket(
            begin: iw.boi, placeholder: vw, end: iw.eoi, count: out.count))
    }
    err(String(format: "[film] %d soft tokens (%d x %d) in %.1fs on %@\n",
               perFrame * shot.images.count, shot.images.count, perFrame,
               Date().timeIntervalSince(t0), metal ? "Metal" : "SIMD"))

    var parts: [ContentPart] = [.video]
    var expand = [vw: block]
    var heard: [Float] = []
    if withAudio {
        let aw = try chat.audioWire()
        let pcm = try await AudioFile.samples(
            url: url, sampleRate: Double(chat.melConfig.sampleRate))
        let a = try gemmaEars(chat, gpu)(pcm)
        heard = a.proj
        err("[film] audio track -> \(a.count) soft tokens\n")
        parts.append(.audio)
        expand[aw.token] = SoftSpan.bracket(
            begin: aw.boa, placeholder: aw.token, end: aw.eoa, count: a.count)
    }
    let audioToken = withAudio ? try chat.audioWire().token : Int32(-1)
    let question = ask ?? "What happens in this video?"
    let ids = try softTurnIds(chat, question, parts, expand)
    let embd = g.cfg.nEmbd
    let drive = gemmaDrive(chat, gpu)
    var seen = 0
    var hear = 0
    let p0 = Date()
    var next = drive.prefill(ids) { id in
        var out: [Float]? = nil
        if id == vw {
            out = Array(feats[(seen * embd)..<((seen + 1) * embd)])
            seen += 1
        } else if id == audioToken {
            out = Array(heard[(hear * embd)..<((hear + 1) * embd)])
            hear += 1
        }
        return out
    }
    err(String(format: "[film] prefill %d tokens (%d soft) in %.1fs\n",
               ids.count, seen + hear, Date().timeIntervalSince(p0)))
    var out: [Int32] = []
    while out.count < (cap ?? 128) && !chat.eosIds.contains(next) {
        out.append(next)
        next = drive.step(next)
    }
    print("ASKED: \(question)")
    print("MODEL: \(chat.decode(out))")
}


// The tower gate feeds the dump's arrays, so the PROCESSOR side (resize,
// patch order, positions, padding) has never been exercised.
@MainActor private func gemmaPatchGate(_ chat: GemmaChat, _ dir: String,
                                       _ image: String) throws {
    let base = URL(fileURLWithPath: dir)
    let wantPx = try npy(base.appendingPathComponent(
        "vision.pixel_values.npy"))
    let wantPos = try npy(base.appendingPathComponent(
        "vision.position_ids.npy"))
    let p = try Gemma4Patchify(chat.model)
    let data = try Data(contentsOf: URL(fileURLWithPath: image))
    let patched = p.patches(data)
    if patched == nil {
        throw GemmaCLIError.msg("could not decode \(image)")
    }
    let got = patched!
    let n = wantPos.values.count / 2
    var posHits = 0
    var realWant = 0
    for i in 0..<n {
        let wx = Int(wantPos.values[i * 2]), wy = Int(wantPos.values[i * 2 + 1])
        if wx >= 0 || wy >= 0 { realWant += 1 }
        if got.pos[i].0 == wx && got.pos[i].1 == wy { posHits += 1 }
    }
    let cPx = Vectors.cosine(wantPx.values, got.pixels)
    let px = min(wantPx.values.count, got.pixels.count)
    var worst: Float = 0
    var total = 0.0
    for i in 0..<px {
        let d = abs(wantPx.values[i] - got.pixels[i])
        worst = max(worst, d)
        total += Double(d)
    }
    print("patches              \(got.pos.count) vs reference \(n)")
    print("real patches         \(got.pos.filter { q in q.0 >= 0 }.count) "
        + "vs \(realWant)")
    print("positions exact      \(posHits)/\(n)")
    print(String(format: "pixels cosine        %.6f", cPx))
    print(String(format: "pixels max abs diff  %.4f  mean %.6f", worst,
                 total / Double(max(px, 1))))
    // What remains is the JPEG DECODER: ImageIO and libjpeg differ by up to 12
    // of 255 before any resample. Read the MEAN; a max reports one pixel.
    let wantTower = try npy(base.appendingPathComponent("vision.tower.npy"))
    let vit = try Gemma4ViT(chat.model)
    let ran = vit.forward(pixels: got.pixels, pos: got.pos)
    let refCount = wantTower.shape.first ?? -1
    let cTower = Vectors.cosine(wantTower.values, ran.tower)
    print("soft tokens          \(ran.count) vs reference \(refCount)")
    print(String(format: "tower cosine         %.6f", cTower))
    let vw = try chat.visionWire()
    let spoke = try gemmaGenerationPoint(
        chat, base, "vision", "vision", .image,
        SoftSpan.bracketed(begin: vw.boi, placeholder: vw.token,
                           end: vw.eoi, count: ran.count,
                           features: ran.proj))
    print(posHits == n && cPx >= 0.9999 && ran.count == refCount && spoke
          ? "GATE PASS" : "GATE FAIL")
}

// Pixels and positions come from the dump, so the TOWER is what is scored.
@MainActor private func gemmaViTGate(_ chat: GemmaChat, _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let px = try npy(base.appendingPathComponent("vision.pixel_values.npy"))
    let ps = try npy(base.appendingPathComponent("vision.position_ids.npy"))
    let want = try npy(base.appendingPathComponent("vision.tower.npy"))
    let wantProj = try npy(base.appendingPathComponent("vision.proj.npy"))
    let n = ps.values.count / 2
    var pos: [(Int, Int)] = []
    for i in 0..<n {
        pos.append((Int(ps.values[i * 2]), Int(ps.values[i * 2 + 1])))
    }
    err("[vit-gate] \(n) patches, reference \(want.shape) tower / "
        + "\(wantProj.shape) proj\n")
    let vit = try Gemma4ViT(chat.model)
    let t0 = Date()
    let got = vit.forward(pixels: px.values, pos: pos)
    let cpuSecs = Date().timeIntervalSince(t0)
    let cTower = Vectors.cosine(want.values, got.tower)
    let cProj = Vectors.cosine(wantProj.values, got.proj)
    let countOK = got.count == (want.shape.first ?? -1)
    // The tower binds through the TEXT engine's context, as in the app; the
    // unread text tensors cost nothing, being mapped rather than resident.
    let gpu = try Gemma4MetalViT(chat.model,
                                 ctx: Gemma4MetalEngine(chat.model).ctx)
    let g0 = Date()
    let mg = gpu.forward(pixels: px.values, pos: pos)
    let gpuSecs = Date().timeIntervalSince(g0)
    let mTower = Vectors.cosine(got.tower, mg.tower)
    let mProj = Vectors.cosine(got.proj, mg.proj)
    print("soft tokens          \(got.count) "
        + (countOK ? "MATCH" : "vs reference \(want.shape.first ?? -1)")
        + ", metal \(mg.count)")
    print(String(format: "tower cosine vs ref  %.6f", cTower))
    print(String(format: "proj  cosine vs ref  %.6f", cProj))
    print(String(format: "metal tower vs simd  %.6f", mTower))
    print(String(format: "metal proj  vs simd  %.6f", mProj))
    print(String(format: "forward simd %.1fs | metal %.2fs", cpuSecs,
                 gpuSecs))
    // 0.95 is the model's own reproducibility band under SRQ, not picked to
    // pass.
    let vw = try chat.visionWire()
    func spanOf(_ r: (tower: [Float], proj: [Float], count: Int)) -> SoftSpan {
        SoftSpan.bracketed(begin: vw.boi, placeholder: vw.token, end: vw.eoi,
                           count: r.count, features: r.proj)
    }
    let simdSpoke = try gemmaGenerationPoint(
        chat, base, "simd", "vision", .image, spanOf(got))
    let metalSpoke = try gemmaGenerationPoint(
        chat, base, "metal", "vision", .image, spanOf(mg))
    print(countOK && mg.count == got.count && mTower >= 0.95
          && mProj >= 0.95 && simdSpoke && metalSpoke
          ? "GATE PASS" : "GATE FAIL")
}

// STRUCTURE (patches and positions equal HF's exactly) and NUMERICS (fed HF's
// OWN patches, cosine, since ours is f32 and the reference bf16) kept apart.
@MainActor private func gemmaUnifiedGate(_ chat: GemmaChat, _ dir: String,
                                         _ image: String,
                                         _ wav: String) throws {
    let base = URL(fileURLWithPath: dir)
    let media = try Gemma4UnifiedMedia(chat.model)
    let cut = try Gemma4Patchify(chat.model)
    var ok = true

    let refPx = try npy(base.appendingPathComponent("vision.pixel_values.npy"))
    let refPos = try npy(base.appendingPathComponent("vision.position_ids.npy"))
    let refEmb = try npy(base.appendingPathComponent("vision.embed.npy"))
    let dim = refPx.shape.count > 1 ? refPx.shape[1] : media.patchDim
    let count = refPx.shape[0]
    print("[unified] vision \(count) patches x \(dim)")

    // HF pads to the whole budget with (-1, -1); a unified checkpoint scatters
    // exactly what it is handed, so the comparison is against the REAL ones.
    var pos: [(Int, Int)] = []
    for i in 0..<count {
        pos.append((Int(refPos.values[i * 2]), Int(refPos.values[i * 2 + 1])))
    }
    var keep: [Int] = []
    for i in 0..<count where pos[i].0 >= 0 { keep.append(i) }
    let real = keep.count

    let mine = cut.merged(try Data(contentsOf: URL(fileURLWithPath: image)),
                          softTokens: cut.maxSoftTokens)
    if let mine {
        let sameCount = mine.pos.count == real
        var posBad = 0
        for i in 0..<min(mine.pos.count, real) {
            let wantX = Int(refPos.values[i * 2])
            let wantY = Int(refPos.values[i * 2 + 1])
            if mine.pos[i].0 != wantX || mine.pos[i].1 != wantY { posBad += 1 }
        }
        var pixMax: Float = 0
        for i in 0..<min(mine.pixels.count, refPx.values.count) {
            pixMax = max(pixMax, abs(mine.pixels[i] - refPx.values[i]))
        }
        ok = ok && sameCount && posBad == 0
        print("  patch count  \(mine.pos.count) vs \(real) real "
            + "(\(count) padded)  " + (sameCount ? "MATCH" : "MISMATCH"))
        print("  positions    \(posBad) wrong of \(real)  "
            + (posBad == 0 ? "MATCH" : "MISMATCH"))
        // Pixels may differ where STRUCTURE may not: ImageIO and libjpeg
        // disagree before any resize.
        print(String(format: "  pixels       max abs %.4f (decoder)",
                     pixMax))
    } else {
        print("  patchify FAILED")
        ok = false
    }

    let pixels = refPx.values
    var packed = [Float](repeating: 0, count: keep.count * dim)
    for (n, i) in keep.enumerated() {
        for j in 0..<dim { packed[n * dim + j] = pixels[i * dim + j] }
    }
    let got = media.image(pixels: packed, pos: keep.map { i in pos[i] })
    // The reference keeps a row per PADDED slot; gather the real ones or every
    // score is diluted by rows never produced.
    let wide = refEmb.values.count / max(count, 1)
    var refRows = [Float](repeating: 0, count: keep.count * wide)
    for (n, i) in keep.enumerated() {
        for j in 0..<wide { refRows[n * wide + j] = refEmb.values[i * wide + j] }
    }
    let rows = got.flatMap { row in row }
    let cosV = Vectors.cosine(rows, refRows)
    ok = ok && cosV >= 0.99
    print(String(format: "  embed        %d tokens  cos %.6f  %@",
                 got.count, cosV, cosV >= 0.99 ? "OK" : "FAIL"))
    // A cosine cannot see MAGNITUDE, and these rows sit beside embeddings that
    // carry embed_scale; gate the amplitude or that bug class passes.
    let scaleV = amplitude(rows) / amplitude(refRows)
    ok = ok && abs(scaleV - 1) <= 0.01
    print(String(format: "  embed scale  %.6f of reference  %@", scaleV,
                 abs(scaleV - 1) <= 0.01 ? "OK" : "FAIL"))

    let refFeat = try npy(base.appendingPathComponent(
        "audio.input_features.npy"))
    let refAud = try npy(base.appendingPathComponent("audio.embed.npy"))
    let frames = refFeat.shape[0]
    let tokens = media.audio(refFeat.values)
    let cosA = Vectors.cosine(tokens.flatMap { row in row }, refAud.values)
    let scaleA = amplitude(tokens.flatMap { row in row })
        / amplitude(refAud.values)
    ok = ok && cosA >= 0.99 && abs(scaleA - 1) <= 0.01
    print(String(format: "[unified] audio %d frames -> %d tokens  cos %.6f "
        + " scale %.6f  %@", frames, tokens.count, cosA, scaleA,
        cosA >= 0.99 && abs(scaleA - 1) <= 0.01 ? "OK" : "FAIL"))
    _ = wav
    print(ok ? "GATE PASS" : "GATE FAIL")
}

// Every stage before this matches HF yet the answers differ, so this scores
// the decoder stack; --ref-embeds feeds HF's inputs_embeds to isolate it.
@MainActor private func gemmaImageGate(_ chat: GemmaChat, _ dir: String,
                                       _ useRef: Bool,
                                       _ metal: Bool) throws {
    let base = URL(fileURLWithPath: dir)
    let ids = try npy(base.appendingPathComponent("img.ids.npy"))
        .values.map { v in Int32(v) }
    let probe = try npy(base.appendingPathComponent("img.probe.npy"))
        .values.map { v in Int(v) }
    let embeds = try npy(base.appendingPathComponent("img.embeds.npy"))
    let refLogits = try npy(base.appendingPathComponent("img.logits.npy"))
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let turn = man["image_turn"] as? [String: Any] ?? [:]
    let image = turn["image"] as? String ?? "LLM/fixtures/vl/elephant.jpeg"
    let embd = chat.model.cfg.nEmbd
    let nLayer = chat.model.cfg.nLayer
    let wire = try chat.visionWire()
    err("[image-gate] \(ids.count) ids, \(probe.count) probes, "
        + (useRef ? "REFERENCE embeddings\n" : "our own embeddings\n"))

    var mine: [Float] = []
    if !useRef {
        let patch = try Gemma4Patchify(chat.model)
        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let merged = patch.merged(data, softTokens: patch.maxSoftTokens)
        if merged == nil {
            throw GemmaCLIError.msg("could not cut \(image)")
        }
        let cut = merged!
        mine = try Gemma4UnifiedMedia(chat.model)
            .image(pixels: cut.pixels, pos: cut.pos).flatMap { r in r }
    }

    // Only the probed positions are kept, which is what the reference stored.
    let empty = [[Float]](repeating: [], count: probe.count)
    var got = [[[Float]]](repeating: empty, count: nLayer + 1)
    let attn = [[[Float]]](repeating: empty, count: nLayer)
    let mlp = [[[Float]]](repeating: empty, count: nLayer)
    let slot = Dictionary(uniqueKeysWithValues:
        probe.enumerated().map { i, p in (p, i) })
    // The BATCHED path, the only one that can express a vision block; `probe` +
    // `onLayer` read the chunk buffers between layer groups.
    let gpu = try Gemma4MetalEngine(chat.model)
    gpu.reset()
    gpu.probe = Set(probe)
    gpu.onLayer = { il, p, v in
        if let at = slot[p] {
            if il < nLayer { got[il][at] = v } else { got[nLayer][at] = v }
        }
    }
    // softAt is asked EXACTLY once per id and in order, so a cursor walks
    // either.
    var soft = 0
    var seen = 0
    _ = gpu.extend(ids) { id in
        var out: [Float]? = nil
        if useRef {
            out = Array(embeds.values[(seen * embd)..<((seen + 1) * embd)])
        } else if id == wire.token {
            out = Array(mine[(soft * embd)..<((soft + 1) * embd)])
            soft += 1
        }
        seen += 1
        return out
    }
    gpu.onLayer = nil

    var first = -1
    for il in 0...nLayer {
        let want = try npy(base.appendingPathComponent(
            String(format: "img.hidden.%02d.npy", il)))
        var worst = 1.0
        var at = -1
        var best = 0.0
        var row = ""
        for (n, p) in probe.enumerated() where !got[il][n].isEmpty {
            let ref = Array(want.values[(n * embd)..<((n + 1) * embd)])
            let c = Vectors.cosine(ref, got[il][n])
            if c < worst { worst = c; at = p }
            best = max(best, c)
            row += String(format: "  %d:%.4f", p, c)
        }
        // The FIRST diverging layer in full: all soft positions wrong or only
        // some is a path bug against a data-dependent one.
        if worst < 0.99 && first < 0 {
            first = il
            print("   probes" + row)
        }
        // The two BRANCHES at the worst position say where to look.
        var branch = ""
        if il < nLayer, let n = probe.firstIndex(of: at) {
            for (name, mine) in [("attn", attn[il]), ("mlp", mlp[il])]
                where !mine[n].isEmpty {
                let url = base.appendingPathComponent(
                    String(format: "img.%@.%02d.npy", name, il))
                if let ref = try? npy(url) {
                    let want = Array(
                        ref.values[(n * embd)..<((n + 1) * embd)])
                    branch += String(format: "  %@ %.6f", name,
                                     Vectors.cosine(want, mine[n]))
                }
            }
        }
        if il % 4 == 0 || worst < 0.99 {
            print(String(format:
                "  layer %02d  worst %.6f at pos %d, best %.6f%@%@", il,
                worst, at, best, branch,
                worst < 0.99 ? "   <-- DIVERGED" : ""))
        }
    }
    let mineTop = Vectors.argmax(gpu.logits())
    var refTop = 0
    var best = -Float.greatestFiniteMagnitude
    for v in 0..<refLogits.values.count where refLogits.values[v] > best {
        best = refLogits.values[v]
        refTop = v
    }
    print("generation point     ours \(mineTop) "
        + chat.decode([Int32(mineTop)]).debugDescription
        + " vs reference \(refTop) "
        + chat.decode([Int32(refTop)]).debugDescription)
    print(first < 0 ? "GATE PASS" : "GATE FAIL (first divergence at layer "
          + "\(first))")
}

// The ONE check an empty serializeState cannot pass: restore reproduces the
// same next token WITHOUT a forward pass, in disk-read time.
@MainActor private func gemmaParkGate(_ chat: GemmaChat,
                                      _ metal: Bool) async throws {
    let backend: any AgentBackend = metal ? try chat.metalBackend()
                                          : chat.backend()
    let ids = chat.encode(String(repeating: "The quarterly logistics review "
        + "covered warehouse throughput and pallet rotation. ", count: 40))
    let t0 = Date()
    let want = try await backend.extend(ids)
    let cooked = Date().timeIntervalSince(t0)
    let state = try await backend.saveState()
    let bytes = await backend.serializeState(state)
    err(String(format: "[park] %d ids cooked in %.2fs -> %d KiB\n",
               ids.count, cooked, bytes.count / 1024))
    if bytes.isEmpty {
        print("park bytes           EMPTY -- this backend re-prefills")
        print("GATE FAIL")
        exit(1)
    }
    await backend.reset()
    let t1 = Date()
    try await backend.loadState(try await backend.deserializeState(bytes))
    let restored = Date().timeIntervalSince(t1)
    // The engine predicts from the state alone, so the SAME token proves the
    // whole KV came back, not merely the position counter.
    let got = try await backend.decode(want)
    let again = try await backend.decode(got)
    await backend.reset()
    _ = try await backend.extend(ids)
    let ref = try await backend.decode(want)
    let ref2 = try await backend.decode(ref)
    print(String(format: "cook                 %.2fs", cooked))
    print(String(format: "restore              %.3fs  (%.0fx faster)",
                 restored, cooked / max(restored, 1e-6)))
    print("next token           \(got) vs \(ref) "
        + (got == ref ? "MATCH" : "DIFFER"))
    print("and the one after    \(again) vs \(ref2) "
        + (again == ref2 ? "MATCH" : "DIFFER"))
    print(got == ref && again == ref2 && restored < cooked / 4
          ? "GATE PASS" : "GATE FAIL")
}

// Root mean square, the size a cosine deliberately divides away.
private func amplitude(_ a: [Float]) -> Float {
    var sum = 0.0
    for v in a { sum += Double(v) * Double(v) }
    return a.isEmpty ? 0 : Float((sum / Double(a.count)).squareRoot())
}

private func npy(_ url: URL) throws -> (values: [Float], shape: [Int]) {
    let d = try Data(contentsOf: url)
    let headerLen = Int(d[8]) | (Int(d[9]) << 8)
    let start = 10 + headerLen
    let header = String(decoding: d[10..<start], as: UTF8.self)
    var shape: [Int] = []
    if let open = header.range(of: "'shape': ("),
       let shut = header[open.upperBound...].firstIndex(of: ")") {
        shape = header[open.upperBound..<shut]
            .split(separator: ",")
            .compactMap { part in Int(part.trimmingCharacters(
                in: .whitespaces)) }
    }
    let count = (d.count - start) / 4
    var values = [Float](repeating: 0, count: count)
    _ = values.withUnsafeMutableBytes { dst in
        d.copyBytes(to: dst, from: start..<d.count)
    }
    return (values, shape)
}

// One flat f32 blob per array, so the Python side needs no Swift parsing.
@MainActor private func gemmaGate(_ chat: GemmaChat, _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    err("[gate] replaying \(ids.count) reference tokens\n")

    let eng = chat.engine
    let nLayer = chat.model.cfg.nLayer
    var hidden = [[Float]](repeating: [], count: nLayer + 1)
    var finals: [Float] = []
    var logits: [Float] = []
    eng.reset()
    let t0 = Date()
    for (i, id) in ids.enumerated() {
        let h = eng.forward(token: Int(id), pos: i) { name, il, v in
            if name == "embed" {
                hidden[0].append(contentsOf: v)
            } else if name == "l_out" {
                hidden[il + 1].append(contentsOf: v)
            }
        }
        finals.append(contentsOf: h)
        logits.append(contentsOf: eng.logits(h))
        err(String(format: "\r  %d/%d tokens (%.0fs)", i + 1, ids.count,
                   Date().timeIntervalSince(t0)))
    }
    err("\n")
    for (i, h) in hidden.enumerated() {
        try writeF32(h, base.appendingPathComponent(
            String(format: "swift.hidden.%02d.bin", i)))
    }
    try writeF32(finals, base.appendingPathComponent("swift.final.bin"))
    try writeF32(logits, base.appendingPathComponent("swift.logits.bin"))
    err("[gate] wrote \(hidden.count) hidden + final + logits to \(dir)\n")
}

private func writeF32(_ a: [Float], _ url: URL) throws {
    try a.withUnsafeBytes { raw in try Data(raw).write(to: url) }
}

// Prefill feeds DECODE, so a wrong chunk shows many tokens later; what must
// match is the state left behind, at a position that is NOT zero.
@MainActor private func gemmaBatchGate(_ chat: GemmaChat,
                                       _ dir: String) throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(
        contentsOf: base.appendingPathComponent("manifest.json"))
    let man = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let text = man["text"] as! [String: Any]
    let ids = (text["input_ids"] as! [Int]).map { id in Int32(id) }
    // The chunk scratch is sized at INIT from defaultBatch; a larger `batch`
    // wrote off every chunk buffer. err + exit: release strips messages.
    if Gemma4MetalEngine.defaultBatch < 2 {
        err("set --gemma-batch=256 (or any N > 1) before this gate: the "
            + "engine sizes its chunk scratch for that many tokens at init, "
            + "and the gate compares batch 1 against it\n")
        exit(2)
    }
    let gpu = try Gemma4MetalEngine(chat.model)
    // Compare against what the engine ALLOCATED, not a literal.
    let wide = Gemma4MetalEngine.defaultBatch
    // The two paths CANNOT agree to 1.0: rint() is discontinuous, so batch vs
    // batch-1 sits at 0.988; the question is accuracy against HF's logits.
    let ref = try npy(base.appendingPathComponent("text.logits.npy"))
    let vocab = ref.shape.last ?? ref.values.count
    let want = Array(ref.values.suffix(vocab))

    func run(_ b: Int, warm: [Int32],
             _ feed: [Int32]) -> (hidden: [Float], logits: [Float]) {
        gpu.batch = b
        gpu.reset()
        // A non-empty warmup is the point: it leaves pos > 0, so the chunk
        // that follows starts mid-KV exactly as a real second turn does.
        if !warm.isEmpty { _ = gpu.extend(warm) }
        _ = gpu.extend(feed)
        return (gpu.hidden(), gpu.logits())
    }
    // The reference prompt is shorter than the window, so the sliding branch
    // NEVER runs; a mutation test proved it. Repeating past the window evicts.
    var long = ids
    while long.count <= chat.model.cfg.slidingWindow { long += ids }
    // With LLM_SRQ=0 the clamp is the identity and the paths must agree to fp
    // rounding; SRQ-on cannot pass ~0.99 however correct the code.
    let clamped = Flags.on("srq")
    let floorHidden = clamped ? 0.95 : 0.999999
    err(clamped
        ? "[batch-gate] SRQ ON: accuracy mode, structural defects are BELOW "
          + "the rint() noise floor here. Re-run with --no-srq.\n"
        : "[batch-gate] SRQ OFF: structural mode, the two paths must agree "
          + "to fp rounding.\n")
    var failed = 0
    for (label, warm, feed) in [("from pos 0", [Int32](), ids),
                                ("mid-KV", Array(ids.prefix(7)), ids),
                                ("windowed", [Int32](), long)] {
        let one = run(1, warm: warm, feed)
        let many = run(wide, warm: warm, feed)
        let ch = Vectors.cosine(one.hidden, many.hidden)
        let cl = Vectors.cosine(one.logits, many.logits)
        var worst: Float = 0
        for i in 0..<min(one.hidden.count, many.hidden.count) {
            worst = max(worst, abs(one.hidden[i] - many.hidden[i]))
        }
        let a1 = Vectors.argmax(one.logits)
        let a2 = Vectors.argmax(many.logits)
        print(String(format: "%-10s hidden %.7f  logits %.7f  maxAbs %.5f  "
                     + "argmax %@", (label as NSString).utf8String!, ch, cl,
                     worst, a1 == a2 ? "MATCH (\(a1))"
                                     : "DIFFER \(a1) vs \(a2)"))
        // Only meaningful with the clamp ON: SRQ-off is a materially different
        // model, so scoring it against HF measures that gap.
        if warm.isEmpty && clamped {
            let hf = Vectors.argmax(want)
            print(String(format: "  vs HF    batch-1 %.7f  batched %.7f  "
                         + "HF argmax %d, ours %d / %d",
                         Vectors.cosine(one.logits, want),
                         Vectors.cosine(many.logits, want), hf, a1, a2))
        }
        if ch < floorHidden || a1 != a2 {
            failed += 1
            print(String(format: "  FAIL %@: hidden %.7f below %.7f",
                         (label as NSString).utf8String!, ch, floorHidden))
        }
    }
    print(failed == 0 ? "GATE PASS" : "GATE FAIL (\(failed))")
    if failed > 0 { exit(1) }
}
