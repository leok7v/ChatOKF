import XCTest
@testable import LLM

final class SamplerTests: XCTestCase {
    private func referenceArgmax(_ v: [Float]) -> Int32 {
        var best = 0
        var i = 1
        while i < v.count {
            if v[i] > v[best] { best = i }
            i += 1
        }
        return Int32(best)
    }

    // sample() mutates its buffer in place; these tests reuse one raw `logits`
    // across calls and samplers, so hand each call a private copy.
    private func pick(_ s: inout Sampler, _ logits: [Float]) -> Int32 {
        var work = logits
        return s.sample(&work)
    }

    func testGreedyEqualsArgmax() {
        let cfg = SamplerConfig(temperature: 0, setMask: [.temperature])
        var s = Sampler(vocabSize: 6, config: cfg)
        let logits: [Float] = [0.1, 2.3, -1.0, 2.31, 0.5, 2.30]
        XCTAssertEqual(pick(&s, logits), referenceArgmax(logits))
        XCTAssertEqual(pick(&s, logits), 3)
    }

    func testTopKTruncatesToArgmax() {
        let cfg = SamplerConfig(
            temperature: 1.0, topP: 1.0, topK: 1, minP: 0.0, seed: 7,
            setMask: [.temperature, .topP, .topK, .minP, .seed])
        var s = Sampler(vocabSize: 5, config: cfg)
        let logits: [Float] = [0.2, 3.0, 0.1, 1.5, 0.4]
        var i = 0
        while i < 8 {
            XCTAssertEqual(pick(&s, logits), referenceArgmax(logits))
            i += 1
        }
    }

    func testSeededDeterminism() {
        let cfg = SamplerConfig(
            temperature: 1.0, topP: 0.95, topK: 40, minP: 0.0, seed: 99,
            setMask: [.temperature, .topP, .topK, .minP, .seed])
        let logits: [Float] = [1.0, 2.0, 0.5, 1.5, 3.0, 0.2, 2.5, 1.1]
        var a = Sampler(vocabSize: logits.count, config: cfg)
        var b = Sampler(vocabSize: logits.count, config: cfg)
        var seqA: [Int32] = []
        var seqB: [Int32] = []
        var i = 0
        while i < 12 {
            seqA.append(pick(&a, logits))
            seqB.append(pick(&b, logits))
            i += 1
        }
        XCTAssertEqual(seqA, seqB)
    }

    func testPresencePenaltyLowersRepeated() {
        let base = SamplerConfig(temperature: 0, setMask: [.temperature])
        let penal = SamplerConfig(
            temperature: 0, presencePenalty: 0.5,
            setMask: [.temperature, .presencePenalty])
        var plain = Sampler(vocabSize: 4, config: base)
        var penalized = Sampler(vocabSize: 4, config: penal)
        let logits: [Float] = [1.0, 0.9, 0.1, 0.2]
        plain.accept(0)
        penalized.accept(0)
        XCTAssertEqual(pick(&plain, logits), 0)
        XCTAssertEqual(pick(&penalized, logits), 1)
    }

    // Wire specials are exempt from the repetition penalties: the markup a tool
    func testPenaltyExemptSkipsWireTokens() {
        let penal = SamplerConfig(
            temperature: 0, presencePenalty: 0.5,
            setMask: [.temperature, .presencePenalty])
        var plain = Sampler(vocabSize: 4, config: penal)
        var exempt = Sampler(vocabSize: 4, config: penal)
        exempt.penaltyExempt = [0]
        let logits: [Float] = [1.0, 0.9, 0.1, 0.2]
        plain.accept(0)
        exempt.accept(0)
        XCTAssertEqual(pick(&plain, logits), 1)    // penalized off the top
        XCTAssertEqual(pick(&exempt, logits), 0)   // exempt keeps its logit
    }

    func testOverthinkPenaltyLowersMarkers() {
        let cfg = SamplerConfig(temperature: 0, setMask: [.temperature])
        var plain = Sampler(vocabSize: 4, config: cfg)
        var penalized = Sampler(vocabSize: 4, config: cfg)
        penalized.overthinkTokens = [1]
        penalized.overthinkLambda = 5.0
        let logits: [Float] = [1.0, 2.0, 0.5, 0.2]   // token 1 is the max
        XCTAssertEqual(pick(&plain, logits), 1)       // greedy picks the marker
        XCTAssertEqual(pick(&penalized, logits), 0)   // pushed below 0
    }

    func testOverthinkOffByDefault() {
        let cfg = SamplerConfig(temperature: 0, setMask: [.temperature])
        var s = Sampler(vocabSize: 4, config: cfg)
        s.overthinkTokens = [1]                        // set, but lambda 0
        let logits: [Float] = [1.0, 2.0, 0.5, 0.2]
        XCTAssertEqual(pick(&s, logits), 1)            // no bias with lambda 0
    }

    // DRY suppresses EXTENDING a verbatim repeat: after [1,2,3] has cycled and
    func testDryPenaltyBreaksVerbatimCycle() {
        let base = SamplerConfig(temperature: 0, setMask: [.temperature])
        var dry = SamplerConfig(temperature: 0, setMask: [.temperature])
        dry.dryMultiplier = 0.8
        var plain = Sampler(vocabSize: 4, config: base)
        var guarded = Sampler(vocabSize: 4, config: dry)
        for t in [Int32(1), 2, 3, 1, 2, 3, 1, 2] {
            plain.accept(t)
            guarded.accept(t)
        }
        let logits: [Float] = [0.0, 0.1, 0.2, 0.5]   // 3 extends the cycle
        XCTAssertEqual(pick(&plain, logits), 3)
        XCTAssertNotEqual(pick(&guarded, logits), 3)
    }

    func testASequenceBreakerEndsTheMatch() {
        var dry = SamplerConfig(temperature: 0, setMask: [.temperature])
        dry.dryMultiplier = 0.8
        var run = Sampler(vocabSize: 5, config: dry)
        var cut = Sampler(vocabSize: 5, config: dry)
        cut.dryBreakers = [4]
        for t in [Int32(1), 2, 4, 3, 1, 2, 4] {
            run.accept(t)
            cut.accept(t)
        }
        let logits: [Float] = [0.0, 0.1, 0.2, 0.5, 0.0]
        XCTAssertNotEqual(pick(&run, logits), 3, "1 2 4 then 3 is a repeat")
        XCTAssertEqual(pick(&cut, logits), 3,
                       "a match may not run across a breaker")
    }

    func testVerbatimSuspendsEveryPenalty() {
        var cfg = SamplerConfig(
            temperature: 0, repeatPenalty: 1.5, presencePenalty: 0.5,
            setMask: [.temperature, .repeatPenalty, .presencePenalty])
        cfg.dryMultiplier = 0.8
        var guarded = Sampler(vocabSize: 4, config: cfg)
        var verbatim = Sampler(vocabSize: 4, config: cfg)
        verbatim.verbatim = true
        verbatim.overthinkTokens = [3]
        verbatim.overthinkLambda = 5.0
        for t in [Int32(1), 2, 3, 1, 2, 3, 1, 2] {
            guarded.accept(t)
            verbatim.accept(t)
        }
        let logits: [Float] = [0.0, 0.1, 0.2, 0.5]
        XCTAssertNotEqual(pick(&guarded, logits), 3)
        XCTAssertEqual(pick(&verbatim, logits), 3)
    }

    func testVerbatimIsGreedy() {
        let cfg = SamplerConfig(
            temperature: 1.5, topP: 1.0, topK: 0, minP: 0.0, seed: 3,
            setMask: [.temperature, .topP, .topK, .minP, .seed])
        var s = Sampler(vocabSize: 8, config: cfg)
        s.verbatim = true
        let flat: [Float] = [1.0, 1.1, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0]
        for _ in 0 ..< 16 { XCTAssertEqual(pick(&s, flat), 1) }
    }

    func testGreedyPresetFields() {
        let p = SamplerConfig.greedy
        XCTAssertEqual(p.temperature, 0.0)
        XCTAssertEqual(p.topP, 1.0)
        XCTAssertEqual(p.topK, 0)
        XCTAssertEqual(p.presencePenalty, 0.0)
        XCTAssertTrue(p.setMask.contains(.temperature))
        XCTAssertTrue(p.setMask.contains(.presencePenalty))
    }

    func testResolvePrecedence() {
        var user = SamplerConfig.default
        user.temperature = 0.3
        user.setMask = [.temperature]
        var model = SamplerConfig.default
        model.temperature = 0.9
        model.topP = 0.5
        model.setMask = [.temperature, .topP]
        let out = SamplerConfig.resolve(
            user: user, model: model, default: .default)
        XCTAssertEqual(out.temperature, 0.3)
        XCTAssertEqual(out.topP, 0.5)
        XCTAssertEqual(out.topK, SamplerConfig.default.topK)
    }

    func testGenerationConfigParsing() throws {
        let json = """
        {"do_sample": true, "temperature": 0.7, "top_p": 0.8,
         "top_k": 50, "unrelated": "x"}
        """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gen_cfg_\(UUID().uuidString).json")
        try json.data(using: .utf8)!.write(to: url)
        let cfg = try SamplerConfig.from(generationConfig: url)
        try? FileManager.default.removeItem(at: url)
        XCTAssertEqual(cfg.temperature, 0.7)
        XCTAssertEqual(cfg.topP, 0.8)
        XCTAssertEqual(cfg.topK, 50)
        XCTAssertTrue(cfg.setMask.contains(.temperature))
        XCTAssertTrue(cfg.setMask.contains(.topP))
        XCTAssertTrue(cfg.setMask.contains(.topK))
        XCTAssertFalse(cfg.setMask.contains(.minP))
    }

    func testPresetsSelectMatrix() {
        let p = SamplingPresets(
            thinkingText: SamplerConfig(temperature: 0.1),
            thinkingVision: SamplerConfig(temperature: 0.2),
            nonThinkingText: SamplerConfig(temperature: 0.3),
            nonThinkingVision: SamplerConfig(temperature: 0.4))
        XCTAssertEqual(p.select(thinking: true, vision: false).temperature, 0.1)
        XCTAssertEqual(p.select(thinking: true, vision: true).temperature, 0.2)
        XCTAssertEqual(p.select(thinking: false, vision: false).temperature,
                       0.3)
        XCTAssertEqual(p.select(thinking: false, vision: true).temperature, 0.4)
    }

    private func writeJSON(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("presets_\(UUID().uuidString).json")
        try json.data(using: .utf8)!.write(to: url)
        return url
    }

    func testPresetsMatrixParsing() throws {
        let url = try writeJSON("""
        {"temperature": 1.0, "top_p": 0.95, "_sampling_presets": {
          "thinking_text": {"temperature": 0.9, "presence_penalty": 1.1},
          "nonthinking_text": {"temperature": 0.3, "top_p": 0.5}}}
        """)
        let p = try XCTUnwrap(SamplingPresets.from(generationConfig: url))
        try? FileManager.default.removeItem(at: url)
        XCTAssertEqual(p.thinkingText.temperature, 0.9)
        XCTAssertEqual(p.thinkingText.presencePenalty, 1.1)
        XCTAssertEqual(p.thinkingText.topP, 0.95)
        XCTAssertEqual(p.nonThinkingText.temperature, 0.3)
        XCTAssertEqual(p.thinkingVision.temperature, 1.0)
    }

    func testTopLevelRowFillsEveryCell() throws {
        let url = try writeJSON("""
        {"temperature": 0.6, "top_k": 20}
        """)
        let p = try XCTUnwrap(SamplingPresets.from(generationConfig: url))
        try? FileManager.default.removeItem(at: url)
        XCTAssertEqual(p.nonThinkingVision.temperature, 0.6)
        XCTAssertEqual(p.thinkingText.topK, 20)
    }

    func testNoSuggestedSamplingIsNil() throws {
        let url = try writeJSON("""
        {"eos_token_id": 7, "transformers_version": "4.0"}
        """)
        let p = try SamplingPresets.from(generationConfig: url)
        try? FileManager.default.removeItem(at: url)
        XCTAssertNil(p)
    }

    // The default seed is FIXED, so recording it is pointless unless a
    // caller varies it. An explicit seed must select a DIFFERENT stream.
    func testSeedSelectsTheStream() {
        let flat = [Float](repeating: 1.0, count: 64)
        func draw(_ seed: UInt64) -> [Int32] {
            var cfg = SamplerConfig(temperature: 1.0, setMask: [.temperature])
            cfg.seed = seed
            var s = Sampler(vocabSize: flat.count, config: cfg)
            return (0 ..< 24).map { _ in pick(&s, flat) }
        }
        XCTAssertEqual(draw(0), draw(0), "the default seed is not fixed")
        XCTAssertNotEqual(draw(0), draw(987654321), "--seed changed nothing")
        XCTAssertEqual(draw(987654321), draw(987654321),
                       "an explicit seed is not reproducible")
    }

    private struct ReferencePenalties {
        var vocabSize: Int
        var repeatPenalty: Float
        var presencePenalty: Float
        var frequencyPenalty: Float
        var dryMultiplier: Float
        var dryBase: Float
        var dryAllowedLength: Int
        var penaltyExempt: Set<Int32>
        var dryBreakers: Set<Int32>

        private func seenEarlier(_ w: [Int32], _ i: Int, from: Int) -> Bool {
            var j = from
            while j < i && w[j] != w[i] { j += 1 }
            return j < i
        }

        private func count(_ w: [Int32], _ tok: Int) -> Int {
            var n = 0
            for t in w where Int(t) == tok { n += 1 }
            return n
        }

        func applyRecent(_ recent: [Int32], _ logits: inout [Float]) {
            if repeatPenalty != 1 {
                var i = 0
                while i < recent.count {
                    let tok = Int(recent[i])
                    let use = tok >= 0 && tok < vocabSize
                        && !seenEarlier(recent, i, from: 0)
                        && !penaltyExempt.contains(recent[i])
                    if use {
                        if logits[tok] > 0 {
                            logits[tok] /= repeatPenalty
                        } else {
                            logits[tok] *= repeatPenalty
                        }
                    }
                    i += 1
                }
            }
            if presencePenalty != 0 || frequencyPenalty != 0 {
                var i = 0
                while i < recent.count {
                    let tok = Int(recent[i])
                    let use = tok >= 0 && tok < vocabSize
                        && !seenEarlier(recent, i, from: 0)
                        && !penaltyExempt.contains(recent[i])
                    if use {
                        logits[tok] -= presencePenalty
                            + frequencyPenalty * Float(count(recent, tok))
                    }
                    i += 1
                }
            }
        }

        private func suffixMatch(_ dry: [Int32], _ j: Int, _ n: Int) -> Int {
            var len = 0
            while len < j && dry[j - 1 - len] == dry[n - 1 - len]
                  && !dryBreakers.contains(dry[n - 1 - len]) {
                len += 1
            }
            return len + 1
        }

        private func bestMatch(_ dry: [Int32], _ tok: Int, _ n: Int) -> Int {
            var best = 0
            var k = 1
            while k < n {
                if Int(dry[k]) == tok {
                    let mk = suffixMatch(dry, k, n)
                    if mk > best { best = mk }
                }
                k += 1
            }
            return best
        }

        func applyDry(_ dry: [Int32], _ logits: inout [Float]) {
            let n = dry.count
            if n >= 2 && dryMultiplier > 0 {
                let allowed = dryAllowedLength > 0 ? dryAllowedLength : 1
                var j = 1
                while j < n {
                    let tok = Int(dry[j])
                    let use = tok >= 0 && tok < vocabSize
                        && !seenEarlier(dry, j, from: 1)
                        && !penaltyExempt.contains(dry[j])
                    if use {
                        let best = bestMatch(dry, tok, n)
                        if best > allowed {
                            logits[tok] -= dryMultiplier
                                * powf(dryBase, Float(best - allowed))
                        }
                    }
                    j += 1
                }
            }
        }

        func apply(_ history: [Int32], recentCap: Int, dryCap: Int,
                   _ logits: inout [Float]) {
            let recent = Array(history.suffix(recentCap))
            let dry = Array(history.suffix(dryCap))
            applyDry(dry, &logits)
            applyRecent(recent, &logits)
        }
    }

    private struct Lcg {
        var state: UInt64
        mutating func next(_ bound: Int) -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(bound))
        }
    }

    private func randomHistory(_ n: Int, alphabet: Int,
                               seed: UInt64) -> [Int32] {
        var rng = Lcg(state: seed)
        return (0 ..< n).map { _ in Int32(rng.next(alphabet)) }
    }

    private func repetitiveHistory(_ n: Int, period: Int,
                                   seed: UInt64) -> [Int32] {
        var rng = Lcg(state: seed)
        let cycle = (0 ..< period).map { i in Int32(i + 1) }
        return (0 ..< n).map { i in
            rng.next(16) == 0 ? Int32(rng.next(48)) : cycle[i % period]
        }
    }

    private func penaltyConfig(lastN: Int, heavy: Bool) -> SamplerConfig {
        var cfg = SamplerConfig(temperature: 0, setMask: [.temperature])
        cfg.repeatLastN = lastN
        cfg.dryMultiplier = 0.8
        if heavy {
            cfg.repeatPenalty = 1.3
            cfg.presencePenalty = 0.5
            cfg.frequencyPenalty = 0.2
        }
        return cfg
    }

    private func reference(_ cfg: SamplerConfig,
                           vocabSize: Int) -> ReferencePenalties {
        ReferencePenalties(
            vocabSize: vocabSize, repeatPenalty: cfg.repeatPenalty,
            presencePenalty: cfg.presencePenalty,
            frequencyPenalty: cfg.frequencyPenalty,
            dryMultiplier: cfg.dryMultiplier, dryBase: cfg.dryBase,
            dryAllowedLength: cfg.dryAllowedLength,
            penaltyExempt: [3, 7, 40], dryBreakers: [5, 44])
    }

    private func windows(_ cfg: SamplerConfig) -> (recent: Int, dry: Int) {
        let penalties = cfg.repeatPenalty != 1 || cfg.presencePenalty != 0
            || cfg.frequencyPenalty != 0
        return (penalties ? cfg.repeatLastN : 0,
                cfg.repeatLastN > 256 ? cfg.repeatLastN : 256)
    }

    private func assertPenaltiesMatch(_ history: [Int32], lastN: Int,
                                      heavy: Bool, _ label: String) {
        let vocabSize = 256
        let cfg = penaltyConfig(lastN: lastN, heavy: heavy)
        let oracle = reference(cfg, vocabSize: vocabSize)
        let caps = windows(cfg)
        var s = Sampler(vocabSize: vocabSize, config: cfg)
        s.penaltyExempt = oracle.penaltyExempt
        s.dryBreakers = oracle.dryBreakers
        var rng = Lcg(state: 11)
        var i = 0
        while i < history.count {
            s.accept(history[i])
            i += 1
            if i % 37 == 0 || i == history.count {
                let logits = (0 ..< vocabSize).map { _ in
                    Float(rng.next(2000)) / 100 - 10
                }
                var want = logits
                oracle.apply(Array(history.prefix(i)), recentCap: caps.recent,
                             dryCap: caps.dry, &want)
                var got = logits
                _ = s.sample(&got)
                XCTAssertEqual(got, want, "\(label) after \(i) tokens")
            }
        }
    }

    func testPenaltiesMatchTheReferenceOnRandomHistories() {
        assertPenaltiesMatch(randomHistory(700, alphabet: 64, seed: 1),
                             lastN: 64, heavy: false, "random dry only")
        assertPenaltiesMatch(randomHistory(700, alphabet: 64, seed: 2),
                             lastN: 64, heavy: true, "random all penalties")
        assertPenaltiesMatch(randomHistory(3000, alphabet: 200, seed: 3),
                             lastN: 2048, heavy: true, "random lastN 2048")
    }

    func testPenaltiesMatchTheReferenceOnRepetitiveHistories() {
        assertPenaltiesMatch(repetitiveHistory(700, period: 5, seed: 4),
                             lastN: 64, heavy: false, "cycle dry only")
        assertPenaltiesMatch(repetitiveHistory(700, period: 7, seed: 5),
                             lastN: 64, heavy: true, "cycle all penalties")
        assertPenaltiesMatch(repetitiveHistory(3000, period: 9, seed: 6),
                             lastN: 2048, heavy: true, "cycle lastN 2048")
    }

    private func perToken(_ history: [Int32], lastN: Int, heavy: Bool)
        -> (old: Double, new: Double) {
        let vocabSize = 256
        let cfg = penaltyConfig(lastN: lastN, heavy: heavy)
        let oracle = reference(cfg, vocabSize: vocabSize)
        let caps = windows(cfg)
        var s = Sampler(vocabSize: vocabSize, config: cfg)
        s.penaltyExempt = oracle.penaltyExempt
        s.dryBreakers = oracle.dryBreakers
        let warm = max(caps.recent, caps.dry)
        for t in history.prefix(warm) { s.accept(t) }
        let logits = [Float](repeating: 0.5, count: vocabSize)
        let steps = history.count - warm
        var oldSeconds = 0.0
        var newSeconds = 0.0
        var i = warm
        while i < history.count {
            var want = logits
            let began = Date()
            oracle.apply(Array(history.prefix(i)), recentCap: caps.recent,
                         dryCap: caps.dry, &want)
            oldSeconds += Date().timeIntervalSince(began)
            var got = logits
            let again = Date()
            _ = s.sample(&got)
            newSeconds += Date().timeIntervalSince(again)
            s.accept(history[i])
            i += 1
        }
        return (oldSeconds / Double(steps) * 1e6,
                newSeconds / Double(steps) * 1e6)
    }

    func testPenaltyCostPerToken() {
        let short = perToken(repetitiveHistory(356, period: 7, seed: 8),
                             lastN: 64, heavy: false)
        let long = perToken(repetitiveHistory(2148, period: 7, seed: 9),
                            lastN: 2048, heavy: true)
        print(String(format: "[penalties] window 256 dry: old %.0f us, new "
                         + "%.0f us per token", short.old, short.new))
        print(String(format: "[penalties] lastN 2048 all: old %.0f us, new "
                         + "%.0f us per token", long.old, long.new))
    }
}
