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
}
