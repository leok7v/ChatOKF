import Foundation

// Set membership rather than a sentinel: 0 and -1 are both legal values.
public enum SamplerField: Sendable, CaseIterable {
    case temperature
    case topP
    case topK
    case minP
    case typicalP
    case repeatPenalty
    case repeatLastN
    case presencePenalty
    case frequencyPenalty
    case seed
    case dryMultiplier
    case dryBase
    case dryAllowedLength
    case xtcProbability
    case xtcThreshold
    case mirostat
    case mirostatTau
    case mirostatEta
}

public struct SamplerConfig: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var typicalP: Float
    public var repeatPenalty: Float
    public var repeatLastN: Int
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    public var dryMultiplier: Float
    public var dryBase: Float
    public var dryAllowedLength: Int
    public var xtcProbability: Float
    public var xtcThreshold: Float
    public var mirostat: Int
    public var mirostatTau: Float
    public var mirostatEta: Float
    public var seed: UInt64
    public var setMask: Set<SamplerField>

    public init(
        temperature: Float = 0.8,
        topP: Float = 0.95,
        topK: Int = 40,
        minP: Float = 0.05,
        typicalP: Float = 1.0,
        repeatPenalty: Float = 1.0,
        repeatLastN: Int = 64,
        presencePenalty: Float = 0.0,
        frequencyPenalty: Float = 0.0,
        dryMultiplier: Float = 0.0,
        dryBase: Float = 1.75,
        dryAllowedLength: Int = 2,
        xtcProbability: Float = 0.0,
        xtcThreshold: Float = 0.1,
        mirostat: Int = 0,
        mirostatTau: Float = 5.0,
        mirostatEta: Float = 0.1,
        seed: UInt64 = 0,
        setMask: Set<SamplerField> = []
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.typicalP = typicalP
        self.repeatPenalty = repeatPenalty
        self.repeatLastN = repeatLastN
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.dryMultiplier = dryMultiplier
        self.dryBase = dryBase
        self.dryAllowedLength = dryAllowedLength
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.mirostat = mirostat
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.seed = seed
        self.setMask = setMask
    }

    public static let `default` = SamplerConfig()

    public static let greedy = SamplerConfig(
        temperature: 0.0, topP: 1.0, topK: 0, minP: 0.0,
        repeatPenalty: 1.0, presencePenalty: 0.0,
        setMask: [.temperature, .topP, .topK, .minP,
                  .presencePenalty, .repeatPenalty])

    private func overlaying(_ src: SamplerConfig,
                            _ field: SamplerField) -> SamplerConfig {
        var out = self
        switch field {
        case .temperature: out.temperature = src.temperature
        case .topP: out.topP = src.topP
        case .topK: out.topK = src.topK
        case .minP: out.minP = src.minP
        case .typicalP: out.typicalP = src.typicalP
        case .repeatPenalty: out.repeatPenalty = src.repeatPenalty
        case .repeatLastN: out.repeatLastN = src.repeatLastN
        case .presencePenalty:
            out.presencePenalty = src.presencePenalty
        case .frequencyPenalty:
            out.frequencyPenalty = src.frequencyPenalty
        case .seed: out.seed = src.seed
        case .dryMultiplier: out.dryMultiplier = src.dryMultiplier
        case .dryBase: out.dryBase = src.dryBase
        case .dryAllowedLength:
            out.dryAllowedLength = src.dryAllowedLength
        case .xtcProbability: out.xtcProbability = src.xtcProbability
        case .xtcThreshold: out.xtcThreshold = src.xtcThreshold
        case .mirostat: out.mirostat = src.mirostat
        case .mirostatTau: out.mirostatTau = src.mirostatTau
        case .mirostatEta: out.mirostatEta = src.mirostatEta
        }
        return out
    }

    public static func resolve(user: SamplerConfig?,
                               model: SamplerConfig?,
                               default def: SamplerConfig)
        -> SamplerConfig {
        var out = def
        if let model {
            for field in SamplerField.allCases
                where model.setMask.contains(field) {
                out = out.overlaying(model, field)
            }
        }
        if let user {
            for field in SamplerField.allCases
                where user.setMask.contains(field) {
                out = out.overlaying(user, field)
            }
        }
        out.setMask = (user?.setMask ?? [])
            .union(model?.setMask ?? [])
        return out
    }

    static func from(gguf g: GGUF) -> SamplerConfig {
        var cfg = SamplerConfig.default
        if let t = g.double("general.sampling.temp") {
            cfg.temperature = Float(t); cfg.setMask.insert(.temperature)
        }
        if let p = g.double("general.sampling.top_p") {
            cfg.topP = Float(p); cfg.setMask.insert(.topP)
        }
        if let k = g.int("general.sampling.top_k") {
            cfg.topK = k; cfg.setMask.insert(.topK)
        }
        if let m = g.double("general.sampling.min_p") {
            cfg.minP = Float(m); cfg.setMask.insert(.minP)
        }
        if let p = g.double("general.sampling.presence") {
            cfg.presencePenalty = Float(p)
            cfg.setMask.insert(.presencePenalty)
        }
        if let r = g.double("general.sampling.repeat_penalty") {
            cfg.repeatPenalty = Float(r)
            cfg.setMask.insert(.repeatPenalty)
        }
        return cfg
    }

    static func row(_ any: Any?, base: SamplerConfig) -> SamplerConfig {
        var cfg = base
        if let d = any as? [String: Any] {
            if let t = d["temperature"] as? NSNumber {
                cfg.temperature = t.floatValue; cfg.setMask.insert(.temperature)
            }
            if let p = d["top_p"] as? NSNumber {
                cfg.topP = p.floatValue; cfg.setMask.insert(.topP)
            }
            if let k = d["top_k"] as? NSNumber {
                cfg.topK = k.intValue; cfg.setMask.insert(.topK)
            }
            if let m = d["min_p"] as? NSNumber {
                cfg.minP = m.floatValue; cfg.setMask.insert(.minP)
            }
            if let pp = d["presence_penalty"] as? NSNumber {
                cfg.presencePenalty = pp.floatValue
                cfg.setMask.insert(.presencePenalty)
            }
            if let rp = d["repetition_penalty"] as? NSNumber {
                cfg.repeatPenalty = rp.floatValue
                cfg.setMask.insert(.repeatPenalty)
            }
        }
        return cfg
    }

    public static func from(generationConfig url: URL) throws
        -> SamplerConfig {
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data)
        var cfg = row(obj, base: .default)
        if let dict = obj as? [String: Any],
           let s = dict["do_sample"] as? NSNumber, !s.boolValue {
            cfg.temperature = 0
            cfg.setMask.insert(.temperature)
        }
        return cfg
    }
}

public struct SamplingPresets: Sendable {
    public let thinkingText: SamplerConfig
    public let thinkingVision: SamplerConfig
    public let nonThinkingText: SamplerConfig
    public let nonThinkingVision: SamplerConfig

    public init(thinkingText: SamplerConfig, thinkingVision: SamplerConfig,
                nonThinkingText: SamplerConfig,
                nonThinkingVision: SamplerConfig) {
        self.thinkingText = thinkingText
        self.thinkingVision = thinkingVision
        self.nonThinkingText = nonThinkingText
        self.nonThinkingVision = nonThinkingVision
    }

    public func select(thinking: Bool, vision: Bool) -> SamplerConfig {
        let result: SamplerConfig
        if thinking {
            result = vision ? thinkingVision : thinkingText
        } else {
            result = vision ? nonThinkingVision : nonThinkingText
        }
        return result
    }

    public static let greedy = SamplingPresets(
        thinkingText: .greedy, thinkingVision: .greedy,
        nonThinkingText: .greedy, nonThinkingVision: .greedy)

    static func uniform(_ row: SamplerConfig) -> SamplingPresets {
        SamplingPresets(thinkingText: row, thinkingVision: row,
                        nonThinkingText: row, nonThinkingVision: row)
    }

    static func from(presets p: [String: Any],
                     base: SamplingPresets) -> SamplingPresets {
        SamplingPresets(
            thinkingText: SamplerConfig.row(
                p["thinking_text"], base: base.thinkingText),
            thinkingVision: SamplerConfig.row(
                p["thinking_vision"], base: base.thinkingVision),
            nonThinkingText: SamplerConfig.row(
                p["nonthinking_text"], base: base.nonThinkingText),
            nonThinkingVision: SamplerConfig.row(
                p["nonthinking_vision"], base: base.nonThinkingVision))
    }

    public static func from(generationConfig url: URL) throws
        -> SamplingPresets? {
        from(generationConfigText: String(
            data: try Data(contentsOf: url), encoding: .utf8))
    }

    static func from(generationConfigText text: String?) -> SamplingPresets? {
        var result: SamplingPresets? = nil
        if let text, let data = text.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) {
            let row = SamplerConfig.row(obj, base: .default)
            result = row.setMask.isEmpty ? nil : uniform(row)
            if let dict = obj as? [String: Any],
               let p = dict["_sampling_presets"] as? [String: Any] {
                result = from(presets: p, base: uniform(row))
            }
        }
        return result
    }

    static func require(gguf g: GGUF, path: String) -> SamplingPresets {
        let dir = URL(fileURLWithPath: path).deletingLastPathComponent()
        var out = from(generationConfigText:
            g.string(Tokenizer.generationConfigKey))
        if out == nil {
            out = try? from(generationConfig: dir.appendingPathComponent(
                "generation_config.json"))
        }
        if out == nil { out = from(gguf: g) }
        if out == nil {
            fatalError("\(path) suggests no sampling: put the card in "
                + "with `chatokf-cli x --meta`, or ship a "
                + "generation_config.json beside it")
        }
        return out!
    }

    static func from(gguf g: GGUF) -> SamplingPresets? {
        let row = SamplerConfig.from(gguf: g)
        return row.setMask.isEmpty ? nil : uniform(row)
    }
}

private let rngMultiplier: UInt64 = 0x2545_F491_4F6C_DD1D
private let rngFloatDivisor: Float = 16_777_216

private struct Rng {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return UInt32(truncatingIfNeeded: (state &* rngMultiplier) >> 32)
    }

    mutating func float() -> Float {
        Float(next32() >> 8) / rngFloatDivisor
    }
}

private struct ProbIndex {
    var prob: Float
    var index: Int
}

public struct Sampler: Sendable {
    public var logitMask: (@Sendable (inout [Float]) -> Void)?

    public var overthinkTokens: Set<Int32> = []
    public var overthinkLambda: Float = 0

    public var penaltyExempt: Set<Int32> = []

    public var dryBreakers: Set<Int32> = []

    public var verbatim = false

    public static let overthinkMarkers: [String] = [
        "Wait", "But", "However", "Alternatively", "Actually", "Hmm",
        "Perhaps", "Maybe", "Instead", "Otherwise", "Conversely",
        "Nevertheless", "Nonetheless", "Although", "Though", "Yet", "Still",
        "Rethinking", "Reconsidering", "reconsider", "Hold", "Unless",
        "Wait,", "Actually,", "However,", "But,", "Alternatively,",
        "Hmm,", "Well", "Okay", "OK", "So",
    ]

    private let vocabSize: Int
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private let typicalP: Float
    private let repeatPenalty: Float
    private let presencePenalty: Float
    private let frequencyPenalty: Float
    private let dryMultiplier: Float
    private let dryBase: Float
    private let dryAllowedLength: Int
    private let xtcProbability: Float
    private let xtcThreshold: Float
    private let mirostat: Int
    private let mirostatTau: Float
    private let mirostatEta: Float
    private var rng: Rng
    private var mirostatMu: Float
    private let recentCap: Int
    private var recent: [Int32]
    private var recentLen: Int
    private var recentPos: Int
    private let dryCap: Int
    private var dry: [Int32]

    public init(vocabSize: Int, config: SamplerConfig) {
        self.vocabSize = vocabSize
        temperature = config.temperature
        topP = config.topP
        topK = config.topK
        minP = config.minP
        typicalP = config.typicalP
        repeatPenalty = config.repeatPenalty
        presencePenalty = config.presencePenalty
        frequencyPenalty = config.frequencyPenalty
        dryMultiplier = config.dryMultiplier
        dryBase = config.dryBase
        dryAllowedLength = config.dryAllowedLength
        xtcProbability = config.xtcProbability
        xtcThreshold = config.xtcThreshold
        mirostat = config.mirostat
        mirostatTau = config.mirostatTau
        mirostatEta = config.mirostatEta
        rng = Rng(state: config.seed != 0 ? config.seed : 42)
        mirostatMu = 2 * config.mirostatTau
        let penalties = config.repeatPenalty != 1
            || config.presencePenalty != 0
            || config.frequencyPenalty != 0
        recentCap = penalties
            ? (config.repeatLastN > 0 ? config.repeatLastN : 64) : 0
        recent = [Int32](repeating: 0, count: recentCap)
        recentLen = 0
        recentPos = 0
        dryCap = config.dryMultiplier > 0
            ? (config.repeatLastN > 256 ? config.repeatLastN : 256) : 0
        dry = []
    }

    public mutating func accept(_ token: Int32) {
        if recentCap > 0 {
            recent[recentPos] = token
            recentPos = (recentPos + 1) % recentCap
            if recentLen < recentCap { recentLen += 1 }
        }
        if dryCap > 0 {
            dry.append(token)
            if dry.count > dryCap { dry.removeFirst() }
        }
    }

    public mutating func resetRecent() {
        recentLen = 0
        recentPos = 0
        dry.removeAll(keepingCapacity: true)
        mirostatMu = 2 * mirostatTau
    }

    public mutating func sample(_ logits: inout [Float]) -> Int32 {
        logitMask?(&logits)
        if !verbatim {
            applyDry(&logits)
            applyRepeatPenalty(&logits)
            applyPresenceFrequency(&logits)
            applyOverthink(&logits)
        }
        var result: Int32 = 0
        if temperature <= 0 || verbatim {
            result = Int32(Vectors.argmax(logits))
        } else if topK > 0 && mirostat != 2 {
            result = sampleTopKFirst(logits)
        } else {
            softmax(&logits)
            if mirostat == 2 {
                result = sampleMirostat(logits)
            } else {
                result = sampleFiltered(logits)
            }
        }
        return result
    }

    private func smallestChild(_ h: [ProbIndex], _ root: Int,
                               _ n: Int) -> Int {
        var smallest = root
        let l = 2 * root + 1
        let r = 2 * root + 2
        if l < n && h[l].prob < h[smallest].prob { smallest = l }
        if r < n && h[r].prob < h[smallest].prob { smallest = r }
        return smallest
    }

    private func siftDown(_ h: inout [ProbIndex], _ start: Int, _ n: Int) {
        var root = start
        var next = smallestChild(h, root, n)
        while next != root {
            h.swapAt(root, next)
            root = next
            next = smallestChild(h, root, n)
        }
    }

    private func selectTopK(_ logits: [Float], _ k: Int) -> [ProbIndex] {
        let cap = min(max(k, 1), logits.count)
        var heap: [ProbIndex] = []
        heap.reserveCapacity(cap)
        var i = 0
        while i < logits.count {
            let v = logits[i]
            if heap.count < cap {
                heap.append(ProbIndex(prob: v, index: i))
                if heap.count == cap {
                    var j = cap / 2 - 1
                    while j >= 0 { siftDown(&heap, j, cap); j -= 1 }
                }
            } else if v > heap[0].prob {
                heap[0] = ProbIndex(prob: v, index: i)
                siftDown(&heap, 0, cap)
            }
            i += 1
        }
        heap.sort { a, b in
            a.prob > b.prob || (a.prob == b.prob && a.index < b.index)
        }
        return heap
    }

    private func softmaxCandidates(_ c: inout [ProbIndex]) {
        if !c.isEmpty {
            let maxVal = c[0].prob
            var sum: Float = 0
            var i = 0
            while i < c.count {
                c[i].prob = expf(c[i].prob - maxVal)
                sum += c[i].prob
                i += 1
            }
            if sum > 0 {
                i = 0
                while i < c.count { c[i].prob /= sum; i += 1 }
            }
        }
    }

    private mutating func sampleTopKFirst(_ logits: [Float]) -> Int32 {
        var c = selectTopK(logits, topK)
        softmaxCandidates(&c)
        if typicalP > 0 && typicalP < 1 { c = applyTypical(c) }
        c = applyTopP(c)
        c = applyMinP(c)
        c = applyXtc(c)
        return sampleFrom(c)
    }

    private func recentSeenEarlier(_ i: Int) -> Bool {
        var j = 0
        while j < i && recent[j] != recent[i] { j += 1 }
        return j < i
    }

    private func recentCount(_ tok: Int) -> Int {
        var count = 0
        var j = 0
        while j < recentLen {
            if Int(recent[j]) == tok { count += 1 }
            j += 1
        }
        return count
    }

    private func applyRepeatPenalty(_ logits: inout [Float]) {
        if repeatPenalty != 1 && recentLen > 0 {
            var i = 0
            while i < recentLen {
                let tok = Int(recent[i])
                let use = tok >= 0 && tok < vocabSize
                    && !recentSeenEarlier(i)
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
    }

    private func applyPresenceFrequency(_ logits: inout [Float]) {
        let active = presencePenalty != 0 || frequencyPenalty != 0
        if active && recentLen > 0 {
            var i = 0
            while i < recentLen {
                let tok = Int(recent[i])
                let use = tok >= 0 && tok < vocabSize
                    && !recentSeenEarlier(i)
                    && !penaltyExempt.contains(recent[i])
                if use {
                    let count = recentCount(tok)
                    logits[tok] -= presencePenalty
                        + frequencyPenalty * Float(count)
                }
                i += 1
            }
        }
    }

    private func drySuffixMatch(_ j: Int, _ n: Int) -> Int {
        var len = 0
        while len < j && dry[j - 1 - len] == dry[n - 1 - len]
              && !dryBreakers.contains(dry[n - 1 - len]) {
            len += 1
        }
        return len + 1
    }

    private func drySeenEarlier(_ j: Int) -> Bool {
        var k = 1
        while k < j && dry[k] != dry[j] { k += 1 }
        return k < j
    }

    private func dryBestMatch(_ tok: Int, _ n: Int) -> Int {
        var best = 0
        var k = 1
        while k < n {
            if Int(dry[k]) == tok {
                let mk = drySuffixMatch(k, n)
                if mk > best { best = mk }
            }
            k += 1
        }
        return best
    }

    private func applyDry(_ logits: inout [Float]) {
        let n = dry.count
        if n >= 2 && dryMultiplier > 0 {
            let allowed = dryAllowedLength > 0 ? dryAllowedLength : 1
            var j = 1
            while j < n {
                let tok = Int(dry[j])
                let use = tok >= 0 && tok < vocabSize
                    && !drySeenEarlier(j)
                    && !penaltyExempt.contains(dry[j])
                if use {
                    let best = dryBestMatch(tok, n)
                    if best > allowed {
                        logits[tok] -= dryMultiplier
                            * powf(dryBase, Float(best - allowed))
                    }
                }
                j += 1
            }
        }
    }

    private func applyOverthink(_ logits: inout [Float]) {
        if overthinkLambda != 0 && !overthinkTokens.isEmpty {
            for t in overthinkTokens {
                let i = Int(t)
                if i >= 0 && i < logits.count {
                    logits[i] -= overthinkLambda
                }
            }
        }
    }

    private func softmax(_ x: inout [Float]) {
        let n = x.count
        var maxVal = n > 0 ? x[0] : 0
        var i = 1
        while i < n {
            if x[i] > maxVal { maxVal = x[i] }
            i += 1
        }
        var sum: Float = 0
        i = 0
        while i < n {
            x[i] = expf(x[i] - maxVal)
            sum += x[i]
            i += 1
        }
        i = 0
        while i < n {
            x[i] /= sum
            i += 1
        }
    }

    private func sortedCandidates(_ probs: [Float],
                                  cutoff: Float) -> [ProbIndex] {
        var c: [ProbIndex] = []
        var i = 0
        while i < probs.count {
            if probs[i] >= cutoff {
                c.append(ProbIndex(prob: probs[i], index: i))
            }
            i += 1
        }
        c.sort { a, b in
            a.prob > b.prob || (a.prob == b.prob && a.index < b.index)
        }
        return c
    }

    private func applyTypical(_ c: [ProbIndex]) -> [ProbIndex] {
        var result = c
        let m = c.count
        var sum: Float = 0
        for e in c { sum += e.prob }
        if m > 1 && sum > 0 {
            var entropy: Float = 0
            for e in c {
                let p = e.prob / sum
                if p > 0 { entropy -= p * logf(p) }
            }
            let surprise: [Float] = c.map { e in
                let p = e.prob / sum
                return abs(-logf(p > 0 ? p : Float(1e-30)) - entropy)
            }
            var order = Array(0..<m)
            order.sort { a, b in surprise[a] < surprise[b] }
            var cum: Float = 0
            var i = 0
            while i < m && cum < typicalP {
                cum += c[order[i]].prob / sum
                i += 1
            }
            var kept: [ProbIndex] = []
            var k = 0
            while k < i {
                kept.append(c[order[k]])
                k += 1
            }
            kept.sort { a, b in
                a.prob > b.prob || (a.prob == b.prob && a.index < b.index)
            }
            result = kept
        }
        return result
    }

    private func applyTopP(_ c: [ProbIndex]) -> [ProbIndex] {
        var result = c
        if topP > 0 && topP < 1 {
            var cum: Float = 0
            var i = 0
            while i < c.count && cum < topP {
                cum += c[i].prob
                i += 1
            }
            result = Array(c.prefix(i))
        }
        return result
    }

    private func applyMinP(_ c: [ProbIndex]) -> [ProbIndex] {
        var result = c
        if minP > 0 && !c.isEmpty {
            let thresh = minP * c[0].prob
            var k = 0
            while k < c.count && c[k].prob >= thresh { k += 1 }
            if k > 0 { result = Array(c.prefix(k)) }
        }
        return result
    }

    private mutating func applyXtc(_ c: [ProbIndex]) -> [ProbIndex] {
        var result = c
        let m = c.count
        let eligible = xtcProbability > 0 && xtcThreshold <= 0.5 && m >= 2
        if eligible && rng.float() <= xtcProbability {
            var posLast = 0
            var i = 0
            while i < m && c[i].prob >= xtcThreshold {
                posLast = i
                i += 1
            }
            if posLast > 0 { result = Array(c.suffix(m - posLast)) }
        }
        return result
    }

    private mutating func sampleFrom(_ c: [ProbIndex]) -> Int32 {
        var weights = c
        var sum: Float = 0
        let tinv = temperature != 1 ? 1 / temperature : 1
        var i = 0
        while i < weights.count {
            if temperature != 1 {
                weights[i].prob = powf(weights[i].prob, tinv)
            }
            sum += weights[i].prob
            i += 1
        }
        var result = weights.isEmpty ? Int32(0) : Int32(weights[0].index)
        if sum > 0 {
            let r = rng.float() * sum
            var cdf: Float = 0
            var k = 0
            while k < weights.count && r >= cdf + weights[k].prob {
                cdf += weights[k].prob
                k += 1
            }
            let idx = k < weights.count
                ? weights[k].index : weights[weights.count - 1].index
            result = Int32(idx)
        }
        return result
    }

    private mutating func sampleFiltered(_ probs: [Float]) -> Int32 {
        let n = probs.count
        var result: Int32 = 0
        if n > 1 {
            let cutoff = (topP > 0 && topP < 1)
                ? (1 - topP) / Float(n - 1) : Float(1e-9)
            var c = sortedCandidates(probs, cutoff: cutoff)
            if c.isEmpty {
                result = Int32(Vectors.argmax(probs))
            } else {
                if topK > 0 && c.count > topK {
                    c = Array(c.prefix(topK))
                }
                if typicalP > 0 && typicalP < 1 { c = applyTypical(c) }
                c = applyTopP(c)
                c = applyMinP(c)
                c = applyXtc(c)
                result = sampleFrom(c)
            }
        }
        return result
    }

    private mutating func sampleMirostat(_ probs: [Float]) -> Int32 {
        var c = sortedCandidates(probs, cutoff: Float(1e-9))
        var result: Int32 = 0
        if c.isEmpty {
            result = Int32(Vectors.argmax(probs))
        } else {
            var keep = 1
            while keep < c.count
                && -log2f(c[keep].prob) <= mirostatMu { keep += 1 }
            c = Array(c.prefix(keep))
            var sum: Float = 0
            for e in c { sum += e.prob }
            if sum <= 0 {
                result = Int32(c[0].index)
            } else {
                let r = rng.float() * sum
                var cdf: Float = 0
                var i = 0
                while i < c.count && r >= cdf + c[i].prob {
                    cdf += c[i].prob
                    i += 1
                }
                let picked = i < c.count ? i : c.count - 1
                let observed = -log2f(c[picked].prob)
                mirostatMu -= mirostatEta * (observed - mirostatTau)
                result = Int32(c[picked].index)
            }
        }
        return result
    }
}
