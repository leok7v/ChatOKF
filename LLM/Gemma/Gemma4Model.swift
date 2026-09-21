import Foundation

func requireInt(_ g: GGUF, _ key: String, _ why: String) throws -> Int {
    let value = g.int(key)
    if value == nil { throw GGUFErr.parse("\(key) is absent; \(why)") }
    return value!
}

public struct Gemma4Config {
    public let nEmbd: Int
    public let nLayer: Int
    let nHead: Int
    let nHeadKV: Int
    let nHeadKVFull: Int
    // The full layers of a unified checkpoint have no value projection: the
    // value is the key projection's output BEFORE k_norm and rope, via v_norm.
    let kEqV: Bool
    let headDimSliding: Int
    let headDimFull: Int
    let eps: Float
    public let nVocab: Int
    let perLayerDim: Int
    public let slidingWindow: Int
    let kvSharedLayers: Int
    let logitSoftcap: Float
    let embedScale: Float
    let perLayerEmbedScale: Float
    let ropeBaseSliding: Float
    let ropeBaseFull: Float
    // `proportional` rope rotates only the first int(pf * head_dim / 2) pairs
    // and leaves the rest as identity; the pairing stays (j, j + head_dim/2).
    let rotatedPairsFull: Int
    let rotatedPairsSliding: Int
    let layerFull: [Bool]
    // One image's tokens see each other in BOTH directions, only on the SLIDING
    // layers; without it a 12B reads nonsense from an image (40 of 48 layers).
    let blockwiseVision: Bool
    // Audio is deliberately NOT a block: HF's block ids come from image and
    // video.
    let visionTokens: Set<Int32>
    let activation: GemmaActivation
    // Every multimodal position gathers THIS row, not the placeholder's.
    let padTokenId: Int

    var firstShared: Int { nLayer - kvSharedLayers }
    public var fullLayers: Int {
        layerFull.filter { full in full }.count
    }
    public var slidingLayers: Int {
        layerFull.filter { full in !full }.count
    }
    func isFull(_ il: Int) -> Bool { layerFull[il] }
    func headDim(_ il: Int) -> Int {
        isFull(il) ? headDimFull : headDimSliding
    }
    func headCountKV(_ il: Int) -> Int {
        isFull(il) ? nHeadKVFull : nHeadKV
    }
    func isShared(_ il: Int) -> Bool { il >= firstShared }
    var hasPerLayerInputs: Bool { perLayerDim > 0 }

    // The LAST non-shared layer of each type (E2B: L13 sliding, L14 full), so
    // L13 keeps an unwindowed history; only the QUERY's own block matters.

    func visionBlocks(_ ids: [Int32], from base: Int) -> [(Int, Int)] {
        var out = [(Int, Int)](repeating: (0, 0), count: ids.count)
        var i = 0
        while i < ids.count {
            var j = i
            while j < ids.count && visionTokens.contains(ids[j]) { j += 1 }
            if j > i {
                for k in i..<j { out[k] = (base + i, base + j) }
                i = j
            } else {
                i += 1
            }
        }
        return out
    }

    private static func span(_ blocks: [(Int, Int)], at i: Int) -> Int {
        let here = blocks[i]
        return here.1 > here.0 ? here.1 - here.0 : 1
    }

    // A block reads FORWARD across itself and a later chunk's keys are not yet
    // appended, so a block rides ONE chunk WHOLE; step one is unconditional.

    static func chunkLength(_ blocks: [(Int, Int)], at i: Int,
                            want: Int) -> Int {
        let limit = min(want, blocks.count - i)
        var out = span(blocks, at: i)
        while out < limit && out + span(blocks, at: i + out) <= limit {
            out += span(blocks, at: i + out)
        }
        return out
    }

    private static func sourceBidirectional(_ g: GGUF) -> String? {
        var out: String? = nil
        if let raw = g.string("gemma4.source.config_json"),
           let data = raw.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data))
               as? [String: Any],
           let text = root["text_config"] as? [String: Any] {
            out = text["use_bidirectional_attention"] as? String
        }
        return out
    }

    func sharedSource(_ il: Int) -> Int {
        var src = -1
        var i = 0
        while i < firstShared {
            if layerFull[i] == layerFull[il] { src = i }
            i += 1
        }
        return src
    }

    init(_ g: GGUF) throws {
        func i(_ k: String) -> Int { g.int("gemma4." + k)! }
        func f(_ k: String) -> Float { Float(g.double("gemma4." + k)!) }
        nEmbd = i("embedding_length")
        nLayer = i("block_count")
        nHead = i("attention.head_count")
        nHeadKV = i("attention.head_count_kv")
        // Both are absent from the mobile checkpoints, where the full layers
        // share the sliding kv-head count and every layer owns a value.
        nHeadKVFull = g.int("gemma4.attention.global_head_count_kv")
            ?? i("attention.head_count_kv")
        kEqV = g.bool("gemma4.attention.k_eq_v") ?? false
        // NOT READABLE BY UPSTREAM llama.cpp: key_length means the global width
        // there.
        headDimSliding = i("attention.key_length")
        headDimFull = i("attention.global_key_length")
        eps = f("attention.layer_norm_rms_epsilon")
        perLayerDim = i("per_layer_dim")
        slidingWindow = i("attention.sliding_window")
        kvSharedLayers = i("attention.kv_shared_layers")
        logitSoftcap = f("logit_softcap")
        embedScale = f("embed_scale")
        perLayerEmbedScale = f("per_layer_embed_scale")
        ropeBaseSliding = f("rope.freq_base_sliding")
        ropeBaseFull = f("rope.freq_base_full")
        let pf = Double(f("rope.partial_factor_full"))
        rotatedPairsFull = Int(pf * Double(headDimFull) / 2)
        rotatedPairsSliding = headDimSliding / 2
        layerFull = g.ints("gemma4.layer_types")!.map { t in t == 1 }
        activation = try GemmaActivation.read(g, "gemma4.activation")
        // The dedicated key wins; a file emitted before it existed still
        // carries its origin config.json verbatim and answers from there.
        let mode = g.string("gemma4.attention.bidirectional")
            ?? Gemma4Config.sourceBidirectional(g)
        blockwiseVision = mode == "vision"
        visionTokens = Set([g.int("gemma4.image_token_id"),
                            g.int("gemma4.video_token_id")]
            .compactMap { id in id.map { v in Int32(v) } })
        padTokenId = try requireInt(
            g, "tokenizer.ggml.padding_token_id",
            "a soft token has no per-layer row to gather without it")
        nVocab = g.tensor("token_embd.weight").dims[1]
    }
}

// The MLP width is read from the tensor, not the config: layers 0-14 are 6144
// wide and 15-34 12288, where KV sharing starts and the QAT drops to 2 bits.
struct Gemma4Layer {
    let attnNorm: [Float]
    let postAttnNorm: [Float]
    let ffnNorm: [Float]
    let postFfnNorm: [Float]
    let perLayerPostNorm: [Float]
    let layerScalar: Float

    let wq: GGUFTensor
    let wo: GGUFTensor
    let qNorm: [Float]
    // Absent on the shared layers, where HF builds no k/v and the repack drops
    // the dead copies; `wv` is also absent wherever K IS V, per layer.
    let wk: GGUFTensor?
    let wv: GGUFTensor?
    let kNorm: [Float]?
    let nHeadKV: Int

    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let nFF: Int

    let perLayerGate: GGUFTensor?
    let perLayerProj: GGUFTensor?

    init(_ g: GGUF, _ il: Int, _ cfg: Gemma4Config) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        func v(_ s: String) -> [Float] { Dense.floats(t(s)) }
        func m(_ s: String) -> GGUFTensor? { g.maybe("blk.\(il).\(s)") }
        attnNorm = v("attn_norm.weight")
        postAttnNorm = v("post_attn_norm.weight")
        ffnNorm = v("ffn_norm.weight")
        postFfnNorm = v("post_ffn_norm.weight")
        perLayerPostNorm = m("per_layer_post_norm.weight")
            .map(Dense.floats) ?? []
        layerScalar = v("layer_scalar").first ?? 1
        wq = t("attn_q.weight")
        wo = t("attn_output.weight")
        qNorm = v("attn_q_norm.weight")
        let shared = cfg.isShared(il)
        wk = shared ? nil : t("attn_k.weight")
        wv = shared ? nil : m("attn_v.weight")
        kNorm = shared ? nil : v("attn_k_norm.weight")
        nHeadKV = wk.map { w in w.dims[1] / cfg.headDim(il) }
            ?? cfg.headCountKV(il)
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        nFF = ffnGate.dims[1]
        perLayerGate = m("per_layer_gate.weight")
        perLayerProj = m("per_layer_proj.weight")
    }
}

struct Gemma4AssistLayer {
    let attnNorm: GGUFTensor
    let postAttnNorm: GGUFTensor
    let ffnNorm: GGUFTensor
    let postFfnNorm: GGUFTensor
    let qNorm: GGUFTensor
    let layerScalar: Float
    let wq: GGUFTensor
    let wo: GGUFTensor
    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let nFF: Int

    init(_ g: GGUF, _ il: Int) {
        func t(_ s: String) -> GGUFTensor { g.tensor("assist.blk.\(il).\(s)") }
        attnNorm = t("attn_norm.weight")
        postAttnNorm = t("post_attn_norm.weight")
        ffnNorm = t("ffn_norm.weight")
        postFfnNorm = t("post_ffn_norm.weight")
        qNorm = t("attn_q_norm.weight")
        layerScalar = Dense.floats(t("layer_scalar")).first ?? 1
        wq = t("attn_q.weight")
        wo = t("attn_output.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        nFF = ffnGate.dims[1]
    }
}

public struct Gemma4Assist {
    let nLayer: Int
    let nEmbd: Int
    let backbone: Int
    let nHead: Int
    let layerFull: [Bool]
    let layers: [Gemma4AssistLayer]
    let preProj: GGUFTensor
    let postProj: GGUFTensor
    let outputNorm: GGUFTensor
    let output: GGUFTensor
    let centroids: GGUFTensor?
    let tokenOrdering: GGUFTensor?

    var clustered: Bool { centroids != nil && tokenOrdering != nil }

    func isFull(_ il: Int) -> Bool { layerFull[il] }

    static func read(_ g: GGUF, _ trunk: Gemma4Config) -> Gemma4Assist? {
        var out: Gemma4Assist? = nil
        let n = g.int("gemma4.assist.block_count") ?? 0
        if n > 0, g.maybe("assist.pre_proj.weight") != nil {
            out = Gemma4Assist(g, n, trunk)
        }
        return out
    }

    private init(_ g: GGUF, _ n: Int, _ trunk: Gemma4Config) {
        nLayer = n
        nEmbd = g.int("gemma4.assist.embedding_length")!
        backbone = g.int("gemma4.assist.backbone_length")!
        nHead = g.int("gemma4.assist.attention.head_count")!
        layerFull = g.ints("gemma4.assist.layer_types")!.map { t in t == 1 }
        layers = (0..<n).map { il in Gemma4AssistLayer(g, il) }
        preProj = g.tensor("assist.pre_proj.weight")
        postProj = g.tensor("assist.post_proj.weight")
        outputNorm = g.tensor("assist.output_norm.weight")
        output = g.tensor("assist.token_embd.weight")
        centroids = g.maybe("assist.centroids.weight")
        tokenOrdering = g.maybe("assist.token_ordering")
        precondition(backbone == trunk.nEmbd,
                     "assist head fits a \(backbone)-wide trunk, this one "
                     + "is \(trunk.nEmbd)")
        precondition(preProj.dims[0] == 2 * backbone,
                     "assist pre_proj reads \(preProj.dims[0]), expected "
                     + "\(2 * backbone)")
        precondition(g.int("gemma4.assist.attention.head_count_kv")
                     == trunk.nHeadKV
                     && g.int("gemma4.assist.attention.global_head_count_kv")
                     == trunk.nHeadKVFull,
                     "assist kv-head counts must match the trunk's: it "
                     + "reads the trunk's pools")
        precondition(g.int("gemma4.assist.attention.key_length")
                     == trunk.headDimSliding
                     && g.int("gemma4.assist.attention.global_key_length")
                     == trunk.headDimFull,
                     "assist head widths must match the trunk's")
    }
}

public final class Gemma4Model {
    let gguf: GGUF
    public let cfg: Gemma4Config
    let layers: [Gemma4Layer]
    let tokEmbd: GGUFTensor
    let perLayerEmbd: GGUFTensor?
    let perLayerModelProj: GGUFTensor?
    let perLayerProjNorm: [Float]
    let outputNorm: [Float]
    let output: GGUFTensor
    let assist: Gemma4Assist?

    public init(path: String) throws {
        gguf = try GGUF(path: path)
        cfg = try Gemma4Config(gguf)
        tokEmbd = gguf.tensor("token_embd.weight")
        perLayerEmbd = gguf.maybe("per_layer_token_embd.weight")
        perLayerModelProj = gguf.maybe("per_layer_model_proj.weight")
        perLayerProjNorm = gguf.maybe("per_layer_proj_norm.weight")
            .map(Dense.floats) ?? []
        outputNorm = Dense.floats(gguf.tensor("output_norm.weight"))
        output = gguf.maybe("output.weight") ?? tokEmbd
        let g = gguf, c = cfg
        layers = (0..<c.nLayer).map { il in Gemma4Layer(g, il, c) }
        assist = Gemma4Assist.read(g, c)
    }

    // A UNIFIED checkpoint records zero and means it: its images and audio go
    // through a projection with no encoder behind it.
    public var hasVisionTower: Bool {
        (gguf.int("gemma4.vision.block_count") ?? 0) > 0
    }
    public var hasAudioTower: Bool {
        (gguf.int("gemma4.audio.block_count") ?? 0) > 0
    }

    public static func isGemma4(path: String) -> Bool {
        var result = false
        if let g = try? GGUF(path: path) {
            result = g.string("general.architecture") == "gemma4"
        }
        return result
    }
}

public struct GemmaChat {
    public let model: Gemma4Model
    public let engine: Gemma4Engine
    let tokenizer: GemmaTokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets

    public init(ggufPath: String) throws {
        model = try Gemma4Model(path: ggufPath)
        engine = Gemma4Engine(model)
        tokenizer = try GemmaTokenizer(gguf: model.gguf)
        chatTemplate = model.gguf.string("tokenizer.chat_template") ?? ""
        samplingPresets = SamplingPresets.require(gguf: model.gguf,
                                                  path: ggufPath)
    }

    // eosIds is the SET: gemma stops on three ids.
    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: true)
    }
    public func decode(_ ids: [Int32]) -> String { tokenizer.decode(ids) }

    public func encodeRaw(_ text: String) -> [Int32] {
        tokenizer.encode(text, addSpecial: false)
    }
    public var eosIds: Set<Int32> { tokenizer.eosIds }
    public var vocabCount: Int { tokenizer.vocabCount }
    public var shape: ModelShape { ModelShape(gguf: model.gguf) }
    public var bosToken: String { tokenizer.bosToken }

    public func weightWarm(drafting: Bool) -> WeightWarm {
        WeightWarm(model.gguf, drafting: drafting)
    }

    public func backend() -> Gemma4Backend {
        Gemma4Backend(engine: engine, tokenizer: tokenizer)
    }

    public func audioWire() throws -> Gemma4AudioWire {
        try Gemma4AudioWire(model.gguf)
    }

    public func visionWire() throws -> Gemma4VisionWire {
        try Gemma4VisionWire(model.gguf)
    }

    public func videoWire() throws -> Gemma4VideoWire {
        try Gemma4VideoWire(model.gguf)
    }

    public var melConfig: Gemma4MelConfig {
        Gemma4MelConfig(model.gguf) ?? .processorDefault
    }

    public func media(ctx: MetalContext? = nil) -> Gemma4Media {
        Gemma4Media(self, ctx: ctx)
    }

    public func metalBackend() throws -> Gemma4MetalBackend {
        Gemma4MetalBackend(engine: try Gemma4MetalEngine(model),
                           tokenizer: tokenizer)
    }
}
