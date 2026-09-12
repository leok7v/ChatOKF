// Read straight from the GGUF metadata; names and shapes follow llama.cpp's
// qwen35 and qwen3 loaders. Nothing is keyed by model name.
import Foundation

public struct QwenConfig {
    public let nEmbd: Int
    let nFF: Int
    public let nLayer: Int
    let nHead: Int
    let nHeadKV: Int
    let headDim: Int
    let nRot: Int
    let ropeBase: Float
    let ropeSections: [Int]
    let eps: Float
    public let nVocab: Int
    let fullAttnInterval: Int
    // dense: no state-space layers, no output gate, no fused q|gate projection.
    let dense: Bool

    let dState: Int
    let nKHead: Int
    let nVHead: Int
    let dInner: Int
    let dConv: Int

    var keyDim: Int { dState * nKHead }
    var valueDim: Int { dState * nVHead }
    var convDim: Int { keyDim * 2 + valueDim }
    func isRecurrent(_ il: Int) -> Bool { (il + 1) % fullAttnInterval != 0 }

    init(_ g: GGUF) {
        let arch = g.string("general.architecture") ?? "qwen35"
        dense = arch == "qwen3"
        func i(_ k: String) -> Int { g.int("\(arch)." + k)! }
        nEmbd = i("embedding_length")
        nFF = i("feed_forward_length")
        let declared = i("block_count")
        let nextn = g.int("\(arch).nextn_predict_layers") ?? 0
        let last = declared - nextn
        nLayer = nextn > 0
            && g.maybe("blk.\(last).nextn.eh_proj.weight") != nil
            ? last : declared
        nHead = i("attention.head_count")
        nHeadKV = i("attention.head_count_kv")
        headDim = i("attention.key_length")
        // Dense qwen3 ships no rope.dimension_count: it rotates the whole head.
        nRot = g.int("\(arch).rope.dimension_count") ?? headDim
        ropeBase = Float(g.double("\(arch).rope.freq_base")!)
        ropeSections = g.ints("\(arch).rope.dimension_sections") ?? [11, 11, 10, 0]
        eps = Float(g.double("\(arch).attention.layer_norm_rms_epsilon")!)
        nVocab = g.tensor("token_embd.weight").dims[1]
        if dense {
            // Interval 1 makes every layer attention.
            fullAttnInterval = 1
            dState = 0; nKHead = 0; nVHead = 0; dInner = 0; dConv = 0
        } else {
            fullAttnInterval = g.int("qwen35.full_attention_interval") ?? 4
            dState = i("ssm.state_size")
            nKHead = i("ssm.group_count")
            nVHead = i("ssm.time_step_rank")
            dInner = i("ssm.inner_size")
            dConv = i("ssm.conv_kernel")
        }
    }
}

struct QwenLayer {
    let attnNorm: GGUFTensor
    let attnPostNorm: GGUFTensor
    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor
    let recurrent: Bool

    var wqkv: GGUFTensor?
    var wqkvGate: GGUFTensor?
    var ssmConv1d: GGUFTensor?
    var ssmDt: GGUFTensor?
    var ssmA: GGUFTensor?
    var ssmBeta: GGUFTensor?
    var ssmAlpha: GGUFTensor?
    var ssmNorm: GGUFTensor?
    var ssmOut: GGUFTensor?

    var wq: GGUFTensor?
    var wk: GGUFTensor?
    var wv: GGUFTensor?
    var wo: GGUFTensor?
    var qNorm: GGUFTensor?
    var kNorm: GGUFTensor?

    init(_ g: GGUF, _ il: Int, _ cfg: QwenConfig) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        attnNorm = t("attn_norm.weight")
        // Dense qwen3 names the pre-FFN norm ffn_norm: same slot, other name.
        attnPostNorm = cfg.dense ? t("ffn_norm.weight")
                                 : t("post_attention_norm.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
        recurrent = cfg.isRecurrent(il)
        if recurrent {
            wqkv = t("attn_qkv.weight")
            wqkvGate = t("attn_gate.weight")
            ssmConv1d = t("ssm_conv1d.weight")
            ssmDt = t("ssm_dt.bias")
            ssmA = t("ssm_a")
            ssmBeta = t("ssm_beta.weight")
            ssmAlpha = t("ssm_alpha.weight")
            ssmNorm = t("ssm_norm.weight")
            ssmOut = t("ssm_out.weight")
        } else {
            wq = t("attn_q.weight")
            wk = t("attn_k.weight")
            wv = t("attn_v.weight")
            wo = t("attn_output.weight")
            qNorm = t("attn_q_norm.weight")
            kNorm = t("attn_k_norm.weight")
        }
    }
}

struct QwenMTP {
    let ehProj: GGUFTensor
    let enorm: GGUFTensor
    let hnorm: GGUFTensor
    let headNorm: GGUFTensor
    let attnNorm: GGUFTensor
    let attnPostNorm: GGUFTensor
    let wq: GGUFTensor
    let wk: GGUFTensor
    let wv: GGUFTensor
    let wo: GGUFTensor
    let qNorm: GGUFTensor
    let kNorm: GGUFTensor
    let ffnGate: GGUFTensor
    let ffnUp: GGUFTensor
    let ffnDown: GGUFTensor

    init(_ g: GGUF, _ il: Int) {
        func t(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
        ehProj = t("nextn.eh_proj.weight")
        enorm = t("nextn.enorm.weight")
        hnorm = t("nextn.hnorm.weight")
        headNorm = t("nextn.shared_head_norm.weight")
        attnNorm = t("attn_norm.weight")
        attnPostNorm = t("post_attention_norm.weight")
        wq = t("attn_q.weight")
        wk = t("attn_k.weight")
        wv = t("attn_v.weight")
        wo = t("attn_output.weight")
        qNorm = t("attn_q_norm.weight")
        kNorm = t("attn_k_norm.weight")
        ffnGate = t("ffn_gate.weight")
        ffnUp = t("ffn_up.weight")
        ffnDown = t("ffn_down.weight")
    }
}

public final class QwenModel {
    let gguf: GGUF
    public let cfg: QwenConfig
    let layers: [QwenLayer]
    let tokEmbd: GGUFTensor
    let outputNorm: GGUFTensor
    let output: GGUFTensor
    let mtp: QwenMTP?

    public init(path: String) throws {
        gguf = try GGUF(path: path)
        cfg = QwenConfig(gguf)
        tokEmbd = gguf.tensor("token_embd.weight")
        outputNorm = gguf.tensor("output_norm.weight")
        output = gguf.maybe("output.weight") ?? tokEmbd
        let g = gguf, c = cfg
        layers = (0..<c.nLayer).map { QwenLayer(g, $0, c) }
        mtp = g.maybe("blk.\(c.nLayer).nextn.eh_proj.weight") != nil
            ? QwenMTP(g, c.nLayer) : nil
    }
}
