import Accelerate
import Foundation

// Weights are dequantized per layer inside the forward rather than held:
// 672 MB as f32 for the whole tower, about 40 MB one layer at a time.

public struct Gemma4VisionConfig {
    public let embd: Int
    public let layers: Int
    public let patchSize: Int
    public let poolKernel: Int
    public let maxSoftTokens: Int
    let headDim: Int
    let heads: Int
    let ff: Int
    let ropeTheta: Float
    let eps: Float
    let posTable: Int
    let activation: GemmaActivation

    public var patchDim: Int { patchSize * patchSize * 3 }

    init(_ g: GGUF) throws {
        func i(_ k: String) -> Int { g.int("gemma4.vision." + k)! }
        embd = i("embedding_length")
        layers = i("block_count")
        patchSize = i("patch_size")
        poolKernel = i("pool_kernel")
        maxSoftTokens = i("max_soft_tokens")
        ropeTheta = Float(g.double("gemma4.vision.rope_theta")!)
        eps = Float(g.double("gemma4.attention.layer_norm_rms_epsilon")!)
        headDim = g.tensor("v.blk.0.attn_q_norm.weight").dims[0]
        heads = g.tensor("v.blk.0.attn_q.weight").dims[1] / headDim
        ff = g.tensor("v.blk.0.ffn_gate.weight").dims[1]
        posTable = g.tensor("v.position_embd.weight").dims[1]
        activation = try GemmaActivation.read(g, "gemma4.vision.activation")
    }
}

// N is the tower's ACTUAL count for this image, following the aspect ratio.

public struct Gemma4VisionWire: Sendable {
    public let boi: Int32
    public let eoi: Int32
    public let token: Int32

    init(_ g: GGUF) throws {
        boi = Int32(try requireInt(g, "gemma4.boi_token_id",
                                   "an image turn has no opening marker"))
        eoi = Int32(try requireInt(g, "gemma4.eoi_token_id",
                                   "an image turn has no closing marker"))
        token = Int32(try requireInt(g, "gemma4.image_token_id",
                                     "an image has no placeholder to expand"))
    }
}

public struct Gemma4VideoWire: Sendable {
    public let token: Int32
    public let frames: Int
    public let softTokensPerFrame: Int

    init(_ g: GGUF) throws {
        token = Int32(try requireInt(g, "gemma4.video_token_id",
                                     "a video has no placeholder to expand"))
        frames = try requireInt(g, "gemma4.video.num_frames",
                                "the frame count is unknown")
        softTokensPerFrame = try requireInt(
            g, "gemma4.video.max_soft_tokens",
            "the per-frame soft-token budget is unknown")
    }
}

public final class Gemma4ViT {
    public let cfg: Gemma4VisionConfig
    private let gguf: GGUF
    private let posEmbd: [Float]

    private let srq: [String: SRQ]

    public init(_ model: Gemma4Model) throws {
        gguf = model.gguf
        cfg = try Gemma4VisionConfig(model.gguf)
        posEmbd = Dense.floats(model.gguf.tensor("v.position_embd.weight"))
        var scales: [String: SRQ] = [:]
        for il in 0..<cfg.layers {
            for tag in ["attn_q", "attn_k", "attn_v", "attn_out",
                        "ffn_gate", "ffn_up", "ffn_down"] {
                let name = "v.blk.\(il).\(tag).weight"
                scales[name] = SRQ(model.gguf, name)
            }
        }
        srq = scales
    }

    // Every quantized projection routes through here so no site skips the
    // clamp.
    private func linear(_ name: String, _ x: [Float], _ n: Int,
                        _ inDim: Int, _ outDim: Int) -> [Float] {
        let s = srq[name] ?? SRQ.none
        var h = x
        SRQ.apply(&h, s.input)
        var out = [Float](repeating: 0, count: n * outDim)
        vDSP_mmul(h, 1, matrix(name, inDim, outDim), 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(outDim), vDSP_Length(inDim))
        SRQ.apply(&out, s.output)
        return out
    }

    public func forward(pixels: [Float], pos: [(Int, Int)])
        -> (tower: [Float], proj: [Float], count: Int) {
        let n = pos.count
        let e = cfg.embd
        var x = patchEmbed(pixels, pos, n)
        let padded = pos.map { p in p.0 < 0 && p.1 < 0 }
        let tables = ropeTables(pos)
        for il in 0..<cfg.layers { block(il, &x, n, padded, tables) }
        let pooled = pool(x, pos, padded, n)
        var tower = pooled.rows
        let scale = Float(e).squareRoot()
        for i in 0..<tower.count { tower[i] *= scale }
        return (tower, project(tower, pooled.count), pooled.count)
    }

    // 2 * (p - 0.5) happens in model code, not the processor, so this is the
    // only place; padding patches keep the projection, lose only position.
    func patchEmbed(_ pixels: [Float], _ pos: [(Int, Int)],
                            _ n: Int) -> [Float] {
        let e = cfg.embd, pd = cfg.patchDim
        var scaled = [Float](repeating: 0, count: n * pd)
        for i in 0..<(n * pd) { scaled[i] = 2 * (pixels[i] - 0.5) }
        var out = [Float](repeating: 0, count: n * e)
        let w = matrix("v.patch_embd.weight", pd, e)
        vDSP_mmul(scaled, 1, w, 1, &out, 1,
                  vDSP_Length(n), vDSP_Length(e), vDSP_Length(pd))
        for s in 0..<n {
            if pos[s].0 >= 0 || pos[s].1 >= 0 {
                let xi = max(pos[s].0, 0), yi = max(pos[s].1, 0)
                let xb = xi * e
                let yb = (cfg.posTable + yi) * e
                for c in 0..<e {
                    out[s * e + c] += posEmbd[xb + c] + posEmbd[yb + c]
                }
            }
        }
        return out
    }

    // 64 wide: the first 32 channels rotate by the patch's x and the second 32
    // by its y, each half pairing (j, j + 16). Every layer shares them.
    func ropeTables(_ pos: [(Int, Int)])
        -> (cos: [Float], sin: [Float]) {
        let d = cfg.headDim
        let per = d / 2
        let half = per / 2
        var c = [Float](repeating: 0, count: pos.count * d)
        var s = [Float](repeating: 0, count: pos.count * d)
        for p in 0..<pos.count {
            for axis in 0..<2 {
                let coord = Float(max(axis == 0 ? pos[p].0 : pos[p].1, 0))
                for j in 0..<half {
                    let f = powf(cfg.ropeTheta,
                                 -2 * Float(j) / Float(per))
                    let a = coord * f
                    let base = p * d + axis * per + j
                    c[base] = cosf(a)
                    s[base] = sinf(a)
                    c[base + half] = cosf(a)
                    s[base + half] = sinf(a)
                }
            }
        }
        return (c, s)
    }

    private func block(_ il: Int, _ x: inout [Float], _ n: Int,
                       _ padded: [Bool],
                       _ rope: (cos: [Float], sin: [Float])) {
        let e = cfg.embd
        func w(_ s: String) -> [Float] {
            Dense.floats(gguf.tensor("v.blk.\(il).\(s)"))
        }
        var h = x
        rmsRows(&h, e, n, w("ln1.weight"))
        var attn = attention(il, h, n, padded, rope)
        rmsRows(&attn, e, n, w("post_attn_norm.weight"))
        vDSP_vadd(x, 1, attn, 1, &x, 1, vDSP_Length(n * e))

        h = x
        rmsRows(&h, e, n, w("ln2.weight"))
        var ff = mlp(il, h, n)
        rmsRows(&ff, e, n, w("post_ffn_norm.weight"))
        vDSP_vadd(x, 1, ff, 1, &x, 1, vDSP_Length(n * e))
    }

    private func mlp(_ il: Int, _ h: [Float], _ n: Int) -> [Float] {
        let e = cfg.embd, f = cfg.ff
        var gate = linear("v.blk.\(il).ffn_gate.weight", h, n, e, f)
        let up = linear("v.blk.\(il).ffn_up.weight", h, n, e, f)
        let act = cfg.activation
        for i in 0..<(n * f) { gate[i] = act.apply(gate[i]) * up[i] }
        return linear("v.blk.\(il).ffn_down.weight", gate, n, f, e)
    }

    // Bidirectional, scale 1.0, and padding patches are masked OUT of every row
    // rather than merely contributing zero.
    private func attention(_ il: Int, _ h: [Float], _ n: Int,
                           _ padded: [Bool],
                           _ rope: (cos: [Float], sin: [Float])) -> [Float] {
        let e = cfg.embd, d = cfg.headDim, nH = cfg.heads
        var q = linear("v.blk.\(il).attn_q.weight", h, n, e, nH * d)
        var k = linear("v.blk.\(il).attn_k.weight", h, n, e, nH * d)
        var v = linear("v.blk.\(il).attn_v.weight", h, n, e, nH * d)
        let qn = Dense.floats(gguf.tensor("v.blk.\(il).attn_q_norm.weight"))
        let kn = Dense.floats(gguf.tensor("v.blk.\(il).attn_k_norm.weight"))
        GK.rmsnormRows(&q, d: d, rows: n * nH, w: qn, eps: cfg.eps)
        GK.rmsnormRows(&k, d: d, rows: n * nH, w: kn, eps: cfg.eps)
        // v_norm is scale-free and leaves no tensor, exactly as in the text
        // tower.
        GK.rmsnormRowsNoWeight(&v, d: d, rows: n * nH, eps: cfg.eps)
        applyRope(&q, n, rope)
        applyRope(&k, n, rope)

        var ctx = [Float](repeating: 0, count: n * e)
        // Heads write disjoint slices and read only q/k/v, so the fill is
        // race-free; serial this is minutes on a 2520-patch image.
        ctx.withUnsafeMutableBufferPointer { cb in
            q.withUnsafeBufferPointer { qb in
                k.withUnsafeBufferPointer { kb in
                    v.withUnsafeBufferPointer { vb in
                        nonisolated(unsafe) let cp = cb.baseAddress!
                        nonisolated(unsafe) let qp = qb.baseAddress!
                        nonisolated(unsafe) let kp = kb.baseAddress!
                        nonisolated(unsafe) let vp = vb.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nH) {
                            head in
                            var scores = [Float](repeating: 0, count: n)
                            for i in 0..<n {
                                for j in 0..<n {
                                    var dot: Float = 0
                                    let qo = (i * nH + head) * d
                                    let ko = (j * nH + head) * d
                                    for c in 0..<d {
                                        dot += qp[qo + c] * kp[ko + c]
                                    }
                                    scores[j] = padded[j]
                                        ? -Float.greatestFiniteMagnitude : dot
                                }
                                GK.softmaxInPlace(&scores, n)
                                let oo = i * e + head * d
                                for j in 0..<n where scores[j] != 0 {
                                    let vo = (j * nH + head) * d
                                    let s = scores[j]
                                    for c in 0..<d {
                                        cp[oo + c] += s * vp[vo + c]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return linear("v.blk.\(il).attn_out.weight", ctx, n, nH * d, e)
    }

    private func applyRope(_ t: inout [Float], _ n: Int,
                           _ rope: (cos: [Float], sin: [Float])) {
        let d = cfg.headDim, nH = cfg.heads
        let per = d / 2, half = per / 2
        for p in 0..<n {
            for head in 0..<nH {
                let b = (p * nH + head) * d
                for axis in 0..<2 {
                    let o = axis * per
                    for j in 0..<half {
                        let c = rope.cos[p * d + o + j]
                        let s = rope.sin[p * d + o + j]
                        let a = t[b + o + j]
                        let bb = t[b + o + j + half]
                        t[b + o + j] = a * c - bb * s
                        t[b + o + j + half] = a * s + bb * c
                    }
                }
            }
        }
    }

    // The divisor is the fixed k^2, not the patches that landed, so a padding
    // patch contributes zero without diluting; an unreached slot is dropped.
    func pool(_ x: [Float], _ pos: [(Int, Int)], _ padded: [Bool],
                      _ n: Int) -> (rows: [Float], count: Int) {
        let e = cfg.embd
        let k = cfg.poolKernel
        let slots = n / (k * k)
        var maxX = 0
        for p in pos { maxX = max(maxX, max(p.0, 0)) }
        let cols = (maxX + 1) / k
        var acc = [Float](repeating: 0, count: slots * e)
        var used = [Bool](repeating: false, count: slots)
        let inv = 1 / Float(k * k)
        for s in 0..<n {
            let xi = max(pos[s].0, 0) / k
            let yi = max(pos[s].1, 0) / k
            let slot = xi + cols * yi
            if slot < slots {
                used[slot] = true
                if !padded[s] {
                    for c in 0..<e { acc[slot * e + c] += x[s * e + c] * inv }
                }
            }
        }
        var rows: [Float] = []
        rows.reserveCapacity(slots * e)
        var count = 0
        for slot in 0..<slots where used[slot] {
            rows.append(contentsOf: acc[(slot * e)..<((slot + 1) * e)])
            count += 1
        }
        return (rows, count)
    }

    // A SCALE-FREE norm then a plain projection tames the tower's four-orders
    // range.
    func project(_ tower: [Float], _ rows: Int) -> [Float] {
        let e = cfg.embd
        let w = gguf.tensor("mm.vision.weight")
        let outDim = w.dims[1]
        var h = tower
        GK.rmsnormRowsNoWeight(&h, d: e, rows: rows, eps: cfg.eps)
        var out = [Float](repeating: 0, count: rows * outDim)
        vDSP_mmul(h, 1, matrix("mm.vision.weight", e, outDim), 1, &out, 1,
                  vDSP_Length(rows), vDSP_Length(outDim), vDSP_Length(e))
        return out
    }

    private func matrix(_ name: String, _ inDim: Int,
                        _ outDim: Int) -> [Float] {
        Dense.transposedFloats(gguf.tensor(name), inDim, outDim)
    }

    private func rmsRows(_ x: inout [Float], _ d: Int, _ rows: Int,
                         _ w: [Float]) {
        GK.rmsnormRows(&x, d: d, rows: rows, w: w, eps: cfg.eps)
    }
}

extension Gemma4ViT {
    // Shared with the GPU tower so the drop-empty-slot rule and the (x, y)
    // lookup have ONE definition rather than two that can drift.
    func embedForGPU(pixels: [Float], pos: [(Int, Int)]) -> [Float] {
        patchEmbed(pixels, pos, pos.count)
    }

    func ropeForGPU(pos: [(Int, Int)]) -> (cos: [Float], sin: [Float]) {
        ropeTables(pos)
    }

    func poolAndProject(_ x: [Float], pos: [(Int, Int)], padded: [Bool])
        -> (tower: [Float], proj: [Float], count: Int) {
        let pooled = pool(x, pos, padded, pos.count)
        var tower = pooled.rows
        let scale = Float(cfg.embd).squareRoot()
        for i in 0..<tower.count { tower[i] *= scale }
        return (tower, project(tower, pooled.count), pooled.count)
    }
}
