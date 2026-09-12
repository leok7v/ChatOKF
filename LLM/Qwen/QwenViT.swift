import Accelerate
import Foundation

// Mirrors llama.cpp clip_graph_qwen3vl; gated byte for byte against the
// numpy reference by `chatokf-cli --vit`. No deepstack layers here.

public struct QwenViTConfig {
    public let imageSize: Int
    public let patchSize: Int
    public let merge: Int
    public let embd: Int
    public let ff: Int
    public let layers: Int
    public let heads: Int
    public let eps: Float
    public let mean: [Float]
    public let std: [Float]
    public let projDim: Int

    public var side: Int { imageSize / patchSize }
    public var patches: Int { side * side }
    public var mergedTokens: Int { patches / (merge * merge) }
    public var headDim: Int { embd / heads }
    public var patchDim: Int { patchSize * patchSize * 3 }

    // Force-read: a missing key means a broken mmproj. merge falls back to a
    // SHAPE derivation, so even that is the file's own geometry.
    init(_ g: GGUF) {
        func i(_ k: String) -> Int { g.int("clip.vision." + k)! }
        imageSize = i("image_size")
        patchSize = i("patch_size")
        embd = i("embedding_length")
        ff = i("feed_forward_length")
        layers = i("block_count")
        heads = i("attention.head_count")
        eps = Float(g.double("clip.vision.attention.layer_norm_epsilon")!)
        mean = g.doubles("clip.vision.image_mean")!.map { Float($0) }
        std = g.doubles("clip.vision.image_std")!.map { Float($0) }
        let mm0In = g.tensor("mm.0.weight").dims[0]
        let m2 = g.int("clip.vision.spatial_merge_size") ?? mm0In / i("embedding_length")
        merge = g.int("clip.vision.spatial_merge_size") != nil
            ? m2 : Int(Double(m2).squareRoot().rounded())
        projDim = g.tensor("mm.2.weight").dims[1]
    }

    // Header-only probe: no weight materialization, so grid sizing is cheap.
    public init(mmprojPath: String) throws {
        self.init(try GGUF(path: mmprojPath))
    }
}

// Weight matrices are stored TRANSPOSED at load ([in][out]) so every
// product is a plain row-major vDSP_mmul, no per-call transpose.
private struct ViTLayer {
    let ln1w: [Float], ln1b: [Float]
    let qkvW: [Float], qkvB: [Float]
    let outW: [Float], outB: [Float]
    let ln2w: [Float], ln2b: [Float]
    let upW: [Float], upB: [Float]
    let downW: [Float], downB: [Float]

    init(_ g: GGUF, _ il: Int, _ e: Int, _ ff: Int) {
        func t(_ s: String) -> [Float] {
            Dense.floats(g.tensor("v.blk.\(il).\(s)"))
        }
        ln1w = t("ln1.weight"); ln1b = t("ln1.bias")
        qkvW = QwenViT.transposed(t("attn_qkv.weight"), 3 * e, e)
        qkvB = t("attn_qkv.bias")
        outW = QwenViT.transposed(t("attn_out.weight"), e, e)
        outB = t("attn_out.bias")
        ln2w = t("ln2.weight"); ln2b = t("ln2.bias")
        upW = QwenViT.transposed(t("ffn_up.weight"), ff, e)
        upB = t("ffn_up.bias")
        downW = QwenViT.transposed(t("ffn_down.weight"), e, ff)
        downB = t("ffn_down.bias")
    }
}

public final class QwenViT {
    public let cfg: QwenViTConfig
    private let patchW: [Float]
    private let patchWPair: [Float]
    private let patchB: [Float]
    private let posEmbd: [Float]
    private let postW: [Float], postB: [Float]
    private let mm0W: [Float], mm0B: [Float]
    private let mm2W: [Float], mm2B: [Float]
    private let blocks: [ViTLayer]
    private var tables: Tables

    struct Tables {
        let h: Int
        let w: Int
        let order: [Int]
        let cos: [Float]
        let sin: [Float]
        let pos: [Float]
    }

    public init(path: String) throws {
        let g = try GGUF(path: path)
        let c = QwenViTConfig(g)
        cfg = c
        let w0 = Dense.floats(g.tensor("v.patch_embd.weight"))
        let w1 = Dense.floats(g.tensor("v.patch_embd.weight.1"))
        var sum = w0
        vDSP_vadd(w0, 1, w1, 1, &sum, 1, vDSP_Length(w0.count))
        let pd = c.patchSize * c.patchSize * 3
        patchW = QwenViT.transposed(sum, c.embd, pd)
        patchWPair = QwenViT.transposed(QwenViT.pairKernel(w0, w1, embd: c.embd, pd: pd),
                                    c.embd, 2 * pd)
        patchB = Dense.floats(g.tensor("v.patch_embd.bias"))
        let pe = Dense.floats(g.tensor("v.position_embd.weight"))
        posEmbd = pe
        postW = Dense.floats(g.tensor("v.post_ln.weight"))
        postB = Dense.floats(g.tensor("v.post_ln.bias"))
        let mmIn = c.embd * c.merge * c.merge
        mm0W = QwenViT.transposed(Dense.floats(g.tensor("mm.0.weight")),
                              mmIn, mmIn)
        mm0B = Dense.floats(g.tensor("mm.0.bias"))
        mm2W = QwenViT.transposed(Dense.floats(g.tensor("mm.2.weight")),
                              c.projDim, mmIn)
        mm2B = Dense.floats(g.tensor("mm.2.bias"))
        let e = c.embd, f = c.ff
        blocks = (0..<c.layers).map { il in ViTLayer(g, il, e, f) }
        tables = QwenViT.tables(c, pe, h: c.side, w: c.side)
    }


    static func mergeOrderTable(_ cfg: QwenViTConfig) -> [Int] {
        mergeOrderTable(h: cfg.side, w: cfg.side, merge: cfg.merge)
    }

    // result[s] is the raster patch index at sequence slot s.
    static func mergeOrderTable(h: Int, w: Int, merge m: Int) -> [Int] {
        var order: [Int] = []
        order.reserveCapacity(h * w)
        for wi in 0..<(h / m) {
            for wj in 0..<(w / m) {
                for dy in 0..<m {
                    for dx in 0..<m {
                        order.append((m * wi + dy) * w + (m * wj + dx))
                    }
                }
            }
        }
        return order
    }

    static func ropeTables(_ cfg: QwenViTConfig, _ order: [Int])
        -> (cos: [Float], sin: [Float]) {
        ropeTables(cfg, order, w: cfg.side)
    }

    // M-RoPE: headDim/2 angle pairs, the first half keyed by the patch row,
    // the second by the column; dims (d, d+headDim/2) rotate together.
    static func ropeTables(_ cfg: QwenViTConfig, _ order: [Int], w: Int)
        -> (cos: [Float], sin: [Float]) {
        let half = cfg.headDim / 2
        let quarter = half / 2
        let n = order.count
        var cosT = [Float](repeating: 0, count: n * half)
        var sinT = [Float](repeating: 0, count: n * half)
        for s in 0..<n {
            let y = Float(order[s] / w)
            let x = Float(order[s] % w)
            for j in 0..<half {
                let pos = j < quarter ? y : x
                let f = pow(10000.0, -Float(j % quarter) / Float(quarter))
                cosT[s * half + j] = cos(pos * f)
                sinT[s * half + j] = sin(pos * f)
            }
        }
        return (cosT, sinT)
    }

    static func patchRows(_ px: [Float], _ cfg: QwenViTConfig,
                          _ order: [Int]) -> [Float] {
        patchRows(px, cfg, order, w: cfg.side)
    }

    // ggml's im2col layout: channel-major within a patch.
    static func patchRows(_ px: [Float], _ cfg: QwenViTConfig,
                          _ order: [Int], w: Int) -> [Float] {
        let n = order.count, pd = cfg.patchDim
        let p = cfg.patchSize, img = w * p
        var rows = [Float](repeating: 0, count: n * pd)
        for s in 0..<n {
            let py = (order[s] / w) * p
            let pxx = (order[s] % w) * p
            for c in 0..<3 {
                for ky in 0..<p {
                    for kx in 0..<p {
                        rows[s * pd + c * p * p + ky * p + kx] =
                            px[((py + ky) * img + pxx + kx) * 3 + c]
                    }
                }
            }
        }
        return rows
    }

    static func pairKernel(_ w0: [Float], _ w1: [Float], embd: Int,
                           pd: Int) -> [Float] {
        var out = [Float](repeating: 0, count: embd * 2 * pd)
        for o in 0..<embd {
            for k in 0..<pd {
                out[o * 2 * pd + k] = w0[o * pd + k]
                out[o * 2 * pd + pd + k] = w1[o * pd + k]
            }
        }
        return out
    }

    static func pairRows(_ a: [Float], _ b: [Float], _ cfg: QwenViTConfig,
                         _ order: [Int], w: Int) -> [Float] {
        let ra = patchRows(a, cfg, order, w: w)
        let rb = patchRows(b, cfg, order, w: w)
        let n = order.count, pd = cfg.patchDim
        var rows = [Float](repeating: 0, count: n * 2 * pd)
        for s in 0..<n {
            for k in 0..<pd {
                rows[s * 2 * pd + k] = ra[s * pd + k]
                rows[s * 2 * pd + pd + k] = rb[s * pd + k]
            }
        }
        return rows
    }

    static func positionTable(_ posEmbd: [Float], embd e: Int,
                              h: Int, w: Int) -> [Float] {
        let n = Int(Double(posEmbd.count / e).squareRoot().rounded())
        var out = [Float](repeating: 0, count: h * w * e)
        for y in 0..<h {
            let sy = h > 1 ? Double(y) * Double(n - 1) / Double(h - 1) : 0
            let y0 = min(Int(sy), n - 1)
            let y1 = min(y0 + 1, n - 1)
            let fy = Float(sy - Double(y0))
            for x in 0..<w {
                let sx = w > 1 ? Double(x) * Double(n - 1) / Double(w - 1) : 0
                let x0 = min(Int(sx), n - 1)
                let x1 = min(x0 + 1, n - 1)
                let fx = Float(sx - Double(x0))
                let base = (y * w + x) * e
                for c in 0..<e {
                    let top = posEmbd[(y0 * n + x0) * e + c] * (1 - fx)
                        + posEmbd[(y0 * n + x1) * e + c] * fx
                    let bottom = posEmbd[(y1 * n + x0) * e + c] * (1 - fx)
                        + posEmbd[(y1 * n + x1) * e + c] * fx
                    out[base + c] = top * (1 - fy) + bottom * fy
                }
            }
        }
        return out
    }

    private static func tables(_ cfg: QwenViTConfig, _ posEmbd: [Float],
                               h: Int, w: Int) -> Tables {
        let order = mergeOrderTable(h: h, w: w, merge: cfg.merge)
        let rope = ropeTables(cfg, order, w: w)
        return Tables(h: h, w: w, order: order, cos: rope.cos, sin: rope.sin,
                      pos: positionTable(posEmbd, embd: cfg.embd, h: h, w: w))
    }

    private func prepare(h: Int, w: Int) -> Tables {
        if tables.h != h || tables.w != w {
            tables = QwenViT.tables(cfg, posEmbd, h: h, w: w)
        }
        return tables
    }

    // pixels are HWC, already (v - mean) / std.
    public func forward(pixels: [Float]) -> [Float] {
        forward(pixels: pixels, gridH: cfg.side, gridW: cfg.side)
    }

    public func forward(pixels: [Float], gridH: Int, gridW: Int) -> [Float] {
        let t = prepare(h: gridH, w: gridW)
        return run(QwenViT.patchRows(pixels, cfg, t.order, w: t.w),
                   cfg.patchDim, patchW, t)
    }

    public func forward(pair a: [Float], _ b: [Float], gridH: Int,
                        gridW: Int) -> [Float] {
        let t = prepare(h: gridH, w: gridW)
        return run(QwenViT.pairRows(a, b, cfg, t.order, w: t.w),
                   2 * cfg.patchDim, patchWPair, t)
    }

    private func run(_ rows: [Float], _ k: Int, _ weight: [Float],
                     _ t: Tables) -> [Float] {
        var x = patchEmbed(rows, k, weight, t)
        for l in blocks { block(l, &x, t) }
        layerNorm(&x, postW, postB)
        return merger(x)
    }

    private func patchEmbed(_ rows: [Float], _ k: Int, _ weight: [Float],
                            _ t: Tables) -> [Float] {
        let n = t.order.count, e = cfg.embd
        var out = [Float](repeating: 0, count: n * e)
        matmul(rows, weight, n, e, k, &out)
        for s in 0..<n {
            let raster = t.order[s]
            for c in 0..<e {
                out[s * e + c] += patchB[c] + t.pos[raster * e + c]
            }
        }
        return out
    }

    private func block(_ l: ViTLayer, _ x: inout [Float], _ t: Tables) {
        let e = cfg.embd
        let n = x.count / e
        var h = x
        layerNorm(&h, l.ln1w, l.ln1b)
        var qkv = [Float](repeating: 0, count: n * 3 * e)
        matmul(h, l.qkvW, n, 3 * e, e, &qkv)
        addBias(&qkv, l.qkvB, n)
        let ctx = attention(qkv, t)
        var attnOut = [Float](repeating: 0, count: n * e)
        matmul(ctx, l.outW, n, e, e, &attnOut)
        addBias(&attnOut, l.outB, n)
        vDSP_vadd(x, 1, attnOut, 1, &x, 1, vDSP_Length(n * e))
        h = x
        layerNorm(&h, l.ln2w, l.ln2b)
        var up = [Float](repeating: 0, count: n * cfg.ff)
        matmul(h, l.upW, n, cfg.ff, e, &up)
        addBias(&up, l.upB, n)
        gelu(&up)
        var down = [Float](repeating: 0, count: n * e)
        matmul(up, l.downW, n, e, cfg.ff, &down)
        addBias(&down, l.downB, n)
        vDSP_vadd(x, 1, down, 1, &x, 1, vDSP_Length(n * e))
    }

    private func attention(_ qkv: [Float], _ t: Tables) -> [Float] {
        let e = cfg.embd, d = cfg.headDim
        let n = qkv.count / (3 * e)
        let scale = 1 / Float(d).squareRoot()
        var ctx = [Float](repeating: 0, count: n * e)
        for head in 0..<cfg.heads {
            var q = [Float](repeating: 0, count: n * d)
            var k = [Float](repeating: 0, count: n * d)
            var v = [Float](repeating: 0, count: n * d)
            for s in 0..<n {
                let base = s * 3 * e + head * d
                for i in 0..<d {
                    q[s * d + i] = qkv[base + i]
                    k[s * d + i] = qkv[base + e + i]
                    v[s * d + i] = qkv[base + 2 * e + i]
                }
                rope(&q, s, t); rope(&k, s, t)
            }
            var kT = [Float](repeating: 0, count: n * d)
            vDSP_mtrans(k, 1, &kT, 1, vDSP_Length(d), vDSP_Length(n))
            var scores = [Float](repeating: 0, count: n * n)
            matmul(q, kT, n, n, d, &scores)
            var sc = scale
            vDSP_vsmul(scores, 1, &sc, &scores, 1, vDSP_Length(n * n))
            softmaxRows(&scores, n)
            var o = [Float](repeating: 0, count: n * d)
            matmul(scores, v, n, d, n, &o)
            for s in 0..<n {
                for i in 0..<d { ctx[s * e + head * d + i] = o[s * d + i] }
            }
        }
        return ctx
    }

    private func rope(_ b: inout [Float], _ s: Int, _ t: Tables) {
        let d = cfg.headDim, half = d / 2
        for j in 0..<half {
            let c = t.cos[s * half + j]
            let sn = t.sin[s * half + j]
            let x0 = b[s * d + j]
            let x1 = b[s * d + j + half]
            b[s * d + j] = x0 * c - x1 * sn
            b[s * d + j + half] = x0 * sn + x1 * c
        }
    }

    // The concat is a pure reshape: 4 consecutive merge-ordered rows.
    private func merger(_ x: [Float]) -> [Float] {
        let m2 = cfg.merge * cfg.merge
        let inDim = cfg.embd * m2
        let n = x.count / inDim
        var h = [Float](repeating: 0, count: n * inDim)
        matmul(x, mm0W, n, inDim, inDim, &h)
        addBias(&h, mm0B, n)
        gelu(&h)
        var out = [Float](repeating: 0, count: n * cfg.projDim)
        matmul(h, mm2W, n, cfg.projDim, inDim, &out)
        addBias(&out, mm2B, n)
        return out
    }

    // ggml [out][in] row-major -> [in][out], so products are plain mmul.
    static func transposed(_ a: [Float], _ rows: Int, _ cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        vDSP_mtrans(a, 1, &out, 1, vDSP_Length(cols), vDSP_Length(rows))
        return out
    }

    private func matmul(_ a: [Float], _ b: [Float], _ m: Int, _ n: Int,
                        _ k: Int, _ c: inout [Float]) {
        vDSP_mmul(a, 1, b, 1, &c, 1,
                  vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }

    private func addBias(_ x: inout [Float], _ b: [Float], _ rows: Int) {
        let n = b.count
        for r in 0..<rows {
            b.withUnsafeBufferPointer { bp in
                x.withUnsafeMutableBufferPointer { xp in
                    vDSP_vadd(xp.baseAddress! + r * n, 1, bp.baseAddress!, 1,
                              xp.baseAddress! + r * n, 1, vDSP_Length(n))
                }
            }
        }
    }

    private func layerNorm(_ x: inout [Float], _ w: [Float], _ b: [Float]) {
        let e = w.count
        let rows = x.count / e
        for r in 0..<rows {
            var mean: Float = 0
            var sq: Float = 0
            x.withUnsafeBufferPointer { xp in
                vDSP_meanv(xp.baseAddress! + r * e, 1, &mean, vDSP_Length(e))
                vDSP_measqv(xp.baseAddress! + r * e, 1, &sq, vDSP_Length(e))
            }
            let inv = 1 / (sq - mean * mean + cfg.eps).squareRoot()
            for i in 0..<e {
                x[r * e + i] = (x[r * e + i] - mean) * inv * w[i] + b[i]
            }
        }
    }

    private func gelu(_ x: inout [Float]) {
        let n = x.count
        var t = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let v = x[i]
            t[i] = 0.7978845608 * (v + 0.044715 * v * v * v)
        }
        var cnt = Int32(n)
        vvtanhf(&t, t, &cnt)
        for i in 0..<n { x[i] = 0.5 * x[i] * (1 + t[i]) }
    }

    private func softmaxRows(_ x: inout [Float], _ n: Int) {
        for r in 0..<n {
            var mx: Float = 0
            var sum: Float = 0
            x.withUnsafeMutableBufferPointer { xp in
                let row = xp.baseAddress! + r * n
                vDSP_maxv(row, 1, &mx, vDSP_Length(n))
                mx = -mx
                vDSP_vsadd(row, 1, &mx, row, 1, vDSP_Length(n))
                var cnt = Int32(n)
                vvexpf(row, row, &cnt)
                vDSP_sve(row, 1, &sum, vDSP_Length(n))
                var inv = 1 / sum
                vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(n))
            }
        }
    }
}
