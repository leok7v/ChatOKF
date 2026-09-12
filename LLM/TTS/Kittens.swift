// Mechanical transliteration of kittens.c from github.com/leok7v/tts.cli.
// The arena resets and their SEGMENTATION are output bits, not bookkeeping.
import Accelerate
import Foundation

struct KittensArch {
    var vocab = 0, maxPos = 0, tokenTypes = 0
    var embdDim = 0, hidden = 0, nLayers = 0, nHeads = 0
    var headDim = 0, ffnDim = 0
    var lnEps: Float = 0.0
    var bertEncDim = 0, styleDim = 0, lstmHidden = 0, durLogits = 0
    var audioPerFrame = 0, istftHop = 0, istftTrim = 0
}

final class KittensWeights {
    var eWord: Tensor!,   ePos: Tensor!,    eType: Tensor!
    var eLnW: Tensor!,    eLnB: Tensor!
    var projW: Tensor!,   projB: Tensor!
    var qW: Tensor!, qB: Tensor!, kW: Tensor!, kB: Tensor!
    var vW: Tensor!, vB: Tensor!, oW: Tensor!, oB: Tensor!
    var attnLnW: Tensor!, attnLnB: Tensor!
    var ffnW: Tensor!, ffnB: Tensor!, ffnOutW: Tensor!, ffnOutB: Tensor!
    var fullLnW: Tensor!, fullLnB: Tensor!
    var bertEncW: Tensor!, bertEncB: Tensor!
    var ptL0fW: Tensor!, ptL0fR: Tensor!, ptL0fb: Tensor!
    var ptL0bW: Tensor!, ptL0bR: Tensor!, ptL0bb: Tensor!
    var ptFc1W: Tensor!, ptFc1B: Tensor!
    var ptL2fW: Tensor!, ptL2fR: Tensor!, ptL2fb: Tensor!
    var ptL2bW: Tensor!, ptL2bR: Tensor!, ptL2bb: Tensor!
    var ptFc3W: Tensor!, ptFc3B: Tensor!
    var durLfW: Tensor!, durLfR: Tensor!, durLfb: Tensor!
    var durLbW: Tensor!, durLbR: Tensor!, durLbb: Tensor!
    var durW: Tensor!, durB: Tensor!
    var acEmbd: Tensor!
    var acC0W: Tensor!, acC0B: Tensor!, acLn0G: Tensor!, acLn0B: Tensor!
    var acC1W: Tensor!, acC1B: Tensor!, acLn1G: Tensor!, acLn1B: Tensor!
    var acLfW: Tensor!, acLfR: Tensor!, acLfb: Tensor!
    var acLbW: Tensor!, acLbR: Tensor!, acLbb: Tensor!
    var shfW: Tensor!, shfR: Tensor!, shfb: Tensor!
    var shbW: Tensor!, shbR: Tensor!, shbb: Tensor!
}

final class KittensCtx {
    let gguf: GGUF
    let weightsArena: Arena
    let scratchArena: Arena
    var arch = KittensArch()
    var W = KittensWeights()
    var cache: [String: Tensor] = [:]
    var err = ""

    init(_ gguf: GGUF, _ weightsArena: Arena, _ scratchArena: Arena) {
        self.gguf = gguf
        self.weightsArena = weightsArena
        self.scratchArena = scratchArena
    }
}

private func setCtxErr(_ ctx: KittensCtx, _ message: String) {
    ctx.err = message
}

func kittensLastError(_ ctx: KittensCtx?) -> String {
    return ctx != nil ? ctx!.err : ""
}

// The shared reader collapses integer KVs to Int and float KVs to Double, so
// these read the VALUE rather than the declared GGUF type the C matched on.
private func ktU32(_ g: GGUF, _ key: String,
                   _ out: inout UInt32) -> Bool {
    let v = g.int(key)
    if let v = v { out = UInt32(truncatingIfNeeded: v) }
    return v != nil
}

private func ktF32(_ g: GGUF, _ key: String,
                   _ out: inout Float) -> Bool {
    let v = g.double(key)
    if let v = v { out = Float(v) }
    return v != nil
}

// The GGUF mapping backs every F32 weight in place, so the ctx must outlive
// them; F16 widens into the weights arena, which is never reset.
private func ktLoadTensor(_ g: GGUF, _ arena: Arena,
                          _ name: String) -> Tensor? {
    var out: Tensor? = nil
    if let t = g.maybe(name) {
        out = Dense.tensor(t, into: arena)
        precondition(out != nil, "gguf: unsupported tensor dtype")
    }
    return out
}

private func bind(_ ctx: KittensCtx, _ name: String) -> Tensor? {
    let t = ktLoadTensor(ctx.gguf, ctx.weightsArena, name)
    if t == nil {
        setCtxErr(ctx, "missing GGUF tensor: \(name)")
    }
    return t
}

private func bindReq(_ ctx: KittensCtx, _ name: String) -> Tensor {
    let t = bind(ctx, name)
    precondition(t != nil, "kittens: required tensor missing: \(name)")
    return t!
}

private func loadArch(_ ctx: KittensCtx) -> Bool {
    var v: UInt32 = 0
    let g = ctx.gguf
    let loaded = ktU32(g, "kittens-tts.vocab_size", &v)
    if !loaded {
        setCtxErr(ctx, "missing arch KV: kittens-tts.vocab_size")
    } else {
        ctx.arch.vocab = Int(v)
        _ = ktU32(g, "kittens-tts.max_position", &v)
        ctx.arch.maxPos = Int(v)
        _ = ktU32(g, "kittens-tts.token_types", &v)
        ctx.arch.tokenTypes = Int(v)
        _ = ktU32(g, "kittens-tts.embedding_dim", &v)
        ctx.arch.embdDim = Int(v)
        _ = ktU32(g, "kittens-tts.hidden_size", &v)
        ctx.arch.hidden = Int(v)
        _ = ktU32(g, "kittens-tts.num_layers", &v)
        ctx.arch.nLayers = Int(v)
        _ = ktU32(g, "kittens-tts.num_heads", &v)
        ctx.arch.nHeads = Int(v)
        _ = ktU32(g, "kittens-tts.head_dim", &v)
        ctx.arch.headDim = Int(v)
        _ = ktU32(g, "kittens-tts.ffn_dim", &v)
        ctx.arch.ffnDim = Int(v)
        _ = ktF32(g, "kittens-tts.layer_norm_eps", &ctx.arch.lnEps)
        _ = ktU32(g, "kittens-tts.bert_enc_dim", &v)
        ctx.arch.bertEncDim = Int(v)
        _ = ktU32(g, "kittens-tts.style_dim", &v)
        ctx.arch.styleDim = Int(v)
        _ = ktU32(g, "kittens-tts.lstm_hidden", &v)
        ctx.arch.lstmHidden = Int(v)
        _ = ktU32(g, "kittens-tts.dur_logits", &v)
        ctx.arch.durLogits = Int(v)
        _ = ktU32(g, "kittens-tts.audio_per_frame", &v)
        ctx.arch.audioPerFrame = Int(v)
        _ = ktU32(g, "kittens-tts.istft_hop", &v)
        ctx.arch.istftHop = Int(v)
        _ = ktU32(g, "kittens-tts.istft_trim", &v)
        ctx.arch.istftTrim = Int(v)
    }
    return loaded
}

private func bindWeights(_ ctx: KittensCtx) {
    let W = ctx.W
    W.eWord   = bindReq(ctx, "embd.word.weight")
    W.ePos    = bindReq(ctx, "embd.pos.weight")
    W.eType   = bindReq(ctx, "embd.type.weight")
    W.eLnW    = bindReq(ctx, "embd.ln.weight")
    W.eLnB    = bindReq(ctx, "embd.ln.bias")
    W.projW   = bindReq(ctx, "embd_to_hidden.weight")
    W.projB   = bindReq(ctx, "embd_to_hidden.bias")
    W.qW      = bindReq(ctx, "layer.attn_q.weight")
    W.qB      = bindReq(ctx, "layer.attn_q.bias")
    W.kW      = bindReq(ctx, "layer.attn_k.weight")
    W.kB      = bindReq(ctx, "layer.attn_k.bias")
    W.vW      = bindReq(ctx, "layer.attn_v.weight")
    W.vB      = bindReq(ctx, "layer.attn_v.bias")
    W.oW      = bindReq(ctx, "layer.attn_out.weight")
    W.oB      = bindReq(ctx, "layer.attn_out.bias")
    W.attnLnW = bindReq(ctx, "layer.attn_ln.weight")
    W.attnLnB = bindReq(ctx, "layer.attn_ln.bias")
    W.ffnW    = bindReq(ctx, "layer.ffn.weight")
    W.ffnB    = bindReq(ctx, "layer.ffn.bias")
    W.ffnOutW = bindReq(ctx, "layer.ffn_out.weight")
    W.ffnOutB = bindReq(ctx, "layer.ffn_out.bias")
    W.fullLnW = bindReq(ctx, "layer.full_ln.weight")
    W.fullLnB = bindReq(ctx, "layer.full_ln.bias")
    W.bertEncW = bindReq(ctx, "bert_enc.weight")
    W.bertEncB = bindReq(ctx, "bert_enc.bias")
    W.ptL0fW = bindReq(ctx, "pred_text.lstm0.fwd.W")
    W.ptL0fR = bindReq(ctx, "pred_text.lstm0.fwd.R")
    W.ptL0fb = bindReq(ctx, "pred_text.lstm0.fwd.b")
    W.ptL0bW = bindReq(ctx, "pred_text.lstm0.bwd.W")
    W.ptL0bR = bindReq(ctx, "pred_text.lstm0.bwd.R")
    W.ptL0bb = bindReq(ctx, "pred_text.lstm0.bwd.b")
    W.ptFc1W = bindReq(ctx, "pred_text.fc1.weight")
    W.ptFc1B = bindReq(ctx, "pred_text.fc1.bias")
    W.ptL2fW = bindReq(ctx, "pred_text.lstm2.fwd.W")
    W.ptL2fR = bindReq(ctx, "pred_text.lstm2.fwd.R")
    W.ptL2fb = bindReq(ctx, "pred_text.lstm2.fwd.b")
    W.ptL2bW = bindReq(ctx, "pred_text.lstm2.bwd.W")
    W.ptL2bR = bindReq(ctx, "pred_text.lstm2.bwd.R")
    W.ptL2bb = bindReq(ctx, "pred_text.lstm2.bwd.b")
    W.ptFc3W = bindReq(ctx, "pred_text.fc3.weight")
    W.ptFc3B = bindReq(ctx, "pred_text.fc3.bias")
    W.durLfW = bindReq(ctx, "dur.lstm.fwd.W")
    W.durLfR = bindReq(ctx, "dur.lstm.fwd.R")
    W.durLfb = bindReq(ctx, "dur.lstm.fwd.b")
    W.durLbW = bindReq(ctx, "dur.lstm.bwd.W")
    W.durLbR = bindReq(ctx, "dur.lstm.bwd.R")
    W.durLbb = bindReq(ctx, "dur.lstm.bwd.b")
    W.durW   = bindReq(ctx, "dur_proj.weight")
    W.durB   = bindReq(ctx, "dur_proj.bias")
    W.acEmbd = bindReq(ctx, "acoustic.embd.weight")
    W.acC0W  = bindReq(ctx, "acoustic.cnn0.weight")
    W.acC0B  = bindReq(ctx, "acoustic.cnn0.bias")
    W.acLn0G = bindReq(ctx, "acoustic.ln0.gamma")
    W.acLn0B = bindReq(ctx, "acoustic.ln0.beta")
    W.acC1W  = bindReq(ctx, "acoustic.cnn1.weight")
    W.acC1B  = bindReq(ctx, "acoustic.cnn1.bias")
    W.acLn1G = bindReq(ctx, "acoustic.ln1.gamma")
    W.acLn1B = bindReq(ctx, "acoustic.ln1.beta")
    W.acLfW  = bindReq(ctx, "acoustic.lstm.fwd.W")
    W.acLfR  = bindReq(ctx, "acoustic.lstm.fwd.R")
    W.acLfb  = bindReq(ctx, "acoustic.lstm.fwd.b")
    W.acLbW  = bindReq(ctx, "acoustic.lstm.bwd.W")
    W.acLbR  = bindReq(ctx, "acoustic.lstm.bwd.R")
    W.acLbb  = bindReq(ctx, "acoustic.lstm.bwd.b")
    W.shfW   = bindReq(ctx, "shared.lstm.fwd.W")
    W.shfR   = bindReq(ctx, "shared.lstm.fwd.R")
    W.shfb   = bindReq(ctx, "shared.lstm.fwd.b")
    W.shbW   = bindReq(ctx, "shared.lstm.bwd.W")
    W.shbR   = bindReq(ctx, "shared.lstm.bwd.R")
    W.shbb   = bindReq(ctx, "shared.lstm.bwd.b")
}

func kittensCreate(_ ggufPath: String) -> KittensCtx? {
    var ctx: KittensCtx? = nil
    var success = false
    if let g = try? GGUF(path: ggufPath) {
        let c = KittensCtx(g, arenaNew(64 * 1024 * 1024),
                           arenaNew(1 * 1024 * 1024))
        ctx = c
        if loadArch(c) {
            bindWeights(c)
            success = true
        }
    }
    var result: KittensCtx? = nil
    if success {
        result = ctx
    } else if let c = ctx {
        kittensDestroy(c)
    }
    return result
}

func kittensDestroy(_ ctx: KittensCtx?) {
    if let ctx = ctx {
        arenaFree(ctx.scratchArena)
        arenaFree(ctx.weightsArena)
        ctx.cache.removeAll()
    }
}

private func named(_ ctx: KittensCtx, _ name: String) -> Tensor? {
    var result: Tensor?
    if let hit = ctx.cache[name] {
        result = hit
    } else {
        let t = ktLoadTensor(ctx.gguf, ctx.weightsArena, name)
        if let t = t { ctx.cache[name] = t }
        result = t
    }
    return result
}

private func layerNorm(_ x: Tensor, _ w: Tensor?, _ b: Tensor?,
                       _ eps: Float) -> Tensor {
    var h = tensorNorm(x, 0, eps)
    if let w = w { h = tensorMul(h, w) }
    if let b = b { h = tensorAdd(h, b) }
    return h
}

private func adaLayerNorm(_ x: Tensor, _ style: Tensor,
                          _ fcW: Tensor, _ fcB: Tensor,
                          _ C: Int) -> Tensor {
    var h = tensorMulMat(fcW, style)
    h = tensorAdd(h, fcB)
    let fsz = MemoryLayout<Float>.size
    let gamma = tensorView1d(h, Int64(C), 0)
    let beta  = tensorView1d(h, Int64(C), C * fsz)
    let n   = tensorNorm(x, 0, 1e-5)
    let nG  = tensorMul(n, gamma)
    var out = tensorAdd(nG, n)
    out = tensorAdd(out, beta)
    return out
}

private func adaIn1d(_ x: Tensor, _ style: Tensor,
                     _ fcW: Tensor, _ fcB: Tensor,
                     _ nW: Tensor?, _ nB: Tensor?,
                     _ C: Int) -> Tensor {
    var h = tensorMulMat(fcW, style)
    h = tensorAdd(h, fcB)
    let fsz = MemoryLayout<Float>.size
    let gamma = tensorView1d(h, Int64(C), 0)
    let beta  = tensorView1d(h, Int64(C), C * fsz)
    let gammaC = tensorCont(gamma)
    let betaC  = tensorCont(beta)
    let xT = tensorCont(tensorTranspose(x))
    let nT = tensorNorm(xT, 0, 1e-5)
    var n  = tensorCont(tensorTranspose(nT))
    if let nW = nW { n = tensorMul(n, nW) }
    if let nB = nB { n = tensorAdd(n, nB) }
    let nG  = tensorMul(n, gammaC)
    var out = tensorAdd(nG, n)
    out = tensorAdd(out, betaC)
    return out
}

private func snake1d(_ x: Tensor, _ alpha: Tensor) -> Tensor {
    let ax = tensorMul(x, alpha)
    let s  = tensorSin(ax)
    let s2 = tensorMul(s, s)
    let s2OverA = tensorDiv(s2, alpha)
    return tensorAdd(x, s2OverA)
}

private func conv1dNlc(_ x: Tensor, _ kw: Tensor, _ kb: Tensor?,
                       _ stride: Int, _ padIn: Int,
                       _ dilation: Int) -> Tensor {
    let K = Int(kw.ne[0])
    var pad = padIn
    if pad < 0 { pad = (K - 1) / 2 }
    let xNcl = tensorCont(tensorTranspose(x))
    let x3d = tensorReshape3d(xNcl, xNcl.ne[0], xNcl.ne[1], 1)
    var y3d = tensorConv1d(kw, x3d, stride, pad, dilation)
    if let kb = kb {
        let b3 = tensorReshape3d(kb, 1, kb.ne[0], 1)
        y3d = tensorAdd(y3d, b3)
    }
    let y2d = tensorReshape2d(y3d, y3d.ne[0], y3d.ne[1])
    return tensorCont(tensorTranspose(y2d))
}

private func repeatInterleave2xNlc(_ x: Tensor) -> Tensor {
    let C = x.ne[0]
    let L = x.ne[1]
    let x4 = tensorReshape4d(x, C, L, 1, 1)
    let stacked = tensorConcat(x4, x4, 2)
    let p = tensorPermute(stacked, 0, 2, 1, 3)
    let pc = tensorCont(p)
    return tensorReshape2d(pc, C, 2 * L)
}

private func insertZeros2xNlc(_ x: Tensor) -> Tensor {
    let C = x.ne[0]
    let L = x.ne[1]
    let zeros = tensorScale(x, 0.0)
    let x4 = tensorReshape4d(x, C, L, 1, 1)
    let z4 = tensorReshape4d(zeros, C, L, 1, 1)
    let stacked = tensorConcat(x4, z4, 2)
    let p = tensorPermute(stacked, 0, 2, 1, 3)
    let pc = tensorCont(p)
    return tensorReshape2d(pc, C, 2 * L)
}

private func upsample2xDwT(_ x: Tensor, _ poolW: Tensor,
                           _ poolB: Tensor?) -> Tensor {
    let K = Int(poolW.ne[0])
    assert(K == 3)
    let xup = insertZeros2xNlc(x)
    let xupNcl = tensorCont(tensorTranspose(xup))
    let xup3d = tensorReshape3d(xupNcl, xupNcl.ne[0],
                                xupNcl.ne[1], 1)
    var y3d = tensorConv1dDw(poolW, xup3d, 1, 1, 1)
    if let poolB = poolB {
        let b3 = tensorReshape3d(poolB, 1, poolB.ne[0], 1)
        y3d = tensorAdd(y3d, b3)
    }
    let y2d = tensorReshape2d(y3d, y3d.ne[0], y3d.ne[1])
    return tensorCont(tensorTranspose(y2d))
}

private func convTranspose1dNlc(_ x: Tensor, _ kw: Tensor,
                                _ kb: Tensor?, _ stride: Int,
                                _ pad: Int) -> Tensor {
    let b = tensorCont(tensorTranspose(x))
    let b3d = tensorReshape3d(b, b.ne[0], b.ne[1], 1)
    var y3d = tensorConvTranspose1d(kw, b3d, stride, pad)
    if let kb = kb {
        let bb = tensorReshape3d(kb, 1, kb.ne[0], 1)
        y3d = tensorAdd(y3d, bb)
    }
    let y2d = tensorReshape2d(y3d, y3d.ne[0], y3d.ne[1])
    return tensorCont(tensorTranspose(y2d))
}

private func reflectionPadLeft(_ x: Tensor, _ n: Int) -> Tensor {
    var result = x
    if n > 0 {
        assert(n == 1)
        let C = x.ne[0]
        let slice = tensorView2d(x, C, 1, Int(x.nb[1]), Int(x.nb[1]))
        let sliceC = tensorCont(slice)
        result = tensorConcat(sliceC, x, 1)
    }
    return result
}

private func styleBcastCxL(_ style: Tensor, _ C: Int,
                           _ L: Int) -> Tensor {
    let s2 = tensorReshape2d(style, Int64(C), 1)
    return tensorRepeatTo(s2, 2, Int64(C), Int64(L), 1, 1)
}

private func lstmDir(_ a: Arena, _ x: Tensor, _ W: Tensor,
                     _ R: Tensor, _ b: Tensor,
                     _ h0: Tensor, _ c0: Tensor,
                     _ H: Int, _ T: Int, _ reverse: Bool) -> Tensor {
    var wxFull = tensorMulMat(W, x)
    wxFull = tensorAdd(wxFull, b)
    let out = tensorNew2d(a, Int64(H), Int64(T))
    var hPrev = h0
    var cPrev = c0
    let fsz = MemoryLayout<Float>.size
    for step in 0..<T {
        let t = reverse ? (T - 1 - step) : step
        let wxT = tensorView1d(wxFull, Int64(4 * H), t * 4 * H * fsz)
        let rh  = tensorMulMat(R, hPrev)
        let z   = tensorAdd(wxT, rh)
        let zi = tensorView1d(z, Int64(H), 0)
        let zf = tensorView1d(z, Int64(H), 1 * H * fsz)
        let zg = tensorView1d(z, Int64(H), 2 * H * fsz)
        let zo = tensorView1d(z, Int64(H), 3 * H * fsz)
        let gi  = tensorSigmoid(zi)
        let gf  = tensorSigmoid(zf)
        let gg  = tensorTanh   (zg)
        let go  = tensorSigmoid(zo)
        let fc  = tensorMul(gf, cPrev)
        let ig  = tensorMul(gi, gg)
        let cT  = tensorAdd(fc, ig)
        let hT  = tensorMul(go, tensorTanh(cT))
        memcpy(UnsafeMutableRawPointer(out.data!) + t * H * fsz,
               hT.data!, H * fsz)
        hPrev = hT
        cPrev = cT
    }
    return out
}

private func bidirLstm(_ a: Arena, _ x: Tensor,
                       _ fW: Tensor, _ fR: Tensor, _ fb: Tensor,
                       _ bW: Tensor, _ bR: Tensor, _ bb: Tensor,
                       _ h0: Tensor, _ c0: Tensor,
                       _ H: Int, _ T: Int) -> Tensor {
    let fwd = lstmDir(a, x, fW, fR, fb, h0, c0, H, T, false)
    let bwd = lstmDir(a, x, bW, bR, bb, h0, c0, H, T, true)
    return tensorConcat(fwd, bwd, 0)
}

private func buildAdaBlock1d(_ ctx: KittensCtx, _ prefix: String,
                             _ x: Tensor, _ style: Tensor,
                             _ shortcutIn: Tensor?,
                             _ divide: Bool) -> Tensor {
    let n1fcW = named(ctx, "\(prefix).n1.fcW")
    let n1fcB = named(ctx, "\(prefix).n1.fcB")
    let n1nW  = named(ctx, "\(prefix).n1.nW")
    let n1nB  = named(ctx, "\(prefix).n1.nB")
    let n2fcW = named(ctx, "\(prefix).n2.fcW")
    let n2fcB = named(ctx, "\(prefix).n2.fcB")
    let n2nW  = named(ctx, "\(prefix).n2.nW")
    let n2nB  = named(ctx, "\(prefix).n2.nB")
    let c1w   = named(ctx, "\(prefix).c1.weight")
    let c1b   = named(ctx, "\(prefix).c1.bias")
    let c2w   = named(ctx, "\(prefix).c2.weight")
    let c2b   = named(ctx, "\(prefix).c2.bias")
    let svw   = named(ctx, "\(prefix).sv.weight")
    let svb   = named(ctx, "\(prefix).sv.bias")
    let poolW = named(ctx, "\(prefix).pool.weight")
    let poolB = named(ctx, "\(prefix).pool.bias")
    assert(n1fcW != nil && n2fcW != nil
           && c1w != nil && c2w != nil)
    let upsample   = (poolW != nil)
    let hasConv1x1 = (svw   != nil)
    let cIn = Int(x.ne[0])
    var h = adaIn1d(x, style, n1fcW!, n1fcB!, n1nW, n1nB, cIn)
    h = tensorLeakyRelu(h, 0.2)
    if upsample { h = upsample2xDwT(h, poolW!, poolB) }
    h = conv1dNlc(h, c1w!, c1b, 1, -1, 1)
    let cMid = Int(c1w!.ne[2])
    h = adaIn1d(h, style, n2fcW!, n2fcB!, n2nW, n2nB, cMid)
    h = tensorLeakyRelu(h, 0.2)
    h = conv1dNlc(h, c2w!, c2b, 1, -1, 1)
    let shortcut = shortcutIn != nil ? shortcutIn! : x
    var res: Tensor
    if upsample {
        let sup = repeatInterleave2xNlc(shortcut)
        res = conv1dNlc(sup, svw!, svb, 1, 0, 1)
    } else if hasConv1x1 {
        res = conv1dNlc(shortcut, svw!, svb, 1, 0, 1)
    } else {
        res = shortcut
    }
    var out = tensorAdd(h, res)
    if divide { out = tensorScale(out, 1.0 / sqrtf(2.0)) }
    return out
}

// savept and hifi_out carry a stage across an arena_reset; in the C they are
// mmap'd pages retired page by page, here plain allocations. No value changes.
struct Savept {
    var ndim: Int32 = 0
    var ne: [Int64] = [1, 1, 1, 1]
    var data: UnsafeMutablePointer<Float>?
}

private func hostAlloc(_ n: Int) -> UnsafeMutablePointer<Float> {
    return UnsafeMutablePointer<Float>.allocate(capacity: n)
}

private func hostFree(_ p: UnsafeMutablePointer<Float>?) {
    if let p = p { p.deallocate() }
}

private func save(_ t: Tensor) -> Savept {
    var p = Savept()
    p.ndim = t.ndim
    for i in 0..<4 { p.ne[i] = t.ne[i] }
    let n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3]
    p.data = hostAlloc(Int(n))
    memcpy(p.data!, t.data!, Int(n) * MemoryLayout<Float>.size)
    return p
}

private func saveptWrap(_ data: UnsafeMutablePointer<Float>,
                        _ n0: Int64, _ n1: Int64) -> Savept {
    var p = Savept()
    p.ndim = 2
    p.ne[0] = n0; p.ne[1] = n1; p.ne[2] = 1; p.ne[3] = 1
    p.data = data
    return p
}

private func restore(_ a: Arena, _ p: Savept) -> Tensor {
    let t = tensorNewNd(a, p.ndim, p.ne)
    let n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3]
    memcpy(t.data!, p.data!, Int(n) * MemoryLayout<Float>.size)
    return t
}

private func saveptFree(_ p: inout Savept) {
    hostFree(p.data)
    p.data = nil
}

struct HifiOut {
    var data: UnsafeMutablePointer<Float>?
    var C: Int = 0
    var L: Int64 = 0
}

private let hifiSegL = 1024
private let hifiSegOverlap = 128
private let dilations = [1, 3, 5]

private func hifiComputeStatsHost(_ host: UnsafePointer<Float>,
                                  _ C: Int, _ L: Int64,
                                  _ mean: UnsafeMutablePointer<Float>,
                                  _ istd: UnsafeMutablePointer<Float>) {
    let tmp = UnsafeMutablePointer<Float>.allocate(capacity: Int(L))
    let Cs = Int(C)
    var c = 0
    while c < C {
        var t = 0
        while t < Int(L) { tmp[t] = host[t * Cs + c]; t += 1 }
        var m: Float = 0.0
        var msq: Float = 0.0
        vDSP_meanv (tmp, 1, &m,   vDSP_Length(L))
        vDSP_measqv(tmp, 1, &msq, vDSP_Length(L))
        let varv = msq - m * m
        mean[c] = m
        istd[c] = 1.0 / sqrtf(varv + 1e-5)
        c += 1
    }
    tmp.deallocate()
}

private func adaIn1dWithStats(_ x: Tensor, _ style: Tensor,
                              _ fcW: Tensor, _ fcB: Tensor,
                              _ nW: Tensor?, _ nB: Tensor?,
                              _ C: Int,
                              _ mean: UnsafePointer<Float>,
                              _ istd: UnsafePointer<Float>) -> Tensor {
    let sa = arenaGetActive()!
    var h = tensorMulMat(fcW, style)
    h = tensorAdd(h, fcB)
    let fsz = MemoryLayout<Float>.size
    let gamma  = tensorView1d(h, Int64(C), 0)
    let beta   = tensorView1d(h, Int64(C), C * fsz)
    let gammaC = tensorCont(gamma)
    let betaC  = tensorCont(beta)
    let negMean = tensorNew1d(sa, Int64(C))
    let istdT   = tensorNew1d(sa, Int64(C))
    var c = 0
    while c < C {
        negMean.data[c] = -mean[c]
        istdT.data[c]   =  istd[c]
        c += 1
    }
    var n = tensorAdd(x, negMean)
    n = tensorMul(n, istdT)
    if let nW = nW { n = tensorMul(n, nW) }
    if let nB = nB { n = tensorAdd(n, nB) }
    let nG  = tensorMul(n, gammaC)
    var out = tensorAdd(nG, n)
    out = tensorAdd(out, betaC)
    return out
}

private func buildHifiBlock(_ ctx: KittensCtx, _ prefix: String,
                            _ px: inout Savept, _ styleIn: Tensor,
                            _ consume: Bool) -> HifiOut {
    let sa = ctx.scratchArena
    let C = Int(px.ne[0])
    let L = px.ne[1]
    arenaSetActive(sa)
    var ps = save(styleIn)
    arenaReset(sa)
    var result = HifiOut(data: nil, C: C, L: L)
    if L <= Int64(hifiSegL + 2 * hifiSegOverlap) {
        var out = restore(sa, px)
        if consume { saveptFree(&px) }
        var style = restore(sa, ps)
        for k in 0..<3 {
            if k > 0 {
                var po = save(out)
                arenaReset(sa)
                out   = restore(sa, po);  saveptFree(&po)
                style = restore(sa, ps)
            }
            let d = dilations[k]
            let a1fcW = named(ctx, "\(prefix).a1.\(k).fcW")
            let a1fcB = named(ctx, "\(prefix).a1.\(k).fcB")
            let a1nW  = named(ctx, "\(prefix).a1.\(k).nW")
            let a1nB  = named(ctx, "\(prefix).a1.\(k).nB")
            let a2fcW = named(ctx, "\(prefix).a2.\(k).fcW")
            let a2fcB = named(ctx, "\(prefix).a2.\(k).fcB")
            let a2nW  = named(ctx, "\(prefix).a2.\(k).nW")
            let a2nB  = named(ctx, "\(prefix).a2.\(k).nB")
            let al1   = named(ctx, "\(prefix).al1.\(k)")
            let al2   = named(ctx, "\(prefix).al2.\(k)")
            let c1w   = named(ctx, "\(prefix).c1.\(k).weight")
            let c1b   = named(ctx, "\(prefix).c1.\(k).bias")
            let c2w   = named(ctx, "\(prefix).c2.\(k).weight")
            let c2b   = named(ctx, "\(prefix).c2.\(k).bias")
            let K1 = Int(c1w!.ne[0])
            let K2 = Int(c2w!.ne[0])
            var h = adaIn1d(out, style, a1fcW!, a1fcB!, a1nW, a1nB, C)
            h = snake1d(h, al1!)
            h = conv1dNlc(h, c1w!, c1b, 1, d * (K1 - 1) / 2, d)
            h = adaIn1d(h, style, a2fcW!, a2fcB!, a2nW, a2nB, C)
            h = snake1d(h, al2!)
            h = conv1dNlc(h, c2w!, c2b, 1, (K2 - 1) / 2, 1)
            out = tensorAdd(out, h)
        }
        let totalFloats = C * Int(L)
        result.data = hostAlloc(totalFloats)
        memcpy(result.data!, out.data!,
               totalFloats * MemoryLayout<Float>.size)
    } else {
        let totalFloats = C * Int(L)
        var outHost: UnsafeMutablePointer<Float>
        if consume {
            outHost = px.data!
            px.data = nil
        } else {
            outHost = hostAlloc(totalFloats)
            memcpy(outHost, px.data!,
                   totalFloats * MemoryLayout<Float>.size)
        }
        let h3Host = UnsafeMutablePointer<Float>
            .allocate(capacity: totalFloats)
        let meanIn  = UnsafeMutablePointer<Float>.allocate(capacity: C)
        let istdIn  = UnsafeMutablePointer<Float>.allocate(capacity: C)
        let meanMid = UnsafeMutablePointer<Float>.allocate(capacity: C)
        let istdMid = UnsafeMutablePointer<Float>.allocate(capacity: C)
        for k in 0..<3 {
            let d = dilations[k]
            let a1fcW = named(ctx, "\(prefix).a1.\(k).fcW")
            let a1fcB = named(ctx, "\(prefix).a1.\(k).fcB")
            let a1nW  = named(ctx, "\(prefix).a1.\(k).nW")
            let a1nB  = named(ctx, "\(prefix).a1.\(k).nB")
            let a2fcW = named(ctx, "\(prefix).a2.\(k).fcW")
            let a2fcB = named(ctx, "\(prefix).a2.\(k).fcB")
            let a2nW  = named(ctx, "\(prefix).a2.\(k).nW")
            let a2nB  = named(ctx, "\(prefix).a2.\(k).nB")
            let al1   = named(ctx, "\(prefix).al1.\(k)")
            let al2   = named(ctx, "\(prefix).al2.\(k)")
            let c1w   = named(ctx, "\(prefix).c1.\(k).weight")
            let c1b   = named(ctx, "\(prefix).c1.\(k).bias")
            let c2w   = named(ctx, "\(prefix).c2.\(k).weight")
            let c2b   = named(ctx, "\(prefix).c2.\(k).bias")
            let K1 = Int(c1w!.ne[0])
            let K2 = Int(c2w!.ne[0])
            hifiComputeStatsHost(outHost, C, L, meanIn, istdIn)
            var a: Int64 = 0
            while a < L {
                var b = a + Int64(hifiSegL)
                if b > L { b = L }
                var lo = a - Int64(hifiSegOverlap)
                if lo < 0 { lo = 0 }
                var hi = b + Int64(hifiSegOverlap)
                if hi > L { hi = L }
                let segW  = hi - lo
                let cmOff = Int(a - lo)
                let cmN   = Int(b - a)
                arenaReset(sa)
                let segIn = tensorNew2d(sa, Int64(C), segW)
                memcpy(segIn.data!, outHost + Int(lo) * C,
                       Int(segW) * C * MemoryLayout<Float>.size)
                let sty = restore(sa, ps)
                var h = adaIn1dWithStats(segIn, sty, a1fcW!, a1fcB!,
                                         a1nW, a1nB, C, meanIn, istdIn)
                h = snake1d(h, al1!)
                h = conv1dNlc(h, c1w!, c1b, 1, d * (K1 - 1) / 2, d)
                memcpy(h3Host + Int(a) * C, h.data! + cmOff * C,
                       cmN * C * MemoryLayout<Float>.size)
                a = b
            }
            hifiComputeStatsHost(h3Host, C, L, meanMid, istdMid)
            a = 0
            while a < L {
                var b = a + Int64(hifiSegL)
                if b > L { b = L }
                var lo = a - Int64(hifiSegOverlap)
                if lo < 0 { lo = 0 }
                var hi = b + Int64(hifiSegOverlap)
                if hi > L { hi = L }
                let segW  = hi - lo
                let cmOff = Int(a - lo)
                let cmN   = Int(b - a)
                arenaReset(sa)
                let segH3 = tensorNew2d(sa, Int64(C), segW)
                memcpy(segH3.data!, h3Host + Int(lo) * C,
                       Int(segW) * C * MemoryLayout<Float>.size)
                let sty = restore(sa, ps)
                var h = adaIn1dWithStats(segH3, sty, a2fcW!, a2fcB!,
                                         a2nW, a2nB, C, meanMid, istdMid)
                h = snake1d(h, al2!)
                h = conv1dNlc(h, c2w!, c2b, 1, (K2 - 1) / 2, 1)
                let src = UnsafePointer<Float>(h.data!) + cmOff * C
                let totalN = cmN * C
                let dst = outHost + Int(a) * C
                var i = 0
                while i < totalN { dst[i] += src[i]; i += 1 }
                a = b
            }
        }
        arenaReset(sa)
        result.data = outHost
        h3Host.deallocate()
        meanIn.deallocate();   istdIn.deallocate()
        meanMid.deallocate();  istdMid.deallocate()
    }
    saveptFree(&ps)
    return result
}

private func buildAlbert(_ ctx: KittensCtx, _ L: Int,
                         _ ids: [Int32], _ pos: [Int32],
                         _ type: [Int32]) -> Tensor {
    let a = ctx.arch
    let W = ctx.W
    let sa = ctx.scratchArena
    var h = tensorGetRows(W.eWord, ids,  L)
    let p = tensorGetRows(W.ePos,  pos,  L)
    let t = tensorGetRows(W.eType, type, L)
    h = tensorAdd(h, p)
    h = tensorAdd(h, t)
    h = layerNorm(h, W.eLnW, W.eLnB, a.lnEps)
    h = tensorMulMat(W.projW, h)
    h = tensorAdd(h, W.projB)
    assert(tensorIsPacked(h))
    do {
        var pH = save(h)
        arenaReset(sa)
        h = restore(sa, pH)
        saveptFree(&pH)
    }
    let kqScale = 1.0 / sqrtf(Float(a.headDim))
    for il in 0..<a.nLayers {
        if il > 0 {
            assert(tensorIsPacked(h))
            var pH = save(h)
            arenaReset(sa)
            h = restore(sa, pH)
            saveptFree(&pH)
        }
        let residual = h
        var q = tensorAdd(tensorMulMat(W.qW, h), W.qB)
        var k = tensorAdd(tensorMulMat(W.kW, h), W.kB)
        var v = tensorAdd(tensorMulMat(W.vW, h), W.vB)
        q = tensorReshape4d(q, Int64(a.headDim), Int64(a.nHeads),
                            Int64(L), 1)
        k = tensorReshape4d(k, Int64(a.headDim), Int64(a.nHeads),
                            Int64(L), 1)
        v = tensorReshape4d(v, Int64(a.headDim), Int64(a.nHeads),
                            Int64(L), 1)
        q = tensorPermute(q, 0, 2, 1, 3)
        k = tensorPermute(k, 0, 2, 1, 3)
        v = tensorPermute(v, 0, 2, 1, 3)
        v = tensorCont(tensorTranspose(v))
        var kq = tensorMulMat(k, q)
        kq = tensorSoftmax(kq, 0, kqScale)
        var kqv = tensorMulMat(v, kq)
        kqv = tensorPermute(kqv, 0, 2, 1, 3)
        kqv = tensorCont2d(kqv, Int64(a.hidden), Int64(L))
        let attOut = tensorAdd(tensorMulMat(W.oW, kqv), W.oB)
        h = tensorAdd(attOut, residual)
        h = layerNorm(h, W.attnLnW, W.attnLnB, a.lnEps)
        let mid = h
        var ffn = tensorAdd(tensorMulMat(W.ffnW, h), W.ffnB)
        ffn = tensorGeluErf(ffn)
        ffn = tensorAdd(tensorMulMat(W.ffnOutW, ffn), W.ffnOutB)
        h = tensorAdd(ffn, mid)
        h = layerNorm(h, W.fullLnW, W.fullLnB, a.lnEps)
    }
    return h
}

private func buildPredText(_ ctx: KittensCtx, _ bertOut: Tensor,
                           _ style: Tensor, _ h0: Tensor,
                           _ c0: Tensor, _ L: Int) -> Tensor {
    let W = ctx.W
    let sa = ctx.scratchArena
    let C = 128
    let H = ctx.arch.lstmHidden
    let sBcast = styleBcastCxL(style, C, L)
    let x = tensorConcat(bertOut, sBcast, 0)
    let y = bidirLstm(sa, x,
                      W.ptL0fW, W.ptL0fR, W.ptL0fb,
                      W.ptL0bW, W.ptL0bR, W.ptL0bb,
                      h0, c0, H, L)
    let y1 = adaLayerNorm(y, style, W.ptFc1W, W.ptFc1B, C)
    let x2 = tensorConcat(y1, sBcast, 0)
    let y2 = bidirLstm(sa, x2,
                       W.ptL2fW, W.ptL2fR, W.ptL2fb,
                       W.ptL2bW, W.ptL2bR, W.ptL2bb,
                       h0, c0, H, L)
    return adaLayerNorm(y2, style, W.ptFc3W, W.ptFc3B, C)
}

private func buildAcoustic(_ ctx: KittensCtx, _ ids: [Int32],
                           _ h0: Tensor, _ c0: Tensor,
                           _ L: Int) -> Tensor {
    let W = ctx.W
    let sa = ctx.scratchArena
    let H = ctx.arch.lstmHidden
    var x = tensorGetRows(W.acEmbd, ids, L)
    for i in 0..<2 {
        let cnnW = (i == 0) ? W.acC0W  : W.acC1W
        let cnnB = (i == 0) ? W.acC0B  : W.acC1B
        let lnG  = (i == 0) ? W.acLn0G : W.acLn1G
        let lnB  = (i == 0) ? W.acLn0B : W.acLn1B
        let K = Int(cnnW!.ne[0])
        let pad = (K - 1) / 2
        let xNcl = tensorCont(tensorTranspose(x))
        let xNcl3d = tensorReshape3d(xNcl, xNcl.ne[0],
                                     xNcl.ne[1], 1)
        var y3d = tensorConv1d(cnnW!, xNcl3d, 1, pad, 1)
        let b3 = tensorReshape3d(cnnB!, 1, cnnB!.ne[0], 1)
        y3d = tensorAdd(y3d, b3)
        let y2d = tensorReshape2d(y3d, y3d.ne[0], y3d.ne[1])
        x = tensorCont(tensorTranspose(y2d))
        x = layerNorm(x, lnG, lnB, 1e-5)
        x = tensorLeakyRelu(x, 0.2)
    }
    let y = bidirLstm(sa, x,
                      W.acLfW, W.acLfR, W.acLfb,
                      W.acLbW, W.acLbR, W.acLbb,
                      h0, c0, H, L)
    return y
}

struct TextstageOuts {
    var prosody256: Tensor
    var text: Tensor
    var durSig: Tensor
}

private func buildTextstage(_ ctx: KittensCtx, _ L: Int,
                            _ ids: [Int32], _ pos: [Int32],
                            _ type: [Int32], _ stylePrIn: Tensor,
                            _ h0In: Tensor,
                            _ c0In: Tensor) -> TextstageOuts {
    let W = ctx.W
    let sa = ctx.scratchArena
    var pSty = save(stylePrIn)
    var pH0  = save(h0In)
    var pC0  = save(c0In)
    let bert = buildAlbert(ctx, L, ids, pos, type)
    let stylePr = restore(sa, pSty)
    let h0      = restore(sa, pH0)
    let c0      = restore(sa, pC0)
    saveptFree(&pSty)
    saveptFree(&pH0)
    saveptFree(&pC0)
    let bertProj = tensorAdd(
        tensorMulMat(W.bertEncW, bert), W.bertEncB)
    let prosody = buildPredText(ctx, bertProj, stylePr, h0, c0, L)
    let sBcast = styleBcastCxL(stylePr, ctx.arch.styleDim, L)
    let prosody256 = tensorConcat(prosody, sBcast, 0)
    let dlstm = bidirLstm(sa, prosody256,
                          W.durLfW, W.durLfR, W.durLfb,
                          W.durLbW, W.durLbR, W.durLbb,
                          h0, c0, ctx.arch.lstmHidden, L)
    let durLogits = tensorAdd(
        tensorMulMat(W.durW, dlstm), W.durB)
    let durSig = tensorSigmoid(durLogits)
    let text = buildAcoustic(ctx, ids, h0, c0, L)
    let r = TextstageOuts(prosody256: prosody256, text: text,
                          durSig: durSig)
    return r
}

struct GenfrontOuts {
    var f0Proj: Tensor
    var nProj: Tensor
}

private func buildGenfront(_ ctx: KittensCtx,
                           _ prosodyLrNlc: Tensor,
                           _ style: Tensor, _ h0: Tensor,
                           _ c0: Tensor, _ F: Int) -> GenfrontOuts {
    let W = ctx.W
    let sa = ctx.scratchArena
    let H = ctx.arch.lstmHidden
    let sh = bidirLstm(sa, prosodyLrNlc,
                       W.shfW, W.shfR, W.shfb,
                       W.shbW, W.shbR, W.shbb,
                       h0, c0, H, F)
    var f0 = buildAdaBlock1d(ctx, "f0.0", sh, style, sh, true)
    f0 = buildAdaBlock1d(ctx, "f0.1", f0, style, f0, true)
    f0 = buildAdaBlock1d(ctx, "f0.2", f0, style, f0, true)
    let f0ProjW = named(ctx, "f0_proj.weight")
    let f0ProjB = named(ctx, "f0_proj.bias")
    let f0p = conv1dNlc(f0, f0ProjW!, f0ProjB, 1, -1, 1)
    var nx = buildAdaBlock1d(ctx, "n.0", sh, style, sh, true)
    nx = buildAdaBlock1d(ctx, "n.1", nx, style, nx, true)
    nx = buildAdaBlock1d(ctx, "n.2", nx, style, nx, true)
    let nProjW = named(ctx, "n_proj.weight")
    let nProjB = named(ctx, "n_proj.bias")
    let np = conv1dNlc(nx, nProjW!, nProjB, 1, -1, 1)
    let r = GenfrontOuts(f0Proj: f0p, nProj: np)
    return r
}

private func buildDecoder(_ ctx: KittensCtx, _ textLr: Tensor,
                          _ f0Proj: Tensor, _ nProj: Tensor,
                          _ styleAco: Tensor) -> Tensor {
    let asrW = named(ctx, "dec.asr.weight")
    let asrB = named(ctx, "dec.asr.bias")
    let f0W  = named(ctx, "dec.f0_conv.weight")
    let f0B  = named(ctx, "dec.f0_conv.bias")
    let nW   = named(ctx, "dec.n_conv.weight")
    let nB   = named(ctx, "dec.n_conv.bias")
    let asr   = conv1dNlc(textLr, asrW!, asrB, 1, 0, 1)
    let f0Dn = conv1dNlc(f0Proj, f0W!, f0B, 2, 1, 1)
    let nDn  = conv1dNlc(nProj,  nW!,  nB,  2, 1, 1)
    var encIn = tensorConcat(textLr, f0Dn, 0)
    encIn = tensorConcat(encIn, nDn, 0)
    var x = buildAdaBlock1d(ctx, "dec.encode", encIn,
                            styleAco, encIn, true)
    for i in 0..<4 {
        let prefix = "dec.decode.\(i)"
        var xCat = tensorConcat(x, asr, 0)
        xCat = tensorConcat(xCat, f0Dn, 0)
        xCat = tensorConcat(xCat, nDn,  0)
        x = buildAdaBlock1d(ctx, prefix, xCat, styleAco, xCat, true)
    }
    return x
}

struct NoiseOuts {
    var nr0: HifiOut
    var nr1: HifiOut
}

private let noiseSegF = 256
private let noiseSegU = 16384
private let noiseSegV = 4096

struct NoiseHosts {
    var fpf: UnsafePointer<Float>
    var ps: UnsafePointer<Float>
    var f0: UnsafePointer<Float>
    var srange: UnsafePointer<Float>
    var tf: Int64
}

private func noiseSingenSegment(_ ctx: KittensCtx,
                                _ h: NoiseHosts,
                                _ fa: Int64, _ fb: Int64,
                                _ outH: UnsafeMutablePointer<Float>) {
    let sa = ctx.scratchArena
    let hop    = 300
    let sr     = Float(24000.0)
    let twoPi  = 2.0 * Float(Double.pi)
    let nf = fb - fa
    let nt = nf * Int64(hop)
    arenaReset(sa)
    let fpfT = tensorNew2d(sa, 9, nf)
    memcpy(fpfT.data!, h.fpf + Int(9 * fa),
           Int(9 * nf) * MemoryLayout<Float>.size)
    let psT = tensorNew2d(sa, 9, nf)
    memcpy(psT.data!, h.ps + Int(9 * fa),
           Int(9 * nf) * MemoryLayout<Float>.size)
    let sT = tensorNew1d(sa, Int64(hop))
    memcpy(sT.data!, h.srange, hop * MemoryLayout<Float>.size)
    let fpf3d = tensorReshape3d(fpfT, 9, nf, 1)
    let s3d   = tensorReshape3d(sT, 1, 1, Int64(hop))
    let fpfX  = tensorRepeatTo(fpf3d, 3, 9, nf, Int64(hop), 1)
    let sX    = tensorRepeatTo(s3d,   3, 9, nf, Int64(hop), 1)
    var within = tensorMul(fpfX, sX)
    within = tensorScale(within, twoPi / sr)
    let ps3d = tensorReshape3d(psT, 9, nf, 1)
    let psX  = tensorRepeatTo(ps3d, 3, 9, nf, Int64(hop), 1)
    var phase = tensorAdd(psX, within)
    phase = tensorCont(tensorPermute(phase, 0, 2, 1, 3))
    phase = tensorReshape2d(phase, 9, nt)
    let sines = tensorScale(tensorSin(phase), 0.1)
    let voiced = tensorNew2d(sa, 1, nt)
    for f in fa..<fb {
        let v: Float = h.f0[Int(f)] > 0.0 ? 1.0 : 0.0
        let dst = voiced.data! + Int(f - fa) * hop
        for s in 0..<hop { dst[s] = v }
    }
    let sinGen = tensorMul(sines, voiced)
    memcpy(outH + Int(9 * fa) * hop, sinGen.data!,
           Int(9 * nt) * MemoryLayout<Float>.size)
}

private func noiseStftSegment(_ ctx: KittensCtx,
                              _ excH: UnsafePointer<Float>,
                              _ tAudio: Int64,
                              _ ua: Int64, _ ub: Int64,
                              _ eps: Float,
                              _ stftH: UnsafeMutablePointer<Float>) {
    let sa = ctx.scratchArena
    let nu = ub - ua
    let W  = (nu - 1) * 5 + 20
    let t0 = ua * 5 - 10
    arenaReset(sa)
    let slice = tensorNew2d(sa, 1, W)
    let e32 = excH
    for i in 0..<W {
        let t = t0 + i
        slice.data![Int(i)] = (t >= 0 && t < tAudio)
            ? e32[Int(t)] : 0.0
    }
    let fr = named(ctx, "stft_fwd.real")
    let fi = named(ctx, "stft_fwd.imag")
    let stR = conv1dNlc(slice, fr!, nil, 5, 0, 1)
    let stI = conv1dNlc(slice, fi!, nil, 5, 0, 1)
    let epsSeg = tensorNew1d(sa, 1)
    epsSeg.data![0] = eps
    let re2 = tensorMul(stR, stR)
    let im2 = tensorMul(stI, stI)
    var mag2 = tensorAdd(re2, im2)
    mag2 = tensorAdd(mag2, epsSeg)
    let mag = tensorSqrt(mag2)
    let phi = tensorAtan2(stI, stR)
    let out = tensorConcat(mag, phi, 0)
    memcpy(stftH + Int(22 * ua), out.data!,
           Int(22 * nu) * MemoryLayout<Float>.size)
}

private func noiseConvBlocks(_ ctx: KittensCtx,
                             _ inH: UnsafePointer<Float>,
                             _ cIn: Int, _ lIn: Int64,
                             _ wName: String, _ bName: String,
                             _ stride: Int, _ pad: Int,
                             _ outH: UnsafeMutablePointer<Float>,
                             _ cOut: Int, _ lOut: Int64) {
    let sa = ctx.scratchArena
    let w = named(ctx, wName)
    let b = named(ctx, bName)
    let K = Int(w!.ne[0])
    var va: Int64 = 0
    while va < lOut {
        var vb = va + Int64(noiseSegV)
        if vb > lOut { vb = lOut }
        let nv = vb - va
        let W  = (nv - 1) * Int64(stride) + Int64(K)
        let u0 = va * Int64(stride) - Int64(pad)
        arenaReset(sa)
        let slice = tensorNew2d(sa, Int64(cIn), W)
        for i in 0..<W {
            let u = u0 + i
            let dst = slice.data! + Int(i) * cIn
            if u >= 0 && u < lIn {
                memcpy(dst, inH + Int(u) * cIn,
                       cIn * MemoryLayout<Float>.size)
            } else {
                memset(dst, 0, cIn * MemoryLayout<Float>.size)
            }
        }
        let y = conv1dNlc(slice, w!, b, stride, 0, 1)
        memcpy(outH + Int(va) * cOut, y.data!,
               Int(nv) * cOut * MemoryLayout<Float>.size)
        va = vb
    }
}

private func buildNoiseContribs(_ ctx: KittensCtx, _ f0Proj: Tensor,
                                _ styleAco: Tensor,
                                _ harmonics: Tensor,
                                _ sRange: Tensor, _ epsT: Tensor,
                                _ F: Int) -> NoiseOuts {
    let sa = ctx.scratchArena
    let tFrames = 2 * F
    let hop     = 300
    let tAudio  = Int64(tFrames) * Int64(hop)
    let sr      = Float(24000.0)
    let twoPi   = 2.0 * Float(Double.pi)
    let f0Repeated = tensorRepeatTo(f0Proj, 2, 9, Int64(tFrames), 1, 1)
    let harm2d = tensorReshape2d(harmonics, 9, 1)
    let f0PerFrame = tensorMul(f0Repeated, harm2d)
    let stepNlc = tensorScale(f0PerFrame, Float(hop) / sr)
    let stepNcl = tensorCont(tensorTranspose(stepNlc))
    let cs      = tensorCumsum(stepNcl, 0)
    var psNcl   = tensorSub(cs, stepNcl)
    psNcl = tensorScale(psNcl, twoPi)
    let phaseStartNlc = tensorCont(tensorTranspose(psNcl))
    var pFpf = save(f0PerFrame)
    var pPs  = save(phaseStartNlc)
    var pF0  = save(f0Proj)
    var pSr  = save(sRange)
    var pSty = save(styleAco)
    let eps = epsT.data![0]
    let h = NoiseHosts(fpf: pFpf.data!, ps: pPs.data!,
                       f0: pF0.data!, srange: pSr.data!,
                       tf: Int64(tFrames))
    let singenH = UnsafeMutablePointer<Float>
        .allocate(capacity: Int(9 * tAudio))
    var fa: Int64 = 0
    while fa < Int64(tFrames) {
        var fb = fa + Int64(noiseSegF)
        if fb > Int64(tFrames) { fb = Int64(tFrames) }
        noiseSingenSegment(ctx, h, fa, fb, singenH)
        fa = fb
    }
    arenaReset(sa)
    let sinGen = tensorWrap2d(sa, singenH, 9, tAudio)
    let lLinW = named(ctx, "l_lin.weight")
    let lLinB = named(ctx, "l_lin.bias")
    let mixed = tensorAdd(
        tensorMulMat(lLinW!, sinGen), lLinB!)
    let excitation = tensorTanh(mixed)
    let excH = UnsafeMutablePointer<Float>
        .allocate(capacity: Int(tAudio))
    memcpy(excH, excitation.data!,
           Int(tAudio) * MemoryLayout<Float>.size)
    singenH.deallocate()
    let l1 = tensorConvOutLen(tAudio, 20, 5, 10, 1)
    let stftH = UnsafeMutablePointer<Float>
        .allocate(capacity: Int(22 * l1))
    var ua: Int64 = 0
    while ua < l1 {
        var ub = ua + Int64(noiseSegU)
        if ub > l1 { ub = l1 }
        noiseStftSegment(ctx, excH, tAudio, ua, ub, eps, stftH)
        ua = ub
    }
    excH.deallocate()
    let nc0W = named(ctx, "nc0.weight")
    let nc1W = named(ctx, "nc1.weight")
    let c0 = Int(nc0W!.ne[2])
    let c1 = Int(nc1W!.ne[2])
    let l0 = tensorConvOutLen(l1, Int(nc0W!.ne[0]), 6, 3, 1)
    let nc0H = hostAlloc(c0 * Int(l0))
    noiseConvBlocks(ctx, stftH, 22, l1, "nc0.weight", "nc0.bias",
                    6, 3, nc0H, c0, l0)
    arenaReset(sa)
    var pNc0 = saveptWrap(nc0H, Int64(c0), l0)
    var sty = restore(sa, pSty)
    let nr0 = buildHifiBlock(ctx, "nr0", &pNc0, sty, true)
    let nc1H = hostAlloc(c1 * Int(l1))
    noiseConvBlocks(ctx, stftH, 22, l1, "nc1.weight", "nc1.bias",
                    1, 0, nc1H, c1, l1)
    stftH.deallocate()
    arenaReset(sa)
    var pNc1 = saveptWrap(nc1H, Int64(c1), l1)
    sty = restore(sa, pSty)
    let nr1 = buildHifiBlock(ctx, "nr1", &pNc1, sty, true)
    saveptFree(&pFpf)
    saveptFree(&pPs)
    saveptFree(&pF0)
    saveptFree(&pSr)
    saveptFree(&pSty)
    let r = NoiseOuts(nr0: nr0, nr1: nr1)
    return r
}

private let istftSegL = 4096

struct IstftStream {
    var r2: UnsafePointer<Float>
    var r3: UnsafePointer<Float>
    var C: Int
    var L: Int64
}

private func istftMixSegment(_ sa: Arena, _ s: IstftStream,
                             _ lo: Int64, _ hi: Int64) -> Tensor {
    let seg = tensorNew2d(sa, Int64(s.C), hi - lo)
    let n   = Int(hi - lo) * s.C
    let off = Int(lo) * s.C
    let dst = seg.data!
    let r2 = s.r2 + off
    let r3 = s.r3 + off
    for i in 0..<n {
        let v = 0.5 * (r2[i] + r3[i])
        let w = 0.1 * v
        dst[i] = v > w ? v : w
    }
    return seg
}

struct IstftSpectra {
    var re: Tensor
    var im: Tensor
}

private func istftSegmentSpectra(_ seg: Tensor, _ cpW: Tensor,
                                 _ cpB: Tensor?) -> IstftSpectra {
    let y = conv1dNlc(seg, cpW, cpB, 1, -1, 1)
    let W = y.ne[1]
    let magLog = tensorCont(
        tensorView2d(y, 11, W, Int(y.nb[1]), 0))
    let phase = tensorCont(
        tensorView2d(y, 11, W, Int(y.nb[1]), Int(11 * y.nb[0])))
    let mag   = tensorExp(magLog)
    let inner = tensorSin(phase)
    let r = IstftSpectra(re: tensorMul(mag, tensorCos(inner)),
                         im: tensorMul(mag, tensorSin(inner)))
    return r
}

private func istftFrameWindow(_ spec: Tensor, _ lo: Int64,
                              _ wf: Int64, _ b: Int64) -> Tensor {
    return tensorCont(tensorView2d(spec, 11, b - wf,
                                   Int(spec.nb[1]),
                                   Int((wf - lo) * spec.nb[1])))
}

private func buildIstftTail(_ ctx: KittensCtx, _ r2: inout HifiOut,
                            _ r3: inout HifiOut) -> Tensor {
    let sa = ctx.scratchArena
    let cpW = named(ctx, "gen.cp.weight")
    let cpB = named(ctx, "gen.cp.bias")
    let sbR = named(ctx, "stft_bwd.real")
    let sbI = named(ctx, "stft_bwd.imag")
    let padCp = (Int(cpW!.ne[0]) - 1) / 2
    let kb    = Int(sbR!.ne[0])
    let frOv  = (kb + 3) / 5
    let s = IstftStream(r2: UnsafePointer(r2.data!),
                        r3: UnsafePointer(r3.data!),
                        C: r2.C, L: r2.L)
    let tFull = (s.L - 1) * 5 + Int64(kb)
    let audioH = UnsafeMutablePointer<Float>
        .allocate(capacity: Int(tFull))
    var a: Int64 = 0
    while a < s.L {
        var b = a + Int64(istftSegL)
        if b > s.L { b = s.L }
        var lo = a - Int64(frOv + padCp)
        if lo < 0 { lo = 0 }
        var hi = b + Int64(padCp)
        if hi > s.L { hi = s.L }
        var wf = a - Int64(frOv)
        if wf < 0 { wf = 0 }
        arenaReset(sa)
        let seg = istftMixSegment(sa, s, lo, hi)
        let sp = istftSegmentSpectra(seg, cpW!, cpB)
        let reW = istftFrameWindow(sp.re, lo, wf, b)
        let imW = istftFrameWindow(sp.im, lo, wf, b)
        let ar = convTranspose1dNlc(reW, sbR!, nil, 5, 0)
        let ai = convTranspose1dNlc(imW, sbI!, nil, 5, 0)
        let chunk = tensorSub(ar, ai)
        let g0 = a * 5
        let g1 = (b == s.L) ? tFull : b * 5
        memcpy(audioH + Int(g0), chunk.data! + Int((a - wf) * 5),
               Int(g1 - g0) * MemoryLayout<Float>.size)
        a = b
    }
    hostFree(r2.data); r2.data = nil
    hostFree(r3.data); r3.data = nil
    let trim = ctx.arch.istftTrim
    assert(tFull > Int64(2 * trim))
    arenaReset(sa)
    let out = tensorNew1d(sa, tFull - Int64(2 * trim))
    memcpy(out.data!, audioH + trim,
           Int(tFull - Int64(2 * trim)) * MemoryLayout<Float>.size)
    audioH.deallocate()
    return out
}

private func hifiMaterialize(_ sa: Arena,
                             _ h: inout HifiOut) -> Tensor {
    let t = tensorNew2d(sa, Int64(h.C), h.L)
    memcpy(t.data!, h.data!,
           h.C * Int(h.L) * MemoryLayout<Float>.size)
    hostFree(h.data)
    h.data = nil
    return t
}

private func buildGenerator(_ ctx: KittensCtx, _ decOut: Tensor,
                            _ nr0: inout HifiOut,
                            _ nr1: inout HifiOut,
                            _ styleAco: Tensor) -> Tensor {
    let u0W = named(ctx, "gen.u0.weight")
    let u0B = named(ctx, "gen.u0.bias")
    let u1W = named(ctx, "gen.u1.weight")
    let u1B = named(ctx, "gen.u1.bias")
    let sa = ctx.scratchArena
    var pSty = save(styleAco)
    var x = tensorLeakyRelu(decOut, 0.1)
    x = convTranspose1dNlc(x, u0W!, u0B, 10, 5)
    x = tensorAdd(x, hifiMaterialize(sa, &nr0))
    var px = save(x)
    var r0 = buildHifiBlock(ctx, "gen.r0", &px, styleAco, false)
    var sty = restore(sa, pSty)
    var r1 = buildHifiBlock(ctx, "gen.r1", &px, sty, true)
    arenaReset(sa)
    x = tensorAdd(hifiMaterialize(sa, &r0), hifiMaterialize(sa, &r1))
    x = tensorScale(x, 0.5)
    x = tensorLeakyRelu(x, 0.1)
    x = convTranspose1dNlc(x, u1W!, u1B, 6, 3)
    px = save(x)
    arenaReset(sa)
    x = restore(sa, px)
    saveptFree(&px)
    x = reflectionPadLeft(x, 1)
    x = tensorAdd(x, hifiMaterialize(sa, &nr1))
    sty = restore(sa, pSty)
    px = save(x)
    var r2 = buildHifiBlock(ctx, "gen.r2", &px, sty, false)
    sty = restore(sa, pSty)
    var r3 = buildHifiBlock(ctx, "gen.r3", &px, sty, true)
    saveptFree(&px)
    saveptFree(&pSty)
    let out = buildIstftTail(ctx, &r2, &r3)
    return out
}

private func fadeIn(_ x: UnsafeMutablePointer<Float>, _ n: Int,
                    _ fade: Int) {
    if fade > 0 && fade <= n {
        for i in 0..<fade {
            let t = Float(i) / Float(fade - 1 > 0 ? fade - 1 : 1)
            x[i] *= 0.5 - 0.5 * cosf(Float(Double.pi) * t)
        }
    }
}

private func fadeOut(_ x: UnsafeMutablePointer<Float>, _ n: Int,
                     _ fade: Int) {
    if fade > 0 && fade <= n {
        let start = n - fade
        for i in 0..<fade {
            let t = Float(i) / Float(fade - 1 > 0 ? fade - 1 : 1)
            x[start + i] *= 0.5 + 0.5 * cosf(Float(Double.pi) * t)
        }
    }
}

struct KittensAudio {
    var samples: [Float] = []
    var nSamples: Int = 0
}

func kittensSynthesize(_ ctx: KittensCtx?,
                              _ phonemes: [Int32], _ nPhonemes: Int,
                              _ style256: [Float],
                              _ speed: Float) -> KittensAudio {
    var out = KittensAudio()
    let usable = ctx != nil && nPhonemes > 0 && speed > 0.0
    if usable {
        let ctx = ctx!
        let L = nPhonemes
        var F = 0
        let A = ctx.arch
        let sa = ctx.scratchArena
        arenaSetActive(sa)
        var durs: UnsafeMutablePointer<Int32>? = nil
        var prosodyH: UnsafeMutablePointer<Float>? = nil
        var textH: UnsafeMutablePointer<Float>? = nil
        var durH: UnsafeMutablePointer<Float>? = nil
        var prosodyLrH: UnsafeMutablePointer<Float>? = nil
        var textLrH: UnsafeMutablePointer<Float>? = nil
        var f0pH: UnsafeMutablePointer<Float>? = nil
        var npH: UnsafeMutablePointer<Float>? = nil
        var decH: UnsafeMutablePointer<Float>? = nil
        var audioBuf: UnsafeMutablePointer<Float>? = nil
        arenaReset(sa)
        do {
            var posIds  = [Int32](repeating: 0, count: L)
            var typeIds = [Int32](repeating: 0, count: L)
            let maxP = A.maxPos > 0 ? A.maxPos - 1 : 0
            for i in 0..<L {
                posIds[i]  = Int32(i <= maxP ? i : maxP)
                typeIds[i] = 0
            }
            let stylePr = tensorNew1d(sa, Int64(A.styleDim))
            _ = style256.withUnsafeBufferPointer { src in
                memcpy(stylePr.data!, src.baseAddress! + A.styleDim,
                       MemoryLayout<Float>.size * A.styleDim)
            }
            let h0 = tensorNew1d(sa, Int64(A.lstmHidden))
            let c0 = tensorNew1d(sa, Int64(A.lstmHidden))
            memset(h0.data!, 0, MemoryLayout<Float>.size * A.lstmHidden)
            memset(c0.data!, 0, MemoryLayout<Float>.size * A.lstmHidden)
            let ts = buildTextstage(ctx, L, phonemes, posIds, typeIds,
                                    stylePr, h0, c0)
            prosodyH = hostAlloc(256 * L)
            textH    = hostAlloc(128 * L)
            durH     = hostAlloc(50 * L)
            memcpy(prosodyH!, ts.prosody256.data!,
                   MemoryLayout<Float>.size * 256 * L)
            memcpy(textH, ts.text.data!,
                   MemoryLayout<Float>.size * 128 * L)
            memcpy(durH, ts.durSig.data!,
                   MemoryLayout<Float>.size * 50 * L)
        }
        durs = UnsafeMutablePointer<Int32>.allocate(capacity: L)
        F = 0
        for i in 0..<L {
            var sum: Float = 0.0
            for j in 0..<50 {
                sum += durH![i * 50 + j]
            }
            var d = Int(lrintf(sum / speed))
            if d < 1 { d = 1 }
            durs![i] = Int32(d)
            F += d
        }
        prosodyLrH = hostAlloc(256 * F)
        textLrH    = hostAlloc(128 * F)
        memset(prosodyLrH!, 0, 256 * F * MemoryLayout<Float>.size)
        memset(textLrH!, 0, 128 * F * MemoryLayout<Float>.size)
        do {
            var t = 0
            for l in 0..<L {
                let d = Int(durs![l])
                for _ in 0..<d {
                    memcpy(prosodyLrH! + t * 256,
                           prosodyH!   + l * 256,
                           MemoryLayout<Float>.size * 256)
                    memcpy(textLrH!    + t * 128,
                           textH!      + l * 128,
                           MemoryLayout<Float>.size * 128)
                    t += 1
                }
            }
        }
        hostFree(prosodyH); prosodyH = nil
        hostFree(textH);    textH    = nil
        hostFree(durH);     durH     = nil
        durs?.deallocate(); durs     = nil
        arenaReset(sa)
        do {
            let prosodyLr = tensorNew2d(sa, 256, Int64(F))
            memcpy(prosodyLr.data!, prosodyLrH!,
                   MemoryLayout<Float>.size * 256 * F)
            let stylePr = tensorNew1d(sa, Int64(A.styleDim))
            _ = style256.withUnsafeBufferPointer { src in
                memcpy(stylePr.data!, src.baseAddress! + A.styleDim,
                       MemoryLayout<Float>.size * A.styleDim)
            }
            let h0 = tensorNew1d(sa, Int64(A.lstmHidden))
            let c0 = tensorNew1d(sa, Int64(A.lstmHidden))
            memset(h0.data!, 0, MemoryLayout<Float>.size * A.lstmHidden)
            memset(c0.data!, 0, MemoryLayout<Float>.size * A.lstmHidden)
            let g = buildGenfront(ctx, prosodyLr, stylePr, h0, c0, F)
            f0pH = hostAlloc(2 * F)
            npH  = hostAlloc(2 * F)
            memcpy(f0pH!, g.f0Proj.data!,
                   MemoryLayout<Float>.size * 2 * F)
            memcpy(npH!, g.nProj.data!,
                   MemoryLayout<Float>.size * 2 * F)
        }
        hostFree(prosodyLrH); prosodyLrH = nil
        arenaReset(sa)
        do {
            let textLr = tensorNew2d(sa, 128, Int64(F))
            let f0pT   = tensorNew2d(sa, 1, Int64(2 * F))
            let npT    = tensorNew2d(sa, 1, Int64(2 * F))
            let styleA = tensorNew1d(sa, 128)
            memcpy(textLr.data!, textLrH!,
                   MemoryLayout<Float>.size * 128 * F)
            memcpy(f0pT.data!, f0pH!, MemoryLayout<Float>.size * 2 * F)
            memcpy(npT.data!,  npH!,  MemoryLayout<Float>.size * 2 * F)
            _ = style256.withUnsafeBufferPointer { src in
                memcpy(styleA.data!, src.baseAddress!,
                       MemoryLayout<Float>.size * 128)
            }
            let decOut = buildDecoder(ctx, textLr, f0pT, npT, styleA)
            decH = hostAlloc(256 * 2 * F)
            memcpy(decH!, decOut.data!,
                   MemoryLayout<Float>.size * 256 * 2 * F)
        }
        hostFree(textLrH); textLrH = nil
        hostFree(npH);     npH     = nil
        var nz = NoiseOuts(nr0: HifiOut(), nr1: HifiOut())
        arenaReset(sa)
        do {
            let f0T   = tensorNew2d(sa, 1, Int64(2 * F))
            let styA  = tensorNew1d(sa, 128)
            let harm  = tensorNew1d(sa, 9)
            let sRng  = tensorNew1d(sa, 300)
            let epsT  = tensorNew1d(sa, 1)
            memcpy(f0T.data!, f0pH!, MemoryLayout<Float>.size * 2 * F)
            _ = style256.withUnsafeBufferPointer { src in
                memcpy(styA.data!, src.baseAddress!,
                       MemoryLayout<Float>.size * 128)
            }
            for i in 0..<9   { harm.data![i] = Float(i + 1) }
            for i in 0..<300 { sRng.data![i] = Float(i) }
            epsT.data![0] = 1e-9
            nz = buildNoiseContribs(ctx, f0T, styA, harm, sRng,
                                    epsT, F)
        }
        hostFree(f0pH); f0pH = nil
        var n = 0
        arenaReset(sa)
        do {
            let decT = tensorNew2d(sa, 256, Int64(2 * F))
            let styA = tensorNew1d(sa, 128)
            memcpy(decT.data!, decH!,
                   MemoryLayout<Float>.size * 256 * 2 * F)
            _ = style256.withUnsafeBufferPointer { src in
                memcpy(styA.data!, src.baseAddress!,
                       MemoryLayout<Float>.size * 128)
            }
            hostFree(decH); decH = nil
            let audioT = buildGenerator(ctx, decT, &nz.nr0, &nz.nr1,
                                        styA)
            let tAudio = Int(audioT.ne[0])
            n = tAudio
            let tailDrop = 3 * 600
            if n > tailDrop { n -= tailDrop }
            audioBuf = hostAlloc(n)
            memcpy(audioBuf!, audioT.data!,
                   MemoryLayout<Float>.size * n)
        }
        fadeIn (audioBuf!, n,  72)
        fadeOut(audioBuf!, n, 960)
        out.samples = [Float](UnsafeBufferPointer(start: audioBuf!,
                                                  count: n))
        out.nSamples = n
        hostFree(audioBuf)
        arenaSetActive(nil)
        arenaReset(sa)
    }
    return out
}
