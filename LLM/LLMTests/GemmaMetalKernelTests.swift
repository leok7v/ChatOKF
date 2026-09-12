import Foundation
import Metal
import Testing
@testable import LLM

// Op-by-op gate for the gemma-4 Metal kernels against the SIMD engine, which is
// itself gated against the Python reference dump.
private struct MetalHarness {
    let model: Gemma4Model
    let ctx: MetalContext

    init(_ path: String) throws {
        model = try Gemma4Model(path: path)
        ctx = try MetalContext(model.gguf)
    }

    func off(_ t: GGUFTensor) -> WeightRef {
        ctx.window(UInt64(t.base - model.gguf.map))
    }

    // Encode one kernel, run it, and hand back the output buffer's floats.
    func run(_ n: Int, _ body: (MetalEnc) -> Void) -> [Float] {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        body(MetalEnc(ctx: ctx, e: e))
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        return []
    }
}

// Cosine of two equal-length vectors; 1.0 is identical direction.
private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<min(a.count, b.count) {
        dot += Double(a[i]) * Double(b[i])
        na += Double(a[i]) * Double(a[i])
        nb += Double(b[i]) * Double(b[i])
    }
    return dot / ((na.squareRoot() * nb.squareRoot()) + 1e-30)
}

private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    var m: Float = 0
    for i in 0..<min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
    return m
}

// A deterministic activation vector: the kernels are linear in x, so a fixed
// pseudo-random spread exercises every sign and magnitude a real hidden does.
private func activations(_ n: Int) -> [Float] {
    var out = [Float](repeating: 0, count: n)
    var s: UInt64 = 0x9E3779B97F4A7C15
    for i in 0..<n {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        out[i] = Float(Int32(truncatingIfNeeded: s >> 33)) / Float(1 << 30)
    }
    return out
}

struct GemmaMetalKernelTests {
    // Q4_0 carries the attention projections, layers 0-14's MLP and the
    @Test(needsGemmaWeights) func q4GemvMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let w = h.model.layers[0].wq
        #expect(w.type == .q4_0)
        let K = w.dims[0], M = w.dims[1]
        let x = activations(K)
        var want = [Float](repeating: 0, count: M)
        GQ.matvec(w, x: x, out: &want)

        let xb = h.ctx.makeF32(x)
        let ob = h.ctx.makeF32(M)
        _ = h.run(M) { f in
            f.gemv(w, x: xb, out: ob, off: h.off(w))
        }
        let got = Array(ob.f32(M))
        let cos = cosine(want, got)
        #expect(cos >= 0.99999,
                "q4_0_gemv cosine \(cos), maxAbs \(maxAbsDiff(want, got))")
    }

    // The Q2_0 path must keep working through the newly type-dispatched gemv:
    @Test(needsGemmaWeights) func q2GemvStillMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let w = h.model.layers[h.model.cfg.nLayer - 1].ffnDown
        #expect([GGUFType.q2_0, .q4_0].contains(w.type),
                "unknown MLP bit plan \(w.type)")
        if w.type == .q2_0 {
            let K = w.dims[0], M = w.dims[1]
            let x = activations(K)
            var want = [Float](repeating: 0, count: M)
            GQ.matvec(w, x: x, out: &want)

            let xb = h.ctx.makeF32(x)
            let ob = h.ctx.makeF32(M)
            _ = h.run(M) { f in
                f.gemv(w, x: xb, out: ob, off: h.off(w))
            }
            let got = Array(ob.f32(M))
            let cos = cosine(want, got)
            #expect(cos >= 0.99999, "q2_0_gemv cosine \(cos)")
        }
    }

    // The embedding gathers.
    @Test(needsGemmaWeights) func dequantRowMatchesSIMDForBothTables() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let c = h.model.cfg
        // Only a checkpoint that HAS a per-layer table can be asked about its
        // split, so the fixture's own is required rather than assumed.
        let ple = try #require(h.model.perLayerEmbd)
        #expect([GGUFType.q2_0, .q4_0].contains(h.model.tokEmbd.type),
                "unknown token_embd bit plan \(h.model.tokEmbd.type)")
        #expect(ple.type == .q4_0)
        let cases = [(h.model.tokEmbd, c.nEmbd, "token_embd"),
                     (ple, c.nLayer * c.perLayerDim,
                      "per_layer_token_embd")]
        for (t, n, label) in cases {
            let row = 1234
            var want = [Float](repeating: 0, count: n)
            GQ.dequantSpan(t, row: row, from: 0, count: n, into: &want)

            let ob = h.ctx.makeF32(n)
            let rowOff = h.off(t) + UInt64(row * GQ.rowBytes(t))
            _ = h.run(n) { f in
                f.dequantRow(weightOff: rowOff, out: ob, n: n,
                             type: t.type)
            }
            let got = Array(ob.f32(n))
            let diff = maxAbsDiff(want, got)
            #expect(diff == 0,
                    "\(label) dequant is a pure unpack: maxAbs \(diff)")
        }
    }

    // Every gemma norm is BF16; reading one as f32 fuses two weights into one
    // garbage float, so this is the case that fails loudly rather than subtly.
    @Test(needsGemmaWeights) func bf16RmsnormMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let t = h.model.gguf.tensor("blk.0.attn_norm.weight")
        #expect(t.type == .bf16)
        let n = h.model.cfg.nEmbd
        let x = activations(n)
        let w = Dense.floats(t)
        let want = GK.rmsnorm(x, w, h.model.cfg.eps)

        let xb = h.ctx.makeF32(x)
        let ob = h.ctx.makeF32(n)
        _ = h.run(n) { f in
            f.rmsnormBF16(x: xb, weightOff: h.off(t), out: ob, n: n,
                          eps: h.model.cfg.eps)
        }
        let got = Array(ob.f32(n))
        let cos = cosine(want, got)
        let diff = maxAbsDiff(want, got)
        #expect(cos >= 0.99999,
                "rmsnorm_bf16 cosine \(cos) maxAbs \(diff)")
    }

    // v_norm leaves no tensor in the checkpoint, so nothing but the source
    // says it exists -- and it divides by the RMS, not the L2 norm.
    @Test(needsGemmaWeights) func noWeightRmsnormMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let hd = h.model.cfg.headDimSliding
        let rows = 4
        var want = activations(hd * rows)
        let x = want
        GK.rmsnormRowsNoWeight(&want, d: hd, rows: rows,
                               eps: h.model.cfg.eps)

        let xb = h.ctx.makeF32(x)
        _ = h.run(hd * rows) { f in
            f.rmsnormRowsNoWeight(x: xb, xoff: 0, d: hd, rows: rows,
                                  eps: h.model.cfg.eps)
        }
        let got = Array(xb.f32(hd * rows))
        let cos = cosine(want, got)
        #expect(cos >= 0.99999, "rmsnorm_rows_noweight cosine \(cos)")
    }

    // gemma's 7 full-attention layers are head_dim 512, which the paged
    @Test(needsGemmaWeights) func pagedAttentionMatchesReferenceAt256And512() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        for (hd, T) in [(256, 40), (512, 40), (256, 640), (512, 640)] {
            let nH = 2, nKV = 1, P = 32
            let pool = MetalKVPool(device: h.ctx.device, P: P,
                                   kvDim: hd * nKV)
            pool.appendBatch(T)
            // Fill the pages directly: this gate is about the read path.
            let src = activations(T * hd * nKV * 2)
            var kHalf = [Float](repeating: 0, count: T * hd)
            var vHalf = [Float](repeating: 0, count: T * hd)
            for t in 0..<T {
                let page = pool.kPages[t / P], vpage = pool.vPages[t / P]
                let kp = page.contents()
                    .assumingMemoryBound(to: Float16.self)
                let vp = vpage.contents()
                    .assumingMemoryBound(to: Float16.self)
                for i in 0..<hd {
                    let kv = src[t * hd + i]
                    let vv = src[T * hd + t * hd + i]
                    kp[(t % P) * hd + i] = Float16(kv)
                    vp[(t % P) * hd + i] = Float16(vv)
                    kHalf[t * hd + i] = Float(Float16(kv))
                    vHalf[t * hd + i] = Float(Float16(vv))
                }
            }
            pool.refreshTable()

            let q = activations(hd * nH)
            var want = [Float](repeating: 0, count: hd * nH)
            for head in 0..<nH {
                var sc = [Float](repeating: 0, count: T)
                for t in 0..<T {
                    var s: Float = 0
                    for i in 0..<hd {
                        s += q[head * hd + i] * kHalf[t * hd + i]
                    }
                    sc[t] = s
                }
                GK.softmaxInPlace(&sc, T)
                for t in 0..<T {
                    for i in 0..<hd {
                        want[head * hd + i] += sc[t] * vHalf[t * hd + i]
                    }
                }
            }

            let qb = h.ctx.makeF32(q)
            let ob = h.ctx.makeF32(hd * nH)
            let gate = h.ctx.makeF32(hd * nH)
            _ = h.run(hd * nH) { f in
                f.attnPaged(q: qb, kAddr: pool.kAddr, vAddr: pool.vAddr,
                            pages: pool.residentPages, gate: gate,
                            out: ob, hd: hd, nH: nH, nKV: nKV, T: T,
                            kvDim: hd * nKV, P: P, scale: 1, gated: 0)
            }
            let got = Array(ob.f32(hd * nH))
            let cos = cosine(want, got)
            let diff = maxAbsDiff(want, got)
            #expect(cos >= 0.9999,
                    "attn_head hd=\(hd) cosine \(cos) maxAbs \(diff)")
        }
    }

    // Rollback is index-based, so this proves the indexes are the whole story:
    // decoding, rewinding, and decoding again must replay the SAME tokens.
    @Test(needsGemmaWeights) func bookmarkRewindReplaysIdentically() throws {
        let path = try #require(gemmaGgufPath)
        let model = try Gemma4Model(path: path)
        let eng = try Gemma4MetalEngine(model)
        eng.reset()
        var next = eng.extend([2, 106, 1645, 236764, 1653])
        let mark = eng.bookmark()
        var first: [Int32] = []
        for _ in 0..<6 {
            first.append(next)
            next = eng.decode(next)
        }
        eng.restore(mark)
        var again: [Int32] = []
        var n2 = first.first ?? next
        for _ in 0..<6 {
            again.append(n2)
            n2 = eng.decode(n2)
        }
        #expect(first == again,
                "rewind replay diverged: \(first) vs \(again)")
    }

    @Test(needsGemmaMTPWeights) func adapterReachesTheEngineLevers() async throws {
        let path = try #require(gemmaMTPGgufPath)
        let backend = try GemmaChat(ggufPath: path).metalBackend()
        backend.requestStop()
        #expect(backend.shouldStop())
        await backend.reset()
        #expect(!backend.shouldStop())
        #expect(await backend.queuedCount() == 0)
        #expect(await backend.supportsSoftTokens() == backend.ctx.matrixUnits)
    }

    @Test(needsGemmaMTPWeights) func extendDropsTheSpeculativeQueue() throws {
        let path = try #require(gemmaMTPGgufPath)
        let model = try Gemma4Model(path: path)
        let eng = try Gemma4MetalEngine(model)
        #expect(eng.hasAssist)
        eng.reset()
        var next = eng.extend([2, 106, 1645, 236764, 1653])
        var steps = 0
        while eng.queued == 0 && steps < 64 {
            let cycles = eng.specCycles
            let accepted = eng.specAccepted
            next = eng.decode(next)
            if eng.specCycles == cycles + 1 {
                #expect(eng.queued == max(eng.specAccepted - accepted - 1, 0),
                        "the bonus token is counted as committed")
            }
            steps += 1
        }
        #expect(eng.queued > 0, "no cycle accepted a draft in \(steps) steps")
        let held = eng.pos
        let cycles = eng.specCycles
        _ = eng.extend([1645, 236764, 1653])
        #expect(eng.queued == 0, "the queue survived an extend")
        #expect(eng.pos == held + 3)
        #expect(eng.specCycles == cycles, "the extend ran a spec cycle")
    }

    // The batched GEMM per weight type, against the SIMD matvec row by row.
    @Test(needsGemmaWeights) func batchedGemmMatchesSIMDPerType() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let cases = [
            ("q4_0 lconv_start", h.model.gguf.tensor(
                "a.blk.0.lconv_start.weight")),
            ("q2_0 ffw1_up", h.model.gguf.tensor(
                "a.blk.0.ffw1_up.weight")),
            ("q8_0 vision q", h.model.gguf.tensor(
                "v.blk.0.attn_q.weight")),
        ]
        for (label, w) in cases {
            let K = w.dims[0], M = w.dims[1]
            let rows = 8
            let x = activations(rows * K)
            var want = [Float](repeating: 0, count: rows * M)
            var one = [Float](repeating: 0, count: M)
            for r in 0..<rows {
                let slice = Array(x[(r * K)..<((r + 1) * K)])
                GQ.matvec(w, x: slice, out: &one)
                for m in 0..<M { want[r * M + m] = one[m] }
            }
            let xb = h.ctx.makeF32(x)
            let ob = h.ctx.makeF32(rows * M)
            _ = h.run(rows * M) { f in
                f.gemm(w, X: xb, out: ob, off: h.off(w), N: rows)
            }
            let got = Array(ob.f32(rows * M))
            let cos = cosine(want, got)
            #expect(cos >= 0.9999, "\(label) gemm cosine \(cos)")
        }
    }

    // The soft-token entry point on both engines.
    @Test(needsGemmaWeights) func softTokenForwardMatchesAcrossEngines() throws {
        let path = try #require(gemmaGgufPath)
        let model = try Gemma4Model(path: path)
        let cpu = Gemma4Engine(model)
        let gpu = try Gemma4MetalEngine(model)
        let e = model.cfg.nEmbd
        // Tower features are order-1, not order-1e-3; a feature small
        // enough to vanish under the residual would prove nothing.
        let one = activations(e).map { v in v * 4 }
        let two = activations(2 * e).suffix(e).map { v in v * 4 }
        // ONE cpu forward: `swift test` builds Debug, where a 35-layer SIMD
        cpu.reset()
        gpu.reset()
        let hCpu = cpu.forward(embedding: one, pos: 0)
        gpu.forward(embedding: one, pos: 0)
        let hGpu = gpu.hidden()
        let across = cosine(hCpu, hGpu)
        #expect(across >= 0.95,
                "soft token cpu vs gpu hidden cosine \(across)")

        // A second feature must move the hidden.
        gpu.reset()
        gpu.forward(embedding: Array(two), pos: 0)
        let moved = cosine(hGpu, gpu.hidden())
        let why = "a different feature left the hidden at \(moved); "
            + "the soft token is being ignored"
        #expect(moved < 0.999, "\(why)")
    }

    // The trained activation clamp.
    @Test(needsGemmaWeights) func srqMatchesSIMDBitExactly() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let g = h.model.gguf
        let sides = ["a.blk.0.attn_q.weight", "a.blk.0.lconv_end.weight",
                     "v.blk.0.attn_q.weight"]
            .map { name in SRQ(g, name) }
            .flatMap { s in [s.input, s.output] }
            .filter { s in s.active }
        #expect(!sides.isEmpty,
                "this file carries neither gemma4.srq.* nor gemma4.clamp.*")
        let n = 4096
        for s in sides {
            let span = s.scale != 0 ? s.scale * 200
                                    : max(abs(s.lo), abs(s.hi)) * 2
            let x = activations(n).map { v in v * span }
            var want = x
            SRQ.apply(&want, s)

            let bIn = h.ctx.makeF32(x)
            let bSrc = h.ctx.makeF32(x)
            let bDst = h.ctx.makeF32(n)
            _ = h.run(n) { f in
                f.srqInPlace(x: bIn, n: n, s: s)
                f.srqTo(src: bSrc, dst: bDst, n: n, s: s)
            }
            let gotIn = Array(bIn.f32(n))
            let gotTo = Array(bDst.f32(n))
            #expect(maxAbsDiff(want, gotIn) == 0,
                    "srq_inplace \(s) maxAbs \(maxAbsDiff(want, gotIn))")
            #expect(maxAbsDiff(want, gotTo) == 0,
                    "srq_to \(s) maxAbs \(maxAbsDiff(want, gotTo))")
            let bound = want.filter { v in
                s.scale != 0 ? abs(v) >= 127 * s.scale
                             : (v == s.lo || v == s.hi)
            }.count
            #expect(bound > 0, "the clamp range was never reached")
        }
    }

    // The macaron residual is a HALF-weight add, so a kernel that adds at
    // full weight still produces a finite tower that drifts everywhere.
    @Test(needsGemmaWeights) func audioSiluAndFmaMatchSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let n = 2048
        let x = activations(n)
        let y = activations(n).map { v in v * 0.5 }
        var wantSilu = [Float](repeating: 0, count: n)
        var wantFma = [Float](repeating: 0, count: n)
        for i in 0..<n {
            wantSilu[i] = x[i] / (1 + expf(-x[i]))
            wantFma[i] = x[i] + y[i] * 0.5
        }
        let bSilu = h.ctx.makeF32(x)
        let bFma = h.ctx.makeF32(x)
        let bY = h.ctx.makeF32(y)
        _ = h.run(n) { f in
            f.siluInPlace(x: bSilu, n: n)
            f.fmaInPlace(x: bFma, y: bY, n: n, s: 0.5)
        }
        let cs = cosine(wantSilu, Array(bSilu.f32(n)))
        let cf = cosine(wantFma, Array(bFma.f32(n)))
        #expect(cs >= 0.99999, "silu_inplace cosine \(cs)")
        #expect(cf >= 0.99999, "fma_inplace cosine \(cf)")
    }

    // The GLU halves a 2e-wide row and the depthwise conv is CAUSAL, so it pads
    @Test(needsGemmaWeights) func audioGluAndDepthwiseMatchSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let a = try Gemma4Audio(h.model)
        let e = a.cfg.embd, k = a.cfg.convKernel
        let n = 50
        let wide = activations(n * 2 * e)
        let dw = Dense.floats(
            h.model.gguf.tensor("a.blk.0.lconv_dw.weight"))
        var wantGlu = [Float](repeating: 0, count: n * e)
        for t in 0..<n {
            for c in 0..<e {
                let lo = wide[t * 2 * e + c]
                let hi = wide[t * 2 * e + e + c]
                wantGlu[t * e + c] = lo / (1 + expf(-hi))
            }
        }
        var wantConv = [Float](repeating: 0, count: n * e)
        for t in 0..<n {
            for c in 0..<e {
                var acc: Float = 0
                for j in 0..<k {
                    let src = t - (k - 1) + j
                    if src >= 0 {
                        acc += wantGlu[src * e + c] * dw[c * k + j]
                    }
                }
                wantConv[t * e + c] = acc
            }
        }
        let bWide = h.ctx.makeF32(wide)
        let bGlu = h.ctx.makeF32(n * e)
        let bDw = h.ctx.makeF32(dw)
        let bConv = h.ctx.makeF32(n * e)
        _ = h.run(n * e) { f in
            f.gluRows(src: bWide, dst: bGlu, rows: n, width: e)
            f.depthwiseCausal(x: bGlu, w: bDw, out: bConv, rows: n,
                              width: e, k: k)
        }
        let cg = cosine(wantGlu, Array(bGlu.f32(n * e)))
        let cc = cosine(wantConv, Array(bConv.f32(n * e)))
        #expect(cg >= 0.99999, "glu_rows cosine \(cg)")
        #expect(cc >= 0.9999, "depthwise_causal cosine \(cc)")
    }

    // per_dim_scale is per HEAD DIMENSION, not per channel of the row: the
    // same 128 factors repeat across all 8 heads.
    @Test(needsGemmaWeights) func audioQueryScaleMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let a = try Gemma4Audio(h.model)
        let e = a.cfg.embd, d = a.cfg.headDim, nH = a.cfg.heads
        let n = 8
        let scales = a.queryScales(0)
        let x = activations(n * e)
        var want = x
        for t in 0..<n {
            for head in 0..<nH {
                for c in 0..<d {
                    want[t * e + head * d + c] *= scales[c]
                }
            }
        }
        let bX = h.ctx.makeF32(x)
        let bS = h.ctx.makeF32(scales)
        _ = h.run(n * e) { f in
            f.scaleHeadDims(x: bX, scales: bS, rows: n, width: e, hd: d)
        }
        let c = cosine(want, Array(bX.f32(n * e)))
        #expect(c >= 0.99999, "scale_head_dims cosine \(c)")
    }

    // The one audio kernel with a contract worth fearing.
    @Test(needsGemmaWeights) func audioWindowedAttentionMatchesSIMD() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let a = try Gemma4Audio(h.model)
        let e = a.cfg.embd
        let n = 50
        #expect(n > 3 * a.cfg.chunk)
        let q = activations(n * e)
        let k = activations(n * e).map { v in v * 0.75 }
        let v = activations(n * e).map { w in w * 1.25 }
        let want = a.windowed(0, q, k, v, n)
        let bQ = h.ctx.makeF32(q)
        let bK = h.ctx.makeF32(k)
        let bV = h.ctx.makeF32(v)
        let bRel = h.ctx.makeF32(a.relativeSlice(0))
        let bOut = h.ctx.makeF32(n * e)
        _ = h.run(n * e) { f in
            f.gemmaAudioAttn(q: bQ, k: bK, v: bV, relk: bRel, out: bOut,
                             rows: n, width: e, heads: a.cfg.heads,
                             hd: a.cfg.headDim, chunk: a.cfg.chunk,
                             context: a.cfg.context,
                             past: a.cfg.pastHorizon,
                             cap: a.cfg.logitCap)
        }
        let got = Array(bOut.f32(n * e))
        let c = cosine(want, got)
        let diff = maxAbsDiff(want, got)
        #expect(c >= 0.9999,
                "gemma_audio_attn cosine \(c) maxAbs \(diff)")
    }

    // Both rope regimes: the sliding layers rotate every pair of a 256-wide
    // head, the full layers only the first 64 of a 512-wide one.
    @Test(needsGemmaWeights) func gemmaRopeMatchesSIMDBothRegimes() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let c = h.model.cfg
        let cases = [(c.headDimSliding, c.rotatedPairsSliding,
                      c.ropeBaseSliding, "sliding"),
                     (c.headDimFull, c.rotatedPairsFull,
                      c.ropeBaseFull, "full")]
        for (hd, rot, base, label) in cases {
            let nH = 2
            var want = activations(hd * nH)
            let x = want
            GK.rope(&want, headDim: hd, nHead: nH, rotated: rot,
                    base: base, pos: 37)

            let xb = h.ctx.makeF32(x)
            _ = h.run(hd * nH) { f in
                f.ropeGemma(x: xb, headDim: hd, nHead: nH, rotated: rot,
                            base: base, pos: 37)
            }
            let got = Array(xb.f32(hd * nH))
            let cos = cosine(want, got)
            let diff = maxAbsDiff(want, got)
            #expect(cos >= 0.99999,
                    "rope_gemma \(label) cosine \(cos) maxAbs \(diff)")
        }
    }

    @Test(needsGemmaWeights) func clampedGemvMatchesThreeDispatches() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let w = h.model.layers[0].wq
        let s = SRQ(h.model.gguf, w.name)
        #expect(s.input.active || s.output.active, "no clamp on \(w.name)")
        let K = w.dims[0], M = w.dims[1]
        let x = activations(K).map { v in v * 8 }
        var xin = x
        SRQ.apply(&xin, s.input)
        var want = [Float](repeating: 0, count: M)
        GQ.matvec(w, x: xin, out: &want)
        SRQ.apply(&want, s.output)
        let xb = h.ctx.makeF32(x)
        let ob = h.ctx.makeF32(M)
        let three = h.ctx.makeF32(M)
        let scratch = h.ctx.makeF32(K)
        _ = h.run(M) { f in
            f.gemvSrq(w, x: xb, out: ob, off: h.off(w), srq: s)
            f.srqTo(src: xb, dst: scratch, n: K, s: s.input)
            f.gemv(w, x: scratch, out: three, off: h.off(w))
            f.srqInPlace(x: three, n: M, s: s.output)
        }
        let got = Array(ob.f32(M))
        let fold = maxAbsDiff(got, Array(three.f32(M)))
        #expect(fold == 0, "the fold changed the arithmetic by \(fold)")
        let cos = cosine(want, got)
        #expect(cos >= 0.99999, "q4_0_gemv_srq cosine \(cos)")
    }

    @Test(needsGemmaWeights) func normAddNormMatchesFourDispatches() throws {
        let path = try #require(gemmaGgufPath)
        let h = try MetalHarness(path)
        let g = h.model.gguf
        let post = g.tensor("blk.0.post_attn_norm.weight")
        let pre = g.tensor("blk.0.ffn_norm.weight")
        let n = h.model.cfg.nEmbd
        let eps = h.model.cfg.eps
        let x = activations(n)
        let c = activations(2 * n).suffix(n).map { v in v * 3 }
        let xb = h.ctx.makeF32(x)
        let cb = h.ctx.makeF32(c)
        let yb = h.ctx.makeF32(n)
        let x4 = h.ctx.makeF32(x)
        let t4 = h.ctx.makeF32(n)
        let y4 = h.ctx.makeF32(n)
        _ = h.run(n) { f in
            f.normAddNormBF16(x: xb, c: cb, post: h.off(post),
                              pre: h.off(pre), y: yb, n: n, rows: 1,
                              eps: eps, s: 0.75)
            f.rmsnormBF16(x: cb, weightOff: h.off(post), out: t4, n: n,
                          eps: eps)
            f.add(x: x4, y: t4, n: n)
            f.scaleInPlace(x: x4, n: n, s: 0.75)
            f.rmsnormBF16(x: x4, weightOff: h.off(pre), out: y4, n: n,
                          eps: eps)
        }
        let dx = maxAbsDiff(Array(xb.f32(n)), Array(x4.f32(n)))
        let dy = maxAbsDiff(Array(yb.f32(n)), Array(y4.f32(n)))
        #expect(dx <= 1e-5 && dy <= 1e-5,
                "norm_add_norm_bf16 residual \(dx) normed \(dy)")
    }
}
