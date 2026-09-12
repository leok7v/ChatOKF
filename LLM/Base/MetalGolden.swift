import Foundation
import Metal

public enum MetalGolden {
    private static func fill(_ n: Int, seed: UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = seed &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: s >> 33)
            out[i] = Float(bits) / Float(UInt32.max) * 2 - 1
        }
        return out
    }

    private struct Case {
        let name: String
        let out: MTLBuffer
        let count: Int
        let encode: (MetalEnc) -> Void
    }

    public static func run(ggufPath: String, dir: String) throws -> String {
        let g = try GGUF(path: ggufPath)
        let ctx = try MetalContext(g)
        try ctx.prewarm()
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)

        func first(_ ok: (GGUFTensor) -> Bool) -> GGUFTensor? {
            var found: GGUFTensor? = nil
            for name in g.tensors.keys.sorted() where found == nil {
                let t = g.tensors[name]!
                if ok(t) { found = t }
            }
            return found
        }
        func vector(_ ty: GGUFType) -> GGUFTensor? {
            first { t in
                t.type == ty && t.dims.count == 1 && t.dims[0] >= 256
            }
        }
        func matrix(_ ty: GGUFType) -> GGUFTensor? {
            first { t in
                t.type == ty && t.dims.count == 2
                    && t.dims[0] >= 256 && t.dims[1] >= 256
                    && t.dims[1] <= 8192
            }
        }
        func off(_ t: GGUFTensor) -> WeightRef {
            ctx.window(UInt64(t.base - g.map))
        }

        var cases: [Case] = []
        let eps: Float = 1e-6

        for (ty, tag) in [(GGUFType.f32, "f32"), (.bf16, "bf16")] {
            if let w = vector(ty) {
                let n = min(w.dims[0], 2048)
                let rows = 3
                let x1 = ctx.makeF32(fill(n, seed: 11))
                let o1 = ctx.makeF32(n)
                cases.append(Case(name: "rmsnorm_\(tag)", out: o1,
                                  count: n) { f in
                    if ty == .bf16 {
                        f.rmsnormBF16(x: x1, weightOff: off(w), out: o1,
                                      n: n, eps: eps)
                    } else {
                        f.rmsnorm(x: x1, weightOff: off(w), out: o1, n: n,
                                  eps: eps)
                    }
                })
                let x2 = ctx.makeF32(fill(rows * n, seed: 12))
                let o2 = ctx.makeF32(rows * n)
                cases.append(Case(name: "rmsnorm_batch_\(tag)", out: o2,
                                  count: rows * n) { f in
                    if ty == .bf16 {
                        f.rmsnormBatchBF16(x: x2, weightOff: off(w), y: o2,
                                           n: n, rows: rows, eps: eps)
                    } else {
                        f.rmsnormBatch(x: x2, weightOff: off(w), y: o2, n: n,
                                       rows: rows, eps: eps)
                    }
                })
                let d = 128, r = 4
                let x3 = ctx.makeF32(fill(d * r, seed: 13))
                cases.append(Case(name: "rmsnorm_rows_\(tag)", out: x3,
                                  count: d * r) { f in
                    if ty == .bf16 {
                        f.rmsnormRowsBF16(x: x3, xoff: 0, d: d, rows: r,
                                          weightOff: off(w), eps: eps)
                    } else {
                        f.rmsnormRows(x: x3, xoff: 0, d: d, rows: r,
                                      weightOff: off(w), eps: eps)
                    }
                })
            }
        }
        let d = 128, r = 4
        let xNo = ctx.makeF32(fill(d * r, seed: 14))
        cases.append(Case(name: "rmsnorm_rows_noweight", out: xNo,
                          count: d * r) { f in
            f.rmsnormRowsNoWeight(x: xNo, xoff: 0, d: d, rows: r, eps: eps)
        })
        let xL2 = ctx.makeF32(fill(d * r, seed: 15))
        cases.append(Case(name: "l2norm_rows", out: xL2, count: d * r) { f in
            f.l2normRows(x: xL2, xoff: 0, d: d, rows: r, eps: eps)
        })
        let toks = 3, perTok = 2, stride = 512
        let xL2B = ctx.makeF32(fill(toks * stride, seed: 16))
        cases.append(Case(name: "l2norm_rows_batch", out: xL2B,
                          count: toks * stride) { f in
            f.l2normRowsBatch(x: xL2B, d: d, rowsPerTok: perTok,
                              tokStride: stride, tokens: toks, eps: eps)
        })
        let lnN = 256, lnRows = 3
        let lnX = ctx.makeF32(fill(lnRows * lnN, seed: 17))
        let lnW = ctx.makeF32(fill(lnN, seed: 18))
        let lnB = ctx.makeF32(fill(lnN, seed: 19))
        let lnY = ctx.makeF32(lnRows * lnN)
        cases.append(Case(name: "vit_layernorm", out: lnY,
                          count: lnRows * lnN) { f in
            f.layerNorm(x: lnX, w: lnW, b: lnB, y: lnY, n: lnN,
                        rows: lnRows, eps: eps)
        })

        let hd = 128, nH = 2
        let rNeox = ctx.makeF32(fill(hd * nH, seed: 20))
        cases.append(Case(name: "rope_neox", out: rNeox,
                          count: hd * nH) { f in
            f.rope(x: rNeox, headDim: hd, nHead: nH, nRot: 64, base: 1e6,
                   pos: 37)
        })
        let nBat = 3
        let rBat = ctx.makeF32(fill(nBat * hd * nH, seed: 21))
        cases.append(Case(name: "rope_batch", out: rBat,
                          count: nBat * hd * nH) { f in
            f.ropeBatch(x: rBat, headDim: hd, nHead: nH, nRot: 64, base: 1e6,
                        basePos: 5, N: nBat)
        })
        let rGem = ctx.makeF32(fill(hd * nH, seed: 22))
        cases.append(Case(name: "rope_gemma", out: rGem,
                          count: hd * nH) { f in
            f.ropeGemma(x: rGem, headDim: hd, nHead: nH, rotated: 16,
                        base: 1e4, pos: 37)
        })
        let rGemB = ctx.makeF32(fill(nBat * hd * nH, seed: 23))
        cases.append(Case(name: "rope_gemma_batch", out: rGemB,
                          count: nBat * hd * nH) { f in
            f.ropeGemmaBatch(x: rGemB, headDim: hd, nHead: nH, rotated: 16,
                             base: 1e4, basePos: 5, N: nBat)
        })
        var pos3: [Int32] = []
        for n in 0..<nBat {
            pos3 += [Int32(n + 1), Int32(n + 4), Int32(n + 9)]
        }
        let mrX = ctx.makeF32(fill(nBat * hd * nH, seed: 24))
        let mrP = ctx.makeF32(nBat * 3)
        pos3.withUnsafeBytes { src in
            mrP.contents().copyMemory(from: src.baseAddress!,
                                      byteCount: src.count)
        }
        cases.append(Case(name: "rope_mrope_batch", out: mrX,
                          count: nBat * hd * nH) { f in
            f.ropeMBatch(x: mrX, pos3: mrP, headDim: hd, nHead: nH, nRot: 64,
                         base: 1e6, N: nBat)
        })
        let vN = 4
        let vX = ctx.makeF32(fill(vN * hd * nH, seed: 25))
        let vCos = ctx.makeF32(fill(vN * hd, seed: 26))
        let vSin = ctx.makeF32(fill(vN * hd, seed: 27))
        cases.append(Case(name: "vit_rope", out: vX,
                          count: vN * hd * nH) { f in
            f.visionRope(x: vX, cos: vCos, sin: vSin, rowStride: hd * nH,
                         off: 0, headDim: hd, nHead: nH, N: vN)
        })
        let gvX = ctx.makeF32(fill(vN * hd * nH, seed: 28))
        cases.append(Case(name: "gemma_vit_rope", out: gvX,
                          count: vN * hd * nH) { f in
            f.gemmaVisionRope(x: gvX, cos: vCos, sin: vSin,
                              rowStride: hd * nH, headDim: hd, nHead: nH,
                              N: vN)
        })

        for ty in [GGUFType.q2_0, .q4_0, .q8_0, .bf16, .f32]
            + QwenMetalSelfTest.gemvTypes {
            if let w = matrix(ty) {
                let K = w.dims[0], M = w.dims[1]
                let tag = "\(ty)"
                let xv = ctx.makeF32(fill(K, seed: 30))
                let ov = ctx.makeF32(M)
                cases.append(Case(name: "gemv_\(tag)", out: ov,
                                  count: M) { f in
                    f.gemv(w, x: xv, out: ov, off: off(w))
                })
                if ty == .q2_0
                    || ty == .q4_0 || ty == .q8_0 || ty == .iq4_nl
                    || Blocks.superBlocked(ty) {
                    for n in [8, 33] {
                        let xm = ctx.makeF32(fill(n * K, seed: 31))
                        let om = ctx.makeF32(n * M)
                        cases.append(Case(name: "gemm_\(tag)_n\(n)", out: om,
                                          count: n * M) { f in
                            f.gemm(w, X: xm, out: om, off: off(w), N: n)
                        })
                    }
                }
            }
        }

        for hd in [256, 512] {
            let nH = 2, nKV = 1, T = 40, P = 8
            let pool = MetalKVPool(device: ctx.device, P: P, kvDim: hd * nKV)
            pool.appendBatch(T)
            let src = fill(2 * T * hd, seed: 40)
            for t in 0..<T {
                let kp = pool.kPages[t / P].contents()
                    .assumingMemoryBound(to: Float16.self)
                let vp = pool.vPages[t / P].contents()
                    .assumingMemoryBound(to: Float16.self)
                for i in 0..<hd {
                    kp[(t % P) * hd + i] = Float16(src[t * hd + i])
                    vp[(t % P) * hd + i] = Float16(src[T * hd + t * hd + i])
                }
            }
            pool.refreshTable()
            let gate = ctx.makeF32(fill(hd * nH, seed: 41))
            for (lo, gated) in [(0, 0), (12, 1)] {
                let q = ctx.makeF32(fill(hd * nH, seed: 42))
                let o = ctx.makeF32(hd * nH)
                cases.append(Case(name: "attn_head_hd\(hd)_lo\(lo)_g\(gated)",
                                  out: o, count: hd * nH) { f in
                    f.attnPaged(q: q, kAddr: pool.kAddr, vAddr: pool.vAddr,
                                pages: pool.residentPages, gate: gate,
                                out: o, hd: hd, nH: nH, nKV: nKV, T: T,
                                kvDim: hd * nKV, P: P, scale: 0.125,
                                gated: gated, lo: lo)
                })
            }
            let nq = 5
            let gateN = ctx.makeF32(fill(nq * hd * nH, seed: 43))
            for (basePos, window) in [(0, 0), (17, 16)] {
                let qN = ctx.makeF32(fill(nq * hd * nH, seed: 44))
                let oN = ctx.makeF32(nq * hd * nH)
                let nm = "attn_batch_hd\(hd)_p\(basePos)_w\(window)"
                cases.append(Case(name: nm, out: oN,
                                  count: nq * hd * nH) { f in
                    f.attnBatch(qN: qN, kAddr: pool.kAddr, vAddr: pool.vAddr,
                                pages: pool.residentPages, gateN: gateN,
                                outN: oN, hd: hd, nH: nH, nKV: nKV,
                                kvDim: hd * nKV, P: P, scale: 0.125,
                                basePos: basePos, N: nq, gated: 0,
                                window: window)
                })
            }
        }

        let aN = 40, vHd = 64, vHeads = 2
        let vE = vHd * vHeads
        let vQkv = ctx.makeF32(fill(aN * 3 * vE, seed: 50))
        let vOut = ctx.makeF32(aN * vE)
        cases.append(Case(name: "vit_attn", out: vOut,
                          count: aN * vE) { f in
            f.visionAttn(qkv: vQkv, out: vOut, n: aN, embd: vE, hd: vHd,
                         nHead: vHeads, scale: 0.125)
        })
        // Gate on the tensors, not on try?: Gemma4AudioConfig force-unwraps its
        // metadata and fatals on a file without an audio tower.
        if g.maybe("a.blk.0.attn_q.weight") != nil,
           let ac = try? Gemma4AudioConfig(g) {
            let an = 4 * ac.chunk
            let ae = ac.embd
            let aq = ctx.makeF32(fill(an * ae, seed: 51))
            let ak = ctx.makeF32(fill(an * ae, seed: 52))
            let av = ctx.makeF32(fill(an * ae, seed: 53))
            let arel = ctx.makeF32(fill(ac.context * ae, seed: 54))
            let aout = ctx.makeF32(an * ae)
            cases.append(Case(name: "gemma_audio_attn", out: aout,
                              count: an * ae) { f in
                f.gemmaAudioAttn(q: aq, k: ak, v: av, relk: arel, out: aout,
                                 rows: an, width: ae, heads: ac.heads,
                                 hd: ac.headDim, chunk: ac.chunk,
                                 context: ac.context, past: ac.pastHorizon,
                                 cap: ac.logitCap)
            })
        }

        var lines = ["  weights in \(ctx.windowMB.count) windows "
                     + "\(ctx.windowMB) MB, GPU caps one at "
                     + "\(ctx.device.maxBufferLength / 1_048_576) MB"]
        var failed = 0
        var wrote = 0
        for c in cases {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            c.encode(MetalEnc(ctx: ctx, e: e))
            e.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            precondition(cb.error == nil,
                         "\(c.name): \(cb.error!)")
            let bytes = c.count * MemoryLayout<Float>.stride
            let got = Data(bytes: c.out.contents(), count: bytes)
            let file = (dir as NSString)
                .appendingPathComponent("\(c.name).bin")
            let old = try? Data(contentsOf: URL(fileURLWithPath: file))
            if old == nil {
                try got.write(to: URL(fileURLWithPath: file))
                wrote += 1
                lines.append("  WROTE  \(c.name) (\(c.count) floats)")
            } else if old! == got {
                lines.append("  MATCH  \(c.name)")
            } else {
                failed += 1
                lines.append("  DIFFER \(c.name)  \(describe(old!, got))")
            }
        }
        let verdict = failed == 0
            ? (wrote > 0 ? "GOLDEN WRITTEN (\(wrote) new)" : "GOLDEN MATCH")
            : "GOLDEN DIFFER (\(failed) of \(cases.count))"
        return lines.joined(separator: "\n") + "\n" + verdict
    }

    private static func describe(_ a: Data, _ b: Data) -> String {
        var out = "sizes \(a.count) vs \(b.count)"
        if a.count == b.count {
            let n = a.count / MemoryLayout<Float>.stride
            let fa = a.withUnsafeBytes { p in
                Array(p.bindMemory(to: Float.self).prefix(n))
            }
            let fb = b.withUnsafeBytes { p in
                Array(p.bindMemory(to: Float.self).prefix(n))
            }
            var firstAt = -1
            var worst: Float = 0
            var count = 0
            for i in 0..<n where fa[i].bitPattern != fb[i].bitPattern {
                if firstAt < 0 { firstAt = i }
                worst = max(worst, abs(fa[i] - fb[i]))
                count += 1
            }
            out = "\(count)/\(n) floats differ, first at \(firstAt) "
                + "(\(fa[max(firstAt, 0)]) vs \(fb[max(firstAt, 0)])), "
                + "maxAbs \(worst)"
        }
        return out
    }
}
