import Foundation
import Metal

public enum QwenMetalSelfTest {
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

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    public static func run(ggufPath: String) throws -> String {
        let model = try QwenModel(path: ggufPath)
        let ctx = try MetalContext(model.gguf)
        func step(_ s: String) { print(s); fflush(stdout) }
        let hd = model.cfg.headDim
        step("device.maxThreadgroupMemoryLength="
            + "\(ctx.device.maxThreadgroupMemoryLength)  "
            + "attn tgmem=\((5 * hd + 8) * 4)B (hd=\(hd), constant in T)")
        step("weights in \(ctx.windowMB.count) windows \(ctx.windowMB) MB, "
            + "GPU caps one at "
            + "\(ctx.device.maxBufferLength / 1_048_576) MB")
        step(try checkGemv(model, ctx))
        step(try checkGemm(model, ctx))
        step(try checkNarrow(model, ctx))
        step(try checkDenseTile(model, ctx))
        step(try checkGemvTypes(model, ctx))
        step(try checkVisionBlocks(ctx))
        step(try checkDequant(model, ctx))
        step(try checkForward(model))
        step(try checkDecode(model))
        step(checkBookmark(model))
        step(try checkBatchedPrefill(model))
        step(try checkLongContext(model))
        step(try checkMetalPaged(model))
        step(try checkMetalBookmark(model))
        step(try checkCeiling(model))
        return "done"
    }

    private static func checkBatchedPrefill(_ model: QwenModel) throws
        -> String {
        var lines: [String] = []
        var failed = 0
        for chunk in [4, 3] {
            let r = try prefillCase(model, chunk: chunk)
            lines.append("  " + r.0)
            if !r.1 { failed += 1 }
        }
        return "batched prefill \(lines.count - failed)/\(lines.count)  "
            + (failed == 0 ? "PASS" : "FAIL") + "\n"
            + lines.joined(separator: "\n")
    }

    private static func prefillCase(_ model: QwenModel, chunk: Int) throws
        -> (String, Bool) {
        let ids: [Int32] = [9707, 11, 264, 1879, 374, 13, 264, 2100, 1110, 320]
        let n = 8
        let last = ids.count % chunk == 0 ? chunk : ids.count % chunk
        let mEng = try QwenMetalEngine(model)
        let sEng = QwenEngine(model)
        mEng.reset(); sEng.reset()
        var mCur = mEng.prefillBatch(ids, chunk: chunk)
        var sCur = sEng.extend(ids)
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return (String(format: "chunk %d, last %d  %d/%d match  %@", chunk,
                       last, match, n, match == n ? "PASS" : "FAIL")
                + "\n    metal=\(mSeq)\n    simd =\(sSeq)", match == n)
    }

    private static func checkLongContext(_ model: QwenModel) throws -> String {
        let count = 150
        var ids: [Int32] = []
        for i in 0..<count { ids.append(Int32(100 + (i * 37) % 900)) }
        let n = 6
        let mEng = try QwenMetalEngine(model)
        let sEng = QwenEngine(model)
        mEng.reset(); sEng.reset()
        var mCur = mEng.prefillBatch(ids, chunk: 48)
        var sCur = sEng.extend(ids)
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return "long-context (150 prefill, chunk 48) \(match)/\(n) match  "
            + (match == n ? "PASS" : "FAIL")
            + "\n  metal=\(mSeq)\n  simd =\(sSeq)"
    }

    private static func checkCeiling(_ model: QwenModel) throws -> String {
        let count = 9000
        var ids: [Int32] = []
        for i in 0..<count { ids.append(Int32(100 + (i * 37) % 900)) }
        let mEng = try QwenMetalEngine(model)
        mEng.reset()
        let tok = mEng.prefillBatch(ids, chunk: 512)
        _ = mEng.decode(tok)
        return "ceiling: prefilled \(count) positions + decoded past the old "
            + "~8160 cap  PASS"
    }

    private static func checkMetalPaged(_ model: QwenModel) throws -> String {
        let ids: [Int32] = [9707, 11, 264, 1879, 374]
        let n = 12
        let mEng = try QwenMetalEngine(model, pageP: 4)
        let sEng = QwenEngine(model)
        mEng.reset(); sEng.reset()
        var mSeq: [Int32] = [], sSeq: [Int32] = []
        var mCur = mEng.extend(ids), sCur = sEng.extend(ids)
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur); sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        return "metal paged P=4 \(match)/\(n) match  \(match == n ? "PASS" : "FAIL")"
    }

    private static func checkMetalBookmark(_ model: QwenModel) throws -> String {
        let e = try QwenMetalEngine(model, pageP: 4)
        e.reset()
        _ = e.extend([9707, 11, 264, 1879, 374, 13])
        let mark = e.bookmark()
        func path() -> [Int32] {
            var out: [Int32] = []
            var cur = e.extend([264, 1879])
            for _ in 0..<5 { out.append(cur); cur = e.decode(cur) }
            return out
        }
        let a = path()
        e.restore(mark)
        let b = path()
        return "metal bookmark replay  \(a == b ? "identical" : "DIVERGED")"
            + "  \(a == b ? "PASS" : "FAIL")"
    }

    private static func checkBookmark(_ model: QwenModel) -> String {
        let e = QwenEngine(model)
        e.reset()
        _ = e.extend([9707, 11, 264])
        let mark = e.bookmark()
        func path() -> [Int32] {
            var out: [Int32] = []
            var cur = e.extend([1879, 374])
            for _ in 0..<5 { out.append(cur); cur = e.decode(cur) }
            return out
        }
        let a = path()
        e.restore(mark)
        let b = path()
        let tag = a == b ? "PASS" : "FAIL"
        return "bookmark restore replay  \(a == b ? "identical" : "DIVERGED")"
            + "  \(tag)\n  a=\(a)\n  b=\(b)"
    }

    private static func checkForward(_ model: QwenModel) throws -> String {
        let mEng = try QwenMetalEngine(model)
        let sEng = QwenEngine(model)
        let tok = 9707
        mEng.reset()
        let mhBuf = mEng.forward(token: tok, pos: 0)
        let mh = Array(mhBuf.f32(model.cfg.nEmbd))
        let ml = Vectors.argmax(mEng.logits(mhBuf))
        sEng.reset()
        let sh = sEng.forward(token: tok, pos: 0)
        let sl = Vectors.argmax(sEng.logits(sh))
        let dh = maxAbsDiff(mh, sh)
        let tag = dh < 5e-2 && ml == sl ? "PASS" : "FAIL"
        return "forward tok=\(tok)  hiddenMaxDiff=\(dh)  "
            + "argmax metal=\(ml) simd=\(sl)  \(tag)"
    }

    private static func checkDecode(_ model: QwenModel) throws -> String {
        let ids: [Int32] = [9707, 11, 264, 1879, 374]
        let n = 12
        let mEng = try QwenMetalEngine(model)
        let sEng = QwenEngine(model)
        mEng.reset(); sEng.reset()
        var mSeq: [Int32] = []
        var sSeq: [Int32] = []
        var mCur = mEng.extend(ids)
        var sCur = sEng.extend(ids)
        for _ in 0..<n {
            mSeq.append(mCur); sSeq.append(sCur)
            mCur = mEng.decode(mCur)
            sCur = sEng.decode(sCur)
        }
        var match = 0
        for i in 0..<n where mSeq[i] == sSeq[i] { match += 1 }
        let tag = match == n ? "PASS" : "FAIL"
        return "decode \(match)/\(n) greedy tokens match  \(tag)\n"
            + "  metal=\(mSeq)\n  simd =\(sSeq)"
    }

    // Every fixture is a SINGLE block, so two block-mask bugs sailed through
    // every gate; the no-table run must MOVE, or a blind kernel passes.
    private static func checkVisionBlocks(_ ctx: MetalContext) throws
        -> String {
        var lines: [String] = []
        var failed = 0
        for basePos in [0, 8] {
            for window in [0, 12] {
                let r = try visionCase(ctx, basePos: basePos, window: window)
                lines.append("  " + r.0)
                if !r.1 { failed += 1 }
            }
        }
        return "vision blocks (2 blocks + text in one chunk)  "
            + (failed == 0 ? "PASS" : "FAIL (\(failed))") + "\n"
            + lines.joined(separator: "\n")
    }

    private static func visionCase(_ ctx: MetalContext, basePos: Int,
                                   window: Int) throws -> (String, Bool) {
        let hd = 256, nH = 2, nKV = 1, P = 8, N = 37
        let scale: Float = 0.125
        let T = basePos + N
        let fix = kvFixture(ctx, hd: hd, nH: nH, nKV: nKV, P: P, T: T, N: N)
        let rows = blockRows(N, basePos, [(2, 14), (15, 31)])
        let table = ctx.makeU32(2 * (N + 8))
        let cells = table.u32(2 * (N + 8))
        for j in 0..<N {
            cells[2 * j] = UInt32(rows[j].0)
            cells[2 * j + 1] = UInt32(rows[j].1)
        }
        let want = attendRef(fix.q, fix.k, fix.v, rows, basePos: basePos,
                             N: N, nH: nH, hd: hd, window: window,
                             scale: scale)
        func run(_ blocks: MTLBuffer?, _ matrix: Bool) -> [Float] {
            attn(ctx, fix, hd: hd, nH: nH, nKV: nKV, P: P, N: N,
                 basePos: basePos, window: window, scale: scale,
                 blocks: blocks, matrix: matrix)
        }
        let scalar = relDiff(run(table, false), want)
        let mm = ctx.matrixUnits ? relDiff(run(table, true), want) : 0
        let blind = ctx.matrixUnits ? relDiff(run(nil, true), want)
                                    : relDiff(run(nil, false), want)
        let ok = scalar < 1e-4 && mm < 1e-2 && blind > 0.1
        return (String(format:
            "p%-2d w%-2d  scalar %.2e  matrix %.2e  no-table %.2e  %@",
            basePos, window, scalar, mm, blind, ok ? "PASS" : "FAIL"), ok)
    }

    private static func blockRows(_ N: Int, _ basePos: Int,
                                  _ runs: [(Int, Int)]) -> [(Int, Int)] {
        var out = [(Int, Int)](repeating: (0, 0), count: N)
        for r in runs {
            for j in r.0..<r.1 { out[j] = (basePos + r.0, basePos + r.1) }
        }
        return out
    }

    // The pages are half, so the oracle scores against the ROUNDED values or
    // it measures the storage format instead of the mask.
    private struct KVFixture {
        let pool: MetalKVPool
        let qBuf: MTLBuffer
        let gate: MTLBuffer
        let out: MTLBuffer
        let q: [Float]
        let k: [Float]
        let v: [Float]
    }

    private static func kvFixture(_ ctx: MetalContext, hd: Int, nH: Int,
                                  nKV: Int, P: Int, T: Int,
                                  N: Int) -> KVFixture {
        precondition(nKV == 1, "the k/v below are indexed as one kv head")
        let pool = MetalKVPool(device: ctx.device, P: P, kvDim: hd * nKV)
        try! pool.attachTemp()
        pool.appendBatch(T)
        let src = fill(2 * T * hd, seed: 60)
        var k = [Float](repeating: 0, count: T * hd)
        var v = [Float](repeating: 0, count: T * hd)
        for t in 0..<T {
            let kp = pool.kPages[t / P].contents()
                .assumingMemoryBound(to: Float16.self) + (t % P) * hd
            let vp = pool.vPages[t / P].contents()
                .assumingMemoryBound(to: Float16.self) + (t % P) * hd
            for i in 0..<hd {
                kp[i] = Float16(src[t * hd + i])
                vp[i] = Float16(src[(T + t) * hd + i])
                k[t * hd + i] = Float(kp[i])
                v[t * hd + i] = Float(vp[i])
            }
        }
        pool.refreshTable()
        let q = fill(N * nH * hd, seed: 61)
        return KVFixture(pool: pool, qBuf: ctx.makeF32(q),
                         gate: ctx.makeF32(N * nH * hd),
                         out: ctx.makeF32(N * nH * hd), q: q, k: k, v: v)
    }

    private static func attn(_ ctx: MetalContext, _ fix: KVFixture, hd: Int,
                             nH: Int, nKV: Int, P: Int, N: Int, basePos: Int,
                             window: Int, scale: Float, blocks: MTLBuffer?,
                             matrix: Bool) -> [Float] {
        let pool = fix.pool
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).attnBatch(
            qN: fix.qBuf, kAddr: pool.kAddr, vAddr: pool.vAddr,
            pages: pool.residentPages, gateN: fix.gate, outN: fix.out,
            hd: hd, nH: nH, nKV: nKV, kvDim: hd * nKV, P: P, scale: scale,
            basePos: basePos, N: N, gated: 0, window: window,
            blocks: blocks, matrix: matrix)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        precondition(cb.error == nil, "attn_batch: \(cb.error!)")
        return Array(fix.out.f32(N * nH * hd))
    }

    private static func attendRef(_ q: [Float], _ k: [Float], _ v: [Float],
                                  _ blocks: [(Int, Int)], basePos: Int,
                                  N: Int, nH: Int, hd: Int, window: Int,
                                  scale: Float) -> [Float] {
        var out = [Float](repeating: 0, count: N * nH * hd)
        for n in 0..<N {
            let p = basePos + n
            let t = p + 1
            let lo = window == 0 || t <= window ? 0 : t - window
            let b = blocks[n]
            let hi = p >= b.0 && p < b.1 ? max(t, b.1) : t
            for h in 0..<nH {
                let row = attendOne(q, (n * nH + h) * hd, k, v, hd, lo, hi,
                                    scale)
                for i in 0..<hd { out[(n * nH + h) * hd + i] = row[i] }
            }
        }
        return out
    }

    private static func attendOne(_ q: [Float], _ qo: Int, _ k: [Float],
                                  _ v: [Float], _ hd: Int, _ lo: Int,
                                  _ hi: Int, _ scale: Float) -> [Float] {
        var s = [Float](repeating: 0, count: hi - lo)
        var top = -Float.greatestFiniteMagnitude
        for t in lo..<hi {
            s[t - lo] = dot(q, qo, k, t * hd, hd) * scale
            top = max(top, s[t - lo])
        }
        var sum: Float = 0
        for i in 0..<s.count {
            s[i] = expf(s[i] - top)
            sum += s[i]
        }
        var out = [Float](repeating: 0, count: hd)
        for t in lo..<hi { axpy(s[t - lo] / sum, v, t * hd, &out, hd) }
        return out
    }

    private static func dot(_ a: [Float], _ ao: Int, _ b: [Float], _ bo: Int,
                            _ n: Int) -> Float {
        var s: Float = 0
        for i in 0..<n { s += a[ao + i] * b[bo + i] }
        return s
    }

    private static func axpy(_ w: Float, _ src: [Float], _ so: Int,
                             _ dst: inout [Float], _ n: Int) {
        for i in 0..<n { dst[i] += w * src[so + i] }
    }

    // RELATIVE: the output is an average of V rows, so its magnitude follows
    // the fixture rather than the kernel.
    private static func relDiff(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0
        var mag: Float = 0
        for i in 0..<min(a.count, b.count) {
            d = max(d, abs(a[i] - b[i]))
            mag = max(mag, abs(b[i]))
        }
        return mag > 0 ? d / mag : d
    }

    private static func checkGemm(_ model: QwenModel,
                                  _ ctx: MetalContext) throws -> String {
        let more = [GGUFType.q5k, .q6k, .iq4_xs, .q3k, .iq2_s, .iq3_xxs,
                    .iq3_s, .iq2_xxs, .iq2_xs, .iq1_s, .iq1_m]
            .compactMap { ty in
            widest(model, ty)
        }
        var lines: [String] = []
        var failed = 0
        for w in [model.layers[0].ffnGate] + more {
            let line = gemmCase(model, ctx, w)
            if !line.hasSuffix("PASS") { failed += 1 }
            lines.append("  " + line)
        }
        return "gemm tile vs Swift oracle \(lines.count - failed)/\(lines.count)  "
            + (failed == 0 ? "PASS" : "FAIL") + "\n"
            + lines.joined(separator: "\n")
    }

    private static func gemmCase(_ model: QwenModel, _ ctx: MetalContext,
                                 _ w: GGUFTensor) -> String {
        let k = w.dims[0], m = w.dims[1]
        let n = MetalEnc.narrowMax + 28
        var xflat: [Float] = []
        for c in 0..<n { xflat += fill(k, seed: UInt64(c + 10)) }
        let xb = ctx.makeF32(xflat)
        let outb = ctx.makeF32(n * m)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemm(
            w, X: xb, out: outb,
            off: ctx.window(UInt64(w.base - model.gguf.map)), N: n)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let gpu = Array(outb.f32(n * m))
        var d: Float = 0
        var mag: Float = 0
        for c in 0..<n {
            let col = Array(xflat[c * k ..< (c + 1) * k])
            var ref = [Float](repeating: 0, count: m)
            QB.matvec(w, x: col, out: &ref)
            for j in 0..<m {
                d = max(d, abs(gpu[c * m + j] - ref[j]))
                mag = max(mag, abs(ref[j]))
            }
        }
        // RELATIVE: an absolute bound on a K-deep dot product says nothing;
        // fp16 staging puts the format under 2e-3 and a defect at O(1).
        let rel = mag > 0 ? d / mag : d
        return String(format: "%@ [%d,%d]@[%d,%d]  rel %.2e  %@", "\(w.type)",
                      n, k, k, m, rel, rel < 2e-3 ? "PASS" : "FAIL")
    }

    private static func checkNarrow(_ model: QwenModel,
                                    _ ctx: MetalContext) throws -> String {
        var lines: [String] = []
        var failed = 0
        var ran = 0
        for ty in [GGUFType.q4_0, .q2_0] + gemvTypes {
            if let w = widest(model, ty) {
                let k = w.dims[0], m = w.dims[1]
                let off = ctx.window(UInt64(w.base - model.gguf.map))
                for n in 2...MetalEnc.narrowMax {
                    var xflat: [Float] = []
                    for c in 0..<n { xflat += fill(k, seed: UInt64(c + 70)) }
                    let xb = ctx.makeF32(xflat)
                    let narrow = ctx.makeF32(2 * n * m)
                    let cb = ctx.queue.makeCommandBuffer()!
                    let e = cb.makeComputeCommandEncoder()!
                    let f = MetalEnc(ctx: ctx, e: e)
                    f.gemm(w, X: xb, out: narrow, off: off, N: n)
                    for c in 0..<n {
                        f.gemv(w, x: xb, out: narrow, off: off,
                               xOff: c * k * 4,
                               outOff: (n + c) * m * 4)
                    }
                    e.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    precondition(cb.error == nil, "narrow: \(cb.error!)")
                    let got = Array(narrow.f32(2 * n * m))
                    var d: Float = 0
                    var mag: Float = 0
                    for i in 0..<(n * m) {
                        d = max(d, abs(got[i] - got[n * m + i]))
                        mag = max(mag, abs(got[n * m + i]))
                    }
                    let rel = mag > 0 ? d / mag : d
                    let ok = rel < 1e-5
                    if !ok { failed += 1 }
                    ran += 1
                    lines.append(String(format:
                        "  %@ N=%d [%d,%d]  rel %.2e  %@", "\(ty)", n, k, m,
                        rel, ok ? "PASS" : "FAIL"))
                }
            }
        }
        return "narrow gemm vs gemv \(ran - failed)/\(ran)  "
            + (failed == 0 ? "PASS" : "FAIL") + "\n"
            + lines.joined(separator: "\n")
    }

    private static func checkDenseTile(_ model: QwenModel,
                                       _ ctx: MetalContext) throws -> String {
        var lines: [String] = []
        var failed = 0
        var ran = 0
        let n = MetalEnc.narrowMax + 28
        for ty in [GGUFType.f16, .bf16, .f32] {
            if let w = widest(model, ty), w.dims[0] % 32 == 0 {
                let k = w.dims[0], m = w.dims[1]
                let off = ctx.window(UInt64(w.base - model.gguf.map))
                var xflat: [Float] = []
                for c in 0..<n { xflat += fill(k, seed: UInt64(c + 90)) }
                let xb = ctx.makeF32(xflat)
                let both = ctx.makeF32(2 * n * m)
                let cb = ctx.queue.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder()!
                let f = MetalEnc(ctx: ctx, e: e)
                f.gemm(w, X: xb, out: both, off: off, N: n)
                for c in 0..<n {
                    f.gemv(w, x: xb, out: both, off: off, xOff: c * k * 4,
                           outOff: (n + c) * m * 4)
                }
                e.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                precondition(cb.error == nil, "dense tile: \(cb.error!)")
                let got = Array(both.f32(2 * n * m))
                var d: Float = 0
                var mag: Float = 0
                for i in 0..<(n * m) {
                    d = max(d, abs(got[i] - got[n * m + i]))
                    mag = max(mag, abs(got[n * m + i]))
                }
                let rel = mag > 0 ? d / mag : d
                let ok = rel < 2e-3
                if !ok { failed += 1 }
                ran += 1
                lines.append(String(format: "  %@ N=%d [%d,%d]  rel %.2e  %@",
                                    "\(ty)", n, k, m, rel,
                                    ok ? "PASS" : "FAIL"))
            }
        }
        return "dense tile vs gemv \(ran - failed)/\(ran)  "
            + (failed == 0 ? "PASS" : "FAIL")
            + (lines.isEmpty ? "" : "\n" + lines.joined(separator: "\n"))
    }

    private static func widest(_ model: QwenModel,
                               _ ty: GGUFType) -> GGUFTensor? {
        var out: GGUFTensor? = nil
        for name in model.gguf.tensors.keys.sorted() {
            let t = model.gguf.tensors[name]!
            if t.type == ty && t.dims.count == 2 && t.dims[0] % 128 == 0
                && (out == nil || t.dims[1] > out!.dims[1]) {
                out = t
            }
        }
        return out
    }

    static let gemvTypes: [GGUFType] = [
        .q4k, .q5k, .q6k, .q2k, .q3k, .iq1_s, .iq1_m, .iq2_xxs,
        .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs, .iq4_nl,
    ]

    private static func checkGemvTypes(_ model: QwenModel,
                                       _ ctx: MetalContext) throws -> String {
        var lines: [String] = []
        var failed = 0
        var ran = 0
        for ty in gemvTypes {
            if let w = widest(model, ty) {
                let k = w.dims[0], m = w.dims[1]
                let x = fill(k, seed: 21)
                var ref = [Float](repeating: 0, count: m)
                QB.matvec(w, x: x, out: &ref)
                let gpu = try gemv(ctx, weight: w,
                                   mapBase: model.gguf.map, x: x)
                let d = maxAbsDiff(ref, gpu)
                let mag = ref.reduce(Float(0)) { peak, v in
                    max(peak, abs(v))
                }
                let rel = mag > 0 ? d / mag : d
                let ok = rel < 1e-4
                if !ok { failed += 1 }
                ran += 1
                lines.append(String(format: "  %@ [%d,%d]  rel %.2e  %@",
                                    "\(ty)", k, m, rel,
                                    ok ? "PASS" : "FAIL"))
            }
        }
        return "gemv vs Swift oracle, per type \(ran - failed)/\(ran)  "
            + (failed == 0 ? "PASS" : "FAIL")
            + (lines.isEmpty ? "" : "\n" + lines.joined(separator: "\n"))
    }

    private static func checkGemv(_ model: QwenModel,
                                  _ ctx: MetalContext) throws -> String {
        let w = model.layers[0].ffnGate
        let k = w.dims[0]
        let m = w.dims[1]
        let x = fill(k, seed: 1)

        var ref = [Float](repeating: 0, count: m)
        QB.matvec(w, x: x, out: &ref)

        let gpu = try gemv(ctx, weight: w, mapBase: model.gguf.map, x: x)
        let d = maxAbsDiff(ref, gpu)
        let mag = ref.reduce(Float(0)) { peak, v in max(peak, abs(v)) }
        let rel = mag > 0 ? d / mag : d
        let tag = rel < 1e-4 ? "PASS" : "FAIL"
        return "gemv ffn_gate[\(k),\(m)]  maxAbsDiff=\(d)  rel=\(rel)  \(tag)"
    }

    private static func checkDequant(_ model: QwenModel,
                                     _ ctx: MetalContext) throws -> String {
        let t = model.tokEmbd
        let k = t.dims[0]
        let id = 100
        let rowBytes = GGUF.rowByteCount(t.type, k)
        let row = ctx.window(
            UInt64((t.base - model.gguf.map) + id * rowBytes))

        var ref = [Float](repeating: 0, count: k)
        ref.withUnsafeMutableBufferPointer { rb in
            QB.dequant(t, row: id, count: k, into: rb.baseAddress!)
        }

        let out = ctx.makeF32(k)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).dequantRow(weightOff: row, out: out, n: k,
                                            type: t.type)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let gpu = Array(out.f32(k))
        let d = maxAbsDiff(ref, gpu)
        let tag = d < 1e-4 ? "PASS" : "FAIL"
        return "dequant tok_embd row[\(k)]  maxAbsDiff=\(d)  \(tag)"
    }

    static func gemv(_ ctx: MetalContext, weight w: GGUFTensor,
                     mapBase: UnsafeRawPointer, x: [Float]) throws -> [Float] {
        let m = w.dims[1]
        let xb = ctx.makeF32(x)
        let out = ctx.makeF32(m)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        MetalEnc(ctx: ctx, e: e).gemv(
            w, x: xb, out: out,
            off: ctx.window(UInt64(w.base - mapBase)))
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return Array(out.f32(m))
    }

    static func dispatch(_ ctx: MetalContext, _ name: String, threads: Int,
                         _ bind: (MTLComputeCommandEncoder) -> Void) throws
        -> MTLCommandBuffer {
        let pipe = try ctx.pipeline(name)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        bind(e)
        let w = pipe.threadExecutionWidth
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
        cb.commit()
        return cb
    }

    static func dispatch(_ ctx: MetalContext, _ name: String, groups: Int,
                         threadsPerGroup: Int,
                         _ bind: (MTLComputeCommandEncoder) -> Void) throws
        -> MTLCommandBuffer {
        let pipe = try ctx.pipeline(name)
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        bind(e)
        e.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup:
                MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        e.endEncoding()
        cb.commit()
        return cb
    }
}

// Matches the MSL `GemvArgs { ulong woff; uint K; uint M; }`.
struct GemvArgs { var woff: UInt64; var K: UInt32; var M: UInt32 }
