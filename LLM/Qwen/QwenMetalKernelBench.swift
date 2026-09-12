// Bytes a kernel MUST move divided by its time: the whole-model bench cannot
// separate a kernel from prefill, thermals and page cache.
import Foundation
import Metal

public enum QwenMetalKernelBench {

    // Needs MTL_CAPTURE_ENABLED=1 in the environment to be allowed at all.
    private static func capture(_ ctx: MetalContext) -> String {
        var note = ""
        if let path = Flags.value("gpu-capture"), !path.isEmpty {
            let mgr = MTLCaptureManager.shared()
            let d = MTLCaptureDescriptor()
            d.captureObject = ctx.queue
            if mgr.supportsDestination(.gpuTraceDocument) {
                d.destination = .gpuTraceDocument
                d.outputURL = URL(fileURLWithPath: path)
                try? FileManager.default.removeItem(atPath: path)
                do {
                    try mgr.startCapture(with: d)
                    note = "capturing to \(path)\n"
                } catch {
                    note = "capture refused: \(error)\n"
                }
            } else {
                note = "this device cannot write a .gputrace document; set "
                    + "MTL_CAPTURE_ENABLED=1\n"
            }
        }
        return note
    }

    public static func run(ggufPath: String) throws -> String {
        let gguf = try GGUF(path: ggufPath)
        let ctx = try MetalContext(gguf)
        try ctx.prewarm()
        let capNote = capture(ctx)
        let arch = gguf.string("general.architecture") ?? ""
        let nLayer = gguf.int(arch + ".block_count") ?? 0
        var picks: [(String, GGUFTensor)] = []
        for name in ["blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                     "blk.0.attn_qkv.weight", "blk.0.attn_q.weight",
                     "blk.0.ssm_out.weight", "blk.0.ssm_alpha.weight",
                     "per_layer_model_proj.weight",
                     "blk.\(nLayer).ffn_down.weight",
                     "blk.\(nLayer).attn_q.weight"] {
            if let t = gguf.maybe(name) { picks.append((name, t)) }
        }
        let head = gguf.maybe("output.weight")
            ?? gguf.maybe("token_embd.weight")
        if let head {
            picks.append(("lm_head", head))
        }
        if let only = Flags.value("gpu-capture-only"), !only.isEmpty {
            picks = picks.filter { p in p.0.contains(only) }
        }
        var out = "kernel bench: "
            + URL(fileURLWithPath: ggufPath).lastPathComponent + "\n"
        out += pad("tensor", 24) + pad("kernel", 10) + pad("N", 6)
            + pad("ms", 10) + pad("GB/s", 8) + "GFLOPS\n"
        let wide = 512
        for (name, t) in picks {
            let k = t.dims[0], m = t.dims[1]
            let bytes = GGUF.rowByteCount(t.type, k) * m
            let x = ctx.makeF32(wide * k)
            let y = ctx.makeF32(wide * m)
            seed(x, wide * k)
            // The first timed dispatch of a tensor pays its page-in.
            _ = time(ctx, t, x, y, n: 1, tile: false, gguf: gguf)
            for n in [1, 2, 3, 4, 5, 32, 128, wide] {
                for kind in ["auto", "tile"] where n <= 5 || kind == "tile"
                    || !Blocks.superBlocked(t.type) {
                    let ms = time(ctx, t, x, y, n: n,
                                  tile: kind == "tile", gguf: gguf)
                    if ms > 0 {
                        let gbs = Double(bytes) / (ms / 1000) / 1e9
                        let flops = 2.0 * Double(k) * Double(m) * Double(n)
                            / (ms / 1000) / 1e9
                        out += pad(name, 24) + pad(kind, 10) + pad("\(n)", 6)
                            + pad(String(format: "%.3f", ms), 10)
                            + pad(String(format: "%.1f", gbs), 8)
                            + String(format: "%.0f", flops) + "\n"
                    }
                }
            }
        }
        out += occupancy(ctx)
        if !capNote.isEmpty {
            MTLCaptureManager.shared().stopCapture()
            out += "\n" + capNote
        }
        return out
    }

    // maxTotalThreadsPerThreadgroup falls as register use rises.
    private static func occupancy(_ ctx: MetalContext) -> String {
        var out = "\nregister pressure (maxThreads/threadgroup, "
            + "1024 = uncontended)\n"
        var names = ["q4_0_gemv", "q4_k_gemv", "iq1_s_gemv",
                     "iq3_xxs_gemv", "iq_gemm_mm_h", "q4_k_gemm_mm_h",
                     "q6_k_gemm_mm_h", "q8_0_gemm_mm_h"]
        for r in 2...5 { names.append("q4_0_gemm_nb_r\(r)") }
        for shape in ["r2s1", "r2s8", "r3s1", "r3s8", "r4s8", "r5s8"] {
            names.append("q4_k_gemm_nb_" + shape)
            names.append("q8_0_gemm_nb_" + shape)
        }
        for n in names {
            if let p = try? ctx.pipeline(n) {
                out += pad(n, 24) + "\(p.maxTotalThreadsPerThreadgroup)\n"
            }
        }
        return out
    }

    private static func pad(_ s: String, _ n: Int) -> String {
        s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
    }

    private static func seed(_ b: MTLBuffer, _ n: Int) {
        let p = b.contents().assumingMemoryBound(to: Float.self)
        var s: UInt64 = 0x2545_F491_4F6C_DD1D
        for i in 0..<n {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            p[i] = Float(Int32(truncatingIfNeeded: s)) / 2.147e9
        }
    }

    // Iterations ride ONE command buffer so the measurement is GPU time for
    // the kernel, not per-commit latency.
    private static func time(_ ctx: MetalContext, _ t: GGUFTensor,
                             _ x: MTLBuffer, _ y: MTLBuffer, n: Int,
                             tile: Bool, gguf: GGUF) -> Double {
        let reps = 20
        let off = ctx.window(UInt64(t.base - gguf.map))
        var result = -1.0
        let tiled = [.q2_0, .q4_0, .q8_0, .f16, .bf16, .f32].contains(t.type)
            || Blocks.superBlocked(t.type)
        for _ in 0 ..< 3 {
            var ms = -1.0
            if tile {
                ms = tiled
                    ? run(ctx, reps) { f in
                        f.gemmTile(t, X: x, out: y, off: off, N: n)
                    } : -1
            } else {
                ms = run(ctx, reps) { f in
                    if n == 1 {
                        f.gemv(t, x: x, out: y, off: off)
                    } else {
                        f.gemm(t, X: x, out: y, off: off, N: n)
                    }
                }
            }
            if ms > 0 && (result < 0 || ms < result) { result = ms }
        }
        return result
    }

    private static func run(_ ctx: MetalContext, _ reps: Int,
                            _ body: (MetalEnc) -> Void) -> Double {
        let cb = ctx.queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        let f = MetalEnc(ctx: ctx, e: e)
        for _ in 0..<reps { body(f) }
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return cb.error == nil
            ? (cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(reps) : -1
    }
}
