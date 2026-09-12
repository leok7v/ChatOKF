import Foundation
import Metal

struct StateRing {
    let slot0: Int
    let slots: Int
    static let inPlace = StateRing(slot0: 0, slots: 1)
}

final class DispatchSync {
    let concurrent: Bool
    var grouped = false

    init(concurrent: Bool) {
        self.concurrent = concurrent
    }
}

struct MetalEnc {
    let ctx: MetalContext
    let e: MTLComputeCommandEncoder
    let sync: DispatchSync

    init(ctx: MetalContext, e: MTLComputeCommandEncoder,
         concurrent: Bool = false) {
        self.ctx = ctx
        self.e = e
        sync = DispatchSync(concurrent: concurrent)
    }

    static let labels = Flags.on("gpu-labels")
    private func push(_ n: String) {
        if MetalEnc.labels { e.pushDebugGroup(n) }
    }
    private func pop() {
        if MetalEnc.labels { e.popDebugGroup() }
        if sync.concurrent && !sync.grouped {
            e.memoryBarrier(scope: .buffers)
        }
    }

    func parallel(_ body: () -> Void) {
        sync.grouped = sync.concurrent
        body()
        sync.grouped = false
        if sync.concurrent { e.memoryBarrier(scope: .buffers) }
    }

    private func grid1D(_ name: String, _ n: Int,
                        _ bind: (MTLComputeCommandEncoder) -> Void) {
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        bind(e)
        let w = min(256, pipe.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: w, height: 1, depth: 1))
        pop()
    }

    private func groups(_ name: String, _ groups: Int, tpg: Int,
                        tgmem: Int = 0,
                        _ bind: (MTLComputeCommandEncoder) -> Void) {
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        bind(e)
        if tgmem > 0 {
            let need = (tgmem + 15) & ~15
            precondition(need <= ctx.device.maxThreadgroupMemoryLength,
                "\(name): threadgroup memory \(need)B exceeds device max "
                + "\(ctx.device.maxThreadgroupMemoryLength)B")
            e.setThreadgroupMemoryLength(need, index: 0)
        }
        e.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        pop()
    }

    static let spanTypes: Set<GGUFType> = [.q8_0, .iq4_nl]

    func gemv(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer, off: WeightRef,
              xOff: Int = 0, outOff: Int = 0) {
        if Blocks.superBlocked(w.type) || MetalEnc.spanTypes.contains(w.type) {
            iqGemv(w, x: x, out: out, off: off, xOff: xOff, outOff: outOff)
        } else {
            packedGemv(w, x: x, out: out, off: off, xOff: xOff,
                       outOff: outOff)
        }
    }

    private func packedGemv(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer,
                            off: WeightRef, xOff: Int, outOff: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        let ty = w.type
        let name: String
        switch ty {
        case .q2_0: name = "q2_0_gemv"
        case .q4_0: name = "q4_0_gemv"
        case .bf16: name = "bf16_gemv"
        case .f16: name = "f16_gemv"
        case .f32: name = "f32_gemv"
        default:
            fatalError("MetalEnc.gemv: no kernel for \(ty) (\(w.name))")
        }
        let rows = ty == .f16 || ty == .f32 ? 1 : 8
        groups(name, (m + rows - 1) / rows, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(x, offset: xOff, index: 1)
            e.setBuffer(out, offset: outOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        }
    }

    private func iqGemv(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer,
                        off: WeightRef, xOff: Int, outOff: Int) {
        var a = iqArgs(off, w.type, k: w.dims[0], m: w.dims[1])
        let rows = 8
        let name: String
        switch w.type {
        case .iq1_s: name = "iq1_s_gemv"
        case .iq1_m: name = "iq1_m_gemv"
        case .q4k: name = "q4_k_gemv"
        case .q5k: name = "q5_k_gemv"
        case .q6k: name = "q6_k_gemv"
        case .q2k: name = "q2_k_gemv"
        case .q3k: name = "q3_k_gemv"
        case .iq2_xxs: name = "iq2_xxs_gemv"
        case .iq2_xs: name = "iq2_xs_gemv"
        case .iq2_s: name = "iq2_s_gemv"
        case .iq3_xxs: name = "iq3_xxs_gemv"
        case .iq3_s: name = "iq3_s_gemv"
        case .iq4_xs: name = "iq4_xs_gemv"
        case .q8_0: name = "q8_0_gemv"
        case .iq4_nl: name = "iq4_nl_gemv"
        default:
            fatalError("MetalEnc.gemv: no kernel for \(w.type) (\(w.name))")
        }
        groups(name, (w.dims[1] + rows - 1) / rows, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(x, offset: xOff, index: 1)
            e.setBuffer(out, offset: outOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<IQArgs>.stride, index: 3)
        }
    }

    private func iqArgs(_ off: WeightRef, _ ty: GGUFType,
                        k: Int, m: Int) -> IQArgs {
        IQArgs(woff: off.local, K: UInt32(k), M: UInt32(m),
               ty: UInt32(bitPattern: ty.rawValue),
               blk: UInt32(GGUF.rowByteCount(ty, Blocks.superBlock)))
    }


    func gemmRows(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                  off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        for j in 0 ..< N {
            gemv(w, x: X, out: out, off: off, xOff: j * k * 4,
                 outOff: j * m * 4)
        }
    }

    static let narrowMax = 5

    static func narrowRows() -> Int { 16 }

    func gemmNarrow(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                    off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        let rows = MetalEnc.narrowRows()
        let name = (MetalEnc.narrowPackedName(w.type) ?? "") + "\(N)"
        groups(name, (m + rows - 1) / rows, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(X, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        }
    }

    static func narrowPackedName(_ t: GGUFType) -> String? {
        let out: String?
        switch t {
        case .q4_0: out = "q4_0_gemm_nb_r"
        case .q2_0: out = "q2_0_gemm_nb_r"
        default: out = nil
        }
        return out
    }

    static func narrowIQName(_ t: GGUFType) -> String? {
        let out: String?
        switch t {
        case .q4k: out = "q4_k_gemm_nb_r"
        case .q5k: out = "q5_k_gemm_nb_r"
        case .q6k: out = "q6_k_gemm_nb_r"
        case .iq4_xs: out = "iq4_xs_gemm_nb_r"
        case .q8_0: out = "q8_0_gemm_nb_r"
        case .iq4_nl: out = "iq4_nl_gemm_nb_r"
        case .q3k: out = "q3_k_gemm_nb_r"
        case .iq2_s: out = "iq2_s_gemm_nb_r"
        case .iq3_xxs: out = "iq3_xxs_gemm_nb_r"
        case .iq3_s: out = "iq3_s_gemm_nb_r"
        case .iq2_xxs: out = "iq2_xxs_gemm_nb_r"
        case .iq2_xs: out = "iq2_xs_gemm_nb_r"
        case .iq1_s: out = "iq1_s_gemm_nb_r"
        case .iq1_m: out = "iq1_m_gemm_nb_r"
        default: out = nil
        }
        return out
    }

    func gemmNarrowIQ(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                      off: WeightRef, N: Int) {
        var a = iqArgs(off, w.type, k: w.dims[0], m: w.dims[1])
        let wide = w.dims[0] >= 8192 || (N == 3 && w.dims[1] >= 65536)
        let nw = N == 3 && MetalEnc.fusedTypes.contains(w.type)
        let sg = nw ? 2 : (N >= 4 || wide ? 8 : 1)
        let rows = 8 * sg
        let base = MetalEnc.narrowIQName(w.type)!
        let name = nw ? String(base.dropLast(4)) + "nw_r\(N)"
                      : base + "\(N)s\(sg)"
        groups(name, (w.dims[1] + rows - 1) / rows, tpg: nw ? 32 : 32 * sg,
               tgmem: nw ? 0 : N * 128 * 16) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(X, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<IQArgs>.stride, index: 3)
        }
    }

    func gemm(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer, off: WeightRef,
              N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        var nn = UInt32(N)
        let narrow = MetalEnc.narrowPackedName(w.type) != nil
            && N > 1 && N <= MetalEnc.narrowMax
        let kqNarrow = MetalEnc.narrowIQName(w.type) != nil
            && N > 1 && N <= MetalEnc.narrowMax
        let iqNarrow = Blocks.superBlocked(w.type)
            && N > 1 && N <= MetalEnc.narrowMax
        // The spill path needs 8192 B of f32 temp; a fully in-bounds tile (M %
        // 64, N % 32) never spills and 6144 B suffices.
        let fullTile = m % 64 == 0 && N % 32 == 0
        let name: String
        switch w.type {
        case .q2_0:
            name = "q2_0_gemm_mm_h"
        case .q4_0:
            name = "q4_0_gemm_mm_h"
        case .iq4_nl:
            name = "iq4_nl_gemm_mm_h"
        case .q8_0:
            name = "q8_0_gemm_mm_h"
        case .f16, .bf16, .f32:
            name = MetalEnc.denseTileName(w.type, K: k, N: N)
        default:
            precondition(Blocks.superBlocked(w.type),
                         "MetalEnc.gemm: no kernel for \(w.type) (\(w.name))")
            name = MetalEnc.kqTileName(w.type) ?? "iq_gemm_mm_h"
        }
        if N == 1 {
            gemmRows(w, X: X, out: out, off: off, N: N)
        } else if narrow {
            gemmNarrow(w, X: X, out: out, off: off, N: N)
        } else if kqNarrow {
            gemmNarrowIQ(w, X: X, out: out, off: off, N: N)
        } else if iqNarrow {
            gemmRows(w, X: X, out: out, off: off, N: N)
        } else if name.hasPrefix("iq_gemm_mm") {
            iqGemmTiled(name, w, X: X, out: out, off: off, N: N,
                        fullTile: fullTile)
        } else if name.isEmpty {
            gemmRows(w, X: X, out: out, off: off, N: N)
        } else {
            gemmTiled(name, w, X: X, out: out, off: off, N: N, a: &a, nn: &nn,
                      fullTile: fullTile)
        }
    }

    static let fusedTypes: Set<GGUFType> = [.q4k, .q5k, .q6k, .iq4_xs]

    static func kqTileName(_ t: GGUFType) -> String? {
        let out: String?
        switch t {
        case .q4k: out = "q4_k_gemm_mm_h"
        case .q5k: out = "q5_k_gemm_mm_h"
        case .q6k: out = "q6_k_gemm_mm_h"
        case .iq4_xs: out = "iq4_xs_gemm_mm_h"
        case .q3k: out = "q3_k_gemm_mm_h"
        case .iq2_s: out = "iq2_s_gemm_mm_h"
        case .iq3_xxs: out = "iq3_xxs_gemm_mm_h"
        case .iq3_s: out = "iq3_s_gemm_mm_h"
        case .iq2_xxs: out = "iq2_xxs_gemm_mm_h"
        case .iq2_xs: out = "iq2_xs_gemm_mm_h"
        case .iq1_s: out = "iq1_s_gemm_mm_h"
        case .iq1_m: out = "iq1_m_gemm_mm_h"
        default: out = nil
        }
        return out
    }

    static func denseTileName(_ t: GGUFType, K: Int, N: Int) -> String {
        var out = ""
        if N > MetalEnc.narrowMax && K % 32 == 0 {
            switch t {
            case .f16: out = "f16_gemm_mm_h"
            case .bf16: out = "bf16_gemm_mm_h"
            case .f32: out = "f32_gemm_mm_h"
            default: out = ""
            }
        }
        return out
    }

    func gemmTile(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer,
                  off: WeightRef, N: Int) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        var nn = UInt32(N)
        let fullTile = m % 64 == 0 && N % 32 == 0
        let name: String
        switch w.type {
        case .q2_0: name = "q2_0_gemm_mm_h"
        case .q4_0: name = "q4_0_gemm_mm_h"
        case .iq4_nl:
            name = "iq4_nl_gemm_mm_h"
        case .q8_0: name = "q8_0_gemm_mm_h"
        case .f16, .bf16, .f32:
            name = MetalEnc.denseTileName(w.type, K: k, N: 32)
        default:
            name = Blocks.superBlocked(w.type)
                ? MetalEnc.kqTileName(w.type) ?? "iq_gemm_mm_h" : ""
        }
        if name.hasPrefix("iq_gemm_mm") {
            iqGemmTiled(name, w, X: X, out: out, off: off, N: N,
                        fullTile: fullTile)
        } else if !name.isEmpty {
            gemmTiled(name, w, X: X, out: out, off: off, N: N, a: &a, nn: &nn,
                      fullTile: fullTile)
        }
    }

    private func iqGemmTiled(_ name: String, _ w: GGUFTensor, X: MTLBuffer,
                             out: MTLBuffer, off: WeightRef, N: Int,
                             fullTile: Bool) {
        let m = w.dims[1]
        var a = iqArgs(off, w.type, k: w.dims[0], m: m)
        var nn = UInt32(N)
        precondition(ctx.matrixUnits,
                     "MetalEnc.gemm: \(name) needs simdgroup matrix units")
        push(name)
        e.setComputePipelineState(try! ctx.pipeline(name))
        e.setBuffer(off.buf, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<IQArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        e.setThreadgroupMemoryLength(
            fullTile ? 6144 : 8192, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    private func gemmTiled(_ name: String, _ w: GGUFTensor, X: MTLBuffer,
                           out: MTLBuffer, off: WeightRef, N: Int,
                           a: inout GemvArgs, nn: inout UInt32,
                           fullTile: Bool) {
        let m = w.dims[1]
        precondition(ctx.matrixUnits,
                     "MetalEnc.gemm: \(name) needs simdgroup matrix units; "
                     + "this GPU has none, so its caller should have been "
                     + "gated (see Gemma4MetalBackend.supportsSoftTokens)")
        let tgmem = fullTile ? 6144 : 8192
        push(name)
        let pipe = try! ctx.pipeline(name)
        e.setComputePipelineState(pipe)
        e.setBuffer(off.buf, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        e.setThreadgroupMemoryLength(tgmem, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    // `outOff` is a BYTE offset, so a batched caller dequants into row r
    // directly.
    func dequantRow(weightOff: WeightRef, out: MTLBuffer, n: Int,
                    type: GGUFType = .q2_0, outOff: Int = 0) {
        if Blocks.superBlocked(type) {
            var a = iqArgs(weightOff, type, k: n, m: n)
            grid1D(MetalEnc.dqRowName(type),
                   n / Blocks.superBlock) { e in
                e.setBuffer(weightOff.buf, offset: 0, index: 0)
                e.setBuffer(out, offset: outOff, index: 1)
                e.setBytes(&a, length: MemoryLayout<IQArgs>.stride, index: 2)
            }
        } else {
            packedDequantRow(weightOff: weightOff, out: out, n: n,
                             type: type, outOff: outOff)
        }
    }

    static func dqRowName(_ t: GGUFType) -> String {
        let out: String
        switch t {
        case .q2k: out = "q2_k_dequant_row"
        case .q4k: out = "q4_k_dequant_row"
        case .q6k: out = "q6_k_dequant_row"
        default: out = "iq_dequant_row"
        }
        return out
    }

    static func embedName(_ t: GGUFType) -> String {
        let out: String
        switch t {
        case .q2k: out = "q2_k_embed_batch"
        case .q4k: out = "q4_k_embed_batch"
        case .q6k: out = "q6_k_embed_batch"
        default: out = "iq_embed_batch"
        }
        return out
    }

    private func packedDequantRow(weightOff: WeightRef, out: MTLBuffer,
                                  n: Int, type: GGUFType, outOff: Int) {
        var a = GemvArgs(woff: weightOff.local, K: UInt32(n), M: UInt32(n))
        let name: String
        switch type {
        case .q2_0: name = "q2_0_dequant_row"
        case .q4_0: name = "q4_0_dequant_row"
        case .q8_0: name = "q8_0_dequant_row"
        case .bf16: name = "bf16_dequant_row"
        case .f16: name = "f16_dequant_row"
        case .f32: name = "f32_dequant_row"
        default:
            fatalError("MetalEnc.dequantRow: no kernel for \(type)")
        }
        grid1D(name, n) { e in
            e.setBuffer(weightOff.buf, offset: 0, index: 0)
            e.setBuffer(out, offset: outOff, index: 1)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 2)
        }
    }

    func assistTopClusters(x: MTLBuffer, out: MTLBuffer, n: Int, k: Int) {
        var a = AssistTopArgs(n: UInt32(n), k: UInt32(k))
        groups("assist_top_clusters", 1, tpg: 256) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<AssistTopArgs>.stride,
                       index: 2)
        }
    }

    func assistClusterArgmax(_ w: GGUFTensor, h: MTLBuffer,
                             ordering: WeightRef, clusters: MTLBuffer,
                             out: MTLBuffer, off: WeightRef, dim: Int,
                             per: Int, k: Int) {
        var a = AssistPickArgs(
            woff: off.local, ooff: ordering.local, dim: UInt32(dim),
            per: UInt32(per), clusters: UInt32(k),
            rowBytes: UInt32(GGUF.rowByteCount(w.type, dim)))
        groups("assist_cluster_argmax", 1, tpg: 256) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(h, offset: 0, index: 1)
            e.setBuffer(ordering.buf, offset: 0, index: 2)
            e.setBuffer(clusters, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<AssistPickArgs>.stride,
                       index: 5)
        }
    }

    func argmaxRows(x: MTLBuffer, out: MTLBuffer, n: Int, rows: Int) {
        var nn = UInt32(n)
        groups("argmax_rows", rows, tpg: 256, tgmem: 0) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func rmsnorm(x: MTLBuffer, weightOff: WeightRef, out: MTLBuffer, n: Int,
                 eps: Float, xOff: Int = 0, outOff: Int = 0) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm", 1, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: xOff, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(out, offset: outOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func rmsnormBatch(x: MTLBuffer, weightOff: WeightRef, y: MTLBuffer, n: Int,
                      rows: Int, eps: Float, xOff: Int = 0, xStride: Int? = nil,
                      yOff: Int = 0, yStride: Int? = nil) {
        var a = NormBatchArgs(woff: weightOff.local, n: UInt32(n), eps: eps,
                              xStride: UInt32(xStride ?? n),
                              yStride: UInt32(yStride ?? n))
        groups("rmsnorm_batch", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: xOff, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(y, offset: yOff, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormBatchArgs>.stride,
                       index: 3)
        }
    }

    func addRmsnormBatch(x: MTLBuffer, r: MTLBuffer, weightOff: WeightRef,
                         y: MTLBuffer, n: Int, rows: Int, eps: Float) {
        var a = NormBatchArgs(woff: weightOff.local, n: UInt32(n), eps: eps,
                              xStride: UInt32(n), yStride: UInt32(n))
        groups("add_rmsnorm_batch", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(r, offset: 0, index: 1)
            e.setBuffer(weightOff.buf, offset: 0, index: 2)
            e.setBuffer(y, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<NormBatchArgs>.stride,
                       index: 4)
        }
    }

    func rmsnormRowsSilu(x: MTLBuffer, z: MTLBuffer, d: Int, rows: Int,
                         weightOff: WeightRef, eps: Float) {
        var a = RowArgs(woff: weightOff.local, d: UInt32(d), xoff: 0,
                        eps: eps)
        groups("rmsnorm_rows_silu", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(z, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 3)
        }
    }

    func rmsnormRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                     weightOff: WeightRef, eps: Float) {
        var a = RowArgs(woff: weightOff.local, d: UInt32(d), xoff: xoff,
                        eps: eps)
        groups("rmsnorm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 2)
        }
    }

    func l2normRows(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                    eps: Float) {
        var a = RowArgs(woff: 0, d: UInt32(d), xoff: xoff, eps: eps)
        groups("l2norm_rows", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 1)
        }
    }

    func l2normRowsBatch(x: MTLBuffer, d: Int, rowsPerTok: Int, tokStride: Int,
                         tokens: Int, eps: Float) {
        var a = RowBatchArgs(d: UInt32(d), rowsPerTok: UInt32(rowsPerTok),
                             tokStride: UInt32(tokStride), eps: eps)
        groups("l2norm_rows_batch", tokens * rowsPerTok, tpg: 128,
               tgmem: 128) {
            e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowBatchArgs>.stride, index: 1)
        }
    }

    func rmsnormBF16(x: MTLBuffer, weightOff: WeightRef, out: MTLBuffer,
                     n: Int, eps: Float) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm_bf16", 1, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func rmsnormRowsBF16(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                         weightOff: WeightRef, eps: Float) {
        var a = RowArgs(woff: weightOff.local, d: UInt32(d), xoff: xoff,
                        eps: eps)
        groups("rmsnorm_rows_bf16", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 2)
        }
    }

    func rmsnormRowsNoWeight(x: MTLBuffer, xoff: UInt32, d: Int, rows: Int,
                             eps: Float) {
        var a = RowArgs(woff: 0, d: UInt32(d), xoff: xoff, eps: eps)
        groups("rmsnorm_rows_noweight", rows, tpg: 128, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RowArgs>.stride, index: 1)
        }
    }

    func scaleInPlace(x: MTLBuffer, n: Int, s: Float, off: Int = 0) {
        var nn = UInt32(n)
        var ss = s
        grid1D("scale_inplace", n) { e in
            e.setBuffer(x, offset: off, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
            e.setBytes(&ss, length: 4, index: 2)
        }
    }

    func addScaled(a: MTLBuffer, b: MTLBuffer, n: Int, s: Float) {
        var nn = UInt32(n)
        var ss = s
        grid1D("add_scaled", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
            e.setBytes(&ss, length: 4, index: 3)
        }
    }

    func geluMul(a: MTLBuffer, b: MTLBuffer, n: Int, bOff: Int = 0) {
        var g = GeluMulArgs(n: UInt32(n), boff: UInt32(bOff))
        grid1D("gelu_mul", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&g, length: MemoryLayout<GeluMulArgs>.stride, index: 2)
        }
    }

    func softcap(x: MTLBuffer, n: Int, cap: Float) {
        var nn = UInt32(n)
        var cc = cap
        grid1D("softcap", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
            e.setBytes(&cc, length: 4, index: 2)
        }
    }

    func rmsnormBatchBF16(x: MTLBuffer, weightOff: WeightRef, y: MTLBuffer,
                          n: Int, rows: Int, eps: Float) {
        var a = NormArgs(woff: weightOff.local, n: UInt32(n), eps: eps)
        groups("rmsnorm_batch_bf16", rows, tpg: 256, tgmem: 128) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(weightOff.buf, offset: 0, index: 1)
            e.setBuffer(y, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<NormArgs>.stride, index: 3)
        }
    }

    func gemmaVisionRope(x: MTLBuffer, cos: MTLBuffer, sin: MTLBuffer,
                         rowStride: Int, headDim: Int, nHead: Int, N: Int) {
        var a = GVRopeArgs(rowStride: UInt32(rowStride), off: 0,
                           headDim: UInt32(headDim), nHead: UInt32(nHead),
                           N: UInt32(N))
        grid1D("gemma_vit_rope", N * nHead * (headDim / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(cos, offset: 0, index: 1)
            e.setBuffer(sin, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GVRopeArgs>.stride, index: 3)
        }
    }

    func gemmaVisionAttn(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                         mask: MTLBuffer, out: MTLBuffer, n: Int, hd: Int,
                         nHead: Int) {
        var a = GVAttnArgs(n: UInt32(n), hd: UInt32(hd),
                           nHead: UInt32(nHead))
        groups("gemma_vit_attn", n * nHead, tpg: 32,
               tgmem: hd * MemoryLayout<Float>.stride) { e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(k, offset: 0, index: 1)
            e.setBuffer(v, offset: 0, index: 2)
            e.setBuffer(mask, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<GVAttnArgs>.stride, index: 5)
        }
    }

    func activateMul(_ act: GemmaActivation, a: MTLBuffer, b: MTLBuffer,
                     n: Int, bOff: Int = 0) {
        switch act {
        case .geluTanh: geluMul(a: a, b: b, n: n, bOff: bOff)
        case .silu:
            precondition(bOff == 0, "silu_mul takes no b offset")
            siluMul(a: a, b: b, n: n)
        }
    }

    func activateMulRows(_ act: GemmaActivation, a: MTLBuffer, b: MTLBuffer,
                         n: Int, rows: Int, aStride: Int, bStride: Int,
                         bOff: Int) {
        precondition(act == .geluTanh,
                     "gelu_mul_rows is the only strided form; \(act) would "
                     + "need its own kernel")
        var g = GeluMulRowsArgs(n: UInt32(n), rows: UInt32(rows),
                                aStride: UInt32(aStride),
                                bStride: UInt32(bStride), boff: UInt32(bOff))
        grid1D("gelu_mul_rows", rows * n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&g, length: MemoryLayout<GeluMulRowsArgs>.stride,
                       index: 2)
        }
    }

    func activateInPlace(_ act: GemmaActivation, x: MTLBuffer, n: Int) {
        switch act {
        case .geluTanh: gelu(x: x, n: n)
        case .silu: siluInPlace(x: x, n: n)
        }
    }

    // A scale of 0 is an uncalibrated linear whose clamp is the identity, so it
    // is skipped rather than dispatched.
    func srqInPlace(x: MTLBuffer, n: Int, s: SRQ.Side) {
        if s.active {
            var a = SrqArgs(n: UInt32(n), s: s.scale, lo: s.lo, hi: s.hi)
            grid1D("srq_inplace", n) { e in
                e.setBuffer(x, offset: 0, index: 0)
                e.setBytes(&a, length: MemoryLayout<SrqArgs>.stride, index: 1)
            }
        }
    }

    func srqTo(src: MTLBuffer, dst: MTLBuffer, n: Int, s: SRQ.Side) {
        var a = SrqArgs(n: UInt32(n), s: s.scale, lo: s.lo, hi: s.hi)
        grid1D("srq_to", n) { e in
            e.setBuffer(src, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<SrqArgs>.stride, index: 2)
        }
    }

    // The input is clamped into `scratch` because it is usually a shared buffer
    // (one norm feeds q, k and v); the output belongs to this linear alone.
    func linear(_ w: GGUFTensor, X: MTLBuffer, out: MTLBuffer, off: WeightRef,
                N: Int, srq: SRQ, scratch: MTLBuffer) {
        var input = X
        if srq.input.active {
            srqTo(src: X, dst: scratch, n: N * w.dims[0], s: srq.input)
            input = scratch
        }
        gemm(w, X: input, out: out, off: off, N: N)
        srqInPlace(x: out, n: N * w.dims[1], s: srq.output)
    }

    func linear(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer, off: WeightRef,
                srq: SRQ, scratch: MTLBuffer) {
        let clamped = srq.input.active || srq.output.active
        if clamped && w.type == .q4_0 {
            gemvSrq(w, x: x, out: out, off: off, srq: srq)
        } else {
            let grouped = sync.grouped
            if grouped { e.memoryBarrier(scope: .buffers) }
            sync.grouped = false
            var input = x
            if srq.input.active {
                srqTo(src: x, dst: scratch, n: w.dims[0], s: srq.input)
                input = scratch
            }
            gemv(w, x: input, out: out, off: off)
            srqInPlace(x: out, n: w.dims[1], s: srq.output)
            sync.grouped = grouped
        }
    }

    func gemvSrq(_ w: GGUFTensor, x: MTLBuffer, out: MTLBuffer,
                 off: WeightRef, srq: SRQ) {
        let k = w.dims[0], m = w.dims[1]
        var a = GemvArgs(woff: off.local, K: UInt32(k), M: UInt32(m))
        var c = SrqPair(sIn: srq.input.scale, loIn: srq.input.lo,
                        hiIn: srq.input.hi, sOut: srq.output.scale,
                        loOut: srq.output.lo, hiOut: srq.output.hi)
        groups("q4_0_gemv_srq", (m + 7) / 8, tpg: 32) { e in
            e.setBuffer(off.buf, offset: 0, index: 0)
            e.setBuffer(x, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<GemvArgs>.stride, index: 3)
            e.setBytes(&c, length: MemoryLayout<SrqPair>.stride, index: 4)
        }
    }

    func normAddNormBF16(x: MTLBuffer, c: MTLBuffer, post: WeightRef,
                         pre: WeightRef?, y: MTLBuffer, n: Int, rows: Int,
                         eps: Float, s: Float) {
        var a = NormAddArgs(woffPost: post.local, woffPre: pre?.local ?? 0,
                            n: UInt32(n), eps: eps, s: s,
                            pre: pre == nil ? 0 : 1)
        groups("norm_add_norm_bf16", rows, tpg: 256, tgmem: 256) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(c, offset: 0, index: 1)
            e.setBuffer(post.buf, offset: 0, index: 2)
            e.setBuffer((pre ?? post).buf, offset: 0, index: 3)
            e.setBuffer(y, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<NormAddArgs>.stride,
                       index: 5)
        }
    }

    func siluInPlace(x: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("silu_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
        }
    }

    func fmaInPlace(x: MTLBuffer, y: MTLBuffer, n: Int, s: Float) {
        var a = FmaArgs(n: UInt32(n), s: s)
        grid1D("fma_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(y, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<FmaArgs>.stride, index: 2)
        }
    }

    func gluRows(src: MTLBuffer, dst: MTLBuffer, rows: Int, width: Int) {
        var a = GluArgs(n: UInt32(rows), e: UInt32(width))
        grid1D("glu_rows", rows * width) { e in
            e.setBuffer(src, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<GluArgs>.stride, index: 2)
        }
    }

    func depthwiseCausal(x: MTLBuffer, w: MTLBuffer, out: MTLBuffer,
                         rows: Int, width: Int, k: Int) {
        var a = DwArgs(n: UInt32(rows), e: UInt32(width), k: UInt32(k))
        grid1D("depthwise_causal", rows * width) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<DwArgs>.stride, index: 3)
        }
    }

    func scaleHeadDims(x: MTLBuffer, scales: MTLBuffer, rows: Int,
                       width: Int, hd: Int) {
        var a = HeadScaleArgs(n: UInt32(rows), e: UInt32(width),
                              hd: UInt32(hd))
        grid1D("scale_head_dims", rows * width) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(scales, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<HeadScaleArgs>.stride,
                       index: 2)
        }
    }

    func gemmaAudioAttn(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                        relk: MTLBuffer, out: MTLBuffer, rows: Int,
                        width: Int, heads: Int, hd: Int, chunk: Int,
                        context: Int, past: Int, cap: Float) {
        precondition(context <= 32,
                     "gemma_audio_attn stages one window's scores in a "
                     + "32-slot thread array")
        var a = AAttnArgs(n: UInt32(rows), e: UInt32(width),
                          heads: UInt32(heads), hd: UInt32(hd),
                          chunk: UInt32(chunk), ctx: UInt32(context),
                          past: UInt32(past), cap: cap)
        grid1D("gemma_audio_attn", rows * heads) { e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(k, offset: 0, index: 1)
            e.setBuffer(v, offset: 0, index: 2)
            e.setBuffer(relk, offset: 0, index: 3)
            e.setBuffer(out, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<AAttnArgs>.stride, index: 5)
        }
    }

    func ropeGemma(x: MTLBuffer, headDim: Int, nHead: Int, rotated: Int,
                   base: Float, pos: Int) {
        var a = RopeGemmaArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                              rotated: UInt32(rotated), base: base,
                              pos: UInt32(pos))
        grid1D("rope_gemma", nHead * rotated) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeGemmaArgs>.stride,
                       index: 1)
        }
    }

    func rope(x: MTLBuffer, headDim: Int, nHead: Int, nRot: Int, base: Float,
              pos: Int) {
        var a = RopeArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                         nRot: UInt32(nRot), base: base, pos: UInt32(pos))
        grid1D("rope_neox", nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeArgs>.stride, index: 1)
        }
    }

    func siluMul(a: MTLBuffer, b: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("silu_mul", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func mulSilu(a: MTLBuffer, b: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("mul_silu", n) { e in
            e.setBuffer(a, offset: 0, index: 0)
            e.setBuffer(b, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func accumOuter(h: MTLBuffer, src: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        push("accum_outer")
        let pipe = try! ctx.pipeline("accum_outer")
        e.setComputePipelineState(pipe)
        e.setBuffer(h, offset: 0, index: 0)
        e.setBuffer(src, offset: 0, index: 1)
        e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: n, height: n, depth: 1),
                          threadsPerThreadgroup:
                            MTLSize(width: 32, height: 8, depth: 1))
        pop()
    }

    func accumSq(dst: MTLBuffer, src: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("accum_sq", n) { e in
            e.setBuffer(dst, offset: 0, index: 0)
            e.setBuffer(src, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    func add(x: MTLBuffer, y: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("add_inplace", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(y, offset: 0, index: 1)
            e.setBytes(&nn, length: 4, index: 2)
        }
    }

    // The one dispatch reading TWO weights: ssm_dt and ssm_a are separate
    // tensors.
    func gdnGate(bPre: MTLBuffer, aPre: MTLBuffer, dtOff: WeightRef,
                 aOff: WeightRef, beta: MTLBuffer, g: MTLBuffer, nV: Int,
                 count: Int) {
        var a = GateArgs(dtOff: dtOff.local, aOff: aOff.local, nV: UInt32(nV))
        var total = UInt32(count)
        grid1D("gdn_gate", count) { e in
            e.setBuffer(bPre, offset: 0, index: 0)
            e.setBuffer(aPre, offset: 0, index: 1)
            e.setBuffer(dtOff.buf, offset: 0, index: 2)
            e.setBuffer(aOff.buf, offset: 0, index: 3)
            e.setBuffer(beta, offset: 0, index: 4)
            e.setBuffer(g, offset: 0, index: 5)
            e.setBytes(&a, length: MemoryLayout<GateArgs>.stride, index: 6)
            e.setBytes(&total, length: 4, index: 7)
        }
    }

    func gdnConv(qkvMix: MTLBuffer, convState: MTLBuffer, cwOff: WeightRef,
                 out: MTLBuffer, convDim: Int, dConv: Int) {
        var a = ConvArgs(cwOff: cwOff.local, convDim: UInt32(convDim),
                         dConv: UInt32(dConv))
        grid1D("gdn_conv", convDim) { e in
            e.setBuffer(qkvMix, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(cwOff.buf, offset: 0, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvArgs>.stride, index: 4)
        }
    }

    func splitQGate(qFull: MTLBuffer, q: MTLBuffer, gate: MTLBuffer, hd: Int,
                    nH: Int) {
        var a = SplitArgs(hd: UInt32(hd), nH: UInt32(nH))
        grid1D("split_qgate", hd * nH) { e in
            e.setBuffer(qFull, offset: 0, index: 0)
            e.setBuffer(q, offset: 0, index: 1)
            e.setBuffer(gate, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<SplitArgs>.stride, index: 3)
        }
    }

    func kvAppend(kCur: MTLBuffer, vCur: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
                  kvDim: Int, pos: Int) {
        var a = KVArgs(kvDim: UInt32(kvDim), pos: UInt32(pos))
        grid1D("kv_append", kvDim) { e in
            e.setBuffer(kCur, offset: 0, index: 0)
            e.setBuffer(vCur, offset: 0, index: 1)
            e.setBuffer(K, offset: 0, index: 2)
            e.setBuffer(V, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<KVArgs>.stride, index: 4)
        }
    }

    // A barrier orders the tail append (directly bound, hazard-tracked) before
    // the bindless page reads.
    func attnPaged(q: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                   pages: [MTLResource], gate: MTLBuffer, out: MTLBuffer,
                   hd: Int, nH: Int, nKV: Int, T: Int, kvDim: Int, P: Int,
                   scale: Float, gated: Int, lo: Int = 0) {
        precondition(hd % 4 == 0 && hd <= 512,
                     "attn_head: half4 V slices need hd % 4 == 0, and the "
                     + "four-slice accumulator caps head dim at 512")
        var a = AttnArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                         T: UInt32(T), kvDim: UInt32(kvDim), P: UInt32(P),
                         scale: scale, gated: UInt32(gated), lo: UInt32(lo))
        e.memoryBarrier(scope: .buffers)
        e.useResources(pages, usage: .read)
        let tgmem = (5 * hd + 8) * MemoryLayout<Float>.stride
        let keys = T - lo
        let want = min(MetalEnc.attnSplits, (keys + 127) / 128)
        if want > 1 {
            attnSplit(q: q, kAddr: kAddr, vAddr: vAddr, gate: gate, out: out,
                      hd: hd, nH: nH, nKV: nKV, T: T, kvDim: kvDim, P: P,
                      scale: scale, gated: gated, lo: lo, want: want,
                      tgmem: tgmem)
        } else {
            groups("attn_head", nH, tpg: 128, tgmem: tgmem) {
                e in
                e.setBuffer(q, offset: 0, index: 0)
                e.setBuffer(kAddr, offset: 0, index: 1)
                e.setBuffer(vAddr, offset: 0, index: 2)
                e.setBuffer(gate, offset: 0, index: 3)
                e.setBuffer(out, offset: 0, index: 4)
                e.setBytes(&a, length: MemoryLayout<AttnArgs>.stride,
                           index: 5)
            }
        }
    }

    static let attnSplits = 8

    private func attnSplit(q: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                           gate: MTLBuffer, out: MTLBuffer, hd: Int, nH: Int,
                           nKV: Int, T: Int, kvDim: Int, P: Int, scale: Float,
                           gated: Int, lo: Int, want: Int, tgmem: Int) {
        let keys = T - lo
        let span = ((keys + want - 1) / want + 31) / 32 * 32
        let splits = (keys + span - 1) / span
        let parts = ctx.scratch(nH * splits * (hd + 2)
                                * MemoryLayout<Float>.stride)
        var a = AttnSplitArgs(hd: UInt32(hd), nH: UInt32(nH),
                              nKV: UInt32(nKV), T: UInt32(T),
                              kvDim: UInt32(kvDim), P: UInt32(P),
                              scale: scale, gated: UInt32(gated),
                              lo: UInt32(lo), splits: UInt32(splits),
                              span: UInt32(span))
        groups("attn_head_part", nH * splits, tpg: 128, tgmem: tgmem) { e in
            e.setBuffer(q, offset: 0, index: 0)
            e.setBuffer(kAddr, offset: 0, index: 1)
            e.setBuffer(vAddr, offset: 0, index: 2)
            e.setBuffer(parts, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<AttnSplitArgs>.stride,
                       index: 4)
        }
        groups("attn_reduce", nH, tpg: 32) { e in
            e.setBuffer(parts, offset: 0, index: 0)
            e.setBuffer(gate, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<AttnSplitArgs>.stride,
                       index: 3)
        }
    }

    func embedBatch(ids: MTLBuffer, weightOff: WeightRef, out: MTLBuffer,
                    nEmbd: Int, N: Int, type: GGUFType = .q2_0) {
        if Blocks.superBlocked(type) {
            iqEmbedBatch(ids: ids, weightOff: weightOff, out: out,
                         nEmbd: nEmbd, N: N, type: type)
        } else {
            packedEmbedBatch(ids: ids, weightOff: weightOff, out: out,
                             nEmbd: nEmbd, N: N, type: type)
        }
    }

    private func iqEmbedBatch(ids: MTLBuffer, weightOff: WeightRef,
                              out: MTLBuffer, nEmbd: Int, N: Int,
                              type: GGUFType) {
        var a = IQEmbedArgs(
            woff: weightOff.local, K: UInt32(nEmbd), M: UInt32(N),
            rowBytes: UInt32(GGUF.rowByteCount(type, nEmbd)),
            ty: UInt32(bitPattern: type.rawValue),
            blk: UInt32(GGUF.rowByteCount(type, Blocks.superBlock)))
        grid1D(MetalEnc.embedName(type),
               N * (nEmbd / Blocks.superBlock)) { e in
            e.setBuffer(weightOff.buf, offset: 0, index: 0)
            e.setBuffer(ids, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<IQEmbedArgs>.stride, index: 3)
        }
    }

    private func packedEmbedBatch(ids: MTLBuffer, weightOff: WeightRef,
                                  out: MTLBuffer, nEmbd: Int, N: Int,
                                  type: GGUFType) {
        var a = EmbedArgs(woff: weightOff.local, K: UInt32(nEmbd),
                          M: UInt32(N),
                          rowBytes: UInt32(GGUF.rowByteCount(type, nEmbd)))
        let name: String
        switch type {
        case .q2_0: name = "embed_batch"
        case .q4_0: name = "q4_0_embed_batch"
        case .q8_0: name = "q8_0_embed_batch"
        case .bf16: name = "bf16_embed_batch"
        case .f16: name = "f16_embed_batch"
        case .f32: name = "f32_embed_batch"
        default:
            fatalError("MetalEnc.embedBatch: no kernel for \(type)")
        }
        grid1D(name, N * nEmbd) { e in
            e.setBuffer(weightOff.buf, offset: 0, index: 0)
            e.setBuffer(ids, offset: 0, index: 1)
            e.setBuffer(out, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<EmbedArgs>.stride, index: 3)
        }
    }

    func gdnConvBatch(qkvMixN: MTLBuffer, convState: MTLBuffer,
                      cwOff: WeightRef, outN: MTLBuffer, convDim: Int,
                      dConv: Int, N: Int, ring: StateRing = .inPlace) {
        var a = ConvBatchArgs(cwOff: cwOff.local, convDim: UInt32(convDim),
                              dConv: UInt32(dConv), N: UInt32(N),
                              slot0: UInt32(ring.slot0),
                              slots: UInt32(ring.slots),
                              stateElems: UInt32(convDim * (dConv - 1)))
        grid1D("gdn_conv_batch", convDim) { e in
            e.setBuffer(qkvMixN, offset: 0, index: 0)
            e.setBuffer(convState, offset: 0, index: 1)
            e.setBuffer(cwOff.buf, offset: 0, index: 2)
            e.setBuffer(outN, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<ConvBatchArgs>.stride,
                       index: 4)
        }
    }

    func gdnScanBatch(convOutN: MTLBuffer, keyDim: Int, valueDim: Int,
                      convDim: Int, gN: MTLBuffer, betaN: MTLBuffer,
                      S: MTLBuffer, oN: MTLBuffer, nV: Int, nK: Int, dS: Int,
                      qScale: Float, N: Int, ring: StateRing = .inPlace) {
        var a = ScanBatchArgs(nV: UInt32(nV), nK: UInt32(nK), dS: UInt32(dS),
                              qScale: qScale, N: UInt32(N),
                              convDim: UInt32(convDim), keyDim: UInt32(keyDim),
                              valueDim: UInt32(valueDim),
                              slot0: UInt32(ring.slot0),
                              slots: UInt32(ring.slots),
                              stateElems: UInt32(nV * dS * dS))
        let bind = { (e: MTLComputeCommandEncoder) in
            e.setBuffer(convOutN, offset: 0, index: 0)
            e.setBuffer(gN, offset: 0, index: 1)
            e.setBuffer(betaN, offset: 0, index: 2)
            e.setBuffer(S, offset: 0, index: 3)
            e.setBuffer(oN, offset: 0, index: 4)
            e.setBytes(&a, length: MemoryLayout<ScanBatchArgs>.stride,
                       index: 5)
        }
        if [64, 128, 256].contains(dS) {
            let name = "gdn_scan_batch_k\(dS / 32)"
            push(name)
            let pipe = try! ctx.pipeline(name)
            e.setComputePipelineState(pipe)
            bind(e)
            e.dispatchThreadgroups(
                MTLSize(width: dS / 4, height: nV, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1,
                                               depth: 1))
            pop()
        } else {
            grid1D("gdn_scan_batch", nV * dS, bind)
        }
    }

    func splitQGateBatch(qFullN: MTLBuffer, qN: MTLBuffer, gateN: MTLBuffer,
                         hd: Int, nH: Int, N: Int) {
        var a = SplitArgs(hd: UInt32(hd), nH: UInt32(nH))
        var nn = UInt32(N)
        grid1D("split_qgate_batch", N * hd * nH) { e in
            e.setBuffer(qFullN, offset: 0, index: 0)
            e.setBuffer(qN, offset: 0, index: 1)
            e.setBuffer(gateN, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<SplitArgs>.stride, index: 3)
            e.setBytes(&nn, length: 4, index: 4)
        }
    }

    func ropeGemmaBatch(x: MTLBuffer, headDim: Int, nHead: Int, rotated: Int,
                        base: Float, basePos: Int, N: Int) {
        var a = RopeGemmaBatchArgs(headDim: UInt32(headDim),
                                   nHead: UInt32(nHead),
                                   rotated: UInt32(rotated), base: base,
                                   basePos: UInt32(basePos), N: UInt32(N))
        grid1D("rope_gemma_batch", N * nHead * rotated) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeGemmaBatchArgs>.stride,
                       index: 1)
        }
    }

    func ropeBatch(x: MTLBuffer, headDim: Int, nHead: Int, nRot: Int,
                   base: Float, basePos: Int, N: Int) {
        var a = RopeBatchArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                              nRot: UInt32(nRot), base: base,
                              basePos: UInt32(basePos), N: UInt32(N))
        grid1D("rope_batch", N * nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&a, length: MemoryLayout<RopeBatchArgs>.stride,
                       index: 1)
        }
    }

    func ropeMBatch(x: MTLBuffer, pos3: MTLBuffer, headDim: Int, nHead: Int,
                    nRot: Int, base: Float, N: Int) {
        var a = RopeMBatchArgs(headDim: UInt32(headDim), nHead: UInt32(nHead),
                               nRot: UInt32(nRot), base: base, N: UInt32(N))
        grid1D("rope_mrope_batch", N * nHead * (nRot / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(pos3, offset: 0, index: 1)
            e.setBytes(&a, length: MemoryLayout<RopeMBatchArgs>.stride,
                       index: 2)
        }
    }

    func kvAppendBatch(kCurN: MTLBuffer, vCurN: MTLBuffer, kAddr: MTLBuffer,
                       vAddr: MTLBuffer, pages: [MTLResource], kvDim: Int,
                       basePos: Int, P: Int, N: Int) {
        var a = KVBatchArgs(kvDim: UInt32(kvDim), basePos: UInt32(basePos),
                            P: UInt32(P), N: UInt32(N))
        e.useResources(pages, usage: .write)
        grid1D("kv_append_batch", N * kvDim) { e in
            e.setBuffer(kCurN, offset: 0, index: 0)
            e.setBuffer(vCurN, offset: 0, index: 1)
            e.setBuffer(kAddr, offset: 0, index: 2)
            e.setBuffer(vAddr, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<KVBatchArgs>.stride, index: 4)
        }
    }

    func f16Gemm(_ w: MTLBuffer, X: MTLBuffer, out: MTLBuffer, K: Int, M: Int,
                 N: Int) {
        var a = F16WArgs(K: UInt32(K), M: UInt32(M))
        var nn = UInt32(N)
        // The spill path needs 8192 B of f32 temp; a fully in-bounds tile (M %
        // 64, N % 32) never spills and 6144 B suffices.
        let fullTile = M % 64 == 0 && N % 32 == 0
        precondition(ctx.matrixUnits,
                     "MetalEnc.f16Gemm: f16w_gemm_mm needs simdgroup matrix "
                     + "units; this GPU has none, so QwenMetalViT should never "
                     + "have been built (see QwenMetalBackend.supportsVision)")
        push("f16w_gemm_mm")
        let pipe = try! ctx.pipeline("f16w_gemm_mm")
        e.setComputePipelineState(pipe)
        e.setBuffer(w, offset: 0, index: 0)
        e.setBuffer(X, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&a, length: MemoryLayout<F16WArgs>.stride, index: 3)
        e.setBytes(&nn, length: 4, index: 4)
        e.setThreadgroupMemoryLength(fullTile ? 6144 : 8192, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 31) / 32, height: (M + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        pop()
    }

    func layerNorm(x: MTLBuffer, w: MTLBuffer, b: MTLBuffer, y: MTLBuffer,
                   n: Int, rows: Int, eps: Float) {
        var a = LNArgs(n: UInt32(n), eps: eps)
        groups("vit_layernorm", rows, tpg: 256, tgmem: 256) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBuffer(b, offset: 0, index: 2)
            e.setBuffer(y, offset: 0, index: 3)
            e.setBytes(&a, length: MemoryLayout<LNArgs>.stride, index: 4)
        }
    }

    func addBiasRows(x: MTLBuffer, bias: MTLBuffer, m: Int, rows: Int) {
        var mm = UInt32(m)
        var total = UInt32(rows * m)
        grid1D("add_bias_rows", rows * m) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(bias, offset: 0, index: 1)
            e.setBytes(&mm, length: 4, index: 2)
            e.setBytes(&total, length: 4, index: 3)
        }
    }

    func gelu(x: MTLBuffer, n: Int) {
        var nn = UInt32(n)
        grid1D("gelu_tanh", n) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBytes(&nn, length: 4, index: 1)
        }
    }

    func visionRope(x: MTLBuffer, cos: MTLBuffer, sin: MTLBuffer,
                    rowStride: Int, off: Int, headDim: Int, nHead: Int,
                    N: Int) {
        var a = VRopeArgs(rowStride: UInt32(rowStride), off: UInt32(off),
                          headDim: UInt32(headDim), nHead: UInt32(nHead),
                          N: UInt32(N))
        grid1D("vit_rope", N * nHead * (headDim / 2)) { e in
            e.setBuffer(x, offset: 0, index: 0)
            e.setBuffer(cos, offset: 0, index: 1)
            e.setBuffer(sin, offset: 0, index: 2)
            e.setBytes(&a, length: MemoryLayout<VRopeArgs>.stride, index: 3)
        }
    }

    func visionAttn(qkv: MTLBuffer, out: MTLBuffer, n: Int, embd: Int,
                    hd: Int, nHead: Int, scale: Float) {
        precondition(hd <= 128, "vit_attn lane slice caps head dim at 128")
        var a = VAttnArgs(n: UInt32(n), e: UInt32(embd), hd: UInt32(hd),
                          nHead: UInt32(nHead), scale: scale)
        let tq = 16
        push("vit_attn")
        let pipe = try! ctx.pipeline("vit_attn")
        e.setComputePipelineState(pipe)
        e.setBuffer(qkv, offset: 0, index: 0)
        e.setBuffer(out, offset: 0, index: 1)
        e.setBytes(&a, length: MemoryLayout<VAttnArgs>.stride, index: 2)
        e.setThreadgroupMemoryLength(
            ((2 * 32 + tq) * hd * MemoryLayout<Float>.stride + 15) & ~15,
            index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (n + tq - 1) / tq, height: nHead, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tq * 32, height: 1,
                                           depth: 1))
        pop()
    }

    // The matrix kernel reads a whole FA_Q=8 tile of rows whatever N is, so
    // `blocks` has to answer 8 rows past the batch.
    func attnBatch(qN: MTLBuffer, kAddr: MTLBuffer, vAddr: MTLBuffer,
                   pages: [MTLResource], gateN: MTLBuffer, outN: MTLBuffer,
                   hd: Int, nH: Int, nKV: Int, kvDim: Int, P: Int,
                   scale: Float,
                   basePos: Int, N: Int, gated: Int, window: Int = 0,
                   blocks: MTLBuffer? = nil,
                   matrix: Bool = true) {
        precondition(hd % 4 == 0 && hd <= 512,
                     "attn_batch: half4 V slices need hd % 4 == 0, and the "
                     + "four-slice accumulator caps head dim at 512")
        var a = AttnBatchArgs(hd: UInt32(hd), nH: UInt32(nH), nKV: UInt32(nKV),
                              kvDim: UInt32(kvDim), P: UInt32(P), scale: scale,
                              basePos: UInt32(basePos), N: UInt32(N),
                              gated: UInt32(gated), window: UInt32(window))
        let blk = blocks ?? ctx.noBlocks(N + 8)
        e.memoryBarrier(scope: .buffers)
        e.useResources(pages, usage: .read)
        // An 8-key block is ONE strided simdgroup_load, contiguous only inside
        // a page, so P is a multiple of 8; the P=4 self-test pool proves it.
        let mm = matrix && ctx.matrixUnits
            && hd % 8 == 0 && (hd / 4) % 8 == 0 && P % 8 == 0
        if mm {
            // qs[8*hd half] | s[8*32 f32] | p[8*32 half] | m[8] | l[8] |
            // corr[64 half]
            let tgmem = 8 * hd * 2 + 256 * 4 + 256 * 2 + 8 * 4 + 8 * 4 + 64 * 2
            let tiles = (N + 7) / 8
            groups("attn_batch_mm", tiles * nH, tpg: 128, tgmem: tgmem) { e in
                e.setBuffer(qN, offset: 0, index: 0)
                e.setBuffer(kAddr, offset: 0, index: 1)
                e.setBuffer(vAddr, offset: 0, index: 2)
                e.setBuffer(gateN, offset: 0, index: 3)
                e.setBuffer(outN, offset: 0, index: 4)
                e.setBytes(&a, length: MemoryLayout<AttnBatchArgs>.stride,
                           index: 5)
                e.setBuffer(blk, offset: 0, index: 6)
            }
        } else {
            let tgmem = (5 * hd + 8) * MemoryLayout<Float>.stride
            groups("attn_batch", N * nH, tpg: 128, tgmem: tgmem) { e in
                e.setBuffer(qN, offset: 0, index: 0)
                e.setBuffer(kAddr, offset: 0, index: 1)
                e.setBuffer(vAddr, offset: 0, index: 2)
                e.setBuffer(gateN, offset: 0, index: 3)
                e.setBuffer(outN, offset: 0, index: 4)
                e.setBytes(&a, length: MemoryLayout<AttnBatchArgs>.stride,
                           index: 5)
                e.setBuffer(blk, offset: 0, index: 6)
            }
        }
    }
}

// Mirrors of the MSL kargs structs, same field order and types. Weight byte
// offsets are UInt64 (a weight buffer exceeds 4 GB) and lead for alignment.
struct NormArgs { var woff: UInt64; var n: UInt32; var eps: Float }
struct NormBatchArgs {
    var woff: UInt64; var n: UInt32; var eps: Float
    var xStride: UInt32; var yStride: UInt32
}
struct RowArgs {
    var woff: UInt64; var d: UInt32; var xoff: UInt32; var eps: Float
}
struct RowBatchArgs {
    var d: UInt32; var rowsPerTok: UInt32; var tokStride: UInt32
    var eps: Float
}
struct RopeGemmaBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var rotated: UInt32
    var base: Float; var basePos: UInt32; var N: UInt32
}
struct RopeArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var pos: UInt32
}
struct GeluMulArgs { var n: UInt32; var boff: UInt32 }
struct GeluMulRowsArgs {
    var n: UInt32; var rows: UInt32; var aStride: UInt32
    var bStride: UInt32; var boff: UInt32
}
struct GVAttnArgs { var n: UInt32; var hd: UInt32; var nHead: UInt32 }
struct GVRopeArgs {
    var rowStride: UInt32; var off: UInt32; var headDim: UInt32
    var nHead: UInt32; var N: UInt32
}
struct RopeGemmaArgs {
    var headDim: UInt32; var nHead: UInt32; var rotated: UInt32
    var base: Float; var pos: UInt32
}
struct AssistTopArgs { var n: UInt32; var k: UInt32 }
struct AssistPickArgs {
    var woff: UInt64; var ooff: UInt64
    var dim: UInt32; var per: UInt32; var clusters: UInt32
    var rowBytes: UInt32
}
struct FmaArgs { var n: UInt32; var s: Float }
struct SrqArgs { var n: UInt32; var s: Float; var lo: Float; var hi: Float }
struct SrqPair {
    var sIn: Float; var loIn: Float; var hiIn: Float
    var sOut: Float; var loOut: Float; var hiOut: Float
}
struct NormAddArgs {
    var woffPost: UInt64; var woffPre: UInt64; var n: UInt32; var eps: Float
    var s: Float; var pre: UInt32
}
struct GluArgs { var n: UInt32; var e: UInt32 }
struct DwArgs { var n: UInt32; var e: UInt32; var k: UInt32 }
struct HeadScaleArgs { var n: UInt32; var e: UInt32; var hd: UInt32 }
struct AAttnArgs {
    var n: UInt32; var e: UInt32; var heads: UInt32; var hd: UInt32
    var chunk: UInt32; var ctx: UInt32; var past: UInt32; var cap: Float
}
struct GateArgs { var dtOff: UInt64; var aOff: UInt64; var nV: UInt32 }
struct ConvArgs { var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32 }
struct SplitArgs { var hd: UInt32; var nH: UInt32 }
struct AttnArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32
    var T: UInt32; var kvDim: UInt32; var P: UInt32; var scale: Float
    var gated: UInt32; var lo: UInt32
}
struct AttnSplitArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32
    var T: UInt32; var kvDim: UInt32; var P: UInt32; var scale: Float
    var gated: UInt32; var lo: UInt32; var splits: UInt32; var span: UInt32
}
struct KVArgs { var kvDim: UInt32; var pos: UInt32 }
struct ConvBatchArgs {
    var cwOff: UInt64; var convDim: UInt32; var dConv: UInt32; var N: UInt32
    var slot0: UInt32; var slots: UInt32; var stateElems: UInt32
}
struct ScanBatchArgs {
    var nV: UInt32; var nK: UInt32; var dS: UInt32; var qScale: Float
    var N: UInt32; var convDim: UInt32; var keyDim: UInt32
    var valueDim: UInt32
    var slot0: UInt32; var slots: UInt32; var stateElems: UInt32
}
struct RopeBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var basePos: UInt32; var N: UInt32
}
struct RopeMBatchArgs {
    var headDim: UInt32; var nHead: UInt32; var nRot: UInt32
    var base: Float; var N: UInt32
}
struct KVBatchArgs {
    var kvDim: UInt32; var basePos: UInt32; var P: UInt32; var N: UInt32
}
struct AttnBatchArgs {
    var hd: UInt32; var nH: UInt32; var nKV: UInt32; var kvDim: UInt32
    var P: UInt32; var scale: Float; var basePos: UInt32; var N: UInt32
    var gated: UInt32; var window: UInt32
}
struct EmbedArgs {
    var woff: UInt64; var K: UInt32; var M: UInt32; var rowBytes: UInt32
}
struct IQArgs {
    var woff: UInt64; var K: UInt32; var M: UInt32
    var ty: UInt32; var blk: UInt32
}
struct IQEmbedArgs {
    var woff: UInt64; var K: UInt32; var M: UInt32; var rowBytes: UInt32
    var ty: UInt32; var blk: UInt32
}
struct F16WArgs { var K: UInt32; var M: UInt32 }
struct LNArgs { var n: UInt32; var eps: Float }
struct VRopeArgs {
    var rowStride: UInt32; var off: UInt32; var headDim: UInt32
    var nHead: UInt32; var N: UInt32
}
struct VAttnArgs {
    var n: UInt32; var e: UInt32; var hd: UInt32; var nHead: UInt32
    var scale: Float
}
