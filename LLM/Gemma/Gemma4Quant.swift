// Q4_0: 32 weights / 18 bytes {f16 d; u8 qs[16]}, byte j -> elements j (low)
// and j+16 (high), w = (q - 8) * d. Q8_0: 34 bytes {f16 d; i8 qs[32]}.
import Foundation

enum IQ4_NL {
    static let qk = 32
    static let blockBytes = 18
    static let kvalues: [Float] = [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113,
    ]

    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        var p = base
        var o = 0
        for _ in 0..<(n / qk) {
            let d = Float(p.loadUnaligned(as: Float16.self))
            for j in 0..<16 {
                let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                out[o + j] = kvalues[Int(b & 0x0F)] * d
                out[o + j + 16] = kvalues[Int(b >> 4)] * d
            }
            p += blockBytes
            o += qk
        }
    }
}

enum Q4_0 {
    static let qk = 32
    static let blockBytes = 18

    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        var p = base
        var o = 0
        for _ in 0..<(n / qk) {
            let d = Float(p.loadUnaligned(as: Float16.self))
            for j in 0..<16 {
                let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                out[o + j] = (Float(b & 0x0F) - 8) * d
                out[o + j + 16] = (Float(b >> 4) - 8) * d
            }
            p += blockBytes
            o += qk
        }
    }
}

enum Q8_0 {
    static let qk = 32
    static let blockBytes = 34

    static func dequant(_ base: UnsafeRawPointer, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        var p = base
        var o = 0
        for _ in 0..<(n / qk) {
            let d = Float(p.loadUnaligned(as: Float16.self))
            for j in 0..<qk {
                let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                out[o + j] = Float(q) * d
            }
            p += blockBytes
            o += qk
        }
    }
}

enum GQ {

    static func rowBytes(_ t: GGUFTensor) -> Int {
        GGUF.rowByteCount(t.type, t.dims[0])
    }

    static func blockElems(_ t: GGUFType) -> Int {
        let out: Int
        switch t {
        case .q2k, .q3k, .q4k, .q5k, .q6k, .q8k,
             .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq3_s, .iq2_s,
             .iq4_xs, .iq1_m: out = Blocks.superBlock
        case .q2_0: out = 128
        default: out = 32
        }
        return out
    }

    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        switch w.type {
        case .q2_0:
            Q2_0.matvec(w, x: x, out: &out)
        case .q4_0:
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var acc: Float = 0
                let d = Float(p.loadUnaligned(as: Float16.self))
                for j in 0..<16 {
                    let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    acc += (Float(b & 0x0F) - 8) * xp[k + j]
                    acc += (Float(b >> 4) - 8) * xp[k + j + 16]
                }
                return acc * d
            }
        case .iq4_nl:
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var acc: Float = 0
                let d = Float(p.loadUnaligned(as: Float16.self))
                for j in 0..<16 {
                    let b = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    acc += IQ4_NL.kvalues[Int(b & 0x0F)] * xp[k + j]
                    acc += IQ4_NL.kvalues[Int(b >> 4)] * xp[k + j + 16]
                }
                return acc * d
            }
        case .q8_0:
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var acc: Float = 0
                let d = Float(p.loadUnaligned(as: Float16.self))
                for j in 0..<32 {
                    let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                    acc += Float(q) * xp[k + j]
                }
                return acc * d
            }
        case .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs,
             .iq3_s, .iq4_xs, .q2k, .q3k, .q4k, .q5k, .q6k:
            let decode = Blocks.decoder(w.type)!
            rowParallel(w, x: x, out: &out) { p, xp, k in
                var w = [Float](repeating: 0, count: 256)
                var acc: Float = 0
                w.withUnsafeMutableBufferPointer { b in
                    decode(p, b.baseAddress!)
                    for i in 0..<256 { acc += b[i] * xp[k + i] }
                }
                return acc
            }
        case .bf16:
            denseMatvec(w, x: x, out: &out)
        case .f16:
            denseMatvec(w, x: x, out: &out)
        case .f32:
            denseMatvec(w, x: x, out: &out)
        default:
            fatalError("GQ.matvec: unsupported type \(w.type) for \(w.name)")
        }
    }

    // The point is DEQUANT REUSE: a mat-vec unpacks a block for ONE dot
    // product; here a block is unpacked once and serves all N rows.
    static func matmul(_ w: GGUFTensor, X: [Float], N: Int,
                       out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        switch w.type {
        case .q4_0, .q8_0:
            blockMatmul(w, X: X, N: N, K: K, M: M, out: &out)
        case .bf16, .f16, .f32:
            denseMatmul(w, X: X, N: N, K: K, M: M, out: &out)
        default:
            // A type with no batched twin still answers one row at a time; only
            // the reuse is lost.
            var row = [Float](repeating: 0, count: K)
            var acc = [Float](repeating: 0, count: M)
            for n in 0..<N {
                for i in 0..<K { row[i] = X[n * K + i] }
                matvec(w, x: row, out: &acc)
                for m in 0..<M { out[n * M + m] = acc[m] }
            }
        }
    }

    private static func blockMatmul(_ w: GGUFTensor, X: [Float], N: Int,
                                    K: Int, M: Int, out: inout [Float]) {
        let q4 = w.type == .q4_0
        let per = q4 ? Q4_0.qk : Q8_0.qk
        let size = q4 ? Q4_0.blockBytes : Q8_0.blockBytes
        let nblk = K / per
        let stride = nblk * size
        // X transposed to [K][N] so the innermost accumulate walks N
        // contiguously against a broadcast weight, the shape that vectorizes.
        var XT = [Float](repeating: 0, count: N * K)
        X.withUnsafeBufferPointer { xb in
            XT.withUnsafeMutableBufferPointer { tb in
                for n in 0..<N {
                    for k in 0..<K { tb[k * N + n] = xb[n * K + k] }
                }
            }
        }
        XT.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xt = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    // Stack scratch; allocating per m would cost more than the
                    // arithmetic it holds.
                    withUnsafeTemporaryAllocation(of: Float.self,
                                                  capacity: per + N) { tmp in
                        let blk = tmp.baseAddress!
                        let acc = blk + per
                        for n in 0..<N { acc[n] = 0 }
                        var p = base + m * stride
                        var k = 0
                        for _ in 0..<nblk {
                            if q4 {
                                Q4_0.dequant(p, count: per, into: blk)
                            } else {
                                Q8_0.dequant(p, count: per, into: blk)
                            }
                            for j in 0..<per {
                                let wj = blk[j]
                                let col = xt + (k + j) * N
                                for n in 0..<N { acc[n] += wj * col[n] }
                            }
                            p += size
                            k += per
                        }
                        for n in 0..<N { op[n * M + m] = acc[n] }
                    }
                }
            }
        }
    }

    private static func denseMatmul(_ w: GGUFTensor, X: [Float], N: Int,
                                    K: Int, M: Int, out: inout [Float]) {
        let stride = rowBytes(w)
        let ty = w.type
        X.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let p = base + m * stride
                    withUnsafeTemporaryAllocation(of: Float.self,
                                                  capacity: N) { acc in
                        for n in 0..<N { acc[n] = 0 }
                        for i in 0..<K {
                            let e = element(p, ty, i)
                            for n in 0..<N { acc[n] += e * xp[n * K + i] }
                        }
                        for n in 0..<N { op[n * M + m] = acc[n] }
                    }
                }
            }
        }
    }

    private static func rowParallel(
        _ w: GGUFTensor, x: [Float], out: inout [Float],
        _ block: @escaping @Sendable (UnsafeRawPointer,
                                      UnsafePointer<Float>, Int) -> Float
    ) {
        let K = w.dims[0]
        let M = w.dims[1]
        let per = blockElems(w.type)
        let nblk = K / per
        let stride = GGUF.rowByteCount(w.type, K)
        let size = stride / nblk
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    var acc: Float = 0
                    var p = base + m * stride
                    var k = 0
                    for _ in 0..<nblk {
                        acc += block(p, xp, k)
                        p += size
                        k += per
                    }
                    op[m] = acc
                }
            }
        }
    }

    private static func denseMatvec(_ w: GGUFTensor, x: [Float],
                                    out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        let stride = rowBytes(w)
        let ty = w.type
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    let p = base + m * stride
                    var acc: Float = 0
                    for i in 0..<K { acc += element(p, ty, i) * xp[i] }
                    op[m] = acc
                }
            }
        }
    }

    private static func element(_ p: UnsafeRawPointer, _ ty: GGUFType,
                                _ i: Int) -> Float {
        let v: Float
        switch ty {
        case .f32:
            v = p.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
        case .f16:
            v = Float(p.loadUnaligned(fromByteOffset: i * 2,
                                      as: Float16.self))
        default:
            let bits = p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
            v = Float(bitPattern: UInt32(bits) << 16)
        }
        return v
    }

    // `from` and `count` must land on block boundaries (256 and 1536 divide
    // both 32 and 128).
    static func dequantSpan(_ t: GGUFTensor, row: Int, from: Int, count: Int,
                            into out: inout [Float]) {
        out.withUnsafeMutableBufferPointer { ob in
            gather(t, row: row, from: from, count: count,
                   into: ob.baseAddress!)
        }
    }

    // Straight into caller memory, never an MTLBuffer, so the tables stay out
    // of the wired set.
    static func gather(_ t: GGUFTensor, row: Int, from: Int, count: Int,
                       into o: UnsafeMutablePointer<Float>) {
        let base = t.base + row * rowBytes(t)
        switch t.type {
        case .q2_0:
            Q2_0.dequant(base + from / Q2_0.qk * Q2_0.blockBytes,
                         count: count, into: o)
        case .q4_0:
            Q4_0.dequant(base + from / Q4_0.qk * Q4_0.blockBytes,
                         count: count, into: o)
        case .iq4_nl:
            IQ4_NL.dequant(base + from / IQ4_NL.qk * IQ4_NL.blockBytes,
                           count: count, into: o)
        case .q8_0:
            Q8_0.dequant(base + from / Q8_0.qk * Q8_0.blockBytes,
                         count: count, into: o)
        default:
            superBlockGather(t, base, from: from, count: count, into: o)
        }
    }

    private static func superBlockGather(_ t: GGUFTensor,
                                         _ base: UnsafeRawPointer,
                                         from: Int, count: Int,
                                         into o: UnsafeMutablePointer<Float>) {
        if let decode = Blocks.decoder(t.type) {
            let per = Blocks.superBlock
            let size = GGUF.rowByteCount(t.type, per)
            for b in 0..<(count / per) {
                decode(base + (from / per + b) * size, o + b * per)
            }
        } else {
            for i in 0..<count { o[i] = element(base, t.type, from + i) }
        }
    }
}

struct SRQ {

    struct Side {
        let scale: Float
        let lo: Float
        let hi: Float

        static let none = Side(scale: 0, lo: 0, hi: 0)

        var active: Bool { scale != 0 || lo < hi }
    }

    let input: Side
    let output: Side

    static let none = SRQ(input: .none, output: .none)

    init(input: Side, output: Side) {
        self.input = input
        self.output = output
    }

    static let enabled = Flags.on("srq")

    init(_ g: GGUF, _ weight: String) {
        let site = weight.replacingOccurrences(of: ".weight", with: "")
        input = SRQ.side(g, site, "in")
        output = SRQ.side(g, site, "out")
    }

    private static func side(_ g: GGUF, _ site: String,
                             _ slot: String) -> Side {
        var out = Side.none
        if SRQ.enabled {
            let scale = Float(g.double("gemma4.srq.\(site).\(slot)") ?? 0)
            let lo = g.double("gemma4.clamp.\(site).\(slot)_lo")
            let hi = g.double("gemma4.clamp.\(site).\(slot)_hi")
            if let lo, let hi {
                out = Side(scale: 0, lo: Float(lo), hi: Float(hi))
            } else if scale != 0 {
                out = Side(scale: scale, lo: 0, hi: 0)
            }
        }
        return out
    }

    // clamp(round(x / s), -128, 127) * s. torch.round is round-half-to-EVEN,
    // so anything else drifts on exact .5 -- which the trained scales hit.
    static func apply(_ x: inout [Float], _ s: Side) {
        if s.scale != 0 {
            let inv = 1 / s.scale
            for i in 0..<x.count {
                let q = (x[i] * inv).rounded(.toNearestOrEven)
                x[i] = min(max(q, -128), 127) * s.scale
            }
        } else if s.lo < s.hi {
            for i in 0..<x.count {
                x[i] = min(max(x[i], s.lo), s.hi)
            }
        }
    }
}
