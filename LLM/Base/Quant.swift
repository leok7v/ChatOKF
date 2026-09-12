import Accelerate
// Q2_0 block (34 bytes, 128 weights): { f16 d; u8 qs[32] }, scale FIRST;
// 2-bit codes LSB-first, weight = (code - 1) * d, the sign lives in the code.
import Foundation

enum Q2_0 {
    static let qk = 128
    static let blockBytes = 34

    static func dequant(_ base: UnsafeRawPointer, count n: Int, into out: UnsafeMutablePointer<Float>) {
        let nb = n / qk
        var p = base
        var o = 0
        for _ in 0..<nb {
            let d = Float(p.loadUnaligned(as: Float16.self))
            let qs = p + 2
            for j in 0..<qk {
                let byte = qs.load(fromByteOffset: j >> 2, as: UInt8.self)
                let code = Int((byte >> UInt8((j & 3) << 1)) & 0x03)
                out[o + j] = Float(code - 1) * d
            }
            p += blockBytes
            o += qk
        }
    }

    // Per block dot = d * (lo + 2 * hi - sumy), which keeps it multiply-light.
    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        let K = w.dims[0]
        let M = w.dims[1]
        precondition(K % qk == 0)
        let nblk = K / qk
        let rowBytes = nblk * blockBytes
        // Each iteration writes its own row and only reads immutable pointers,
        // which strict concurrency cannot prove for raw pointers; hence unsafe.
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                nonisolated(unsafe) let base = w.base
                nonisolated(unsafe) let xp = xb.baseAddress!
                nonisolated(unsafe) let op = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: M) { m in
                    var acc: Float = 0
                    var p = base + m * rowBytes
                    var k = 0
                    for _ in 0..<nblk {
                        let d = Float(p.loadUnaligned(as: Float16.self))
                        let qs = p + 2
                        var lo: Float = 0, hi: Float = 0, sy: Float = 0
                        for j in 0..<qk {
                            let byte = qs.load(fromByteOffset: j >> 2, as: UInt8.self)
                            let code = (byte >> UInt8((j & 3) << 1)) & 0x03
                            let xv = xp[k + j]
                            sy += xv
                            if code & 1 != 0 { lo += xv }
                            if code & 2 != 0 { hi += xv }
                        }
                        acc += d * (lo + 2 * hi - sy)
                        p += blockBytes
                        k += qk
                    }
                    op[m] = acc
                }
            }
        }
    }
}

enum QB {
    static func matvec(_ w: GGUFTensor, x: [Float], out: inout [Float]) {
        GQ.matvec(w, x: x, out: &out)
    }

    static func dequant(_ w: GGUFTensor, row: Int, count n: Int,
                        into out: UnsafeMutablePointer<Float>) {
        GQ.gather(w, row: row, from: 0, count: n, into: out)
    }
}

enum Dense {
    // ggml stores [out][in] row-major; vDSP_mmul wants [in][out].
    static func transposedFloats(_ t: GGUFTensor, _ inDim: Int,
                                 _ outDim: Int) -> [Float] {
        var flat: [Float]
        switch t.type {
        case .q2_0, .q4_0:
            flat = [Float](repeating: 0, count: inDim * outDim)
            var row = [Float](repeating: 0, count: inDim)
            for m in 0..<outDim {
                GQ.dequantSpan(t, row: m, from: 0, count: inDim, into: &row)
                for i in 0..<inDim { flat[m * inDim + i] = row[i] }
            }
        default:
            flat = floats(t)
        }
        var out = [Float](repeating: 0, count: flat.count)
        vDSP_mtrans(flat, 1, &out, 1, vDSP_Length(inDim), vDSP_Length(outDim))
        return out
    }

    // F32 is WRAPPED IN PLACE over the mapping, so the GGUF must outlive every
    // tensor made this way; F16 widens into the arena; a block type gives nil.
    static func tensor(_ t: GGUFTensor, into arena: Arena) -> Tensor? {
        var out: Tensor? = nil
        var ne: [Int64] = [1, 1, 1, 1]
        for d in 0..<t.dims.count { ne[d] = Int64(t.dims[d]) }
        let nDims = Int32(t.dims.count)
        if t.type == .f32 {
            let data = UnsafeMutableRawPointer(mutating: t.base)
                .assumingMemoryBound(to: Float.self)
            out = tensorWrapNd(arena, nDims, data, ne)
        } else if t.type == .f16 {
            let o = tensorNewNd(arena, nDims, ne)
            let total = Int(tensorNelements(o))
            for i in 0..<total {
                let h = (t.base + i * 2).loadUnaligned(as: UInt16.self)
                o.data[i] = Float(Float16(bitPattern: h))
            }
            out = o
        }
        if let o = out { tensorSetName(o, t.name) }
        return out
    }

    static func floats(_ t: GGUFTensor) -> [Float] {
        let n = t.count
        return [Float](unsafeUninitializedCapacity: n) { buf, cnt in
            switch t.type {
            case .f32:
                memcpy(buf.baseAddress!, t.base, n * 4)
            case .f16:
                let src = t.base.assumingMemoryBound(to: Float16.self)
                for i in 0..<n { buf[i] = Float(src[i]) }
            case .bf16:
                let src = t.base.assumingMemoryBound(to: UInt16.self)
                for i in 0..<n {
                    buf[i] = Float(bitPattern: UInt32(src[i]) << 16)
                }
            case .q8_0:
                var p = t.base
                var o = 0
                for _ in 0..<(n / 32) {
                    let d = Float(p.loadUnaligned(as: Float16.self))
                    for j in 0..<32 {
                        let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                        buf[o + j] = d * Float(q)
                    }
                    p += 34
                    o += 32
                }
            default:
                fatalError("Dense.floats: unsupported type \(t.type) "
                    + "for \(t.name)")
            }
            cnt = n
        }
    }
}

enum F32T {
    static func array(_ t: GGUFTensor) -> [Float] {
        precondition(t.type == .f32)
        let n = t.count
        return [Float](unsafeUninitializedCapacity: n) { buf, cnt in
            memcpy(buf.baseAddress!, t.base, n * 4); cnt = n
        }
    }
    static func ptr(_ t: GGUFTensor) -> UnsafePointer<Float> {
        precondition(t.type == .f32)
        return t.base.assumingMemoryBound(to: Float.self)
    }
}
