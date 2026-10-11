import Dispatch
import Foundation

enum EncoderKernels {

    static let cores = ProcessInfo.processInfo.activeProcessorCount

    static func dequant(_ t: GGUFTensor, _ start: Int, _ count: Int,
                        into buf: inout [Float]) {
        buf.withUnsafeMutableBufferPointer { bp in
            dequant(t, start, count, into: bp.baseAddress!)
        }
    }

    static func dequant(_ t: GGUFTensor, _ start: Int, _ count: Int,
                        into buf: UnsafeMutablePointer<Float>) {
        let base = t.base
        switch t.type {
        case .f32:
            _ = memcpy(buf, base + start * 4, count * 4)
        case .f16, .bf16:
            for i in 0..<count {
                let h = base.loadUnaligned(
                    fromByteOffset: (start + i) * 2, as: UInt16.self)
                buf[i] = t.type == .f16
                    ? Float(Float16(bitPattern: h))
                    : Float(bitPattern: UInt32(h) << 16)
            }
        case .q4_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 18
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    let lo = Int(byte & 0x0f) - 8
                    let hi = Int(byte >> 4) - 8
                    buf[b * 32 + j] = Float(lo) * d
                    buf[b * 32 + j + 16] = Float(hi) * d
                }
            }
        case .q4_1:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 20
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                let m = Float(Float16(bitPattern:
                    p.loadUnaligned(fromByteOffset: 2, as: UInt16.self)))
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: 4 + j, as: UInt8.self)
                    buf[b * 32 + j] = Float(byte & 0x0f) * d + m
                    buf[b * 32 + j + 16] = Float(byte >> 4) * d + m
                }
            }
        case .q5_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 22
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                let qh = p.loadUnaligned(fromByteOffset: 2, as: UInt32.self)
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: 6 + j, as: UInt8.self)
                    let top = Int((qh >> UInt32(j)) & 1) << 4
                    let topHi = Int((qh >> UInt32(j + 16)) & 1) << 4
                    let lo = Int(byte & 0x0f) | top
                    let hi = Int(byte >> 4) | topHi
                    buf[b * 32 + j] = Float(lo - 16) * d
                    buf[b * 32 + j + 16] = Float(hi - 16) * d
                }
            }
        case .q8_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 34
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<32 {
                    let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                    buf[b * 32 + j] = Float(q) * d
                }
            }
        default:
            preconditionFailure("encoder: unsupported ggml type \(t.type)")
        }
    }

    static func dequant(_ t: GGUFTensor) -> [Float] {
        var out = [Float](repeating: 0, count: t.count)
        dequant(t, 0, t.count, into: &out)
        return out
    }

    private static func quad(_ p: UnsafeRawPointer,
                             _ i: Int) -> SIMD4<Float> {
        p.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)
    }

    static func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                    _ n: Int) -> Float {
        let ra = UnsafeRawPointer(a)
        let rb = UnsafeRawPointer(b)
        var q0 = SIMD4<Float>(repeating: 0)
        var q1 = SIMD4<Float>(repeating: 0)
        var q2 = SIMD4<Float>(repeating: 0)
        var q3 = SIMD4<Float>(repeating: 0)
        var i = 0
        while i + 16 <= n {
            q0 += quad(ra, i) * quad(rb, i)
            q1 += quad(ra, i + 4) * quad(rb, i + 4)
            q2 += quad(ra, i + 8) * quad(rb, i + 8)
            q3 += quad(ra, i + 12) * quad(rb, i + 12)
            i += 16
        }
        while i + 4 <= n {
            q0 += quad(ra, i) * quad(rb, i)
            i += 4
        }
        var total = ((q0 + q1) + (q2 + q3)).sum()
        while i < n {
            total += a[i] * b[i]
            i += 1
        }
        return total
    }

    static func linearBatch(_ w: GGUFTensor, _ b: [Float]?, _ x: [Float],
                            T: Int, inn: Int, out: Int,
                            _ y: inout [Float]) {
        let lanes = min(cores, out)
        let span = (out + lanes - 1) / lanes
        let bias = b ?? [Float](repeating: 0, count: out)
        nonisolated(unsafe) let weight = w
        x.withUnsafeBufferPointer { xp in
            bias.withUnsafeBufferPointer { bp in
                y.withUnsafeMutableBufferPointer { yp in
                    nonisolated(unsafe) let xb = xp.baseAddress!
                    nonisolated(unsafe) let bb = bp.baseAddress!
                    nonisolated(unsafe) let yb = yp.baseAddress!
                    DispatchQueue.concurrentPerform(
                        iterations: lanes) { lane in
                        var row = [Float](repeating: 0, count: inn)
                        let first = lane * span
                        let last = min(out, first + span)
                        row.withUnsafeMutableBufferPointer { rp in
                            let rb = rp.baseAddress!
                            for o in first..<last {
                                dequant(weight, o * inn, inn, into: rb)
                                for t in 0..<T {
                                    yb[t * out + o] = dot(
                                        rb, xb + t * inn, inn) + bb[o]
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static func layerNorm(_ x: inout [Float], off: Int, n: Int, eps: Float,
                          _ w: [Float], _ b: [Float]?) {
        var mean: Float = 0
        for i in 0..<n { mean += x[off + i] }
        mean /= Float(n)
        var varr: Float = 0
        for i in 0..<n {
            let d = x[off + i] - mean
            varr += d * d
        }
        varr /= Float(n)
        let inv = 1 / (varr + eps).squareRoot()
        if let b {
            for i in 0..<n {
                x[off + i] = (x[off + i] - mean) * inv * w[i] + b[i]
            }
        } else {
            for i in 0..<n { x[off + i] = (x[off + i] - mean) * inv * w[i] }
        }
    }

    private static func softmax(_ s: UnsafeMutablePointer<Float>, n: Int) {
        var mx = s[0]
        for i in 1..<n where s[i] > mx { mx = s[i] }
        var sum: Float = 0
        for i in 0..<n {
            s[i] = expf(s[i] - mx)
            sum += s[i]
        }
        let inv = 1 / sum
        for i in 0..<n { s[i] *= inv }
    }

    private static func transposed(_ k: [Float], T: Int, ne: Int) -> [Float] {
        var kT = [Float](repeating: 0, count: T * ne)
        for j in 0..<T {
            for c in 0..<ne { kT[c * T + j] = k[j * ne + c] }
        }
        return kT
    }

    private static func wide(_ p: UnsafeRawPointer,
                             _ i: Int) -> SIMD16<Float> {
        p.loadUnaligned(fromByteOffset: i * 4, as: SIMD16<Float>.self)
    }

    private static func scores(_ qi: UnsafePointer<Float>,
                               _ kT: UnsafePointer<Float>,
                               _ s: UnsafeMutablePointer<Float>,
                               count: Int, stride: Int, hd: Int,
                               scale: Float) {
        let kr = UnsafeRawPointer(kT)
        let sr = UnsafeMutableRawPointer(s)
        var j = 0
        while j + 16 <= count {
            var acc = SIMD16<Float>(repeating: 0)
            for d in 0..<hd { acc += qi[d] * wide(kr, d * stride + j) }
            sr.storeBytes(of: acc * scale, toByteOffset: j * 4,
                          as: SIMD16<Float>.self)
            j += 16
        }
        while j + 4 <= count {
            var acc = SIMD4<Float>(repeating: 0)
            for d in 0..<hd { acc += qi[d] * quad(kr, d * stride + j) }
            sr.storeBytes(of: acc * scale, toByteOffset: j * 4,
                          as: SIMD4<Float>.self)
            j += 4
        }
        while j < count {
            var acc: Float = 0
            for d in 0..<hd { acc += qi[d] * kT[d * stride + j] }
            s[j] = acc * scale
            j += 1
        }
    }

    private static func weighted(_ s: UnsafePointer<Float>,
                                 _ v: UnsafePointer<Float>,
                                 _ c: UnsafeMutablePointer<Float>,
                                 count: Int, ne: Int, hd: Int) {
        for d in 0..<hd { c[d] = 0 }
        let cr = UnsafeMutableRawPointer(c)
        for j in 0..<count {
            let wj = s[j]
            let vj = UnsafeRawPointer(v + j * ne)
            var d = 0
            while d + 16 <= hd {
                cr.storeBytes(of: wide(cr, d) + wj * wide(vj, d),
                              toByteOffset: d * 4, as: SIMD16<Float>.self)
                d += 16
            }
            while d + 4 <= hd {
                cr.storeBytes(of: quad(cr, d) + wj * quad(vj, d),
                              toByteOffset: d * 4, as: SIMD4<Float>.self)
                d += 4
            }
            while d < hd {
                c[d] += wj * v[j * ne + d]
                d += 1
            }
        }
    }

    private static func attendRows(_ q: UnsafePointer<Float>,
                                   _ kT: UnsafePointer<Float>,
                                   _ v: UnsafePointer<Float>,
                                   _ c: UnsafeMutablePointer<Float>,
                                   rows: Range<Int>, T: Int, ne: Int,
                                   hd: Int, scale: Float, window: Int) {
        var s = [Float](repeating: 0, count: T)
        s.withUnsafeMutableBufferPointer { sp in
            let sb = sp.baseAddress!
            for i in rows {
                let lo = window > 0 ? max(0, i - window) : 0
                let hi = window > 0 ? min(T, i + window + 1) : T
                scores(q + i * ne, kT + lo, sb, count: hi - lo, stride: T,
                       hd: hd, scale: scale)
                softmax(sb, n: hi - lo)
                weighted(sb, v + lo * ne, c + i * ne, count: hi - lo,
                         ne: ne, hd: hd)
            }
        }
    }

    static func attention(_ q: [Float], _ k: [Float], _ vv: [Float],
                          _ ctx: inout [Float], T: Int, ne: Int,
                          heads: Int, scale: Float, window: Int = 0) {
        let hd = ne / heads
        let kT = transposed(k, T: T, ne: ne)
        let block = 32
        let blocks = (T + block - 1) / block
        q.withUnsafeBufferPointer { qp in
            kT.withUnsafeBufferPointer { kp in
                vv.withUnsafeBufferPointer { vp in
                    ctx.withUnsafeMutableBufferPointer { cp in
                        nonisolated(unsafe) let qb = qp.baseAddress!
                        nonisolated(unsafe) let kb = kp.baseAddress!
                        nonisolated(unsafe) let vb = vp.baseAddress!
                        nonisolated(unsafe) let cb = cp.baseAddress!
                        DispatchQueue.concurrentPerform(
                            iterations: heads * blocks) { item in
                            let off = (item / blocks) * hd
                            let first = (item % blocks) * block
                            attendRows(
                                qb + off, kb + off * T, vb + off, cb + off,
                                rows: first ..< min(T, first + block),
                                T: T, ne: ne, hd: hd, scale: scale,
                                window: window)
                        }
                    }
                }
            }
        }
    }

    static func pooled(_ x: [Float], width ne: Int, into out: inout [Float]) {
        let T = x.count / ne
        if T > 0 {
            for d in 0..<ne { out[d] = 0 }
            for t in 0..<T {
                for d in 0..<ne { out[d] += x[t * ne + d] }
            }
            let invT = 1 / Float(T)
            for d in 0..<ne { out[d] *= invT }
            var ss: Float = 0
            for d in 0..<ne { ss += out[d] * out[d] }
            let inv = ss > 0 ? 1 / ss.squareRoot() : 0
            for d in 0..<ne { out[d] *= inv }
        }
    }
}
