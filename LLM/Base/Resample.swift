import CoreGraphics
import Foundation

// Pillow's separable bicubic, a = -0.5 where GGML and PyTorch use -0.75;
// the 22-bit fixed point and half-unit bias are Pillow's own arithmetic.

public enum Resample {
    // 32 bits less 8 for a pixel and 2 of headroom: a cubic overshoots, so a
    // tap sum reaches about 255 * 1.2 * 2^22.
    private static let bits = 22

    private struct Step {
        let tap: Int
        let line: Int
    }

    private struct Kernel {
        let size: Int
        let start: [Int]
        let count: [Int]
        let weight: [Int32]
    }

    public static func bicubicRGB(_ img: CGImage, _ w: Int,
                                  _ h: Int) -> [UInt8]? {
        var out: [UInt8]? = nil
        if var rows = raster(img) {
            if w != img.width {
                rows = convolve(rows, kernel(img.width, w),
                                lines: img.height,
                                from: Step(tap: 4, line: img.width * 4),
                                to: Step(tap: 4, line: w * 4))
            }
            if h != img.height {
                rows = convolve(rows, kernel(img.height, h), lines: w,
                                from: Step(tap: w * 4, line: 4),
                                to: Step(tap: w * 4, line: 4))
            }
            out = packed(rows, w * h)
        }
        return out
    }

    // Pillow's cubic: a = -0.5, zero outside [-2, 2].

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let t = abs(x)
        var w = 0.0
        if t < 1 {
            w = ((a + 2) * t - (a + 3)) * t * t + 1
        } else if t < 2 {
            w = (((t - 5) * t + 8) * t - 4) * a
        }
        return w
    }

    private static func kernel(_ inSize: Int, _ outSize: Int) -> Kernel {
        let scale = Double(inSize) / Double(outSize)
        let spread = max(scale, 1.0)
        let support = 2.0 * spread
        let size = Int(support.rounded(.up)) * 2 + 1
        let unit = Double(1 << bits)
        var start = [Int](repeating: 0, count: outSize)
        var count = [Int](repeating: 0, count: outSize)
        var weight = [Int32](repeating: 0, count: outSize * size)
        for o in 0..<outSize {
            let center = (Double(o) + 0.5) * scale
            let lo = max(Int(center - support + 0.5), 0)
            let hi = min(Int(center + support + 0.5), inSize)
            var taps = [Double](repeating: 0, count: max(hi - lo, 0))
            var sum = 0.0
            for t in 0..<taps.count {
                taps[t] = cubic((Double(t + lo) - center + 0.5) / spread)
                sum += taps[t]
            }
            // Normalizing to sum one preserves brightness where an edge
            // truncates the support.
            for t in 0..<taps.count {
                let v = (sum != 0 ? taps[t] / sum : 0) * unit
                weight[o * size + t] = Int32(v + (v < 0 ? -0.5 : 0.5))
            }
            start[o] = lo
            count[o] = taps.count
        }
        return Kernel(size: size, start: start, count: count, weight: weight)
    }

    private static func blend(_ s: UnsafePointer<UInt8>, _ step: Int,
                              _ w: UnsafePointer<Int32>,
                              _ n: Int) -> SIMD4<UInt8> {
        var acc = SIMD4<Int32>(repeating: 1 << (bits - 1))
        for t in 0..<n {
            let p = t * step
            let px = SIMD4<UInt8>(s[p], s[p + 1], s[p + 2], s[p + 3])
            acc &+= SIMD4<Int32>(truncatingIfNeeded: px)
                &* SIMD4<Int32>(repeating: w[t])
        }
        return SIMD4<UInt8>(clamping: acc &>> Int32(bits))
    }

    private static func convolve(_ src: [UInt8], _ k: Kernel, lines: Int,
                                 from: Step, to: Step) -> [UInt8] {
        let outs = k.start.count
        var out = [UInt8](repeating: 0, count: outs * lines * 4)
        src.withUnsafeBufferPointer { s in
            k.weight.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { d in
                    for line in 0..<lines {
                        for o in 0..<outs {
                            let px = blend(
                                s.baseAddress! + k.start[o] * from.tap
                                    + line * from.line, from.tap,
                                w.baseAddress! + o * k.size, k.count[o])
                            let q = o * to.tap + line * to.line
                            for c in 0..<4 { d[q + c] = px[c] }
                        }
                    }
                }
            }
        }
        return out
    }

    private static func raster(_ img: CGImage) -> [UInt8]? {
        var out: [UInt8]? = nil
        let w = img.width, h = img.height
        let ctx = space(img).flatMap { space in
            CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: space,
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }
        if let ctx, let base = ctx.data {
            ctx.interpolationQuality = .none
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            let p = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
            out = [UInt8](UnsafeBufferPointer(start: p, count: w * h * 4))
        }
        return out
    }

    // The image's OWN space, so the draw copies rather than converts: a fixed
    // space applies a conversion PIL never applies (11 of 255 on a gAMA PNG).

    private static func space(_ img: CGImage) -> CGColorSpace? {
        var out = CGColorSpace(name: CGColorSpace.sRGB)
        if let own = img.colorSpace, own.model == .rgb { out = own }
        return out
    }

    private static func packed(_ rgba: [UInt8], _ pixels: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: pixels * 3)
        for i in 0..<pixels {
            for c in 0..<3 { out[i * 3 + c] = rgba[i * 4 + c] }
        }
        return out
    }
}
