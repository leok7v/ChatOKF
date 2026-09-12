import Foundation
import Metal

struct WeightRef {
    let buf: MTLBuffer
    let local: UInt64

    static func + (base: WeightRef, delta: UInt64) -> WeightRef {
        WeightRef(buf: base.buf, local: base.local + delta)
    }
}

public final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    // Apple7 (A14 / M1) and up. Building a simdgroup-matrix kernel on an older
    // GPU takes the shader compiler service down instead of returning an error.
    let matrixUnits: Bool

    private struct Window {
        let buf: MTLBuffer
        let start: Int
        let end: Int
    }
    private let windows: [Window]

    static let minWindows = 64

    init(_ gguf: GGUF) throws {
        let dev = MTLCreateSystemDefaultDevice()
        let q = dev?.makeCommandQueue()
        let limit = dev?.maxBufferLength ?? 0
        let ranges = MetalContext.cuts(gguf, limit: limit)
        let widest = ranges.map { r in r.count }.max() ?? 0
        let mapped = MetalContext.mapped(gguf, ranges, dev, limit)
        if let fault = MetalContext.fault(dev, q, widest: widest,
                                          limit: limit, mapped: mapped.count,
                                          want: ranges.count) {
            throw fault
        }
        device = dev!
        queue = q!
        library = try device.makeDefaultLibrary(bundle: MetalContext.bundle)
        windows = mapped
        matrixUnits = device.supportsFamily(.apple7)
        for t in gguf.tensors.values {
            let r = window(UInt64(t.base - gguf.map))
            precondition(Int(r.local) + t.byteCount <= r.buf.length,
                         "\(t.name) straddles a weight window")
        }
    }

    private static var bundle: Bundle { Bundle(for: MetalContext.self) }

    // No window splits a tensor: every kernel reads a whole tensor through one
    // binding. Windows round outward to pages and may overlap by a page.
    private static func cuts(_ g: GGUF, limit: Int) -> [Range<Int>] {
        let page   = Int(getpagesize())
        let total  = (g.mapSize + page - 1) / page * page
        let target = min(limit, (total + minWindows - 1) / minWindows)
        let spans = g.tensors.values
            .map { t in (start: t.base - g.map,
                         end: t.base - g.map + t.byteCount) }
            .sorted { a, b in a.start < b.start }
        var out: [Range<Int>] = []
        var open = 0
        var last = 0
        for s in spans {
            let next = s.start / page * page
            if s.end - open > target && next > open && last > open {
                out.append(open ..< min((last + page - 1) / page * page,
                                        total))
                open = next
            }
            last = max(last, s.end)
        }
        out.append(open ..< total)
        return out
    }

    // A range wider than `limit` is skipped, not asked for: Metal aborts on an
    // oversized bytesNoCopy instead of returning nil.
    private static func mapped(_ g: GGUF, _ ranges: [Range<Int>],
                               _ dev: MTLDevice?, _ limit: Int) -> [Window] {
        let base = UnsafeMutableRawPointer(mutating: g.map)
        return ranges.compactMap { r in
            let buf = r.count <= limit
                ? dev?.makeBuffer(bytesNoCopy: base + r.lowerBound,
                                  length: r.count,
                                  options: .storageModeShared,
                                  deallocator: nil)
                : nil
            return buf.map { b in
                Window(buf: b, start: r.lowerBound, end: r.upperBound)
            }
        }
    }

    private static func fault(_ dev: MTLDevice?, _ queue: MTLCommandQueue?,
                              widest: Int, limit: Int, mapped: Int,
                              want: Int) -> MetalErr? {
        var out: MetalErr? = nil
        if dev == nil {
            out = .noDevice
        } else if queue == nil {
            out = .noQueue
        } else if widest > limit {
            out = .tooBig(widest, limit)
        } else if mapped < want {
            out = .noBuffer
        }
        return out
    }

    var windowMB: [Int] { windows.map { w in (w.end - w.start) / 1_048_576 } }

    // The LAST window starting at or before `off`: where two overlap, only the
    // later one is guaranteed to hold the whole tensor.
    func window(_ off: UInt64) -> WeightRef {
        var i = 0
        while i + 1 < windows.count && UInt64(windows[i + 1].start) <= off {
            i += 1
        }
        return WeightRef(buf: windows[i].buf,
                         local: off - UInt64(windows[i].start))
    }

    static func needsMatrixUnits(_ name: String) -> Bool {
        name.hasSuffix("_mm") || name.hasSuffix("_mm_h")
    }

    func prewarm() throws {
        let t0 = Date()
        let all = library.functionNames.count
        var built = 0
        for name in library.functionNames
        where matrixUnits || !MetalContext.needsMatrixUnits(name) {
            _ = try pipeline(name)
            built += 1
        }
        Diag.shared.report(.perf, String(
            format: "[metal] prewarm %d of %d pipelines in %.2fs",
            built, all, Date().timeIntervalSince(t0)))
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        var state = pipelines[name]
        if state == nil, let fn = library.makeFunction(name: name) {
            let made = try device.makeComputePipelineState(function: fn)
            pipelines[name] = made
            state = made
        }
        if state == nil { throw MetalErr.noFunction(name) }
        return state!
    }

    func makeF32(_ count: Int) -> MTLBuffer {
        device.makeBuffer(length: max(count, 1) * MemoryLayout<Float>.stride,
                          options: .storageModeShared)!
    }

    func makeF32(_ values: [Float]) -> MTLBuffer {
        values.withUnsafeBytes { src in
            device.makeBuffer(bytes: src.baseAddress!,
                              length: max(src.count, 1),
                              options: .storageModeShared)!
        }
    }

    // Zeroed here: Metal hands back whatever was in the page.
    func makeU32(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(
            length: max(count, 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }

    private var blank: MTLBuffer?

    func noBlocks(_ rows: Int) -> MTLBuffer {
        let need = rows * 2 * MemoryLayout<UInt32>.stride
        if blank == nil || blank!.length < need { blank = makeU32(rows * 2) }
        return blank!
    }

    private var spare: MTLBuffer?

    func scratch(_ bytes: Int) -> MTLBuffer {
        if spare == nil || spare!.length < bytes {
            spare = device.makeBuffer(length: max(bytes, 16),
                                      options: .storageModeShared)!
        }
        return spare!
    }
}

enum MetalErr: Error, CustomStringConvertible {
    case noDevice, noQueue, noBuffer
    case tooBig(Int, Int)
    case noFunction(String)

    var description: String {
        let out: String
        switch self {
        case .noDevice: out = "no Metal device"
        case .noQueue: out = "no Metal command queue"
        case .noBuffer: out = "Metal buffer allocation failed"
        case let .tooBig(want, cap):
            out = "one weight tensor needs \(want / 1_048_576) MB but this "
                + "GPU maps at most \(cap / 1_048_576) MB in one buffer"
        case let .noFunction(n): out = "no Metal kernel named \(n)"
        }
        return out
    }
}

extension MTLBuffer {
    func f32(_ count: Int) -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(
            start: contents().assumingMemoryBound(to: Float.self), count: count)
    }

    func u32(_ count: Int) -> UnsafeMutableBufferPointer<UInt32> {
        UnsafeMutableBufferPointer(
            start: contents().assumingMemoryBound(to: UInt32.self),
            count: count)
    }
}
