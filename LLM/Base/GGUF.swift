import Foundation

enum GGUFType: Int32 {
    case f32 = 0, f16 = 1
    case q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8, q8_1 = 9
    case q2k = 10, q3k = 11, q4k = 12, q5k = 13, q6k = 14, q8k = 15
    case iq2_xxs = 16, iq2_xs = 17, iq3_xxs = 18
    case iq1_s = 19
    case iq4_nl = 20, iq3_s = 21, iq2_s = 22, iq4_xs = 23
    case iq1_m = 29
    case bf16 = 30
    case q1_0 = 41         // PrismML 1-bit, 128/block, 18 bytes
    case q2_0 = 42         // PrismML ternary, 128/block, 34 bytes
}

enum GGUFValue {
    case int(Int64)
    case double(Double)
    case string(String)
    case ints([Int64])
    case doubles([Double])
    case stringTable(offset: Int, count: Int)
    case bool(Bool)
}

struct GGUFStringTable {
    let base: UnsafeRawPointer
    let count: Int

    func forEach(_ body: (Int, UnsafeRawBufferPointer) -> Void) {
        var at = 0
        for _ in 0..<count {
            let n = Int((base + at).loadUnaligned(as: UInt64.self))
            body(at, UnsafeRawBufferPointer(start: base + at + 8, count: n))
            at += 8 + n
        }
    }

    func decoded() -> [String] {
        var out: [String] = []
        out.reserveCapacity(count)
        forEach { _, bytes in
            out.append(String(decoding: bytes, as: UTF8.self))
        }
        return out
    }
}

struct GGUFTensor {
    let name: String
    let dims: [Int]           // ggml order: ne[0] is the fastest-varying axis
    let type: GGUFType
    let base: UnsafeRawPointer
    let byteCount: Int

    var count: Int { dims.reduce(1, *) }
}

// Owned as an object of its own so a parse that throws still frees them: a
// throwing class init runs no deinit, but releases initialized properties.
private final class GGUFMapping {

    let base: UnsafeRawPointer
    let size: Int
    private let fd: Int32

    init(path: String) throws {
        let opened = open(path, O_RDONLY)
        var st = stat()
        var pages: UnsafeMutableRawPointer? = nil
        var failure = ""
        if opened < 0 {
            failure = "open \(path)"
        } else if fstat(opened, &st) != 0 {
            failure = "fstat"
        } else {
            pages = mmap(nil, Int(st.st_size), PROT_READ, MAP_PRIVATE,
                         opened, 0)
            if pages == nil || pages == MAP_FAILED {
                pages = nil
                failure = "mmap"
            }
        }
        if let pages {
            fd = opened
            size = Int(st.st_size)
            base = UnsafeRawPointer(pages)
        } else {
            if opened >= 0 { close(opened) }
            throw GGUFErr.io(failure)
        }
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), size)
        close(fd)
    }
}

final class GGUF {
    private let mapping: GGUFMapping
    let meta: [String: GGUFValue]
    let tensors: [String: GGUFTensor]

    var map: UnsafeRawPointer { mapping.base }
    var mapSize: Int { mapping.size }

    init(path: String) throws {
        let mapped = try GGUFMapping(path: path)
        mapping = mapped
        let map = mapped.base

        var c = Cursor(base: map, limit: mapped.size)
        if c.u32() != 0x4655_4747 { throw GGUFErr.parse("bad magic") }
        _ = c.u32()
        let nTensors = Int(c.u64())
        let nKV = Int(c.u64())

        var meta: [String: GGUFValue] = [:]
        var alignment = 32
        for _ in 0..<nKV {
            let key = c.str()
            let v = c.value()
            if key == "general.alignment", case let .int(a) = v { alignment = Int(a) }
            meta[key] = v
        }
        self.meta = meta

        struct Raw { let name: String; let dims: [Int]; let type: Int32; let off: Int }
        var raws: [Raw] = []
        raws.reserveCapacity(nTensors)
        for _ in 0..<nTensors {
            let name = c.str()
            let nd = Int(c.u32())
            var dims: [Int] = []
            for _ in 0..<nd { dims.append(Int(c.u64())) }
            let type = Int32(bitPattern: c.u32())
            let off = Int(c.u64())
            raws.append(Raw(name: name, dims: dims, type: type, off: off))
        }
        let dataStart = (c.pos + alignment - 1) / alignment * alignment

        var tensors: [String: GGUFTensor] = [:]
        tensors.reserveCapacity(nTensors)
        for r in raws {
            let found = GGUFType(rawValue: r.type)
            if found == nil {
                throw GGUFErr.parse("unknown ggml type \(r.type) for \(r.name)")
            }
            let ty = found!
            let n = r.dims.reduce(1, *)
            let bytes = GGUF.rowByteCount(ty, n)
            tensors[r.name] = GGUFTensor(
                name: r.name, dims: r.dims, type: ty,
                base: map + dataStart + r.off, byteCount: bytes)
        }
        self.tensors = tensors
    }

    static func rowByteCount(_ ty: GGUFType, _ n: Int) -> Int {
        switch ty {
        case .f32:  return n * 4
        case .f16, .bf16: return n * 2
        case .q2_0: return n / 128 * 34
        case .q2k: return n / 256 * 84
        case .q3k: return n / 256 * 110
        case .q4k: return n / 256 * 144
        case .q5k: return n / 256 * 176
        case .q6k: return n / 256 * 210
        case .q8k: return n / 256 * 292
        case .iq2_xxs: return n / 256 * 66
        case .iq2_xs: return n / 256 * 74
        case .iq2_s: return n / 256 * 82
        case .iq3_xxs: return n / 256 * 98
        case .iq3_s: return n / 256 * 110
        case .iq4_xs: return n / 256 * 136
        case .iq4_nl: return n / 32 * 18
        case .iq1_s: return n / 256 * 50
        case .iq1_m: return n / 256 * 56
        case .q1_0: return n / 128 * 18
        case .q4_0: return n / 32 * 18
        case .q8_0: return n / 32 * 34
        default:    return n
        }
    }

    func int(_ k: String) -> Int? {
        switch meta[k] {
        case let .int(v): Int(v)
        default: nil
        }
    }

    func double(_ k: String) -> Double? {
        switch meta[k] {
        case let .double(v): v
        case let .int(v): Double(v)
        default: nil
        }
    }

    // A writer may store a flag as GGUF bool OR int; a bool key read through
    // `int` comes back nil and looks absent.
    func bool(_ k: String) -> Bool? {
        switch meta[k] {
        case let .bool(v): v
        case let .int(v): v != 0
        default: nil
        }
    }

    func ints(_ k: String) -> [Int]? {
        switch meta[k] {
        case let .ints(v): v.map(Int.init)
        default: nil
        }
    }

    func doubles(_ k: String) -> [Double]? {
        switch meta[k] {
        case let .doubles(v): v
        default: nil
        }
    }

    func string(_ k: String) -> String? {
        switch meta[k] {
        case let .string(v): v
        default: nil
        }
    }

    func strings(_ k: String) -> [String]? {
        stringTable(k)?.decoded()
    }

    func stringTable(_ k: String) -> GGUFStringTable? {
        switch meta[k] {
        case let .stringTable(offset, count):
            GGUFStringTable(base: map + offset, count: count)
        default: nil
        }
    }

    // MADV_RANDOM over the table's pages: the default read-ahead clusters are
    // right for a weight walked end to end, wrong for a row-at-a-time gather.
    func gathered(_ t: GGUFTensor) {
        let page = Int(getpagesize())
        let from = (t.base - map) / page * page
        let upto = (t.base - map + t.byteCount + page - 1) / page * page
        _ = madvise(UnsafeMutableRawPointer(mutating: map + from),
                    upto - from, MADV_RANDOM)
    }

    func readByEveryToken(drafting: Bool) -> [GGUFTensor] {
        let untied = tensors["output.weight"] != nil
        return tensors.values.filter { t in
            let head = t.name.hasPrefix("assist.")
                || t.name.contains(".nextn.")
            let gathered = t.name == "per_layer_token_embd.weight"
                || (untied && t.name == "token_embd.weight")
            return ModelShape.tower(of: t.name) == "text" && !gathered
                && (drafting || !head)
        }
    }

    private var pageSink: UInt8 = 0

    func fault(_ picked: [GGUFTensor]) -> Int {
        let page = Int(getpagesize())
        let spans = picked.map { t in
            (from: (t.base - map) / page * page,
             upto: min((t.base - map + t.byteCount + page - 1) / page * page,
                       mapSize))
        }.sorted { a, b in a.from < b.from }
        var covered = 0
        var done = 0
        for span in spans where span.upto > max(span.from, done) {
            let from = max(span.from, done)
            _ = madvise(UnsafeMutableRawPointer(mutating: map + from),
                        span.upto - from, MADV_WILLNEED)
            for at in stride(from: from, to: span.upto, by: page) {
                pageSink &+= (map + at).load(as: UInt8.self)
            }
            covered += span.upto - from
            done = span.upto
        }
        return covered
    }

    func tensor(_ name: String) -> GGUFTensor {
        let t = tensors[name]
        precondition(t != nil, "missing tensor \(name)")
        return t!
    }

    func maybe(_ name: String) -> GGUFTensor? { tensors[name] }
}

public struct WeightWarm: @unchecked Sendable {
    public let bytes: Int
    private let gguf: GGUF
    private let picked: [GGUFTensor]

    init(_ gguf: GGUF, drafting: Bool) {
        self.gguf = gguf
        picked = gguf.readByEveryToken(drafting: drafting)
        bytes = picked.reduce(0) { sum, t in sum + t.byteCount }
    }

    public func run() -> Int { gguf.fault(picked) }
}

enum GGUFErr: Error { case io(String), parse(String) }

private struct Cursor {
    let base: UnsafeRawPointer
    let limit: Int
    var pos: Int = 0

    mutating func bytes(_ n: Int) -> UnsafeRawPointer {
        let p = base + pos
        pos += n
        precondition(pos <= limit, "gguf overrun")
        return p
    }

    mutating func u32() -> UInt32 { bytes(4).loadUnaligned(as: UInt32.self) }

    mutating func u64() -> UInt64 { bytes(8).loadUnaligned(as: UInt64.self) }

    mutating func str() -> String {
        let n = Int(u64())
        let p = bytes(n)
        return String(decoding: UnsafeRawBufferPointer(start: p, count: n),
                      as: UTF8.self)
    }

    mutating func value() -> GGUFValue {
        let t = u32()
        return scalar(t)
    }

    mutating func scalar(_ t: UInt32) -> GGUFValue {
        switch t {
        case 0:  return .int(Int64(bytes(1).loadUnaligned(as: UInt8.self)))
        case 1:  return .int(Int64(bytes(1).loadUnaligned(as: Int8.self)))
        case 2:  return .int(Int64(bytes(2).loadUnaligned(as: UInt16.self)))
        case 3:  return .int(Int64(bytes(2).loadUnaligned(as: Int16.self)))
        case 4:  return .int(Int64(bytes(4).loadUnaligned(as: UInt32.self)))
        case 5:  return .int(Int64(bytes(4).loadUnaligned(as: Int32.self)))
        case 6:  return .double(Double(bytes(4).loadUnaligned(as: Float32.self)))
        case 7:  return .bool(bytes(1).loadUnaligned(as: UInt8.self) != 0)
        case 8:  return .string(str())
        case 9:  return array()
        case 10: return .int(Int64(bitPattern: u64()))
        case 11: return .int(bytes(8).loadUnaligned(as: Int64.self))
        case 12: return .double(bytes(8).loadUnaligned(as: Double.self))
        default: fatalError("bad kv type \(t)")
        }
    }

    mutating func array() -> GGUFValue {
        let et = u32()
        let n = Int(u64())
        if et == 8 {
            let start = pos
            for _ in 0..<n { _ = bytes(Int(u64())) }
            return .stringTable(offset: start, count: n)
        }
        var out: [Int64] = []; out.reserveCapacity(n)
        var dbl = false; var dvals: [Double] = []
        for _ in 0..<n {
            let v = scalar(et)
            switch v {
            case let .int(i): out.append(i)
            case let .bool(b): out.append(b ? 1 : 0)
            case let .double(d): dbl = true; dvals.append(d)
            default: break
            }
        }
        return dbl ? .doubles(dvals) : .ints(out)
    }
}
