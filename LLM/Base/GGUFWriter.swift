import Foundation

enum GGUFValueOut {
    case u32(UInt32)
    case i32(Int32)
    case f32(Float)
    case bool(Bool)
    case u64(UInt64)
    case string(String)
    case strings([String])
    case i32s([Int32])
    case f32s([Float])
}

enum GGUFWriteErr: Error, CustomStringConvertible {
    case io(String)
    case order(String)

    var description: String {
        let out: String
        switch self {
        case let .io(m): out = "gguf write: \(m)"
        case let .order(m): out = "gguf write: \(m)"
        }
        return out
    }
}

final class GGUFWriter {
    private struct Declared {
        let name: String
        let dims: [Int]
        let type: GGUFType
        let offset: Int
        let bytes: Int
    }

    private let handle: FileHandle
    private let scratch: String
    private let target: String
    private let alignment: Int
    private var kv: [(String, Data)] = []
    private var declared: [Declared] = []
    private var next = 0
    private var written = 0
    private var headerDone = false

    init(path: String, alignment: Int = 16384) throws {
        let temp = path + ".partial"
        try? FileManager.default.removeItem(atPath: temp)
        FileManager.default.createFile(atPath: temp, contents: nil)
        let opened = FileHandle(forWritingAtPath: temp)
        if opened == nil { throw GGUFWriteErr.io("cannot open \(temp)") }
        handle = opened!
        scratch = temp
        target = path
        self.alignment = alignment
    }

    static func encoded(_ value: GGUFValueOut) -> Data { encode(value) }

    func meta(_ key: String, _ value: GGUFValueOut) {
        kv.append((key, GGUFWriter.encode(value)))
    }

    func metaRaw(_ key: String, _ encoded: Data) {
        kv.append((key, encoded))
    }

    func declare(_ name: String, dims: [Int], type: GGUFType) {
        let size = GGUF.rowByteCount(type, dims.reduce(1, *))
        declared.append(Declared(name: name, dims: dims, type: type,
                                 offset: next, bytes: size))
        next += size + GGUFWriter.padding(size, alignment)
    }

    private static func padding(_ size: Int, _ alignment: Int) -> Int {
        (alignment - size % alignment) % alignment
    }

    var declaredCount: Int { declared.count }
    var declaredBytes: Int { next }

    func finishHeader() {
        var out = Data()
        out.append(GGUFWriter.u32(0x4655_4747))
        out.append(GGUFWriter.u32(3))
        out.append(GGUFWriter.u64(UInt64(declared.count)))
        out.append(GGUFWriter.u64(UInt64(kv.count)))
        for (key, blob) in kv {
            out.append(GGUFWriter.str(key))
            out.append(blob)
        }
        for t in declared {
            out.append(GGUFWriter.str(t.name))
            out.append(GGUFWriter.u32(UInt32(t.dims.count)))
            for d in t.dims { out.append(GGUFWriter.u64(UInt64(d))) }
            out.append(GGUFWriter.u32(UInt32(bitPattern: t.type.rawValue)))
            out.append(GGUFWriter.u64(UInt64(t.offset)))
        }
        let pad = GGUFWriter.padding(out.count, alignment)
        if pad > 0 { out.append(Data(count: pad)) }
        handle.write(out)
        headerDone = true
    }

    func append(_ bytes: UnsafeRawBufferPointer, expecting name: String) throws {
        var fault: GGUFWriteErr? = nil
        if !headerDone {
            fault = .order("finishHeader before append")
        } else if written >= declared.count {
            fault = .order("append past the declared tensor count")
        } else if declared[written].name != name {
            fault = .order("expected \(declared[written].name), got \(name)")
        } else if bytes.count != declared[written].bytes {
            fault = .order("\(name): \(bytes.count) bytes, declared "
                           + "\(declared[written].bytes)")
        }
        if let fault { throw fault }
        handle.write(Data(bytes))
        let pad = GGUFWriter.padding(bytes.count, alignment)
        if pad > 0 { handle.write(Data(count: pad)) }
        written += 1
    }

    func close() throws {
        let missing = declared.count - written
        try handle.close()
        if missing != 0 {
            try? FileManager.default.removeItem(atPath: scratch)
            throw GGUFWriteErr.order("\(missing) tensor(s) never appended")
        }
        if rename(scratch, target) != 0 {
            throw GGUFWriteErr.io("rename \(scratch) -> \(target)")
        }
    }

    private static func u32(_ v: UInt32) -> Data {
        withUnsafeBytes(of: v.littleEndian) { raw in Data(raw) }
    }

    private static func u64(_ v: UInt64) -> Data {
        withUnsafeBytes(of: v.littleEndian) { raw in Data(raw) }
    }

    private static func str(_ s: String) -> Data {
        let raw = Array(s.utf8)
        var out = u64(UInt64(raw.count))
        out.append(contentsOf: raw)
        return out
    }

    private static func tag(_ value: GGUFValueOut) -> UInt32 {
        let out: UInt32
        switch value {
        case .u32: out = 4
        case .i32: out = 5
        case .f32: out = 6
        case .bool: out = 7
        case .u64: out = 10
        case .string: out = 8
        case .strings, .i32s, .f32s: out = 9
        }
        return out
    }

    private static func encode(_ value: GGUFValueOut) -> Data {
        var out = u32(tag(value))
        switch value {
        case let .u32(v): out.append(u32(v))
        case let .i32(v): out.append(u32(UInt32(bitPattern: v)))
        case let .f32(v): out.append(u32(v.bitPattern))
        case let .bool(v): out.append(Data([v ? 1 : 0]))
        case let .u64(v): out.append(u64(v))
        case let .string(v): out.append(str(v))
        case let .strings(items):
            out.append(u32(8))
            out.append(u64(UInt64(items.count)))
            for s in items { out.append(str(s)) }
        case let .i32s(items):
            out.append(u32(5))
            out.append(u64(UInt64(items.count)))
            for v in items { out.append(u32(UInt32(bitPattern: v))) }
        case let .f32s(items):
            out.append(u32(6))
            out.append(u64(UInt64(items.count)))
            for v in items { out.append(u32(v.bitPattern)) }
        }
        return out
    }
}
