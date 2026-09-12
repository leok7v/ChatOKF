import Foundation

enum SafetensorsErr: Error, CustomStringConvertible {
    case io(String)
    case parse(String)

    var description: String {
        let out: String
        switch self {
        case let .io(m): out = "safetensors: \(m)"
        case let .parse(m): out = "safetensors: \(m)"
        }
        return out
    }
}

struct SafetensorsEntry {
    let dtype: String
    let shape: [Int]
    let shard: Int
    let offset: Int
    let bytes: Int

    var count: Int { shape.reduce(1, *) }
}

final class Safetensors {

    private final class Shard {
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
                failure = "fstat \(path)"
            } else {
                pages = mmap(nil, Int(st.st_size), PROT_READ, MAP_PRIVATE,
                             opened, 0)
                if pages == nil || pages == MAP_FAILED {
                    pages = nil
                    failure = "mmap \(path)"
                }
            }
            if let pages {
                fd = opened
                size = Int(st.st_size)
                base = UnsafeRawPointer(pages)
            } else {
                if opened >= 0 { close(opened) }
                throw SafetensorsErr.io(failure)
            }
        }

        deinit {
            munmap(UnsafeMutableRawPointer(mutating: base), size)
            close(fd)
        }
    }

    static let widths = ["F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1]

    private let shards: [Shard]
    private let dataStart: [Int]
    let entries: [String: SafetensorsEntry]
    let shardNames: [String]

    var names: [String] { entries.keys.sorted() }

    init(dir: String) throws {
        let listing = ((try? FileManager.default
            .contentsOfDirectory(atPath: dir)) ?? [])
            .filter { n in n.hasSuffix(".safetensors") }
            .sorted()
        if listing.isEmpty {
            throw SafetensorsErr.io("no .safetensors under \(dir)")
        }
        var opened: [Shard] = []
        var starts: [Int] = []
        var found: [String: SafetensorsEntry] = [:]
        for (i, file) in listing.enumerated() {
            let shard = try Shard(path: dir + "/" + file)
            let n = Int(shard.base.loadUnaligned(as: UInt64.self))
            if n <= 0 || n + 8 > shard.size {
                throw SafetensorsErr.parse("\(file): header length \(n)")
            }
            let raw = Data(bytes: shard.base + 8, count: n)
            let root = (try JSONSerialization.jsonObject(with: raw))
                as? [String: Any]
            if root == nil {
                throw SafetensorsErr.parse("\(file): header is not an object")
            }
            for (key, any) in root! where key != "__metadata__" {
                found[key] = try Safetensors.entry(key, any, i)
            }
            opened.append(shard)
            starts.append(8 + n)
        }
        shards = opened
        dataStart = starts
        entries = found
        shardNames = listing
    }

    private static func entry(_ key: String, _ any: Any,
                              _ shard: Int) throws -> SafetensorsEntry {
        let item = any as? [String: Any] ?? [:]
        let span = item["data_offsets"] as? [Int] ?? []
        let shape = item["shape"] as? [Int] ?? []
        let dtype = item["dtype"] as? String ?? ""
        if span.count != 2 || dtype.isEmpty {
            throw SafetensorsErr.parse("\(key): malformed entry")
        }
        if widths[dtype] == nil {
            throw SafetensorsErr.parse("\(key): unsupported dtype \(dtype)")
        }
        let want = shape.reduce(1, *) * widths[dtype]!
        if span[1] - span[0] != want {
            throw SafetensorsErr.parse(
                "\(key): \(span[1] - span[0]) bytes, shape wants \(want)")
        }
        return SafetensorsEntry(dtype: dtype, shape: shape, shard: shard,
                                offset: span[0], bytes: span[1] - span[0])
    }

    func shape(_ name: String) throws -> [Int] {
        let e = entries[name]
        if e == nil { throw SafetensorsErr.parse("no tensor \(name)") }
        return e!.shape
    }

    func values(_ name: String) throws -> [Float] {
        let e = entries[name]
        if e == nil { throw SafetensorsErr.parse("no tensor \(name)") }
        var out = decode(e!)
        let key = name.hasSuffix(".weight")
            ? String(name.dropLast(7)) + ".weight_scale_inv" : ""
        if let scale = entries[key], e!.shape.count == 2 {
            unblock(&out, e!.shape, decode(scale), scale.shape)
        }
        return out
    }

    private func decode(_ e: SafetensorsEntry) -> [Float] {
        let p = shards[e.shard].base + dataStart[e.shard] + e.offset
        let n = e.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { ob in
            let dst = ob.baseAddress!
            switch e.dtype {
            case "F32":
                memcpy(dst, p, n * 4)
            case "F16":
                for i in 0..<n {
                    let h = (p + i * 2).loadUnaligned(as: UInt16.self)
                    dst[i] = Float(Float16(bitPattern: h))
                }
            case "BF16":
                BF16.decode(p, n, dst)
            default:
                FP8.decode(p.assumingMemoryBound(to: UInt8.self), n, dst)
            }
        }
        return out
    }

    private func unblock(_ values: inout [Float], _ shape: [Int],
                         _ scale: [Float], _ grid: [Int]) {
        let rows = shape[0]
        let cols = shape[1]
        let stride = grid.count == 2 ? grid[1] : 1
        let block = 128
        values.withUnsafeMutableBufferPointer { vb in
            let v = vb.baseAddress!
            for r in 0..<rows {
                let band = (r / block) * stride
                for c in 0..<cols {
                    v[r * cols + c] *= scale[band + c / block]
                }
            }
        }
    }
}
