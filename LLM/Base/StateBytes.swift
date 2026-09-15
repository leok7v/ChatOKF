import Foundation

enum StateBytes {

    // A state file names itself: a stale file from a build with the empty-Data
    // default would deserialize to pos 0 with no KV and SUCCEED.
    static let magic: [UInt8] = Array("GDNS".utf8)
    static let version = 5

    static func putHeader(_ out: inout Data) {
        out.append(contentsOf: magic)
        putInt(&out, version)
    }

    static func putInt(_ out: inout Data, _ v: Int) {
        var x = Int64(v).littleEndian
        withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }

    static func putRaw(_ out: inout Data, _ p: UnsafeRawPointer, _ n: Int) {
        putInt(&out, n)
        out.append(p.assumingMemoryBound(to: UInt8.self), count: n)
    }

    static func putKeyed<V>(_ out: inout Data, _ dict: [Int: V],
                            _ payload: (inout Data, Int, V) -> Void) {
        putInt(&out, dict.count)
        for key in dict.keys.sorted() {
            putInt(&out, key)
            payload(&out, key, dict[key]!)
        }
    }

    static func read<T>(_ data: Data, named: Bool = true,
                        _ body: (inout Reader) -> T) -> T? {
        data.withUnsafeBytes { raw in
            var r = Reader(raw)
            var out: T? = nil
            if !named || r.header() { out = body(&r) }
            return out
        }
    }

    static func keyed<V>(_ r: inout Reader,
                         _ payload: (inout Reader) -> V) -> [Int: V] {
        var out: [Int: V] = [:]
        for _ in 0..<r.int() {
            let key = r.int()
            out[key] = payload(&r)
        }
        return out
    }

    struct Reader {
        let raw: UnsafeRawBufferPointer
        var at = 0

        init(_ raw: UnsafeRawBufferPointer) { self.raw = raw }

        mutating func header() -> Bool {
            var named = raw.count >= magic.count + 8
            if named {
                for i in 0..<magic.count where raw[i] != magic[i] {
                    named = false
                }
                at = magic.count
            }
            return named && int() == version
        }

        mutating func int() -> Int {
            var out = 0
            if at + 8 <= raw.count {
                out = Int(Int64(littleEndian: raw.loadUnaligned(
                    fromByteOffset: at, as: Int64.self)))
            }
            at += 8
            return out
        }

        mutating func bytes() -> UnsafeRawBufferPointer {
            let n = max(int(), 0)
            let whole = at + n <= raw.count
            let out = UnsafeRawBufferPointer(
                rebasing: raw[at ..< (whole ? at + n : at)])
            at += n
            return out
        }

        mutating func bytes(into b: UnsafeMutableRawPointer, count: Int) {
            let src = bytes()
            if src.count == count {
                _ = memcpy(b, src.baseAddress!, count)
            }
        }
    }
}
