import Compression
import Foundation

public enum XETError: Error, CustomStringConvertible, Sendable {
    case header(String)
    case truncated(String)
    case decode(String)
    case http(Int, String)
    case malformed(String)
    case file(String)

    public var description: String {
        switch self {
        case .header(let what): return "xet chunk header: \(what)"
        case .truncated(let what): return "xet stream ended inside \(what)"
        case .decode(let what): return "xet chunk: \(what)"
        case .http(let code, let what): return "xet http \(code): \(what)"
        case .malformed(let what): return "xet reconstruction: \(what)"
        case .file(let what): return "xet file: \(what)"
        }
    }
}

public struct ChunkHeader: Equatable, Sendable {
    public static let bytes = 8
    public static let largest = 128 << 10
    public let version: UInt8
    public let compressed: Int
    public let scheme: UInt8
    public let uncompressed: Int

    public init(version: UInt8 = 0, compressed: Int, scheme: UInt8,
                uncompressed: Int) {
        self.version = version
        self.compressed = compressed
        self.scheme = scheme
        self.uncompressed = uncompressed
    }

    public static func parse(_ b: UnsafeRawBufferPointer) throws
        -> ChunkHeader {
        let h = ChunkHeader(
            version: b[0],
            compressed: Int(b[1]) | Int(b[2]) << 8 | Int(b[3]) << 16,
            scheme: b[4],
            uncompressed: Int(b[5]) | Int(b[6]) << 8 | Int(b[7]) << 16)
        var fault: String? = nil
        if h.version != 0 {
            fault = "version \(h.version)"
        } else if h.scheme > 2 {
            fault = "scheme \(h.scheme)"
        } else if h.uncompressed > ChunkHeader.largest {
            fault = "\(h.uncompressed) bytes uncompressed"
        } else if h.scheme == 0 && h.compressed != h.uncompressed {
            fault = "\(h.compressed) stored bytes for \(h.uncompressed)"
        }
        if let fault { throw XETError.header(fault) }
        return h
    }

    public var encoded: [UInt8] {
        [version,
         UInt8(compressed & 0xFF), UInt8(compressed >> 8 & 0xFF),
         UInt8(compressed >> 16 & 0xFF),
         scheme,
         UInt8(uncompressed & 0xFF), UInt8(uncompressed >> 8 & 0xFF),
         UInt8(uncompressed >> 16 & 0xFF)]
    }
}

public enum Chunk {

    public static func decode(_ h: ChunkHeader,
                              _ payload: UnsafeRawBufferPointer,
                              into out: UnsafeMutableRawBufferPointer,
                              scratch: inout [UInt8]) throws {
        if h.uncompressed > 0 {
            if scratch.count < h.uncompressed + 1 {
                scratch = [UInt8](repeating: 0, count: h.uncompressed + 1)
            }
            switch h.scheme {
            case 0:
                out.copyMemory(from: payload)
            case 1:
                try lz4(payload, h.uncompressed, &scratch)
                scratch.withUnsafeBytes { s in
                    out.copyMemory(from: UnsafeRawBufferPointer(
                        rebasing: s[0..<h.uncompressed]))
                }
            default:
                try lz4(payload, h.uncompressed, &scratch)
                regroup(scratch, h.uncompressed, into: out)
            }
        }
    }

    static func lz4(_ src: UnsafeRawBufferPointer, _ n: Int,
                    _ dst: inout [UInt8]) throws {
        let framed = src.count >= 7 && src[0] == 0x04 && src[1] == 0x22
            && src[2] == 0x4D && src[3] == 0x18
        let got = framed ? try frame(src, n, &dst)
                         : raw(src, into: &dst, at: 0, want: n)
        if got != n {
            throw XETError.decode("lz4 gave \(got) of \(n) bytes")
        }
    }

    static func raw(_ src: UnsafeRawBufferPointer, into dst: inout [UInt8],
                    at: Int, want: Int) -> Int {
        dst.withUnsafeMutableBytes { d in
            compression_decode_buffer(
                d.baseAddress!.assumingMemoryBound(to: UInt8.self) + at,
                want + 1,
                src.baseAddress!.assumingMemoryBound(to: UInt8.self),
                src.count, nil, COMPRESSION_LZ4_RAW)
        }
    }

    static func frame(_ src: UnsafeRawBufferPointer, _ n: Int,
                      _ dst: inout [UInt8]) throws -> Int {
        let flg = src[4]
        let independent = flg & 0x20 != 0
        let checksummed = flg & 0x10 != 0
        var i = 7 + (flg & 0x08 != 0 ? 8 : 0) + (flg & 0x01 != 0 ? 4 : 0)
        var done = 0
        var fault: String? = flg >> 6 == 1 ? nil : "frame version \(flg >> 6)"
        var more = fault == nil
        while more {
            if i + 4 > src.count {
                fault = "frame cut at a block size"
            } else {
                let size = Int(src[i]) | Int(src[i + 1]) << 8
                    | Int(src[i + 2]) << 16 | Int(src[i + 3] & 0x7F) << 24
                let stored = src[i + 3] & 0x80 != 0
                i += 4
                if size == 0 {
                    more = false
                } else if i + size > src.count || (stored && done + size > n) {
                    fault = "frame cut inside a block"
                } else {
                    let block = UnsafeRawBufferPointer(
                        rebasing: src[i..<i + size])
                    let got: Int
                    if stored {
                        dst.withUnsafeMutableBytes { d in
                            UnsafeMutableRawBufferPointer(
                                rebasing: d[done..<done + size])
                                .copyMemory(from: block)
                        }
                        got = size
                    } else if independent {
                        got = raw(block, into: &dst, at: done, want: n - done)
                    } else {
                        got = sequences(block, into: &dst, at: done,
                                        want: n - done)
                    }
                    if got <= 0 { fault = "block \(i) did not decode" }
                    done += max(got, 0)
                    i += size + (checksummed ? 4 : 0)
                }
            }
            more = more && fault == nil
        }
        if let fault { throw XETError.decode("lz4 " + fault) }
        return done
    }

    static func sequences(_ src: UnsafeRawBufferPointer,
                          into dst: inout [UInt8], at: Int, want: Int)
        -> Int {
        var i = 0
        var o = at
        var bad = false
        while i < src.count && !bad {
            let token = Int(src[i])
            i += 1
            var lit = token >> 4
            var ext = lit == 15
            while ext, i < src.count {
                lit += Int(src[i])
                ext = src[i] == 255
                i += 1
            }
            bad = i + lit > src.count || o + lit > at + want
            if !bad {
                dst.withUnsafeMutableBytes { d in
                    UnsafeMutableRawBufferPointer(rebasing: d[o..<o + lit])
                        .copyMemory(from: UnsafeRawBufferPointer(
                            rebasing: src[i..<i + lit]))
                }
                i += lit
                o += lit
            }
            if !bad && i < src.count {
                bad = i + 2 > src.count
                let off = bad ? 0 : Int(src[i]) | Int(src[i + 1]) << 8
                i += 2
                var m = (token & 15) + 4
                ext = token & 15 == 15
                while ext, i < src.count {
                    m += Int(src[i])
                    ext = src[i] == 255
                    i += 1
                }
                bad = bad || off == 0 || off > o || o + m > at + want
                var k = 0
                while !bad && k < m {
                    dst[o] = dst[o - off]
                    o += 1
                    k += 1
                }
            }
        }
        return bad ? 0 : o - at
    }

    public static func regroup(_ grouped: [UInt8], _ n: Int,
                               into out: UnsafeMutableRawBufferPointer) {
        let split = n / 4
        let rem = n % 4
        for g in 0..<4 {
            var i = g
            var j = g * split + min(g, rem)
            while i < n {
                out[i] = grouped[j]
                i += 4
                j += 1
            }
        }
    }

    public static func group(_ plain: [UInt8]) -> [UInt8] {
        let n = plain.count
        let split = n / 4
        let rem = n % 4
        var out = [UInt8](repeating: 0, count: n)
        for g in 0..<4 {
            var i = g
            var j = g * split + min(g, rem)
            while i < n {
                out[j] = plain[i]
                i += 4
                j += 1
            }
        }
        return out
    }
}

public final class ChunkStream {
    private var buffer: [UInt8] = []
    private var start = 0
    private var out: [UInt8] = []
    private var scratch: [UInt8] = []
    public private(set) var ordinal = 0

    public init() {}

    public var pending: Int { buffer.count - start }

    public func push(_ data: Data,
                     _ sink: (Int, UnsafeRawBufferPointer) throws -> Void)
        throws {
        buffer.append(contentsOf: data)
        var more = true
        while more {
            let header = try peek()
            if let h = header, pending >= ChunkHeader.bytes + h.compressed {
                try take(h, sink)
            } else {
                more = false
            }
        }
        if start > 0 && start * 2 > buffer.count {
            buffer.removeSubrange(0..<start)
            start = 0
        }
    }

    private func peek() throws -> ChunkHeader? {
        var h: ChunkHeader? = nil
        if pending >= ChunkHeader.bytes {
            h = try buffer.withUnsafeBytes { raw in
                try ChunkHeader.parse(UnsafeRawBufferPointer(
                    rebasing: raw[start..<start + ChunkHeader.bytes]))
            }
        }
        return h
    }

    private func take(_ h: ChunkHeader,
                      _ sink: (Int, UnsafeRawBufferPointer) throws -> Void)
        throws {
        if out.count < h.uncompressed {
            out = [UInt8](repeating: 0,
                          count: max(h.uncompressed, ChunkHeader.largest))
        }
        let from = start + ChunkHeader.bytes
        let decoded = try buffer.withUnsafeBytes { raw in
            try out.withUnsafeMutableBytes { o in
                try Chunk.decode(
                    h, UnsafeRawBufferPointer(
                        rebasing: raw[from..<from + h.compressed]),
                    into: UnsafeMutableRawBufferPointer(
                        rebasing: o[0..<h.uncompressed]),
                    scratch: &scratch)
                return h.uncompressed
            }
        }
        start = from + h.compressed
        try out.withUnsafeBytes { o in
            try sink(ordinal, UnsafeRawBufferPointer(rebasing: o[0..<decoded]))
        }
        ordinal += 1
    }
}
