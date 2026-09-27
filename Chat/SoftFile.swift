import Foundation
import LLM

public struct SoftTurn: Sendable {
    public let stamp: String
    public let labelled: Bool
    public let parts: [ContentPart]
    public let spans: [SoftSpan]

    public init(stamp: String, labelled: Bool, parts: [ContentPart],
                spans: [SoftSpan]) {
        self.stamp = stamp
        self.labelled = labelled
        self.parts = parts
        self.spans = spans
    }

    public var rows: Int {
        spans.reduce(0) { sum, span in sum + span.rows }
    }
}

public enum SoftFileError: Error, Equatable {
    case range
    case format
}

public enum SoftFile {

    static let magic: [UInt8] = Array("SOFT".utf8)
    static let version = 1
    static let none = -1
    static let f16Limit: Float = 65504

    nonisolated public static let dir: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        let dir = support.appendingPathComponent("soft", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    public static func url(_ name: String) -> URL {
        dir.appendingPathComponent(name)
    }

    public static func inRange(_ spans: [SoftSpan]) -> Bool {
        spans.allSatisfy { span in
            span.features.allSatisfy { f in f.isFinite && abs(f) <= f16Limit }
        }
    }

    public static func write(_ turn: SoftTurn, to url: URL) throws {
        if !inRange(turn.spans) { throw SoftFileError.range }
        var out = Data()
        out.append(contentsOf: magic)
        putInt(&out, version)
        putBytes(&out, Array(turn.stamp.utf8))
        putInt(&out, turn.labelled ? 1 : 0)
        putInt(&out, turn.parts.count)
        for part in turn.parts {
            switch part {
            case .text(let s):
                putInt(&out, 0)
                putBytes(&out, Array(s.utf8))
            case .image: putInt(&out, 1)
            case .audio: putInt(&out, 2)
            case .video: putInt(&out, 3)
            }
        }
        putInt(&out, turn.spans.count)
        for span in turn.spans {
            putInt(&out, Int(span.placeholder))
            putInt(&out, span.ids.count)
            span.ids.withUnsafeBytes { raw in out.append(contentsOf: raw) }
            putInt(&out, span.grid?.h ?? none)
            putInt(&out, span.grid?.w ?? none)
            putInt(&out, span.wrap.map { w in Int(w.begin) } ?? none)
            putInt(&out, span.wrap.map { w in Int(w.end) } ?? none)
            let halves = span.features.map { f in Float16(f) }
            putInt(&out, halves.count)
            halves.withUnsafeBytes { raw in out.append(contentsOf: raw) }
        }
        try out.write(to: url, options: .atomic)
    }

    public static func read(_ url: URL) throws -> SoftTurn {
        let data = try Data(contentsOf: url)
        var r = Reader(data)
        if !r.header() { throw SoftFileError.format }
        let stamp = String(decoding: r.bytes(), as: UTF8.self)
        let labelled = r.int() == 1
        var parts: [ContentPart] = []
        for _ in 0 ..< r.int() {
            switch r.int() {
            case 0: parts.append(.text(String(decoding: r.bytes(),
                                              as: UTF8.self)))
            case 1: parts.append(.image)
            case 2: parts.append(.audio)
            default: parts.append(.video)
            }
        }
        var spans: [SoftSpan] = []
        for _ in 0 ..< r.int() {
            let placeholder = Int32(r.int())
            let ids = r.ints32(r.int())
            let h = r.int()
            let w = r.int()
            let begin = r.int()
            let end = r.int()
            let features = r.halves(r.int())
            spans.append(SoftSpan(
                placeholder: placeholder, ids: ids, features: features,
                grid: h == none ? nil : (h: h, w: w),
                wrap: begin == none ? nil : (begin: Int32(begin),
                                             end: Int32(end))))
        }
        if !r.whole { throw SoftFileError.format }
        return SoftTurn(stamp: stamp, labelled: labelled, parts: parts,
                        spans: spans)
    }

    public static func stamp(of url: URL) -> String? {
        var out: String? = nil
        if let handle = try? FileHandle(forReadingFrom: url),
           let head = try? handle.read(upToCount: 4096) {
            var r = Reader(head)
            if r.header() { out = String(decoding: r.bytes(), as: UTF8.self) }
            try? handle.close()
        }
        return out
    }

    private static func putInt(_ out: inout Data, _ v: Int) {
        var x = Int64(v).littleEndian
        withUnsafeBytes(of: &x) { raw in out.append(contentsOf: raw) }
    }

    private static func putBytes(_ out: inout Data, _ bytes: [UInt8]) {
        putInt(&out, bytes.count)
        out.append(contentsOf: bytes)
    }

    private struct Reader {
        let data: Data
        var at = 0
        var whole = true

        init(_ data: Data) { self.data = data }

        mutating func header() -> Bool {
            var named = data.count >= magic.count + 8
            if named {
                for i in 0 ..< magic.count where data[i] != magic[i] {
                    named = false
                }
                at = magic.count
            }
            return named && int() == version
        }

        mutating func int() -> Int {
            var out = 0
            if at + 8 <= data.count {
                var x: Int64 = 0
                _ = withUnsafeMutableBytes(of: &x) { raw in
                    data.copyBytes(to: raw, from: at ..< (at + 8))
                }
                out = Int(Int64(littleEndian: x))
            } else {
                whole = false
            }
            at += 8
            return out
        }

        mutating func bytes() -> [UInt8] {
            let n = max(int(), 0)
            return take(n) { raw in Array(raw) }
        }

        mutating func ints32(_ n: Int) -> [Int32] {
            take(n * 4) { raw in
                Array(raw.bindMemory(to: Int32.self))
            }
        }

        mutating func halves(_ n: Int) -> [Float] {
            take(n * 2) { raw in
                raw.bindMemory(to: Float16.self).map { h in Float(h) }
            }
        }

        private mutating func take<T>(
            _ n: Int, _ body: (UnsafeRawBufferPointer) -> [T]) -> [T] {
            var out: [T] = []
            if n >= 0, at + n <= data.count {
                out = data.subdata(in: at ..< (at + n)).withUnsafeBytes(body)
            } else {
                whole = false
            }
            at += max(n, 0)
            return out
        }
    }

}
