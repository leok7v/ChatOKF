import Accelerate
import Foundation

public struct TranscriptPiece: Sendable, Equatable {
    public let offset: Int
    public let sample: Int
    public let score: Float
}

public struct Transcript: Sendable, Equatable {
    public let text: String
    public let bytes: [UInt8]
    public let pieces: [TranscriptPiece]
    public let steps: Int

    public func piece(_ index: Int) -> String {
        let from = pieces[index].offset
        let to = index + 1 < pieces.count ?
            pieces[index + 1].offset : bytes.count
        return String(decoding: bytes[from..<to], as: UTF8.self)
    }
}

public final class Transcriber: @unchecked Sendable {

    public static let sampleRate = 16000
    public static let windowSeconds = 30.0
    public static let languages: [String] = Whistle.codes

    private let engine: Whistle
    private let lock = NSLock()

    public init?(pack path: String, packed: Bool = false, beam: Int = 5,
                 depth: Int = 8) {
        let fits = (1...Model.beams).contains(beam) &&
            (2...Model.layers).contains(depth)
        if fits, let opened = packMap(path, packed: packed) {
            engine = Whistle(pack: opened, beams: beam, depth: depth)
        } else {
            return nil
        }
    }

    public func transcribe(_ samples: [Float],
                           language: String? = nil) -> Transcript {
        lock.lock()
        defer { lock.unlock() }
        let out = samples.withUnsafeBufferPointer { buffer in
            buffer.baseAddress.map { first in
                engine.transcribe(first, buffer.count, language ?? "")
            } ?? Output()
        }
        return Transcript(text: String(decoding: out.text, as: UTF8.self),
                          bytes: out.text, pieces: out.pieces,
                          steps: out.steps)
    }

    public static func resampled(_ frames: [Float],
                                 rate: Double) -> [Float] {
        let target = Double(sampleRate)
        let count = Int(Double(frames.count) * target / rate)
        var out = [Float](repeating: 0, count: frames.isEmpty ? 0 : count)
        for i in 0..<out.count {
            let at = Double(i) * rate / target
            let a = min(Int(at), frames.count - 1)
            let b = a + 1 < frames.count ? a + 1 : a
            let mix = Float(at - Double(a))
            out[i] = (frames[b] * mix).addingProduct(frames[a], 1 - mix)
        }
        return out
    }

    public static func samples(wav data: Data) -> [Float] {
        let bytes = [UInt8](data)
        let riff = bytes.count >= 12 && bytes[0..<4].elementsEqual(
            "RIFF".utf8) && bytes[8..<12].elementsEqual("WAVE".utf8)
        let format = riff ? wavChunk(bytes, "fmt ") : nil
        let body = riff ? wavChunk(bytes, "data") : nil
        var out: [Float] = []
        if let format, let body, format.count >= 16,
           word(bytes, format.at) == 1, word(bytes, format.at + 14) == 16 {
            let channels = max(word(bytes, format.at + 2), 1)
            let rate = Double(word(bytes, format.at + 4) |
                              word(bytes, format.at + 6) << 16)
            let frames = (0..<(body.count / (channels * 2))).map { i in
                wavFrame(bytes, body.at + i * channels * 2, channels)
            }
            out = resampled(frames, rate: rate)
        }
        return out
    }
}

func word(_ bytes: [UInt8], _ at: Int) -> Int {
    Int(bytes[at]) | Int(bytes[at + 1]) << 8
}

func wavChunk(_ bytes: [UInt8], _ id: String) -> (at: Int, count: Int)? {
    var at = 12
    var found: (at: Int, count: Int)? = nil
    while found == nil && at + 8 <= bytes.count {
        let n = word(bytes, at + 4) | word(bytes, at + 6) << 16
        if bytes[at..<(at + 4)].elementsEqual(id.utf8) {
            found = (at + 8, min(n, bytes.count - at - 8))
        }
        at += 8 + n + (n & 1)
    }
    return found
}

func wavFrame(_ bytes: [UInt8], _ at: Int, _ channels: Int) -> Float {
    var sum = 0
    for c in 0..<channels {
        sum += Int(Int16(bitPattern: UInt16(word(bytes, at + c * 2))))
    }
    return Float(sum) / (32768 * Float(channels))
}

struct Output {
    var text: [UInt8] = []
    var pieces: [TranscriptPiece] = []
    var steps = 0
}

final class Whistle {
    static let codes = ["en", "de", "fr", "es", "it", "nl", "pl"]

    let pack: WhistlePack
    let decoder: WhistleDecoder

    init(pack: WhistlePack, beams: Int, depth: Int) {
        self.pack = pack
        decoder = WhistleDecoder(pack: pack, beams: beams, depth: depth)
    }

    static func tag(_ language: String) -> Int {
        let at = codes.firstIndex(of: language)
        return at.map { i in Model.text + i } ?? 0
    }

    private func byte(_ at: Int) -> UInt8 {
        pack.map.load(fromByteOffset: at, as: UInt8.self)
    }

    private func hex(_ at: Int, _ size: Int) -> UInt8 {
        let bytes = (0..<max(size - 3, 0)).map { i in byte(at + 3 + i) }
        let digits = bytes.prefix { b in
            (48...57).contains(b) || (65...70).contains(b) ||
                (97...102).contains(b)
        }
        let value = Int(String(decoding: digits, as: UTF8.self),
                        radix: 16) ?? 0
        return UInt8(truncatingIfNeeded: value)
    }

    private func pieceWrite(_ text: inout [UInt8], _ token: Int) {
        let piece = pack.pieces[token]
        let p = piece.at
        let n = piece.size
        if piece.kind == Layout.pieceByte {
            text.append(hex(p, n))
        } else if piece.kind == Layout.pieceNormal {
            var i = 0
            while i < n {
                let space = i + 2 < n && byte(p + i) == 0xE2 &&
                    byte(p + i + 1) == 0x96 && byte(p + i + 2) == 0x81
                if !space {
                    text.append(byte(p + i))
                } else if !text.isEmpty && text.last != 32 {
                    text.append(32)
                }
                i += space ? 3 : 1
            }
        }
    }

    private func pieceEnds(_ token: Int) -> Bool {
        let piece = pack.pieces[token]
        let last = piece.size > 0 ? byte(piece.at + piece.size - 1) : 0
        return piece.kind == Layout.pieceNormal &&
            (last == 46 || last == 33 || last == 63)
    }

    private func pieceStarts(_ token: Int) -> Bool {
        let piece = pack.pieces[token]
        return piece.size > 2 && byte(piece.at) == 0xE2 &&
            byte(piece.at + 1) == 0x96 && byte(piece.at + 2) == 0x81
    }

    private func sentenceEnd(_ best: Best) -> Int {
        var stop = 0
        for i in stride(from: 2, to: best.token.count - 1, by: 1) {
            let end = pieceEnds(best.token[i]) &&
                pieceStarts(best.token[i + 1]) &&
                best.frame[i + 1] > best.frame[i - 1]
            stop = end ? i : stop
        }
        return stop
    }

    private func clip(_ samples: UnsafePointer<Float>, _ count: Int,
                      _ base: Int, _ forced: Int, _ more: Bool,
                      _ out: inout Output) -> Int {
        let m = memoryOpen(pack, decoder.layers, samples, count)
        var best = Best()
        if m.spread >= Model.steady {
            out.steps += decoder.search(m, forced, &best)
        }
        let stop = more ? sentenceEnd(best) : 0
        let last = stop > 0 ? stop + 1 : best.token.count
        for i in stride(from: 2, to: last, by: 1) {
            out.pieces.append(TranscriptPiece(
                offset: out.text.count,
                sample: base + best.frame[i] * Model.stride,
                score: best.gain[i]))
            pieceWrite(&out.text, best.token[i])
        }
        decoder.release()
        m.free()
        return stop > 0 ?
            (best.frame[stop - 1] + best.frame[stop + 1]) * Model.stride / 2 :
            count
    }

    private func windowEnd(_ samples: UnsafePointer<Float>, _ at: Int,
                           _ count: Int) -> Int {
        let long = count - at > Model.samples
        let last = at + Model.samples - Model.slice
        var end = long ? at + Model.samples : count
        var least = Float.infinity
        var i = at + Model.samples - Model.search
        while long && i <= last {
            let energy = dot(samples + i, samples + i, Model.slice)
            if energy < least {
                least = energy
                end = i + Model.slice / 2
            }
            i += Model.hop
        }
        return end
    }

    func transcribe(_ samples: UnsafePointer<Float>, _ count: Int,
                    _ language: String) -> Output {
        let forced = Whistle.tag(language)
        var out = Output()
        var at = 0
        while at < count {
            let end = windowEnd(samples, at, count)
            at += clip(samples + at, end - at, at, forced, end < count, &out)
            if at < count && !out.text.isEmpty {
                out.text.append(32)
            }
        }
        return out
    }
}

typealias Floats = UnsafeMutablePointer<Float>

enum Model {
    static let dim = 512
    static let heads = 8
    static let groups = 2
    static let qk = 48
    static let vd = 64
    static let lanes = 4
    static let layers = 8
    static let mels = 80
    static let bins = 257
    static let fft = 512
    static let win = 400
    static let hop = 160
    static let stem = 128
    static let kernel = 9
    static let taps = 3
    static let rank = 8
    static let ha = 16
    static let hb = 32
    static let group = 128
    static let slots = 18432
    static let tables = 4
    static let sub = 128
    static let sites = 2
    static let etaps = 4
    static let dilation = 3
    static let text = 8192
    static let vocab = 8199
    static let tokens = 320
    static let samples = 480000
    static let eos = 1
    static let bos = 2
    static let gates = 20
    static let search = 80000
    static let slice = 1600
    static let stride = 1280
    static let beams = 8
    static let steady: Float = 0.7
}

enum Slot {
    static let embed = 0
    static let decoder = 1
    static let decoderStride = 27
    static let decoderLanes = 217
    static let p1 = 226
    static let p2 = 227
    static let engram = 228
    static let decoderFinal = 236
    static let manifest = 237
    static let cross = 238
    static let crossStride = 10
    static let peGate = 318
    static let encoder = 319
    static let encoderStride = 43
    static let encoderLanes = 663
    static let encoderFinal = 672
    static let stem = 673
    static let mel = 679
    static let tokenizer = 680
    static let count = 681
}

enum Layout {
    static let header = 196
    static let record = 44
    static let books = 28
    static let fp16 = 1
    static let fp32 = 2
    static let cq = 3
    static let raw = 4
    static let tag = 0x05E1_2A84
    static let pieceNormal = 0
    static let pieceByte = 4
    static let manifest: Float = 3246
}

struct Piece {
    let at: Int
    let size: Int
    let kind: Int
}

func floats(_ count: Int) -> Floats {
    let n = max(count, 1)
    let p = Floats.allocate(capacity: n)
    p.initialize(repeating: 0, count: n)
    return p
}

func le8(_ map: UnsafeRawPointer, _ at: Int) -> Int {
    Int(map.load(fromByteOffset: at, as: UInt8.self))
}

func le16(_ map: UnsafeRawPointer, _ at: Int) -> Int {
    Int(UInt16(littleEndian: map.loadUnaligned(fromByteOffset: at,
                                               as: UInt16.self)))
}

func le32(_ map: UnsafeRawPointer, _ at: Int) -> Int {
    Int(UInt32(littleEndian: map.loadUnaligned(fromByteOffset: at,
                                               as: UInt32.self)))
}

func le64(_ map: UnsafeRawPointer, _ at: Int) -> Int {
    Int(truncatingIfNeeded: UInt64(littleEndian: map.loadUnaligned(
        fromByteOffset: at, as: UInt64.self)))
}

func half(_ map: UnsafeRawPointer, _ at: Int) -> Float {
    Float(Float16(bitPattern: UInt16(le16(map, at))))
}

func single(_ map: UnsafeRawPointer, _ at: Int) -> Float {
    Float(bitPattern: UInt32(le32(map, at)))
}

func recordAt(_ index: Int) -> Int {
    Layout.header + Layout.books * 4 + index * Layout.record
}

func hadamard128(_ v: Floats) {
    var h = 1
    while h < Model.group {
        for i in stride(from: 0, to: Model.group, by: 2 * h) {
            for j in i..<(i + h) {
                let a = v[j]
                let b = v[j + h]
                v[j] = a + b
                v[j + h] = a - b
            }
        }
        h *= 2
    }
}

func recordBytes(_ map: UnsafeRawPointer, _ record: Int) -> Int {
    let kind = le8(map, record)
    let rank = min(le8(map, record + 1), 4)
    let count = (0..<rank).reduce(1) { n, d in
        n * le32(map, record + 4 + d * 4)
    }
    let rows = le32(map, record + 4)
    let groups = (le32(map, record + 8) + Model.group - 1) / Model.group
    let bits = le32(map, record + 40)
    let bytes: Int
    switch kind {
    case Layout.fp16:
        bytes = count * 2
    case Layout.fp32:
        bytes = count * 4
    case Layout.cq:
        bytes = rows * groups * (Model.group * bits / 8 + 2)
    default:
        bytes = 0
    }
    return bytes
}

func recordSound(_ map: UnsafeRawPointer, _ size: Int, _ index: Int) -> Bool {
    let record = recordAt(index)
    let kind = le8(map, record)
    let bits = le32(map, record + 40)
    let blob = le64(map, record + 20)
    let shape = kind != Layout.cq || bits == 2 || bits == 4
    return shape && blob >= 0 && blob + recordBytes(map, record) <= size
}

func whistleValid(_ map: UnsafeRawPointer, _ size: Int) -> Bool {
    let head = size >= recordAt(Slot.count) &&
        le32(map, 0) == Layout.tag && le32(map, 4) == Slot.count &&
        le32(map, 8) == Layout.books && le32(map, 20) == Model.vocab &&
        le32(map, 28) == Model.dim
    let records = head && (0..<Slot.count).allSatisfy { index in
        recordSound(map, size, index)
    }
    let blob = records ? le64(map, recordAt(Slot.manifest) + 20) : 0
    return records && blob + 4 <= size &&
        single(map, blob) == Layout.manifest
}

func piecesLoad(_ map: UnsafeRawPointer, _ size: Int) -> [Piece] {
    let record = recordAt(Slot.tokenizer)
    let start = le64(map, record + 20)
    let ok = le8(map, record) == Layout.raw && start >= 0 &&
        start + 24 <= size && le32(map, start) == Model.vocab
    var out: [Piece] = []
    var p = start + 24
    while ok && out.count < Model.vocab && p + 7 <= size {
        let n = le16(map, p + 5)
        out.append(Piece(at: p + 7, size: n, kind: le8(map, p + 4)))
        p += 7 + n
    }
    return out.count == Model.vocab && p <= size ? out : []
}

func packMap(_ path: String, packed: Bool) -> WhistlePack? {
    let file = open(path, O_RDONLY)
    var status = stat()
    var pack: WhistlePack? = nil
    if file >= 0 && fstat(file, &status) == 0 && status.st_size > 0 {
        let bytes = Int(status.st_size)
        let map: UnsafeMutableRawPointer = mmap(nil, bytes, PROT_READ,
                                                MAP_PRIVATE, file, 0)
        let mapped = map != MAP_FAILED
        let pieces = mapped && whistleValid(map, bytes) ?
            piecesLoad(map, bytes) : []
        if !pieces.isEmpty {
            pack = WhistlePack(map: map, size: bytes, packed: packed,
                               pieces: pieces)
        } else if mapped {
            munmap(map, bytes)
        }
    }
    if file >= 0 { close(file) }
    return pack
}

func linear(_ y: Floats, _ x: UnsafePointer<Float>,
            _ w: UnsafePointer<Float>, _ rows: Int, _ out: Int,
            _ inputs: Int) {
    y.update(repeating: 0, count: rows * out)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows),
                Int32(out), Int32(inputs), 1, x, Int32(inputs), w,
                Int32(inputs), 1, y, Int32(out))
}

func turned(_ x: UnsafePointer<Float>, _ rows: Int, _ inputs: Int,
            _ wide: Int) -> Floats {
    let scale = 1 / sqrtf(Float(Model.group))
    let t = floats(rows * wide)
    for r in 0..<rows {
        let row = t + r * wide
        row.update(from: x + r * inputs, count: inputs)
        for g in stride(from: 0, to: wide, by: Model.group) {
            hadamard128(row + g)
        }
        for i in 0..<wide {
            row[i] *= scale
        }
    }
    return t
}

final class WhistlePack {
    let map: UnsafeMutableRawPointer
    let size: Int
    let packed: Bool
    let pieces: [Piece]
    let books: Floats
    let ropeRate: Floats
    var tensors: [Floats?]
    var p1: [Int] = []
    var p2: [Int] = []

    init(map: UnsafeMutableRawPointer, size: Int, packed: Bool,
         pieces: [Piece]) {
        self.map = map
        self.size = size
        self.packed = packed
        self.pieces = pieces
        books = floats(Layout.books)
        ropeRate = floats(Model.qk / 2)
        tensors = [Floats?](repeating: nil, count: Slot.count)
        load()
    }

    deinit {
        for tensor in tensors {
            tensor?.deallocate()
        }
        books.deallocate()
        ropeRate.deallocate()
        munmap(map, size)
    }

    private func load() {
        for i in 0..<Layout.books {
            books[i] = single(map, Layout.header + i * 4)
        }
        for i in 0..<Slot.count where !lazy(i) {
            tensorLoad(i)
        }
        let a = tensors[Slot.p1]
        let b = tensors[Slot.p2]
        p1 = (0..<Model.dim).map { i in Int(a?[i] ?? 0) }
        p2 = (0..<Model.dim).map { i in Int(b?[i] ?? 0) }
        for i in 0..<Model.qk / 2 {
            ropeRate[i] = powf(100000, -Float(2 * i) / Float(Model.qk))
        }
    }

    func tensor(_ index: Int) -> Floats {
        tensors[index]!
    }

    func lazy(_ index: Int) -> Bool {
        let table = index == Slot.engram || index == Slot.engram + 4
        let block = index >= Slot.encoder && index < Slot.encoderLanes
        return table ||
            ((block || packed) && le8(map, recordAt(index)) == Layout.cq)
    }

    private func groupDecode(_ out: Floats, _ row: Int, _ at: Int,
                             _ bits: Int, _ norm: Float, _ count: Int) {
        let book = books + (bits == 2 ? 0 : 12)
        let scale = 1 / sqrtf(Float(Model.group))
        let mask = (1 << bits) - 1
        let v = floats(Model.group)
        for k in 0..<Model.group {
            let bit = (at * Model.group + k) * bits
            let index = (le8(map, row + (bit >> 3)) >> (bit & 7)) & mask
            v[k] = book[index] * norm
        }
        hadamard128(v)
        for k in 0..<count {
            out[k] = v[k] * scale
        }
        v.deallocate()
    }

    private func matrixDecode(_ out: Floats, _ blob: Int, _ rows: Int,
                              _ columns: Int, _ bits: Int) {
        let groups = (columns + Model.group - 1) / Model.group
        let rowBytes = groups * Model.group * bits / 8
        let norms = blob + rows * rowBytes
        for r in 0..<rows {
            for g in 0..<groups {
                let left = columns - g * Model.group
                groupDecode(out + r * columns + g * Model.group,
                            blob + r * rowBytes, g, bits,
                            half(map, norms + (r * groups + g) * 2),
                            min(left, Model.group))
            }
        }
    }

    func rowDecode(_ out: Floats, _ index: Int, _ row: Int) {
        let record = recordAt(index)
        let rows = le32(map, record + 4)
        let columns = le32(map, record + 8)
        let bits = le32(map, record + 40)
        let groups = (columns + Model.group - 1) / Model.group
        let rowBytes = groups * Model.group * bits / 8
        let blob = le64(map, record + 20)
        let norms = blob + rows * rowBytes
        for g in 0..<groups {
            let left = columns - g * Model.group
            groupDecode(out + g * Model.group, blob + row * rowBytes, g,
                        bits, half(map, norms + (row * groups + g) * 2),
                        min(left, Model.group))
        }
    }

    private func tensorLoad(_ index: Int) {
        let record = recordAt(index)
        let kind = le8(map, record)
        let blob = le64(map, record + 20)
        let rank = min(le8(map, record + 1), 4)
        let count = (0..<rank).reduce(1) { n, d in
            n * le32(map, record + 4 + d * 4)
        }
        if kind != Layout.raw {
            let out = floats(count)
            tensors[index] = out
            if kind == Layout.fp16 {
                for i in 0..<count {
                    out[i] = half(map, blob + i * 2)
                }
            } else if kind == Layout.fp32 {
                for i in 0..<count {
                    out[i] = single(map, blob + i * 4)
                }
            } else if kind == Layout.cq {
                matrixDecode(out, blob, le32(map, record + 4),
                             le32(map, record + 8),
                             le32(map, record + 40))
            }
        }
    }

    func layerLoad(_ l: Int, _ wanted: Bool) {
        let base = Slot.encoder + l * Slot.encoderStride
        for i in base..<(base + Slot.encoderStride) where lazy(i) {
            if wanted {
                tensorLoad(i)
            } else {
                tensors[i]?.deallocate()
                tensors[i] = nil
            }
        }
    }

    private func groupValues(_ v: Floats, _ codes: Int,
                             _ lut: UnsafePointer<Float>, _ per: Int) {
        for i in 0..<(Model.group / per) {
            (v + i * per).update(from: lut + le8(map, codes + i) * per,
                                 count: per)
        }
    }

    private func packedProduct(_ y: Floats, _ x: UnsafePointer<Float>,
                               _ codes: Int, _ norms: Int, _ bits: Int,
                               _ out: Int, _ inputs: Int, _ rows: Int) {
        let per = 8 / bits
        let groups = (inputs + Model.group - 1) / Model.group
        let wide = groups * Model.group
        let groupBytes = Model.group / per
        let book = books + (bits == 2 ? 0 : 12)
        let mask = (1 << bits) - 1
        let t = turned(x, rows, inputs, wide)
        let lut = floats(256 * 4)
        let v = floats(Model.group)
        for b in 0..<(256 * per) {
            lut[b] = book[((b / per) >> ((b % per) * bits)) & mask]
        }
        y.update(repeating: 0, count: rows * out)
        for r in 0..<out {
            for g in 0..<groups {
                let at = r * groups + g
                let norm = half(map, norms + at * 2)
                groupValues(v, codes + at * groupBytes, lut, per)
                for i in 0..<rows {
                    let k = i * out + r
                    y[k] = y[k].addingProduct(
                        norm, dot(v, t + i * wide + g * Model.group,
                                  Model.group))
                }
            }
        }
        t.deallocate()
        lut.deallocate()
        v.deallocate()
    }

    func product(_ y: Floats, _ x: UnsafePointer<Float>, _ index: Int,
                 _ first: Int, _ out: Int, _ rows: Int) {
        let record = recordAt(index)
        let inputs = le32(map, record + 8)
        let bits = le32(map, record + 40)
        let groups = (inputs + Model.group - 1) / Model.group
        let rowBytes = groups * Model.group * bits / 8
        let blob = le64(map, record + 20)
        if let w = tensors[index] {
            linear(y, x, w + first * inputs, rows, out, inputs)
        } else {
            packedProduct(y, x, blob + first * rowBytes,
                          blob + le32(map, record + 4) * rowBytes +
                              first * groups * 2,
                          bits, out, inputs, rows)
        }
    }
}

struct Attention {
    let q: Int
    let k: Int
    let v: Int
    let gate: Int
    let out: Int
}

struct Hadamard {
    let d1: Floats
    let d2: Floats
    let b2: Floats
    let d3: Floats
    let d4: Floats
    let w1a: Floats
    let w1b: Floats
    let w2a: Floats
    let w2b: Floats
    let w3a: Floats
    let w3b: Floats
    let condV: Floats
    let condU: Floats

    init(_ pack: WhistlePack, _ base: Int) {
        d1 = pack.tensor(base)
        d2 = pack.tensor(base + 1)
        b2 = pack.tensor(base + 2)
        d3 = pack.tensor(base + 3)
        d4 = pack.tensor(base + 4)
        w1a = pack.tensor(base + 5)
        w1b = pack.tensor(base + 6)
        w2a = pack.tensor(base + 7)
        w2b = pack.tensor(base + 8)
        w3a = pack.tensor(base + 9)
        w3b = pack.tensor(base + 10)
        condV = pack.tensor(base + 11)
        condU = pack.tensor(base + 12)
    }
}

struct Lanes {
    let aPre: Floats
    let aPost: Floats
    let aRes: Floats
    let bPre: Floats
    let bPost: Floats
    let bRes: Floats
    let phi: Int

    init(_ pack: WhistlePack, _ base: Int) {
        aPre = pack.tensor(base)
        aPost = pack.tensor(base + 1)
        aRes = pack.tensor(base + 2)
        bPre = pack.tensor(base + 3)
        bPost = pack.tensor(base + 4)
        bRes = pack.tensor(base + 5)
        phi = base + 6
    }
}

func sigmoid(_ x: Float) -> Float {
    1 / (1 + expf(-x))
}

func silu(_ x: Float) -> Float {
    x * sigmoid(x)
}

func lane(_ p: UnsafeRawPointer, _ i: Int) -> SIMD4<Float> {
    p.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)
}

func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
         _ n: Int) -> Float {
    let ra = UnsafeRawPointer(a)
    let rb = UnsafeRawPointer(b)
    let whole = n - n % 16
    var s0 = SIMD4<Float>(repeating: 0)
    var s1 = SIMD4<Float>(repeating: 0)
    var s2 = SIMD4<Float>(repeating: 0)
    var s3 = SIMD4<Float>(repeating: 0)
    var i = 0
    while i < whole {
        s0 = s0.addingProduct(lane(ra, i), lane(rb, i))
        s1 = s1.addingProduct(lane(ra, i + 4), lane(rb, i + 4))
        s2 = s2.addingProduct(lane(ra, i + 8), lane(rb, i + 8))
        s3 = s3.addingProduct(lane(ra, i + 12), lane(rb, i + 12))
        i += 16
    }
    let v = s3 + (s2 + (s1 + s0))
    var sum = (v[0] + v[1]) + (v[2] + v[3])
    for j in whole..<n {
        sum = sum.addingProduct(a[j], b[j])
    }
    return sum
}

func norm(_ y: Floats, _ x: UnsafePointer<Float>,
          _ scale: UnsafePointer<Float>?, _ rows: Int, _ n: Int) {
    for r in 0..<rows {
        let source = x + r * n
        let out = y + r * n
        let gain = 1 / sqrtf(dot(source, source, n) / Float(n) + 1e-6)
        if let scale {
            for i in 0..<n {
                out[i] = (1 + scale[i]) * source[i] * gain
            }
        } else {
            for i in 0..<n {
                out[i] = source[i] * gain
            }
        }
    }
}

func rope(_ x: Floats, _ heads: Int, _ at: Int,
          _ rate: UnsafePointer<Float>) {
    let half = Model.qk / 2
    let c = floats(half)
    let s = floats(half)
    for i in 0..<half {
        c[i] = cosf(Float(at) * rate[i])
        s[i] = sinf(Float(at) * rate[i])
    }
    for h in 0..<heads {
        let p = x + h * Model.qk
        for i in 0..<half {
            let a = p[i]
            let b = p[i + half]
            p[i] = (-(b * s[i])).addingProduct(a, c[i])
            p[i + half] = (a * s[i]).addingProduct(b, c[i])
        }
    }
    c.deallocate()
    s.deallocate()
}

func attend(_ out: Floats, _ q: UnsafePointer<Float>,
            _ k: UnsafePointer<Float>, _ v: UnsafePointer<Float>,
            _ count: Int, _ groups: Int) {
    let score = floats(count)
    let root = sqrtf(Float(Model.qk))
    for h in 0..<Model.heads {
        let g = h * groups / Model.heads
        let o = out + h * Model.vd
        var top = -Float.infinity
        var sum: Float = 0
        for s in 0..<count {
            score[s] = dot(q + h * Model.qk,
                           k + (s * groups + g) * Model.qk, Model.qk) / root
            top = fmaxf(top, score[s])
        }
        for s in 0..<count {
            score[s] = expf(score[s] - top)
            sum += score[s]
        }
        o.update(repeating: 0, count: Model.vd)
        for s in 0..<count {
            let from = v + (s * groups + g) * Model.vd
            let w = score[s] / sum
            for d in 0..<Model.vd {
                o[d] = o[d].addingProduct(w, from[d])
            }
        }
    }
    score.deallocate()
}

func kron(_ out: Floats, _ z: UnsafePointer<Float>,
          _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) {
    let t = floats(Model.ha * Model.hb)
    for i in 0..<Model.ha {
        for l in 0..<Model.hb {
            var sum: Float = 0
            for j in 0..<Model.hb {
                sum = sum.addingProduct(z[i * Model.hb + j],
                                        b[j * Model.hb + l])
            }
            t[i * Model.hb + l] = sum
        }
    }
    for k in 0..<Model.ha {
        for l in 0..<Model.hb {
            var sum: Float = 0
            for i in 0..<Model.ha {
                sum = sum.addingProduct(a[i * Model.ha + k],
                                        t[i * Model.hb + l])
            }
            out[k * Model.hb + l] = sum
        }
    }
    t.deallocate()
}

func condition(_ cond: Floats, _ x: UnsafePointer<Float>,
               _ h: Hadamard) {
    let soft = floats(Model.rank)
    var top = -Float.infinity
    var sum: Float = 0
    for c in 0..<Model.dim {
        for j in 0..<Model.rank {
            soft[j] = soft[j].addingProduct(x[c], h.condV[c * Model.rank + j])
        }
    }
    for j in 0..<Model.rank {
        top = fmaxf(top, soft[j])
    }
    for j in 0..<Model.rank {
        soft[j] = expf(soft[j] - top)
        sum += soft[j]
    }
    cond.update(repeating: 1, count: Model.dim)
    for j in 0..<Model.rank {
        for i in 0..<Model.dim {
            cond[i] = cond[i].addingProduct(soft[j] / sum,
                                            h.condU[j * Model.dim + i])
        }
    }
    soft.deallocate()
}

func hadamardRow(_ pack: WhistlePack, _ out: Floats,
                 _ x: UnsafePointer<Float>, _ h: Hadamard) {
    let cond = floats(Model.dim)
    let z = floats(Model.dim)
    let t = floats(Model.dim)
    condition(cond, x, h)
    for i in 0..<Model.dim {
        z[i] = h.d1[i] * x[i]
    }
    kron(t, z, h.w1a, h.w1b)
    for i in 0..<Model.dim {
        z[i] = silu(h.b2[i].addingProduct(h.d2[i] * cond[i],
                                          t[pack.p1[i]]))
    }
    kron(t, z, h.w2a, h.w2b)
    for i in 0..<Model.dim {
        z[i] = h.d3[i] * t[pack.p2[i]]
    }
    kron(t, z, h.w3a, h.w3b)
    for i in 0..<Model.dim {
        out[i] = h.d4[i] * t[i]
    }
    cond.deallocate()
    z.deallocate()
    t.deallocate()
}

func hadamardAdd(_ pack: WhistlePack, _ x: Floats,
                 _ scale: UnsafePointer<Float>, _ h: Hadamard,
                 _ rows: Int, _ share: Float) {
    let source = floats(Model.dim)
    let out = floats(Model.dim)
    for r in 0..<rows {
        let row = x + r * Model.dim
        norm(source, row, scale, 1, Model.dim)
        hadamardRow(pack, out, source, h)
        for i in 0..<Model.dim {
            row[i] = row[i].addingProduct(share, out[i])
        }
    }
    source.deallocate()
    out.deallocate()
}

func logSum(_ v: UnsafePointer<Float>, _ step: Int) -> Float {
    var top = -Float.infinity
    var sum: Float = 0
    for i in 0..<Model.lanes {
        top = fmaxf(top, v[i * step])
    }
    for i in 0..<Model.lanes {
        sum += expf(v[i * step] - top)
    }
    return top + logf(sum)
}

func sinkhorn(_ k: Floats) {
    let n = Model.lanes
    for _ in 0..<20 {
        for i in 0..<n {
            let total = logSum(k + i * n, 1)
            for j in 0..<n {
                k[i * n + j] -= total
            }
        }
        for j in 0..<n {
            let total = logSum(k + j, n)
            for i in 0..<n {
                k[i * n + j] -= total
            }
        }
    }
    for i in 0..<(n * n) {
        k[i] = expf(k[i])
    }
}

func lanesGates(_ m: Lanes, _ l: Int, _ own: Int, _ u: Floats,
                _ gate: Floats, _ x: UnsafePointer<Float>,
                _ mix: UnsafePointer<Float>) {
    let n = Model.lanes
    u.update(repeating: 0, count: Model.dim)
    for j in 0..<n {
        let pre = sigmoid(m.bPre[l * n + j].addingProduct(m.aPre[l], mix[j]) +
                          (j == own ? 4 : -4))
        gate[j] = 2 * sigmoid(m.bPost[l * n + j]
            .addingProduct(m.aPost[l], mix[n + j]) + (j == own ? 0 : -4))
        for i in 0..<Model.dim {
            u[i] = u[i].addingProduct(pre, x[j * Model.dim + i])
        }
    }
    for i in 0..<(n * n) {
        gate[n + i] = m.bRes[l * n * n + i].addingProduct(m.aRes[l],
                                                          mix[2 * n + i])
    }
    sinkhorn(gate + n)
}

func lanesEnter(_ pack: WhistlePack, _ m: Lanes, _ l: Int, _ own: Int,
                _ stream: UnsafePointer<Float>, _ u: Floats,
                _ gate: Floats, _ rows: Int) {
    let n = Model.lanes
    let wide = n * Model.dim
    let mixes = 2 * n + n * n
    let unit = floats(rows * wide)
    let mix = floats(rows * mixes)
    let part = floats(rows * n * n)
    norm(unit, stream, nil, rows, wide)
    for k in 0..<3 {
        let count = k == 2 ? n * n : n
        pack.product(part, unit, m.phi + k, l * count, count, rows)
        for r in 0..<rows {
            (mix + r * mixes + k * n).update(from: part + r * count,
                                             count: count)
        }
    }
    for r in 0..<rows {
        lanesGates(m, l, own, u + r * Model.dim, gate + r * Model.gates,
                   stream + r * wide, mix + r * mixes)
    }
    unit.deallocate()
    mix.deallocate()
    part.deallocate()
}

func lanesLeave(_ stream: Floats, _ u: UnsafePointer<Float>,
                _ y: UnsafePointer<Float>, _ gate: UnsafePointer<Float>,
                _ rows: Int) {
    let n = Model.lanes
    let dim = Model.dim
    let old = floats(n * dim)
    for r in 0..<rows {
        let x = stream + r * n * dim
        let g = gate + r * Model.gates
        old.update(from: x, count: n * dim)
        for i in 0..<n {
            for d in 0..<dim {
                x[i * dim + d] = g[i] * (y[r * dim + d] - u[r * dim + d])
            }
            for j in 0..<n {
                let w = g[n + i * n + j]
                for d in 0..<dim {
                    x[i * dim + d] = x[i * dim + d].addingProduct(
                        w, old[j * dim + d])
                }
            }
        }
    }
    old.deallocate()
}

func lanesMean(_ y: Floats, _ stream: UnsafePointer<Float>,
               _ scale: UnsafePointer<Float>, _ rows: Int) {
    let dim = Model.dim
    for r in 0..<rows {
        let x = stream + r * Model.lanes * dim
        let out = y + r * dim
        for d in 0..<dim {
            out[d] = (x[d] + x[dim + d] + x[2 * dim + d] + x[3 * dim + d]) /
                Float(Model.lanes)
        }
    }
    norm(y, y, scale, rows, dim)
}

func gated(_ pack: WhistlePack, _ y: Floats, _ mix: UnsafePointer<Float>,
           _ x: UnsafePointer<Float>, _ a: Attention, _ rows: Int) {
    let gate = floats(rows * Model.dim)
    let open = floats(rows * Model.dim)
    pack.product(gate, x, a.gate, 0, Model.dim, rows)
    for i in 0..<(rows * Model.dim) {
        open[i] = mix[i] * sigmoid(gate[i])
    }
    pack.product(y, open, a.out, 0, Model.dim, rows)
    gate.deallocate()
    open.deallocate()
}

struct Memory {
    let k: Floats
    let v: Floats
    let rows: Int
    let length: Int
    let spread: Float

    func free() {
        k.deallocate()
        v.deallocate()
    }
}

func encoderRows(_ count: Int) -> Int {
    let frames = 1 + count / Model.hop
    let all = ((((frames - 1) / 2 + 1) - 1) / 2 + 1 - 1) / 2 + 1
    return all + all % 2
}

func fft(_ re: Floats, _ im: Floats) {
    let pi = Float(Double.pi)
    var j = 0
    for i in 1..<Model.fft {
        var bit = Model.fft >> 1
        while (j & bit) != 0 {
            j ^= bit
            bit >>= 1
        }
        j ^= bit
        if i < j {
            let a = re[i]
            let b = im[i]
            re[i] = re[j]
            im[i] = im[j]
            re[j] = a
            im[j] = b
        }
    }
    var size = 2
    while size <= Model.fft {
        let step = -2 * pi / Float(size)
        for i in stride(from: 0, to: Model.fft, by: size) {
            for k in 0..<(size / 2) {
                let c = cosf(step * Float(k))
                let s = sinf(step * Float(k))
                let a = i + k
                let b = a + size / 2
                let x = (-(im[b] * s)).addingProduct(re[b], c)
                let y = (im[b] * c).addingProduct(re[b], s)
                re[b] = re[a] - x
                im[b] = im[a] - y
                re[a] += x
                im[a] += y
            }
        }
        size *= 2
    }
}

func spectrum(_ bank: UnsafePointer<Float>, _ mel: Floats,
              _ wave: UnsafePointer<Float>, _ count: Int,
              _ frame: Int) -> Float {
    let pi = Float(Double.pi)
    var energy: Float = 0
    let re = floats(Model.fft)
    let im = floats(Model.fft)
    for n in 0..<Model.win {
        let at = frame * Model.hop - Model.win / 2 + n
        let window = Float(0.5).addingProduct(
            -0.5, cosf(2 * pi * Float(n) / Float(Model.win - 1)))
        re[n] = at >= 0 && at < count ? wave[at] * window : 0
    }
    fft(re, im)
    mel.update(repeating: 0, count: Model.mels)
    for b in 0..<Model.bins {
        let power = (im[b] * im[b]).addingProduct(re[b], re[b])
        energy += power
        for m in 0..<Model.mels {
            mel[m] = mel[m].addingProduct(power, bank[b * Model.mels + m])
        }
    }
    for m in 0..<Model.mels {
        mel[m] = logf(mel[m] + 0x1p-24)
    }
    re.deallocate()
    im.deallocate()
    return logf(energy + 0x1p-24)
}

func deviation(_ v: UnsafePointer<Float>, _ count: Int) -> Float {
    var mean: Float = 0
    var spread: Float = 0
    for i in 0..<count {
        mean += v[i]
    }
    mean /= Float(count > 1 ? count : 1)
    for i in 0..<count {
        spread = spread.addingProduct(v[i] - mean, v[i] - mean)
    }
    return sqrtf(spread / Float(count > 2 ? count - 1 : 1))
}

func features(_ pack: WhistlePack, _ x: Floats, _ wave: UnsafePointer<Float>,
              _ count: Int, _ rows: Int, _ length: Int) -> Float {
    let mels = Model.mels
    let energy = floats(length + 1)
    for f in 0..<length {
        energy[f] = spectrum(pack.tensor(Slot.mel), x + f * mels, wave,
                             count, f)
    }
    let spread = deviation(energy, length)
    energy.deallocate()
    for m in 0..<mels {
        var mean: Float = 0
        var wide: Float = 0
        for f in 0..<length {
            mean += x[f * mels + m]
        }
        mean /= Float(length > 1 ? length : 1)
        for f in 0..<length {
            let d = x[f * mels + m] - mean
            wide = wide.addingProduct(d, d)
        }
        wide = sqrtf(wide / Float(length > 2 ? length - 1 : 1))
        for f in 0..<rows {
            let v = x[f * mels + m]
            x[f * mels + m] = f < length ? (v - mean) / (wide + 1e-5) : 0
        }
    }
    return spread
}

func tap(_ x: UnsafePointer<Float>, _ rows: Int, _ columns: Int,
         _ depth: Int, _ r: Int, _ c: Int, _ d: Int) -> Float {
    r >= 0 && r < rows && c >= 0 && c < columns ?
        x[(r * columns + c) * depth + d] : 0
}

func stemFirst(_ pack: WhistlePack, _ y: Floats, _ x: UnsafePointer<Float>,
               _ rows: Int, _ length: Int) {
    let w = pack.tensor(Slot.stem)
    let half = Model.mels / 2
    for t in 0..<length {
        for f in 0..<half {
            let out = y + (t * half + f) * Model.stem
            for k in 0..<9 {
                let v = tap(x, rows, Model.mels, 1, 2 * t + k / 3 - 1,
                            2 * f + k % 3 - 1, 0)
                for c in 0..<Model.stem {
                    out[c] = out[c].addingProduct(w[k * Model.stem + c], v)
                }
            }
        }
    }
}

func stemDown(_ pack: WhistlePack, _ y: Floats, _ x: UnsafePointer<Float>,
              _ w: UnsafePointer<Float>, _ pw: Int, _ rows: Int,
              _ columns: Int, _ length: Int) {
    let half = columns / 2
    let mid = floats(length * half * Model.stem)
    for t in 0..<length {
        for f in 0..<half {
            let out = mid + (t * half + f) * Model.stem
            for k in 0..<9 {
                for c in 0..<Model.stem {
                    out[c] = out[c].addingProduct(
                        w[k * Model.stem + c],
                        silu(tap(x, rows, columns, Model.stem,
                                 2 * t + k / 3 - 1, 2 * f + k % 3 - 1, c)))
                }
            }
        }
    }
    pack.product(y, mid, pw, 0, Model.stem, length * half)
    mid.deallocate()
}

func stem(_ pack: WhistlePack, _ y: Floats, _ mel: UnsafePointer<Float>,
          _ rows: Int, _ length: Int) -> Int {
    let r1 = (rows - 1) / 2 + 1
    let r2 = (r1 - 1) / 2 + 1
    let r3 = (r2 - 1) / 2 + 1
    let l1 = (length + 1) / 2
    let l2 = (l1 + 1) / 2
    let l3 = (l2 + 1) / 2
    let a = floats(r1 * 40 * Model.stem)
    let b = floats(r2 * 20 * Model.stem)
    let c = floats(r3 * 10 * Model.stem)
    let flat = floats(r3 * 10 * Model.stem)
    stemFirst(pack, a, mel, rows, l1)
    stemDown(pack, b, a, pack.tensor(Slot.stem + 1), Slot.stem + 2, r1, 40,
             l2)
    stemDown(pack, c, b, pack.tensor(Slot.stem + 3), Slot.stem + 4, r2, 20,
             l3)
    for t in 0..<l3 {
        for f in 0..<10 {
            for d in 0..<Model.stem {
                flat[(t * Model.stem + d) * 10 + f] =
                    silu(c[(t * 10 + f) * Model.stem + d])
            }
        }
    }
    pack.product(y, flat, Slot.stem + 5, 0, Model.dim, l3)
    a.deallocate()
    b.deallocate()
    c.deallocate()
    flat.deallocate()
    return l3
}

func encoderAttention(_ pack: WhistlePack, _ x: Floats, _ base: Int,
                      _ rows: Int, _ length: Int) {
    let a = Attention(q: base + 6, k: base + 7, v: base + 8, gate: base + 9,
                      out: base + 10)
    let dim = Model.dim
    let wq = Model.heads * Model.qk
    let wk = Model.groups * Model.qk
    let wv = Model.groups * Model.vd
    let h = floats(rows * dim)
    let q = floats(rows * wq)
    let k = floats(rows * wk)
    let v = floats(rows * wv)
    let mix = floats(rows * dim)
    norm(h, x, pack.tensor(base + 1), rows, dim)
    pack.product(q, h, a.q, 0, wq, rows)
    pack.product(k, h, a.k, 0, wk, rows)
    pack.product(v, h, a.v, 0, wv, rows)
    norm(q, q, pack.tensor(base + 11), rows * Model.heads, Model.qk)
    norm(k, k, pack.tensor(base + 12), rows * Model.groups, Model.qk)
    for t in 0..<rows {
        rope(q + t * wq, Model.heads, t, pack.ropeRate)
        rope(k + t * wk, Model.groups, t, pack.ropeRate)
    }
    for t in 0..<rows {
        attend(mix + t * dim, q + t * wq, k, v, length, Model.groups)
    }
    gated(pack, h, mix, h, a, rows)
    norm(h, h, pack.tensor(base + 2), rows, dim)
    let share = sigmoid(pack.tensor(base + 13)[0])
    for i in 0..<(rows * dim) {
        x[i] = x[i].addingProduct(share, h[i])
    }
    h.deallocate()
    q.deallocate()
    k.deallocate()
    v.deallocate()
    mix.deallocate()
}

func encoderConv(_ pack: WhistlePack, _ x: Floats, _ base: Int,
                 _ rows: Int, _ length: Int) {
    let dim = Model.dim
    let dw = pack.tensor(base + 16)
    let h = floats(rows * dim)
    let wide = floats(rows * 2 * dim)
    let g = floats(rows * dim)
    norm(h, x, pack.tensor(base + 3), rows, dim)
    pack.product(wide, h, base + 14, 0, 2 * dim, rows)
    for t in 0..<length {
        for d in 0..<dim {
            g[t * dim + d] = wide[t * 2 * dim + d] *
                sigmoid(wide[t * 2 * dim + dim + d])
        }
    }
    h.update(repeating: 0, count: rows * dim)
    for t in 0..<rows {
        for k in 0..<Model.kernel {
            for d in 0..<dim {
                h[t * dim + d] = h[t * dim + d].addingProduct(
                    dw[k * dim + d],
                    tap(g, rows, 1, dim, t + k - Model.kernel / 2, 0, d))
            }
        }
    }
    norm(h, h, pack.tensor(base + 4), rows, dim)
    for i in 0..<(rows * dim) {
        h[i] = silu(h[i])
    }
    pack.product(g, h, base + 15, 0, dim, rows)
    for i in 0..<(rows * dim) {
        x[i] += g[i]
    }
    h.deallocate()
    wide.deallocate()
    g.deallocate()
}

func encoderBlock(_ pack: WhistlePack, _ x: Floats, _ l: Int, _ rows: Int,
                  _ length: Int) {
    let base = Slot.encoder + l * Slot.encoderStride
    pack.layerLoad(l, true)
    let first = Hadamard(pack, base + 17)
    let last = Hadamard(pack, base + 30)
    hadamardAdd(pack, x, pack.tensor(base), first, rows, 0.5)
    encoderAttention(pack, x, base, rows, length)
    encoderConv(pack, x, base, rows, length)
    hadamardAdd(pack, x, pack.tensor(base + 5), last, rows, 0.5)
    pack.layerLoad(l, false)
}

func encoder(_ pack: WhistlePack, _ y: Floats, _ x: UnsafePointer<Float>,
             _ rows: Int, _ length: Int) {
    let dim = Model.dim
    let m = Lanes(pack, Slot.encoderLanes)
    let stream = floats(rows * Model.lanes * dim)
    let u = floats(rows * dim)
    let out = floats(rows * dim)
    let gate = floats(rows * Model.gates)
    for r in 0..<rows {
        for j in 0..<Model.lanes {
            (stream + (r * Model.lanes + j) * dim).update(from: x + r * dim,
                                                          count: dim)
        }
    }
    for l in 0..<Model.layers {
        lanesEnter(pack, m, l, l % Model.lanes, stream, u, gate, rows)
        out.update(from: u, count: rows * dim)
        encoderBlock(pack, out, l, rows, length)
        lanesLeave(stream, u, out, gate, rows)
    }
    lanesMean(y, stream, pack.tensor(Slot.encoderFinal), rows)
    stream.deallocate()
    u.deallocate()
    out.deallocate()
    gate.deallocate()
}

func positions(_ enc: Floats, _ rows: Int, _ gate: Float) {
    let dim = Model.dim
    let rate = -Float(log(Double(10000))) / Float(dim / 2 - 1)
    for t in 0..<rows {
        for i in 0..<(dim / 2) {
            let angle = Float(t) * expf(rate * Float(i))
            enc[t * dim + i] = enc[t * dim + i].addingProduct(gate,
                                                              sinf(angle))
            enc[t * dim + dim / 2 + i] =
                enc[t * dim + dim / 2 + i].addingProduct(gate, cosf(angle))
        }
    }
}

func memoryOpen(_ pack: WhistlePack, _ layers: [Int],
                _ wave: UnsafePointer<Float>, _ count: Int) -> Memory {
    let frames = 1 + count / Model.hop
    let rows = encoderRows(count)
    let wq = Model.heads * Model.qk
    let wv = Model.heads * Model.vd
    let mel = floats(frames * Model.mels)
    let x = floats(rows * Model.dim)
    let enc = floats(rows * Model.dim)
    let k = floats(Model.layers * rows * wq)
    let v = floats(Model.layers * rows * wv)
    let spread = features(pack, mel, wave, count, frames, count / Model.hop)
    let length = stem(pack, x, mel, frames, count / Model.hop)
    if spread >= Model.steady {
        encoder(pack, enc, x, rows, length)
    }
    positions(enc, rows, sigmoid(pack.tensor(Slot.peGate)[0]))
    for l in layers {
        let base = Slot.cross + l * Slot.crossStride
        let key = k + l * rows * wq
        pack.product(key, enc, base + 2, 0, wq, rows)
        norm(key, key, pack.tensor(base + 7), rows * Model.heads, Model.qk)
        pack.product(v + l * rows * wv, enc, base + 3, 0, wv, rows)
    }
    mel.deallocate()
    x.deallocate()
    enc.deallocate()
    return Memory(k: k, v: v, rows: rows, length: max(length, 1),
                  spread: spread)
}

final class Node {
    static let wq = Model.heads * Model.qk
    static let wk = Model.groups * Model.qk
    static let wv = Model.groups * Model.vd
    static let rawQ = 0
    static let rawK = rawQ + Model.layers * wq
    static let rawV = rawK + Model.layers * wk
    static let key = rawV + Model.layers * wv
    static let value = key + Model.layers * wk
    static let engram = value + Model.layers * wv
    static let size = engram + Model.sites * Model.dim

    let token: Int
    let gain: Float
    let cache: Floats
    var frame = 0

    init(token: Int, gain: Float) {
        self.token = token
        self.gain = gain
        cache = floats(Node.size)
    }

    deinit {
        cache.deallocate()
    }

    func rawQ(_ l: Int) -> Floats { cache + Node.rawQ + l * Node.wq }
    func rawK(_ l: Int) -> Floats { cache + Node.rawK + l * Node.wk }
    func rawV(_ l: Int) -> Floats { cache + Node.rawV + l * Node.wv }
    func key(_ l: Int) -> Floats { cache + Node.key + l * Node.wk }
    func value(_ l: Int) -> Floats { cache + Node.value + l * Node.wv }
    func engram(_ s: Int) -> Floats { cache + Node.engram + s * Model.dim }
}

struct Beam {
    var path: [Int]
    var score: Float
}

struct Candidate {
    let beam: Int
    let token: Int
    let score: Float
}

struct Best {
    var token: [Int] = []
    var frame: [Int] = []
    var gain: [Float] = []
    var score = -Float.infinity
}

final class WhistleDecoder {
    let pack: WhistlePack
    let width: Int
    let layers: [Int]
    let viewK: Floats
    let viewV: Floats
    var nodes: [Node] = []

    init(pack: WhistlePack, beams: Int, depth: Int) {
        let rank = [0, 4, 5, 2, 6, 3, 7, 1]
        self.pack = pack
        width = beams
        layers = (0..<Model.layers).filter { l in rank[l] < depth }
        viewK = floats((Model.tokens + 1) * Node.wk)
        viewV = floats((Model.tokens + 1) * Node.wv)
    }

    deinit {
        viewK.deallocate()
        viewV.deallocate()
    }

    private func nodeNew(_ token: Int, _ gain: Float) -> Int {
        nodes.append(Node(token: token, gain: gain))
        return nodes.count - 1
    }

    private func node(_ b: Beam, _ position: Int) -> Node {
        nodes[b.path[position]]
    }

    private func engramSlot(_ b: Beam, _ at: Int, _ table: Int) -> Int {
        let order = table / 2 + 2
        var acc: UInt32 = 0x9E37_79B9 &* UInt32(table + 1)
        for j in 0..<order {
            let token = at - j >= 0 ? UInt32(node(b, at - j).token) : 0
            acc = (acc ^ token) &* 0x0100_0193
        }
        acc ^= acc >> 15
        return Int(acc % UInt32(Model.slots))
    }

    private func engram(_ b: Beam, _ key: Floats, _ value: Floats) {
        let at = b.path.count - 1
        let now = node(b, at)
        let e = floats(Model.dim)
        for s in 0..<Model.sites {
            let base = Slot.engram + s * 4
            for table in 0..<Model.tables where at >= table / 2 + 1 {
                let row = table * Model.slots + engramSlot(b, at, table)
                pack.rowDecode(e + table * Model.sub, base, row)
            }
            pack.product(key + s * Model.dim, e, base + 1, 0, Model.dim, 1)
            pack.product(now.engram(s), e, base + 2, 0, Model.dim, 1)
            let out = value + s * Model.dim
            let taps = pack.tensor(base + 3)
            out.update(repeating: 0, count: Model.dim)
            var j = 0
            while j < Model.etaps && at - j * Model.dilation >= 0 {
                let past = node(b, at - j * Model.dilation).engram(s)
                for d in 0..<Model.dim {
                    out[d] = out[d].addingProduct(taps[j * Model.dim + d],
                                                  past[d])
                }
                j += 1
            }
        }
        e.deallocate()
    }

    private func recall(_ x: Floats, _ key: UnsafePointer<Float>,
                        _ value: UnsafePointer<Float>) {
        let a = floats(Model.dim)
        let b = floats(Model.dim)
        norm(a, x, nil, 1, Model.dim)
        norm(b, key, nil, 1, Model.dim)
        let alpha = sigmoid(dot(a, b, Model.dim) / sqrtf(Float(Model.dim)))
        for d in 0..<Model.dim {
            x[d] = x[d].addingProduct(alpha, value[d])
        }
        a.deallocate()
        b.deallocate()
    }

    private func tapped(_ y: Floats, _ past: [Floats],
                        _ taps: UnsafePointer<Float>, _ width: Int) {
        y.update(repeating: 0, count: width)
        for j in 0..<past.count {
            for d in 0..<width {
                y[d] = y[d].addingProduct(taps[j * width + d], past[j][d])
            }
        }
    }

    private func pathView(_ b: Beam, _ l: Int) {
        for p in 0..<b.path.count {
            let one = node(b, p)
            (viewK + p * Node.wk).update(from: one.key(l), count: Node.wk)
            (viewV + p * Node.wv).update(from: one.value(l), count: Node.wv)
        }
    }

    private func attention(_ x: Floats, _ l: Int, _ b: Beam) {
        let base = Slot.decoder + l * Slot.decoderStride
        let at = b.path.count - 1
        let a = Attention(q: base + 1, k: base + 2, v: base + 3,
                          gate: base + 9, out: base + 10)
        let now = node(b, at)
        let back = (0..<min(Model.taps, at + 1)).map { j in node(b, at - j) }
        let h = floats(Model.dim)
        let q = floats(Node.wq)
        let mix = floats(Model.dim)
        norm(h, x, pack.tensor(base), 1, Model.dim)
        pack.product(now.rawQ(l), h, a.q, 0, Node.wq, 1)
        pack.product(now.rawK(l), h, a.k, 0, Node.wk, 1)
        pack.product(now.rawV(l), h, a.v, 0, Node.wv, 1)
        tapped(q, back.map { one in one.rawQ(l) }, pack.tensor(base + 4),
               Node.wq)
        tapped(now.key(l), back.map { one in one.rawK(l) },
               pack.tensor(base + 5), Node.wk)
        tapped(now.value(l), back.map { one in one.rawV(l) },
               pack.tensor(base + 6), Node.wv)
        norm(q, q, pack.tensor(base + 7), Model.heads, Model.qk)
        norm(now.key(l), now.key(l), pack.tensor(base + 8), Model.groups,
             Model.qk)
        rope(q, Model.heads, at, pack.ropeRate)
        rope(now.key(l), Model.groups, at, pack.ropeRate)
        pathView(b, l)
        attend(mix, q, viewK, viewV, at + 1, Model.groups)
        gated(pack, h, mix, h, a, 1)
        norm(h, h, pack.tensor(base + 11), 1, Model.dim)
        let share = sigmoid(pack.tensor(base + 12)[0])
        for d in 0..<Model.dim {
            x[d] = x[d].addingProduct(share, h[d])
        }
        h.deallocate()
        q.deallocate()
        mix.deallocate()
    }

    private func peak(_ q: UnsafePointer<Float>, _ k: UnsafePointer<Float>,
                      _ count: Int) -> Int {
        let total = floats(count)
        let score = floats(count)
        let root = sqrtf(Float(Model.qk))
        var pick = 0
        for h in 0..<Model.heads {
            var top = -Float.infinity
            var sum: Float = 0
            for s in 0..<count {
                score[s] = dot(q + h * Model.qk,
                               k + (s * Model.heads + h) * Model.qk, Model.qk)
                top = fmaxf(top, score[s])
            }
            for s in 0..<count {
                score[s] = expf((score[s] - top) / root)
                sum += score[s]
            }
            for s in 0..<count {
                total[s] += score[s] / sum
            }
        }
        for s in 1..<max(count, 1) {
            pick = total[s] > total[pick] ? s : pick
        }
        total.deallocate()
        score.deallocate()
        return pick
    }

    private func cross(_ x: Floats, _ l: Int, _ m: Memory, _ now: Node) {
        let base = Slot.cross + l * Slot.crossStride
        let a = Attention(q: base + 1, k: 0, v: 0, gate: base + 4,
                          out: base + 5)
        let wq = Model.heads * Model.qk
        let k = m.k + l * m.rows * wq
        let h = floats(Model.dim)
        let q = floats(wq)
        let mix = floats(Model.dim)
        norm(h, x, pack.tensor(base), 1, Model.dim)
        pack.product(q, h, a.q, 0, wq, 1)
        norm(q, q, pack.tensor(base + 6), Model.heads, Model.qk)
        if l == Model.layers - 1 {
            now.frame = peak(q, k, m.length)
        }
        attend(mix, q, k, m.v + l * m.rows * Model.heads * Model.vd,
               m.length, Model.heads)
        gated(pack, h, mix, h, a, 1)
        norm(h, h, pack.tensor(base + 8), 1, Model.dim)
        let share = sigmoid(pack.tensor(base + 9)[0])
        for d in 0..<Model.dim {
            x[d] = x[d].addingProduct(share, h[d])
        }
        h.deallocate()
        q.deallocate()
        mix.deallocate()
    }

    private func step(_ logits: Floats, _ m: Memory, _ b: Beam) {
        let dim = Model.dim
        let mix = Lanes(pack, Slot.decoderLanes)
        let now = node(b, b.path.count - 1)
        let embed = floats(dim)
        let stream = floats(Model.lanes * dim)
        let u = floats(dim)
        let out = floats(dim)
        let gate = floats(Model.gates)
        let key = floats(Model.sites * dim)
        let value = floats(Model.sites * dim)
        let root = sqrtf(Float(dim))
        pack.rowDecode(embed, Slot.embed, now.token)
        for i in 0..<(Model.lanes * dim) {
            stream[i] = embed[i % dim] * root
        }
        engram(b, key, value)
        for (i, l) in layers.enumerated() {
            let base = Slot.decoder + l * Slot.decoderStride
            let h = Hadamard(pack, base + 14)
            lanesEnter(pack, mix, l, i % Model.lanes, stream, u, gate, 1)
            out.update(from: u, count: dim)
            if l % 4 == 3 {
                recall(out, key + l / 4 * dim, value + l / 4 * dim)
            }
            attention(out, l, b)
            cross(out, l, m, now)
            hadamardAdd(pack, out, pack.tensor(base + 13), h, 1, 1)
            lanesLeave(stream, u, out, gate, 1)
        }
        lanesMean(out, stream, pack.tensor(Slot.decoderFinal), 1)
        pack.product(logits, out, Slot.embed, 0, Model.vocab, 1)
        embed.deallocate()
        stream.deallocate()
        u.deallocate()
        out.deallocate()
        gate.deallocate()
        key.deallocate()
        value.deallocate()
    }

    private func best(_ logits: UnsafePointer<Float>, _ from: Int,
                      _ to: Int) -> Int {
        var pick = from
        for i in (from + 1)..<to {
            pick = logits[i] > logits[pick] ? i : pick
        }
        return pick
    }

    private func logTotal(_ logits: UnsafePointer<Float>, _ from: Int,
                          _ to: Int) -> Float {
        var top = -Float.infinity
        var sum: Float = 0
        for i in from..<to {
            top = fmaxf(top, logits[i])
        }
        for i in from..<to {
            sum += expf(logits[i] - top)
        }
        return top + logf(sum)
    }

    private func candidates(_ list: inout [Candidate], _ logits: Floats,
                            _ b: Beam, _ index: Int, _ forced: Int,
                            _ width: Int) {
        let first = b.path.count == 1
        let from = first ? Model.text : 0
        let to = first ? Model.vocab : Model.text
        let total = logTotal(logits, from, to)
        if first && forced != 0 {
            list.append(Candidate(beam: index, token: forced,
                                  score: b.score))
        } else {
            for _ in 0..<width {
                let pick = best(logits, from, to)
                list.append(Candidate(beam: index, token: pick,
                                      score: b.score + logits[pick] - total))
                logits[pick] = -Float.infinity
            }
        }
    }

    private func order(_ list: inout [Candidate]) {
        for i in 1..<max(list.count, 1) {
            let c = list[i]
            var j = i
            while j > 0 && list[j - 1].score < c.score {
                list[j] = list[j - 1]
                j -= 1
            }
            list[j] = c
        }
    }

    private func keep(_ b: Beam, _ score: Float, _ best: inout Best) {
        if score > best.score {
            let path = b.path.map { at in nodes[at] }
            best.score = score
            best.token = path.map { one in one.token }
            best.gain = path.map { one in one.gain }
            best.frame = [0] + path.map { one in one.frame }
        }
    }

    private func advance(_ cur: [Beam], _ list: [Candidate],
                         _ take: Int, _ best: inout Best) -> [Beam] {
        var next: [Beam] = []
        for c in list.prefix(take) {
            let from = cur[c.beam]
            if c.token == Model.eos {
                keep(from, c.score / Float(from.path.count), &best)
            } else {
                let at = nodeNew(c.token, c.score - from.score)
                next.append(Beam(path: from.path + [at], score: c.score))
            }
        }
        return next
    }

    func search(_ m: Memory, _ forced: Int, _ best: inout Best) -> Int {
        let logits = floats(Model.vocab)
        var beams = [Beam(path: [nodeNew(Model.bos, 0)], score: 0)]
        var width = self.width
        var steps = 0
        while !beams.isEmpty && width > 0 &&
            beams[0].path.count <= Model.tokens {
            var list: [Candidate] = []
            for (i, b) in beams.enumerated() {
                step(logits, m, b)
                candidates(&list, logits, b, i, forced, width)
            }
            steps += beams.count
            order(&list)
            let take = min(list.count, width)
            beams = advance(beams, list, take, &best)
            width -= take - beams.count
        }
        for b in beams {
            keep(b, b.score / Float(b.path.count), &best)
        }
        logits.deallocate()
        return steps
    }

    func release() {
        nodes = []
    }
}
