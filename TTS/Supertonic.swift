import Foundation
import Accelerate

let DP    = "duration_predictor.tts.dp"
let TE    = "text_encoder.tts.ttl"
let FIELD = "vector_estimator.vector_estimator.tts.ttl.vector_field"
let MASK  = "vector_estimator.vector_estimator.tts.ttl.uncond_masker"
let AE    = "vocoder.tts.ae"

let textCapacity     = 1024
let arenaFloats      = 1 << 26
let sampleRate       = 44100
let styleHeads       = 2
let fieldGroups      = 4
let chunkLimit       = 300
let koreanChunkLimit = 120
let frameLimit       = 4096
let arenaAlign       = 32
let widenFloats      = 1 << 18
let vocoderSlice     = 256
let q4Block          = 32
let q4Group          = 8
let q4Head           = 20

let encoderDilations        = [1, 1, 2, 2, 4, 4]
let fieldDilations          = [1, 2, 4, 8]
let vocoderDilations        = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
let layerNormEpsilon: Float = 1e-6
let batchNormEpsilon: Float = 1e-5
let timeScale: Float        = 1000.0
let guidance: Float         = 3.0
let chunkSilence            = 0.3

enum Record {
    static let bits   = 120
    static let shape  = 128
    static let count  = 144
    static let offset = 152
    static let size   = 160
}

struct Tensor {
    var name:  UnsafePointer<CChar>
    var data:  UnsafePointer<Float>
    var count: Int
    var shape: (Int, Int, Int, Int)
    var bits:  Int
}

struct Pack {
    var tensors: [Tensor] = []
    var map:     UnsafeMutableRawPointer
    var bytes:   Int
}

final class Engine {
    let weights: Pack
    let arena:   UnsafeMutablePointer<Float>
    var used     = 0
    init(_ weights: Pack, _ arena: UnsafeMutablePointer<Float>) {
        self.weights = weights
        self.arena   = arena
    }
    deinit {
        munmap(arena, arenaFloats * MemoryLayout<Float>.size)
        munmap(weights.map, weights.bytes)
    }
}

struct Condition {
    var text:   UnsafePointer<Float>
    var length: Int
    var keys:   UnsafePointer<Float>
    var values: UnsafePointer<Float>
    var tokens: Int
}

struct Voice {
    var ttl:    UnsafePointer<Float>
    var dp:     UnsafePointer<Float>
    var tokens: Int
}

struct Audio {
    var samples: UnsafeMutablePointer<Float>
    var count:   Int
    var seconds: Float
}

func cast<T>(_ data: UnsafePointer<Float>,
             _ type: T.Type) -> UnsafePointer<T> {
    return UnsafeRawPointer(data).assumingMemoryBound(to: type)
}

func ascii(_ scalar: Unicode.Scalar) -> UInt32 {
    return scalar.value
}

struct Mt19937 {
    var key      = [UInt32](repeating: 0, count: 624)
    var pos      = 0
    var hasGauss = false
    var gauss    = 0.0
}

func mtSeed(_ mt: inout Mt19937, _ seed: UInt32) {
    mt.key[0] = seed
    for i in 1..<624 {
        let prev = mt.key[i - 1]
        mt.key[i] = 1812433253 &* (prev ^ (prev >> 30)) &+ UInt32(i)
    }
    mt.pos      = 624
    mt.hasGauss = false
    mt.gauss    = 0.0
}

func mtTwist(_ mt: inout Mt19937) {
    for i in 0..<624 {
        let y = (mt.key[i] & 0x80000000) |
                (mt.key[(i + 1) % 624] & 0x7fffffff)
        let mag: UInt32 = (y & 1) != 0 ? 0x9908b0df : 0
        mt.key[i] = mt.key[(i + 397) % 624] ^ (y >> 1) ^ mag
    }
    mt.pos = 0
}

func mtNext(_ mt: inout Mt19937) -> UInt32 {
    if mt.pos >= 624 { mtTwist(&mt) }
    var y = mt.key[mt.pos]
    mt.pos += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >> 18
    return y
}

func mtDouble(_ mt: inout Mt19937) -> Double {
    let a = mtNext(&mt) >> 5
    let b = mtNext(&mt) >> 6
    return (Double(a) * 67108864.0 + Double(b)) / 9007199254740992.0
}

func mtGauss(_ mt: inout Mt19937) -> Double {
    var result = mt.gauss
    if mt.hasGauss {
        mt.hasGauss = false
        mt.gauss    = 0.0
    } else {
        var x1 = 0.0
        var x2 = 0.0
        var r2 = 0.0
        repeat {
            x1 = 2.0 * mtDouble(&mt) - 1.0
            x2 = 2.0 * mtDouble(&mt) - 1.0
            r2 = x1 * x1 + x2 * x2
        } while r2 >= 1.0 || r2 == 0.0
        let f = sqrt(-2.0 * log(r2) / r2)
        mt.gauss    = f * x1
        mt.hasGauss = true
        result      = f * x2
    }
    return result
}

struct Scalars {
    var cp:    UnsafeMutablePointer<UInt32>
    var spare: UnsafeMutablePointer<UInt32>
    var count: Int
}

func textDecode(_ utf8: UnsafePointer<UInt8>,
                _ cp: UnsafeMutablePointer<UInt32>) -> Int {
    var s     = utf8
    var count = 0
    while s[0] != 0 {
        let extra = s[0] < 0x80 ? 0 : s[0] < 0xE0 ? 1 : s[0] < 0xF0 ? 2 : 3
        var value = extra == 0 ? UInt32(s[0]) :
                                 UInt32(s[0]) & (0x3F >> extra)
        var i     = 0
        s += 1
        while i < extra && (s[0] & 0xC0) == 0x80 {
            value = value << 6 | UInt32(s[0] & 0x3F)
            s += 1
            i += 1
        }
        cp[count] = value
        count += 1
    }
    return count
}

func textDecompose(_ raw: UnsafePointer<UInt32>, _ count: Int,
                   _ offsets: UnsafePointer<Int32>,
                   _ parts: UnsafePointer<Int32>,
                   _ cp: UnsafeMutablePointer<UInt32>?) -> Int {
    var made = 0
    for i in 0..<count {
        let basic = raw[i] <= 0xFFFF
        let from  = basic ? Int(offsets[Int(raw[i])]) : 0
        let size  = basic ? Int(offsets[Int(raw[i]) + 1]) - from : 0
        if let cp = cp {
            if size == 0 { cp[made] = raw[i] }
            for k in 0..<size {
                cp[made + k] = UInt32(bitPattern: parts[from + k])
            }
        }
        made += size > 0 ? size : 1
    }
    return made
}

func textClass(_ classes: UnsafePointer<Int32>, _ cp: UInt32) -> Int32 {
    return cp <= 0xFFFF ? classes[Int(cp)] : 0
}

func textReorder(_ t: inout Scalars, _ classes: UnsafePointer<Int32>) {
    for i in stride(from: 1, to: t.count, by: 1) {
        let cp   = t.cp[i]
        let rank = textClass(classes, cp)
        var j    = i
        while j > 0 && rank > 0 && textClass(classes, t.cp[j - 1]) > rank {
            t.cp[j] = t.cp[j - 1]
            j -= 1
        }
        t.cp[j] = cp
    }
}

func textAmong(_ cp: UInt32, _ set: [Unicode.Scalar]) -> Bool {
    var i = 0
    while i < set.count && set[i].value != cp { i += 1 }
    return i < set.count
}

func textIsEmoji(_ cp: UInt32) -> Bool {
    return (0x1F300 <= cp && cp <= 0x1F64F) ||
           (0x1F680 <= cp && cp <= 0x1F8FF) ||
           (0x1F900 <= cp && cp <= 0x1FAFF) ||
           (0x1F1E6 <= cp && cp <= 0x1F1FF) ||
           (0x2600 <= cp && cp <= 0x27BF)
}

func textIsSpace(_ cp: UInt32) -> Bool {
    return (0x09 <= cp && cp <= 0x0D) || (0x1C <= cp && cp <= 0x20) ||
           (0x2000 <= cp && cp <= 0x200A) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000
}

func textSpaces(_ cp: UnsafePointer<UInt32>, _ left: Int) -> Int {
    var i = 0
    while i < left && textIsSpace(cp[i]) { i += 1 }
    return i
}

func textSymbol(_ cp: UInt32) -> UInt32 {
    let map: [(Unicode.Scalar, Unicode.Scalar)] = [
        ("\u{2013}", "-"),  ("\u{2011}", "-"),  ("\u{2014}", "-"),
        ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{2018}", "'"),
        ("\u{2019}", "'"),  ("\u{00B4}", "'"),  ("`",        "'"),
        ("\u{00AF}", " "),  ("_",        " "),  ("[",        " "),
        ("]",        " "),  ("|",        " "),  ("/",        " "),
        ("#",        " "),  ("\u{2192}", " "),  ("\u{2190}", " ")]
    var i = 0
    while i < map.count && map[i].0.value != cp { i += 1 }
    return i < map.count ? map[i].1.value : cp
}

func textClean(_ t: inout Scalars) {
    var kept = 0
    for i in 0..<t.count {
        let cp = textSymbol(t.cp[i])
        if !textIsEmoji(cp) && cp != 0xA9 && cp != ascii("\\") {
            t.cp[kept] = cp
            kept += 1
        }
    }
    t.count = kept
}

func textStarts(_ cp: UnsafePointer<UInt32>, _ left: Int,
                _ prefix: [UInt8]) -> Bool {
    var i = 0
    while i < prefix.count && i < left && cp[i] == UInt32(prefix[i]) {
        i += 1
    }
    return i == prefix.count
}

func textEnds(_ cp: UnsafePointer<UInt32>, _ count: Int,
              _ suffix: [UInt8]) -> Bool {
    let size = suffix.count
    return count >= size && textStarts(cp + count - size, size, suffix)
}

func textReplace(_ t: inout Scalars, _ from: String, _ to: String) {
    let prefix = Array(from.utf8)
    let skip   = prefix.count
    var made   = 0
    var i      = 0
    while i < t.count {
        if textStarts(t.cp + i, t.count - i, prefix) {
            for byte in to.utf8 {
                t.spare[made] = UInt32(byte)
                made += 1
            }
            i += skip
        } else {
            t.spare[made] = t.cp[i]
            made += 1
            i += 1
        }
    }
    let swapped = t.cp
    t.cp    = t.spare
    t.spare = swapped
    t.count = made
}

func textRewrite(_ t: inout Scalars) {
    let pairs = [
        ("@", " at "), ("e.g.,", "for example, "),
        ("i.e.,", "that is, "), (" ,", ","), (" .", "."),
        (" !", "!"), (" ?", "?"), (" ;", ";"), (" :", ":"),
        (" '", "'")]
    for i in 0..<pairs.count {
        textReplace(&t, pairs[i].0, pairs[i].1)
    }
}

func textSqueeze(_ t: inout Scalars) {
    var kept     = 0
    var previous = UInt32(0)
    for i in 0..<t.count {
        let cp       = t.cp[i]
        let repeated = (cp == ascii("\"") || cp == ascii("'")) &&
                       cp == previous
        if !textIsSpace(cp) {
            if !repeated {
                t.cp[kept] = cp
                kept += 1
            }
        } else if kept > 0 && t.cp[kept - 1] != ascii(" ") {
            t.cp[kept] = ascii(" ")
            kept += 1
        }
        previous = cp
    }
    if kept > 0 && t.cp[kept - 1] == ascii(" ") { kept -= 1 }
    t.count = kept
}

func textEndSentence(_ t: inout Scalars) {
    let ending: [Unicode.Scalar] = [
        ".", "!", "?", ";", ":", ",", "'", "\"", ")", "]", "}", "\u{2026}",
        "\u{3002}", "\u{300D}", "\u{300F}", "\u{3011}", "\u{3009}",
        "\u{300B}", "\u{203A}", "\u{00BB}"]
    if t.count == 0 || !textAmong(t.cp[t.count - 1], ending) {
        t.cp[t.count] = ascii(".")
        t.count += 1
    }
}

func textWrap(_ t: inout Scalars, _ lang: String) {
    let open  = Array("<\(lang)>".utf8)
    let close = Array("</\(lang)>".utf8)
    let head  = open.count
    let tail  = close.count
    memmove(t.cp + head, t.cp, t.count * MemoryLayout<UInt32>.size)
    for i in 0..<head { t.cp[i] = UInt32(open[i]) }
    for i in 0..<tail { t.cp[head + t.count + i] = UInt32(close[i]) }
    t.count += head + tail
}

func textKnown(_ t: inout Scalars, _ indexer: UnsafePointer<Int32>) {
    for i in 0..<t.count {
        let cp    = t.cp[i]
        let known = cp <= 0xFFFF && indexer[Int(cp)] >= 0
        if !known && !textIsSpace(cp) { t.cp[i] = ascii(" ") }
    }
}

func textLookup(_ t: Scalars, _ indexer: UnsafePointer<Int32>,
                _ ids: UnsafeMutablePointer<Int32>) {
    for i in 0..<t.count {
        ids[i] = indexer[Int(t.cp[i])]
        precondition(ids[i] >= 0)
    }
}

func packRecords(_ map: UnsafeMutableRawPointer, _ bytes: Int,
                 _ header: Int) -> Pack {
    let index   = UnsafeRawPointer(map) + 8 + header
    var pack    = Pack(map: map, bytes: bytes)
    let count   = Int(index.load(fromByteOffset: 8, as: Int64.self))
    let records = index + 16
    pack.tensors.reserveCapacity(count)
    for i in 0..<count {
        let record = records + i * Record.size
        let shape  = (record + Record.shape)
                         .assumingMemoryBound(to: Int32.self)
        let offset = record.load(fromByteOffset: Record.offset,
                                 as: Int64.self)
        pack.tensors.append(Tensor(
            name: record.assumingMemoryBound(to: CChar.self),
            data: (map + Int(offset)).assumingMemoryBound(to: Float.self),
            count: Int(record.load(fromByteOffset: Record.count,
                                   as: Int64.self)),
            shape: (Int(shape[0]), Int(shape[1]), Int(shape[2]),
                    Int(shape[3])),
            bits: Int(record.load(fromByteOffset: Record.bits,
                                  as: Int32.self))))
    }
    return pack
}

func packOpen(_ path: String) -> Pack? {
    let file   = open(path, O_RDONLY)
    var status = stat()
    var pack: Pack? = nil
    if file >= 0 && fstat(file, &status) == 0 {
        let bytes = Int(status.st_size)
        let map: UnsafeMutableRawPointer = mmap(nil, bytes, PROT_READ,
                                                MAP_PRIVATE, file, 0)
        let mapped = map != MAP_FAILED && bytes >= 24
        let header = mapped ? map.load(as: Int.self) : -1
        if header >= 0 && header <= bytes - 24 &&
           memcmp(map + 8 + header, "SUPERTON", 8) == 0 {
            pack = packRecords(map, bytes, header)
        } else if map != MAP_FAILED {
            munmap(map, bytes)
        }
    }
    if file >= 0 { close(file) }
    return pack
}

func packFind(_ pack: Pack, _ name: String) -> Tensor? {
    var low  = 0
    var high = pack.tensors.count
    while low < high {
        let middle = (low + high) / 2
        if strcmp(pack.tensors[middle].name, name) < 0 {
            low = middle + 1
        } else {
            high = middle
        }
    }
    let found = low < pack.tensors.count &&
                strcmp(pack.tensors[low].name, name) == 0
    return found ? pack.tensors[low] : nil
}

func has(_ pack: Pack, _ name: String) -> Bool {
    return packFind(pack, name) != nil
}

func weight(_ pack: Pack, _ name: String) -> Tensor {
    if let tensor = packFind(pack, name) {
        return tensor
    } else {
        fatalError("no tensor named \(name)")
    }
}

func textIds(_ raw: UnsafePointer<UInt32>, _ count: Int, _ lang: String,
             _ w: Pack, _ ids: UnsafeMutablePointer<Int32>,
             _ capacity: Int) -> Int {
    let offsets = cast(weight(w, "nfkd.offsets").data, Int32.self)
    let parts   = cast(weight(w, "nfkd.data").data, Int32.self)
    let indexer = cast(weight(w, "indexer").data, Int32.self)
    let wrap    = 2 * lang.utf8.count + 5
    let room    = 4 * textDecompose(raw, count, offsets, parts, nil) +
                  wrap + 3
    var t       = Scalars(cp: .allocate(capacity: room),
                          spare: .allocate(capacity: room), count: 0)
    precondition(lang.utf8.count < 12)
    t.count = textDecompose(raw, count, offsets, parts, t.cp)
    textReorder(&t, cast(weight(w, "nfkd.class").data, Int32.self))
    textClean(&t)
    textRewrite(&t)
    textKnown(&t, indexer)
    textSqueeze(&t)
    if t.count > 0 {
        textEndSentence(&t)
        t.count = min(t.count, capacity - wrap)
        textWrap(&t, lang)
        textLookup(t, indexer, ids)
    }
    t.cp.deallocate()
    t.spare.deallocate()
    return t.count
}

struct Chunks {
    var raw:    UnsafeMutablePointer<UInt32>
    var cp:     UnsafeMutablePointer<UInt32>
    var edges:  Tensor
    var count:  Int
    var limit:  Int
    var at:     Int
    var length: Int
}

func chunkOpen(_ w: Pack, _ text: String, _ limit: Int) -> Chunks {
    let room = text.utf8.count + 1
    var c    = Chunks(raw: .allocate(capacity: room),
                      cp: .allocate(capacity: room),
                      edges: weight(w, "word.edges"), count: 0,
                      limit: limit, at: 0, length: 0)
    c.count = textDecode(text, c.raw)
    c.at    = textSpaces(c.raw, c.count)
    return c
}

func chunkIsWord(_ c: Chunks, _ cp: UInt32) -> Bool {
    let edges = cast(c.edges.data, UInt32.self)
    var i     = 0
    while i < c.edges.count && edges[i] <= cp { i += 1 }
    return i % 2 == 1
}

func chunkAbbreviated(_ c: Chunks, _ at: Int) -> Bool {
    let known = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ph.D.", "etc.",
        "e.g.", "i.e.", "vs.", "Inc.", "Ltd.", "Co.", "Corp.", "St.", "Ave.",
        "Blvd."]
    let cp = UnsafePointer(c.raw)
    var i  = 0
    while i < known.count && !textEnds(cp, at, Array(known[i].utf8)) {
        i += 1
    }
    let initial = at >= 2 && cp[at - 1] == ascii(".") &&
                  ascii("A") <= cp[at - 2] && cp[at - 2] <= ascii("Z") &&
                  (at == 2 || !chunkIsWord(c, cp[at - 3]))
    return i < known.count || initial
}

func chunkParagraph(_ c: Chunks, _ from: Int, _ to: Int) -> Bool {
    var lines = 0
    for i in stride(from: from, to: to, by: 1) {
        lines += c.raw[i] == ascii("\n") ? 1 : 0
    }
    return lines >= 2
}

func chunkSentence(_ c: Chunks, _ at: Int) -> Int {
    let stops: [Unicode.Scalar] = [".", "!", "?"]
    let cp   = UnsafePointer(c.raw)
    var end  = at
    var next = at
    repeat {
        end = next
        while end < c.count && !textIsSpace(cp[end]) { end += 1 }
        next = end + textSpaces(cp + end, c.count - end)
    } while next < c.count && !chunkParagraph(c, end, next) &&
            (!textAmong(cp[end - 1], stops) || chunkAbbreviated(c, end))
    return end
}

func chunkCut(_ c: Chunks, _ at: Int) -> Int {
    var size = c.limit
    while size > c.limit / 2 && !textIsSpace(c.raw[at + size]) { size -= 1 }
    return textIsSpace(c.raw[at + size]) ? size : c.limit
}

func chunkNext(_ c: inout Chunks) -> Bool {
    var from = c.at
    var end  = chunkSentence(c, c.at)
    c.length = 0
    while c.at < c.count &&
          (c.length == 0 || (!chunkParagraph(c, from, c.at) &&
                             c.length + end - c.at + 1 <= c.limit)) {
        let whole = end - c.at
        let size  = c.length == 0 && whole > c.limit ? chunkCut(c, c.at) :
                                                       whole
        if c.length > 0 {
            c.cp[c.length] = ascii(" ")
            c.length += 1
        }
        memcpy(c.cp + c.length, c.raw + c.at,
               size * MemoryLayout<UInt32>.size)
        c.length += size
        from = c.at + size
        c.at = from + textSpaces(c.raw + from, c.count - from)
        end  = chunkSentence(c, c.at)
    }
    return c.length > 0
}

func floats(_ tts: Engine, _ count: Int) -> UnsafeMutablePointer<Float> {
    tts.used = (tts.used + arenaAlign - 1) / arenaAlign * arenaAlign
    precondition(tts.used + count <= arenaFloats, "the arena is full")
    let data = tts.arena + tts.used
    tts.used += count
    return data
}

func fill(_ y: UnsafeMutablePointer<Float>, _ value: Float, _ n: Int) {
    for i in 0..<n { y[i] = value }
}

func axpy(_ y: UnsafeMutablePointer<Float>, _ a: Float,
          _ x: UnsafePointer<Float>, _ n: Int) {
    for i in 0..<n { y[i] += a * x[i] }
}

func dot(_ a: UnsafePointer<Float>, _ aStride: Int,
         _ b: UnsafePointer<Float>, _ bStride: Int, _ n: Int) -> Float {
    var sum: Float = 0.0
    for i in 0..<n {
        sum += a[i &* aStride] * b[i &* bStride]
    }
    return sum
}

func matmul(_ y: UnsafeMutablePointer<Float>, _ x: UnsafePointer<Float>,
            _ n: Int, _ w: UnsafePointer<Float>, _ co: Int, _ ci: Int,
            _ transposed: Bool) {
    cblas_sgemm(CblasRowMajor, transposed ? CblasTrans : CblasNoTrans,
                CblasNoTrans, Int32(co), Int32(n), Int32(ci), 1.0, w,
                Int32(transposed ? co : ci), x, Int32(n), 1.0, y, Int32(n))
}

func geluErf(_ e: UnsafeMutablePointer<Float>, _ x: UnsafePointer<Float>,
             _ count: Int) {
    var size = Int32(count)
    for i in 0..<count {
        let z = x[i] / 1.4142135
        e[i] = -z * z
    }
    vvexpf(e, e, &size)
    for i in 0..<count {
        let t: Float = 1.0 / (1.0 + 0.3275911 * fabsf(x[i] / 1.4142135))
        let tail: Float = ((((1.061405429 * t - 1.453152027) * t +
                             1.421413741) * t - 0.284496736) * t +
                           0.254829592) * t * e[i]
        e[i] = copysignf(1.0 - tail, x[i])
    }
}

func unpacked(_ wide: UnsafeMutablePointer<Float>,
              _ slice: UnsafePointer<UInt8>, _ blocks: Int) {
    let codes = slice + (blocks + q4Group - 1) / q4Group * q4Head
    for i in 0..<blocks * (q4Block / 2) {
        wide[2 &* i]      = Float(codes[i] & 15)
        wide[2 &* i &+ 1] = Float(codes[i] >> 4)
    }
    for b in 0..<blocks {
        let head  = slice + b / q4Group * q4Head
        let unit  = UnsafeRawPointer(head).assumingMemoryBound(
            to: Float16.self)
        let scale = Float(unit[0]) * Float(head[4 + b % q4Group])
        let least = Float(unit[1]) * Float(head[12 + b % q4Group])
        let out   = wide + b * q4Block
        for i in 0..<q4Block {
            out[i] = scale * out[i] - least
        }
    }
}

func widened(_ tts: Engine, _ w: Tensor, _ from: Int,
             _ size: Int) -> UnsafePointer<Float> {
    let slice  = w.count / w.shape.0
    var result = w.data + from * slice
    if w.bits == 16 {
        let half = cast(w.data, Float16.self)
        let wide = floats(tts, size * slice)
        for i in 0..<size * slice {
            wide[i] = Float(half[from &* slice &+ i])
        }
        result = UnsafePointer(wide)
    } else if w.bits == 8 {
        let codes  = cast(w.data, Int8.self)
        let scales = weight(tts.weights,
                            String(cString: w.name) + ".scale").data
        let wide   = floats(tts, size * slice)
        for row in 0..<size {
            let scale = scales[from + row]
            let code  = codes + (from + row) * slice
            for i in 0..<slice {
                wide[row &* slice &+ i] = scale * Float(code[i])
            }
        }
        result = UnsafePointer(wide)
    } else if w.bits == 4 {
        let blocks = (slice + q4Block - 1) / q4Block
        let bytes  = (blocks + q4Group - 1) / q4Group * q4Head +
                     blocks * (q4Block / 2)
        let packed = cast(w.data, UInt8.self)
        let wide   = floats(tts, size * slice + q4Block)
        for row in 0..<size {
            unpacked(wide + row * slice, packed + (from + row) * bytes,
                     blocks)
        }
        result = UnsafePointer(wide)
    }
    return result
}

func product(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
             _ x: UnsafePointer<Float>, _ n: Int, _ w: Tensor,
             _ bias: Tensor?, _ transposed: Bool) {
    let rows  = w.shape.0
    let slice = w.count / rows
    let co    = transposed ? slice : rows
    let fit   = widenFloats / slice / 4 * 4
    let block = w.bits == 0 ? rows : fit < 4 ? 4 : fit
    let mark  = tts.used
    for o in 0..<co {
        fill(y + o * n, bias?.data[o] ?? 0.0, n)
    }
    for from in stride(from: 0, to: rows, by: block) {
        let size = rows - from < block ? rows - from : block
        let part = widened(tts, w, from, size)
        if transposed {
            matmul(y, x + from * n, n, part, co, size, true)
        } else {
            matmul(y + from * n, x, n, part, size, slice, false)
        }
        tts.used = mark
    }
}

func unfolded(_ tts: Engine, _ x: UnsafePointer<Float>, _ ci: Int, _ n: Int,
              _ k: Int, _ dilation: Int,
              _ causal: Bool) -> UnsafePointer<Float> {
    let span = dilation * (k - 1)
    let left = causal ? span : span / 2
    let taps = floats(tts, ci * k * n)
    for row in 0..<ci * k {
        let channel = x + row / k * n
        let shift   = row % k * dilation - left
        let head    = shift > 0 ? 0 : -shift < n ? -shift : n
        let tail    = shift < 0 ? 0 : shift < n ? shift : n
        let out     = taps + row * n
        fill(out, channel[0], head)
        for t in head..<n - tail {
            out[t] = channel[t &+ shift]
        }
        fill(out + n - tail, channel[n - 1], tail)
    }
    return UnsafePointer(taps)
}

func conv(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
          _ x: UnsafePointer<Float>, _ ci: Int, _ n: Int, _ w: Tensor,
          _ bias: Tensor?, _ dilation: Int, _ causal: Bool) {
    let co        = w.shape.0
    let k         = w.shape.2
    let depthwise = w.shape.1 == 1 && ci > 1
    let mark      = tts.used
    let taps      = k == 1 ? x : unfolded(tts, x, ci, n, k, dilation,
                                          causal)
    if depthwise {
        for i in 0..<co * k {
            let row = y + i / k * n
            if i % k == 0 { fill(row, bias?.data[i / k] ?? 0.0, n) }
            axpy(row, w.data[i], taps + i * n, n)
        }
    } else {
        product(tts, y, taps, n, w, bias, false)
    }
    tts.used = mark
}

func project(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
             _ x: UnsafePointer<Float>, _ n: Int, _ w: Tensor,
             _ bias: Tensor?) {
    product(tts, y, x, n, w, bias, true)
}

func layerNorm(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
               _ n: Int, _ gain: Tensor, _ bias: Tensor) {
    let mark  = tts.used
    let mean  = floats(tts, n)
    let scale = floats(tts, n)
    fill(mean, 0.0, n)
    fill(scale, 0.0, n)
    for i in 0..<c { axpy(mean, 1.0, x + i * n, n) }
    for t in 0..<n { mean[t] /= Float(c) }
    for i in 0..<c {
        for t in 0..<n {
            let d = x[i &* n &+ t] - mean[t]
            scale[t] += d * d
        }
    }
    for t in 0..<n {
        scale[t] = 1.0 / sqrtf(scale[t] / Float(c) + layerNormEpsilon)
    }
    for i in 0..<c {
        let row = x + i * n
        for t in 0..<n {
            row[t] = (row[t] - mean[t]) * scale[t] * gain.data[i] +
                     bias.data[i]
        }
    }
    tts.used = mark
}

func gelu(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
          _ n: Int) {
    let mark = tts.used
    let e    = floats(tts, c * n)
    geluErf(e, x, c * n)
    for i in 0..<c * n {
        x[i] = x[i] * (e[i] + 1.0) * 0.5
    }
    tts.used = mark
}

func prelu(_ x: UnsafeMutablePointer<Float>, _ n: Int, _ slope: Float) {
    for i in 0..<n {
        x[i] = x[i] < 0.0 ? x[i] * slope : x[i]
    }
}

func softmax(_ x: UnsafeMutablePointer<Float>, _ n: Int) {
    var top = x[0]
    var sum: Float = 0.0
    for i in 1..<n { top = fmaxf(top, x[i]) }
    for i in 0..<n {
        x[i] = expf(x[i] - top)
        sum += x[i]
    }
    for i in 0..<n { x[i] /= sum }
}

func transposed(_ tts: Engine, _ rows: UnsafePointer<Float>, _ m: Int,
                _ c: Int) -> UnsafeMutablePointer<Float> {
    let columns = floats(tts, m * c)
    for i in 0..<m {
        for j in 0..<c {
            columns[j &* m &+ i] = rows[i &* c &+ j]
        }
    }
    return columns
}

func convnext(_ tts: Engine, _ x: UnsafePointer<Float>, _ c: Int, _ n: Int,
              _ dilation: Int, _ causal: Bool,
              _ at: String) -> UnsafeMutablePointer<Float> {
    let w     = tts.weights
    let dw    = causal ? "dwconv.net" : "dwconv"
    let w1    = weight(w, "\(at).pwconv1.weight")
    let gamma = weight(w, "\(at).gamma")
    let wide  = w1.shape.0
    let y     = floats(tts, c * n)
    let mark  = tts.used
    let h     = floats(tts, c * n)
    let z     = floats(tts, wide * n)
    conv(tts, h, x, c, n, weight(w, "\(at).\(dw).weight"),
         weight(w, "\(at).\(dw).bias"), dilation, causal)
    layerNorm(tts, h, c, n, weight(w, "\(at).norm.norm.weight"),
              weight(w, "\(at).norm.norm.bias"))
    conv(tts, z, h, c, n, w1, weight(w, "\(at).pwconv1.bias"), 1, false)
    gelu(tts, z, wide, n)
    conv(tts, y, z, wide, n, weight(w, "\(at).pwconv2.weight"),
         weight(w, "\(at).pwconv2.bias"), 1, false)
    for i in 0..<c * n {
        y[i] = x[i] + gamma.data[i / n] * y[i]
    }
    tts.used = mark
    return y
}

func stack(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
           _ n: Int, _ dilations: [Int]?, _ causal: Bool,
           _ at: String) -> UnsafeMutablePointer<Float> {
    var x = x
    var i = 0
    while has(tts.weights, "\(at).convnext.\(i).gamma") {
        let layer = "\(at).convnext.\(i)"
        x = convnext(tts, x, c, n, dilations?[i] ?? 1, causal, layer)
        i += 1
    }
    return x
}

func relativeRow(_ o: UnsafeMutablePointer<Float>,
                 _ p: UnsafeMutablePointer<Float>,
                 _ q: UnsafePointer<Float>, _ k: UnsafePointer<Float>,
                 _ v: UnsafePointer<Float>, _ relK: Tensor,
                 _ relV: Tensor, _ n: Int, _ i: Int) {
    let d      = relK.shape.2
    let window = relK.shape.1 / 2
    for j in 0..<n {
        let r = j - i + window
        p[j] = dot(q + i, n, k + j, n, d)
        if r >= 0 && r <= 2 * window {
            p[j] += dot(q + i, n, relK.data + r * d, 1, d)
        }
    }
    softmax(p, n)
    for e in 0..<d {
        var relative: Float = 0.0
        for r in 0...2 * window {
            let j = i + r - window
            if j >= 0 && j < n {
                relative += p[j] * relV.data[r * d + e]
            }
        }
        o[e * n + i] = dot(p, 1, v + e * n, 1, n) + relative
    }
}

func relativeAttention(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
                       _ x: UnsafePointer<Float>, _ c: Int, _ n: Int,
                       _ at: String) {
    let w    = tts.weights
    let relK = weight(w, "\(at).emb_rel_k")
    let relV = weight(w, "\(at).emb_rel_v")
    let d    = relK.shape.2
    let size = c * n
    let mark = tts.used
    let q    = floats(tts, size)
    let k    = floats(tts, size)
    let v    = floats(tts, size)
    let o    = floats(tts, size)
    let p    = floats(tts, n)
    conv(tts, q, x, c, n, weight(w, "\(at).conv_q.weight"),
         weight(w, "\(at).conv_q.bias"), 1, false)
    conv(tts, k, x, c, n, weight(w, "\(at).conv_k.weight"),
         weight(w, "\(at).conv_k.bias"), 1, false)
    conv(tts, v, x, c, n, weight(w, "\(at).conv_v.weight"),
         weight(w, "\(at).conv_v.bias"), 1, false)
    for i in 0..<size { q[i] /= sqrtf(Float(d)) }
    for h in 0..<c / d {
        let base = h * d * n
        for i in 0..<n {
            relativeRow(o + base, p, q + base, k + base, v + base, relK,
                        relV, n, i)
        }
    }
    conv(tts, y, o, c, n, weight(w, "\(at).conv_o.weight"),
         weight(w, "\(at).conv_o.bias"), 1, false)
    tts.used = mark
}

func feedForward(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
                 _ x: UnsafePointer<Float>, _ c: Int, _ n: Int,
                 _ at: String) {
    let w    = tts.weights
    let w1   = weight(w, "\(at).conv_1.weight")
    let wide = w1.shape.0
    let mark = tts.used
    let h    = floats(tts, wide * n)
    conv(tts, h, x, c, n, w1, weight(w, "\(at).conv_1.bias"), 1, false)
    prelu(h, wide * n, 0.0)
    conv(tts, y, h, wide, n, weight(w, "\(at).conv_2.weight"),
         weight(w, "\(at).conv_2.bias"), 1, false)
    tts.used = mark
}

func encoder(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
             _ n: Int, _ dilations: [Int]?,
             _ at: String) -> UnsafeMutablePointer<Float> {
    let w    = tts.weights
    let size = c * n
    let x    = stack(tts, x, c, n, dilations, false, "\(at).convnext")
    let h    = floats(tts, size)
    let y    = floats(tts, size)
    var i    = 0
    memcpy(h, x, size * MemoryLayout<Float>.size)
    while has(w, "\(at).attn_encoder.attn_layers.\(i).emb_rel_k") {
        var name = "\(at).attn_encoder.attn_layers.\(i)"
        relativeAttention(tts, y, h, c, n, name)
        axpy(h, 1.0, y, size)
        name = "\(at).attn_encoder.norm_layers_1.\(i).norm"
        layerNorm(tts, h, c, n, weight(w, "\(name).weight"),
                  weight(w, "\(name).bias"))
        name = "\(at).attn_encoder.ffn_layers.\(i)"
        feedForward(tts, y, h, c, n, name)
        axpy(h, 1.0, y, size)
        name = "\(at).attn_encoder.norm_layers_2.\(i).norm"
        layerNorm(tts, h, c, n, weight(w, "\(name).weight"),
                  weight(w, "\(name).bias"))
        i += 1
    }
    axpy(h, 1.0, x, size)
    return h
}

func embed(_ y: UnsafeMutablePointer<Float>, _ stride: Int,
           _ table: Tensor, _ ids: UnsafePointer<Int32>, _ count: Int) {
    let c = table.shape.1
    for i in 0..<c {
        for t in 0..<count {
            y[i * stride + t] = table.data[Int(ids[t]) * c + i]
        }
    }
}

func duration(_ tts: Engine, _ ids: UnsafePointer<Int32>, _ count: Int,
              _ style: UnsafePointer<Float>) -> Float {
    let w      = tts.weights
    let token  = weight(w, DP + ".sentence_encoder.sentence_token")
    let table  = weight(w, DP + ".sentence_encoder.text_embedder" +
                           ".char_embedder.weight")
    let first  = weight(w, DP + ".predictor.layers.0.weight")
    let c      = table.shape.1
    let n      = count + 1
    let mark   = tts.used
    let x      = floats(tts, c * n)
    let input  = floats(tts, first.shape.1)
    let hidden = floats(tts, first.shape.0)
    var out: Float = 0.0
    for i in 0..<c { x[i * n] = token.data[i] }
    embed(x + 1, n, table, ids, count)
    let h = encoder(tts, x, c, n, nil, DP + ".sentence_encoder")
    for i in 0..<c { hidden[i] = h[i * n] }
    conv(tts, input, hidden, c, 1,
         weight(w, DP + ".sentence_encoder.proj_out.net.weight"), nil, 1,
         false)
    memcpy(input + c, style,
           (first.shape.1 - c) * MemoryLayout<Float>.size)
    conv(tts, hidden, input, first.shape.1, 1, first,
         weight(w, DP + ".predictor.layers.0.bias"), 1, false)
    prelu(hidden, first.shape.0,
          weight(w, DP + ".predictor.activation.weight").data[0])
    conv(tts, &out, hidden, first.shape.0, 1,
         weight(w, DP + ".predictor.layers.1.weight"),
         weight(w, DP + ".predictor.layers.1.bias"), 1, false)
    tts.used = mark
    return expf(out)
}

func attend(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
            _ q: UnsafePointer<Float>, _ n: Int, _ k: UnsafePointer<Float>,
            _ v: UnsafePointer<Float>, _ m: Int, _ c: Int, _ heads: Int,
            _ scale: Float) {
    let d       = c / heads
    let mark    = tts.used
    let p       = floats(tts, m)
    let o       = floats(tts, d)
    let columns = transposed(tts, v, c, m)
    for h in 0..<heads {
        let base = h * d
        for i in 0..<n {
            fill(p, 0.0, m)
            fill(o, 0.0, d)
            for e in 0..<d {
                axpy(p, q[(base + e) * n + i], k + (base + e) * m, m)
            }
            for j in 0..<m { p[j] /= scale }
            softmax(p, m)
            for j in 0..<m {
                axpy(o, p[j], columns + j * c + base, d)
            }
            for e in 0..<d { y[(base + e) * n + i] = o[e] }
        }
    }
    tts.used = mark
}

func styleAttention(_ tts: Engine, _ y: UnsafeMutablePointer<Float>,
                    _ x: UnsafePointer<Float>, _ n: Int,
                    _ condition: Condition, _ at: String) {
    let w    = tts.weights
    let wq   = weight(w, "\(at).W_query.linear.weight")
    let wk   = weight(w, "\(at).W_key.linear.weight")
    let c    = wq.shape.1
    let m    = condition.tokens
    let mark = tts.used
    let q    = floats(tts, c * n)
    let k    = floats(tts, c * m)
    let v    = floats(tts, c * m)
    let o    = floats(tts, c * n)
    project(tts, q, x, n, wq, weight(w, "\(at).W_query.linear.bias"))
    project(tts, k, condition.keys, m, wk,
            weight(w, "\(at).W_key.linear.bias"))
    project(tts, v, condition.values, m,
            weight(w, "\(at).W_value.linear.weight"),
            weight(w, "\(at).W_value.linear.bias"))
    for i in 0..<c * m { k[i] = tanhf(k[i]) }
    attend(tts, o, q, n, k, v, m, c, styleHeads,
           sqrtf(Float(wk.shape.0)))
    project(tts, y, o, n, weight(w, "\(at).out_fc.linear.weight"),
            weight(w, "\(at).out_fc.linear.bias"))
    tts.used = mark
}

func textEncoder(_ tts: Engine, _ ids: UnsafePointer<Int32>, _ n: Int,
                 _ condition: Condition) -> UnsafeMutablePointer<Float> {
    let w     = tts.weights
    let table = weight(w, TE + ".text_encoder.text_embedder" +
                          ".char_embedder.weight")
    let c     = table.shape.1
    let size  = c * n
    let emb   = floats(tts, size)
    let mark  = tts.used
    let x     = floats(tts, size)
    let y     = floats(tts, size)
    embed(x, n, table, ids, n)
    let e = encoder(tts, x, c, n, encoderDilations, TE + ".text_encoder")
    styleAttention(tts, y, e, n, condition,
                   TE + ".speech_prompted_text_encoder.attention1")
    axpy(y, 1.0, e, size)
    styleAttention(tts, emb, y, n, condition,
                   TE + ".speech_prompted_text_encoder.attention2")
    axpy(emb, 1.0, e, size)
    layerNorm(tts, emb, c, n,
              weight(w, TE + ".speech_prompted_text_encoder.norm.norm" +
                        ".weight"),
              weight(w, TE + ".speech_prompted_text_encoder.norm.norm" +
                        ".bias"))
    tts.used = mark
    return emb
}

func timeEmbedding(_ tts: Engine, _ out: UnsafeMutablePointer<Float>,
                   _ t: Float) {
    let w     = tts.weights
    let w0    = weight(w, FIELD + ".time_encoder.mlp.0.linear.weight")
    let half  = w0.shape.1 / 2
    let wide  = w0.shape.0
    let freqs = weight(w, "time.freqs").data
    let mark  = tts.used
    let waves = floats(tts, 2 * half)
    let h     = floats(tts, wide)
    for i in 0..<half {
        let wave = __sincosf_stret(t * timeScale * freqs[i])
        waves[i] = wave.__sinval
        waves[half + i] = wave.__cosval
    }
    conv(tts, h, waves, 2 * half, 1, w0,
         weight(w, FIELD + ".time_encoder.mlp.0.linear.bias"), 1, false)
    for i in 0..<wide {
        h[i] = h[i] * tanhf(log1pf(expf(h[i])))
    }
    conv(tts, out, h, wide, 1,
         weight(w, FIELD + ".time_encoder.mlp.2.linear.weight"),
         weight(w, FIELD + ".time_encoder.mlp.2.linear.bias"), 1, false)
    tts.used = mark
}

func rotate(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
            _ n: Int, _ theta: Tensor) {
    let half = theta.count
    for i in 0..<c / 2 * n {
        let pair  = i / n
        let t     = i % n
        let a     = x + (pair / half * 2 * half + pair % half) * n + t
        let b     = a + half * n
        let angle = Float(t) / Float(n) * theta.data[pair % half]
        let turn  = __sincosf_stret(angle)
        let real  = a.pointee * turn.__cosval - b.pointee * turn.__sinval
        b.pointee = a.pointee * turn.__sinval + b.pointee * turn.__cosval
        a.pointee = real
    }
}

func textCondition(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
                   _ n: Int, _ condition: Condition, _ at: String) {
    let w     = tts.weights
    let theta = weight(w, FIELD + ".main_blocks.3.attn.theta")
    let wk    = weight(w, "\(at).attn.W_key.linear.weight")
    let m     = condition.length
    let heads = c / (2 * theta.count)
    let mark  = tts.used
    let q     = floats(tts, c * n)
    let k     = floats(tts, c * m)
    let v     = floats(tts, c * m)
    let o     = floats(tts, c * n)
    project(tts, q, x, n, weight(w, "\(at).attn.W_query.linear.weight"),
            weight(w, "\(at).attn.W_query.linear.bias"))
    project(tts, k, condition.text, m, wk,
            weight(w, "\(at).attn.W_key.linear.bias"))
    project(tts, v, condition.text, m,
            weight(w, "\(at).attn.W_value.linear.weight"),
            weight(w, "\(at).attn.W_value.linear.bias"))
    rotate(tts, q, c, n, theta)
    rotate(tts, k, c, m, theta)
    attend(tts, o, q, n, k, v, m, c, heads, sqrtf(Float(wk.shape.0)))
    project(tts, q, o, n, weight(w, "\(at).attn.out_fc.linear.weight"),
            weight(w, "\(at).attn.out_fc.linear.bias"))
    axpy(x, 1.0, q, c * n)
    tts.used = mark
}

func fieldBlock(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
                _ n: Int, _ block: Int, _ time: UnsafePointer<Float>,
                _ condition: Condition) -> UnsafeMutablePointer<Float> {
    let w    = tts.weights
    let mark = tts.used
    let at   = "\(FIELD).main_blocks.\(block)"
    var y    = x
    if block % 2 == 0 {
        y = stack(tts, x, c, n, block % 6 == 0 ? fieldDilations : nil,
                  false, at)
    } else if block % 6 == 1 {
        let shift = floats(tts, c)
        project(tts, shift, time, 1,
                weight(w, "\(at).linear.linear.weight"),
                weight(w, "\(at).linear.linear.bias"))
        for i in 0..<c * n { x[i] += shift[i / n] }
        tts.used = mark
    } else {
        if block % 6 == 3 {
            textCondition(tts, x, c, n, condition, at)
        } else {
            let styled = floats(tts, c * n)
            styleAttention(tts, styled, x, n, condition, at + ".attention")
            axpy(x, 1.0, styled, c * n)
            tts.used = mark
        }
        layerNorm(tts, x, c, n, weight(w, "\(at).norm.norm.weight"),
                  weight(w, "\(at).norm.norm.bias"))
    }
    return y
}

func velocity(_ tts: Engine, _ latent: UnsafePointer<Float>, _ n: Int,
              _ time: UnsafePointer<Float>,
              _ condition: Condition) -> UnsafeMutablePointer<Float> {
    let w      = tts.weights
    let input  = weight(w, FIELD + ".proj_in.net.weight")
    let out    = weight(w, FIELD + ".proj_out.net.weight")
    let c      = input.shape.0
    let blocks = 6 * fieldGroups
    let v      = floats(tts, out.shape.0 * n)
    let mark   = tts.used
    var x      = floats(tts, c * n)
    conv(tts, x, latent, input.shape.1, n, input, nil, 1, false)
    for block in 0..<blocks {
        x = fieldBlock(tts, x, c, n, block, time, condition)
    }
    x = stack(tts, x, c, n, nil, false, FIELD + ".last_convnext")
    conv(tts, v, x, c, n, out, nil, 1, false)
    tts.used = mark
    return v
}

func flow(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ size: Int,
          _ n: Int, _ spoken: Condition, _ silent: Condition,
          _ steps: Int) {
    let out = weight(tts.weights,
                     FIELD + ".time_encoder.mlp.2.linear.weight")
    for step in 0..<steps {
        let mark = tts.used
        let time = floats(tts, out.shape.0)
        timeEmbedding(tts, time, Float(step) / Float(steps))
        let with    = velocity(tts, x, n, time, spoken)
        let without = velocity(tts, x, n, time, silent)
        for i in 0..<size {
            x[i] += 1.0 / Float(steps) * (with[i] * (1.0 + guidance) -
                                          without[i] * guidance)
        }
        tts.used = mark
    }
}

func vocoderHead(_ tts: Engine, _ x: UnsafeMutablePointer<Float>, _ c: Int,
                 _ n: Int) -> UnsafePointer<Float> {
    let w      = tts.weights
    let norm   = AE + ".decoder.final_norm.norm"
    let head   = weight(w, AE + ".decoder.head.layer1.net.weight")
    let last   = weight(w, AE + ".decoder.head.layer2.weight")
    let shift  = weight(w, norm + ".running_mean").data
    let spread = weight(w, norm + ".running_var").data
    let gain   = weight(w, norm + ".weight").data
    let bias   = weight(w, norm + ".bias").data
    let wide   = head.shape.0
    let y      = floats(tts, last.shape.0 * n)
    let mark   = tts.used
    let h      = floats(tts, wide * n)
    for i in 0..<c * n {
        let at = i / n
        x[i] = (x[i] - shift[at]) / sqrtf(spread[at] + batchNormEpsilon) *
               gain[at] + bias[at]
    }
    conv(tts, h, x, c, n, head,
         weight(w, AE + ".decoder.head.layer1.net.bias"), 1, true)
    prelu(h, wide * n, weight(w, AE + ".decoder.head.act.weight").data[0])
    conv(tts, y, h, wide, n, last, nil, 1, false)
    tts.used = mark
    return UnsafePointer(y)
}

func vocoderSpan(_ tts: Engine, _ wav: UnsafeMutablePointer<Float>,
                 _ latent: UnsafePointer<Float>, _ frames: Int,
                 _ from: Int, _ to: Int) {
    let w      = tts.weights
    let first  = weight(w, AE + ".decoder.embed.net.weight")
    let head   = weight(w, AE + ".decoder.head.layer1.net.weight")
    let last   = weight(w, AE + ".decoder.head.layer2.weight")
    let wing   = weight(w, AE + ".decoder.convnext.0.dwconv.net.weight")
    let mean   = weight(w, AE + ".latent_mean").data
    let std    = weight(w, AE + ".latent_std").data
    let scale  = weight(w, "vocoder.tts.ttl.normalizer.scale").data[0]
    let ld     = first.shape.1
    let c      = first.shape.0
    let hop    = last.shape.0
    let factor = weight(w, FIELD + ".proj_in.net.weight").shape.1 / ld
    var context = first.shape.2 - 1 + head.shape.2 - 1
    for i in 0..<vocoderDilations.count {
        context += vocoderDilations[i] * (wing.shape.2 - 1)
    }
    let start = from > context ? from - context : 0
    let n     = to - start
    let mark  = tts.used
    let z     = floats(tts, ld * n)
    var x     = floats(tts, c * n)
    for i in 0..<ld * n {
        let t = start + i % n
        z[i] = latent[(i / n * factor + t % factor) * frames + t / factor] /
               scale * std[i / n] + mean[i / n]
    }
    conv(tts, x, z, ld, n, first,
         weight(w, AE + ".decoder.embed.net.bias"), 1, true)
    x = stack(tts, x, c, n, vocoderDilations, true, AE + ".decoder")
    let y = vocoderHead(tts, x, c, n)
    for i in from * hop..<to * hop {
        wav[i] = y[i % hop * n + i / hop - start]
    }
    tts.used = mark
}

func vocoder(_ tts: Engine, _ latent: UnsafePointer<Float>, _ frames: Int,
             _ samples: inout Int) -> UnsafeMutablePointer<Float> {
    let w   = tts.weights
    let hop = weight(w, AE + ".decoder.head.layer2.weight").shape.0
    let n   = frames * weight(w, FIELD + ".proj_in.net.weight").shape.1 /
              weight(w, AE + ".decoder.embed.net.weight").shape.1
    let wav = floats(tts, n * hop)
    for from in stride(from: 0, to: n, by: vocoderSlice) {
        vocoderSpan(tts, wav, latent, frames, from,
                    n - from < vocoderSlice ? n : from + vocoderSlice)
    }
    samples = n * hop
    return wav
}

func voiceOpen(_ tts: Engine, _ name: String) -> Voice {
    let w   = tts.weights
    let ttl = weight(w, "voice.\(name).ttl")
    return Voice(ttl: transposed(tts, ttl.data, ttl.shape.0, ttl.shape.1),
                 dp: weight(w, "voice.\(name).dp").data,
                 tokens: ttl.shape.0)
}

func unconditional(_ tts: Engine, _ n: Int, _ m: Int, _ c: Int) -> Condition {
    let w       = tts.weights
    let blank   = weight(w, MASK + ".text_special_token")
    let nothing = floats(tts, blank.shape.1 * n)
    for i in 0..<blank.shape.1 * n {
        nothing[i] = blank.data[i / n]
    }
    return Condition(
        text: nothing, length: n,
        keys: transposed(
            tts, weight(w, MASK + ".style_key_special_token").data, m, c),
        values: transposed(
            tts, weight(w, MASK + ".style_value_special_token").data, m, c),
        tokens: m)
}

func synthesize(_ tts: Engine, _ text: UnsafePointer<UInt32>, _ count: Int,
                _ lang: String, _ voice: Voice, _ mt: inout Mt19937,
                _ steps: Int, _ speed: Float) -> Audio {
    let w     = tts.weights
    let keys  = weight(w, TE + ".style_encoder.style_token_layer.style_key")
    let rows  = weight(w, FIELD + ".proj_in.net.weight").shape.1
    let chunk = weight(w, AE + ".decoder.head.layer2.weight").shape.0 *
                rows / weight(w, AE + ".decoder.embed.net.weight").shape.1
    let m     = voice.tokens
    var ids   = [Int32](repeating: 0, count: textCapacity)
    let n     = textIds(text, count, lang, w, &ids, textCapacity)
    var audio = Audio(samples: tts.arena, count: 0, seconds: 0.0)
    if n > 0 {
        var spoken = Condition(
            text: voice.ttl, length: n,
            keys: transposed(tts, keys.data, m, keys.shape.2),
            values: voice.ttl, tokens: m)
        let silent  = unconditional(tts, n, m, keys.shape.2)
        let seconds = duration(tts, ids, n, voice.dp) / speed
        spoken.text = UnsafePointer(textEncoder(tts, ids, n, spoken))
        let wanted = Int((seconds * Float(sampleRate) + Float(chunk) - 1.0) /
                         Float(chunk))
        let frames = min(wanted, frameLimit)
        let x      = floats(tts, rows * frames)
        for i in 0..<rows * frames { x[i] = Float(mtGauss(&mt)) }
        flow(tts, x, rows * frames, frames, spoken, silent, steps)
        audio.samples = vocoder(tts, x, frames, &audio.count)
        audio.seconds = seconds
    }
    return audio
}

func pcm16(_ sample: Float) -> Int16 {
    let scaled = sample * 2147483648.0
    var value: Int16 = 0
    if Double(scaled) >= 2147483647.0 {
        value = Int16.max
    } else if scaled <= -2147483648.0 {
        value = Int16.min
    } else {
        value = Int16(truncatingIfNeeded: lrintf(scaled) >> 16)
    }
    return value
}

func engineOpen(_ path: String) -> Engine? {
    let bytes = arenaFloats * MemoryLayout<Float>.size
    var engine: Engine? = nil
    if let weights = packOpen(path) {
        let arena: UnsafeMutableRawPointer = mmap(
            nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1,
            0)
        if arena != MAP_FAILED {
            engine = Engine(weights, arena.bindMemory(to: Float.self,
                                                      capacity: arenaFloats))
        } else {
            munmap(weights.map, weights.bytes)
        }
    }
    return engine
}

func narrate(_ tts: Engine, _ text: String, _ lang: String, _ name: String,
             _ seed: UInt32, _ steps: Int, _ speed: Float) -> [Float] {
    let limit = lang == "ko" ? koreanChunkLimit : chunkLimit
    let pause = Int(chunkSilence * Double(sampleRate))
    let bytes = arenaFloats * MemoryLayout<Float>.size
    var whole = [Float]()
    var mt    = Mt19937()
    madvise(tts.arena, bytes, MADV_FREE_REUSE)
    tts.used = 0
    let voice = voiceOpen(tts, name)
    let mark  = tts.used
    var c     = chunkOpen(tts.weights, text, limit)
    mtSeed(&mt, seed)
    while chunkNext(&c) {
        let part = synthesize(tts, c.cp, c.length, lang, voice, &mt, steps,
                              speed)
        if part.count > 0 {
            if !whole.isEmpty {
                whole.append(contentsOf: repeatElement(0.0, count: pause))
            }
            whole.append(contentsOf: UnsafeBufferPointer(
                start: part.samples, count: part.count))
        }
        tts.used = mark
    }
    c.raw.deallocate()
    c.cp.deallocate()
    madvise(tts.arena, bytes, MADV_FREE_REUSABLE)
    return whole
}
