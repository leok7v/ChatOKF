// A PURE SYNCHRONOUS function of (text, voice, speed): a queue would make
// the output depend on when it was asked, and this side is gated byte for byte.
import Foundation

// The speed prior is part of the voice's identity: each was recorded at its
// own pace, so a caller's `speed` multiplies it rather than replacing it.
public struct SpeechVoice: Sendable, Hashable, Identifiable {
    public let name: String
    let tensor: String
    let speedPrior: Float

    public var id: String { name }
}

public final class Speech {

    public static let sampleRate = 24000

    public static let voices: [SpeechVoice] = [
        SpeechVoice(name: "Hugo",   tensor: "expr-voice-4-m", speedPrior: 0.9),
        SpeechVoice(name: "Luna",   tensor: "expr-voice-3-f", speedPrior: 0.8),
        SpeechVoice(name: "Kiki",   tensor: "expr-voice-5-f", speedPrior: 0.8),
        SpeechVoice(name: "Leo",    tensor: "expr-voice-5-m", speedPrior: 0.8),
        SpeechVoice(name: "Bella",  tensor: "expr-voice-2-f", speedPrior: 0.8),
        SpeechVoice(name: "Jasper", tensor: "expr-voice-2-m", speedPrior: 0.8),
        SpeechVoice(name: "Bruno",  tensor: "expr-voice-3-m", speedPrior: 0.8),
        SpeechVoice(name: "Rosie",  tensor: "expr-voice-4-f", speedPrior: 0.8),
    ]

    public static let defaultVoice = voices[2]

    // Case-insensitive, so a stale preference or a CLI name resolves.
    public static func voice(named name: String) -> SpeechVoice? {
        let wanted = name.lowercased()
        return voices.first { v in v.name.lowercased() == wanted }
    }

    private let tts: KittensCtx
    private let phonemizer: Phonemizer
    private let voicesPath: String
    private var voiceRows: [String: [Float]] = [:]
    private let lock = NSLock()

    static func bundled(_ name: String, _ ext: String?) -> URL? {
        Res.url(name, ext,
               dev: URL(fileURLWithPath: #filePath)
                   .deletingLastPathComponent())
    }

    public convenience init?() {
        let gguf = Speech.bundled("kitten_full", "gguf")
        let voices = Speech.bundled("voices", "safetensors")
        let rules = Speech.bundled("en_rules", nil)
        let list = Speech.bundled("en_list", nil)
        if let gguf, let voices, let rules, let list {
            self.init(ggufPath: gguf.path, voicesPath: voices.path,
                      rulesPath: rules.path, listPath: list.path)
        } else {
            return nil
        }
    }

    public init?(ggufPath: String, voicesPath: String,
                 rulesPath: String, listPath: String) {
        let phon = Phonemizer(rulesPath: rulesPath, listPath: listPath,
                              dialect: "en-us")
        let ctx = kittensCreate(ggufPath)
        if let phon, let ctx,
           FileManager.default.fileExists(atPath: voicesPath) {
            self.tts = ctx
            self.phonemizer = phon
            self.voicesPath = voicesPath
        } else {
            kittensDestroy(ctx)
            return nil
        }
    }

    deinit {
        kittensDestroy(tts)
    }

    // Serialized: the engine holds one scratch arena every stage resets, so two
    // runs cannot share an instance; real concurrency wants a second Speech.
    public func synthesize(_ text: String,
                           voice: SpeechVoice? = nil,
                           speed: Float = 1.0) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let picked = voice ?? Speech.defaultVoice
        var out = [Float]()
        if let rows = rowsFor(picked), speed > 0 {
            let ctx = SynthContext(tts: tts, phonemizer: phonemizer,
                                   voice: rows,
                                   speed: speed * picked.speedPrior)
            synthDocument(ctx, Array(text.utf8), &out)
        }
        return out
    }

    private func rowsFor(_ voice: SpeechVoice) -> [Float]? {
        var rows = voiceRows[voice.name]
        if rows == nil {
            rows = voiceEmbedding(voicesPath, voice.tensor)
            voiceRows[voice.name] = rows
        }
        return rows
    }
}

let ttsStyleDim  = 256
let ttsVoiceRows = 400
let ttsParaSilMS = 180   // silence before a new paragraph
let ttsSentSilMS = 80    // silence before a new sentence

// kittenSymbolCP[i] is the code point for id i; duplicates resolve last-wins.
let kittenSymbolCP: [UInt32] = [
    0x0024, 0x003b, 0x003a, 0x002c, 0x002e, 0x0021, 0x003f, 0x00a1,
    0x00bf, 0x2014, 0x2026, 0x0022, 0x00ab, 0x00bb, 0x0022, 0x0022,
    0x0020, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005a, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065,
    0x0066, 0x0067, 0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d,
    0x006e, 0x006f, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075,
    0x0076, 0x0077, 0x0078, 0x0079, 0x007a, 0x0251, 0x0250, 0x0252,
    0x00e6, 0x0253, 0x0299, 0x03b2, 0x0254, 0x0255, 0x00e7, 0x0257,
    0x0256, 0x00f0, 0x02a4, 0x0259, 0x0258, 0x025a, 0x025b, 0x025c,
    0x025d, 0x025e, 0x025f, 0x0284, 0x0261, 0x0260, 0x0262, 0x029b,
    0x0266, 0x0267, 0x0127, 0x0265, 0x029c, 0x0268, 0x026a, 0x029d,
    0x026d, 0x026c, 0x026b, 0x026e, 0x029f, 0x0271, 0x026f, 0x0270,
    0x014b, 0x0273, 0x0272, 0x0274, 0x00f8, 0x0275, 0x0278, 0x03b8,
    0x0153, 0x0276, 0x0298, 0x0279, 0x027a, 0x027e, 0x027b, 0x0280,
    0x0281, 0x027d, 0x0282, 0x0283, 0x0288, 0x02a7, 0x0289, 0x028a,
    0x028b, 0x2c71, 0x028c, 0x0263, 0x0264, 0x028d, 0x03c7, 0x028e,
    0x028f, 0x0291, 0x0290, 0x0292, 0x0294, 0x02a1, 0x0295, 0x02a2,
    0x01c0, 0x01c1, 0x01c2, 0x01c3, 0x02c8, 0x02cc, 0x02d0, 0x02d1,
    0x02bc, 0x02b4, 0x02b0, 0x02b1, 0x02b2, 0x02b7, 0x02e0, 0x02e4,
    0x02de, 0x2193, 0x2191, 0x2192, 0x2197, 0x2198, 0x0027, 0x0329,
    0x0027, 0x1d7b,
]

let kittenSymbolN = 178
let symMapSize    = 0x2300

func symbolMapBuild() -> [Int32] {
    var m = [Int32](repeating: -1, count: symMapSize)
    var i = 0
    while i < kittenSymbolN {
        let cp = Int(kittenSymbolCP[i])
        if cp < symMapSize { m[cp] = Int32(i) }
        i += 1
    }
    return m
}

let gSymID = symbolMapBuild()

func symbolID(_ cp: UInt32) -> Int {
    var id = -1
    if cp < UInt32(symMapSize) { id = Int(gSymID[Int(cp)]) }
    return id
}

struct TtsUtf8Decoded {
    var cp: UInt32 = 0
    var next = 0
    var ok = false
}

func ttsUtf8Decode(_ s: [UInt8], _ n: Int, _ i: Int) -> TtsUtf8Decoded {
    var r = TtsUtf8Decoded()
    let c0 = s[i]
    if c0 < 0x80 {
        r.cp = UInt32(c0); r.next = i + 1; r.ok = true
    } else if (c0 & 0xE0) == 0xC0 && i + 1 < n {
        r.cp = (UInt32(c0 & 0x1F) << 6) | UInt32(s[i + 1] & 0x3F)
        r.next = i + 2; r.ok = true
    } else if (c0 & 0xF0) == 0xE0 && i + 2 < n {
        r.cp = (UInt32(c0 & 0x0F) << 12)
             | (UInt32(s[i + 1] & 0x3F) << 6)
             | UInt32(s[i + 2] & 0x3F)
        r.next = i + 3; r.ok = true
    } else if (c0 & 0xF8) == 0xF0 && i + 3 < n {
        r.cp = (UInt32(c0 & 0x07) << 18)
             | (UInt32(s[i + 1] & 0x3F) << 12)
             | (UInt32(s[i + 2] & 0x3F) << 6)
             | UInt32(s[i + 3] & 0x3F)
        r.next = i + 4; r.ok = true
    } else {
        r.next = i + 1   // resync past a bad byte
    }
    return r
}

// [0, ids, term?, 10, 0]: 0 is pad, 10 the ellipsis marker the model wants as
// a terminator; `term` re-adds the punctuation the phonemizer dropped.
func sentenceToIDs(_ ipa: [UInt8], _ term: UInt8, _ cap: Int) -> [Int32] {
    var ids = [Int32]()
    ids.append(0)
    var i = 0
    let len = ipa.count
    var lastEmitted = -1
    while i < len && ids.count < cap - 3 {
        let d = ttsUtf8Decode(ipa, len, i)
        i = d.next
        if d.ok {
            let id = symbolID(d.cp)
            if id >= 0 {
                ids.append(Int32(id))
                lastEmitted = id
            }
        }
    }
    if term != 0 && ids.count < cap - 3 {
        let tid = symbolID(UInt32(term))
        if tid >= 0 && tid != lastEmitted { ids.append(Int32(tid)) }
    }
    ids.append(10)
    ids.append(0)
    return ids
}

// safetensors: u64 header length, JSON header, packed F32 [400 * 256].
func voiceEmbedding(_ path: String, _ tensorName: String) -> [Float]? {
    var rows: [Float]? = nil
    if let blob = FileManager.default.contents(atPath: path) {
        let hlen = blob.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        }
        let dataBase = 8 + Int(hlen)
        let json = blob.subdata(in: 8..<dataBase)
        let header = try? JSONSerialization.jsonObject(with: json)
        let entry = (header as? [String: Any])?[tensorName]
        let offsets = (entry as? [String: Any])?["data_offsets"]
        if let start = (offsets as? [Any])?.first as? Int {
            let count = ttsVoiceRows * ttsStyleDim
            var v = [Float](repeating: 0, count: count)
            blob.withUnsafeBytes { raw in
                for i in 0..<count {
                    v[i] = raw.loadUnaligned(
                        fromByteOffset: dataBase + start + i * 4,
                        as: Float.self)
                }
            }
            rows = v
        }
    }
    return rows
}

final class SynthContext {
    let tts: KittensCtx
    let phonemizer: Phonemizer
    let voice: [Float]     // [400 * 256]
    let speed: Float       // already x the voice's speed prior

    init(tts: KittensCtx, phonemizer: Phonemizer, voice: [Float],
         speed: Float) {
        self.tts = tts
        self.phonemizer = phonemizer
        self.voice = voice
        self.speed = speed
    }
}

func audioPushSilence(_ b: inout [Float], _ samples: Int) {
    if samples > 0 {
        b.append(contentsOf: [Float](repeating: 0, count: samples))
    }
}

func synthSentence(_ s: SynthContext, _ text: [UInt8], _ term: UInt8,
                   _ silenceMS: Int, _ out: inout [Float]) {
    let ipa = s.phonemizer.phonemize(text)
    let ids = sentenceToIDs(ipa, term, 4096)
    // The style ROW is chosen by the sentence's character length, not its
    // phoneme count and not row 0.
    var ref = text.count
    if ref > ttsVoiceRows - 1 { ref = ttsVoiceRows - 1 }
    var style = [Float](repeating: 0, count: ttsStyleDim)
    for i in 0..<ttsStyleDim { style[i] = s.voice[ref * ttsStyleDim + i] }
    let sil = Int(Double(silenceMS) * (Double(Speech.sampleRate) / 1000.0)
                  / Double(s.speed))
    audioPushSilence(&out, sil)
    let a = kittensSynthesize(s.tts, ids, ids.count, style, s.speed)
    if a.nSamples > 0 {
        out.append(contentsOf: a.samples[0..<a.nSamples])
    }
}

func synthParagraph(_ s: SynthContext, _ para: [UInt8], _ len: Int,
                    _ firstPara: Bool, _ out: inout [Float],
                    _ emitted: inout Bool) {
    var i = 0
    while i < len {
        let start = i
        while i < len && para[i] != UInt8(ascii: ".")
              && para[i] != UInt8(ascii: "!")
              && para[i] != UInt8(ascii: "?") {
            i += 1
        }
        let term: UInt8 = (i < len) ? para[i] : 0
        let end = (i < len) ? i + 1 : i
        var a = start
        var b = end
        while a < b && isSpaceC(para[a]) { a += 1 }
        while b > a && isSpaceC(para[b - 1]) { b -= 1 }
        // A lone terminator is dropped.
        let hasText = (b > a)
            && !(b - a == 1 && para[a] == term && term != 0)
        if hasText {
            var m = b - a
            if m > 2047 { m = 2047 }
            let buf = Array(para[a..<(a + m)])
            let firstSentence = (start == 0)
            let sil = !emitted ? 0
                    : ((firstPara && firstSentence) ? ttsParaSilMS
                                                    : ttsSentSilMS)
            synthSentence(s, buf, term, sil, &out)
            emitted = true
        }
        i = end
    }
}

func synthDocument(_ s: SynthContext, _ text: [UInt8],
                   _ out: inout [Float]) {
    var emitted = false
    var i = 0
    let len = text.count
    while i < len {
        let start = i
        while i < len && text[i] != UInt8(ascii: "\n") { i += 1 }
        // Every paragraph boundary, the first one included, takes the
        // paragraph silence grade for its opening sentence.
        synthParagraph(s, Array(text[start..<i]), i - start, true,
                       &out, &emitted)
        if i < len { i += 1 }
    }
}
