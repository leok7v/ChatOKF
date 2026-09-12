import Foundation
import LLM

// The acceptance test is a BYTE COMPARISON against the reference binary, so
// this writes the same 24 kHz mono PCM WAV: cmp ours.wav ref.wav.

// A NaN sample reaches the conversion: the reference C's cast lowers to
// fcvtzs and answers 0 on arm64, where Int16(Float) traps.
func wavClamp(_ v: Float) -> Int16 {
    var s = v
    if s > 1.0 { s = 1.0 }
    if s < -1.0 { s = -1.0 }
    return s.isNaN ? 0 : Int16(s * 32767.0)
}

func wavPutU32(_ f: inout [UInt8], _ v: UInt32) {
    f.append(UInt8(truncatingIfNeeded: v))
    f.append(UInt8(truncatingIfNeeded: v >> 8))
    f.append(UInt8(truncatingIfNeeded: v >> 16))
    f.append(UInt8(truncatingIfNeeded: v >> 24))
}

func wavPutU16(_ f: inout [UInt8], _ v: UInt16) {
    f.append(UInt8(truncatingIfNeeded: v))
    f.append(UInt8(truncatingIfNeeded: v >> 8))
}

func wavBytes(_ pcm: [Float], rate: Int) -> [UInt8] {
    var f = [UInt8]()
    let dataBytes = UInt32(pcm.count * 2)
    f.append(contentsOf: Array("RIFF".utf8))
    wavPutU32(&f, 36 + dataBytes)
    f.append(contentsOf: Array("WAVE".utf8))
    f.append(contentsOf: Array("fmt ".utf8))
    wavPutU32(&f, 16)
    wavPutU16(&f, 1)                        // PCM
    wavPutU16(&f, 1)                        // mono
    wavPutU32(&f, UInt32(rate))
    wavPutU32(&f, UInt32(rate * 2))         // byte rate
    wavPutU16(&f, 2)                        // block align
    wavPutU16(&f, 16)                       // bits
    f.append(contentsOf: Array("data".utf8))
    wavPutU32(&f, dataBytes)
    for sample in pcm {
        wavPutU16(&f, UInt16(bitPattern: wavClamp(sample)))
    }
    return f
}

@MainActor func probeTTS() throws {
    let listing = args.flag("--tts-voices")
    let outPath = args.value("--tts-out") ?? "tts.wav"
    let voiceName = args.value("--tts-voice")
    let speed = args.float("--tts-speed") ?? 1.0
    let text = args.text("--tts")
    if listing {
        for v in Speech.voices { err("  \(v.name)\n") }
        exit(0)
    }
    // --tts-md runs the text through the app's chunker; --tts stays raw, since
    // the byte gate compares against a reference that has no chunker.
    let markdown = args.text("--tts-md")
    let chunked = markdown.map { source -> String in
        var chunker = SpeakableText()
        var segments = chunker.push(source)
        segments.append(contentsOf: chunker.finish())
        for s in segments { err("  say: \(s.spoken)\n") }
        // One segment per line, so the engine's splitter gives each the pause.
        return segments.map { s in s.spoken }.joined(separator: "\n")
    }
    if let text = chunked ?? text {
        let voice = voiceName.flatMap { name in Speech.voice(named: name) }
        if voiceName != nil && voice == nil {
            err("unknown voice '\(voiceName!)' (try --tts-voices)\n")
            exit(2)
        }
        let t0 = Date()
        let speech = Speech()
        if speech == nil {
            err("speech engine unavailable (bundled resources missing)\n")
            exit(1)
        }
        let load = Date().timeIntervalSince(t0)
        let t1 = Date()
        let pcm = speech!.synthesize(text, voice: voice, speed: speed)
        let synth = Date().timeIntervalSince(t1)
        if pcm.isEmpty {
            err("no audio produced\n")
            exit(1)
        }
        let url = URL(fileURLWithPath: outPath)
        try Data(wavBytes(pcm, rate: Speech.sampleRate)).write(to: url)
        let seconds = Double(pcm.count) / Double(Speech.sampleRate)
        err(String(format:
            "wrote %@  (%.2fs audio, voice %@, speed %.2f)\n"
            + "[tts] load %.2fs, synth %.2fs, %.1fx realtime\n",
            outPath, seconds, (voice ?? Speech.defaultVoice).name, speed,
            load, synth, synth > 0 ? seconds / synth : 0))
        exit(0)
    }
}
