import Foundation
import LLM
import TTS

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
    wavPutU16(&f, 1)
    wavPutU16(&f, 1)
    wavPutU32(&f, UInt32(rate))
    wavPutU32(&f, UInt32(rate * 2))
    wavPutU16(&f, 2)
    wavPutU16(&f, 16)
    f.append(contentsOf: Array("data".utf8))
    wavPutU32(&f, dataBytes)
    for sample in pcm {
        wavPutU16(&f, UInt16(bitPattern: Speech.pcm(sample)))
    }
    return f
}

@MainActor func ttsPack() -> String {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    return args.value("--tts-pack")
        ?? home + "/huggingface.co/leok7v/supertonic/supertonic-q8.safetensors"
}

func ttsFootprint() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.stride
            / MemoryLayout<natural_t>.stride)
    let kr = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw,
                      &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
}

@MainActor func probeTTS() throws {
    let listing = args.flag("--tts-voices")
    let outPath = args.value("--tts-out") ?? "tts.wav"
    let voiceName = args.value("--tts-voice")
    let speed = args.float("--tts-speed") ?? 1.0
    let language = args.value("--tts-lang") ?? "na"
    let text = args.text("--tts")
    if listing {
        for v in Speech.voices { err("  \(v.id) \(v.name): \(v.detail)\n") }
        exit(0)
    }
    let markdown = args.text("--tts-md")
    let shaped = markdown != nil
    let chunked = markdown.map { source -> [String] in
        var chunker = SpeakableText()
        var segments = chunker.push(source)
        segments.append(contentsOf: chunker.finish())
        for s in segments { err("  say: \(s.spoken)\n") }
        return segments.map { s in s.spoken }
    }
    if let pieces = chunked ?? text.map({ whole in [whole] }) {
        let voice = voiceName.flatMap { name in Speech.voice(named: name) }
        if voiceName != nil && voice == nil {
            err("unknown voice '\(voiceName!)' (try --tts-voices)\n")
            exit(2)
        }
        let t0 = Date()
        let speech = Speech(pack: ttsPack())
        if speech == nil {
            err("no voice pack at \(ttsPack()) (try --tts-pack)\n")
            exit(1)
        }
        let load = Date().timeIntervalSince(t0)
        let t1 = Date()
        var pcm: [Float] = []
        var peak = 0.0
        for piece in pieces {
            let whole = speech!.synthesize(piece, voice: voice,
                                           speed: speed, language: language)
            pcm += shaped ? Speech.trimmed(whole) : whole
            peak = max(peak, ttsFootprint())
        }
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
            + "[tts] load %.2fs, synth %.2fs, %.1fx realtime, "
            + "footprint %.0f MB after a piece at most, %.0f MB at rest\n",
            outPath, seconds, (voice ?? Speech.defaultVoice).name, speed,
            load, synth, synth > 0 ? seconds / synth : 0, peak,
            ttsFootprint()))
        exit(0)
    }
}
