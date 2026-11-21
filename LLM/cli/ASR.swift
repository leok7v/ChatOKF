import ASR
import Foundation

@MainActor func asrPack() -> String {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    return args.value("--asr-pack")
        ?? home + "/huggingface.co/leok7v/whistle/whistle.cact"
}

@MainActor func probeASR() throws {
    let path = args.value("--asr")
    let language = args.value("--asr-lang") ?? ""
    let beam = args.int("--asr-beam") ?? 5
    let depth = args.int("--asr-depth") ?? 8
    let packed = args.value("--asr-weights") == "packed"
    let listing = args.flag("--asr-pieces")
    if let path {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let samples = Transcriber.samples(wav: data)
        let t0 = Date()
        let opened = Transcriber(pack: asrPack(), packed: packed, beam: beam,
                                 depth: depth)
        if opened == nil || samples.isEmpty {
            err("no whistle pack at \(asrPack()), or \(path) is not a wav\n")
            exit(1)
        }
        let t1 = Date()
        let heard = opened!.transcribe(samples, language: language)
        let t2 = Date()
        print(heard.text)
        for i in 0..<heard.pieces.count where listing {
            let one = heard.pieces[i]
            print(String(format: "%d\t%.6f\t", one.sample, Double(one.score))
                  + heard.piece(i))
        }
        err(String(format:
            "audio %.2f s, open %.2f s, transcribe %.2f s, %d steps\n",
            Double(samples.count) / Double(Transcriber.sampleRate),
            t1.timeIntervalSince(t0), t2.timeIntervalSince(t1),
            heard.steps))
        exit(0)
    }
}
