import Foundation
import Testing
import TTS
@testable import ASR

@Suite struct WhistleTests {

    static let skipped = ProcessInfo.processInfo
        .environment["CHATOKF_SKIP_WEIGHTS"] == "1"

    static func readable(_ tail: String) -> String? {
        let path = NSHomeDirectory() + "/huggingface.co/leok7v/" + tail
        let found = FileManager.default.isReadableFile(atPath: path)
        return found && !skipped ? path : nil
    }

    static let pack = readable("whistle/whistle.cact")
    static let voice = readable("supertonic/supertonic-q4.safetensors")

    static func le(_ v: Int, _ n: Int) -> [UInt8] {
        (0..<n).map { i in UInt8(truncatingIfNeeded: v >> (8 * i)) }
    }

    static func wav(_ frames: [Int16], rate: Int, channels: Int) -> Data {
        let head = Array("RIFF".utf8) + le(36 + frames.count * 2, 4) +
            Array("WAVEfmt ".utf8) + le(16, 4) + le(1, 2) + le(channels, 2) +
            le(rate, 4) + le(rate * channels * 2, 4) + le(channels * 2, 2) +
            le(16, 2) + Array("data".utf8) + le(frames.count * 2, 4)
        let body = frames.flatMap { v in le(Int(v), 2) }
        return Data(head + body)
    }

    @Test func aFileThatIsNotAPackIsRefused() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".cact")
        try Data(repeating: 7, count: 65536).write(to: url)
        #expect(Transcriber(pack: url.path) == nil)
        #expect(Transcriber(pack: url.path + ".absent") == nil)
        try FileManager.default.removeItem(at: url)
    }

    @Test func theLanguagesAreTheSevenOfTheModelInTokenOrder() {
        #expect(Transcriber.languages ==
                ["en", "de", "fr", "es", "it", "nl", "pl"])
        #expect(Whistle.tag("en") == 8192)
        #expect(Whistle.tag("pl") == 8198)
        #expect(Whistle.tag("ru") == 0)
        #expect(Whistle.tag("") == 0)
    }

    @Test func aStereoWavIsMixedAndResampledLinearly() {
        let frames: [Int16] = [0, 0, 16384, 16384, -16384, 0, 8192, 8192]
        let data = WhistleTests.wav(frames, rate: 32000, channels: 2)
        let out = Transcriber.samples(wav: data)
        #expect(out == [0, -0.25])
        #expect(Transcriber.samples(wav: Data("not a wav".utf8)).isEmpty)
        let up = Transcriber.resampled([0, 1], rate: 8000)
        #expect(up == [0, 0.5, 1, 1])
    }

    @Test(.enabled(if: WhistleTests.pack != nil))
    func aBeamOrADepthOutOfRangeIsRefused() throws {
        let path = try #require(WhistleTests.pack)
        #expect(Transcriber(pack: path, beam: 0) == nil)
        #expect(Transcriber(pack: path, beam: 9) == nil)
        #expect(Transcriber(pack: path, depth: 1) == nil)
        #expect(Transcriber(pack: path, beam: 1, depth: 2) != nil)
    }

    @Test(.enabled(if: WhistleTests.pack != nil))
    func aSteadySoundGivesNoText() throws {
        let path = try #require(WhistleTests.pack)
        let whistle = try #require(Transcriber(pack: path))
        let count = 3 * Transcriber.sampleRate
        var seed: UInt64 = 7
        let noise = (0..<count).map { _ -> Float in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: seed >> 11) % 2001 - 1000) / 1000
        }
        let tone = (0..<count).map { i in
            0.3 * sinf(2 * Float(Double.pi) * 120 * Float(i) / 16000)
        }
        let clips: [[Float]] = [
            [Float](repeating: 0, count: count),
            noise.map { v in v * 0.001 },
            noise.map { v in v * 0.01 },
            tone,
            [],
        ]
        for clip in clips {
            let heard = whistle.transcribe(clip, language: "en")
            #expect(heard.text.isEmpty)
            #expect(heard.pieces.isEmpty)
            #expect(heard.steps == 0)
        }
    }

    @Test(.enabled(if: WhistleTests.pack != nil &&
                   WhistleTests.voice != nil))
    func theReadingVoiceIsHeardInBothWeightForms() throws {
        let speech = try #require(WhistleTests.voice.flatMap { path in
            Speech(pack: path)
        })
        let path = try #require(WhistleTests.pack)
        let said = speech.synthesize("Hello world.",
                                     voice: Speech.voice(named: "M1"))
        let clip = Transcriber.resampled(said,
                                         rate: Double(Speech.sampleRate))
        for packed in [false, true] {
            let whistle = try #require(Transcriber(pack: path,
                                                   packed: packed))
            let heard = whistle.transcribe(clip, language: "en")
            #expect(heard.text == "Hello world.")
            #expect(heard.pieces.count >= 2)
            let offsets = heard.pieces.map { one in one.offset }
            #expect(offsets == offsets.sorted())
            #expect(heard.pieces.allSatisfy { one in
                one.score <= 0 && one.sample >= 0 &&
                    one.sample < clip.count
            })
            let joined = (0..<heard.pieces.count).map { i in
                heard.piece(i)
            }.joined()
            #expect(joined == heard.text)
        }
    }
}
