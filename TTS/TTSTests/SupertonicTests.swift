import Foundation
import Testing
@testable import TTS

@Suite struct SupertonicTests {

    static let pack: String? = {
        let skipped = ProcessInfo.processInfo
            .environment["CHATOKF_SKIP_WEIGHTS"] == "1"
        let path = NSHomeDirectory()
            + "/huggingface.co/leok7v/supertonic/supertonic-q4.safetensors"
        let readable = FileManager.default.isReadableFile(atPath: path)
        return readable && !skipped ? path : nil
    }()

    private static func speech() -> Speech? {
        pack.flatMap { path in Speech(pack: path) }
    }

    @Test func aFileThatIsNotAPackIsRefused() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".safetensors")
        try Data(repeating: 7, count: 4096).write(to: url)
        #expect(Speech(pack: url.path) == nil)
        #expect(Speech(pack: url.path + ".absent") == nil)
        try FileManager.default.removeItem(at: url)
    }

    @Test func everyVoiceHasANameAndAnId() {
        #expect(Speech.voices.count == 10)
        #expect(Speech.voice(named: "sarah")?.id == "F1")
        #expect(Speech.voice(named: "m5")?.name == "Daniel")
        #expect(Speech.voice(named: "Kiki") == nil)
        #expect(Speech.defaultVoice.name == "Emily")
    }

    @Test(.enabled(if: SupertonicTests.pack != nil))
    func helloIsTheReferenceLengthAndRepeats() throws {
        let speech = try #require(SupertonicTests.speech())
        let alex = Speech.voice(named: "M1")
        let once = speech.synthesize("Hello world.", voice: alex)
        let twice = speech.synthesize("Hello world.", voice: alex)
        #expect(once.count == 61440)
        #expect(once == twice)
        #expect(once.allSatisfy { sample in sample.isFinite })
        #expect(once.contains { sample in abs(sample) > 0.05 })
    }

    @Test(.enabled(if: SupertonicTests.pack != nil))
    func whatCannotBeReadIsSilenceAndNeverATrap() throws {
        let speech = try #require(SupertonicTests.speech())
        #expect(speech.synthesize("\u{1F600}\u{1F680}").isEmpty)
        #expect(speech.synthesize("... !!").isEmpty)
        #expect(speech.synthesize("").isEmpty)
        let odd = "Fine \u{2265} 20\u{00B0}C \u{2192} ok\u{2026} \u{2713}"
        #expect(!speech.synthesize(odd).isEmpty)
        #expect(!speech.synthesize("word", language: "xx").isEmpty)
    }

    @Test(.enabled(if: SupertonicTests.pack != nil))
    func aRunWithNoStopAndNoSpaceIsCutIntoChunks() throws {
        let speech = try #require(SupertonicTests.speech())
        let run = String(repeating: "a", count: chunkLimit + 40)
        let alone = speech.synthesize(String(run.prefix(chunkLimit)))
        let whole = speech.synthesize(run)
        let pause = Int(chunkSilence * Double(Speech.sampleRate))
        #expect(whole.count > alone.count + pause)
    }

}
