import Foundation
import Testing
@testable import LLM

// The mic gate's state machine, on synthetic audio so every assertion is exact
struct SpeechGateTests {
    private static let rate = 16000.0

    // A tone the pre-emphasised RMS reads far above a quiet room.
    private static func tone(_ seconds: Double, level: Float = 0.3) -> [Float] {
        let n = Int(rate * seconds)
        return (0 ..< n).map { i in
            level * sinf(2 * .pi * 440 * Float(i) / Float(rate))
        }
    }

    // Room tone: low, and deterministic so a run cannot flake.
    private static func room(_ seconds: Double) -> [Float] {
        let n = Int(rate * seconds)
        var s: UInt64 = 0x9E3779B97F4A7C15
        return (0 ..< n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: s >> 33))
                / Float(1 << 30) * 0.0008
        }
    }

    private func gate() -> SpeechGate {
        SpeechGate(rate: SpeechGateTests.rate, maxSeconds: 30)
    }

    private func run(_ blocks: [[Float]]) -> [SpeechGate.Utterance] {
        let g = gate()
        var out: [SpeechGate.Utterance] = []
        for b in blocks { out.append(contentsOf: g.push(b)) }
        out.append(contentsOf: g.finish())
        return out
    }

    // Two spoken stretches with a long gap: two utterances, and the silence
    // between them never reaches the caller.
    @Test func twoUtterancesDropTheSilenceBetween() {
        let said = run([SpeechGateTests.room(1.0),
                        SpeechGateTests.tone(0.8),
                        SpeechGateTests.room(1.5),
                        SpeechGateTests.tone(0.8),
                        SpeechGateTests.room(1.5)])
        #expect(said.count == 2, "expected two utterances, got \(said.count)")
        let kept = said.reduce(0.0) { sum, u in
            sum + Double(u.samples.count) / SpeechGateTests.rate
        }
        // 0.8 of speech plus its margins, twice -- against 5.6 s of clock.
        #expect(kept < 3.0, "kept \(kept)s, which is most of the clock")
        #expect(kept > 1.6, "kept \(kept)s, less than the speech itself")
    }

    // The margins are the measured ones and they are ASYMMETRIC: 150 ms of
    // pre-roll, 300 ms of post-roll.
    @Test func utteranceCarriesItsMargins() {
        let said = run([SpeechGateTests.room(1.0),
                        SpeechGateTests.tone(1.0),
                        SpeechGateTests.room(1.5)])
        #expect(said.count == 1)
        let secs = Double(said[0].samples.count) / SpeechGateTests.rate
        // 1.0 speech + 0.15 pre + 0.30 post, within a hop either way.
        #expect(abs(secs - 1.45) < 0.05,
                "utterance is \(secs)s, expected ~1.45s of speech + margins")
    }

    // A click is shorter than a syllable. Dropping it is not tidiness: a span
    // with no speech in it comes back as invented speech.
    @Test func aClickIsNotAnUtterance() {
        let said = run([SpeechGateTests.room(1.0),
                        SpeechGateTests.tone(0.05),
                        SpeechGateTests.room(1.5)])
        #expect(said.isEmpty, "a 50 ms blip became \(said.count) utterance(s)")
    }

    // An empty room produces nothing at all, however long it is listened to.
    @Test func silenceProducesNothing() {
        #expect(run([SpeechGateTests.room(6.0)]).isEmpty)
    }

    // Speech that never pauses is cut at the tower's ceiling and continues,
    // rather than growing into something that cannot be encoded.
    @Test func unbrokenSpeechIsCutAtTheCeiling() {
        let g = SpeechGate(rate: SpeechGateTests.rate, maxSeconds: 2)
        var said = g.push(SpeechGateTests.room(1.0))
        said.append(contentsOf: g.push(SpeechGateTests.tone(5.0)))
        said.append(contentsOf: g.finish())
        #expect(said.count >= 2,
                "unbroken speech under a 2 s ceiling gave \(said.count)")
        #expect(said.dropLast().allSatisfy { u in u.hardCut },
                "a ceiling cut must say it was one")
        for u in said {
            let secs = Double(u.samples.count) / SpeechGateTests.rate
            #expect(secs <= 2.3, "utterance of \(secs)s exceeds the ceiling")
        }
    }

    // A microphone hands over a stretch of exact zeros while the hardware spins
    // up.
    @Test func aSilentOpenDoesNotLatchTheGate() {
        let said = run([[Float](repeating: 0, count: 4096),
                        SpeechGateTests.room(1.0),
                        SpeechGateTests.tone(0.8),
                        SpeechGateTests.room(1.5)])
        #expect(said.count == 1, "expected one utterance, got \(said.count)")
        let kept = said.reduce(0.0) { sum, u in
            sum + Double(u.samples.count) / SpeechGateTests.rate
        }
        #expect(kept < 1.6, "kept \(kept)s of a 3.5 s clock: the gate latched")
    }

    // The block size the caller happens to use must not change what is heard:
    // a mic hands over whatever its hardware buffer holds.
    @Test func blockSizeDoesNotChangeTheResult() {
        let whole = SpeechGateTests.room(1.0) + SpeechGateTests.tone(0.8)
            + SpeechGateTests.room(1.5)
        let oneGo = run([whole])
        let g = gate()
        var dribbled: [SpeechGate.Utterance] = []
        var at = 0
        while at < whole.count {
            let n = min(511, whole.count - at)
            dribbled.append(contentsOf: g.push(Array(whole[at ..< at + n])))
            at += n
        }
        dribbled.append(contentsOf: g.finish())
        #expect(oneGo.count == dribbled.count)
        #expect(oneGo.first?.samples.count == dribbled.first?.samples.count,
                "block size changed the utterance length")
    }
}
