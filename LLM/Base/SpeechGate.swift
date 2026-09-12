import Accelerate
import Foundation

public final class SpeechGate: @unchecked Sendable {
    private let lock = NSLock()

    public struct Utterance: Sendable {
        public let samples: [Float]
        public let startSeconds: Double
        public let hardCut: Bool
        public let rate: Double

        public var seconds: Double { Double(samples.count) / rate }
    }

    private static let preRoll = 0.150
    private static let postRoll = 0.300
    private static let hangover = 0.700
    private static let minSpeech = 0.250
    private static let hop = 0.020
    private static let speechOverFloor: Float = 3
    private static let floorWindow = 5.0
    private static let floorPercentile = 0.20
    // Bounds a zero background; must stay far BELOW any real room, since an
    // absolute inside a relative measure stops scaling on a quieter device.
    private static let floorAtLeast: Float = 5e-5
    // Under this the input is not running yet: a microphone hands over zeros
    // while the hardware spins up, and that must not become the background.
    private static let notAudio: Float = 1e-5

    private let rate: Double
    private let ceiling: Double
    private let hopSize: Int
    private var pending: [Float] = []
    private var preRollRing: [Float] = []
    private var current: [Float] = []
    private var quiet: [Float] = []
    private var speaking = false
    private var recent: Float = 0
    private var openPreRoll = 0
    private var loudestFrame: Float = 0
    private var quietRun = 0
    private var consumed = 0
    private var startedAt = 0

    public init(rate: Double, maxSeconds: Double) {
        self.rate = rate
        self.ceiling = maxSeconds
        self.hopSize = max(Int(rate * SpeechGate.hop), 1)
    }

    public func push(_ block: [Float]) -> [Utterance] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(contentsOf: block)
        var out: [Utterance] = []
        while pending.count >= hopSize {
            let frame = Array(pending[0 ..< hopSize])
            pending.removeFirst(hopSize)
            if let done = step(frame) { out.append(done) }
            consumed += hopSize
        }
        return out
    }

    public func finish() -> [Utterance] {
        lock.lock()
        defer { lock.unlock() }
        var out: [Utterance] = []
        if !pending.isEmpty {
            current.append(contentsOf: speaking ? pending : [])
            pending = []
        }
        if let done = close(hard: false) { out.append(done) }
        return out
    }

    private func step(_ frame: [Float]) -> Utterance? {
        let e = energy(frame)
        let bar = threshold()
        recent = bar > 0 ? e / bar : 0
        if e > loudestFrame { loudestFrame = e }
        let loud = e > bar
        var out: Utterance? = nil
        if loud {
            if !speaking {
                speaking = true
                startedAt = consumed - preRollRing.count
                openPreRoll = preRollRing.count
                current = preRollRing
                preRollRing = []
            }
            current.append(contentsOf: frame)
            quietRun = 0
            if Double(current.count) / rate >= ceiling {
                out = close(hard: true)
            }
        } else if speaking {
            current.append(contentsOf: frame)
            quietRun += 1
            if Double(quietRun) * SpeechGate.hop >= SpeechGate.hangover {
                out = close(hard: false)
            }
        } else {
            note(energy(frame))
            preRollRing.append(contentsOf: frame)
            let cap = Int(rate * SpeechGate.preRoll)
            if preRollRing.count > cap {
                preRollRing.removeFirst(preRollRing.count - cap)
            }
        }
        return out
    }

    private func close(hard: Bool) -> Utterance? {
        let keep = Int(rate * SpeechGate.postRoll)
        let tail = min(Int(Double(quietRun) * SpeechGate.hop * rate),
                       current.count)
        let cut = current.count - max(tail - keep, 0)
        let voiced = Double(max(0, current.count - openPreRoll - tail)) / rate
        var out: Utterance? = nil
        if voiced >= SpeechGate.minSpeech {
            out = Utterance(samples: Array(current[0 ..< cut]),
                            startSeconds: Double(startedAt) / rate,
                            hardCut: hard, rate: rate)
        }
        current = []
        preRollRing = []
        speaking = hard
        openPreRoll = 0
        startedAt = hard ? consumed : startedAt
        quietRun = 0
        return out
    }

    private func energy(_ frame: [Float]) -> Float {
        var lifted = frame
        for i in stride(from: frame.count - 1, to: 0, by: -1) {
            lifted[i] = frame[i] - 0.97 * frame[i - 1]
        }
        var rms: Float = 0
        vDSP_rmsqv(lifted, 1, &rms, vDSP_Length(lifted.count))
        return rms
    }

    private func note(_ e: Float) {
        if e > SpeechGate.notAudio {
            quiet.append(e)
            let cap = Int(SpeechGate.floorWindow / SpeechGate.hop)
            if quiet.count > cap { quiet.removeFirst(quiet.count - cap) }
        }
    }

    private func threshold() -> Float {
        let ranked = quiet.sorted()
        var out = Float.greatestFiniteMagnitude
        if ranked.count >= 8 {
            let at = min(Int(Double(ranked.count) * SpeechGate.floorPercentile),
                         ranked.count - 1)
            out = max(ranked[at], SpeechGate.floorAtLeast)
                * SpeechGate.speechOverFloor
        }
        return out
    }

    public var speechThreshold: Float {
        lock.lock()
        defer { lock.unlock() }
        return threshold()
    }

    public var hearing: Bool {
        lock.lock()
        defer { lock.unlock() }
        return speaking
    }

    public var loudestFrameEnergy: Float {
        lock.lock()
        defer { lock.unlock() }
        return loudestFrame
    }

    public var level: Float {
        lock.lock()
        defer { lock.unlock() }
        let over = max(0, recent - 1)
        return min(1, over / 4)
    }
}
