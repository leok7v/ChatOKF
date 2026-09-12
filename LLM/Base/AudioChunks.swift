import Accelerate
import Foundation

public enum AudioChunks {
    public struct Chunk: Sendable {
        public let range: Range<Int>
        public let hardCut: Bool
    }

    private static let hopSeconds = 0.020
    private static let pauseSeconds = 0.300
    private static let speechOverFloor: Float = 3
    private static let floorPercentile = 0.20

    public static func split(_ pcm: [Float], rate: Double,
                             maxSeconds: Double) -> [Chunk] {
        let hop = max(Int(rate * hopSeconds), 1)
        let cap = max(Int(rate * maxSeconds), hop)
        var out: [Chunk] = []
        if pcm.count <= cap {
            out = [Chunk(range: 0 ..< pcm.count, hardCut: false)]
        } else {
            let cuts = pauseCuts(pcm, hop: hop)
            var start = 0
            while start < pcm.count {
                let ceiling = start + cap
                let end = ceiling >= pcm.count
                    ? pcm.count
                    : (cuts.last { at in at > start && at <= ceiling }
                        ?? ceiling)
                out.append(Chunk(range: start ..< end,
                                 hardCut: end < pcm.count && !cuts.contains(end)))
                start = end
            }
        }
        return out
    }

    // Cut at the MIDDLE of a pause: an edge clips the breath before the next
    // sentence or leaves it dangling, and the tower pads semicausally.
    private static func pauseCuts(_ pcm: [Float], hop: Int) -> [Int] {
        let energy = frameEnergies(pcm, hop: hop)
        let quiet = energy.sorted()
        let floor = quiet.isEmpty
            ? 0 : quiet[min(Int(Double(quiet.count) * floorPercentile),
                            quiet.count - 1)]
        let threshold = floor * speechOverFloor
        let minRun = max(Int(pauseSeconds / hopSeconds), 1)
        var out: [Int] = []
        var run = 0
        for i in 0 ... energy.count {
            let quietHere = i < energy.count && energy[i] <= threshold
            if quietHere {
                run += 1
            } else {
                if run >= minRun { out.append((i - run / 2) * hop) }
                run = 0
            }
        }
        return out
    }

    private static func frameEnergies(_ pcm: [Float], hop: Int) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(pcm.count / hop + 1)
        pcm.withUnsafeBufferPointer { p in
            var at = 0
            while at < pcm.count {
                let n = min(hop, pcm.count - at)
                var rms: Float = 0
                vDSP_rmsqv(p.baseAddress! + at, 1, &rms, vDSP_Length(n))
                out.append(rms)
                at += n
            }
        }
        return out
    }
}
