import Foundation

public enum VisionPositions {

    static let visionMerge = 2

    public static func visionPositionsMulti(_ n: Int,
        _ spans: [(start: Int, gh: Int, gw: Int)],
        startScalar: Int = 0)
        -> (pos: [(Int32, Int32, Int32)], next: Int) {
        let ordered = spans.sorted { $0.start < $1.start }
        var pos = [(Int32, Int32, Int32)](repeating: (0, 0, 0), count: n)
        var cur = startScalar
        var i = 0
        var si = 0
        while i < n {
            if si < ordered.count && i == ordered[si].start {
                let sp = ordered[si]
                let gm = sp.gh / visionMerge
                let gwm = sp.gw / visionMerge
                let nImg = gm * gwm
                let s = Int32(cur)
                for r in 0 ..< nImg {
                    pos[sp.start + r] = (s, s + Int32(r / gwm), s + Int32(r % gwm))
                }
                cur += max(sp.gh, sp.gw) / visionMerge
                i = sp.start + nImg
                si += 1
            } else {
                pos[i] = (Int32(cur), Int32(cur), Int32(cur))
                cur += 1
                i += 1
            }
        }
        return (pos, cur)
    }
}
