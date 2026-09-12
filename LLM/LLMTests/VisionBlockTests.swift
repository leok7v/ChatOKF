import Testing
@testable import LLM

struct VisionBlockChunkTests {
    // The absolute [lo, hi) each row attends across, given the runs of vision
    // placeholders in a turn. Mirrors what Gemma4Config.visionBlocks builds.
    private func table(_ n: Int, _ runs: [(Int, Int)]) -> [(Int, Int)] {
        var out = [(Int, Int)](repeating: (0, 0), count: n)
        for r in runs {
            for j in r.0..<r.1 { out[j] = r }
        }
        return out
    }

    @Test func plainTextTakesTheWholeWant() {
        let blocks = table(200, [])
        #expect(Gemma4Config.chunkLength(blocks, at: 0, want: 128) == 128)
        #expect(Gemma4Config.chunkLength(blocks, at: 150, want: 128) == 50)
    }

    // The win this walk exists for: a video is many small blocks, and packing
    // several per chunk is what takes prefill off the 63-token curve.
    @Test func severalWholeBlocksRideOneChunk() {
        // 63-token frames separated by a 2-token timestamp, as a video turn is.
        var runs: [(Int, Int)] = []
        var at = 0
        for _ in 0..<4 {
            runs.append((at, at + 63))
            at += 65
        }
        let blocks = table(260, runs)
        // 63 + 2 + 63 = 128 exactly, and the third frame would overflow.
        #expect(Gemma4Config.chunkLength(blocks, at: 0, want: 128) == 128)
    }

    // A chunk stops SHORT of a block it cannot hold whole rather than taking
    // part of it. This is the regression that silently dropped the mask.
    @Test func aChunkStopsShortOfABlockItCannotHold() {
        let blocks = table(400, [(100, 163)])
        // 100 text ids fit; the block needs 63 more and only 28 remain.
        #expect(Gemma4Config.chunkLength(blocks, at: 0, want: 128) == 100)
        // Starting on the block it goes whole, then text fills the chunk.
        #expect(Gemma4Config.chunkLength(blocks, at: 100, want: 128) == 128)
    }

    // A block wider than the batch still rides ONE chunk: a partial one cannot
    @Test func anOversizeBlockRidesAloneRatherThanSplitting() {
        let blocks = table(400, [(0, 300)])
        #expect(Gemma4Config.chunkLength(blocks, at: 0, want: 128) == 300)
    }

    // The whole-turn invariant.
    @Test func theWalkTilesATurnAndNeverCutsABlock() {
        var runs: [(Int, Int)] = []
        var at = 7
        for _ in 0..<9 {
            runs.append((at, at + 63))
            at += 63 + 3
        }
        let n = at + 40
        let blocks = table(n, runs)
        var edges: Set<Int> = []
        var i = 0
        while i < n {
            let step = Gemma4Config.chunkLength(blocks, at: i,
                                                want: min(128, n - i))
            #expect(step >= 1, "the walk stalled at \(i)")
            i += step
            edges.insert(i)
        }
        #expect(i == n, "the chunks covered \(i) of \(n) ids")
        for r in runs {
            let cut = edges.filter { e in e > r.0 && e < r.1 }
            #expect(cut.isEmpty, "a chunk boundary \(cut) split block \(r)")
        }
    }
}
