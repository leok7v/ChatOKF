import Foundation
import Testing
@testable import LLM

struct VLTowerTests {

    private func pixels(_ w: Int, _ h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h * 3)
        for i in 0 ..< out.count {
            out[i] = Float((i * 7919) % 255) / 127.5 - 1
        }
        return out
    }

    @Test func positionTableIsIdentityOnTheBakedGrid() {
        let n = 4, e = 3
        let table = (0 ..< n * n * e).map { i in Float(i) }
        #expect(QwenViT.positionTable(table, embd: e, h: n, w: n) == table)
    }

    @Test func positionTableInterpolatesWithCornersAligned() {
        let e = 1
        let table: [Float] = [0, 1, 2, 10, 11, 12, 20, 21, 22]
        let wide = QwenViT.positionTable(table, embd: e, h: 1, w: 5)
        #expect(wide == [0, 0.5, 1, 1.5, 2])
        let tall = QwenViT.positionTable(table, embd: e, h: 2, w: 3)
        #expect(tall == [0, 1, 2, 20, 21, 22])
        let dense = QwenViT.positionTable(table, embd: e, h: 5, w: 5)
        #expect(dense[12] == 11)
        #expect(dense[6] == 5.5)
    }

    @Test func mergeOrderWalksBlocksAcrossARectangle() {
        let order = QwenViT.mergeOrderTable(h: 2, w: 4, merge: 2)
        #expect(order == [0, 1, 4, 5, 2, 3, 6, 7])
    }

    @Test func nativeSizeFollowsSmartResize() {
        let big = VisionPreprocess.nativeSize(
            width: 1000, height: 600, factor: 32,
            maxPixels: 280 * 1024, minPixels: 4 * 1024)
        #expect(big.w == 672 && big.h == 384)
        let kept = VisionPreprocess.nativeSize(
            width: 100, height: 50, factor: 32,
            maxPixels: 280 * 1024, minPixels: 4 * 1024)
        #expect(kept.w == 96 && kept.h == 64)
        let tiny = VisionPreprocess.nativeSize(
            width: 20, height: 20, factor: 32,
            maxPixels: 280 * 1024, minPixels: 4 * 1024)
        #expect(tiny.w == 64 && tiny.h == 64)
        let exact = VisionPreprocess.nativeSize(
            width: 768, height: 768, factor: 32,
            maxPixels: 576 * 1024, minPixels: 4 * 1024)
        #expect(exact.w == 768 && exact.h == 768)
    }

    @Test func expandSpansDropsTheTemplatesWrap() {
        let span = SoftSpan(placeholder: 9, ids: [5, 7, 9, 9, 8, 6, 7, 9, 8],
                            features: [Float](repeating: 0, count: 3),
                            wrap: (begin: 7, end: 8))
        let out = Continuation.expandSpans([1, 7, 9, 8, 2], [span])
        #expect(out == [1, 5, 7, 9, 9, 8, 6, 7, 9, 8, 2])
        let bare = SoftSpan(placeholder: 9, ids: [9, 9],
                            features: [Float](repeating: 0, count: 2))
        #expect(Continuation.expandSpans([1, 7, 9, 8, 2], [bare])
                == [1, 7, 9, 9, 8, 2])
    }

    private func divergence(_ a: [Float], _ b: [Float]) -> (Float, Float) {
        var worst: Float = 0
        var scale: Float = 0
        for i in 0 ..< min(a.count, b.count) {
            worst = max(worst, abs(a[i] - b[i]))
            scale = max(scale, abs(a[i]))
        }
        return (worst, scale)
    }

    @Test(needsQwenWeights) func aPairOfTheSameFrameIsTheStill() throws {
        let path = try #require(qwenGgufPath)
        let cpu = try QwenViT(path: path)
        let gpu = try QwenMetalViT(path: path)
        let p = cpu.cfg.patchSize
        let gh = 16, gw = 24
        let px = pixels(gw * p, gh * p)
        let still = cpu.forward(pixels: px, gridH: gh, gridW: gw)
        let pair = cpu.forward(pair: px, px, gridH: gh, gridW: gw)
        #expect(pair.count == still.count)
        let cpuGap = divergence(still, pair)
        #expect(cpuGap.0 <= cpuGap.1 * 0.001, "cpu pair vs still \(cpuGap)")
        let metal = gpu.forward(pair: px, px, gridH: gh, gridW: gw)
        let gap = divergence(still, metal)
        #expect(gap.0 <= gap.1 * 0.02, "metal pair vs cpu still \(gap)")
    }

    @Test(needsQwenWeights) func nativeVideoTurnSeesTheDogs() async throws {
        let path = try #require(qwenGgufPath)
        let chat = try QwenMetalChat(ggufPath: path)
        let backend = chat.backend()
        let media = try #require(backend.media())
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("App/Resources/dogs-beach.mp4")
        var frames = 0
        let got = try await media.video(url: url, budget: 280) { _, _ in
            frames += 1
        }
        let span = try #require(got.spans.first)
        #expect(frames >= 4)
        #expect(span.wrap != nil)
        #expect(span.rows * 2 >= frames * 16 / 2)
        #expect(span.ids.filter { id in id == span.placeholder }.count
                == span.rows)
        let session = ChatSession(
            backend: backend, template: chat.chatTemplate,
            system: "You are a helpful assistant.",
            vocabSize: chat.tokenizer.vocabCount, presets: .greedy,
            maxTokens: 60)
        let ask = "What animals are in this video? Answer in one sentence."
        var answer = ""
        for await piece in session.replySoft(
            ask, parts: got.parts + [.text(ask)], spans: got.spans) {
            answer += piece
        }
        #expect(answer.lowercased().contains("dog"), "answer: \(answer)")
    }

    @Test(needsQwenWeights) func adapterReachesTheEngineLevers() async throws {
        let path = try #require(qwenGgufPath)
        let backend = try QwenMetalChat(ggufPath: path).backend()
        backend.requestStop()
        #expect(backend.shouldStop())
        await backend.reset()
        #expect(!backend.shouldStop())
        #expect(await backend.queuedCount() == 0)
        #expect(await backend.supportsVision())
        #expect(backend.drainSpecTurn() == nil)
    }

    @Test(needsQwenWeights) func nativeImageTurnSeesTheDogs() async throws {
        let path = try #require(qwenGgufPath)
        let chat = try QwenMetalChat(ggufPath: path)
        let backend = chat.backend()
        let media = try #require(backend.media())
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/ViT-512x382.jpeg")
        let got = try media.image(try Data(contentsOf: url), budget: 280)
        let grid = try #require(got.spans.first?.grid)
        #expect(grid.w == 32 && grid.h == 24, "512x382 -> \(grid)")
        #expect(got.rows == 192)
        let session = ChatSession(
            backend: backend, template: chat.chatTemplate,
            system: "You are a helpful assistant.",
            vocabSize: chat.tokenizer.vocabCount, presets: .greedy,
            maxTokens: 60)
        let ask = "What animals are in this picture? Answer in one sentence."
        var answer = ""
        for await piece in session.replySoft(
            ask, parts: got.parts + [.text(ask)], spans: got.spans) {
            answer += piece
        }
        #expect(answer.lowercased().contains("dog"), "answer: \(answer)")
    }

    @Test(needsQwenWeights) func nativeGridMatchesAcrossEngines() throws {
        let path = try #require(qwenGgufPath)
        let cpu = try QwenViT(path: path)
        let gpu = try QwenMetalViT(path: path)
        let p = cpu.cfg.patchSize
        let gh = 24, gw = 40
        let px = pixels(gw * p, gh * p)
        let a = cpu.forward(pixels: px, gridH: gh, gridW: gw)
        let b = gpu.forward(pixels: px, gridH: gh, gridW: gw)
        #expect(a.count == gh * gw / 4 * cpu.cfg.projDim)
        #expect(b.count == a.count)
        var worst: Float = 0
        var scale: Float = 0
        for i in 0 ..< a.count {
            worst = max(worst, abs(a[i] - b[i]))
            scale = max(scale, abs(a[i]))
        }
        #expect(worst <= scale * 0.02,
                "cpu and metal diverge by \(worst) at scale \(scale)")
    }
}
