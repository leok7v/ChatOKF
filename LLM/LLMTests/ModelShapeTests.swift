import Foundation
import Testing
@testable import LLM

// ModelShape is what the app's status line says about a model before any turn
// has produced a number, so everything in it is read from the file.
struct ModelShapeTests {

    @Test(needsGemmaWeights) func shapeAgreesWithTheTensors() throws {
        let g = try GGUF(path: gemmaGgufPath!)
        let shape = ModelShape(gguf: g)
        // The stated width IS the embedding table's row length.
        #expect(shape.embedding == g.tensor("token_embd.weight").dims[0])
        // E2B and E4B state the trained context only inside the origin
        // config.json they carry; the 12B keys it directly.
        #expect(shape.trainedContext > 0)
        // Every tensor lands in exactly one tower, so the towers account for
        let payload = g.tensors.values.reduce(0) { sum, t in
            sum + t.byteCount
        }
        let towers = shape.towers.reduce(0) { sum, t in sum + t.bytes }
        #expect(towers == payload)
        // The trunk leads, and outweighs any sense grafted onto it.
        #expect(shape.towers.first?.name == "text")
        let trunk = shape.towers.first?.bytes ?? 0
        #expect(shape.towers.dropFirst().allSatisfy { t in t.bytes < trunk })
    }

    static let qwen2B = ModelShape.Eye(
        weights: 302_776_320, layers: 24, width: 1024,
        projector: 25_165_824, merge: 2, k: ModelShape.clipK)
    static let qwen2BText = 1_373_159_424
    static let gemmaE2B = ModelShape.Eye(
        weights: 151_584_768, layers: 16, width: 768,
        projector: 1_179_648, merge: 3, k: ModelShape.gemmaK)
    static let gemmaE2BText = 1_876_819_968

    @Test func aPictureCostsWhatItsTowerAndThePrefillRateSay() {
        let cases: [(eye: ModelShape.Eye, text: Int, soft: Int, pp: Double,
                     seen: Double)] = [
            (ModelShapeTests.qwen2B, ModelShapeTests.qwen2BText, 192, 129.5,
             5.9),
            (ModelShapeTests.qwen2B, ModelShapeTests.qwen2BText, 270, 147.4,
             5.7),
            (ModelShapeTests.gemmaE2B, ModelShapeTests.gemmaE2BText, 270,
             104.2, 19.1),
        ]
        for c in cases {
            let look = ModelShape.lookSeconds(c.eye, textWeights: c.text,
                                              softTokens: c.soft, pp: c.pp,
                                              k: c.eye.k)
            #expect(look > c.seen * 0.55 && look < c.seen * 1.75)
            let whole = ModelShape.pictureSeconds(
                c.eye, textWeights: c.text, softTokens: c.soft, pp: c.pp,
                k: c.eye.k)
            #expect(abs(whole - look - Double(c.soft) / c.pp) < 1e-9)
        }
        let sizes = [70, 140, 280, 560, 1120]
        for eye in [ModelShapeTests.qwen2B, ModelShapeTests.gemmaE2B] {
            let times = sizes.map { n in
                ModelShape.pictureSeconds(eye, textWeights: 2_000_000_000,
                                          softTokens: n, pp: 100, k: eye.k)
            }
            #expect(zip(times, times.dropFirst()).allSatisfy { a, b in a < b })
        }
    }

    @Test(needsGemmaWeights) func theEyeIsReadFromTheTowerTensors() throws {
        let shape = ModelShape(gguf: try GGUF(path: gemmaGgufPath!))
        let eye = try #require(shape.eye)
        let want = ModelShapeTests.gemmaE2B
        #expect(eye.weights == want.weights && eye.layers == want.layers)
        #expect(eye.width == want.width && eye.merge == want.merge)
        #expect(eye.projector == want.projector && eye.k == want.k)
        #expect(shape.textWeights == ModelShapeTests.gemmaE2BText)
        if let path = TestWeights.find("Qwen3.5-2B") {
            let qwen = ModelShape(gguf: try GGUF(path: path))
            let seen = try #require(qwen.eye)
            let wanted = ModelShapeTests.qwen2B
            #expect(seen.weights == wanted.weights)
            #expect(seen.layers == wanted.layers && seen.merge == wanted.merge)
            #expect(seen.width == wanted.width && seen.k == wanted.k)
            #expect(qwen.textWeights == ModelShapeTests.qwen2BText)
        }
    }
}
