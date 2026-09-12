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
}
