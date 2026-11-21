import Foundation
import Testing
@testable import LLM

let bonsai2Paths: [String] = ["2B", "4B"].compactMap { size in
    TestWeights.clone("prism-ml/Ternary-Bonsai-2-\(size)-gguf",
                      "Ternary-Bonsai-2-\(size)-Q2_0.gguf")
}

let needsBonsai2Weights = ConditionTrait.enabled(
    if: bonsai2Paths.count == 2,
    Comment(rawValue: "no Ternary-Bonsai-2-{2B,4B}-Q2_0.gguf under "
        + "~/huggingface.co/prism-ml or ~/Models/prism-ml"))

let bonsai2SmokePath = TestWeights.find("Ternary-Bonsai-2-2B")

let needsBonsai2Smoke = ConditionTrait.enabled(
    if: bonsai2SmokePath != nil,
    Comment(rawValue: TestWeights.missing("Ternary-Bonsai-2-2B")))

struct HadamardTests {

    private func fill(_ n: Int, seed: UInt64) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var s = seed &+ 0x9E37_79B9_7F4A_7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: s >> 33)
            out[i] = Float(bits) / Float(UInt32.max) * 2 - 1
        }
        return out
    }

    private func relDiff(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0
        var mag: Float = 0
        for i in 0..<min(a.count, b.count) {
            d = max(d, abs(a[i] - b[i]))
            mag = max(mag, abs(b[i]))
        }
        return mag > 0 ? d / mag : d
    }

    @Test func theButterflyIsTheSylvesterMatrix() {
        for n in [512, 1024, 4096] {
            let h = Hadamard.dense(n)
            let x = fill(n, seed: UInt64(n))
            var want = [Float](repeating: 0, count: n)
            for r in 0..<n {
                var s: Float = 0
                for c in 0..<n { s += h[r * n + c] * x[c] }
                want[r] = s
            }
            var got = x
            Hadamard.rows(&got, width: n, block: n, signs: nil,
                          inverse: false)
            #expect(relDiff(got, want) < 1e-5, "block \(n)")
        }
    }

    @Test func forwardThenInverseIsTheIdentity() {
        for block in [512, 1024, 4096] {
            let width = 2 * block
            let x = fill(3 * width, seed: 7)
            let signs = fill(width, seed: 8).map { v in
                v < 0 ? Float(-1) : 1
            }
            var y = x
            Hadamard.rows(&y, width: width, block: block, signs: signs,
                          inverse: false)
            #expect(relDiff(y, x) > 0.1, "the transform moved nothing")
            Hadamard.rows(&y, width: width, block: block, signs: signs,
                          inverse: true)
            #expect(relDiff(y, x) < 1e-5, "round trip at \(block)")
        }
    }

    @Test func theGroupingIsThePermutationUpstreamApplies() {
        let g = HadamardFold.Grouping(hd: 2, nk: 3, rep: 2)
        var x: [Float] = (0..<12).map { i in Float(i) }
        Hadamard.grouped(&x, width: 12, g)
        var want = [Float](repeating: 0, count: 12)
        for k in 0..<3 {
            for rep in 0..<2 {
                for d in 0..<2 {
                    want[d + 2 * (rep + 2 * k)] = Float(d + 2 * (k + 3 * rep))
                }
            }
        }
        #expect(x == want)
    }

    private struct Header {
        var keys = true
        var version = 2
        var tied = true
        var block = 512
        var widths: [Int32] = [512]
        var values: [Int32]? = nil
        var names = ["blk.0.ffn_up.weight", "blk.0.ssm_out.weight"]
        var latent = ["token_embd.weight"]
        var output = false
        var tensorWidth = 512
        var grouped = false
    }

    private func write(_ h: Header) throws -> String {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".gguf").path
        let w = try GGUFWriter(path: path, alignment: 32)
        let k = HadamardFold.key
        w.meta("general.architecture", .string("qwen35"))
        if h.keys {
            w.meta(k + "version", .u32(UInt32(h.version)))
            w.meta(k + "tied_output", .bool(h.tied))
            w.meta(k + "block_size", .u32(UInt32(h.block)))
            w.meta(k + "transform",
                   .string("normalized-sylvester-walsh-hadamard"))
            w.meta(k + "axis", .string("input-last-dimension"))
            w.meta(k + "sign_mode", .string("explicit"))
            w.meta(k + "sign_widths", .i32s(h.widths))
            let total = Int(h.widths.reduce(0, +))
            w.meta(k + "sign_values", .i32s(h.values ?? (0..<total).map { i in
                i % 3 == 0 ? -1 : 1
            }))
            w.meta(k + "weight_names", .strings(h.names))
            w.meta(k + "inverse_weight_names", .strings(h.latent))
            if h.grouped { w.meta(k + "gdn_v_grouped", .bool(true)) }
        }
        var declared = ["token_embd.weight"] + h.names
        if h.output { declared.append("output.weight") }
        for name in declared {
            w.declare(name, dims: [h.tensorWidth, 4], type: .f32)
        }
        w.finishHeader()
        let zeros = [Float](repeating: 0, count: h.tensorWidth * 4)
        for name in declared {
            try zeros.withUnsafeBytes { raw in
                try w.append(raw, expecting: name)
            }
        }
        try w.close()
        return path
    }

    @Test func theReaderAcceptsAWrittenHeader() throws {
        let g = try GGUF(path: try write(Header()))
        let fold = try #require(try HadamardFold.read(g, nV: 4, nK: 2))
        #expect(fold.version == 2 && fold.tied && fold.block == 512)
        #expect(fold.explicit)
        #expect(fold.rotated == ["blk.0.ffn_up.weight",
                                 "blk.0.ssm_out.weight",
                                 "token_embd.weight"])
        #expect(fold.inverse == ["token_embd.weight"])
        #expect(fold.signs[512]?.count == 512)
        #expect(fold.grouping == nil)
        let t = fold.transform(g.tensor("blk.0.ffn_up.weight"))
        #expect(t?.signs?.count == 512 && t?.block == 512)
        #expect(fold.transform(g.tensor("token_embd.weight")) != nil)
        #expect(fold.inverseTransform(g.tensor("token_embd.weight")) != nil)
        #expect(fold.inverseTransform(g.tensor("blk.0.ffn_up.weight")) == nil)
    }

    @Test func theReaderCarriesTheGroupedValueHeads() throws {
        var h = Header()
        h.grouped = true
        let g = try GGUF(path: try write(h))
        let fold = try #require(try HadamardFold.read(g, nV: 4, nK: 2))
        #expect(fold.grouping == HadamardFold.Grouping(hd: 128, nk: 2,
                                                       rep: 2))
        #expect(fold.transform(g.tensor("blk.0.ssm_out.weight"))?
            .grouping != nil)
        #expect(fold.transform(g.tensor("blk.0.ffn_up.weight"))?
            .grouping == nil)
        #expect(throws: HadamardErr.self) {
            try HadamardFold.read(g, nV: 0, nK: 0)
        }
    }

    @Test func aFileWithoutTheKeysReadsAsNoFold() throws {
        var h = Header()
        h.keys = false
        let g = try GGUF(path: try write(h))
        #expect(try HadamardFold.read(g, nV: 4, nK: 2) == nil)
    }

    @Test func theReaderRefusesABadHeader() throws {
        var cases: [(String, Header)] = []
        var v = Header()
        v.version = 3
        cases.append(("version 3", v))
        v = Header()
        v.tied = false
        cases.append(("version 2 untied", v))
        v = Header()
        v.block = 500
        cases.append(("block 500", v))
        v = Header()
        v.widths = [100]
        cases.append(("width 100", v))
        v = Header()
        v.values = [Int32](repeating: 2, count: 512)
        cases.append(("sign 2", v))
        v = Header()
        v.names = ["blk.0.attn_norm.weight"]
        cases.append(("a norm among the names", v))
        v = Header()
        v.names = ["blk.0.ffn_up.weight", "blk.0.ffn_up.weight"]
        cases.append(("a duplicate name", v))
        v = Header()
        v.output = true
        cases.append(("tied with output.weight present", v))
        v = Header()
        v.tensorWidth = 520
        cases.append(("a width the block does not divide", v))
        v = Header()
        v.latent = []
        cases.append(("tied without a latent table", v))
        for (label, h) in cases {
            let g = try GGUF(path: try write(h))
            #expect(throws: HadamardErr.self, "\(label)") {
                try HadamardFold.read(g, nV: 4, nK: 2)
            }
        }
    }

    @Test(needsBonsai2Smoke) func theBonsai22BNamesTheCapitalOfFrance()
        async throws {
        let path = try #require(bonsai2SmokePath)
        let backend: any AgentBackend = MetalContext.supported
            ? try QwenMetalChat(ggufPath: path).backend()
            : try QwenChat(ggufPath: path).backend()
        await backend.useSampler(nil)
        await backend.reset()
        var next = try await backend.extend(
            backend.encode("The capital of France is"))
        var out: [Int32] = []
        while out.count < 8 {
            out.append(next)
            next = try await backend.decode(next)
        }
        let text = backend.text(out)
        print("bonsai2 2B smoke, "
            + (MetalContext.supported ? "Metal" : "CPU")
            + ": \(out) \(text.debugDescription)")
        #expect(text.contains("Paris"), "decoded: \(text)")
    }

    @Test(needsBonsai2Weights) func theBonsai2FilesRead() throws {
        for (path, size) in zip(bonsai2Paths, ["2B", "4B"]) {
            let g = try GGUF(path: path)
            #expect(g.q2Block == 64, "\(size) carries the upstream block")
            #expect(g.tensor("token_embd.weight").type == .q2_64)
            #expect(g.tensor("token_embd.weight").byteCount
                    == g.tensor("token_embd.weight").count / 64 * 18)
            _ = try QwenModel(path: path)
            let nV = try #require(g.int("qwen35.ssm.time_step_rank"))
            let nK = try #require(g.int("qwen35.ssm.group_count"))
            let fold = try #require(try HadamardFold.read(g, nV: nV, nK: nK))
            #expect(fold.block == 512 && fold.version == 2 && fold.tied)
            #expect(fold.inverse == ["token_embd.weight"])
            #expect(fold.rotated.count == (size == "2B" ? 151 : 201))
            let want = size == "2B" ? nil
                : HadamardFold.Grouping(hd: 128, nk: 16, rep: 2)
            #expect(fold.grouping == want, "\(size)")
            let table = g.tensor("token_embd.weight")
            #expect(fold.transform(table) != nil)
            #expect(fold.inverseTransform(table) != nil)
            print("hadamard \(size): block \(fold.block) version "
                + "\(fold.version) tied \(fold.tied) rotated "
                + "\(fold.rotated.count) widths \(fold.signs.keys.sorted()) "
                + "grouping \(String(describing: fold.grouping))")
        }
    }
}
