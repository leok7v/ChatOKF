import XCTest
@testable import LLM

final class IQ1CodecTests: XCTestCase {

    // ggml's own dequantisation of the same file, produced by
    // `llama-quantize --allow-requantize <iq1_s> <out> F32`.
    private var source: String {
        NSHomeDirectory()
            + "/Models/mradermacher/Qwen3.5-0.8B-i1-GGUF/"
            + "Qwen3.5-0.8B.i1-IQ1_S.gguf"
    }
    private let reference = "tmp/59/iq1s-ref-f32.gguf"

    private func compare(_ type: GGUFType, _ qk: Int, _ bytes: Int,
                         _ decode: (UnsafeRawPointer,
                                    UnsafeMutablePointer<Float>) -> Void,
                         _ src: String,
                         _ ref: String? = nil) throws {
        let up = TestWeights.repoRoot + "/"
        let refPath = up + (ref ?? reference)
        let src = src.hasPrefix("tmp/") ? up + src : src
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: src)
            && FileManager.default.fileExists(atPath: refPath),
            "needs the quantized model and its F32 reference")
        let q = try GGUF(path: src)
        let f = try GGUF(path: refPath)
        var checked = 0
        var worst: Float = 0
        for (name, t) in q.tensors.sorted(by: { a, b in a.key < b.key })
            where t.type == type {
            guard let r = f.maybe(name) else { continue }
            let want = Dense.floats(r)
            var got = [Float](repeating: 0, count: t.count)
            got.withUnsafeMutableBufferPointer { g in
                for b in 0..<(t.count / qk) {
                    decode(t.base + b * bytes, g.baseAddress! + b * qk)
                }
            }
            for i in 0..<t.count { worst = max(worst, abs(got[i] - want[i])) }
            checked += 1
            if checked >= 3 { break }
        }
        XCTAssertGreaterThan(checked, 0, "no \(type) tensor was compared")
        XCTAssertEqual(worst, 0, accuracy: 1e-6,
                       "\(type) must reproduce ggml bit for bit")
    }

    func testKQuantsMatchGgmlExactly() throws {
        try compare(.q2k, 256, 84, KQuant.q2K, source)
    }

    func testQ3KMatchesGgmlExactly() throws {
        try compare(.q3k, 256, 110, KQuant.q3K, "tmp/59/fix-q3k.gguf",
                    "tmp/59/fix-q3k-f32.gguf")
    }

    private func fixture(_ name: String, _ type: GGUFType, _ bytes: Int,
                         _ decode: (UnsafeRawPointer,
                                    UnsafeMutablePointer<Float>) -> Void)
        throws {
        try compare(type, 256, bytes, decode,
                    "tmp/59/fx-\(name).gguf", "tmp/59/fx-\(name)-f32.gguf")
    }

    func testQ4KMatchesGgmlExactly() throws {
        try fixture("Q4_K_M", .q4k, 144, KQuant.q4K)
    }

    func testIQ2XSMatchesGgmlExactly() throws {
        try fixture("IQ2_XS", .iq2_xs, 74, IQX.iq2XS)
    }

    func testIQ2SMatchesGgmlExactly() throws {
        try fixture("IQ2_M", .iq2_s, 82, IQX.iq2S)
    }

    func testIQ3XXSMatchesGgmlExactly() throws {
        try fixture("IQ3_XXS", .iq3_xxs, 98, IQX.iq3XXS)
    }

    func testIQ3SMatchesGgmlExactly() throws {
        try fixture("IQ3_S", .iq3_s, 110, IQX.iq3S)
    }

    func testIQ4XSMatchesGgmlExactly() throws {
        try fixture("IQ4_XS", .iq4_xs, 136, IQX.iq4XS)
    }

    func testQ5KMatchesGgmlExactly() throws {
        try compare(.q5k, 256, 176, KQuant.q5K, source)
    }

    func testQ6KMatchesGgmlExactly() throws {
        try fixture("Q6_K", .q6k, 210, KQuant.q6K)
    }

    func testIQ2XXSMatchesGgmlExactly() throws {
        try compare(.iq2_xxs, IQ2XXS.qk, IQ2XXS.blockBytes,
                    IQ2XXS.dequant, source)
    }

    func testDequantMatchesGgmlExactly() throws {
        let refPath = TestWeights.repoRoot + "/" + reference
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: source)
            && FileManager.default.fileExists(atPath: refPath),
            "needs the IQ1_S model and its F32 reference")
        let q = try GGUF(path: source)
        let f = try GGUF(path: refPath)
        var checked = 0
        var worst: Float = 0
        for (name, t) in q.tensors where t.type == .iq1_s {
            guard let r = f.maybe(name) else { continue }
            XCTAssertEqual(t.count, r.count, name)
            let want = Dense.floats(r)
            var got = [Float](repeating: 0, count: t.count)
            let blocks = t.count / IQ1.qk
            got.withUnsafeMutableBufferPointer { g in
                for b in 0..<blocks {
                    IQ1.dequant(t.base + b * IQ1.blockBytes,
                                into: g.baseAddress! + b * IQ1.qk)
                }
            }
            for i in 0..<t.count {
                worst = max(worst, abs(got[i] - want[i]))
            }
            checked += 1
            if checked >= 4 { break }
        }
        XCTAssertGreaterThan(checked, 0, "no iq1_s tensor was compared")
        XCTAssertEqual(worst, 0, accuracy: 1e-6,
                       "our dequant must reproduce ggml's bit for bit")
    }
}
