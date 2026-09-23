import Foundation
import LLM

// Each exits the process itself; the dispatcher in main.swift keeps the order.

// The smallest templated prompt that still elicits a direct answer; the
// empty <think></think> is the reasoning-effort-none form.
func probeWrap(_ user: String) -> String {
    "<|im_start|>user\n\(user)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

@MainActor func probeVit() throws {
    if args.flag("--vit") {
        let mmproj = args.value("--vit") ?? ""
        let vit = try QwenViT(path: mmproj)
        let c = vit.cfg
        err("[vit] \(c.layers) blocks, \(c.embd) wide, \(c.imageSize)px, "
            + "\(c.mergedTokens) merged tokens -> \(c.projDim)\n")
        let pixPath = args.value("--vit-pixels")
        var pixels = [Float](repeating: 0, count: c.imageSize * c.imageSize * 3)
        if let pixPath {
            let data = try Data(contentsOf: URL(fileURLWithPath: pixPath))
            precondition(data.count == pixels.count * 4,
                         "pixels.bin size mismatch: \(data.count)")
            _ = pixels.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        }
        let t0 = Date()
        let out = vit.forward(pixels: pixels)
        err(String(format: "[vit] forward %.2fs\n", Date().timeIntervalSince(t0)))
        // f16 weights are the only delta, so cos must be ~1; the second forward
        // times the resident steady state.
        if let mvit = try? QwenMetalViT(path: mmproj) {
            let m0 = Date()
            let mout = try mvit.forward(pixels: pixels)
            let m1 = Date()
            _ = try mvit.forward(pixels: pixels)
            err(String(format: "[vit] metal forward %.2fs (warm %.2fs)\n",
                       m1.timeIntervalSince(m0),
                       Date().timeIntervalSince(m1)))
            let mcos = Vectors.cosine(out, mout)
            print(String(format: "VIT metal-vs-cpu cos=%.7f %@", mcos,
                         mcos > 0.999 ? "MATCH" : "MISMATCH"))
            if mcos <= 0.999 { exit(1) }
        } else {
            err("[vit] no Metal device; GPU cross-gate skipped\n")
        }
        if let refPath = args.value("--vit-ref") {
            let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
            var ref = [Float](repeating: 0, count: data.count / 4)
            _ = ref.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            precondition(ref.count == out.count,
                         "ref count \(ref.count) != out \(out.count)")
            let cos = Vectors.cosine(out, ref)
            var worst: Float = 0
            for i in 0 ..< out.count { worst = max(worst, abs(out[i] - ref[i])) }
            print(String(format: "VIT cos=%.7f maxdiff=%.5f %@", cos, worst,
                         cos > 0.9999 ? "MATCH" : "MISMATCH"))
            exit(cos > 0.9999 ? 0 : 1)
        }
        print("VIT ok (no ref given): \(out.count) values")
        exit(0)
    }
}

@MainActor func probeNet() async {
    if args.flag("--web") {
        let q = args.value("--web") ?? ""
        print(await Tools.websearch(q, count: 5))
        exit(0)
    }
    if args.flag("--news") {
        let topic = args.value("--news")
        print(await Tools.news(topic))
        exit(0)
    }
    if args.flag("--fetch") {
        let u = args.value("--fetch") ?? ""
        print(await Tools.fetch(u, limit: 1200, offset: 0))
        exit(0)
    }
    if args.flag("--weather") {
        let loc = args.value("--weather") ?? ""
        print(await Tools.weather(loc))
        exit(0)
    }
}
