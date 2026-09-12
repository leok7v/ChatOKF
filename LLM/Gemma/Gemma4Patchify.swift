import CoreGraphics
import Foundation
import ImageIO

public struct Gemma4Patchify {
    public let patchSize: Int
    public let poolKernel: Int
    public let maxSoftTokens: Int
    private let rescale: Float
    private let normalize: Bool
    private let mean: [Float]
    private let std: [Float]

    // A still image and a video FRAME get different budgets from the file (280
    // vs 70).
    public var maxPatches: Int { patchBudget(maxSoftTokens) }
    public var patchDim: Int { patchSize * patchSize * 3 }
    public var mergedSide: Int { poolKernel * patchSize }
    public var mergedDim: Int { mergedSide * mergedSide * 3 }

    public func patchBudget(_ softTokens: Int) -> Int {
        softTokens * poolKernel * poolKernel
    }

    public init(_ model: Gemma4Model) throws {
        let g = model.gguf
        func i(_ k: String) throws -> Int {
            try requireInt(g, "gemma4.vision." + k,
                           "an image cannot be cut without it")
        }
        patchSize = try i("patch_size")
        poolKernel = try i("pool_kernel")
        maxSoftTokens = try i("max_soft_tokens")
        rescale = Float(g.double("gemma4.vision.rescale_factor") ?? (1 / 255))
        normalize = g.bool("gemma4.vision.normalize") ?? false
        mean = (g.doubles("gemma4.vision.image_mean") ?? [0, 0, 0])
            .map { v in Float(v) }
        std = (g.doubles("gemma4.vision.image_std") ?? [1, 1, 1])
            .map { v in Float(v) }
        let want = g.string("gemma4.vision.resample") ?? "bicubic"
        // Only Pillow's bicubic is implemented and the filter shapes the pixels
        // the model sees; a near one is a wrong answer, not an approximation.
        if want != "bicubic" {
            throw GGUFErr.parse("gemma4.vision.resample is \(want); this "
                                + "build resizes bicubic")
        }
    }

    // A side rounding to zero needs an image thinner than one pooled block:
    // refused.
    func target(width: Int, height: Int,
                budget: Int) throws -> (w: Int, h: Int) {
        let patch = patchSize
        let sideMult = mergedSide
        let targetPx = Double(budget * patch * patch)
        let totalPx = Double(width * height)
        let factor = (targetPx / totalPx).squareRoot()
        let w = Int((factor * Double(width) / Double(sideMult)).rounded(.down))
            * sideMult
        let h = Int((factor * Double(height) / Double(sideMult)).rounded(.down))
            * sideMult
        if w == 0 || h == 0 {
            throw GGUFErr.parse(
                "\(width)x\(height) resizes to \(w)x\(h); a side must reach "
                + "\(sideMult)px, one pooled block")
        }
        return (w, h)
    }

    public func patches(_ data: Data)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let img = VisionPreprocess.decodeCapped(data) {
            out = patches(img, budget: maxPatches)
        }
        return out
    }

    public func merged(_ data: Data, softTokens: Int)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let img = VisionPreprocess.decodeCapped(data) {
            out = merged(img, softTokens: softTokens)
        }
        return out
    }

    // A decoded frame at an explicit budget. Video frames all share one size,
    // so every frame yields the same count and none of them pads.
    public func patches(_ img: CGImage, budget: Int)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let size = try? target(width: img.width, height: img.height,
                                  budget: budget),
           let rgb = Resample.bicubicRGB(img, size.w, size.h) {
            out = split(rgb, size.w, size.h, budget, side: patchSize)
        }
        return out
    }

    // Cutting the block whole is what upstream's merge computes: its permute
    // reassembles each k*k group into one contiguous block in [y][x][c] order.
    public func merged(_ img: CGImage, softTokens: Int)
        -> (pixels: [Float], pos: [(Int, Int)])? {
        var out: (pixels: [Float], pos: [(Int, Int)])? = nil
        if let size = try? target(width: img.width, height: img.height,
                                  budget: patchBudget(softTokens)),
           let rgb = Resample.bicubicRGB(img, size.w, size.h) {
            // No padding: a unified checkpoint scatters exactly the tokens it
            // is handed.
            let count = (size.w / mergedSide) * (size.h / mergedSide)
            out = split(rgb, size.w, size.h, count, side: mergedSide)
        }
        return out
    }

    private func split(_ rgb: [UInt8], _ w: Int, _ h: Int, _ budget: Int,
                       side patch: Int)
        -> (pixels: [Float], pos: [(Int, Int)]) {
        let cols = w / patch, rows = h / patch
        let dim = patch * patch * 3
        var pixels = [Float](repeating: 0, count: budget * dim)
        var pos = [(Int, Int)](repeating: (-1, -1), count: budget)
        for y in 0..<rows {
            for x in 0..<cols {
                let slot = y * cols + x
                pos[slot] = (x, y)
                for iy in 0..<patch {
                    let row = (y * patch + iy) * w + x * patch
                    for ix in 0..<patch {
                        for c in 0..<3 {
                            let raw = Float(rgb[(row + ix) * 3 + c])
                            let v = raw * rescale
                            let n = normalize ? (v - mean[c]) / std[c] : v
                            pixels[slot * dim + (iy * patch + ix) * 3 + c] = n
                        }
                    }
                }
            }
        }
        return (pixels, pos)
    }
}
