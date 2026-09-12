import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum VisionError: Error { case decode }

public enum VisionPreprocess {
    static let maxDecodePx = 4096

    // Scales down at decode time, so the full bitmap is never materialized. No
    // EXIF transform: the tower path matches HF's raw pixel order.
    static func decodeCapped(_ data: Data) -> CGImage? {
        decode(data, maxPx: maxDecodePx, transform: false)
    }

    public static func thumbnail(_ data: Data, maxPx: Int) -> CGImage? {
        decode(data, maxPx: maxPx, transform: true)
    }

    public static func image(_ data: Data) -> CGImage? {
        CGImageSourceCreateWithData(data as CFData, nil).flatMap { src in
            CGImageSourceCreateImageAtIndex(src, 0, nil)
        }
    }

    public static func jpeg(_ cg: CGImage, quality: Double = 0.7) -> Data? {
        var out: Data? = nil
        let buf = NSMutableData()
        if let dst = CGImageDestinationCreateWithData(
            buf, UTType.jpeg.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(dst, cg, [
                kCGImageDestinationLossyCompressionQuality: quality,
            ] as CFDictionary)
            if CGImageDestinationFinalize(dst) { out = buf as Data }
        }
        return out
    }

    private static func decode(_ data: Data, maxPx: Int,
                               transform: Bool) -> CGImage? {
        var result: CGImage? = nil
        if let src = CGImageSourceCreateWithData(data as CFData, nil) {
            let opts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPx,
                kCGImageSourceCreateThumbnailWithTransform: transform,
            ]
            result = CGImageSourceCreateThumbnailAtIndex(
                src, 0, opts as CFDictionary)
        }
        return result
    }

    public static func nativeSize(width: Int, height: Int, factor: Int,
                                  maxPixels: Int, minPixels: Int)
        -> (w: Int, h: Int) {
        func by(_ v: Double, _ rule: FloatingPointRoundingRule) -> Int {
            max(factor, Int((v / Double(factor)).rounded(rule)) * factor)
        }
        var w = by(Double(width), .toNearestOrEven)
        var h = by(Double(height), .toNearestOrEven)
        if w * h > maxPixels {
            let beta = (Double(width * height) / Double(maxPixels))
                .squareRoot()
            w = by(Double(width) / beta, .down)
            h = by(Double(height) / beta, .down)
        } else if w * h < minPixels {
            let beta = (Double(minPixels) / Double(width * height))
                .squareRoot()
            w = by(Double(width) * beta, .up)
            h = by(Double(height) * beta, .up)
        }
        return (w, h)
    }

    public static func native(_ img: CGImage, factor: Int, maxPixels: Int,
                              minPixels: Int) throws
        -> (pixels: [Float], w: Int, h: Int) {
        let size = nativeSize(width: img.width, height: img.height,
                              factor: factor, maxPixels: maxPixels,
                              minPixels: minPixels)
        var out: (pixels: [Float], w: Int, h: Int)? = nil
        if let rgb = Resample.bicubicRGB(img, size.w, size.h) {
            out = (rgb.map { v in Float(v) / 127.5 - 1 }, size.w, size.h)
        }
        if out == nil { throw VisionError.decode }
        return out!
    }
}

public enum VLPrompt {
    public static let defaultPrompt =
        "The attached image is already encoded by your vision tower; you can "
        + "see it. Describe in detail what you see in this picture."
}
