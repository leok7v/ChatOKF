import AVFoundation
import CoreGraphics
import ImageIO
import Foundation

public enum VideoFrames {
    public enum Failure: Error {
        case noVideoTrack(String)
    }

    // HOLDS THEM ALL: a decoded 4K frame is about 33 MB and thirty-two of them
    // a gigabyte. Prefer `stream` wherever frames are consumed one at a time.
    public static func sample(url: URL, count: Int) async throws
        -> (images: [CGImage], seconds: [Double]) {
        var images: [CGImage] = []
        var seconds: [Double] = []
        try await stream(url: url, count: count) { img, at in
            images.append(img)
            seconds.append(at)
        }
        return (images, seconds)
    }

    public static func stream(url: URL, count: Int,
                              _ each: (CGImage, Double) throws -> Void)
        async throws {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        if tracks.isEmpty {
            throw Failure.noVideoTrack(url.lastPathComponent)
        } else {
            let duration = try await asset.load(.duration).seconds
            let rate = try await tracks[0].load(.nominalFrameRate)
            let total = max(1, Int((duration * Double(rate)).rounded()))
            let take = min(count, total)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            // Exact frames, not the nearest keyframe: a 19 s clip has few
            // keyframes and snapping would sample the same picture repeatedly.
            gen.requestedTimeToleranceBefore = .zero
            gen.requestedTimeToleranceAfter = .zero
            for i in 0..<take {
                try Task.checkCancellation()
                let index = take == 1 ? 0
                    : Int((Double(i) * Double(total - 1)
                           / Double(take - 1)).rounded())
                let at = Double(index) / Double(max(rate, 1))
                let time = CMTime(seconds: at, preferredTimescale: 600)
                if let img = try? await gen.image(at: time).image {
                    try each(img, at)
                }
            }
        }
    }

    public static func stream(url: URL, fps: Double, minFrames: Int,
                              maxFrames: Int,
                              _ each: (CGImage, Double) throws -> Void)
        async throws {
        let duration = try await AVURLAsset(url: url).load(.duration).seconds
        let count = min(max(Int(duration * fps), minFrames), maxFrames)
        try await stream(url: url, count: count, each)
    }

    public static func poster(url: URL, maxPx: Int) async -> Data? {
        let asset = AVURLAsset(url: url)
        var out: Data? = nil
        let items = (try? await asset.load(.commonMetadata)) ?? []
        for item in items
        where out == nil && item.commonKey == .commonKeyArtwork {
            if let data = try? await item.load(.dataValue),
               let cg = VisionPreprocess.thumbnail(data, maxPx: maxPx) {
                out = VisionPreprocess.jpeg(cg)
            }
        }
        if out == nil, let duration = try? await asset.load(.duration) {
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            gen.maximumSize = CGSize(width: maxPx, height: maxPx)
            let at = CMTimeMultiplyByFloat64(duration, multiplier: 0.25)
            if let cg = try? await gen.image(at: at).image {
                out = VisionPreprocess.jpeg(cg)
            }
        }
        return out
    }

    public static func stamp(_ seconds: Double) -> String {
        String(format: "%02d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }
}

public extension VideoFrames {
    static func write(_ images: [CGImage], to dir: String) throws {
        let base = URL(fileURLWithPath: dir)
        try FileManager.default.createDirectory(
            at: base, withIntermediateDirectories: true)
        for (i, img) in images.enumerated() {
            let url = base.appendingPathComponent(String(format: "%02d.png", i))
            if let dst = CGImageDestinationCreateWithURL(
                url as CFURL, "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(dst, img, nil)
                CGImageDestinationFinalize(dst)
            }
        }
    }
}
