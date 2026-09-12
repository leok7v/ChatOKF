import CoreGraphics
import Foundation

public final class QwenMedia: MediaEncoder, @unchecked Sendable {
    private let backend: QwenMetalBackend
    private let tokenizer: Tokenizer
    private let pad: Int32

    init(backend: QwenMetalBackend, tokenizer: Tokenizer, pad: Int32) {
        self.backend = backend
        self.tokenizer = tokenizer
        self.pad = pad
    }

    public var modalities: Modalities {
        Modalities(images: true, audio: false, video: true)
    }

    public var audioSampleRate: Double { 0 }
    public var maxAudioSeconds: Double { 0 }

    public func image(_ data: Data, budget: Int) throws -> Attached {
        let cfg = try backend.tower().cfg
        let factor = cfg.patchSize * cfg.merge
        var out: Attached? = nil
        if let img = VisionPreprocess.decodeCapped(data) {
            let cut = try VisionPreprocess.native(
                img, factor: factor, maxPixels: budget * factor * factor,
                minPixels: 4 * factor * factor)
            let gh = cut.h / cfg.patchSize
            let gw = cut.w / cfg.patchSize
            let rows = try backend.encode(pixels: cut.pixels,
                                          gridH: gh, gridW: gw)
            let merged = (gh / cfg.merge) * (gw / cfg.merge)
            out = Attached(parts: [.image], spans: [SoftSpan(
                placeholder: pad,
                ids: [Int32](repeating: pad, count: merged),
                features: rows, grid: (h: gh, w: gw))])
        }
        if out == nil { throw MediaError("That picture could not be read.") }
        return out!
    }

    public func audio(_ pcm: [Float]) throws -> [SoftSpan] {
        throw MediaError("This model cannot hear.")
    }

    public func audio(url: URL) async throws -> [SoftSpan] {
        throw MediaError("This model cannot hear.")
    }

    private struct Frame {
        let pixels: [Float]
        let w: Int
        let h: Int
        let at: Double
    }

    private func single(_ text: String) -> Int32 {
        tokenizer.encode(text, addSpecial: true)[0]
    }

    public func video(url: URL, budget: Int,
                      onFrame: ((CGImage, Double) -> Void)?)
        async throws -> Attached {
        let cfg = try backend.tower().cfg
        let factor = cfg.patchSize * cfg.merge
        let perPair = max(16, budget / 4)
        var strip = VideoStrip(placeholder: single("<|video_pad|>"),
                               begin: single("<|vision_start|>"),
                               end: single("<|vision_end|>"))
        var held: Frame? = nil
        func encode(_ a: Frame, _ b: Frame) throws {
            let gh = a.h / cfg.patchSize
            let gw = a.w / cfg.patchSize
            let rows = try backend.encode(pair: a.pixels, b.pixels,
                                          gridH: gh, gridW: gw)
            let stamp = String(format: "<%.1f seconds>", (a.at + b.at) / 2)
            strip.add(stamp: tokenizer.encode(stamp, addSpecial: false),
                      rows: rows,
                      count: (gh / cfg.merge) * (gw / cfg.merge),
                      grid: (h: gh, w: gw))
        }
        try await VideoFrames.stream(url: url, fps: 2, minFrames: 4,
                                     maxFrames: 768) { img, at in
            let cut = try VisionPreprocess.native(
                img, factor: factor, maxPixels: perPair * factor * factor,
                minPixels: 4 * factor * factor)
            onFrame?(img, at)
            let frame = Frame(pixels: cut.pixels, w: cut.w, h: cut.h, at: at)
            if let first = held {
                try encode(first, frame)
                held = nil
            } else {
                held = frame
            }
        }
        if let last = held { try encode(last, last) }
        return Attached(parts: [.video], spans: [try strip.span(bracket: .block)])
    }
}
