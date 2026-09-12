import CoreGraphics
import Foundation

// A CLASS with cached towers: building one reads its config and, on the GPU
// arm, a CPU twin; every send is serialized behind `busy`, so never concurrent.
public final class Gemma4Media: MediaEncoder, @unchecked Sendable {
    let model: Gemma4Model
    let tokenizer: GemmaTokenizer
    // A context IS the arm choice: the GPU arm needs one and cannot make its
    // own (the text engine owns the single mapping), so nil is the CPU arm.
    let ctx: MetalContext?
    private var vision: VisionTower?
    private var audio: AudioTower?

    // Image and video come BEFORE the text and audio AFTER it, per gemma's own
    // guidance; audio placed first answers a different question.
    public static func ordered(_ parts: [ContentPart],
                               around text: ContentPart) -> [ContentPart] {
        var lead: [ContentPart] = []
        var trail: [ContentPart] = []
        for part in parts {
            if case .audio = part { trail.append(part) } else {
                lead.append(part)
            }
        }
        return lead + [text] + trail
    }

    public init(_ chat: GemmaChat, ctx: MetalContext? = nil) {
        model = chat.model
        tokenizer = chat.tokenizer
        self.ctx = ctx
    }

    public var modalities: Modalities {
        Modalities(images: true, audio: true, video: true)
    }

    public func image(_ data: Data, budget: Int) throws -> Attached {
        Attached(parts: [.image], spans: [try image(data, softTokens: budget)])
    }

    // A smaller budget is what a MULTI-image turn wants: 28 of 35 layers window
    // at 512, so two images at the full ceiling push the first out of view.
    public func image(_ data: Data,
                      softTokens: Int? = nil) throws -> SoftSpan {
        let wire = try Gemma4VisionWire(model.gguf)
        let patch = try Gemma4Patchify(model)
        var out: SoftSpan? = nil
        let budget = softTokens.map { n in patch.patchBudget(n) }
            ?? patch.maxPatches
        if !model.hasVisionTower {
            // Encoder-free: the 48-pixel block IS the token, so no tower runs
            // and the grid the image resized to decides the count.
            let media = try Gemma4UnifiedMedia(model)
            let want = softTokens ?? patch.maxSoftTokens
            if let img = VisionPreprocess.decodeCapped(data),
               let cut = patch.merged(img, softTokens: want) {
                let rows = media.image(pixels: cut.pixels, pos: cut.pos)
                out = SoftSpan.bracketed(
                    begin: wire.boi, placeholder: wire.token, end: wire.eoi,
                    count: rows.count, features: rows.flatMap { r in r })
            }
        } else if let img = VisionPreprocess.decodeCapped(data),
                  let cut = patch.patches(img, budget: budget) {
            let got = try tower(cut.pixels, cut.pos)
            out = SoftSpan.bracketed(begin: wire.boi, placeholder: wire.token,
                                     end: wire.eoi, count: got.count,
                                     features: got.proj)
        }
        if out == nil {
            throw MediaError("That picture could not be read.")
        }
        return out!
    }

    // SEVERAL spans: the tower hears `maxSeconds` at a time, so the clip is cut
    // at its own pauses and each piece becomes a span.
    public func audio(_ pcm: [Float]) throws -> [SoftSpan] {
        let wire = try Gemma4AudioWire(model.gguf)
        if !model.hasAudioTower {
            // Encoder-free: a token is the next frame of raw samples, the
            // framing itself.
            let media = try Gemma4UnifiedMedia(model)
            let rate = Double(model.gguf.int("gemma4.audio.sample_rate")
                              ?? 16000)
            return AudioChunks.split(pcm, rate: rate,
                                     maxSeconds: wire.maxSeconds)
                .map { chunk in
                    let rows = media.audio(Array(pcm[chunk.range]))
                    return SoftSpan.bracketed(
                        begin: wire.boa, placeholder: wire.token,
                        end: wire.eoa, count: rows.count,
                        features: rows.flatMap { r in r })
                }
        }
        let mel = Gemma4Mel(Gemma4MelConfig(model.gguf) ?? .processorDefault)
        let tower = try audioTower()
        return AudioChunks.split(pcm, rate: Double(mel.cfg.sampleRate),
                                 maxSeconds: wire.maxSeconds)
            .map { chunk in
                let feats = mel.features(Array(pcm[chunk.range]))
                let got = tower.run(feats.values, feats.frames, mel.cfg.bins)
                return SoftSpan.bracketed(
                    begin: wire.boa, placeholder: wire.token, end: wire.eoa,
                    count: got.count, features: got.proj)
            }
    }

    // The vision tower per frame, since there are no video weights; each frame
    // carries its own mm:ss stamp and begin/end pair, telling the model when.
    public func video(frames: [CGImage], seconds: [Double]) throws -> SoftSpan {
        let film = try Gemma4VideoWire(model.gguf)
        let encode = try frameEncoder(film)
        var strip = try videoStrip(film)
        var i = 0
        while i < frames.count, let got = encode(frames[i]) {
            strip.add(stamp: stamp(seconds[i]), rows: got.proj,
                      count: got.count)
            i += 1
        }
        if i < frames.count {
            throw MediaError("Frame \(i) of that video could not be read.")
        }
        return try strip.span(bracket: .template)
    }

    // The tower is built ONCE: a construction prewarms a Metal context and a
    // CPU twin, which per frame would dominate the encode.
    private func frameEncoder(_ film: Gemma4VideoWire)
        throws -> (CGImage) -> (proj: [Float], count: Int)? {
        let patch = try Gemma4Patchify(model)
        let budget = patch.patchBudget(film.softTokensPerFrame)
        let encode: (CGImage) -> (proj: [Float], count: Int)?
        if model.hasVisionTower {
            let vit = try visionTower()
            encode = { frame in
                patch.patches(frame, budget: budget).map { cut in
                    let got = vit.run(cut.pixels, cut.pos)
                    return (got.proj, got.count)
                }
            }
        } else {
            let media = try Gemma4UnifiedMedia(model)
            encode = { frame in
                patch.merged(frame, softTokens: film.softTokensPerFrame)
                    .map { cut in
                        let rows = media.image(pixels: cut.pixels,
                                               pos: cut.pos)
                        return (rows.flatMap { r in r }, rows.count)
                    }
            }
        }
        return encode
    }

    private func videoStrip(_ film: Gemma4VideoWire) throws -> VideoStrip {
        let iw = try Gemma4VisionWire(model.gguf)
        return VideoStrip(placeholder: film.token, begin: iw.boi, end: iw.eoi)
    }

    private func stamp(_ seconds: Double) -> [Int32] {
        tokenizer.encode(VideoFrames.stamp(seconds) + " ", addSpecial: false)
    }

    public func audio(url: URL) async throws -> [SoftSpan] {
        let rate = Double((Gemma4MelConfig(model.gguf)
            ?? .processorDefault).sampleRate)
        return try audio(await AudioFile.samples(url: url, sampleRate: rate))
    }

    // STREAMED one frame at a time (a 4K frame decodes to about 33 MB);
    // `onFrame` runs off the main actor; its frame is released on return.
    public func video(url: URL, budget: Int,
                      onFrame: ((CGImage, Double) -> Void)?)
        async throws -> Attached {
        let film = try Gemma4VideoWire(model.gguf)
        let encode = try frameEncoder(film)
        var strip = try videoStrip(film)
        var read = 0
        try await VideoFrames.stream(url: url, count: film.frames) { img, at in
            if let got = encode(img) {
                onFrame?(img, at)
                strip.add(stamp: stamp(at), rows: got.proj, count: got.count)
                read += 1
            } else {
                throw MediaError("Frame \(read) of that video could not be read.")
            }
        }
        return Attached(parts: [.video], spans: [try strip.span(bracket: .template)])
    }

    public var audioSampleRate: Double {
        Double((Gemma4MelConfig(model.gguf) ?? .processorDefault).sampleRate)
    }

    public var maxAudioSeconds: Double {
        ((try? Gemma4AudioWire(model.gguf))?.maxSeconds) ?? 0
    }

    struct VisionTower {
        let run: ([Float], [(Int, Int)]) -> (tower: [Float], proj: [Float],
                                             count: Int)
    }

    func visionTower() throws -> VisionTower {
        if vision == nil {
            if let ctx {
                let gpu = try Gemma4MetalViT(model, ctx: ctx)
                vision = VisionTower { pixels, pos in
                    gpu.forward(pixels: pixels, pos: pos)
                }
            } else {
                let cpu = try Gemma4ViT(model)
                vision = VisionTower { pixels, pos in
                    cpu.forward(pixels: pixels, pos: pos)
                }
            }
        }
        return vision!
    }

    private func tower(_ pixels: [Float], _ pos: [(Int, Int)])
        throws -> (tower: [Float], proj: [Float], count: Int) {
        try visionTower().run(pixels, pos)
    }

    struct AudioTower {
        let run: ([Float], Int, Int) -> (tower: [Float], proj: [Float],
                                         count: Int)
    }

    func audioTower() throws -> AudioTower {
        if audio == nil {
            if let ctx {
                let gpu = try Gemma4MetalAudio(model, ctx: ctx)
                audio = AudioTower { mel, frames, bins in
                    gpu.forward(mel: mel, frames: frames, bins: bins)
                }
            } else {
                let cpu = try Gemma4Audio(model)
                audio = AudioTower { mel, frames, bins in
                    cpu.forward(mel: mel, frames: frames, bins: bins)
                }
            }
        }
        return audio!
    }
}
