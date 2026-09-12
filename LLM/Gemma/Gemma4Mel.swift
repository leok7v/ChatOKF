import Accelerate
import Foundation

// Semicausal padding (frame/2 zeros PREPENDED), a PERIODIC Hann, and log of
// mel + floor rather than max(mel, floor): each shifts every frame.

public struct Gemma4MelConfig: Sendable {
    public let sampleRate: Int
    public let bins: Int
    public let fftLength: Int
    public let frameLength: Int
    public let hopLength: Int
    public let melFloor: Float
    public let minHz: Float
    public let maxHz: Float

    public static let processorDefault = Gemma4MelConfig(
        sampleRate: 16000, bins: 128, fftLength: 512, frameLength: 320,
        hopLength: 160, melFloor: 0.001, minHz: 0, maxHz: 8000)

    init?(_ g: GGUF) {
        func i(_ k: String) -> Int? { g.int("gemma4.audio.mel." + k) }
        guard let sr = i("sample_rate"), let b = i("bins"),
              let fft = i("fft_length"), let fr = i("frame_length"),
              let hop = i("hop_length"),
              let floor = g.double("gemma4.audio.mel.floor"),
              let lo = g.double("gemma4.audio.mel.min_hz"),
              let hi = g.double("gemma4.audio.mel.max_hz") else {
            return nil
        }
        sampleRate = sr
        bins = b
        fftLength = fft
        frameLength = fr
        hopLength = hop
        melFloor = Float(floor)
        minHz = Float(lo)
        maxHz = Float(hi)
    }

    public init(sampleRate: Int, bins: Int, fftLength: Int, frameLength: Int,
                hopLength: Int, melFloor: Float, minHz: Float, maxHz: Float) {
        self.sampleRate = sampleRate
        self.bins = bins
        self.fftLength = fftLength
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.melFloor = melFloor
        self.minHz = minHz
        self.maxHz = maxHz
    }
}

public final class Gemma4Mel {
    public let cfg: Gemma4MelConfig
    private let window: [Float]
    private let filters: [Float]
    private let setup: FFTSetup
    private let log2n: vDSP_Length

    public init(_ cfg: Gemma4MelConfig) {
        self.cfg = cfg
        let n = cfg.frameLength
        window = (0..<n).map { i in
            0.5 - 0.5 * cosf(2 * .pi * Float(i) / Float(n))
        }
        filters = Gemma4Mel.melFilterBank(cfg)
        log2n = vDSP_Length(log2(Double(cfg.fftLength)).rounded())
        setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    // HTK triangles, unnormalized: `norm=None` upstream peaks at 1, not scaled
    // by bandwidth.
    static func melFilterBank(_ cfg: Gemma4MelConfig) -> [Float] {
        let freqBins = cfg.fftLength / 2 + 1
        let m = cfg.bins
        func toMel(_ hz: Float) -> Float {
            2595 * log10f(1 + hz / 700)
        }
        func toHz(_ mel: Float) -> Float {
            700 * (powf(10, mel / 2595) - 1)
        }
        let loMel = toMel(cfg.minHz), hiMel = toMel(cfg.maxHz)
        let edges = (0..<(m + 2)).map { i in
            toHz(loMel + (hiMel - loMel) * Float(i) / Float(m + 1))
        }
        let nyquist = Float(cfg.sampleRate) / 2
        let fftFreqs = (0..<freqBins).map { i in
            nyquist * Float(i) / Float(freqBins - 1)
        }
        var out = [Float](repeating: 0, count: freqBins * m)
        for j in 0..<m {
            let lo = edges[j], mid = edges[j + 1], hi = edges[j + 2]
            for i in 0..<freqBins {
                let f = fftFreqs[i]
                let down = (f - lo) / (mid - lo)
                let up = (hi - f) / (hi - mid)
                out[i * m + j] = max(0, min(down, up))
            }
        }
        return out
    }

    public func features(_ samples: [Float])
        -> (values: [Float], frames: Int) {
        let cfg = self.cfg
        let pad = cfg.frameLength / 2
        var wave = [Float](repeating: 0, count: pad + samples.count)
        for i in 0..<samples.count { wave[pad + i] = samples[i] }
        // The extractor unfolds frame_length + 1 samples and drops the last
        // without preemphasis, so the usable span is one sample past a frame.
        let span = cfg.frameLength + 1
        let frames = wave.count >= span
            ? (wave.count - span) / cfg.hopLength + 1 : 0
        let freqBins = cfg.fftLength / 2 + 1
        var mag = [Float](repeating: 0, count: freqBins)
        var out = [Float](repeating: 0, count: frames * cfg.bins)
        for t in 0..<frames {
            spectrum(wave, t * cfg.hopLength, &mag)
            for j in 0..<cfg.bins {
                var acc: Float = 0
                for i in 0..<freqBins {
                    acc += mag[i] * filters[i * cfg.bins + j]
                }
                out[t * cfg.bins + j] = logf(acc + cfg.melFloor)
            }
        }
        return (out, frames)
    }

    // vDSP's packed real FFT puts DC in real[0], Nyquist in imag[0], scaled by
    // two.
    private func spectrum(_ wave: [Float], _ start: Int,
                          _ mag: inout [Float]) {
        let n = cfg.fftLength
        var padded = [Float](repeating: 0, count: n)
        for i in 0..<cfg.frameLength {
            padded[i] = wave[start + i] * window[i]
        }
        var real = [Float](repeating: 0, count: n / 2)
        var imag = [Float](repeating: 0, count: n / 2)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!,
                                            imagp: ip.baseAddress!)
                padded.withUnsafeBufferPointer { p in
                    p.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: n / 2) { c in
                        vDSP_ctoz(c, 2, &split, 1, vDSP_Length(n / 2))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n,
                              FFTDirection(FFT_FORWARD))
            }
        }
        mag[0] = abs(real[0]) / 2
        mag[n / 2] = abs(imag[0]) / 2
        for k in 1..<(n / 2) {
            mag[k] = (real[k] * real[k] + imag[k] * imag[k]).squareRoot() / 2
        }
    }
}
