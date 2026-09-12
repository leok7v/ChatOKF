// The tap allocates on the real-time thread (one array per block), a known
// compromise with 4096 frames of slack. Never a model, GPU or await here.
@preconcurrency import AVFoundation
import Foundation

public final class Microphone: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case denied
        case unavailable(String)

        public var description: String {
            let out: String
            switch self {
            case .denied:
                out = "Microphone access was refused."
            case .unavailable(let why):
                out = "The microphone is unavailable: \(why)"
            }
            return out
        }
    }

    private let engine = AVAudioEngine()
    private let rate: Double
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private var running = false

    public init(rate: Double) { self.rate = rate }

    // Ask once, up front: a tap installed before the answer arrives records
    // silence on iOS and throws on macOS.
    public static func permission() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    public func start(_ onSamples: @escaping @Sendable ([Float]) -> Void)
        throws {
        lock.lock()
        defer { lock.unlock() }
        if !running {
            let input = engine.inputNode
            let native = input.outputFormat(forBus: 0)
            let made: AVAudioFormat? = native.sampleRate > 0
                ? AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: rate, channels: 1,
                                interleaved: false)
                : nil
            if made == nil {
                throw Failure.unavailable("no input format")
            }
            let want = made!
            converter = AVAudioConverter(from: native, to: want)
            input.installTap(onBus: 0, bufferSize: 4096, format: native) {
                [weak self] buffer, _ in
                if let s = self?.mono(buffer, to: want), !s.isEmpty {
                    onSamples(s)
                }
            }
            engine.prepare()
            do {
                try engine.start()
            } catch {
                input.removeTap(onBus: 0)
                throw Failure.unavailable(error.localizedDescription)
            }
            running = true
        }
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        if running {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            running = false
        }
    }

    // The block runs SYNCHRONOUSLY inside convert() on the audio thread and is
    // never shared, which is what makes @unchecked Sendable true.
    private final class OneShot: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?
        init(_ b: AVAudioPCMBuffer) { buffer = b }
        func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>)
            -> AVAudioPCMBuffer? {
            let out = buffer
            status.pointee = out == nil ? .noDataNow : .haveData
            buffer = nil
            return out
        }
    }

    private func mono(_ buffer: AVAudioPCMBuffer,
                      to want: AVAudioFormat) -> [Float] {
        var out: [Float] = []
        let ratio = want.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        if let converter,
           let dst = AVAudioPCMBuffer(pcmFormat: want, frameCapacity: cap) {
            // Hand the one buffer over on the first call and report starvation
            // after, or the converter loops asking for input that never comes.
            let source = OneShot(buffer)
            var err: NSError?
            converter.convert(to: dst, error: &err) { _, status in
                source.next(status)
            }
            if err == nil, let ch = dst.floatChannelData {
                out = Array(UnsafeBufferPointer(start: ch[0],
                                                count: Int(dst.frameLength)))
            }
        }
        return out
    }
}
