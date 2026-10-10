import AVFoundation
import Chat
import ReplayKit

final class DemoCapture: @unchecked Sendable {

    static let shared = DemoCapture()

    private let lock = NSLock()
    private var writer: AVAssetWriter?
    private var video: AVAssetWriterInput?
    private var audio: AVAssetWriterInput?
    private var url: URL?
    private var frames = 0
    private var audioBuffers = 0
    private var refused = false

    var waiting: Bool {
        lock.lock()
        defer { lock.unlock() }
        return frames == 0 && !refused
    }

    func start(to url: URL, done: @escaping @Sendable (String?) -> Void) {
        try? FileManager.default.removeItem(at: url)
        lock.lock()
        self.url = url
        lock.unlock()
        let recorder = RPScreenRecorder.shared()
        recorder.isMicrophoneEnabled = false
        recorder.startCapture(handler: { [weak self] sample, kind, _ in
            self?.append(sample, kind)
        }, completionHandler: { [weak self] error in
            if error != nil { self?.refuse() }
            done(error?.localizedDescription)
        })
    }

    private func refuse() {
        lock.lock()
        refused = true
        lock.unlock()
    }

    func stop(done: @escaping @Sendable (String) -> Void) {
        RPScreenRecorder.shared().stopCapture { [weak self] error in
            self?.finish(error?.localizedDescription, done: done)
        }
    }

    private func finish(_ error: String?, done: @escaping @Sendable (String)
                        -> Void) {
        lock.lock()
        let writer = writer
        let video = video
        let audio = audio
        let frames = frames
        let audioBuffers = audioBuffers
        lock.unlock()
        video?.markAsFinished()
        audio?.markAsFinished()
        if let writer {
            writer.finishWriting { [self] in
                done("capture: \(frames) frames, \(audioBuffers) audio "
                     + "buffers, " + (failure() ?? error ?? "written"))
            }
        } else {
            done("capture: nothing written, " + (error ?? "no samples"))
        }
    }

    private func failure() -> String? {
        lock.lock()
        defer { lock.unlock() }
        let text = writer?.error.map { e in "\(e)" }
        writer = nil
        return text
    }

    private func append(_ sample: CMSampleBuffer, _ kind: RPSampleBufferType) {
        lock.lock()
        if writer == nil, kind == .video, CMSampleBufferDataIsReady(sample) {
            open(sample)
        }
        let input = kind == .video ? video : (kind == .audioApp ? audio : nil)
        let ready = writer?.status == .writing
            && input?.isReadyForMoreMediaData == true
        if ready, let input, input.append(sample) {
            if kind == .video { frames += 1 } else { audioBuffers += 1 }
        }
        lock.unlock()
    }

    private func open(_ first: CMSampleBuffer) {
        if let url, let format = CMSampleBufferGetFormatDescription(first),
           let made = try? AVAssetWriter(outputURL: url, fileType: .mp4) {
            let size = CMVideoFormatDescriptionGetDimensions(format)
            let v = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: Int(size.width),
                AVVideoHeightKey: Int(size.height),
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 12_000_000,
                    AVVideoProfileLevelKey:
                        AVVideoProfileLevelH264HighAutoLevel,
                ],
            ])
            v.expectsMediaDataInRealTime = true
            let a = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 160_000,
            ])
            a.expectsMediaDataInRealTime = true
            if made.canAdd(v) { made.add(v) }
            if made.canAdd(a) { made.add(a) }
            if made.startWriting() {
                let at = CMSampleBufferGetPresentationTimeStamp(first)
                made.startSession(atSourceTime: at)
                writer = made
                video = v
                audio = a
            }
        }
    }

}
