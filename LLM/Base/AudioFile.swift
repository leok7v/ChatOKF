// AVAssetReader, not AVAudioFile: the source may be a VIDEO container, and
// the reader resamples and downmixes itself through its output settings.
import AVFoundation
import Foundation

public enum AudioFile {
    public enum Failure: Error {
        case noAudioTrack(String)
        case unreadable(String)
    }

    public static func samples(url: URL, sampleRate: Double,
                               offset: Double = 0,
                               seconds: Double? = nil) async throws
        -> [Float] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        var out: [Float] = []
        if tracks.isEmpty {
            throw Failure.noAudioTrack(url.lastPathComponent)
        } else {
            let reader = try AVAssetReader(asset: asset)
            let scale = CMTimeScale(sampleRate)
            let from = CMTime(seconds: max(0, offset), preferredTimescale: scale)
            let span = seconds.map { s in
                CMTime(seconds: max(0, s), preferredTimescale: scale)
            } ?? CMTime.positiveInfinity
            reader.timeRange = CMTimeRange(start: from, duration: span)
            let output = AVAssetReaderAudioMixOutput(
                audioTracks: tracks, audioSettings: settings(sampleRate))
            reader.add(output)
            reader.startReading()
            out = drain(output)
            if reader.status == .failed {
                throw Failure.unreadable(
                    reader.error?.localizedDescription ?? "read failed")
            }
        }
        return out
    }

    private static func settings(_ rate: Double) -> [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM,
         AVSampleRateKey: rate,
         AVNumberOfChannelsKey: 1,
         AVLinearPCMBitDepthKey: 32,
         AVLinearPCMIsFloatKey: true,
         AVLinearPCMIsBigEndianKey: false,
         AVLinearPCMIsNonInterleaved: false]
    }

    private static func drain(_ output: AVAssetReaderOutput) -> [Float] {
        var out: [Float] = []
        var sample = output.copyNextSampleBuffer()
        while sample != nil {
            if let block = CMSampleBufferGetDataBuffer(sample!) {
                let n = CMBlockBufferGetDataLength(block)
                var bytes = [UInt8](repeating: 0, count: n)
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: n,
                                           destination: &bytes)
                bytes.withUnsafeBytes { raw in
                    let f = raw.bindMemory(to: Float.self)
                    out.append(contentsOf: f)
                }
            }
            sample = output.copyNextSampleBuffer()
        }
        return out
    }
}
