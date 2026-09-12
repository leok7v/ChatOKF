import AVFoundation
import Chat
import Foundation
import LLM

final class VoicePlayer: @unchecked Sendable {

    private static let readAhead = 2
    private static let primerSeconds = 0.25
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let synthQueue = DispatchQueue(label: "chatokf.tts.synth",
                                           qos: .userInitiated)
    private let lock = NSLock()

    // AVAudioPlayerNode.stop() blocks on the render thread, so transport runs
    // here, not on the main actor; without `qos:` a queue is UNSPECIFIED.
    private let nodeQueue = DispatchQueue(label: "chatokf.tts.node",
                                          qos: .default)
    private var speech: Speech?
    private var unsynthesized: [(text: String, tag: Int)] = []
    private var bufferedTags: [Int] = []
    private var synthesizing = 0
    private var buffered = 0
    private var epoch = 0
    private var running = false
    private var isPaused = false

    var onActivity: (@Sendable (Bool) -> Void)?
    var onSpeaking: (@Sendable (Int?) -> Void)?

    static let speechFloorGB = Flags.int("speech-floor") ?? 4

    init?() {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(Speech.sampleRate),
                                channels: 1, interleaved: false)
        if let fmt, installedGB >= VoicePlayer.speechFloorGB {
            format = fmt
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: fmt)
        } else {
            return nil
        }
    }

    private func ready() -> Speech? {
        lock.lock()
        let have = speech
        lock.unlock()
        var result = have
        if result == nil {
            let built = Speech()
            lock.lock()
            if speech == nil { speech = built }
            result = speech
            lock.unlock()
        }
        return result
    }

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return synthesizing > 0 || buffered > 0 || !unsynthesized.isEmpty
    }

    private func log(_ what: @autoclosure () -> String) {
        if DiagGate.voice.on { Diag.shared.report(.voice, "[tts] " + what()) }
    }

    private func counts() -> String {
        lock.lock()
        defer { lock.unlock() }
        return "synthesizing \(synthesizing) buffered \(buffered) "
            + "unsynthesized \(unsynthesized.count) epoch \(epoch) "
            + "paused \(isPaused) running \(running)"
    }

    func enqueue(_ text: String, tag: Int, voice: String, speed: Float) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            lock.lock()
            unsynthesized.append((trimmed, tag))
            lock.unlock()
            log("enqueue \(trimmed.count) chars | \(counts())")
            nodeQueue.async { [weak self] in self?.startEngine() }
            pump(voice: voice, speed: speed)
        }
    }

    func stop() {
        lock.lock()
        unsynthesized.removeAll()
        epoch += 1
        buffered = 0
        isPaused = false
        bufferedTags.removeAll()
        lock.unlock()
        nodeQueue.async { [weak self] in
            self?.node.stop()
            self?.node.reset()
        }
        log("stop | \(counts())")
        notify()
        notifySpeaking()
    }

    func pause() {
        lock.lock()
        isPaused = true
        lock.unlock()
        nodeQueue.async { [weak self] in self?.node.pause() }
        log("pause | \(counts())")
    }

    func resume() {
        lock.lock()
        isPaused = false
        lock.unlock()
        nodeQueue.async { [weak self] in
            self?.startEngine()
            self?.node.play()
        }
        log("resume | \(counts())")
    }

    private func pump(voice: String, speed: Float) {
        lock.lock()
        let room = synthesizing + buffered < VoicePlayer.readAhead
        let next = (room && !unsynthesized.isEmpty)
            ? unsynthesized.removeFirst() : nil
        if next != nil { synthesizing += 1 }
        let mark = epoch
        lock.unlock()
        if let next {
            notify()
            synthQueue.async { [weak self] in
                self?.render(next.text, tag: next.tag, voice: voice,
                             speed: speed, mark: mark)
            }
        }
    }

    private func render(_ text: String, tag: Int, voice: String,
                        speed: Float, mark: Int) {
        let picked = Speech.voice(named: voice)
        let t0 = Date()
        let pcm = ready()?.synthesize(text, voice: picked, speed: speed) ?? []
        lock.lock()
        synthesizing -= 1
        let stale = mark != epoch
        lock.unlock()
        log(String(format: "render %d chars -> %d samples in %.2fs, %@ | %@",
                   text.count, pcm.count, Date().timeIntervalSince(t0),
                   stale ? "STALE" : "live", counts()))
        if !stale && !pcm.isEmpty {
            schedule(pcm, tag: tag, mark: mark, voice: voice, speed: speed)
        } else {
            notify()
            pump(voice: voice, speed: speed)
        }
    }

    private func schedule(_ pcm: [Float], tag: Int, mark: Int, voice: String,
                          speed: Float) {
        let frames = AVAudioFrameCount(pcm.count)
        let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                      frameCapacity: frames)
        if let buffer, let channel = buffer.floatChannelData {
            buffer.frameLength = frames
            pcm.withUnsafeBufferPointer { src in
                channel[0].update(from: src.baseAddress!, count: pcm.count)
            }
            lock.lock()
            let live = mark == epoch
            if live { buffered += 1; bufferedTags.append(tag) }
            let paused = isPaused
            lock.unlock()
            if live {
                startEngine()
                node.scheduleBuffer(buffer, completionCallbackType:
                                        .dataPlayedBack) { [weak self] _ in
                    self?.finished(mark: mark, voice: voice, speed: speed)
                }
                if !paused { node.play() }
            }
            notify()
            notifySpeaking()
        }
        pump(voice: voice, speed: speed)
    }

    private func finished(mark: Int, voice: String, speed: Float) {
        lock.lock()
        if mark == epoch && buffered > 0 { buffered -= 1 }
        if mark == epoch && !bufferedTags.isEmpty {
            bufferedTags.removeFirst()
        }
        lock.unlock()
        notify()
        notifySpeaking()
        pump(voice: voice, speed: speed)
    }

    private func startEngine() {
        lock.lock()
        let already = running
        running = true
        lock.unlock()
        if !already {
            AudioSession.beginPlayback()
            try? engine.start()
            prime()
        }
    }

    private func prime() {
        let frames = AVAudioFrameCount(Double(Speech.sampleRate)
                                       * VoicePlayer.primerSeconds)
        let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                      frameCapacity: frames)
        lock.lock()
        let paused = isPaused
        lock.unlock()
        if let buffer, let channel = buffer.floatChannelData {
            buffer.frameLength = frames
            channel[0].update(repeating: 0, count: Int(frames))
            node.scheduleBuffer(buffer, completionHandler: nil)
            if !paused { node.play() }
        }
    }

    func idle() {
        lock.lock()
        let quiet = synthesizing == 0 && buffered == 0
            && unsynthesized.isEmpty
        let wasRunning = running
        if quiet { running = false }
        lock.unlock()
        if quiet && wasRunning {
            nodeQueue.async { [weak self] in
                self?.engine.pause()
                AudioSession.endPlayback()
            }
        }
    }

    private func notify() {
        onActivity?(isActive)
    }

    private func notifySpeaking() {
        lock.lock()
        let audible = bufferedTags.first
        lock.unlock()
        onSpeaking?(audible)
    }

}
