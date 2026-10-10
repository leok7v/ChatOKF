import AVFoundation
import Chat
import Foundation
import LLM
import TTS

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
    private var pending: [Scheduled] = []
    private var nextSerial = 0

    private struct Scheduled {
        let serial: Int
        let buffer: AVAudioPCMBuffer
        let mark: Int
        let voice: String
        let speed: Float
    }
    private var synthesizing = 0
    private var buffered = 0
    private var epoch = 0
    private var running = false
    private var isPaused = false

    var onActivity: (@Sendable (Bool) -> Void)?
    var onSpeaking: (@Sendable (Int?) -> Void)?
    private var holding = false
    private var demoFile: AVAudioFile?
    private var demoHost: UInt64 = 0
    private var demoWritten = 0.0
    private var demoConverter: AVAudioConverter?

    static let speechFloorGB = Flags.int("speech-floor") ?? 4

    init?() {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                sampleRate: Double(Speech.sampleRate),
                                channels: 1, interleaved: false)
        if let fmt, installedGB >= VoicePlayer.speechFloorGB {
            format = fmt
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: fmt)
            NotificationCenter.default.addObserver(
                forName: .AVAudioEngineConfigurationChange, object: engine,
                queue: nil) { [weak self] _ in self?.recover() }
        } else {
            return nil
        }
    }

    private func recover() {
        nodeQueue.async { [weak self] in
            if let self {
                self.lock.lock()
                let lost = self.pending
                let wasRunning = self.running
                let paused = self.isPaused
                self.pending = []
                self.running = false
                self.lock.unlock()
                self.log("engine reconfigured, \(lost.count) buffer(s) "
                         + "replayed")
                self.node.stop()
                self.node.reset()
                if wasRunning || !lost.isEmpty {
                    self.startEngine()
                    for item in lost {
                        self.scheduleKept(item.buffer, mark: item.mark,
                                          voice: item.voice, speed: item.speed)
                    }
                    if !paused, self.engine.isRunning { self.node.play() }
                }
            }
        }
    }

    private func scheduleKept(_ buffer: AVAudioPCMBuffer, mark: Int,
                              voice: String, speed: Float) {
        lock.lock()
        nextSerial += 1
        let serial = nextSerial
        pending.append(Scheduled(serial: serial, buffer: buffer, mark: mark,
                                 voice: voice, speed: speed))
        lock.unlock()
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) {
            [weak self] _ in
            self?.finished(serial: serial, mark: mark, voice: voice,
                           speed: speed)
        }
    }

    private func ready() -> Speech? {
        lock.lock()
        let have = speech
        lock.unlock()
        var result = have
        if result == nil {
            let built = VoicePack.path.flatMap { path in Speech(pack: path) }
            lock.lock()
            if speech == nil { speech = built }
            result = speech
            lock.unlock()
        }
        return result
    }

    func forgetPack() {
        lock.lock()
        speech = nil
        lock.unlock()
    }

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return synthesizing > 0 || buffered > 0 || !unsynthesized.isEmpty
    }

    var holdsOpen: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return holding
        }
        set {
            lock.lock()
            holding = newValue
            lock.unlock()
        }
    }

    private func log(_ what: @autoclosure () -> String) {
        if DiagGate.voice.on { Diag.shared.report(.voice, "[tts] " + what()) }
    }

    private static func memory() -> String {
        let mb = 1_048_576.0
        return Footprint.sample().map { s in
            String(format: "footprint %.0f MB, headroom %.0f MB",
                   Double(s.footprint) / mb,
                   Double(s.headroomToJetsam ?? 0) / mb)
        } ?? "footprint unknown"
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

    func warm() {
        nodeQueue.async { [weak self] in self?.startEngine() }
    }

    func endRecording() {
        node.removeTap(onBus: 0)
        engine.mainMixerNode.outputVolume = 1
        lock.lock()
        demoFile = nil
        lock.unlock()
    }

    func record(to url: URL, muted: Bool) {
        try? FileManager.default.removeItem(at: url)
        nodeQueue.async { [weak self] in
            if let self {
                self.startEngine()
                if muted { self.engine.mainMixerNode.outputVolume = 0 }
                let fmt = self.node.outputFormat(forBus: 0)
                self.lock.lock()
                self.demoFile = try? AVAudioFile(forWriting: url,
                                                 settings: fmt.settings)
                self.demoHost = mach_absolute_time()
                self.demoWritten = 0
                self.lock.unlock()
                Diag.shared.report(.turn, String(
                    format: "[demo] recording %.0f Hz %d ch, engine %@, "
                        + "speaker %@", fmt.sampleRate, fmt.channelCount,
                    self.engine.isRunning ? "running" : "STOPPED",
                    muted ? "muted" : "on"))
                self.node.installTap(onBus: 0, bufferSize: 4096, format: fmt) {
                    [weak self] buffer, when in
                    self?.demoRendered(buffer, at: when)
                }
            }
        }
    }

    private func converted(_ buffer: AVAudioPCMBuffer, to fmt: AVAudioFormat)
        -> AVAudioPCMBuffer? {
        var out: AVAudioPCMBuffer? = buffer
        if buffer.format != fmt {
            lock.lock()
            if demoConverter?.inputFormat != buffer.format {
                demoConverter = AVAudioConverter(from: buffer.format, to: fmt)
            }
            let converter = demoConverter
            lock.unlock()
            let ratio = fmt.sampleRate / buffer.format.sampleRate
            let frames = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
            out = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames + 32)
            var fed = false
            if let converter, let out {
                converter.convert(to: out, error: nil) { _, status in
                    status.pointee = fed ? .noDataNow : .haveData
                    let next: AVAudioBuffer? = fed ? nil : buffer
                    fed = true
                    return next
                }
            }
        }
        return out
    }

    private static func silence(_ seconds: Double, in fmt: AVAudioFormat)
        -> AVAudioPCMBuffer? {
        let frames = AVAudioFrameCount(max(0, seconds) * fmt.sampleRate)
        let buffer = frames > 0
            ? AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) : nil
        if let buffer, let channels = buffer.floatChannelData {
            buffer.frameLength = frames
            for c in 0..<Int(fmt.channelCount) {
                channels[c].update(repeating: 0, count: Int(frames))
            }
        }
        return buffer
    }

    private func demoRendered(_ buffer: AVAudioPCMBuffer,
                              at when: AVAudioTime) {
        lock.lock()
        let file = demoFile
        let began = when.isHostTimeValid
            ? AVAudioTime.seconds(forHostTime: when.hostTime)
                - AVAudioTime.seconds(forHostTime: demoHost)
            : demoWritten
        let gap = began - demoWritten
        let seconds = Double(buffer.frameLength) / buffer.format.sampleRate
        if file != nil { demoWritten = max(demoWritten, began) + seconds }
        lock.unlock()
        if let file {
            let fmt = file.processingFormat
            if gap > 0.02, let quiet = VoicePlayer.silence(gap, in: fmt) {
                try? file.write(from: quiet)
            }
            if let out = converted(buffer, to: fmt) {
                try? file.write(from: out)
            }
        }
    }

    func stop() {
        lock.lock()
        unsynthesized.removeAll()
        epoch += 1
        buffered = 0
        isPaused = false
        bufferedTags.removeAll()
        pending.removeAll()
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
            if let self {
                self.startEngine()
                if self.engine.isRunning { self.node.play() }
            }
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
        let whole = ready()?.synthesize(
            text, voice: picked, speed: speed,
            language: Flags.value("tts-lang") ?? "na",
            abandon: { [weak self] in self?.superseded(mark) ?? true }) ?? []
        let pcm = Speech.trimmed(whole)
        lock.lock()
        synthesizing -= 1
        let stale = mark != epoch
        lock.unlock()
        log(String(format: "render %d chars -> %d samples of %d in %.2fs, "
                   + "%@, %@ | %@", text.count, pcm.count, whole.count,
                   Date().timeIntervalSince(t0), stale ? "STALE" : "live",
                   VoicePlayer.memory(), counts()))
        if !stale && !pcm.isEmpty {
            schedule(pcm, tag: tag, mark: mark, voice: voice, speed: speed)
        } else {
            notify()
            pump(voice: voice, speed: speed)
        }
    }

    private func superseded(_ mark: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return mark != epoch
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
                nodeQueue.sync { startEngine() }
                scheduleKept(buffer, mark: mark, voice: voice, speed: speed)
                if !paused, engine.isRunning { node.play() }
            }
            notify()
            notifySpeaking()
        }
        pump(voice: voice, speed: speed)
    }

    private func finished(serial: Int, mark: Int, voice: String,
                          speed: Float) {
        lock.lock()
        let played = pending.firstIndex { item in item.serial == serial }
        if let played {
            pending.remove(at: played)
            if mark == epoch && buffered > 0 { buffered -= 1 }
            if mark == epoch && !bufferedTags.isEmpty {
                bufferedTags.removeFirst()
            }
        }
        lock.unlock()
        notify()
        notifySpeaking()
        pump(voice: voice, speed: speed)
    }

    private func startEngine() {
        lock.lock()
        let already = running && engine.isRunning
        running = true
        lock.unlock()
        if !already {
            AudioSession.beginPlayback()
            if (try? engine.start()) != nil {
                prime()
            } else {
                lock.lock()
                running = false
                lock.unlock()
                log("engine did not start | \(counts())")
            }
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
            && unsynthesized.isEmpty && !holding
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
