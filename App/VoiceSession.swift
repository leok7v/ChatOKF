import Foundation
import LLM
import Observation

@MainActor @Observable final class VoiceSession {

    enum Mode: String, CaseIterable, Identifiable, Sendable {

        case off, replies, everything

        var id: String { rawValue }

        var label: String {
            switch self {
            case .off: return "Off"
            case .replies: return "Replies"
            case .everything: return "Everything"
            }
        }

        var detail: String {
            switch self {
            case .off:
                return "Replies are shown, never spoken."
            case .replies:
                return "The answer is read aloud. Thinking stays silent, so "
                    + "a long think is a pause."
            case .everything:
                return "The thinking is read as it happens, then the "
                    + "answer, so a long think is something to listen to "
                    + "rather than a silence."
            }
        }
    }

    var mode: Mode = VoiceSession.startMode() {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: "voiceMode")
            if mode != .off {
                UserDefaults.standard.set(mode.rawValue, forKey: "voiceLast")
            }
            if mode == .off { stopSpeaking() }
        }
    }

    private static func startMode() -> Mode {
        let d = UserDefaults.standard
        var result = Mode(rawValue: d.string(forKey: "voiceMode") ?? "")
        if result == nil {
            let spoken = d.bool(forKey: "speakReplies")
            let reasoning = d.object(forKey: "speakReasoning") as? Bool ?? false
            result = spoken ? (reasoning ? .everything : .replies) : .off
        }
        return result!
    }

    var enabled: Bool {
        get { mode != .off }
        set {
            if newValue {
                let last = UserDefaults.standard.string(forKey: "voiceLast")
                mode = Mode(rawValue: last ?? "") ?? .replies
            } else {
                mode = .off
            }
        }
    }

    var speakReasoning: Bool { mode == .everything }

    var voiceName: String = UserDefaults.standard
        .string(forKey: "voiceName") ?? Speech.defaultVoice.name {
        didSet { UserDefaults.standard.set(voiceName, forKey: "voiceName") }
    }

    var speed: Double = {
        let v = UserDefaults.standard.double(forKey: "voiceSpeed")
        return v > 0 ? v : 1.0
    }() {
        didSet { UserDefaults.standard.set(speed, forKey: "voiceSpeed") }
    }

    private(set) var speaking = false
    private(set) var paused = false
    private(set) var spokenText: String?

    private(set) var engaged = false
    @ObservationIgnored private var generating = false
    @ObservationIgnored private var turnSilenced = false

    @ObservationIgnored private let player = VoicePlayer()
    @ObservationIgnored private var answer = SpeakableText()
    @ObservationIgnored private var reasoning = SpeakableText()
    @ObservationIgnored private var cueTask: Task<Void, Never>?
    @ObservationIgnored private var cued = false
    private static let cueDelay = 3.5

    @ObservationIgnored private var answerByTag: [Int: String] = [:]
    @ObservationIgnored private var nextTag = 0

    init() {
        player?.onActivity = { [weak self] _ in
            Task { @MainActor in
                let active = self?.player?.isActive ?? false
                self?.speaking = active
                if active { self?.engaged = true }
                if !active {
                    self?.player?.idle()
                    self?.settle()
                }
            }
        }
        player?.onSpeaking = { [weak self] tag in
            Task { @MainActor in
                self?.spokenText = tag.flatMap { t in self?.answerByTag[t] }
            }
        }
    }

    private func settle() {
        if !generating && !(player?.isActive ?? false) { engaged = false }
    }

    var available: Bool { player != nil }

    func beginTurn(cue: SpokenCue.Kind = .thinking) {
        cueKind = cue
        stopSpeaking()
        answer = SpeakableText()
        reasoning = SpeakableText()
        answerByTag = [:]
        turnSilenced = false
        if enabled {
            engaged = true
            generating = true
        }
        armCue()
    }

    func answerArrived(_ piece: String) {
        disarmCue()
        if enabled {
            for segment in answer.push(piece) {
                say(segment.spoken, shown: segment.shown)
            }
        }
    }

    func reasoningArrived(_ piece: String) {
        disarmCue()
        if enabled && speakReasoning {
            for segment in reasoning.push(piece) { say(segment.spoken) }
        }
    }

    func endTurn() {
        disarmCue()
        if enabled {
            for segment in reasoning.finish() { say(segment.spoken) }
            for segment in answer.finish() {
                say(segment.spoken, shown: segment.shown)
            }
        }
        answer = SpeakableText()
        reasoning = SpeakableText()
        generating = false
        settle()
    }

    @ObservationIgnored private var cueKind: SpokenCue.Kind = .thinking

    private func armCue() {
        cueTask?.cancel()
        cued = false
        cueTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(VoiceSession.cueDelay))
            if !Task.isCancelled && enabled && !cued {
                cued = true
                say(SpokenCue.phrase(cueKind))
            }
        }
    }

    private func disarmCue() {
        cueTask?.cancel()
        cueTask = nil
    }

    private static let toolCueDelay = 1.2

    func toolStarted(_ name: String) {
        cueTask?.cancel()
        cueTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(VoiceSession.toolCueDelay))
            if !Task.isCancelled && enabled {
                say(SpokenCue.forTool(name))
            }
        }
    }

    func toolFinished() {
        disarmCue()
    }

    func stopSpeaking() {
        disarmCue()
        player?.stop()
        paused = false
        speaking = false
        spokenText = nil
        generating = false
        engaged = false
        turnSilenced = true
    }

    private func say(_ text: String, shown: String? = nil) {
        if !turnSilenced {
            let tag = nextTag
            nextTag += 1
            if let shown {
                answerByTag[tag] = shown
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            player?.enqueue(terminated(text), tag: tag, voice: voiceName,
                            speed: Float(speed))
        }
    }

    // The synthesizer renders the last phoneme's release off the terminal
    // punctuation; unterminated text comes back cut short.

    private func terminated(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let closed = trimmed.last.map { last in
            last == "." || last == "!" || last == "?"
        } ?? true
        return closed ? trimmed : trimmed + "."
    }

    func pause() {
        player?.pause()
        paused = true
    }

    func resume() {
        player?.resume()
        paused = false
    }

    func preview(_ voice: SpeechVoice) {
        player?.stop()
        player?.enqueue(SpokenCue.preview, tag: -1, voice: voice.name,
                        speed: Float(speed))
    }
}

enum SpokenCue {

    enum Kind { case thinking, consulting, looking, watching, reading }

    static let preview = "This is how I will read your replies."

    private static let thinking = [
        "Let me think about that.",
        "One moment.",
        "Thinking that through.",
    ]

    private static let consulting = [
        "Looking that up.",
        "Let me check a source.",
        "Just a moment while I look.",
    ]

    private static let looking = [
        "I am studying your picture with a magnifying glass.",
        "Let me take a good look at that.",
        "I need a moment to take this in.",
    ]

    private static let watching = [
        "I am looking at your video frame by frame.",
        "Watching this through.",
        "Give me a moment with the footage.",
    ]

    private static let reading = [
        "I need a moment to read what you attached.",
        "Let me look at what is inside that.",
        "Reading through it now.",
    ]

    private static let byTool: [String: String] = [
        "calculator": "Let me work that out.",
        "get_current_time": "Let me check the time.",
        "web_search": "Let me search the web.",
        "fetch_url": "Let me open that page.",
        "get_news": "Let me check the headlines.",
        "get_weather": "Let me check the forecast.",
        "wikipedia_query": "Let me see what Wikipedia says.",
    ]

    @MainActor static func forTool(_ name: String) -> String {
        byTool[name] ?? phrase(.consulting)
    }

    nonisolated(unsafe) private static var turn = 0

    @MainActor static func phrase(_ kind: Kind) -> String {
        let list: [String]
        switch kind {
        case .thinking: list = thinking
        case .consulting: list = consulting
        case .looking: list = looking
        case .watching: list = watching
        case .reading: list = reading
        }
        let pick = list[turn % list.count]
        turn += 1
        return pick
    }

}
