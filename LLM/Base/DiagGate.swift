import Foundation

public enum DiagGate: String, CaseIterable, Identifiable, Sendable {

    case fault
    case load
    case net
    case turn
    case tools
    case voice
    case perf
    case memory
    case pressure
    case transcript

    public var id: String { rawValue }

    public static let switchable: [DiagGate] =
        allCases.filter { gate in gate != .fault }

    public var label: String {
        let out: String
        switch self {
        case .fault: out = "Faults"
        case .load: out = "Startup and models"
        case .net: out = "Downloads"
        case .turn: out = "Turns"
        case .tools: out = "Tools and web"
        case .voice: out = "Microphone and speech"
        case .perf: out = "Stalls and timing"
        case .memory: out = "Memory detail"
        case .pressure: out = "Memory pressure"
        case .transcript: out = "Session transcript"
        }
        return out
    }

    public var detail: String {
        let out: String
        switch self {
        case .fault:
            out = "Errors that stop something working. Always recorded."
        case .load:
            out = "What the app found at launch, and each model as it is "
                + "prepared, primed and released."
        case .net:
            out = "Each piece of a model download as it lands."
        case .turn:
            out = "One line per reply, with the token counts and speeds, "
                + "and a line per attachment."
        case .tools:
            out = "Every tool the assistant calls, what it asked for, and "
                + "what came back."
        case .voice:
            out = "What the microphone hears and what the reading voice is "
                + "doing."
        case .perf:
            out = "Moments the app freezes or a screen redraw runs long, "
                + "and how fast the graphics work finishes."
        case .memory:
            out = "Record memory use step by step during downloads and "
                + "while a prompt is processed, so a climb shows as a curve."
        case .pressure:
            out = "Log the system's low memory warnings."
        case .transcript:
            out = "Save everything sent to and received from the model, "
                + "word for word, in a local log file. Starts at the next "
                + "launch."
        }
        return out
    }

    public var flag: String { "log-\(rawValue)" }

    static let ladder: [[DiagGate]] = [
        [],
        [.load, .turn],
        [.load, .turn, .net, .tools, .voice],
        [.load, .turn, .net, .tools, .voice, .perf, .memory, .pressure,
         .transcript],
    ]

    static let asked: String = Flags.value("diagnostics") ?? ""

    public static let masterKey = "statusLine"

    public static let debug: Bool = Flags.on("debug")
        || (Flags.int("verbosity") ?? 0) > 0
        || !asked.isEmpty
        || UserDefaults.standard.bool(forKey: DiagGate.masterKey)

    public static let verbosity: Int = {
        let given = Flags.int("verbosity") ?? (DiagGate.debug ? 1 : 0)
        return max(0, min(given, DiagGate.ladder.count - 1))
    }()

    static let chosen: [DiagGate] = {
        let named = asked.split(separator: ",").compactMap { name in
            DiagGate(rawValue: name.trimmingCharacters(in: .whitespaces))
        }
        return named.isEmpty ? ladder[verbosity] : named
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var answers: [String: Bool] = [:]

    public var wanted: Bool {
        DiagGate.lock.lock()
        defer { DiagGate.lock.unlock() }
        if DiagGate.answers[rawValue] == nil {
            let text = Flags.value(flag) ?? ""
            DiagGate.answers[rawValue] = text.isEmpty
                ? DiagGate.chosen.contains(self)
                : Flags.truthy(text)
        }
        return DiagGate.answers[rawValue] ?? false
    }

    public var on: Bool {
        self == .fault || (DiagGate.debug && wanted)
    }

    public func set(_ value: Bool) {
        DiagGate.lock.lock()
        DiagGate.answers[rawValue] = value
        DiagGate.lock.unlock()
        Flags.store(flag, value)
    }

}
