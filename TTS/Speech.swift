import Foundation

public struct SpeechVoice: Sendable, Hashable, Identifiable {
    public let id: String
    public let name: String
    public let detail: String
}

public final class Speech {

    public static let sampleRate = TTS.sampleRate

    public static let voices: [SpeechVoice] = [
        SpeechVoice(id: "F1", name: "Sarah",
                    detail: "Calm, slightly low, steady and composed."),
        SpeechVoice(id: "F2", name: "Lily",
                    detail: "Bright and cheerful, lively and youthful."),
        SpeechVoice(id: "F3", name: "Jessica",
                    detail: "Clear and professional, an announcer."),
        SpeechVoice(id: "F4", name: "Olivia",
                    detail: "Crisp and confident, expressive."),
        SpeechVoice(id: "F5", name: "Emily",
                    detail: "Kind and gentle, soft-spoken and soothing."),
        SpeechVoice(id: "M1", name: "Alex",
                    detail: "Lively and upbeat, a standard clear tone."),
        SpeechVoice(id: "M2", name: "James",
                    detail: "Deep and robust, calm and serious."),
        SpeechVoice(id: "M3", name: "Robert",
                    detail: "Polished and authoritative, trustworthy."),
        SpeechVoice(id: "M4", name: "Sam",
                    detail: "Soft and neutral, friendly and youthful."),
        SpeechVoice(id: "M5", name: "Daniel",
                    detail: "Warm and soft-spoken, a storyteller."),
    ]

    public static let defaultVoice = voices[4]

    public static let languages: Set<String> = [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
        "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    ]

    public static func voice(named name: String) -> SpeechVoice? {
        let wanted = name.lowercased()
        return voices.first { v in
            v.name.lowercased() == wanted || v.id.lowercased() == wanted
        }
    }

    public static func pcm(_ sample: Float) -> Int16 { pcm16(sample) }

    static let pace: Float = 1.05
    static let flowSteps = 8
    static let noiseSeed: UInt32 = 0

    private let engine: Engine
    private let lock = NSLock()

    public init?(pack path: String) {
        if let opened = engineOpen(path) {
            engine = opened
        } else {
            return nil
        }
    }

    public func synthesize(_ text: String, voice: SpeechVoice? = nil,
                           speed: Float = 1.0,
                           language: String = "en") -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let picked = voice ?? Speech.defaultVoice
        let lang = Speech.languages.contains(language) ? language : "en"
        let voiced = text.unicodeScalars.contains { s in
            s.properties.isAlphabetic || s.properties.numericType != nil
        }
        var out = [Float]()
        if voiced, speed > 0 {
            out = narrate(engine, text, lang, picked.id, Speech.noiseSeed,
                          Speech.flowSteps, speed * Speech.pace)
        }
        return out
    }
}
