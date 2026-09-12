import Foundation

public enum Whimsical {

    public enum Stage: String {
        case reasoning, prefill, documents, vision, consulting,
             downloading
        case listening, mulling
        case heard
    }

    static let reasoning = load("reasoning").shuffled()
    static let prefill = load("prefill").shuffled()
    static let documents = load("documents").shuffled()
    static let vision = load("vision").shuffled()
    static let consulting = load("consulting").shuffled()
    static let downloading = load("downloading").shuffled()
    static let listening = load("listening").shuffled()
    static let mulling = load("mulling").shuffled()
    static let heard = load("heard").shuffled()

    static func list(_ stage: Stage) -> [String] {
        let result: [String]
        switch stage {
        case .reasoning: result = reasoning
        case .prefill: result = prefill
        case .documents: result = documents
        case .vision: result = vision
        case .consulting: result = consulting
        case .downloading: result = downloading
        case .listening: result = listening
        case .mulling: result = mulling
        case .heard: result = heard
        }
        return result
    }

    private static func load(_ name: String) -> [String] {
        let url = Bundle.main.url(forResource: name, withExtension: "txt")
        let text = url.flatMap { u in
            try? String(contentsOf: u, encoding: .utf8)
        }
        var seen: Set<String> = []
        var out: [String] = []
        for line in (text ?? "").split(whereSeparator: \.isNewline) {
            let phrase = line.trimmingCharacters(in: .whitespaces)
            if !phrase.isEmpty, seen.insert(phrase).inserted {
                out.append(phrase)
            }
        }
        return out.isEmpty ? ["Thinking"] : out
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cursors: [Stage: (Int, Int)] = [:]
    nonisolated(unsafe) private static var held:
        [Stage: (idx: Int, at: Date)] = [:]

    public static func current(_ stage: Stage,
                        hold: TimeInterval = 7) -> String {
        let all = list(stage)
        var result = "Thinking"
        if !all.isEmpty {
            lock.lock()
            var (idx, at) = held[stage] ?? (0, Date())
            if Date().timeIntervalSince(at) >= hold {
                idx = (idx + 1) % all.count
                at = Date()
            }
            held[stage] = (idx, at)
            result = all[idx]
            lock.unlock()
        }
        return result
    }

    public static func pair(_ stage: Stage) -> (first: String, second: String) {
        let all = list(stage)
        let n = all.count
        var result = (first: "Thinking", second: "Thinking")
        if n > 0 {
            lock.lock()
            var (up, down) = cursors[stage] ?? (0, n - 1)
            if up == down { down = (down + n - 1) % n }
            result = (first: all[up], second: all[down])
            cursors[stage] = ((up + 1) % n, (down + n - 1) % n)
            lock.unlock()
        }
        return result
    }

}
