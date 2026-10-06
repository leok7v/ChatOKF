import Foundation

public struct JudgeItem: Decodable, Sendable {
    public let text: String
    public let gold: String
}

public struct JudgeTask: Decodable, Sendable {
    public let name: String
    public let system: String?
    public let frame: String
    public let options: [String]
    public let items: [JudgeItem]
}

final class LogitTap: @unchecked Sendable {
    private let lock = NSLock()
    private var kept: [Float] = []

    func keep(_ logits: [Float]) {
        lock.lock()
        kept = logits
        lock.unlock()
    }

    var logits: [Float] {
        lock.lock()
        defer { lock.unlock() }
        return kept
    }
}

public struct Judge: Sendable {

    let backend: any AgentBackend
    let template: String
    let vocabSize: Int

    public init(backend: any AgentBackend, template: String, vocabSize: Int) {
        self.backend = backend
        self.template = template
        self.vocabSize = vocabSize
    }

    static let sentinel = "ZqItemZq"

    static func spellings(_ option: String) -> [String] {
        let forms = [option, option.lowercased(), option.capitalized,
                     option.uppercased()]
        return forms + forms.map { form in " " + form }
    }

    func ids(_ option: String) -> Set<Int32> {
        var out: Set<Int32> = []
        for spelling in Judge.spellings(option) {
            if let first = backend.encode(spelling).first { out.insert(first) }
        }
        return out
    }

    static func shares(_ logits: [Float], _ ids: [Set<Int32>]) -> [Double] {
        let top = logits.max() ?? 0
        var total = 0.0
        for value in logits { total += exp(Double(value - top)) }
        return ids.map { set in
            var sum = 0.0
            for id in set where Int(id) < logits.count {
                sum += exp(Double(logits[Int(id)] - top))
            }
            return total > 0 ? sum / total : 0
        }
    }

    func prompt(_ task: JudgeTask, _ item: String) -> String {
        var messages: [AgentMessage] = []
        if let system = task.system, !system.isEmpty {
            messages.append(AgentMessage(role: "system", content: system))
        }
        messages.append(AgentMessage(
            role: "user",
            content: task.frame.replacingOccurrences(of: "{item}",
                                                     with: item)))
        return (try? renderPrompt(
            template: template, messages: messages, tools: [],
            addGenerationPrompt: true, enableThinking: false,
            bosToken: backend.bosToken)) ?? ""
    }

    static func common(_ a: [Int32], _ b: [Int32]) -> Int {
        var n = 0
        while n < a.count && n < b.count && a[n] == b[n] { n += 1 }
        return n
    }

    func run(_ task: JudgeTask, _ tap: LogitTap,
             _ emit: @Sendable (String) -> Void) async throws {
        let ids = task.options.map { option in self.ids(option) }
        let probe = backend.encode(prompt(task, Judge.sentinel))
        let first = backend.encode(prompt(task, task.items.first?.text ?? ""))
        let head = Array(probe.prefix(Judge.common(probe, first)))
        await backend.reset()
        if !head.isEmpty { _ = try await backend.extend(head) }
        try await backend.mark()
        for (index, item) in task.items.enumerated() {
            let full = backend.encode(prompt(task, item.text))
            let shared = Judge.common(head, full) == head.count
            let began = Date()
            if shared {
                try await backend.rewind()
                _ = try await backend.extend(Array(full[head.count...]))
            } else {
                await backend.reset()
                _ = try await backend.extend(full)
            }
            let ms = Date().timeIntervalSince(began) * 1000
            let logits = tap.logits
            let best = logits.indices.max { a, b in
                logits[a] < logits[b]
            } ?? 0
            let row: [String: Any] = [
                "task": task.name, "index": index, "gold": item.gold,
                "text": item.text, "options": task.options,
                "shares": Judge.shares(logits, ids), "ms": ms,
                "laid": shared ? full.count - head.count : full.count,
                "head": shared ? head.count : 0,
                "top": backend.text([Int32(best)]),
            ]
            if let data = try? JSONSerialization.data(withJSONObject: row) {
                emit(String(decoding: data, as: UTF8.self))
            }
        }
    }

    public func run(file: URL,
                    _ emit: @Sendable (String) -> Void) async throws {
        let tasks = try JSONDecoder().decode(
            [JudgeTask].self, from: Data(contentsOf: file))
        let tap = LogitTap()
        var sampler = Sampler(vocabSize: vocabSize, config: .greedy)
        sampler.logitMask = { logits in tap.keep(logits) }
        await backend.useSampler(sampler)
        for task in tasks { try await run(task, tap, emit) }
        await backend.useSampler(nil)
        await backend.reset()
    }
}
