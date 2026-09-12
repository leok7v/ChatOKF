import Foundation
import LLM

private final class Reasoning: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""

    func add(_ piece: String) {
        lock.lock()
        text += piece
        lock.unlock()
    }

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return text
    }
}

func runPuzzleGate(_ args: CommandArgs) async {
    let label = args.value("--puzzle-gate") ?? "gate"
    let json = args.value("--json") ?? ""
    let model = args.rest.first ?? ""
    let cap = capVal ?? 4096
    var status: Int32 = 0
    if !isModelFile(model) {
        err("usage: chatokf-cli <model.gguf> --puzzle-gate [label] "
            + "[--json out.json], -n caps the turn\n")
        status = 2
    } else {
        do {
            let runs = try await puzzleRuns(model, cap)
            let (passed, stalled) = PuzzleGate.score(runs)
            let seconds = runs.reduce(0) { total, r in total + r.seconds }
            err("\nGATE \(label)  \(passed)/\(runs.count) pass, "
                + "\(stalled) incomplete  [\(seconds)s]\n")
            if !json.isEmpty {
                let blob = try JSONBytes.reproducible(runs)
                try blob.write(to: URL(fileURLWithPath: json))
                err("[puzzle] wrote \(json)\n")
            }
        } catch {
            err("[puzzle] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runPuzzleRescore(_ args: CommandArgs) {
    let path = args.value("--puzzle-rescore") ?? ""
    var status: Int32 = 0
    if path.isEmpty {
        err("usage: chatokf-cli x --puzzle-rescore <runs.json>\n")
        status = 2
    } else {
        do {
            let blob = try Data(contentsOf: URL(fileURLWithPath: path))
            let saved = try JSONDecoder().decode([PuzzleRun].self, from: blob)
            let byPrompt = Dictionary(
                uniqueKeysWithValues: PuzzleGate.puzzles.map { puzzle in
                    (puzzle.prompt, puzzle)
                })
            var runs: [PuzzleRun] = []
            for var run in saved {
                if let puzzle = byPrompt[run.prompt] {
                    run.verdict = PuzzleGate.verdict(puzzle,
                                                     content: run.content)
                }
                report(run)
                runs.append(run)
            }
            let (passed, stalled) = PuzzleGate.score(runs)
            err("\n\(passed)/\(runs.count) pass, \(stalled) incomplete\n")
        } catch {
            err("[puzzle] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

private func report(_ run: PuzzleRun) {
    let head = run.prompt.prefix(44)
    err(String(format: "  %-4@ %-9@ %4d s  %@\n", run.verdict.rawValue,
               run.category, run.seconds, String(head)))
}

private func puzzleRuns(_ model: String, _ cap: Int) async throws
    -> [PuzzleRun] {
    err("loading GGUF \(model)...\n")
    let chat = try QwenMetalChat(ggufPath: model)
    let session = ChatSession(
        backend: chat.backend(), template: chat.chatTemplate,
        system: "You are a helpful assistant.",
        vocabSize: chat.tokenizer.vocabCount,
        presets: SamplingPresets.greedy, enableThinking: true,
        reasoningEffort: "on", maxTokens: cap)
    err("ready (Metal/GPU; vocab \(chat.tokenizer.vocabCount)).\n")
    var runs: [PuzzleRun] = []
    for puzzle in PuzzleGate.puzzles {
        let started = Date()
        await session.reset()
        let reasoning = Reasoning()
        var content = ""
        for await piece in session.reply(puzzle.prompt,
                                         onReasoning: { r in
                                             reasoning.add(r)
                                         }) {
            content += piece
        }
        var run = PuzzleRun()
        run.category = puzzle.category
        run.prompt = puzzle.prompt
        run.verdict = PuzzleGate.verdict(puzzle, content: content)
        run.seconds = Int(Date().timeIntervalSince(started))
        run.reasoning = reasoning.value
        run.content = content
        report(run)
        runs.append(run)
    }
    return runs
}
