import Foundation
import LLM

private final class Reasoning: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    private var pieces = 0

    func add(_ piece: String) -> Int {
        lock.lock()
        text += piece
        pieces += 1
        let count = pieces
        lock.unlock()
        return count
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
            + "[--json out.json], -n caps the turn, --quick-after N "
            + "presses Quick Answer after N reasoning pieces\n")
        status = 2
    } else {
        do {
            let runs = try await puzzleRuns(model, cap,
                                            quickAfterVal ?? cap)
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

private func puzzleRuns(_ model: String, _ cap: Int,
                        _ thinkCap: Int) async throws -> [PuzzleRun] {
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
            if reasoning.add(r) == thinkCap {
                err("  [quick] Quick Answer pressed\n")
                Task { await session.requestQuickAnswer() }
            }
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

struct RecallRNG {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func below(_ n: Int) -> Int {
        Int(next() % UInt64(max(n, 1)))
    }
}

enum RecallBench {
    nonisolated(unsafe) static var names: [String] = []
    nonisolated(unsafe) static var filler: [String] = []
    nonisolated(unsafe) static var things: [String] = []
    static let system = "I will tell you what people have. Reply 👍 to "
        + "facts. When I ask what someone has, reply with one word: the "
        + "thing."

    struct Fact {
        let key: String
        let thing: String
        let turn: Int
    }

    struct Turn {
        let facts: String
        let question: String
        let asked: Fact
        let back: Int
    }

    static func article(_ w: String) -> String {
        "aeiou".contains(w.first ?? "x") ? "an \(w)" : "a \(w)"
    }

    nonisolated(unsafe) static var fill = 8

    static func turns(_ count: Int, per: Int, seed: UInt64) -> [Turn] {
        var rng = RecallRNG(state: seed)
        var keys = names
        for i in stride(from: keys.count - 1, to: 0, by: -1) {
            keys.swapAt(i, rng.below(i + 1))
        }
        var stated: [String] = []
        var current: [String: String] = [:]
        var last: [String: Int] = [:]
        var out: [Turn] = []
        for t in 1 ... max(count, 1) {
            var said: [String] = []
            for _ in 0 ..< per {
                let key = stated.count < keys.count
                    ? keys[stated.count] : stated[rng.below(stated.count)]
                if current[key] == nil { stated.append(key) }
                let free = things.filter { w in w != current[key] }
                let thing = free[rng.below(free.count)]
                current[key] = thing
                last[key] = t
                said.append("\(key) has \(article(thing)).")
            }
            for _ in 0 ..< fill {
                said.append(filler[rng.below(filler.count)])
            }
            let want = rng.below(t)
            let start = rng.below(stated.count)
            let order = (0 ..< stated.count).map { i in
                stated[(start + i) % stated.count]
            }
            let key = order.min { x, y in
                abs(t - (last[x] ?? t) - want) < abs(t - (last[y] ?? t) - want)
            } ?? order[0]
            let asked = Fact(key: key, thing: current[key] ?? "",
                             turn: last[key] ?? t)
            last[key] = t
            out.append(Turn(facts: said.joined(separator: " ")
                                + " Reply 👍.",
                            question: ask(asked).trimmingCharacters(
                                in: .whitespaces),
                            asked: asked, back: t - asked.turn))
        }
        return out
    }

    nonisolated(unsafe) static var style = "cloze"

    static let words = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("words").path

    static func list(_ dir: String, _ name: String) throws -> [String] {
        let path = dir + "/" + name + ".txt"
        let out = try String(contentsOfFile: path, encoding: .utf8)
            .split(separator: "\n").map(String.init)
            .filter { line in !line.isEmpty }
        if out.isEmpty {
            throw CocoaError(.fileReadCorruptFile,
                             userInfo: [NSFilePathErrorKey: path])
        }
        return out
    }

    static func seed(_ key: String) -> String {
        var out = ""
        if style == "has" {
            out = "\(key) has a"
        } else if style == "thing" {
            out = "\(key)'s thing is the"
        }
        return out
    }

    static func ask(_ f: Fact) -> String {
        var out = " Fill in the missing word: \(f.key) has a ___."
        if style == "say" {
            out = " What does \(f.key) have? Answer in a short sentence."
        } else if style != "cloze" {
            out = " What does \(f.key) have? One word."
        }
        return out
    }

    static func reply(_ text: String) -> (word: String, single: Bool) {
        let words = text.lowercased()
            .split(whereSeparator: { c in !c.isLetter })
            .map(String.init)
            .filter { w in !["a", "an", "the"].contains(w) }
        let hit = words.lazy.compactMap { w in
            things.contains(w) ? w
                : w.hasSuffix("s") && things.contains(String(w.dropLast()))
                ? String(w.dropLast()) : nil
        }.first ?? ""
        return (hit, words.count == 1)
    }

    static func stated(_ facts: String, _ names: Set<String>) -> [String] {
        facts.split(separator: ".").compactMap { sentence in
            let words = sentence.split(separator: " ")
            let key = words.first.map(String.init) ?? ""
            return words.count == 4 && words[1] == "has" && names.contains(key)
                ? key : nil
        }
    }

    static func bucket(_ n: Int, _ edges: [Int], _ names: [String])
        -> String {
        let at = edges.firstIndex { e in n <= e } ?? edges.count
        return names[min(at, names.count - 1)]
    }
}

@MainActor func runRecallBench(_ args: CommandArgs) async {
    let count = Int(args.value("--recall-bench") ?? "") ?? 40
    let per = Int(args.value("--per-turn") ?? "") ?? 1
    RecallBench.fill = Int(args.value("--filler") ?? "") ?? 8
    RecallBench.style = args.value("--ask") ?? "cloze"
    let model = args.rest.first ?? ""
    var status: Int32 = 0
    if !isModelFile(model) {
        err("usage: chatokf <model> --recall-bench N [--per-turn F] "
            + "[--seed S] [--greedy] [--think]\n")
        status = 2
    } else {
        do {
            try await recallRuns(model, count, per,
                                 seedVal == 0 ? 1 : seedVal, args)
        } catch {
            err("[recall] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

private func spaced(_ thing: String) -> String {
    RecallBench.style == "has" || RecallBench.style == "thing"
        ? " " + thing : thing
}

@MainActor private func recallRuns(_ model: String, _ count: Int,
                                    _ per: Int, _ seed: UInt64,
                                    _ args: CommandArgs) async throws {
    let loaded = Gemma4Model.isGemma4(path: model)
        ? try await loadGemma(model, args, capVal) : try loadBonsai(model)
    let oneToken: (String) -> Bool = { w in
        [w, w.capitalized, " " + w, " " + w.capitalized].allSatisfy { s in
            loaded.backend.encode(s).count == 1
        }
    }
    let words = args.value("--words") ?? RecallBench.words
    RecallBench.names = try RecallBench.list(words, "names").filter(oneToken)
    RecallBench.things = try RecallBench.list(words, "things")
    RecallBench.filler = try RecallBench.list(words, "filler")
    let nouns = Set(RecallBench.things)
    RecallBench.names = RecallBench.names.filter { n in
        !nouns.contains(n.lowercased())
    }
    RecallBench.things = RecallBench.things.filter(oneToken)
    print("single-token words for this model: \(RecallBench.names.count) "
          + "names, \(RecallBench.things.count) things")
    let session = ChatSession(
        backend: loaded.backend, template: loaded.template,
        system: RecallBench.system, vocabSize: loaded.vocabCount,
        presets: SamplingPresets.greedy, maxTokens: 64, seed: seed,
        context: contextVal)
    let explain = Int(args.value("--explain") ?? "") ?? 0
    let names = Set(RecallBench.names)
    var mentioned: [String: Int] = [:]
    var held: [Bool: (Int, Int, Double)] = [:]
    let backs = ([5, 50, 150], ["0-5", "6-50", "51-150", "151+"])
    let ctxs = ([4096, 8192, 16384], ["<=4K", "4-8K", "8-16K", "16K+"])
    var byBack: [String: (Int, Int, Double)] = [:]
    var byCtx: [String: (Int, Int, Double)] = [:]
    var right = 0
    var mass = 0.0
    var peak = 0
    let started = Date()
    let plan = RecallBench.turns(count, per: per, seed: seed)
    let things = RecallBench.things
    for (i, turn) in plan.enumerated() {
        _ = try await session.replay([ChatSession.ReplayTurn(
            prompt: turn.facts, answer: "👍", parts: [], spans: [],
            labelled: false)])
        for key in RecallBench.stated(turn.facts, names) {
            mentioned[key] = 4 * i + 1
        }
        let gone = 4 * i + 3 - (await session.messageCount)
        let kept = (mentioned[turn.asked.key] ?? 0) > gone
        let shares = await session.judge(
            frame: turn.question, items: [""], options: things,
            label: "recall", seed: RecallBench.seed(turn.asked.key))
        let row = shares.first ?? []
        let best = row.indices.max { x, y in row[x] < row[y] }
        let pick = best.map { k in things[k] } ?? ""
        let want = things.firstIndex(of: turn.asked.thing)
        let p = want.map { k in k < row.count ? row[k] : 0 } ?? 0
        let ok = pick == turn.asked.thing
        if explain > 0, (i + 1) % explain == 0 {
            let saved = try await session.park()
            var said = ""
            let why = "In one short sentence of twenty words or less, "
                + "explain why \(turn.asked.key)'s thing is the \(pick)."
            for await piece in session.reply(why) { said += piece }
            let end = await session.lastMetrics.endReason
            try await session.resume(saved)
            print("explain \(i + 1) [\(end)] \(said)")
        }
        _ = try await session.replay([ChatSession.ReplayTurn(
            prompt: turn.question,
            answer: RecallBench.seed(turn.asked.key)
                + spaced(turn.asked.thing),
            parts: [],
            spans: [], labelled: false)])
        mentioned[turn.asked.key] = 4 * i + 4
        let vh = held[kept] ?? (0, 0, 0)
        held[kept] = (vh.0 + (ok ? 1 : 0), vh.1 + 1, vh.2 + p)
        let ctx = await session.lastMetrics.ctx
        peak = max(peak, ctx)
        right += ok ? 1 : 0
        mass += p
        let b = RecallBench.bucket(turn.back, backs.0, backs.1)
        let c = RecallBench.bucket(ctx, ctxs.0, ctxs.1)
        let vb = byBack[b] ?? (0, 0, 0)
        byBack[b] = (vb.0 + (ok ? 1 : 0), vb.1 + 1, vb.2 + p)
        let vc = byCtx[c] ?? (0, 0, 0)
        byCtx[c] = (vc.0 + (ok ? 1 : 0), vc.1 + 1, vc.2 + p)
        err(String(format: "[recall] %d ctx %d back %d %@ want %@ got %@ "
                   + "p %.2f %@\n", i + 1, ctx, turn.back, turn.asked.key,
                   turn.asked.thing, pick, p, kept ? "kept" : "dropped"))
        if !ok {
            print(String(format: "%4d ctx %6d back %4d want %-9@ got %-9@ "
                         + "p %.2f", i + 1, ctx, turn.back,
                         turn.asked.thing, pick, p))
        }
    }
    let n = max(plan.count, 1)
    print(String(format: "\nrecall %d/%d right, mean p %.2f, context %d "
                 + "tokens, %ds", right, plan.count, mass / Double(n), peak,
                 Int(Date().timeIntervalSince(started))))
    for (key, name) in [(true, "kept"), (false, "dropped")] {
        let v = held[key] ?? (0, 0, 0)
        print(String(format: "  %@: %d/%d  p %.2f", name, v.0, v.1,
                     v.1 > 0 ? v.2 / Double(v.1) : 0))
    }
    for (name, table) in [("back", (backs.1, byBack)),
                          ("ctx", (ctxs.1, byCtx))] {
        for key in table.0 {
            let v = table.1[key] ?? (0, 0, 0)
            if v.1 > 0 {
                print(String(format: "  %@ %@: %d/%d  p %.2f", name, key,
                             v.0, v.1, v.2 / Double(v.1)))
            }
        }
    }
}
