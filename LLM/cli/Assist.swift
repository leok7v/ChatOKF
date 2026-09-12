import Foundation
import LLM

private func asked(_ args: CommandArgs) -> String {
    args.turns.last
        ?? "Explain, step by step, how a bicycle stays upright while it "
        + "is moving, and why it falls over when it stops."
}

private func sampling(_ chat: GemmaChat, _ args: CommandArgs) -> Sampler? {
    var out: Sampler? = nil
    if args.flag("--sampler") {
        out = Sampler(vocabSize: chat.vocabCount,
                      config: chat.samplingPresets.select(thinking: false,
                                                          vision: false))
    }
    return out
}

private func templated(_ chat: GemmaChat, _ question: String) -> [Int32] {
    var prompt = question
    if !chat.chatTemplate.isEmpty {
        prompt = (try? renderPrompt(
            template: chat.chatTemplate,
            messages: [AgentMessage(role: "user", content: question)],
            tools: [], addGenerationPrompt: true,
            enableThinking: false, bosToken: chat.bosToken)) ?? question
    }
    return chat.encode(prompt)
}

func runAssistBench(_ path: String, _ args: CommandArgs) throws {
    let gen = args.int("--assist-bench") ?? 128
    let chat = try GemmaChat(ggufPath: path)
    let engine = try Gemma4MetalEngine(chat.model)
    var status: Int32 = 0
    if !engine.hasAssist {
        err("\(path) carries no assist head\n")
        status = 2
    } else {
        let ids = templated(chat, asked(args))
        engine.reset()
        engine.sampler = sampling(chat, args)
        var warm = engine.extend(ids)
        for _ in 0..<8 { warm = engine.decode(warm) }

        engine.reset()
        engine.sampler = sampling(chat, args)
        var plain: [Int32] = []
        var next = engine.extend(ids)
        let g0 = Date()
        while plain.count < gen {
            plain.append(next)
            next = engine.decodePlain(next)
        }
        let plainSec = Date().timeIntervalSince(g0)

        engine.reset()
        engine.sampler = sampling(chat, args)
        var spec: [Int32] = []
        var cur = engine.extend(ids)
        let s0 = Date()
        while spec.count < gen {
            spec.append(cur)
            cur = engine.decode(cur)
        }
        let specSec = Date().timeIntervalSince(s0)

        var diff = -1
        var i = 0
        while i < gen && diff < 0 {
            if plain[i] != spec[i] { diff = i }
            i += 1
        }
        let cycles = max(engine.specCycles, 1)
        let tpc = Double(engine.specCommitted) / Double(cycles)
        let acc = Double(engine.specAccepted)
            / Double(max(engine.specDrafted, 1))
        print(chat.decode(spec))
        print("VERIFY-ASSIST (n=\(Gemma4MetalEngine.specN), gen=\(gen)): "
              + (diff < 0 ? "\(gen)/\(gen) EXACT"
                          : "DIVERGES at \(diff) "
                            + "(plain \(plain[diff]) spec \(spec[diff]))"))
        print(String(format: "  %.2f tok/cycle over %d cycles, accept %.0f%%",
                     tpc, cycles, acc * 100))
        print(String(format: "  tg%d %.2f t/s  |  assist n=%d %.2f t/s  %.2fx",
                     gen, Double(gen) / plainSec,
                     Gemma4MetalEngine.specN, Double(gen) / specSec,
                     plainSec / specSec))
    }
    exit(status)
}

func runAssistProbe(_ path: String, _ args: CommandArgs) throws {
    let want = args.int("--assist-probe") ?? 128
    let chat = try GemmaChat(ggufPath: path)
    let engine = try Gemma4MetalEngine(chat.model)
    var status: Int32 = 0
    if !engine.hasAssist {
        err("\(path) carries no assist head\n")
        status = 2
    } else {
        let ids = templated(chat, asked(args))
        var next = engine.extend(ids)
        var agree = 0
        var total = 0
        var draftSecs = 0.0
        var trunkSecs = 0.0
        var produced: [Int32] = []
        while total < want && !chat.eosIds.contains(next) {
            let d0 = Date()
            let drafted = engine.assistDraft(next, count: 1).first ?? -1
            draftSecs += Date().timeIntervalSince(d0)
            let t0 = Date()
            let following = engine.decodePlain(next)
            trunkSecs += Date().timeIntervalSince(t0)
            if drafted == following { agree += 1 }
            total += 1
            produced.append(next)
            next = following
        }
        let rate = total > 0 ? 100 * Double(agree) / Double(total) : 0
        let draftMs = total > 0 ? 1000 * draftSecs / Double(total) : 0
        let trunkMs = total > 0 ? 1000 * trunkSecs / Double(total) : 0
        print(chat.decode(produced))
        print(String(format: "[assist] %d steps  top1 agree %.1f%%  "
                     + "draft %.2f ms  trunk %.2f ms  cost %.2fx",
                     total, rate, draftMs, trunkMs,
                     trunkMs > 0 ? draftMs / trunkMs : 0))
    }
    exit(status)
}
