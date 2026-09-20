import Accelerate
import Foundation
import LLM

struct TopK {
    static let k = 64
    static let magic: UInt32 = 0x444C_4B31
    static let stride = 8 + k * 8
}

func runDivergence(_ path: String, _ args: CommandArgs) throws {
    let dump = args.value("--kld-dump")
    let against = args.value("--kld")
    let corpus = args.value("--ppl") ?? ""
    let ctx = args.int("--ppl-ctx") ?? 512
    let cap = args.int("--ppl-chunks") ?? Int.max
    let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
    if text.isEmpty {
        err("usage: chatokf-cli <model.gguf> --ppl <corpus.txt> "
            + "[--kld-dump <top.bin> | --kld <top.bin>] "
            + "[--ppl-ctx N] [--ppl-chunks N]\n")
        exit(2)
    }
    let chat = try QwenMetalChat(ggufPath: path)
    let ids = chat.tokenizer.encode(text, addSpecial: true)
    let chunks = min(ids.count / ctx, cap)
    let vocab = chat.tokenizer.vocabCount
    let first = ctx / 2
    var teacher = Data()
    if let against {
        teacher = try Data(contentsOf: URL(fileURLWithPath: against))
        let head = teacher.withUnsafeBytes { raw in
            (raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
             raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        if head.0 != TopK.magic || Int(head.1) != TopK.k {
            err("[kld] \(against) is not a top-\(TopK.k) dump\n")
            exit(2)
        }
    }
    var sink = Data()
    if dump != nil {
        sink.append(contentsOf: withUnsafeBytes(of: TopK.magic) { b in
            Array(b)
        })
        sink.append(contentsOf: withUnsafeBytes(of: UInt32(TopK.k)) { b in
            Array(b)
        })
    }
    var tally = Divergence()
    let t0 = Date()
    for c in 0..<chunks {
        let base = c * ctx
        chat.engine.chunkCost(Array(ids[base..<(base + ctx)]),
                              from: first) { _, want, lp in
            let top = rank(lp, vocab)
            if dump != nil {
                append(&sink, want, lp, top, vocab)
            }
            if against != nil {
                tally.add(lp, vocab, teacher, top)
            }
            tally.tokens += 1
            tally.cost += cost(lp, vocab, want)
        }
        err(String(format: "\r[kld] chunk %d/%d  %.0f%%   ", c + 1, chunks,
                   100 * Double(c + 1) / Double(chunks)))
    }
    err("\n")
    if let dump {
        try sink.write(to: URL(fileURLWithPath: dump))
        err("[kld] wrote \(tally.tokens) positions, "
            + "\(sink.count / 1_000_000) MB -> \(dump)\n")
    }
    let name = (path as NSString).lastPathComponent
    print(name + String(repeating: " ", count: max(1, 40 - name.count))
          + String(format: "ppl %8.4f", exp(tally.cost / Double(tally.tokens)))
          + (against == nil ? "" : tally.line))
    err(String(format: "[kld] %.0fs\n", Date().timeIntervalSince(t0)))
    exit(0)
}

struct Divergence {
    var tokens = 0
    var cost = 0.0
    var kl = 0.0
    var top1 = 0
    var top5 = 0
    var tau = 0.0
    var peak = 0.0
    var covered = 0.0
    var at = 0

    var line: String {
        let n = Double(max(tokens, 1))
        return String(format: "   kl %.5f  top1 %.2f%%  top5 %.2f%%  "
            + "tau %.4f  maxdp %.4f  cover %.3f", kl / n,
            100 * Double(top1) / n, 100 * Double(top5) / (5 * n), tau / n,
            peak, covered / n)
    }

    mutating func add(_ lp: UnsafePointer<Float>, _ vocab: Int,
                      _ teacher: Data, _ mine: [Int32]) {
        let off = 8 + at * TopK.stride
        at += 1
        var ids = [Int32](repeating: 0, count: TopK.k)
        var logits = [Float](repeating: 0, count: TopK.k)
        var lse = Float(0)
        teacher.withUnsafeBytes { raw in
            lse = raw.loadUnaligned(fromByteOffset: off + 4, as: Float.self)
            for i in 0..<TopK.k {
                let seat = off + 8 + i * 8
                ids[i] = raw.loadUnaligned(fromByteOffset: seat,
                                           as: Int32.self)
                logits[i] = raw.loadUnaligned(fromByteOffset: seat + 4,
                                              as: Float.self)
            }
        }
        let qlse = logSumExp(lp, vocab)
        var pMass = 0.0
        var qMass = 0.0
        var sum = 0.0
        var worst = 0.0
        for i in 0..<TopK.k {
            let p = Double(exp(logits[i] - lse))
            let q = Double(exp(lp[Int(ids[i])] - qlse))
            pMass += p
            qMass += q
            sum += p * (log(max(p, 1e-30)) - log(max(q, 1e-30)))
            worst = max(worst, abs(p - q))
        }
        let pTail = max(1 - pMass, 1e-12)
        let qTail = max(1 - qMass, 1e-12)
        kl += sum + pTail * log(pTail / qTail)
        covered += pMass
        peak = max(peak, worst)
        top1 += ids[0] == mine[0] ? 1 : 0
        top5 += Set(ids.prefix(5)).intersection(mine.prefix(5)).count
        tau += kendall(ids, lp)
    }

    func kendall(_ ids: [Int32], _ lp: UnsafePointer<Float>) -> Double {
        var same = 0
        var flipped = 0
        for i in 0..<TopK.k {
            for j in (i + 1)..<TopK.k {
                let a = lp[Int(ids[i])]
                let b = lp[Int(ids[j])]
                same += a > b ? 1 : 0
                flipped += a < b ? 1 : 0
            }
        }
        let pairs = same + flipped
        return pairs > 0 ? Double(same - flipped) / Double(pairs) : 1
    }
}

func rank(_ lp: UnsafePointer<Float>, _ vocab: Int) -> [Int32] {
    var ids = [Int32](repeating: 0, count: TopK.k)
    var vals = [Float](repeating: -Float.infinity, count: TopK.k)
    var floor = -Float.infinity
    for i in 0..<vocab where lp[i] > floor {
        var at = TopK.k - 1
        while at > 0 && lp[i] > vals[at - 1] {
            vals[at] = vals[at - 1]
            ids[at] = ids[at - 1]
            at -= 1
        }
        vals[at] = lp[i]
        ids[at] = Int32(i)
        floor = vals[TopK.k - 1]
    }
    return ids
}

func append(_ sink: inout Data, _ want: Int32,
                    _ lp: UnsafePointer<Float>, _ top: [Int32],
                    _ vocab: Int) {
    var lse = logSumExp(lp, vocab)
    sink.append(contentsOf: withUnsafeBytes(of: want) { b in Array(b) })
    sink.append(contentsOf: withUnsafeBytes(of: &lse) { b in Array(b) })
    for id in top {
        var seat = id
        var v = lp[Int(id)]
        sink.append(contentsOf: withUnsafeBytes(of: &seat) { b in Array(b) })
        sink.append(contentsOf: withUnsafeBytes(of: &v) { b in Array(b) })
    }
}

func logSumExp(_ logits: UnsafePointer<Float>, _ len: Int) -> Float {
    var peak: Float = 0
    let n = vDSP_Length(len)
    vDSP_maxv(logits, 1, &peak, n)
    var shifted = [Float](repeating: 0, count: len)
    var minus = -peak
    vDSP_vsadd(logits, 1, &minus, &shifted, 1, n)
    var count = Int32(len)
    vvexpf(&shifted, shifted, &count)
    var sum: Float = 0
    vDSP_sve(shifted, 1, &sum, n)
    return peak + log(sum)
}

struct PplitTurn {
    let prompt: String
    let response: String
}

func parsePplitCorpus(_ text: String) -> [[PplitTurn]] {
    let conv = "<|@CONV@|>", ask = "<|@PROMPT@|>", reply = "<|@RESPONSE@|>"
    let multi = text.contains(conv + "\n")
    var convs: [[PplitTurn]] = []
    var cur: [PplitTurn] = []
    var prompt = "", response = ""
    var at = 0
    for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
        let line = String(raw)
        let have = !prompt.isEmpty && !response.isEmpty
        if line == conv {
            if have { cur.append(PplitTurn(prompt: prompt, response: response)) }
            if !cur.isEmpty { convs.append(cur) }
            cur = []; prompt = ""; response = ""; at = 0
        } else if line == ask {
            if have { cur.append(PplitTurn(prompt: prompt, response: response)) }
            if have && !multi { convs.append(cur); cur = [] }
            prompt = ""; response = ""; at = 1
        } else if line == reply {
            at = 2
        } else if at == 1 {
            prompt += prompt.isEmpty ? line : "\n" + line
        } else if at == 2 {
            response += response.isEmpty ? line : "\n" + line
        }
    }
    if !prompt.isEmpty && !response.isEmpty {
        cur.append(PplitTurn(prompt: prompt, response: response))
    }
    if !cur.isEmpty { convs.append(cur) }
    return convs
}

func pplitMessages(_ conv: [PplitTurn], upto: Int) -> [AgentMessage] {
    var out: [AgentMessage] = []
    for i in 0...upto {
        out.append(AgentMessage(role: "user", content: conv[i].prompt))
        if i < upto {
            out.append(AgentMessage(role: "assistant", content: conv[i].response))
        }
    }
    return out
}

struct PplitScorer {
    let vocab: Int
    let render: ([PplitTurn], Int) throws -> String
    let framed: (String) -> [Int32]
    let plain: (String) -> [Int32]
    let score: ([Int32], Int, (Int, Int32, UnsafePointer<Float>) -> Void) -> Void
}

func pplitScorer(_ path: String) throws -> PplitScorer {
    let out: PplitScorer
    if Gemma4Model.isGemma4(path: path) {
        let chat = try GemmaChat(ggufPath: path)
        let engine = try Gemma4MetalEngine(chat.model)
        out = PplitScorer(
            vocab: chat.vocabCount,
            render: { conv, upto in
                try renderPrompt(template: chat.chatTemplate,
                                 messages: pplitMessages(conv, upto: upto),
                                 tools: [], addGenerationPrompt: true,
                                 enableThinking: false, bosToken: chat.bosToken)
            },
            framed: { s in chat.encode(s) },
            plain: { s in chat.encodeRaw(s) },
            score: { ids, from, sink in engine.chunkCost(ids, from: from, want: sink) })
    } else {
        let chat = try QwenMetalChat(ggufPath: path)
        out = PplitScorer(
            vocab: chat.tokenizer.vocabCount,
            render: { conv, upto in
                try renderPrompt(template: chat.chatTemplate,
                                 messages: pplitMessages(conv, upto: upto),
                                 tools: [], addGenerationPrompt: true,
                                 enableThinking: false)
            },
            framed: { s in chat.tokenizer.encode(s, addSpecial: true) },
            plain: { s in chat.tokenizer.encode(s, addSpecial: false) },
            score: { ids, from, sink in chat.engine.chunkCost(ids, from: from, want: sink) })
    }
    return out
}

struct PplitTotals {
    var nll = 0.0
    var n = 0
    var top1 = 0
    var top5 = 0
    var top10 = 0
    var ranks: [Int32] = []
    var turns = 0
    var skipped = 0

    var ppl: Double { n > 0 ? exp(nll / Double(n)) : 0 }

    mutating func add(_ lp: UnsafePointer<Float>, _ vocab: Int, _ want: Int32) {
        nll += cost(lp, vocab, want)
        let w = lp[Int(want)]
        var r: Int32 = 0
        var i = 0
        while i < vocab {
            if lp[i] > w { r += 1 }
            i += 1
        }
        top1 += r == 0 ? 1 : 0
        top5 += r < 5 ? 1 : 0
        top10 += r < 10 ? 1 : 0
        ranks.append(r)
        n += 1
    }

    func line(_ name: String, convs: Int) -> String {
        let sorted = ranks.sorted()
        let last = Double(max(sorted.count - 1, 0))
        let d = Double(max(n, 1))
        let pad = name.count < 30 ? String(repeating: " ", count: 30 - name.count) : ""
        return name + pad + String(
            format: " ppl %8.4f  top1 %6.2f%%  top5 %6.2f%%  top10 %6.2f%%  "
                + "rank p50 %d p90 %d   over %d tokens, %d turns, %d convs",
            ppl, 100 * Double(top1) / d, 100 * Double(top5) / d,
            100 * Double(top10) / d,
            sorted.isEmpty ? 0 : Int(sorted[Int(0.5 * last)]),
            sorted.isEmpty ? 0 : Int(sorted[Int(0.9 * last)]), n, turns, convs)
    }
}

func pplitScore(_ s: PplitScorer, _ convs: [[PplitTurn]], ctx: Int, cap: Int,
                dump: String?, against: String?,
                progress: (Int, Int, Double) -> Void) throws
    -> (PplitTotals, Divergence?) {
    var teacher = Data()
    if let against {
        teacher = try Data(contentsOf: URL(fileURLWithPath: against))
        let head = teacher.withUnsafeBytes { raw in
            (raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self),
             raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
        }
        if head.0 != TopK.magic || Int(head.1) != TopK.k {
            throw NSError(domain: "pplit", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "\(against) is not a top-\(TopK.k) dump"])
        }
    }
    var sink = Data()
    if dump != nil {
        sink.append(contentsOf: withUnsafeBytes(of: TopK.magic) { b in Array(b) })
        sink.append(contentsOf: withUnsafeBytes(of: UInt32(TopK.k)) { b in Array(b) })
    }
    var totals = PplitTotals()
    var tally = Divergence()
    for (c, conv) in convs.enumerated() {
        for t in 0..<conv.count {
            let rendered = try s.render(conv, t)
            let pid = s.framed(rendered)
            var rid = s.plain(conv[t].response)
            if cap > 0, rid.count > cap { rid = Array(rid.prefix(cap)) }
            let ids = pid + rid
            if rendered.isEmpty || rid.count < 2 || ids.count > ctx {
                totals.skipped += 1
            } else {
                s.score(ids, pid.count) { _, want, lp in
                    totals.add(lp, s.vocab, want)
                    if dump != nil || against != nil {
                        let top = rank(lp, s.vocab)
                        if dump != nil { append(&sink, want, lp, top, s.vocab) }
                        if against != nil {
                            tally.add(lp, s.vocab, teacher, top)
                            tally.tokens += 1
                        }
                    }
                }
                totals.turns += 1
            }
        }
        progress(c + 1, convs.count, totals.ppl)
    }
    if let dump { try sink.write(to: URL(fileURLWithPath: dump)) }
    return (totals, against == nil ? nil : tally)
}

func pplitCorpusBeside(_ file: String = #filePath) -> String {
    URL(fileURLWithPath: file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("LLM/fixtures/pplit/pplit-corpus.txt").path
}

func runPplit(_ path: String, _ args: CommandArgs) throws {
    let given = args.value("--pplit") ?? ""
    let corpus = given.isEmpty ? pplitCorpusBeside() : given
    let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
    if text.isEmpty {
        err("no corpus at \(corpus)\n"
            + "usage: chatokf <model.ggxf> --pplit [pplit-corpus.txt] "
            + "[--pplit-ctx 8192] [--items N] [--cap 320] "
            + "[--kld-dump <top.bin> | --kld <top.bin>]\n")
        exit(2)
    }
    var convs = parsePplitCorpus(text)
    if let items = args.int("--items"), items > 0, convs.count > items {
        convs = Array(convs.prefix(items))
    }
    let t0 = Date()
    let scorer = try pplitScorer(path)
    let (totals, kl) = try pplitScore(
        scorer, convs, ctx: args.int("--pplit-ctx") ?? 8192,
        cap: args.int("--cap") ?? 320, dump: args.value("--kld-dump"),
        against: args.value("--kld")) { done, total, ppl in
        err(String(format: "\r[pplit] %d/%d  ppl %.4f   ", done, total, ppl))
    }
    err("\n")
    print(totals.line((path as NSString).lastPathComponent, convs: convs.count))
    if let kl { print(String(repeating: " ", count: 30) + kl.line) }
    err(String(format: "[pplit] %d turns scored, %d skipped, %.0fs\n",
               totals.turns, totals.skipped, Date().timeIntervalSince(t0)))
    exit(0)
}
