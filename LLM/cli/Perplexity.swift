import Accelerate
import Foundation
import LLM

func runPerplexity(_ path: String, _ args: CommandArgs) throws {
    let corpus = args.value("--ppl") ?? ""
    let ctx = args.int("--ppl-ctx") ?? 512
    let cap = args.int("--ppl-chunks") ?? Int.max
    let text = (try? String(contentsOfFile: corpus, encoding: .utf8)) ?? ""
    if text.isEmpty {
        err("usage: chatokf-cli <model.gguf> --ppl <corpus.txt> "
            + "[--ppl-ctx 512] [--ppl-chunks N]\n")
        exit(2)
    }
    let gemma = Gemma4Model.isGemma4(path: path)
    var ids: [Int32] = []
    var vocab = 0
    var score: ([Int32], Int,
                (Int, Int32, UnsafePointer<Float>) -> Void) -> Void
    var stepper: ((Int32) -> [Float])? = nil
    var rewind: () -> Void = { }
    if gemma {
        let chat = try GemmaChat(ggufPath: path)
        let engine = try Gemma4MetalEngine(chat.model)
        if args.flag("--ppl-serial") { engine.batch = 1 }
        ids = chat.encode(text)
        vocab = chat.vocabCount
        score = { rows, from, sink in
            engine.chunkCost(rows, from: from, want: sink)
        }
    } else {
        let chat = try QwenMetalChat(ggufPath: path)
        ids = chat.tokenizer.encode(text, addSpecial: true)
        vocab = chat.tokenizer.vocabCount
        score = { rows, from, sink in
            chat.engine.chunkCost(rows, from: from, want: sink)
        }
        stepper = { t in chat.engine.step(t) }
        rewind = { chat.engine.reset() }
    }
    let chunks = min(ids.count / ctx, cap)
    if chunks == 0 {
        err("[ppl] corpus is \(ids.count) tokens, need \(ctx)\n")
        exit(2)
    }
    let first = ctx / 2
    let serial = args.flag("--ppl-serial") && stepper != nil
    var total = 0.0
    var scored = 0
    let t0 = Date()
    for c in 0..<chunks {
        let base = c * ctx
        if serial {
            rewind()
            for i in 0..<(ctx - 1) {
                let logits = stepper!(ids[base + i])
                if i + 1 >= first {
                    total += cost(logits, logits.count, ids[base + i + 1])
                    scored += 1
                }
            }
        } else {
            score(Array(ids[base..<(base + ctx)]), first) { _, want, lp in
                total += cost(lp, vocab, want)
                scored += 1
            }
        }
        let done = Double(c + 1) / Double(chunks)
        err(String(format: "\r[ppl] chunk %d/%d  ppl %.4f  %.0f%%   ",
                   c + 1, chunks, exp(total / Double(scored)), done * 100))
    }
    let ppl = exp(total / Double(scored))
    err("\n")
    let name = (path as NSString).lastPathComponent
    print(name + String(repeating: " ", count: max(1, 40 - name.count))
          + String(format: "ppl %8.4f   (%d tokens, ctx %d, %.0fs)",
                   ppl, scored, ctx, Date().timeIntervalSince(t0)))
    exit(0)
}

func cost(_ logits: UnsafePointer<Float>, _ len: Int,
                  _ want: Int32) -> Double {
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
    return Double(peak) + Double(log(sum)) - Double(logits[Int(want)])
}
