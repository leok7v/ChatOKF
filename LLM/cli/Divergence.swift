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
