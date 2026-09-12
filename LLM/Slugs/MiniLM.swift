// A pure-Swift port of the reference llm.c embedder; the parity gate is
// cosine ~1.0 against llama.cpp. No KV cache, causal mask or sampling.
import Foundation

// Big matrices stay QUANTIZED in the mmap and dequantize row by row, so a
// live encoder pins ~36 MB of clean file-backed pages, not ~90 MB dirty.
final class MiniLM {
    struct Layer {
        let wq: GGUFTensor, bq: [Float]
        let wk: GGUFTensor, bk: [Float]
        let wv: GGUFTensor, bv: [Float]
        let wo: GGUFTensor, bo: [Float]
        let attnNormW: [Float], attnNormB: [Float]
        let wUp: GGUFTensor, bUp: [Float]
        let wDown: GGUFTensor, bDown: [Float]
        let outNormW: [Float], outNormB: [Float]
    }

    let nLayer: Int
    let nEmbd: Int
    let nFF: Int
    let nHead: Int
    let nCtx: Int
    let lnEps: Float
    private let clsId: Int32
    private let sepId: Int32
    private let unkId: Int32

    private let gguf: GGUF
    private let tokEmb: GGUFTensor
    private let posEmb: [Float]
    private let typeEmb: [Float]
    private let embNormW: [Float]
    private let embNormB: [Float]
    private let layers: [Layer]
    private let vocab: [String: Int32]
    // One reused row buffer; the encoder is single-threaded per embed.
    private var rowScratch: [Float]

    var dim: Int { nEmbd }

    init(gguf g: GGUF) {
        nLayer = g.int("bert.block_count") ?? 6
        nEmbd = g.int("bert.embedding_length") ?? 384
        nFF = g.int("bert.feed_forward_length") ?? 1536
        nHead = g.int("bert.attention.head_count") ?? 12
        nCtx = g.int("bert.context_length") ?? 512
        lnEps = Float(g.double("bert.attention.layer_norm_epsilon") ?? 1e-12)
        clsId = Int32(g.int("tokenizer.ggml.cls_token_id") ?? 101)
        sepId = Int32(g.int("tokenizer.ggml.seperator_token_id") ?? 102)
        unkId = Int32(g.int("tokenizer.ggml.unknown_token_id") ?? 100)

        gguf = g
        tokEmb = g.tensor("token_embd.weight")
        posEmb = MiniLM.dequant(g.tensor("position_embd.weight"))
        typeEmb = MiniLM.dequant(g.tensor("token_types.weight"))
        embNormW = MiniLM.dequant(g.tensor("token_embd_norm.weight"))
        embNormB = MiniLM.dequant(g.tensor("token_embd_norm.bias"))
        rowScratch = [Float](repeating: 0, count: nFF)
        let n = nLayer
        layers = (0..<n).map { il in
            func w(_ s: String) -> GGUFTensor { g.tensor("blk.\(il).\(s)") }
            func f(_ s: String) -> [Float] {
                MiniLM.dequant(g.tensor("blk.\(il).\(s)"))
            }
            return Layer(
                wq: w("attn_q.weight"), bq: f("attn_q.bias"),
                wk: w("attn_k.weight"), bk: f("attn_k.bias"),
                wv: w("attn_v.weight"), bv: f("attn_v.bias"),
                wo: w("attn_output.weight"), bo: f("attn_output.bias"),
                attnNormW: f("attn_output_norm.weight"),
                attnNormB: f("attn_output_norm.bias"),
                wUp: w("ffn_up.weight"), bUp: f("ffn_up.bias"),
                wDown: w("ffn_down.weight"), bDown: f("ffn_down.bias"),
                outNormW: f("layer_output_norm.weight"),
                outNormB: f("layer_output_norm.bias"))
        }
        var v = [String: Int32](minimumCapacity: 30720)
        if let tokens = g.strings("tokenizer.ggml.tokens") {
            for (i, piece) in tokens.enumerated() { v[piece] = Int32(i) }
        }
        vocab = v
    }

    // Block types take `start`/`count` in multiples of 32: rows are 32-wide
    // multiples, so a row maps to whole blocks.
    static func dequant(_ t: GGUFTensor, _ start: Int, _ count: Int,
                        into buf: inout [Float]) {
        let base = t.base
        switch t.type {
        case .f32:
            buf.withUnsafeMutableBytes {
                _ = memcpy($0.baseAddress!, base + start * 4, count * 4)
            }
        case .f16, .bf16:
            for i in 0..<count {
                let h = base.loadUnaligned(
                    fromByteOffset: (start + i) * 2, as: UInt16.self)
                buf[i] = Float(Float16(bitPattern: h))
            }
        case .q4_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 18
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: 2 + j, as: UInt8.self)
                    let lo = Int(byte & 0x0f) - 8
                    let hi = Int(byte >> 4) - 8
                    buf[b * 32 + j] = Float(lo) * d
                    buf[b * 32 + j + 16] = Float(hi) * d
                }
            }
        case .q8_0:
            let b0 = start / 32
            for b in 0..<(count / 32) {
                let p = base + (b0 + b) * 34
                let d = Float(Float16(bitPattern:
                    p.loadUnaligned(as: UInt16.self)))
                for j in 0..<32 {
                    let q = p.load(fromByteOffset: 2 + j, as: Int8.self)
                    buf[b * 32 + j] = Float(q) * d
                }
            }
        default:
            preconditionFailure("MiniLM: unsupported ggml type \(t.type)")
        }
    }

    static func dequant(_ t: GGUFTensor) -> [Float] {
        var out = [Float](repeating: 0, count: t.count)
        dequant(t, 0, t.count, into: &out)
        return out
    }

    private func isSpace(_ c: UInt8) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D || c == 0x0C
            || c == 0x0B
    }
    private func isPunct(_ c: UInt8) -> Bool {
        (c >= 33 && c <= 47) || (c >= 58 && c <= 64)
            || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)
    }
    private func lowerAscii(_ c: UInt8) -> UInt8 {
        (c >= 0x41 && c <= 0x5A) ? c + 32 : c
    }

    // BERT's basic tokenizer NFD-decomposes then drops marks, so accented Latin
    // folds to its ASCII base; '.' marks the non-decomposable, left as [UNK].
    private static func foldTable(_ s: String) -> [UInt8] {
        s.utf8.map { $0 == UInt8(ascii: ".") ? 0 : $0 }
    }
    private static let latin1Fold = MiniLM.foldTable(
        "aaaaaa.ceeeeiiii" + ".nooooo..uuuuy.." +
        "aaaaaa.ceeeeiiii" + ".nooooo..uuuuy.y")
    private static let latinAFold = MiniLM.foldTable(
        "aaaaaaccccccccdd" + "..eeeeeeeeeegggg" +
        "gggghh..iiiiiiii" + "ii..jjkk.lllllll" +
        "l..nnnnnn...oooo" + "oo..rrrrrrssssss" +
        "sstttt..uuuuuuuu" + "uuuuwwyyyzzzzzzs")

    // Well-formed UTF-8 is assumed; a truncated lead returns its raw byte.
    private func utf8Decode(_ bytes: [UInt8], _ i: Int, _ len: Int)
        -> (cp: UInt32, adv: Int) {
        let c = bytes[i]
        var cp: UInt32
        var n: Int
        if c < 0x80 { cp = UInt32(c); n = 1 }
        else if c & 0xE0 == 0xC0 { cp = UInt32(c & 0x1F); n = 2 }
        else if c & 0xF0 == 0xE0 { cp = UInt32(c & 0x0F); n = 3 }
        else if c & 0xF8 == 0xF0 { cp = UInt32(c & 0x07); n = 4 }
        else { cp = UInt32(c); n = 1 }
        if i + n > len { cp = UInt32(c); n = 1 }
        var k = 1
        while k < n {
            cp = (cp << 6) | UInt32(bytes[i + k] & 0x3F)
            k += 1
        }
        return (cp, n)
    }

    private enum CpKind { case chars, pass, punct, space, drop }

    // Unicode dashes and quotes count as punctuation, so they bound words.
    private func foldCp(_ cp: UInt32) -> (kind: CpKind, folded: UInt8) {
        var kind: CpKind = .pass
        var folded: UInt8 = 0
        if cp < 0x80 {
            let c = UInt8(cp)
            if isSpace(c) { kind = .space }
            else if isPunct(c) { kind = .punct; folded = c }
            else { kind = .chars; folded = lowerAscii(c) }
        } else if cp >= 0x0300 && cp <= 0x036F {
            kind = .drop                                  // combining marks
        } else if cp == 0x00A0 {
            kind = .space                                 // no-break space
        } else if cp >= 0x00C0 && cp <= 0x00FF
            && MiniLM.latin1Fold[Int(cp - 0x00C0)] != 0 {
            kind = .chars; folded = MiniLM.latin1Fold[Int(cp - 0x00C0)]
        } else if cp >= 0x0100 && cp <= 0x017F
            && MiniLM.latinAFold[Int(cp - 0x0100)] != 0 {
            kind = .chars; folded = MiniLM.latinAFold[Int(cp - 0x0100)]
        } else if cp >= 0x2010 && cp <= 0x2015 {
            kind = .punct; folded = 0x2D                  // hyphens / dashes
        } else if cp == 0x2018 || cp == 0x2019 || cp == 0x02BB {
            kind = .punct; folded = 0x27                  // curly / okina '
        } else if cp == 0x201C || cp == 0x201D {
            kind = .punct; folded = 0x22                  // curly double "
        } else {
            kind = .pass                                  // CJK, Greek, ...
        }
        return (kind, folded)
    }

    // The U+2581 prefix is llama.cpp's phantom space: word-initial pieces carry
    // it, so the whole word matches front to back. nil = out of vocab.
    private func wordpiece(_ w: [UInt8]) -> [Int32]? {
        var word1: [UInt8] = [0xE2, 0x96, 0x81]
        word1.append(contentsOf: w)
        let n = word1.count
        var pieces: [Int32] = []
        var i = 0
        var alive = true
        while i < n && alive {
            var j = n
            var found: Int32? = nil
            while j > i && found == nil {
                let key = String(decoding: word1[i..<j], as: UTF8.self)
                found = vocab[key]
                if found == nil { j -= 1 }
            }
            if let found {
                pieces.append(found)
                i = j
            } else {
                alive = false
            }
        }
        return alive ? pieces : nil
    }

    private func addWord(_ w: [UInt8], _ ids: inout [Int32]) {
        if let pieces = wordpiece(w) {
            ids.append(contentsOf: pieces)
        } else {
            ids.append(unkId)
        }
    }

    func tokenize(_ text: String) -> [Int32] {
        var ids: [Int32] = [clsId]
        let bytes = Array(text.utf8)
        let n = bytes.count
        var word: [UInt8] = []
        var i = 0
        while i < n && ids.count < nCtx - 1 {
            let (cp, adv) = utf8Decode(bytes, i, n)
            let (kind, folded) = foldCp(cp)
            switch kind {
            case .chars:
                word.append(folded)
            case .pass:
                word.append(contentsOf: bytes[i ..< i + adv])
            case .space, .punct:
                if !word.isEmpty {
                    addWord(word, &ids)
                    word.removeAll(keepingCapacity: true)
                }
                if kind == .punct { addWord([folded], &ids) }
            case .drop:
                break
            }
            i += adv
        }
        if !word.isEmpty && ids.count < nCtx - 1 { addWord(word, &ids) }
        ids.append(sepId)
        if ids.count > nCtx { ids.removeLast(ids.count - nCtx) }
        return ids
    }

    // W's row is dequantized into rowScratch just before its dot; the loop
    // stays a plain contiguous run so the optimizer vectorizes it.
    private func linear(_ w: GGUFTensor, _ b: [Float], _ x: [Float],
                        xoff: Int, inn: Int, out: Int,
                        _ y: inout [Float], yoff: Int) {
        x.withUnsafeBufferPointer { xp in
            let xb = xp.baseAddress! + xoff
            for o in 0..<out {
                MiniLM.dequant(w, o * inn, inn, into: &rowScratch)
                var acc: Float = 0
                rowScratch.withUnsafeBufferPointer { rp in
                    let row = rp.baseAddress!
                    for i in 0..<inn { acc += row[i] * xb[i] }
                }
                y[yoff + o] = acc + b[o]
            }
        }
    }

    private func layerNorm(_ x: inout [Float], off: Int, n: Int,
                           _ w: [Float], _ b: [Float]) {
        var mean: Float = 0
        for i in 0..<n { mean += x[off + i] }
        mean /= Float(n)
        var varr: Float = 0
        for i in 0..<n {
            let d = x[off + i] - mean
            varr += d * d
        }
        varr /= Float(n)
        let inv = 1 / (varr + lnEps).squareRoot()
        for i in 0..<n {
            x[off + i] = (x[off + i] - mean) * inv * w[i] + b[i]
        }
    }

    private func gelu(_ x: inout [Float], n: Int) {
        let k: Float = 0.7978845608028654   // sqrt(2/pi)
        for i in 0..<n {
            let v = x[i]
            x[i] = 0.5 * v * (1 + tanhf(k * (v + 0.044715 * v * v * v)))
        }
    }

    private func softmax(_ s: inout [Float], n: Int) {
        var mx = s[0]
        for i in 1..<n where s[i] > mx { mx = s[i] }
        var sum: Float = 0
        for i in 0..<n {
            s[i] = expf(s[i] - mx)
            sum += s[i]
        }
        let inv = 1 / sum
        for i in 0..<n { s[i] *= inv }
    }

    // Full bidirectional attention; post-LN, so the residual add precedes it.
    private func encoderLayer(_ L: Layer, _ x: inout [Float], T: Int) {
        let ne = nEmbd, hd = nEmbd / nHead
        let scale = 1 / Float(hd).squareRoot()
        var q = [Float](repeating: 0, count: T * ne)
        var k = [Float](repeating: 0, count: T * ne)
        var vv = [Float](repeating: 0, count: T * ne)
        var ctx = [Float](repeating: 0, count: T * ne)
        for t in 0..<T {
            linear(L.wq, L.bq, x, xoff: t * ne, inn: ne, out: ne, &q, yoff: t * ne)
            linear(L.wk, L.bk, x, xoff: t * ne, inn: ne, out: ne, &k, yoff: t * ne)
            linear(L.wv, L.bv, x, xoff: t * ne, inn: ne, out: ne, &vv, yoff: t * ne)
        }
        var scores = [Float](repeating: 0, count: T)
        for h in 0..<nHead {
            let off = h * hd
            for i in 0..<T {
                q.withUnsafeBufferPointer { qp in
                    k.withUnsafeBufferPointer { kp in
                        let qi = qp.baseAddress! + i * ne + off
                        for j in 0..<T {
                            let kj = kp.baseAddress! + j * ne + off
                            var acc: Float = 0
                            for d in 0..<hd { acc += qi[d] * kj[d] }
                            scores[j] = acc * scale
                        }
                    }
                }
                softmax(&scores, n: T)
                for d in 0..<hd { ctx[i * ne + off + d] = 0 }
                for j in 0..<T {
                    let wj = scores[j]
                    for d in 0..<hd {
                        ctx[i * ne + off + d] += wj * vv[j * ne + off + d]
                    }
                }
            }
        }
        var tmp = [Float](repeating: 0, count: ne)
        var ff = [Float](repeating: 0, count: nFF)
        for t in 0..<T {
            linear(L.wo, L.bo, ctx, xoff: t * ne, inn: ne, out: ne, &tmp, yoff: 0)
            for d in 0..<ne { x[t * ne + d] += tmp[d] }
            layerNorm(&x, off: t * ne, n: ne, L.attnNormW, L.attnNormB)
            linear(L.wUp, L.bUp, x, xoff: t * ne, inn: ne, out: nFF, &ff, yoff: 0)
            gelu(&ff, n: nFF)
            linear(L.wDown, L.bDown, ff, xoff: 0, inn: nFF, out: ne, &tmp, yoff: 0)
            for d in 0..<ne { x[t * ne + d] += tmp[d] }
            layerNorm(&x, off: t * ne, n: ne, L.outNormW, L.outNormB)
        }
    }

    // Type is always 0 (one segment); id*ne lands on a block boundary.
    private func embedTokens(_ ids: [Int32], _ x: inout [Float]) {
        let ne = nEmbd
        for t in 0..<ids.count {
            MiniLM.dequant(tokEmb, Int(ids[t]) * ne, ne, into: &rowScratch)
            let pe = t * ne
            for d in 0..<ne {
                x[t * ne + d] = rowScratch[d] + posEmb[pe + d] + typeEmb[d]
            }
            layerNorm(&x, off: t * ne, n: ne, embNormW, embNormB)
        }
    }

    @discardableResult
    func embed(_ text: String, into out: inout [Float]) -> Int {
        let ids = tokenize(text)
        let T = ids.count
        if T > 0 {
            let ne = nEmbd
            var x = [Float](repeating: 0, count: T * ne)
            embedTokens(ids, &x)
            for l in 0..<nLayer { encoderLayer(layers[l], &x, T: T) }
            for d in 0..<ne { out[d] = 0 }
            for t in 0..<T {
                for d in 0..<ne { out[d] += x[t * ne + d] }
            }
            let invT = 1 / Float(T)
            for d in 0..<ne { out[d] *= invT }
            var ss: Float = 0
            for d in 0..<ne { ss += out[d] * out[d] }
            let inv = ss > 0 ? 1 / ss.squareRoot() : 0
            for d in 0..<ne { out[d] *= inv }
        }
        return T
    }

    func embed(_ text: String) -> [Float] {
        var out = [Float](repeating: 0, count: nEmbd)
        embed(text, into: &out)
        return out
    }
}
