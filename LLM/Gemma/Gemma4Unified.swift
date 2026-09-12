import Foundation

// TWO EPSILONS: the three LayerNorms carry torch's DEFAULT 1e-5, which no
// config ships, while the RMSNorms use the model's rms_norm_eps.

public struct Gemma4UnifiedMedia {
    let embd: Int
    let normEps: Float
    let visionRmsEps: Float
    let audioRmsEps: Float
    public let samplesPerToken: Int
    public let audioMaxTokens: Int

    private let ln1w, ln1b: [Float]
    private let ln2w, ln2b: [Float]
    private let ln3w, ln3b: [Float]
    private let dense: GGUFTensor
    private let denseBias: [Float]
    private let posEmbd: GGUFTensor
    private let visionProj: GGUFTensor
    private let audioProj: GGUFTensor

    public init(_ model: Gemma4Model) throws {
        let g = model.gguf
        func v(_ name: String) -> [Float] { Dense.floats(g.tensor(name)) }
        embd = model.cfg.nEmbd
        normEps = Float(g.double("gemma4.vision.norm_epsilon") ?? 1e-5)
        visionRmsEps = Float(g.double("gemma4.vision.rms_epsilon") ?? 1e-6)
        audioRmsEps = Float(g.double("gemma4.audio.rms_epsilon") ?? 1e-6)
        samplesPerToken = try requireInt(
            g, "gemma4.audio.samples_per_token",
            "a soft token IS a frame of raw samples, so its width is the "
            + "whole audio frontend")
        audioMaxTokens = g.int("gemma4.audio.max_soft_tokens") ?? 750
        ln1w = v("v.patch_norm.1.weight")
        ln1b = v("v.patch_norm.1.bias")
        ln2w = v("v.patch_norm.2.weight")
        ln2b = v("v.patch_norm.2.bias")
        ln3w = v("v.patch_norm.3.weight")
        ln3b = v("v.patch_norm.3.bias")
        dense = g.tensor("v.patch_embd.weight")
        denseBias = v("v.patch_embd.bias")
        posEmbd = g.tensor("v.position_embd.weight")
        visionProj = g.tensor("mm.vision.weight")
        audioProj = g.tensor("mm.audio.weight")
    }

    // Off the projection itself, so a mismatch fails loudly instead of reading
    // past a patch.
    public var patchDim: Int { dense.dims[0] }

    public func image(pixels: [Float],
                      pos: [(Int, Int)]) -> [[Float]] {
        let dim = patchDim
        var out = [[Float]](repeating: [], count: pos.count)
        for i in 0..<pos.count {
            let patch = Array(pixels[(i * dim)..<((i + 1) * dim)])
            out[i] = embedPatch(patch, pos[i])
        }
        return out
    }

    private func embedPatch(_ patch: [Float],
                            _ at: (x: Int, y: Int)) -> [Float] {
        var h = layerNorm(patch, ln1w, ln1b, normEps)
        var y = [Float](repeating: 0, count: embd)
        GQ.matvec(dense, x: h, out: &y)
        for i in 0..<embd { y[i] += denseBias[i] }
        y = layerNorm(y, ln2w, ln2b, normEps)
        // Factorized: the table is [posemb_size][2][embd], so the row index is
        // the coordinate times two plus the axis.
        addPosition(&y, row: at.x * 2)
        addPosition(&y, row: at.y * 2 + 1)
        y = layerNorm(y, ln3w, ln3b, normEps)
        h = y
        GK.rmsnormRowsNoWeight(&h, d: embd, rows: 1, eps: visionRmsEps)
        var out = [Float](repeating: 0, count: embd)
        GQ.matvec(visionProj, x: h, out: &out)
        return out
    }

    private func addPosition(_ y: inout [Float], row: Int) {
        var slice = [Float](repeating: 0, count: embd)
        slice.withUnsafeMutableBufferPointer { buf in
            GQ.gather(posEmbd, row: row, from: 0, count: embd,
                      into: buf.baseAddress!)
        }
        for i in 0..<embd { y[i] += slice[i] }
    }

    // No mel, no window, no overlap: a token is the next `samplesPerToken`
    // samples, zero-padded.
    public func audio(_ pcm: [Float]) -> [[Float]] {
        let per = samplesPerToken
        let frames = min((pcm.count + per - 1) / per, audioMaxTokens)
        var out = [[Float]](repeating: [], count: frames)
        for f in 0..<frames {
            var frame = [Float](repeating: 0, count: per)
            let lo = f * per
            let hi = min(lo + per, pcm.count)
            if lo < hi {
                for i in lo..<hi { frame[i - lo] = pcm[i] }
            }
            GK.rmsnormRowsNoWeight(&frame, d: per, rows: 1, eps: audioRmsEps)
            var token = [Float](repeating: 0, count: embd)
            GQ.matvec(audioProj, x: frame, out: &token)
            out[f] = token
        }
        return out
    }

    // Torch's LayerNorm divides the variance by N, not N-1.
    private func layerNorm(_ x: [Float], _ w: [Float], _ b: [Float],
                           _ eps: Float) -> [Float] {
        let n = Float(x.count)
        var sum: Float = 0
        for v in x { sum += v }
        let mean = sum / n
        var sq: Float = 0
        for v in x { sq += (v - mean) * (v - mean) }
        let inv = 1 / (sq / n + eps).squareRoot()
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = (x[i] - mean) * inv * w[i] + b[i] }
        return out
    }
}
