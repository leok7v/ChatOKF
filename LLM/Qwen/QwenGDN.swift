// One token at a time; mirrors qwen35.cpp build_layer_attn_linear. The
// per-token recurrence is exact, so the GPU's chunked path is gated on it.
import Foundation

final class GDNState {
    var conv: [Float]   // [convDim*(dConv-1)] time-minor
    var rec: [Float]    // [nVHead*dState*dState] head-major
    init(_ cfg: QwenConfig) {
        conv = [Float](repeating: 0, count: cfg.convDim * (cfg.dConv - 1))
        rec = [Float](repeating: 0, count: cfg.nVHead * cfg.dState * cfg.dState)
    }
}

enum GDN {
    static func step(_ n: [Float], _ L: QwenLayer, _ st: GDNState, _ cfg: QwenConfig) -> [Float] {
        let dS = cfg.dState, nK = cfg.nKHead, nV = cfg.nVHead
        let keyDim = cfg.keyDim, valDim = cfg.valueDim, convDim = cfg.convDim
        let kc = cfg.dConv

        var qkvMix = [Float](repeating: 0, count: convDim)
        QB.matvec(L.wqkv!, x: n, out: &qkvMix)
        var z = [Float](repeating: 0, count: valDim)
        QB.matvec(L.wqkvGate!, x: n, out: &z)

        var betaPre = [Float](repeating: 0, count: nV)
        QB.matvec(L.ssmBeta!, x: n, out: &betaPre)
        var alphaPre = [Float](repeating: 0, count: nV)
        QB.matvec(L.ssmAlpha!, x: n, out: &alphaPre)

        let dt = F32T.ptr(L.ssmDt!)
        let aNeg = F32T.ptr(L.ssmA!)
        var beta = [Float](repeating: 0, count: nV)
        var g = [Float](repeating: 0, count: nV)
        for h in 0..<nV {
            beta[h] = sigmoidf(betaPre[h])
            g[h] = softplusf(alphaPre[h] + dt[h]) * aNeg[h]
        }

        // conv1d weight [dConv, convDim], dConv fastest: w(j,c) at c*kc + j.
        let cw = F32T.ptr(L.ssmConv1d!)
        var convOut = [Float](repeating: 0, count: convDim)
        let cs = st.conv
        for c in 0..<convDim {
            var acc: Float = 0
            for j in 0..<(kc - 1) { acc += cs[j * convDim + c] * cw[c * kc + j] }
            acc += qkvMix[c] * cw[c * kc + (kc - 1)]
            convOut[c] = silu(acc)
        }
        if kc >= 2 {
            for j in 0..<(kc - 2) {
                for c in 0..<convDim { st.conv[j * convDim + c] = st.conv[(j + 1) * convDim + c] }
            }
            for c in 0..<convDim { st.conv[(kc - 2) * convDim + c] = qkvMix[c] }
        }

        var q = Array(convOut[0..<keyDim])
        var k = Array(convOut[keyDim..<(2 * keyDim)])
        let v = Array(convOut[(2 * keyDim)..<(2 * keyDim + valDim)])
        Kern.l2normRows(&q, d: dS, rows: nK, eps: cfg.eps)
        Kern.l2normRows(&k, d: dS, rows: nK, eps: cfg.eps)

        let qScale = 1 / Float(dS).squareRoot()
        var o = [Float](repeating: 0, count: valDim)

        q.withUnsafeBufferPointer { qBuf in
        k.withUnsafeBufferPointer { kBuf in
        st.rec.withUnsafeMutableBufferPointer { recBuf in
            let rec = recBuf.baseAddress!
            for hv in 0..<nV {
                let hk = hv % nK  // ggml_repeat tiling
                let qh = qBuf.baseAddress! + hk * dS
                let kh = kBuf.baseAddress! + hk * dS
                let voff = hv * dS
                let S = rec + hv * dS * dS
                let gamma = expf(g[hv])
                let b = beta[hv]
                var sk = [Float](repeating: 0, count: dS)
                for i in 0..<dS {
                    let ki = kh[i]
                    let row = S + i * dS
                    for j in 0..<dS {
                        let s = row[j] * gamma
                        row[j] = s
                        sk[j] += s * ki
                    }
                }
                var d = [Float](repeating: 0, count: dS)
                for j in 0..<dS { d[j] = b * (v[voff + j] - sk[j]) }
                for i in 0..<dS {
                    let ki = kh[i]
                    let qi = qh[i] * qScale
                    let row = S + i * dS
                    for j in 0..<dS {
                        row[j] += ki * d[j]
                        o[voff + j] += row[j] * qi
                    }
                }
            }
        }}}

        let sn = F32T.ptr(L.ssmNorm!)
        Kern.rmsnormRows(&o, d: dS, rows: nV, w: sn, eps: cfg.eps)
        for i in 0..<valDim { o[i] *= silu(z[i]) }

        var out = [Float](repeating: 0, count: cfg.nEmbd)
        QB.matvec(L.ssmOut!, x: o, out: &out)
        return out
    }
}
