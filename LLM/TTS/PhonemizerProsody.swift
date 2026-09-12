// The prosody sub-steps, in place on the phoneme-code buffer, in the C's order.
import Foundation

func applyVelarNasalAssimilation(_ ph: inout [UInt8]) {
    var i = 0
    while i + 1 < ph.count {
        if ph[i] == asc("n")
            && (ph[i + 1] == asc("k") || ph[i + 1] == asc("g")) {
            ph[i] = asc("N")
        }
        i += 1
    }
}

let diphthongBefore = bs("eaOoU")
let allVowelsHappy = bs("aAeEiIOUV03o")

func tenseFinalBareI(_ ph: inout [UInt8]) {
    let prev: UInt8 = ph.count >= 2 ? ph[ph.count - 2] : 0
    let partOfDiph = prev != 0 && strchrHit(diphthongBefore, prev)
    if prev != asc("'") && prev != asc(",") && !partOfDiph {
        var vowelCount = 0
        for i in 0..<ph.count where strchrHit(allVowelsHappy, ph[i]) {
            vowelCount += 1
        }
        if vowelCount > 1 { ph[ph.count - 1] = asc("i") }
    }
}

func applyHappyTensing(_ ph: inout [UInt8]) {
    if ph.count >= 2 && ph[ph.count - 2] == asc("%")
        && ph[ph.count - 1] == asc("I") {
        ph.removeLast(2)
        ph.append(asc("i"))
    } else if ph.count >= 2 && ph[ph.count - 2] == asc("I")
              && ph[ph.count - 1] == asc("2") {
        ph.removeLast(2)
        ph.append(asc("i"))
    } else if ph.count > 0 && ph[ph.count - 1] == asc("I") {
        tenseFinalBareI(&ph)
    }
}

struct Repl {
    var from: [UInt8]
    var to: [UInt8]
    var notFollowedBy: UInt8
}

func applyVowelReduction(_ ph: inout [UInt8]) {
    let reductions = [
        Repl(from: bs("%A:"), to: bs("%@"), notFollowedBy: 0),
        Repl(from: bs("=A:"), to: bs("%@"), notFollowedBy: 0),
        Repl(from: bs("%A"), to: bs("%@"), notFollowedBy: asc("@")),
        Repl(from: bs("=A"), to: bs("%@"), notFollowedBy: asc("@")),
    ]
    for r in reductions {
        var rpos = 0
        while rpos + r.from.count <= ph.count {
            let hit = equalRange(ph, rpos, r.from, r.from.count)
            let after = rpos + r.from.count
            let blocked = hit && r.notFollowedBy != 0
                && after < ph.count && ph[after] == r.notFollowedBy
            if hit && !blocked {
                ph.replaceSubrange(rpos..<after, with: r.to)
                rpos += r.to.count
            } else if blocked {
                rpos = after
            } else {
                rpos += 1
            }
        }
    }
}

func applyLotPlusRMerge(_ ph: inout [UInt8]) {
    var rpos = 0
    while rpos + 2 <= ph.count {
        if ph[rpos] == asc("0") && ph[rpos + 1] == asc("r") {
            ph[rpos] = asc("O")
            ph.insert(asc(":"), at: rpos + 1)
            rpos += 3
        } else {
            rpos += 1
        }
    }
}

func stripMorphemeSchwaR(_ ph: inout [UInt8]) {
    var i = 0
    while i + 2 < ph.count {
        if ph[i] == asc("@") && ph[i + 1] == asc("-")
            && ph[i + 2] == asc("r") {
            ph.removeSubrange(i..<(i + 2))
        } else {
            i += 1
        }
    }
}

let vowelStartsRhotic = bs("aAeEiIoOuUV03@")

func collapseAHashBeforeR(_ ph: inout [UInt8]) {
    var i = 0
    while i + 2 < ph.count {
        if ph[i] == asc("a") && ph[i + 1] == asc("#")
            && ph[i + 2] == asc("r") {
            ph[i] = asc("@")
            ph.remove(at: i + 1)
        }
        i += 1
    }
}

func skipStressMarkers(_ ph: [UInt8], _ from: Int) -> Int {
    var pos = from
    while pos < ph.count
        && (ph[pos] == asc("'") || ph[pos] == asc(",")
            || ph[pos] == asc("%") || ph[pos] == asc("=")) {
        pos += 1
    }
    return pos
}

// A schwa that closes a diphthong keeps its own quality.
func schwaClosesDiphthong(_ ph: [UInt8], _ rpos: Int) -> Bool {
    rpos > 0 && (ph[rpos - 1] == asc("o") || ph[rpos - 1] == asc("A")
        || ph[rpos - 1] == asc("U") || ph[rpos - 1] == asc("O")
        || ph[rpos - 1] == asc("e") || ph[rpos - 1] == asc("i")
        || ph[rpos - 1] == asc("I") || ph[rpos - 1] == asc("a"))
}

// The "r" is absorbed into the rhotic schwa unless a vowel follows it,
// and absorbed anyway when the schwa was unstressed-prefixed.
func absorbsR(_ ph: [UInt8], _ rpos: Int, _ rPos: Int) -> Bool {
    let afterR = skipStressMarkers(ph, rPos + 1)
    let nextIsVowel = afterR < ph.count
        && strchrHit(vowelStartsRhotic, ph[afterR])
    let unstressedPre = rpos > 0
        && (ph[rpos - 1] == asc("=") || ph[rpos - 1] == asc("%"))
    return !nextIsVowel || unstressedPre
}

func applyBareSchwaToRhotic(_ ph: inout [UInt8]) {
    collapseAHashBeforeR(&ph)
    var rpos = 0
    while rpos + 1 < ph.count {
        if ph[rpos] == asc("@") {
            let rPos = skipStressMarkers(ph, rpos + 1)
            if rPos < ph.count && ph[rPos] == asc("r")
                && !schwaClosesDiphthong(ph, rpos) {
                ph[rpos] = asc("3")
                if absorbsR(ph, rpos, rPos) { ph.remove(at: rPos) }
            }
        }
        rpos += 1
    }
}

func rhoticCodeLen(_ ph: [UInt8], _ rpos: Int) -> Int {
    var codeLen = 0
    if ph[rpos] == asc("3") {
        codeLen = 1
        if rpos + 1 < ph.count && ph[rpos + 1] == asc(":") {
            codeLen = 2
        }
    } else if ph[rpos] == asc("U") && rpos + 1 < ph.count
              && ph[rpos + 1] == asc("@") {
        codeLen = 2
    } else if ph[rpos] == asc("A") && rpos + 1 < ph.count
              && ph[rpos + 1] == asc("@") {
        codeLen = 2
    }
    return codeLen
}

// Stress markers may stand between the rhotic and the vowel.
func vowelFollowsMarkers(_ ph: [UInt8], _ afterCode: Int) -> Bool {
    var after = afterCode
    while after < ph.count
        && (ph[after] == asc("'") || ph[after] == asc(",")
            || ph[after] == asc("%") || ph[after] == asc("=")) {
        after += 1
    }
    return after < ph.count
        && strchrHit(vowelStartsRhotic, ph[after])
}

func applyLinkingR(_ ph: inout [UInt8]) {
    var rpos = 0
    while rpos < ph.count {
        let codeLen = rhoticCodeLen(ph, rpos)
        let afterCode = rpos + codeLen
        let alreadyR = afterCode < ph.count
            && ph[afterCode] == asc("r")
        if codeLen > 0 && !alreadyR
            && vowelFollowsMarkers(ph, afterCode) {
            ph.insert(asc("r"), at: afterCode)
            rpos += codeLen
        }
        rpos += 1
    }
}

// The tail after the 'n' distinguishes "-tion(s)" from "-ciency".
func isTionTerminalAfterN(_ ph: [UInt8], _ pn: Int,
                          _ afterN: Int) -> Bool {
    var trulyTerminal = true
    var ki = afterN
    while ki < pn && trulyTerminal {
        let c = ph[ki]
        let isModifier = c == asc("%") || c == asc("=") || c == asc("-")
                      || c == asc("'") || c == asc(",")
        if !isModifier && c != asc("i") && c != asc("z") {
            trulyTerminal = false
        }
        ki += 1
    }
    return trulyTerminal
}

func schwaNStartsTion(_ ph: [UInt8], _ pn: Int, _ ki: Int) -> Bool {
    var foundTion = false
    var ki2 = ki + 1
    var nFound = false
    var nStop = false
    while ki2 < pn && ki2 < ki + 4 && !nFound && !nStop {
        let c2 = ph[ki2]
        let c2Modifier = c2 == asc("-") || c2 == asc("=")
                      || c2 == asc("%")
        if c2 == asc("n") || c2 == asc("N") {
            if isTionTerminalAfterN(ph, pn, ki2 + 1) {
                foundTion = true
            }
            nFound = true
        } else if !c2Modifier {
            nStop = true
        }
        ki2 += 1
    }
    return foundTion
}

func hasTionAfterS(_ ph: [UInt8], _ pn: Int, _ afterS: Int) -> Bool {
    var foundTion = false
    var stop = false
    var ki = afterS
    while ki < pn && ki < afterS + 10 && !foundTion && !stop {
        let c = ph[ki]
        let isModifier = c == asc("=") || c == asc("%") || c == asc("-")
                      || c == asc("'") || c == asc(",")
        if c == asc("@") {
            foundTion = schwaNStartsTion(ph, pn, ki)
            stop = true
        } else if !isModifier {
            stop = true
        }
        ki += 1
    }
    return foundTion
}

func findTionSuffixSPos(_ ph: [UInt8], _ pn: Int) -> Int {
    var sPos = -1
    var k = pn
    while k > 0 && sPos < 0 {
        let idx = k - 1
        if ph[idx] == asc("S") && hasTionAfterS(ph, pn, idx + 1) {
            sPos = idx
        }
        k -= 1
    }
    return sPos
}

func primaryAlreadyOnVowel(_ ph: [UInt8], _ vowelPos: Int) -> Bool {
    var primary = false
    var stop = false
    var bi = vowelPos - 1
    while bi >= 0 && !primary && !stop {
        let bc = ph[bi]
        if bc == asc("'") {
            primary = true
        } else if strchrHit(vowelCodeChars, bc) {
            stop = true
        }
        bi -= 1
    }
    return primary
}

func moveStressToVowel(_ ph: inout [UInt8], _ vowelPos0: Int) {
    var vowelPos = vowelPos0
    let primePos = firstIndexOf(ph, asc("'"))
    let primaryAfter = primePos >= 0 && primePos >= vowelPos
    if !primaryAfter {
        if primePos >= 0 { ph[primePos] = asc(",") }
        if vowelPos > 0 && ph[vowelPos - 1] == asc(",") {
            ph.remove(at: vowelPos - 1)
            vowelPos -= 1
        }
        ph.insert(asc("'"), at: vowelPos)
    }
}

// The .cpp's three multi-char tables here are byte-identical.
let mcUnitAware = stressedMC

struct VowelsBefore {
    var count = 0
    var lastPos = -1
}

func countVowelsBeforeUnitAware(_ ph: [UInt8],
                                _ end: Int) -> VowelsBefore {
    var r = VowelsBefore()
    var vi = 0
    while vi < end {
        let c = ph[vi]
        if c == asc("'") || c == asc(",") || c == asc("%")
            || c == asc("=") {
            vi += 1
        } else {
            var code = [UInt8]()
            var haveCode = false
            var mi = 0
            while mi < mcUnitAware.count && !haveCode {
                let mc = mcUnitAware[mi]
                if vi + mc.count <= end && equalRange(ph, vi, mc,
                                                      mc.count) {
                    code = mc
                    haveCode = true
                }
                mi += 1
            }
            if !haveCode { code = [ph[vi]] }
            if strchrHit(vowelCodeChars, code[0]) {
                r.count += 1
                r.lastPos = vi
            }
            vi += code.count
        }
    }
    return r
}

func applyTionStressFix(_ ph: inout [UInt8]) {
    let sPos = findTionSuffixSPos(ph, ph.count)
    if sPos >= 0 {
        let v = countVowelsBeforeUnitAware(ph, sPos)
        if v.lastPos >= 0 && !primaryAlreadyOnVowel(ph, v.lastPos) {
            moveStressToVowel(&ph, v.lastPos)
        }
    }
}

func matchesOlogyAt(_ ph: [UInt8], _ pn: Int, _ k: Int) -> Bool {
    var matched = false
    if k + 2 < pn && ph[k] == asc("0") && ph[k + 1] == asc("l") {
        var a = k + 2
        while a < pn && (ph[a] == asc("=") || ph[a] == asc("%")
                         || ph[a] == asc(",")) {
            a += 1
        }
        matched = a + 2 < pn && ph[a] == asc("@")
               && ph[a + 1] == asc("d") && ph[a + 2] == asc("Z")
    }
    return matched
}

func applyOlogyStressFix(_ ph: inout [UInt8]) {
    var olPos = -1
    var k = 0
    while k + 2 < ph.count && olPos < 0 {
        if matchesOlogyAt(ph, ph.count, k) { olPos = k }
        k += 1
    }
    if olPos >= 0 {
        let primaryOnOl = olPos > 0 && ph[olPos - 1] == asc("'")
        if !primaryOnOl {
            let prime = firstIndexOf(ph, asc("'"))
            if prime >= 0 && prime < olPos { ph[prime] = asc(",") }
            if olPos > 0 && ph[olPos - 1] == asc(",") {
                ph.remove(at: olPos - 1)
                olPos -= 1
            }
            ph.insert(asc("'"), at: olPos)
        }
    }
}

func lastKPos(_ ph: [UInt8]) -> Int {
    var at = -1
    var i = ph.count
    while i > 0 && at < 0 {
        if ph[i - 1] == asc("k") { at = i - 1 }
        i -= 1
    }
    return at
}

// kPos == 1 gives -1, the same sentinel the C's size_t underflow lands
// on, so the fix is skipped alike.
func unstressedIkPos(_ ph: [UInt8]) -> Int {
    var ikPos = -1
    let kPos = lastKPos(ph)
    if kPos >= 1 && ph[kPos - 1] == asc("I") {
        let iUnstressed = (kPos >= 2
            && (ph[kPos - 2] == asc("=") || ph[kPos - 2] == asc("%")))
            || kPos == 1
        if iUnstressed { ikPos = kPos - 2 }
    }
    return ikPos
}

func applyIcStressFix(_ ph: inout [UInt8]) {
    let ikPos = unstressedIkPos(ph)
    if ikPos >= 0 {
        let v = countVowelsBeforeUnitAware(ph, ikPos)
        let alreadyOnLast = v.lastPos > 0
            && ph[v.lastPos - 1] == asc("'")
        if v.count >= 2 && v.lastPos >= 0 && !alreadyOnLast {
            moveStressToVowel(&ph, v.lastPos)
        }
    }
}

// Removes 0x01 bytes; `rba` marks the positions that carried one.
func stripRuleBoundaryMarkers(_ ph: inout [UInt8],
                              _ rba: inout [Bool]) -> Int {
    var outN = 0
    let any = containsByte(ph, phBnd)
    if any {
        let rbaCap = rba.count
        var w = 0
        var stripped = [UInt8]()
        stripped.reserveCapacity(ph.count)
        for r in 0..<ph.count {
            if ph[r] == phBnd {
                if w > 0 && w - 1 < rbaCap { rba[w - 1] = true }
            } else {
                if w < rbaCap { rba[w] = false }
                stripped.append(ph[r])
                w += 1
            }
        }
        ph = stripped
        outN = w
    }
    return outN
}

func reduceBareZeroAfterUtion(_ ph: inout [UInt8]) {
    var hasUS = false
    var i = 0
    while i + 2 < ph.count && !hasUS {
        if ph[i] == asc("u") && ph[i + 1] == asc(":")
            && ph[i + 2] == asc("S") {
            hasUS = true
        }
        i += 1
    }
    if hasUS {
        for pi in 0..<ph.count where ph[pi] == asc("0") {
            let stressed = pi > 0
                && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
            if !stressed { ph[pi] = asc("@") }
        }
    }
}

func reduceVBetweenSecAndPrimary(_ ph: inout [UInt8]) {
    let pp = firstIndexOf(ph, asc("'"))
    let sp = firstIndexOf(ph, asc(","))
    if pp >= 0 && sp >= 0 && sp < pp {
        for pi in (sp + 1)..<pp where ph[pi] == asc("V") {
            let stressed = pi > 0
                && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
            if !stressed { ph[pi] = asc("@") }
        }
    }
}

// An 'a' before I, U, :, @ or # heads a longer code, so it is not bare.
func aStartsLongerCode(_ ph: [UInt8], _ pi: Int) -> Bool {
    return pi + 1 < ph.count
        && (ph[pi + 1] == asc("I") || ph[pi + 1] == asc("U")
            || ph[pi + 1] == asc(":")
            || ph[pi + 1] == asc("@")
            || ph[pi + 1] == asc("#"))
}

func reduceABetweenPrimaryAndSec(_ ph: inout [UInt8],
                                 _ ruleDerived: Bool) {
    let pp = ruleDerived ? firstIndexOf(ph, asc("'")) : -1
    var sp = ruleDerived ? firstIndexOf(ph, asc(",")) : -1
    if pp >= 0 && sp >= 0 && pp < sp {
        var pi = pp + 1
        while pi < sp {
            let stressed = pi > 0
                && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
            if ph[pi] == asc("a") && !aStartsLongerCode(ph, pi)
                && !stressed {
                ph.insert(asc("#"), at: pi + 1)
                sp += 1
                pi += 1
            }
            pi += 1
        }
    }
}

func reduceZeroBetweenSecAndPrimary(_ ph: inout [UInt8]) {
    let pp = firstIndexOf(ph, asc("'"))
    let sp = firstIndexOf(ph, asc(","))
    if pp >= 0 && sp >= 0 && sp < pp {
        var pi = sp + 1
        while pi < pp {
            if ph[pi] == asc("0") {
                let isVariant = pi + 1 < ph.count
                    && (ph[pi + 1] == asc("#") || ph[pi + 1] == asc("2"))
                if isVariant {
                    pi += 1
                } else {
                    let stressed = pi > 0
                        && (ph[pi - 1] == asc("'")
                            || ph[pi - 1] == asc(","))
                    if !stressed { ph[pi] = asc("@") }
                }
            }
            pi += 1
        }
    }
}

func reduceZeroBetweenPrimaryAndSec(_ ph: inout [UInt8]) {
    let pp = firstIndexOf(ph, asc("'"))
    if pp >= 0 {
        let sp = firstIndexOf(ph, asc(","), from: pp + 1)
        if sp >= 0 {
            var pi = pp + 1
            while pi < sp {
                if ph[pi] == asc("0") {
                    let isVariant = pi + 1 < ph.count
                        && (ph[pi + 1] == asc("#")
                            || ph[pi + 1] == asc("2"))
                    if isVariant {
                        pi += 1
                    } else {
                        let stressed = pi > 0
                            && (ph[pi - 1] == asc("'")
                                || ph[pi - 1] == asc(","))
                        if !stressed { ph[pi] = asc("@") }
                    }
                }
                pi += 1
            }
        }
    }
}

func firstInterVowelIsUnstressed(_ ph: [UInt8], _ start: Int,
                                 _ end: Int) -> Bool {
    var unstressed = false
    var found = false
    var si = start
    while si < end && !found {
        let c = ph[si]
        if strchrHit(vowelCodeChars, c) {
            let sv = si > 0
                && (ph[si - 1] == asc("'") || ph[si - 1] == asc(","))
            if !sv { unstressed = true }
            found = true
        }
        si += 1
    }
    return unstressed
}

func reduceZeroHashBeforePrimary(_ ph: inout [UInt8]) {
    let prime = firstIndexOf(ph, asc("'"))
    if prime >= 0 {
        var pp = prime
        var pi = 0
        while pi < pp {
            if ph[pi] == asc("0") {
                let stressed = pi > 0
                    && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
                let hasHash = pi + 1 < ph.count
                    && ph[pi + 1] == asc("#")
                let is02 = pi + 1 < ph.count && ph[pi + 1] == asc("2")
                if !stressed && is02 {
                    pi += 1
                } else if !stressed && hasHash
                    && !firstInterVowelIsUnstressed(ph, pi + 2, pp) {
                    ph.remove(at: pi + 1)
                    ph[pi] = asc("@")
                    pp -= 1
                }
            }
            pi += 1
        }
    }
}

func bareZeroReduces(_ ph: [UInt8], _ pi: Int, _ rba: [Bool],
                     _ rbaN: Int) -> Bool {
    let marked = pi > 0
        && (ph[pi - 1] == asc("'")
            || ph[pi - 1] == asc(",")
            || ph[pi - 1] == asc("%")
            || ph[pi - 1] == asc("="))
    let isStandalone = pi < rbaN && pi < rba.count && rba[pi]
    var hasPriorVowel = false
    var k = 0
    while k < pi && !hasPriorVowel {
        if strchrHit(vowelCodeChars, ph[k]) { hasPriorVowel = true }
        k += 1
    }
    return !marked && isStandalone && hasPriorVowel
}

func reduceBareZeroBeforePrimary(_ ph: inout [UInt8],
                                 _ rba: [Bool], _ rbaN: Int) {
    let pp = rbaN > 0 ? firstIndexOf(ph, asc("'")) : -1
    if pp >= 0 {
        var pi = 0
        while pi < pp {
            if ph[pi] == asc("0") {
                let isVariant = pi + 1 < ph.count
                    && (ph[pi + 1] == asc("#")
                        || ph[pi + 1] == asc("2"))
                if isVariant {
                    pi += 1
                } else if bareZeroReduces(ph, pi, rba, rbaN) {
                    ph[pi] = asc("@")
                }
            }
            pi += 1
        }
    }
}

func pctSyllableBefore(_ ph: [UInt8], _ pi: Int) -> Bool {
    var pctPos = -1
    for j in 0..<pi where ph[j] == asc("%") { pctPos = j }
    var hasVowelBetween = false
    if pctPos >= 0 {
        var k = pctPos + 1
        while k < pi && !hasVowelBetween {
            if strchrHit(vowelCodeChars, ph[k]) {
                hasVowelBetween = true
            }
            k += 1
        }
    }
    return hasVowelBetween
}

func reduceEPreTonicAfterPctNasal(_ ph: inout [UInt8]) {
    let pp = firstIndexOf(ph, asc("'"))
    if pp >= 0 {
        var pi = 1
        while pi < pp {
            if ph[pi] == asc("E") {
                let isVariant = pi + 1 < ph.count
                    && (ph[pi + 1] == asc("2") || ph[pi + 1] == asc("#"))
                if isVariant {
                    pi += 1
                } else if ph[pi - 1] != asc("'") && ph[pi - 1] != asc(",")
                    && pi + 1 < ph.count && ph[pi + 1] == asc("n")
                    && pctSyllableBefore(ph, pi) {
                    ph[pi] = asc("@")
                }
            }
            pi += 1
        }
    }
}

let step5bVowels = bs("aAeEiIoOuUV03@")
let diphBefore5b = bs("aAeEoOuU")

func findAtLPos(_ ph: [UInt8]) -> Int {
    var at = -1
    var i = 0
    while i + 1 < ph.count && at < 0 {
        if ph[i] == asc("@") && ph[i + 1] == asc("L") { at = i }
        i += 1
    }
    return at
}

func anyVowelBetween(_ ph: [UInt8], _ from: Int,
                     _ to: Int) -> Bool {
    var seen = false
    var k = from
    while k < to && !seen {
        if strchrHit(step5bVowels, ph[k]) { seen = true }
        k += 1
    }
    return seen
}

// A plain 'I' (not I2, I#) right before the syllabic l, not a diphthong tail.
func isTensableI(_ ph: [UInt8], _ pi: Int) -> Bool {
    let isPlainI = ph[pi] == asc("I")
        && !(pi + 1 < ph.count
             && (ph[pi + 1] == asc("2") || ph[pi + 1] == asc("#")))
    let directlyBeforeAl = pi + 2 < ph.count
        && ph[pi + 1] == asc("@") && ph[pi + 2] == asc("L")
    let partOfDiph = pi > 0 && strchrHit(diphBefore5b, ph[pi - 1])
    return isPlainI && directlyBeforeAl && !partOfDiph
}

func postStressIBeforeSyllabicL(_ ph: inout [UInt8]) {
    let sp = firstIndexOf(ph, asc("'"))
    let alPos = sp >= 0 ? findAtLPos(ph) : -1
    if alPos > sp {
        for pi in (sp + 1)..<alPos {
            if isTensableI(ph, pi)
                && anyVowelBetween(ph, sp + 1, pi) {
                ph[pi] = asc("i")
            }
        }
    }
}

// A 3: shortens when unmarked, with a vowel before it, and pre-tonic or
// followed by a secondary the primary's own '%' does not cancel.
func long3ShouldShorten(_ ph: [UInt8], _ pi: Int,
                        _ primePos: Int) -> Bool {
    let isStressed = pi > 0
        && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
    let hasExplicitUnstress = pi > 0 && ph[pi - 1] == asc("%")
    let isPretonic = pi + 2 < primePos
    var hasSecondaryAfter = false
    var q = pi + 2
    while q < ph.count && !hasSecondaryAfter {
        if ph[q] == asc(",") { hasSecondaryAfter = true }
        q += 1
    }
    let pctBeforePrimary = primePos > 0
        && ph[primePos - 1] == asc("%")
    return !isStressed && !hasExplicitUnstress
        && anyVowelBetween(ph, 0, pi)
        && (isPretonic
            || (hasSecondaryAfter && !pctBeforePrimary))
}

func reduce3LongToShort(_ ph: inout [UInt8]) {
    let stress = firstIndexOf(ph, asc("'"))
    var primePos = stress
    var pi = 0
    while stress >= 0 && pi + 1 < ph.count {
        if ph[pi] == asc("3") && ph[pi + 1] == asc(":") {
            if long3ShouldShorten(ph, pi, primePos) {
                ph.remove(at: pi + 1)
                if primePos > 0 { primePos -= 1 }
            } else {
                pi += 2
            }
        } else {
            pi += 1
        }
    }
}

let flapVowelChars = bs("aAeEIiOUVu03@oY")

// "*3n/m/N" is a de-flapped sequence, not a flap.
func deflapBeforeSyllabicNasal(_ ph: inout [UInt8]) {
    var pi = 0
    while pi + 2 < ph.count {
        if ph[pi] == asc("*") && ph[pi + 1] == asc("3")
            && (ph[pi + 2] == asc("n") || ph[pi + 2] == asc("m")
                || ph[pi + 2] == asc("N")) {
            ph[pi] = asc("t")
        }
        pi += 1
    }
}

func flapPrevIsVowel(_ ph: [UInt8], _ pi: Int) -> Bool {
    let prev = ph[pi - 1]
    return prev == asc(":") || prev == asc("#") || prev == asc("r")
        || strchrHit(flapVowelChars, prev)
        || (prev == asc("2") && pi >= 2
            && strchrHit(flapVowelChars, ph[pi - 2]))
        || (prev == asc("L") && pi >= 2 && ph[pi - 2] == asc("@"))
}

// After a "%" or "=" marker the vowel is unstressed unless it leads
// into a syllabic "n" that is not "3:".
func flapAfterMarker(_ ph: [UInt8], _ nxtPos: Int) -> Bool {
    var r = false
    if nxtPos + 1 < ph.count
        && strchrHit(flapVowelChars, ph[nxtPos + 1]) {
        let v2Long = nxtPos + 2 < ph.count
            && ph[nxtPos + 2] == asc(":")
        let n2p = v2Long ? nxtPos + 3 : nxtPos + 2
        let n2n = n2p < ph.count && ph[n2p] == asc("n")
        let v23c = ph[nxtPos + 1] == asc("3") && v2Long
        r = !(n2n && !v23c)
    }
    return r
}

func flapBeforeVowel(_ ph: [UInt8], _ nxtPos: Int,
                     _ nxt: UInt8) -> Bool {
    let is3colon = nxt == asc("3") && nxtPos + 1 < ph.count
        && ph[nxtPos + 1] == asc(":")
    let vowelIsLong = nxtPos + 1 < ph.count
        && ph[nxtPos + 1] == asc(":")
    var next2Pos = vowelIsLong ? nxtPos + 2 : nxtPos + 1
    if nxt == asc("@") && next2Pos < ph.count
        && ph[next2Pos] == asc("L") {
        next2Pos += 1
    }
    let next2IsN = next2Pos < ph.count && ph[next2Pos] == asc("n")
    return !(next2IsN && !is3colon)
}

func flapNextIsUnstressedVowel(_ ph: [UInt8], _ pi: Int) -> Bool {
    let nxtPos = ph[pi + 1] == asc("#") ? pi + 2 : pi + 1
    let nxt: UInt8 = nxtPos < ph.count ? ph[nxtPos] : 0
    var r = false
    if nxt == asc("%") || nxt == asc("=") {
        r = flapAfterMarker(ph, nxtPos)
    } else if nxt != 0 && nxt != asc("'") && nxt != asc(",")
        && strchrHit(flapVowelChars, nxt) {
        r = flapBeforeVowel(ph, nxtPos, nxt)
    }
    return r
}

func applyFlapRule(_ ph: inout [UInt8]) {
    deflapBeforeSyllabicNasal(&ph)
    var pi = 1
    while pi + 1 < ph.count {
        if ph[pi] == asc("t") && flapPrevIsVowel(ph, pi)
            && flapNextIsUnstressedVowel(ph, pi) {
            ph[pi] = asc("*")
        }
        pi += 1
    }
}


func reduceNessSuffix(_ ph: inout [UInt8]) {
    let plen = ph.count
    if plen >= 4 && ph[plen - 1] == asc("s") && ph[plen - 2] == asc("E")
        && ph[plen - 3] == asc(",") && ph[plen - 4] == asc("n") {
        ph[plen - 3] = asc("@")
        ph[plen - 2] = asc("s")
        ph.removeLast()
    } else if plen >= 3 && ph[plen - 1] == asc("s")
        && ph[plen - 2] == asc("E") && ph[plen - 3] == asc("n") {
        ph[plen - 2] = asc("@")
        ph[plen - 1] = asc("s")
    }
}

func prevSyllableIsPrimary(_ ph: [UInt8], _ before: Int) -> Bool {
    var found = false
    var stop = false
    var j = before - 1
    while j >= 0 && !found && !stop {
        let c = ph[j]
        if c == asc("'") { found = true }
        else if c == asc(",") || c == asc("%") { stop = true }
        j -= 1
    }
    return found
}

func nextSyllableIsPrimary(_ ph: [UInt8], _ pn: Int,
                           _ after: Int) -> Bool {
    var found = false
    var stop = false
    var j = after
    while j < pn && !found && !stop {
        let c = ph[j]
        if c == asc("'") { found = true }
        else if c == asc(",") || strchrHit(vowelCodeChars, c) {
            stop = true
        }
        j += 1
    }
    return found
}

func reduceOuHashCompound(_ ph: inout [UInt8]) {
    var i = 0
    while i + 2 < ph.count {
        if ph[i] == asc("o") && ph[i + 1] == asc("U")
            && ph[i + 2] == asc("#") {
            let selfPrimary = i > 0 && ph[i - 1] == asc("'")
            if selfPrimary {
                ph.removeSubrange((i + 1)..<(i + 3))
                ph[i] = asc("0")
            } else if prevSyllableIsPrimary(ph, i) {
                ph.removeSubrange((i + 1)..<(i + 3))
                ph[i] = asc("@")
            } else if nextSyllableIsPrimary(ph, ph.count, i + 3) {
                ph.removeSubrange((i + 1)..<(i + 3))
                ph[i] = asc("@")
            } else {
                ph.remove(at: i + 2)
            }
        }
        i += 1
    }
}

func elideIcallySchwa(_ ph: inout [UInt8]) {
    if ph.count >= 4 && ph[ph.count - 4] == asc("k")
        && ph[ph.count - 3] == asc("@") && ph[ph.count - 2] == asc("l")
        && ph[ph.count - 1] == asc("i") {
        ph[ph.count - 3] = asc("l")
        ph[ph.count - 2] = asc("i")
        ph.removeLast()
    }
}

func endsInSyllabicN(_ ph: [UInt8]) -> Bool {
    return ph.count >= 2 && ph[ph.count - 1] == asc("-")
        && ph[ph.count - 2] == asc("n")
}

// Stress markers are transparent to the vowel-group count.
func countVowelGroupsBefore(_ ph: [UInt8], _ end: Int) -> Int {
    var vgroups = 0
    var inV = false
    for vi in 0..<end {
        let c = ph[vi]
        if c != asc("'") && c != asc(",") && c != asc("%")
            && c != asc("=") {
            if strchrHit(vowelCodeChars, c) {
                if !inV { vgroups += 1; inV = true }
            } else {
                inV = false
            }
        }
    }
    return vgroups
}

func hasUnstressedPrefixBefore(_ ph: [UInt8],
                               _ primePos: Int) -> Bool {
    var found = false
    var k = 0
    while k < primePos && !found {
        if ph[k] == asc("%") || ph[k] == asc("=") { found = true }
        k += 1
    }
    return found
}

func addCompoundSyllabicNStress(_ ph: inout [UInt8]) {
    if endsInSyllabicN(ph) && containsByte(ph, asc("'")) {
        var snStart = ph.count - 2
        if snStart > 0 && ph[snStart - 1] == asc("?") { snStart -= 1 }
        let vgroups = countVowelGroupsBefore(ph, snStart)
        let primePos = firstIndexOf(ph, asc("'"))
        if vgroups >= 2 && !hasUnstressedPrefixBefore(ph, primePos)
            && (snStart == 0 || ph[snStart - 1] != asc(",")) {
            ph.insert(asc(","), at: snStart)
        }
    }
}

let maxSyl = 256

struct Syl {
    var pos = 0
    var code = [UInt8]()
    var codeLen = 0
    var codeFromTable = false
    var level = -1
    var isPrimary = false
    var isSecondary = false
    var stressable = false
}

// Reduces right-to-left so positions stay valid.

// '@-' is a morpheme boundary, not a syllable nucleus.
func collectZeroSyllables(_ ph: [UInt8], _ max: Int) -> [Syl] {
    var syls = [Syl]()
    var pi = 0
    var curLevel = -1
    while pi < ph.count && syls.count < max {
        let c = ph[pi]
        if c == asc("'") { curLevel = 4; pi += 1 }
        else if c == asc(",") { curLevel = 2; pi += 1 }
        else if c == asc("%") || c == asc("=") {
            curLevel = 1; pi += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                if pc.len == 1 && pc.code[0] == asc("@")
                    && pi + 1 < ph.count && ph[pi + 1] == asc("-") {
                    pi += pc.len
                } else {
                    var s = Syl()
                    s.pos = pi
                    s.code = pc.code
                    s.codeLen = pc.len
                    s.level = curLevel
                    syls.append(s)
                    curLevel = -1
                    pi += pc.len
                }
            } else if ph[pi] == asc("-") {
                pi += 1
            } else {
                pi += pc.len
            }
        }
    }
    return syls
}

func reduceZeroSyllables(_ ph: inout [UInt8], _ syls: [Syl]) {
    let n = syls.count
    var v = n - 1
    while v >= 0 {
        let vnum = v + 1
        var eligible = syls[v].codeLen == 1
            && syls[v].code[0] == asc("0")
            && syls[v].level <= 1
            && vnum != 1 && vnum != n
        if eligible && vnum == n - 1 && syls[n - 1].level <= 1 {
            eligible = false
        }
        if eligible { ph[syls[v].pos] = asc("@") }
        v -= 1
    }
}

func reduceZeroDiminished(_ ph: inout [UInt8]) {
    if containsByte(ph, asc("0")) {
        let syls = collectZeroSyllables(ph, maxSyl)
        reduceZeroSyllables(&ph, syls)
    }
}

func demoteAdjacentPrimaries(_ ph: inout [UInt8], _ phIn: [UInt8]) {
    if containsByte(phIn, asc("'")) {
        var syls = [Syl]()
        var pi = 0
        var prim = false
        while pi < ph.count && syls.count < maxSyl {
            let c = ph[pi]
            if c == asc("'") { prim = true; pi += 1 }
            else if c == asc(",") || c == asc("%") || c == asc("=") {
                prim = false; pi += 1
            } else {
                let pc = findPhonemeCode(ph, ph.count, pi)
                if isVowelCode(pc.code, 0, pc.len) {
                    var s = Syl()
                    s.pos = pi
                    s.isPrimary = prim
                    syls.append(s)
                    prim = false
                }
                pi += pc.len
            }
        }
        var si = syls.count - 1
        while si >= 1 {
            if syls[si].isPrimary && syls[si - 1].isPrimary {
                if syls[si - 1].pos > 0 {
                    let cp = syls[si - 1].pos - 1
                    if cp < ph.count && ph[cp] == asc("'") {
                        ph[cp] = asc(",")
                        syls[si - 1].isPrimary = false
                    }
                }
            }
            si -= 1
        }
    }
}

func posAfterPctVowel(_ ph: [UInt8], _ from: Int,
                      _ primaryPos: Int) -> Int {
    var pctScan = from
    var pastVowel = false
    while pctScan < primaryPos && !pastVowel {
        let pc = findPhonemeCode(ph, ph.count, pctScan)
        if isVowelCode(pc.code, 0, pc.len) {
            pctScan += pc.len
            pastVowel = true
        } else {
            pctScan += 1
        }
    }
    return pctScan
}

func posAfterFirstVowel(_ ph: [UInt8], _ primaryPos: Int) -> Int {
    var genScan = 0
    var foundFirst = false
    while genScan < primaryPos && !foundFirst {
        if ph[genScan] == asc("'") || ph[genScan] == asc(",")
            || ph[genScan] == asc("%") {
            genScan += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, genScan)
            genScan += pc.len
            foundFirst = isVowelCode(pc.code, 0, pc.len)
        }
    }
    return genScan
}

// Reduction may start after a secondary or a "%"-syllable before the
// primary, else after the first (protected) vowel; -1 when nothing qualifies.
func aReductionScanStart(_ ph: [UInt8], _ primaryPos: Int) -> Int {
    let sec = firstIndexOf(ph, asc(","))
    let pct = firstIndexOf(ph, asc("%"))
    var scanStart = -1
    if sec >= 0 && sec < primaryPos { scanStart = sec + 1 }
    if pct >= 0 && pct < primaryPos {
        let pctScan = posAfterPctVowel(ph, pct + 1, primaryPos)
        if pctScan < primaryPos
            && (scanStart < 0 || pctScan < scanStart) {
            scanStart = pctScan
        }
    }
    if scanStart < 0 {
        let genScan = posAfterFirstVowel(ph, primaryPos)
        if genScan < primaryPos { scanStart = genScan }
    }
    return scanStart
}

// "aI", "aU", "a:", "a@" and "a#" are codes in their own right, and an
// "a" directly after a stress mark is protected.
func aIsReducible(_ ph: [UInt8], _ pi: Int) -> Bool {
    let isDiphthongStart = pi + 1 < ph.count
        && (ph[pi + 1] == asc("I") || ph[pi + 1] == asc("U")
            || ph[pi + 1] == asc(":") || ph[pi + 1] == asc("@")
            || ph[pi + 1] == asc("#"))
    let protectedVowel = pi > 0
        && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(",")
            || ph[pi - 1] == asc("%"))
    return !isDiphthongStart && !protectedVowel
}

func reduceABeforePrimary(_ ph: inout [UInt8], _ ruleDerived: Bool) {
    let prim = ruleDerived ? firstIndexOf(ph, asc("'")) : -1
    if prim >= 0 {
        var primaryPos = prim
        let scanStart = aReductionScanStart(ph, primaryPos)
        if scanStart >= 0 {
            var pi = scanStart
            while pi < primaryPos {
                if ph[pi] == asc("a") && aIsReducible(ph, pi) {
                    ph.insert(asc("#"), at: pi + 1)
                    primaryPos += 1
                    pi += 1
                }
                pi += 1
            }
        }
    }
}

func isUnstressedCode(_ c: [UInt8], _ cn: Int) -> Bool {
    isSchwaFamily(c, cn)
        || (cn == 2 && c[0] == asc("I") && c[1] == asc("#"))
}

// "E#" and "E2" are codes in their own right, not a bare "E".
func eIsVariant(_ ph: [UInt8], _ pi: Int) -> Bool {
    pi + 1 < ph.count
        && (ph[pi + 1] == asc("#") || ph[pi + 1] == asc("2"))
}

func eIsStressed(_ ph: [UInt8], _ pi: Int) -> Bool {
    pi > 0 && (ph[pi - 1] == asc("'") || ph[pi - 1] == asc(","))
}

func reduceEAt(_ ph: inout [UInt8], _ pi: Int,
               _ beforeN: Bool) -> Int {
    var grew = 0
    if beforeN {
        ph[pi] = asc("@")
    } else {
        ph[pi] = asc("I")
        ph.insert(asc("2"), at: pi + 1)
        grew = 1
    }
    return grew
}

// `end` is the later mark and moves right with each insertion.
func reduceEBetweenStresses(_ ph: inout [UInt8], _ start: Int,
                            _ end: inout Int) {
    var pi = start
    while pi < end {
        if ph[pi] == asc("E") {
            if eIsVariant(ph, pi) {
                pi += 1
            } else if !eIsStressed(ph, pi) {
                let beforeN = pi + 1 < end && ph[pi + 1] == asc("n")
                let grew = reduceEAt(&ph, pi, beforeN)
                end += grew
                pi += grew
            }
        }
        pi += 1
    }
}

// A "%"-initial word's first syllable is already unstressed, so every
// bare "E" after it becomes "I2" whether or not an "n" follows.
func reduceEAfterPctLead(_ ph: inout [UInt8],
                         _ primaryPos: inout Int) {
    var scanStart = 1
    while scanStart < ph.count
        && (ph[scanStart] == asc("'") || ph[scanStart] == asc(",")
            || ph[scanStart] == asc("%")
            || ph[scanStart] == asc("=")) {
        scanStart += 1
    }
    if scanStart < ph.count {
        let pc = findPhonemeCode(ph, ph.count, scanStart)
        if isVowelCode(pc.code, 0, pc.len) { scanStart += pc.len }
    }
    var pi = scanStart
    while pi < primaryPos {
        if ph[pi] == asc("E") {
            if eIsVariant(ph, pi) {
                pi += 1
            } else if !eIsStressed(ph, pi) {
                primaryPos += reduceEAt(&ph, pi, false)
                pi += 1
            }
        }
        pi += 1
    }
}

func firstStressMark(_ ph: [UInt8]) -> Int {
    var firstStress = -1
    var pi = 0
    while pi < ph.count && firstStress < 0 {
        if ph[pi] == asc("'") || ph[pi] == asc(",") { firstStress = pi }
        pi += 1
    }
    return firstStress
}

func countVowelsAfter(_ ph: [UInt8], _ pi: Int,
                      _ lastCode: inout [UInt8],
                      _ lastCodeLen: inout Int) -> Int {
    var vowelsAfter = 0
    lastCode = []
    lastCodeLen = 0
    var pj = pi + 1
    while pj < ph.count {
        let pc = findPhonemeCode(ph, ph.count, pj)
        if isVowelCode(pc.code, 0, pc.len) {
            vowelsAfter += 1
            lastCode = pc.code
            lastCodeLen = pc.len
        }
        pj += pc.len
    }
    return vowelsAfter
}

// A single trailing syllable that is itself reduced does not justify
// reducing this one.
func eShouldReduce(_ ph: [UInt8], _ pi: Int) -> Bool {
    var lastCode = [UInt8]()
    var lastCodeLen = 0
    let vowelsAfter = countVowelsAfter(ph, pi, &lastCode, &lastCodeLen)
    return vowelsAfter >= 2
        || (vowelsAfter == 1
            && !isUnstressedCode(lastCode, lastCodeLen))
}

func reduceEAfterFirstStress(_ ph: inout [UInt8]) {
    let firstStress = firstStressMark(ph)
    if firstStress >= 0 {
        var pi = firstStress + 1
        while pi < ph.count {
            if ph[pi] != asc("E") {
                pi += 1
            } else if eIsVariant(ph, pi) {
                pi += 2
            } else {
                var grew = 0
                if !eIsStressed(ph, pi) && eShouldReduce(ph, pi) {
                    let beforeN = pi + 1 < ph.count
                        && ph[pi + 1] == asc("n")
                    grew = reduceEAt(&ph, pi, beforeN)
                }
                pi += grew + 1
            }
        }
    }
}

func reduceEUnstressed(_ ph: inout [UInt8], _ rbaN: Int) {
    let ruleDerived = rbaN > 0
    let prim = firstIndexOf(ph, asc("'"))
    if prim >= 0 {
        var primaryPos = prim
        var secondaryPos = firstIndexOf(ph, asc(","))
        if ruleDerived && secondaryPos >= 0
            && secondaryPos < primaryPos {
            reduceEBetweenStresses(&ph, secondaryPos + 1, &primaryPos)
        }
        if ruleDerived && secondaryPos >= 0
            && secondaryPos > primaryPos {
            reduceEBetweenStresses(&ph, primaryPos + 1, &secondaryPos)
        }
        if ph.count > 0 && ph[0] == asc("%") {
            reduceEAfterPctLead(&ph, &primaryPos)
        }
        if ruleDerived {
            reduceEAfterFirstStress(&ph)
        }
    }
}


func lastIndexOf(_ ph: [UInt8], _ c: UInt8) -> Int {
    var i = ph.count
    var at = -1
    while i > 0 && at < 0 {
        if ph[i - 1] == c { at = i - 1 }
        i -= 1
    }
    return at
}

func hasPrimaryBetween(_ ph: [UInt8], _ from: Int, _ to: Int) -> Bool {
    var i = from
    while i < to && ph[i] != asc("'") { i += 1 }
    return i < to
}

func lastShiftableVowel(_ ph: [UInt8], _ primPos: Int,
                        _ eqPos: Int) -> Int {
    var lastSvPos = -1
    var scan = primPos + 1
    while scan < eqPos {
        let c = ph[scan]
        if c == asc("'") || c == asc(",") || c == asc("%")
            || c == asc("=") || c == asc("*") {
            scan += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, scan)
            let isStressed = scan > 0
                && (ph[scan - 1] == asc("'")
                    || ph[scan - 1] == asc(","))
            if isVowelCode(pc.code, 0, pc.len)
                && !isWeakVowel(pc.code, pc.len) && !isStressed {
                lastSvPos = scan
            }
            scan += pc.len
        }
    }
    return lastSvPos
}

func firstVowelFrom(_ ph: [UInt8], _ from: Int) -> Int {
    var pos = from
    var foundVowel = false
    while pos < ph.count && !foundVowel {
        let pc = findPhonemeCode(ph, ph.count, pos)
        if isVowelCode(pc.code, 0, pc.len) {
            foundVowel = true
        } else {
            pos += pc.len
        }
    }
    return pos
}

// Fires only when a "=" follows the primary with no second primary between;
// the caller then suppresses the backward secondary pass.
func applyEqualsSuffixStressShift(_ ph: inout [UInt8]) -> Bool {
    var fired = false
    let primPos = firstIndexOf(ph, asc("'"))
    let eqPos = primPos >= 0 ? lastIndexOf(ph, asc("=")) : -1
    let shiftable = eqPos >= 0 && primPos < eqPos
        && !hasPrimaryBetween(ph, primPos + 1, eqPos)
    if shiftable {
        let lastSvPos = lastShiftableVowel(ph, primPos, eqPos)
        if lastSvPos >= 0 {
            let primVowelPos = firstVowelFrom(ph, primPos + 1)
            if primVowelPos != lastSvPos {
                ph[primPos] = asc(",")
                if lastSvPos > 0 && ph[lastSvPos - 1] == asc("*") {
                    ph[lastSvPos - 1] = asc("t")
                }
                ph.insert(asc("'"), at: lastSvPos)
                fired = true
            }
        }
    }
    return fired
}


func collectMarkedSyllables(_ ph: [UInt8]) -> [Syl] {
    var syls = [Syl]()
    var pi = 0
    var prim = false
    var sec = false
    while pi < ph.count && syls.count < maxSyl {
        let c = ph[pi]
        if c == asc("'") { prim = true; pi += 1 }
        else if c == asc(",") { sec = true; pi += 1 }
        else if c == asc("%") || c == asc("=") { pi += 1 }
        else {
            let pc = findPhonemeCode(ph, ph.count, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                var s = Syl()
                s.pos = pi
                s.isPrimary = prim
                s.isSecondary = sec
                syls.append(s)
                prim = false
                sec = false
            }
            pi += pc.len
        }
    }
    return syls
}

func commaBeforeSyllable(_ ph: [UInt8], _ sylPos: Int) -> Int {
    var bp = sylPos - 1
    while bp >= 0 && ph[bp] != asc(",") && ph[bp] != asc("'")
        && ph[bp] != asc("%") {
        bp -= 1
    }
    let hit = bp >= 0 && ph[bp] == asc(",")
    return hit ? bp : -1
}

// A rule boundary between the two marks means they came from different rules.
func sameRuleSpan(_ a: Int, _ b: Int, _ rba: [Bool],
                  _ rbaN: Int) -> Bool {
    let lo = min(a, b)
    let hi = max(a, b)
    var sameRule = true
    var rp = lo
    while rp < hi && rp < rbaN && rp < rba.count && sameRule {
        if rba[rp] { sameRule = false }
        rp += 1
    }
    return sameRule
}

// Drops a "," one syllable from the primary when 5a did not run and the
// "," is not inside the same rule as the primary.
func cleanupAdjacentSecondary(_ ph: inout [UInt8],
                              _ step5aRan: Bool,
                              _ isRuleLeadingComma: Bool,
                              _ rba: [Bool], _ rbaN: Int) {
    let live = !step5aRan && !isRuleLeadingComma && rbaN > 0
        && containsByte(ph, asc("'")) && containsByte(ph, asc(","))
    if live {
        let syls = collectMarkedSyllables(ph)
        let primIdx = findPrimarySyllable(syls)
        if primIdx >= 0 {
            let primPhPos = firstIndexOf(ph, asc("'"))
            var toRemove = [Int]()
            for sj in 0..<syls.count {
                let adjacent = syls[sj].isSecondary
                    && (sj == primIdx + 1 || sj == primIdx - 1)
                let commaPos = adjacent
                    ? commaBeforeSyllable(ph, syls[sj].pos) : -1
                if commaPos >= 0
                    && !sameRuleSpan(commaPos, primPhPos, rba, rbaN) {
                    toRemove.append(commaPos)
                }
            }
            sortMarksDesc(&toRemove)
            for pos in toRemove { ph.remove(at: pos) }
        }
    }
}


func isSchwaFamily(_ code: [UInt8], _ n: Int) -> Bool {
    isSchwaVowel(code, n)
        || (n == 2 && code[0] == asc("@")
            && (code[1] == asc("2") || code[1] == asc("5")
                || code[1] == asc("L")))
}

// The C re-reads a single-byte code through a pointer into ph after this
// step's insertions, so a later read sees the byte the buffer holds now.
func trochLiveCode(_ s: Syl, _ ph: [UInt8]) -> [UInt8] {
    var r = s.code
    if !s.codeFromTable && s.codeLen == 1 {
        r = [s.pos < ph.count ? ph[s.pos] : 0]
    }
    return r
}

func collectTrochaicSyllables(_ ph: [UInt8]) -> [Syl] {
    var syls = [Syl]()
    var pi = 0
    var curLevel = -1
    while pi < ph.count && syls.count < maxSyl {
        let c = ph[pi]
        if c == asc("'") { curLevel = 4; pi += 1 }
        else if c == asc(",") { curLevel = 2; pi += 1 }
        else if c == asc("%") || c == asc("=") {
            curLevel = 1; pi += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                var s = Syl()
                s.pos = pi
                s.code = pc.code
                s.codeLen = pc.len
                s.codeFromTable = pc.len > 1
                s.level = curLevel
                syls.append(s)
            }
            curLevel = -1
            pi += pc.len
        }
    }
    return syls
}

func effectiveLevel(_ syls: [Syl], _ ph: [UInt8], _ sv: Int,
                    _ step: Int) -> Int {
    var lv = -1
    var nv = sv + step
    while nv >= 0 && nv < syls.count
        && isSchwaFamily(trochLiveCode(syls[nv], ph),
                         syls[nv].codeLen) {
        nv += step
    }
    if nv >= 0 && nv < syls.count { lv = syls[nv].level }
    return lv
}

// With no primary in the input, the first trochaic assignment becomes the
// primary: the pick_last one is pulled out and shifted indices corrected.
func promoteTrochaicPrimary(_ ph: inout [UInt8],
                            _ syls: inout [Syl], _ sv: Int) {
    let n = syls.count
    let pp = firstIndexOf(ph, asc("'"))
    if pp >= 0 {
        ph.remove(at: pp)
        for k in 0..<n {
            if syls[k].pos > pp { syls[k].pos -= 1 }
            if syls[k].level == 4 { syls[k].level = -1 }
        }
    }
    ph.insert(asc("'"), at: syls[sv].pos)
    for k in (sv + 1)..<n { syls[k].pos += 1 }
    syls[sv].level = 4
}

func insertTrochaicSecondary(_ ph: inout [UInt8],
                             _ syls: inout [Syl], _ sv: Int) {
    ph.insert(asc(","), at: syls[sv].pos)
    for k in (sv + 1)..<syls.count { syls[k].pos += 1 }
}

func trochaicCompoundPrefix(_ ph: inout [UInt8], _ phIn: [UInt8],
                            _ step5aRan: Bool,
                            _ startsWithSecondary: Bool) {
    let live = !step5aRan && !startsWithSecondary
        && containsByte(ph, asc("'"))
        && containsByte(phIn, asc(","))
    if live {
        var syls = collectTrochaicSyllables(ph)
        let n = syls.count
        let inputHasPrimary = containsByte(phIn, asc("'"))
            || containsByte(phIn, asc("="))
        var trochaicPrimaryDone = false
        var stopOuter = false
        var sv = 0
        while sv < n && !stopOuter {
            let candidate = syls[sv].level == -1
                && !isWeakVowel(trochLiveCode(syls[sv], ph),
                                syls[sv].codeLen)
                && effectiveLevel(syls, ph, sv, -1) <= 1
                && effectiveLevel(syls, ph, sv, 1) <= 1
            if candidate {
                if !inputHasPrimary && !trochaicPrimaryDone {
                    promoteTrochaicPrimary(&ph, &syls, sv)
                    trochaicPrimaryDone = true
                } else {
                    insertTrochaicSecondary(&ph, &syls, sv)
                    stopOuter = true
                }
            }
            sv += 1
        }
    }
}


func collectFinalSyllables(_ ph: [UInt8]) -> [Syl] {
    var syls = [Syl]()
    var pi = 0
    var curLevel = -1
    while pi < ph.count && syls.count < maxSyl {
        let c = ph[pi]
        if c == asc("'") { curLevel = 4; pi += 1 }
        else if c == asc(",") { curLevel = 2; pi += 1 }
        else if c == asc("%") || c == asc("=") {
            curLevel = 1; pi += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                if !isHyphenatedSchwa(ph, pi, pc.code, pc.len) {
                    var s = Syl()
                    s.pos = pi
                    s.code = pc.code
                    s.codeLen = pc.len
                    s.level = curLevel
                    syls.append(s)
                    curLevel = -1
                }
                pi += pc.len
            } else if ph[pi] == asc("-") {
                pi += 1
            } else {
                pi += pc.len
            }
        }
    }
    return syls
}

func finalSyllableSecondary(_ ph: inout [UInt8]) {
    if containsByte(ph, asc("'")) {
        let syls = collectFinalSyllables(ph)
        let n = syls.count
        if n >= 2 {
            let last = syls[n - 1]
            let prev = syls[n - 2]
            let prevUnstressed =
                isReducedOrSchwa(prev.code, prev.codeLen)
                || prev.level == 1
            if last.level == -1
                && !isWeakVowel(last.code, last.codeLen)
                && prevUnstressed {
                ph.insert(asc(","), at: last.pos)
            }
        }
    }
}


// Schwa, the schwa family and bare 3 are too weak to be the strong vowel here.
func isWeakNucleusCode(_ code: [UInt8], _ cl: Int) -> Bool {
    return (cl == 1 && code[0] == asc("@"))
        || (cl == 2 && code[0] == asc("@")
            && (code[1] == asc("2") || code[1] == asc("5")
                || code[1] == asc("L")))
        || (cl == 1 && code[0] == asc("3"))
}

func hasStrongAfter(_ ph: [UInt8], _ pn: Int, _ from: Int) -> Bool {
    var pi = from
    var unst = false
    var found = false
    var stop = false
    while pi < pn && !found && !stop {
        let c = ph[pi]
        if c == asc("%") || c == asc("=") || c == asc(",") {
            unst = true; pi += 1
        } else if c == asc("'") {
            stop = true
        } else {
            let pc = findPhonemeCode(ph, pn, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                if !unst && !isWeakNucleusCode(pc.code, pc.len) {
                    found = true
                }
                unst = false
            }
            pi += pc.len
        }
    }
    return found
}

let strongDiph = bsv([
    "oU", "aI", "eI", "aU", "OI", "aI@", "aI3", "aU@", "i:",
    "u:", "A:", "E:", "3:", "o:", "U:",
])

func isWeakTwoCharVowel(_ code: [UInt8]) -> Bool {
    (code[0] == asc("@") && (code[1] == asc("2") || code[1] == asc("5")
                             || code[1] == asc("L")))
        || (code[0] == asc("I") && (code[1] == asc("#")
                                    || code[1] == asc("2")))
        || (code[0] == asc("a") && code[1] == asc("#"))
}

// After a compound's "=", a bare "i" and "i@" are also too weak for the
// primary.
func isWeakTailVowel(_ code: [UInt8], _ n: Int) -> Bool {
    var r = false
    if n == 1 {
        r = code[0] == asc("@") || code[0] == asc("3")
            || code[0] == asc("i")
    } else if n == 2 {
        r = isWeakTwoCharVowel(code)
            || (code[0] == asc("i") && code[1] == asc("@"))
    }
    return r
}

func isReducedVowel(_ code: [UInt8], _ n: Int) -> Bool {
    (n == 2 && isWeakTwoCharVowel(code))
        || (n == 1 && code[0] == asc("i"))
}

func isSchwaVowel(_ code: [UInt8], _ n: Int) -> Bool {
    n == 1 && (code[0] == asc("@") || code[0] == asc("3"))
}

// A compound tail with no strong vowel after the last "=" takes the primary
// last.
func computePickLast(_ ph: [UInt8], _ forceFinalStress: Bool) -> Bool {
    var pickLast = forceFinalStress
    var lastEq = -1
    var i = ph.count
    while i > 0 && lastEq < 0 {
        if ph[i - 1] == asc("=") { lastEq = i - 1 }
        i -= 1
    }
    if lastEq >= 0 {
        var hasStrong = false
        var pi2 = lastEq + 1
        while pi2 < ph.count && !hasStrong {
            let pc = findPhonemeCode(ph, ph.count, pi2)
            if isVowelCode(pc.code, 0, pc.len)
                && !isWeakTailVowel(pc.code, pc.len) {
                hasStrong = true
            }
            pi2 += pc.len
        }
        pickLast = !hasStrong
    }
    return pickLast
}

func leadingSecondaryComma(_ ph: [UInt8]) -> Int {
    var spi = 0
    while spi < ph.count && ph[spi] != asc("'") && ph[spi] != asc(",") {
        spi += 1
    }
    let hit = spi < ph.count && ph[spi] == asc(",") && spi > 0
    return hit ? spi : -1
}

func posAfterSecondaryVowel(_ ph: [UInt8], _ commaPi: Int) -> Int {
    var scan = commaPi + 1
    while scan < ph.count
        && (ph[scan] == asc("'") || ph[scan] == asc(",")
            || ph[scan] == asc("%") || ph[scan] == asc("=")) {
        scan += 1
    }
    if scan < ph.count {
        let pc = findPhonemeCode(ph, ph.count, scan)
        if isVowelCode(pc.code, 0, pc.len) { scan += pc.len }
    }
    return scan
}

// A strong diphthong after a leading secondary moves the primary to the last.
func strongDiphthongAfterSecondary(_ ph: [UInt8]) -> Bool {
    var found = false
    let commaPi = leadingSecondaryComma(ph)
    if commaPi >= 0 {
        var scan = posAfterSecondaryVowel(ph, commaPi)
        while scan < ph.count && !found && ph[scan] != asc("'") {
            let pc = findPhonemeCode(ph, ph.count, scan)
            var si = 0
            while si < strongDiph.count && !found {
                let sd = strongDiph[si]
                if pc.len == sd.count
                    && equalRange(pc.code, 0, sd, sd.count) {
                    found = true
                }
                si += 1
            }
            if !found { scan += pc.len }
        }
    }
    return found
}

// `done` ends the walk, at a pre-existing "'" or where the primary will go.
struct PrimaryScan {
    var pos = 0
    var insertPos = -1
    var lastStrongPos = -1
    var lastSchwaPos = -1
    var secondaryVowelPos = -1
    var unstressed = false
    var secondaryNext = false
    var done = false
}

func primaryScanVowel(_ s: inout PrimaryScan, _ ph: [UInt8],
                      _ code: [UInt8], _ codeLen: Int,
                      _ pickLast: Bool, _ secPickLast: Bool) {
    if s.secondaryNext {
        s.secondaryNext = false
        s.unstressed = false
        if s.secondaryVowelPos < 0 { s.secondaryVowelPos = s.pos }
        s.pos += codeLen
    } else if s.unstressed {
        s.unstressed = false
        s.pos += codeLen
    } else if isReducedVowel(code, codeLen) {
        s.pos += codeLen
    } else if isSchwaVowel(code, codeLen) {
        if !hasStrongAfter(ph, ph.count, s.pos + codeLen) {
            s.lastSchwaPos = s.pos
            if !pickLast && s.insertPos < 0 { s.insertPos = s.pos }
        }
        s.pos += codeLen
    } else if pickLast {
        if secPickLast && codeLen == 1 && code[0] == asc("I") {
            s.lastSchwaPos = s.pos
        } else {
            s.lastStrongPos = s.pos
        }
        s.pos += codeLen
    } else {
        s.insertPos = s.pos
        s.done = true
    }
}

func scanForPrimary(_ ph: [UInt8], _ pickLast: Bool,
                    _ ignoreComma: Bool,
                    _ secPickLast: Bool) -> PrimaryScan {
    var s = PrimaryScan()
    while s.pos < ph.count && !s.done {
        let c = ph[s.pos]
        if c == asc("'") { s.done = true }
        else if c == asc(",") {
            if !ignoreComma || s.pos == 0 { s.secondaryNext = true }
            s.pos += 1
        } else if c == asc("%") || c == asc("=") {
            s.unstressed = true
            s.pos += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, s.pos)
            if isVowelCode(pc.code, 0, pc.len) {
                primaryScanVowel(&s, ph, pc.code, pc.len,
                                 pickLast, secPickLast)
            } else {
                s.pos += pc.len
            }
        }
    }
    if pickLast {
        s.insertPos = s.lastStrongPos >= 0 ? s.lastStrongPos
                                           : s.lastSchwaPos
    }
    return s
}

// A primary on a schwa is dropped after a leading "%" or beside a secondary.
func suppressPrimaryOnSchwa(_ ph: [UInt8], _ insertPos: Int,
                            _ secondaryVowelPos: Int) -> Bool {
    let atSchwaOrR = insertPos >= 0 && insertPos < ph.count
        && (ph[insertPos] == asc("@")
            || (ph[insertPos] == asc("3")
                && (insertPos + 1 >= ph.count
                    || ph[insertPos + 1] != asc(":"))))
    let pctLead = ph.count > 0 && ph[0] == asc("%")
    let hasSecondary = secondaryVowelPos >= 0
    return atSchwaOrR && (pctLead || hasSecondary)
}

// 3 and 8 are phoneme codes in their own right, never variant digits.
func isVariantDigit(_ c: UInt8) -> Bool {
    c >= asc("1") && c <= asc("9") && c != asc("3") && c != asc("8")
}

func stripAHashVariantDigits(_ ph: inout [UInt8], _ at: Int) {
    let hasVariantDigit = at + 3 < ph.count && isVariantDigit(ph[at + 3])
    if hasVariantDigit {
        ph.remove(at: at + 2)
        while at + 2 < ph.count && isVariantDigit(ph[at + 2]) {
            ph.remove(at: at + 2)
        }
    }
}

// Last resort when no vowel took the primary: stress "a#" as "a".
func stressAHashLastResort(_ ph: inout [UInt8]) {
    var pi2 = 0
    var done = false
    while pi2 < ph.count && !done {
        let pc = findPhonemeCode(ph, ph.count, pi2)
        if pc.len == 2 && pc.code[0] == asc("a")
            && pc.code[1] == asc("#") {
            ph.insert(asc("'"), at: pi2)
            stripAHashVariantDigits(&ph, pi2)
            done = true
        } else if isVowelCode(pc.code, 0, pc.len) {
            done = true
        } else {
            pi2 += pc.len
        }
    }
}

func insertPrimaryStress(_ ph: inout [UInt8],
                         _ forceFinalStress: Bool,
                         _ rbaN: Int) {
    let startsWithSecondary = ph.count > 0 && ph[0] == asc(",")
    let isRuleLeadingComma = startsWithSecondary && rbaN > 0
    let active = !containsByte(ph, asc("'"))
        && (!startsWithSecondary || isRuleLeadingComma)
    if active {
        var pickLast = computePickLast(ph, forceFinalStress)
        var ignoreComma = rbaN > 0
        let secPickLast = ignoreComma
            && strongDiphthongAfterSecondary(ph)
        if secPickLast {
            pickLast = true
            ignoreComma = false
        }
        let s = scanForPrimary(ph, pickLast, ignoreComma, secPickLast)
        if s.insertPos >= 0
            && !suppressPrimaryOnSchwa(ph, s.insertPos,
                                       s.secondaryVowelPos) {
            ph.insert(asc("'"), at: s.insertPos)
        } else {
            stressAHashLastResort(&ph)
        }
    }
}

func isWeakVowel(_ code: [UInt8], _ n: Int) -> Bool {
    isSchwaVowel(code, n) || isReducedVowel(code, n)
}

func isReducedOrSchwa(_ code: [UInt8], _ n: Int) -> Bool {
    isSchwaVowel(code, n)
        || (n == 2 && isWeakTwoCharVowel(code))
}

// "@-" is a syllable-boundary marker, not a syllable of its own.
func isHyphenatedSchwa(_ ph: [UInt8], _ pi: Int, _ code: [UInt8],
                       _ codeLen: Int) -> Bool {
    codeLen == 1 && code[0] == asc("@") && pi + 1 < ph.count
        && ph[pi + 1] == asc("-")
}

// A rule boundary inside a centering diphthong splits it, leaving the
// trailing "@"/"3" as its own syllable; `piOrig` is the pre-marker offset.
func splitCenteringDiphthong(_ ph: [UInt8], _ pi: Int,
                             _ code: [UInt8], _ codeLen: Int,
                             _ rba: [Bool], _ rbaN: Int) -> Int {
    var r = codeLen
    let last = code[codeLen - 1]
    if last == asc("@") || last == asc("3") {
        var piOrig = pi
        for q in 0..<pi {
            let qc = ph[q]
            if qc == asc("'") || qc == asc(",") || qc == asc("%")
                || qc == asc("=") {
                piOrig -= 1
            }
        }
        var hasBoundary = false
        var k = 0
        while k + 1 < codeLen && !hasBoundary {
            let checkPos = piOrig + k
            if checkPos >= 0 && checkPos < rbaN
                && checkPos < rba.count && rba[checkPos] {
                hasBoundary = true
            }
            k += 1
        }
        if hasBoundary { r -= 1 }
    }
    return r
}

func collectSecondarySyllables(_ ph: [UInt8], _ rba: [Bool],
                               _ rbaN: Int) -> [Syl] {
    var syls = [Syl]()
    var pi = 0
    var unstressedPrefix = false
    var secondaryPrefix = false
    var primaryPrefix = false
    while pi < ph.count && syls.count < maxSyl {
        let c = ph[pi]
        if c == asc("'") { primaryPrefix = true; pi += 1 }
        else if c == asc(",") { secondaryPrefix = true; pi += 1 }
        else if c == asc("%") || c == asc("=") {
            unstressedPrefix = true; pi += 1
        } else {
            let pc = findPhonemeCode(ph, ph.count, pi)
            var code = pc.code
            var codeLen = pc.len
            if codeLen >= 3 && !primaryPrefix && !secondaryPrefix {
                codeLen = splitCenteringDiphthong(ph, pi, code,
                                                  codeLen, rba, rbaN)
                code = pre(code, codeLen)
            }
            if isVowelCode(code, 0, codeLen)
                && !isHyphenatedSchwa(ph, pi, code, codeLen) {
                var s = Syl()
                s.pos = pi
                s.code = code
                s.codeLen = codeLen
                s.isPrimary = primaryPrefix
                s.isSecondary = secondaryPrefix
                s.stressable = !isWeakVowel(code, codeLen)
                    && !unstressedPrefix
                syls.append(s)
                primaryPrefix = false
                secondaryPrefix = false
                unstressedPrefix = false
            }
            pi += codeLen
        }
    }
    return syls
}

func findPrimarySyllable(_ syls: [Syl]) -> Int {
    var primaryIdx = -1
    var si = 0
    while si < syls.count && primaryIdx < 0 {
        if syls[si].isPrimary { primaryIdx = si }
        si += 1
    }
    return primaryIdx
}

func sylTakesSecondary(_ s: Syl) -> Bool {
    s.stressable && !s.isSecondary && !s.isPrimary
}

func hasLaterPrimary(_ syls: [Syl], _ from: Int) -> Bool {
    var later = false
    var k = from
    while k < syls.count && !later {
        if syls[k].isPrimary { later = true }
        k += 1
    }
    return later
}

// Leftmost stressable at distance >= 2 from the primary, a cascade every 2
// rightward, then an even-distance pass back that skips within 2 of a mark.
func markSecondaryBackward(_ syls: [Syl], _ primaryIdx: Int,
                           _ toMark: inout [Int]) {
    var firstSec = -1
    var idx = 0
    while idx <= primaryIdx - 2 && firstSec < 0 {
        if sylTakesSecondary(syls[idx]) { firstSec = idx }
        idx += 1
    }
    if firstSec >= 0 {
        toMark.append(firstSec)
        var j = firstSec + 2
        while j <= primaryIdx - 2 {
            if sylTakesSecondary(syls[j]) { toMark.append(j) }
            j += 2
        }
        var dist = 2
        while primaryIdx - dist >= 0 {
            let k = primaryIdx - dist
            if sylTakesSecondary(syls[k]) {
                var tooClose = false
                for m in 0..<toMark.count {
                    var diff = toMark[m] - k
                    if diff < 0 { diff = -diff }
                    if diff < 2 { tooClose = true }
                }
                if !tooClose { toMark.append(k) }
            }
            dist += 2
        }
    }
}

// An unstressable landing spot defers to the syllable one further on.
func markSecondaryForward(_ syls: [Syl], _ primaryIdx: Int,
                          _ toMark: inout [Int]) {
    let n = syls.count
    var stressableAfter = 0
    var idx2 = primaryIdx + 1
    while idx2 < n {
        if syls[idx2].stressable { stressableAfter += 1 }
        idx2 += 1
    }
    if stressableAfter >= 1 {
        var stopFwd = false
        var dist = 2
        while primaryIdx + dist < n && !stopFwd {
            let idx = primaryIdx + dist
            if hasLaterPrimary(syls, idx + 1) {
                stopFwd = true
            } else if sylTakesSecondary(syls[idx]) {
                toMark.append(idx)
            } else if !syls[idx].stressable && !syls[idx].isPrimary {
                let idx2b = primaryIdx + dist + 1
                if idx2b < n && sylTakesSecondary(syls[idx2b])
                    && !hasLaterPrimary(syls, idx2b + 1) {
                    toMark.append(idx2b)
                }
            }
            dist += 2
        }
    }
}

// Descending, so each insertion leaves the earlier positions valid.
func sortMarksDesc(_ toMark: inout [Int]) {
    for i in 0..<toMark.count {
        for j in (i + 1)..<toMark.count where toMark[j] > toMark[i] {
            toMark.swapAt(i, j)
        }
    }
}

func placeSecondaryStress(_ ph: inout [UInt8], _ phIn: [UInt8],
                          _ step50Fired: Bool,
                          _ rba: [Bool], _ rbaN: Int) -> Bool {
    let ran = containsByte(ph, asc("'"))
        && !containsByte(phIn, asc(","))
    if ran {
        let syls = collectSecondarySyllables(ph, rba, rbaN)
        let primaryIdx = findPrimarySyllable(syls)
        if primaryIdx >= 0 && syls.count >= 3 {
            var toMark = [Int]()
            if primaryIdx >= 2 && !step50Fired
                && !containsByte(ph, asc(",")) {
                markSecondaryBackward(syls, primaryIdx, &toMark)
            }
            markSecondaryForward(syls, primaryIdx, &toMark)
            sortMarksDesc(&toMark)
            for idx in toMark {
                ph.insert(asc(","), at: syls[idx].pos)
            }
        }
    }
    return ran
}


let maxPh = 512

func applySegmentalPhases(_ out: inout [UInt8], _ isEnUs: Bool) {
    applyVelarNasalAssimilation(&out)
    if isEnUs {
        applyHappyTensing(&out)
        applyVowelReduction(&out)
        applyLotPlusRMerge(&out)
    }
    stripMorphemeSchwaR(&out)
    if isEnUs {
        applyBareSchwaToRhotic(&out)
        applyLinkingR(&out)
    }
    if isEnUs {
        applyTionStressFix(&out)
        applyOlogyStressFix(&out)
        applyIcStressFix(&out)
    }
}

func applyStressPhases(_ out: inout [UInt8], _ phIn: [UInt8],
                       _ isStrend: Bool, _ rba: [Bool],
                       _ rbaN: Int) {
    let startsWithSecondary = out.count > 0 && out[0] == asc(",")
    let isRuleLeadingComma = startsWithSecondary && rbaN > 0
    insertPrimaryStress(&out, isStrend, rbaN)
    let step50Fired = applyEqualsSuffixStressShift(&out)
    let step5aRan = placeSecondaryStress(&out, phIn, step50Fired,
                                         rba, rbaN)
    cleanupAdjacentSecondary(&out, step5aRan, isRuleLeadingComma,
                             rba, rbaN)
    trochaicCompoundPrefix(&out, phIn, step5aRan, startsWithSecondary)
    finalSyllableSecondary(&out)
}

func applyEnUsReductions(_ out: inout [UInt8], _ rba: [Bool],
                         _ rbaN: Int) {
    reduceBareZeroAfterUtion(&out)
    reduceEUnstressed(&out, rbaN)
    reduceVBetweenSecAndPrimary(&out)
    reduceABeforePrimary(&out, rbaN > 0)
    reduceABetweenPrimaryAndSec(&out, rbaN > 0)
    reduceZeroBetweenSecAndPrimary(&out)
    reduceZeroBetweenPrimaryAndSec(&out)
    reduceZeroHashBeforePrimary(&out)
    reduceBareZeroBeforePrimary(&out, rba, rbaN)
    reduceEPreTonicAfterPctNasal(&out)
    postStressIBeforeSyllabicL(&out)
    reduce3LongToShort(&out)
    applyFlapRule(&out)
    reduceNessSuffix(&out)
    reduceOuHashCompound(&out)
    elideIcallySchwa(&out)
}

func applyReductionPhases(_ out: inout [UInt8], _ phIn: [UInt8],
                          _ isEnUs: Bool, _ rba: [Bool],
                          _ rbaN: Int) {
    if isEnUs { reduceZeroDiminished(&out) }
    demoteAdjacentPrimaries(&out, phIn)
    if isEnUs { applyEnUsReductions(&out, rba, rbaN) }
    addCompoundSyllabicNStress(&out)
}

// `isStrend` becomes forceFinalStress; en-us is the only dialect shipped.
func processPhonemeString(_ input: [UInt8], _ isStrend: Bool) -> [UInt8] {
    // The pre-strip snapshot: the stress phases consult it for the ',' /
    // '\'' the input carried before the boundary markers were removed.
    let phIn = input
    var out = input
    var rba = [Bool](repeating: false, count: maxPh)
    let rbaN = stripRuleBoundaryMarkers(&out, &rba)
    for i in 0..<out.count where out[i] == phSec2 {
        out[i] = asc(",")
    }
    let isEnUs = true
    applySegmentalPhases(&out, isEnUs)
    applyStressPhases(&out, phIn, isStrend, rba, rbaN)
    applyReductionPhases(&out, phIn, isEnUs, rba, rbaN)
    return out
}
