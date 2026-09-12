// Rule application: free functions, as in the C; only the ruleset, no state.
import Foundation

func applyReplacements(_ out: inout [UInt8],
                       _ reps: [ReplaceRule]) {
    for ri in 0..<reps.count {
        let from = reps[ri].from
        let to = reps[ri].to
        if from.count > 0 {
            var pos = 0
            while pos + from.count <= out.count {
                let hit = equalRange(out, pos, from, from.count)
                if hit {
                    out.replaceSubrange(pos..<(pos + from.count),
                                        with: to)
                    pos += to.count
                } else {
                    pos += 1
                }
            }
        }
    }
}

func matchLgroupAt(_ lgroup: [[UInt8]], _ word: [UInt8],
                   _ wn: Int, _ pos: Int) -> Int {
    var best = 0
    if pos >= 0 && pos < wn {
        for i in 0..<lgroup.count {
            let item = lgroup[i]
            let ilen = item.count
            if ilen > 0 && pos + ilen <= wn {
                var ok = true
                var j = 0
                while j < ilen && ok {
                    let wc = toLowerC(word[pos + j])
                    let ic = toLowerC(item[j])
                    if wc != ic { ok = false }
                    j += 1
                }
                if ok && ilen > best { best = ilen }
            }
        }
    }
    return best
}

struct CtxScore {
    var score = 0
    var matched = false
}

func matchGroup(_ groups: LetterGroups, _ g: UInt8,
                _ word: [UInt8], _ wn: Int, _ pos: Int) -> Bool {
    var r = false
    if pos >= 0 && pos < wn {
        let u = Int(toLowerC(word[pos]))
        switch g {
        case asc("A"): r = groups.groupA[u]
        case asc("B"): r = groups.groupB[u]
        case asc("C"): r = groups.groupC[u]
        case asc("F"): r = groups.groupF[u]
        case asc("G"): r = groups.groupG[u]
        case asc("H"): r = groups.groupH[u]
        case asc("Y"): r = groups.groupY[u]
        case asc("K"): r = groups.groupK[u]
        default: break
        }
    }
    return r
}

let stressedMC = bsv([
    "aI@3", "aU@r", "i@3r", "aI@", "aI3", "aU@", "i@3",
    "3:r", "A:r", "o@r", "A@r", "e@r",
    "eI", "aI", "aU", "OI", "oU", "IR", "VR",
    "e@", "i@", "U@", "A@", "O@", "o@",
    "3:", "A:", "i:", "u:", "O:", "e:", "a:", "aa",
    "@L", "@2", "@5",
    "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02",
    "O2", "A~", "O~", "A#",
])

let vowelPhStressable = bs("aAeEiIoOuUV03")

func findStressedCode(_ ph: [UInt8], _ pn: Int,
                      _ pi: Int) -> [UInt8] {
    var code = [UInt8]()
    var haveCode = false
    var mi = 0
    while mi < stressedMC.count && !haveCode {
        let mc = stressedMC[mi]
        if pi + mc.count <= pn && equalRange(ph, pi, mc, mc.count) {
            code = mc
            haveCode = true
        }
        mi += 1
    }
    if !haveCode { code = [ph[pi]] }
    return code
}

// Schwa, bare i and the reduced two-char vowels never carry stress.
func codeIsInherentlyUnstressed(_ code: [UInt8]) -> Bool {
    let codeLen = code.count
    return code[0] == asc("@")
        || (codeLen == 1 && code[0] == asc("i"))
        || (codeLen == 2 && code[0] == asc("a")
            && code[1] == asc("#"))
        || (codeLen == 2 && code[0] == asc("I")
            && code[1] == asc("#"))
        || (codeLen == 2 && code[0] == asc("I")
            && code[1] == asc("2"))
}

// A vowel code silenced neither by a marker nor by its own quality.
func codeCarriesStress(_ code: [UInt8],
                       _ prevUnstressed: Bool) -> Bool {
    let isVowel = code.count > 0
        && strchrHit(vowelPhStressable, code[0])
    return isVowel && !codeIsInherentlyUnstressed(code)
        && !prevUnstressed
}

func stressedInPhonemes(_ ph: [UInt8], _ pn: Int) -> Bool {
    var found = false
    var pi = 0
    var prevUnstressed = false
    while pi < pn && !found {
        let pc = ph[pi]
        if pc == asc("%") || pc == asc("=") || pc == asc(",") {
            prevUnstressed = true
            pi += 1
        } else if pc == asc("'") {
            prevUnstressed = false
            pi += 1
        } else {
            let code = findStressedCode(ph, pn, pi)
            found = codeCarriesStress(code, prevUnstressed)
            if !found {
                prevUnstressed = false
                pi += code.count
            }
        }
    }
    return found
}

let vowelPhAt = bs("aAeEIiOUVu03@o")

func isLetterGroupRef(_ cc: UInt8) -> Bool {
    cc == asc("A") || cc == asc("B") || cc == asc("C")
        || cc == asc("F") || cc == asc("G") || cc == asc("H")
        || cc == asc("Y")
}

// `ok` is the loop predicate: a token that fails the match clears it and
// the scan stops where it stands.
struct LctxScan {
    var score = 0
    var wordPos = 0
    var ci = 0
    var distanceLeft = -2
    var prevChar: UInt8 = 0
    var ok = true
}

func lctxAdvance(_ s: inout LctxScan, _ pts: Int) {
    s.distanceLeft += 2
    if s.distanceLeft > 19 { s.distanceLeft = 19 }
    s.score += pts - s.distanceLeft
}

func lctxWordBound(_ s: inout LctxScan) {
    if s.wordPos >= 0 {
        s.ok = false
    } else {
        s.ci -= 1
        s.score += 4
    }
}

// With no phonemes yet, any vowel to the left counts as the stress.
func lctxStressed(_ s: inout LctxScan, _ word: [UInt8], _ pos: Int,
                  _ phSoFar: [UInt8], _ phN: Int) {
    var foundStressed = false
    if phN > 0 {
        foundStressed = stressedInPhonemes(phSoFar, phN)
    } else {
        var k = 0
        while k < pos && !foundStressed {
            if isVowelLetter(word[k]) { foundStressed = true }
            k += 1
        }
    }
    if !foundStressed {
        s.ok = false
    } else {
        s.ci -= 1
        s.score += 19
    }
}

func countLeftVowelGroups(_ word: [UInt8], _ pos: Int,
                          _ phSoFar: [UInt8], _ phN: Int) -> Int {
    var vowelGroups = 0
    var inV2 = false
    if phN > 0 {
        for k in 0..<phN {
            let v = strchrHit(vowelPhAt, phSoFar[k])
            if v && !inV2 {
                vowelGroups += 1
                inV2 = true
            } else if !v {
                inV2 = false
            }
        }
    } else {
        for wp in 0..<pos {
            let v = isVowelLetter(word[wp])
            if v && !inV2 {
                vowelGroups += 1
                inV2 = true
            } else if !v {
                inV2 = false
            }
        }
    }
    return vowelGroups
}

// "@" reads the distance without bumping it: it scores a syllable
// count, it does not consume a letter.
func lctxSyllables(_ s: inout LctxScan, _ ctxStr: [UInt8],
                   _ word: [UInt8], _ pos: Int,
                   _ phSoFar: [UInt8], _ phN: Int) {
    var syllableCount = 0
    while s.ci >= 0 && ctxStr[s.ci] == asc("@") {
        syllableCount += 1
        s.ci -= 1
    }
    let vowelGroups = countLeftVowelGroups(word, pos, phSoFar, phN)
    if syllableCount > vowelGroups {
        s.ok = false
    } else {
        var dist = s.distanceLeft + 2
        if dist > 19 { dist = 19 }
        s.score += 18 + syllableCount - dist
    }
}

func lctxNoVowelsLeft(_ s: inout LctxScan, _ word: [UInt8]) {
    var foundVowel = false
    var k = 0
    while k <= s.wordPos && !foundVowel {
        if isVowelLetter(word[k]) { foundVowel = true }
        k += 1
    }
    if foundVowel {
        s.ok = false
    } else {
        s.ci -= 1
        s.score += 3
    }
}

func lctxDoubleLetter(_ s: inout LctxScan, _ word: [UInt8]) {
    if s.wordPos < 0 {
        s.ok = false
    } else {
        let cur = toLowerC(word[s.wordPos])
        let nxt = toLowerC(s.prevChar)
        if cur != nxt {
            s.ok = false
        } else {
            s.prevChar = word[s.wordPos]
            s.wordPos -= 1
            s.ci -= 1
            lctxAdvance(&s, 21)
        }
    }
}

func lctxLetterGroup(_ s: inout LctxScan, _ cc: UInt8,
                     _ word: [UInt8], _ wn: Int, _ rs: Ruleset) {
    if !matchGroup(rs.groups, cc, word, wn, s.wordPos) {
        s.ok = false
    } else {
        let lgPts = (cc == asc("C")) ? 19 : 20
        s.prevChar = word[s.wordPos]
        s.wordPos -= 1
        s.ci -= 1
        lctxAdvance(&s, lgPts)
    }
}

func lctxNonVowel(_ s: inout LctxScan, _ word: [UInt8]) {
    if s.wordPos < 0 || isVowelLetter(word[s.wordPos]) {
        s.ok = false
    } else {
        s.prevChar = word[s.wordPos]
        s.wordPos -= 1
        s.ci -= 1
        lctxAdvance(&s, 20)
    }
}

func lctxDigitLetter(_ s: inout LctxScan, _ word: [UInt8]) {
    if s.wordPos < 0 || !isDigitC(word[s.wordPos]) {
        s.ok = false
    } else {
        s.prevChar = word[s.wordPos]
        s.wordPos -= 1
        s.ci -= 1
        lctxAdvance(&s, 21)
    }
}

func lctxNonAlpha(_ s: inout LctxScan, _ word: [UInt8]) {
    if s.wordPos < 0 || isAlphaC(word[s.wordPos]) {
        s.ok = false
    } else {
        s.prevChar = word[s.wordPos]
        s.wordPos -= 1
        s.ci -= 1
        lctxAdvance(&s, 21)
    }
}

// Read right-to-left, an L-group arrives digits first: "L12" reaches this
// at the "2". Digits with no "L" behind them are a plain skip.
func lctxLgroup(_ s: inout LctxScan, _ cc: UInt8, _ ctxStr: [UInt8],
                _ word: [UInt8], _ wn: Int, _ rs: Ruleset) {
    var gid = Int(cc - asc("0"))
    var ci2 = s.ci - 1
    if ci2 >= 0 && ctxStr[ci2] >= asc("0") && ctxStr[ci2] <= asc("9") {
        gid += Int(ctxStr[ci2] - asc("0")) * 10
        ci2 -= 1
    }
    if ci2 >= 0 && ctxStr[ci2] == asc("L") {
        if gid > 0 && gid < 100 {
            let matched = matchLgroupAt(rs.groups.lgroups[gid],
                                        word, wn, s.wordPos)
            if matched == 0 {
                s.ok = false
            } else {
                if matched > 0 {
                    s.prevChar = word[s.wordPos - matched + 1]
                }
                s.wordPos -= matched
                lctxAdvance(&s, 20)
            }
        }
        s.ci = ci2 - 1
    } else {
        s.ci -= 1
    }
}

func lctxLiteral(_ s: inout LctxScan, _ cc: UInt8, _ word: [UInt8]) {
    if s.wordPos < 0 {
        s.ok = false
    } else {
        let wc = toLowerC(word[s.wordPos])
        let mc = toLowerC(cc)
        if wc != mc {
            s.ok = false
        } else {
            s.prevChar = word[s.wordPos]
            s.wordPos -= 1
            s.ci -= 1
            lctxAdvance(&s, 21)
        }
    }
}

// The two token sets are disjoint, so the split preserves the arm order.
func lctxBoundaryToken(_ s: inout LctxScan, _ cc: UInt8,
                       _ ctxStr: [UInt8], _ word: [UInt8], _ pos: Int,
                       _ phSoFar: [UInt8], _ phN: Int) -> Bool {
    var handled = true
    if cc == asc("_") { lctxWordBound(&s) }
    else if cc == asc("&") { lctxStressed(&s, word, pos, phSoFar, phN) }
    else if cc == asc("@") {
        lctxSyllables(&s, ctxStr, word, pos, phSoFar, phN)
    } else if cc == asc("!") { s.ci -= 1 }
    else if cc == asc("+") { s.score += 20; s.ci -= 1 }
    else if cc == asc("<") { s.score -= 20; s.ci -= 1 }
    else if cc == asc("X") { lctxNoVowelsLeft(&s, word) }
    else if cc == asc("E") { s.ok = false }
    else { handled = false }
    return handled
}

func lctxConsumingToken(_ s: inout LctxScan, _ cc: UInt8,
                        _ ctxStr: [UInt8], _ word: [UInt8],
                        _ wn: Int, _ rs: Ruleset) {
    if cc == asc("%") { lctxDoubleLetter(&s, word) }
    else if isLetterGroupRef(cc) {
        lctxLetterGroup(&s, cc, word, wn, rs)
    } else if cc == asc("K") { lctxNonVowel(&s, word) }
    else if cc == asc("D") { lctxDigitLetter(&s, word) }
    else if cc == asc("Z") { lctxNonAlpha(&s, word) }
    else if cc >= asc("0") && cc <= asc("9") {
        lctxLgroup(&s, cc, ctxStr, word, wn, rs)
    } else {
        lctxLiteral(&s, cc, word)
    }
}

func matchLeftContextScore(_ ctxStr: [UInt8], _ ctxN: Int,
                           _ word: [UInt8], _ wn: Int, _ pos: Int,
                           _ rs: Ruleset,
                           _ phSoFar: [UInt8], _ phN: Int) -> CtxScore {
    var result = CtxScore()
    var s = LctxScan()
    s.wordPos = pos - 1
    s.ci = ctxN - 1
    s.prevChar = (pos > 0 && pos < wn) ? word[pos] : 0
    while s.ci >= 0 && s.ok {
        let cc = ctxStr[s.ci]
        if !lctxBoundaryToken(&s, cc, ctxStr, word, pos, phSoFar, phN) {
            lctxConsumingToken(&s, cc, ctxStr, word, wn, rs)
        }
    }
    if s.ok {
        result.score = s.score
        result.matched = true
    }
    return result
}

struct RightCtxScore {
    var score = 0
    var delFwdStart = -1
    var delFwdCount = 0
    var matched = false
}

struct RctxScan {
    var score = 0
    var wordPos = 0
    var ci = 0
    var distanceRight = -6
    var delFwdPos = -1
    var prevChar: UInt8 = 0
    var ok = true
}

// Each token consumed is worth less than the one before, to a floor of 19.
func rctxAdvance(_ s: inout RctxScan, _ pts: Int) {
    s.distanceRight += 6
    if s.distanceRight > 19 { s.distanceRight = 19 }
    s.score += pts - s.distanceRight
}

func rctxWordEnd(_ s: inout RctxScan, _ wn: Int) {
    if s.wordPos < wn {
        s.ok = false
    } else {
        s.ci += 1
        rctxAdvance(&s, 21)
    }
}

func rctxLetterGroup(_ s: inout RctxScan, _ cc: UInt8,
                     _ word: [UInt8], _ wn: Int, _ rs: Ruleset) {
    if !matchGroup(rs.groups, cc, word, wn, s.wordPos) {
        s.ok = false
    } else {
        let lgPts = (cc == asc("C")) ? 19 : 20
        s.prevChar = word[s.wordPos]
        s.wordPos += 1
        s.ci += 1
        rctxAdvance(&s, lgPts)
    }
}

// K matches a non-vowel; the null at word end is non-vowel too.
func rctxNonVowel(_ s: inout RctxScan, _ word: [UInt8], _ wn: Int) {
    if s.wordPos < wn && isVowelLetter(word[s.wordPos]) {
        s.ok = false
    } else {
        if s.wordPos < wn { s.prevChar = word[s.wordPos] }
        s.wordPos += 1
        s.ci += 1
        rctxAdvance(&s, 20)
    }
}

func rctxNoVowelsLeft(_ s: inout RctxScan, _ word: [UInt8],
                      _ wn: Int) {
    var found = false
    var k = s.wordPos
    while k < wn && !found {
        if isVowelLetter(word[k]) { found = true }
        k += 1
    }
    if found {
        s.ok = false
    } else {
        s.ci += 1
        rctxAdvance(&s, 19)
    }
}

func rctxDigitLetter(_ s: inout RctxScan, _ word: [UInt8], _ wn: Int) {
    if s.wordPos >= wn || !isDigitC(word[s.wordPos]) {
        s.ok = false
    } else {
        s.prevChar = word[s.wordPos]
        s.wordPos += 1
        s.ci += 1
        rctxAdvance(&s, 21)
    }
}

func rctxNonAlpha(_ s: inout RctxScan, _ word: [UInt8], _ wn: Int) {
    if s.wordPos >= wn || isAlphaC(word[s.wordPos]) {
        s.ok = false
    } else {
        s.prevChar = word[s.wordPos]
        s.wordPos += 1
        s.ci += 1
        rctxAdvance(&s, 21)
    }
}

func rctxDoubleLetter(_ s: inout RctxScan, _ word: [UInt8],
                      _ wn: Int) {
    if s.wordPos >= wn {
        s.ok = false
    } else {
        let cur = toLowerC(word[s.wordPos])
        let prv = toLowerC(s.prevChar)
        if cur != prv {
            s.ok = false
        } else {
            s.prevChar = word[s.wordPos]
            s.wordPos += 1
            s.ci += 1
            rctxAdvance(&s, 21)
        }
    }
}

func rctxSyllables(_ s: inout RctxScan, _ ctxStr: [UInt8],
                   _ clen: Int, _ word: [UInt8], _ wn: Int) {
    var syllableCount = 0
    while s.ci < clen && ctxStr[s.ci] == asc("@") {
        syllableCount += 1
        s.ci += 1
    }
    var vowelGroups = 0
    var inV = false
    var wp = s.wordPos
    while wp < wn {
        let v = isVowelLetter(word[wp])
        if v && !inV {
            vowelGroups += 1
            inV = true
        } else if !v {
            inV = false
        }
        wp += 1
    }
    if syllableCount > vowelGroups {
        s.ok = false
    } else {
        rctxAdvance(&s, 18 + syllableCount)
    }
}

// RULE_DEL_FWD: the first "e" between match start and cursor is deleted later.
func rctxScanDelFwd(_ s: inout RctxScan, _ word: [UInt8], _ pos: Int) {
    if s.delFwdPos < 0 {
        var sp = pos
        while sp < s.wordPos && s.delFwdPos < 0 {
            if word[sp] == asc("e") { s.delFwdPos = sp }
            sp += 1
        }
    }
    s.ci += 1
}

func rctxWordAlt(_ s: inout RctxScan, _ ctxStr: [UInt8], _ clen: Int,
                 _ wordAltFlags: Int) {
    let isAltRef = s.ci + 6 < clen && ctxStr[s.ci + 1] == asc("w")
        && ctxStr[s.ci + 2] == asc("_")
        && ctxStr[s.ci + 3] == asc("a")
        && ctxStr[s.ci + 4] == asc("l")
        && ctxStr[s.ci + 5] == asc("t")
        && ctxStr[s.ci + 6] >= asc("1")
        && ctxStr[s.ci + 6] <= asc("6")
    if !isAltRef {
        s.ok = false
    } else {
        let altN = Int(ctxStr[s.ci + 6] - asc("0"))
        let altBit = 1 << (altN - 1)
        if (wordAltFlags & altBit) == 0 {
            s.ok = false
        } else {
            s.ci += 7
        }
    }
}

func rctxSuffixRemovedCond(_ s: inout RctxScan, _ ctxStr: [UInt8],
                           _ clen: Int, _ suffixRemoved: Bool) {
    if s.ci + 1 < clen && isDigitC(ctxStr[s.ci + 1]) {
        s.ci += 2
    } else if suffixRemoved {
        s.ok = false
    } else {
        s.score += 1
        s.ci += 1
    }
}

func rctxSkipToSpace(_ s: inout RctxScan, _ ctxStr: [UInt8],
                     _ clen: Int) {
    while s.ci < clen && !isSpaceC(ctxStr[s.ci]) { s.ci += 1 }
}

// "S<N>[flags]" is read by the caller, not scored here.
func rctxSuffixDirective(_ s: inout RctxScan, _ ctxStr: [UInt8],
                         _ clen: Int) {
    s.ci += 1
    while s.ci < clen && isDigitC(ctxStr[s.ci]) { s.ci += 1 }
    while s.ci < clen && isAlphaC(ctxStr[s.ci]) { s.ci += 1 }
}

// An out-of-range or empty L-group consumes its digits and scores
// nothing, rather than failing the match.
func rctxLgroup(_ s: inout RctxScan, _ ctxStr: [UInt8], _ clen: Int,
                _ word: [UInt8], _ wn: Int, _ rs: Ruleset) {
    var gid = 0
    s.ci += 1
    while s.ci < clen && isDigitC(ctxStr[s.ci]) {
        gid = gid * 10 + Int(ctxStr[s.ci] - asc("0"))
        s.ci += 1
    }
    if gid > 0 && gid < 100 && rs.groups.lgroups[gid].count > 0 {
        let matched = matchLgroupAt(rs.groups.lgroups[gid],
                                    word, wn, s.wordPos)
        if matched == 0 {
            s.ok = false
        } else {
            if matched > 0 {
                s.prevChar = word[s.wordPos + matched - 1]
            }
            s.wordPos += matched
            rctxAdvance(&s, 20)
        }
    }
}

func rctxReplacedE(_ s: inout RctxScan, _ word: [UInt8], _ wn: Int,
                   _ replacedEArr: [Bool]?, _ reN: Int) {
    let reMarked = replacedEArr != nil && s.wordPos < wn
        && s.wordPos < reN && replacedEArr![s.wordPos]
    if reMarked {
        s.prevChar = word[s.wordPos]
        s.wordPos += 1
        s.ci += 1
        rctxAdvance(&s, 21)
    } else {
        s.ok = false
    }
}

func rctxLiteral(_ s: inout RctxScan, _ cc: UInt8, _ word: [UInt8],
                 _ wn: Int) {
    if s.wordPos >= wn {
        s.ok = false
    } else {
        let wc = toLowerC(word[s.wordPos])
        let mc = toLowerC(cc)
        if wc != mc {
            s.ok = false
        } else {
            s.prevChar = word[s.wordPos]
            s.wordPos += 1
            s.ci += 1
            rctxAdvance(&s, 21)
        }
    }
}

// The two token sets are disjoint, so the split preserves the arm order.
func rctxDirectiveToken(_ s: inout RctxScan, _ cc: UInt8,
                        _ ctxStr: [UInt8], _ clen: Int,
                        _ word: [UInt8], _ pos: Int,
                        _ wordAltFlags: Int,
                        _ suffixRemoved: Bool) -> Bool {
    var handled = true
    if cc == asc("#") { rctxScanDelFwd(&s, word, pos) }
    else if cc == asc("+") { s.score += 20; s.ci += 1 }
    else if cc == asc("<") { s.score -= 20; s.ci += 1 }
    else if cc == asc("&") { s.ok = false }
    else if cc == asc("!") { s.ci += 1 }
    else if cc == asc("$") {
        rctxWordAlt(&s, ctxStr, clen, wordAltFlags)
    } else if cc == asc("N") {
        rctxSuffixRemovedCond(&s, ctxStr, clen, suffixRemoved)
    } else if cc == asc("P") {
        rctxSkipToSpace(&s, ctxStr, clen)
    } else if cc == asc("S") {
        rctxSuffixDirective(&s, ctxStr, clen)
    } else if isDigitC(cc) {
        s.ci += 1
    } else {
        handled = false
    }
    return handled
}

func rctxConsumingToken(_ s: inout RctxScan, _ cc: UInt8,
                        _ ctxStr: [UInt8], _ clen: Int,
                        _ word: [UInt8], _ wn: Int, _ rs: Ruleset,
                        _ replacedEArr: [Bool]?, _ reN: Int) {
    if cc == asc("_") { rctxWordEnd(&s, wn) }
    else if isLetterGroupRef(cc) {
        rctxLetterGroup(&s, cc, word, wn, rs)
    } else if cc == asc("K") { rctxNonVowel(&s, word, wn) }
    else if cc == asc("X") { rctxNoVowelsLeft(&s, word, wn) }
    else if cc == asc("D") { rctxDigitLetter(&s, word, wn) }
    else if cc == asc("Z") { rctxNonAlpha(&s, word, wn) }
    else if cc == asc("%") { rctxDoubleLetter(&s, word, wn) }
    else if cc == asc("@") {
        rctxSyllables(&s, ctxStr, clen, word, wn)
    } else if cc == asc("L") && s.ci + 1 < clen
              && isDigitC(ctxStr[s.ci + 1]) {
        rctxLgroup(&s, ctxStr, clen, word, wn, rs)
    } else if cc == asc("E") {
        rctxReplacedE(&s, word, wn, replacedEArr, reN)
    } else {
        rctxLiteral(&s, cc, word, wn)
    }
}

func matchRightContextScore(_ ctxStr: [UInt8], _ ctxN: Int,
                            _ word: [UInt8], _ wn: Int, _ pos: Int,
                            _ rs: Ruleset,
                            _ initialPrevChar: UInt8,
                            _ wordAltFlags: Int,
                            _ replacedEArr: [Bool]?, _ reN: Int,
                            _ suffixRemoved: Bool) -> RightCtxScore {
    var result = RightCtxScore()
    var s = RctxScan()
    s.wordPos = pos
    s.prevChar = initialPrevChar
    let clen = ctxN
    while s.ci < clen && s.ok {
        let cc = ctxStr[s.ci]
        if !rctxDirectiveToken(&s, cc, ctxStr, clen, word, pos,
                               wordAltFlags, suffixRemoved) {
            rctxConsumingToken(&s, cc, ctxStr, clen, word, wn, rs,
                               replacedEArr, reN)
        }
    }
    if s.ok {
        result.score = s.score
        result.matched = true
        if s.delFwdPos >= 0 {
            result.delFwdStart = s.delFwdPos
            result.delFwdCount = 1
        }
    }
    return result
}

struct RuleMatch {
    var score = -1
    var phonemes = [UInt8]()
    var advance = 1
    var delStart = -1
    var delCount = 0
    var isPrefix = false
    var isSuffix = false
    var suffixStripLen = 0
    var suffixFlags = 0
}

struct MatchRuleResult {
    var score = -1
    var phonemes = [UInt8]()
    var advance = 0
    var delFwdStart = -1
    var delFwdCount = 0
}

func ruleKeyMatches(_ match: [UInt8], _ word: [UInt8],
                    _ pos: Int, _ mlen: Int) -> Bool {
    var keyOk = true
    var i = 0
    while i < mlen && keyOk {
        let wc = toLowerC(word[pos + i])
        let mc = toLowerC(match[i])
        if wc != mc { keyOk = false }
        i += 1
    }
    return keyOk
}

// Everything from a '$' onwards is a flag annotation, not phonemes.
func ruleTrimmedPhonemes(_ rule: PhonemeRule) -> [UInt8] {
    let phn = rule.phonemes.count
    var dollar = phn
    var k = 0
    while k < phn && dollar == phn {
        if rule.phonemes[k] == asc("$") { dollar = k }
        k += 1
    }
    return trim(rule.phonemes, 0, dollar)
}

// An absent left context matches with score 0.
func ruleLeftScore(_ rs: Ruleset, _ rule: PhonemeRule,
                   _ word: [UInt8], _ wn: Int, _ pos: Int,
                   _ phSoFar: [UInt8], _ phN: Int) -> CtxScore {
    var ls = CtxScore()
    ls.matched = true
    if rule.leftCtx.count > 0 {
        ls = matchLeftContextScore(rule.leftCtx, rule.leftCtx.count,
                                   word, wn, pos, rs, phSoFar, phN)
    }
    return ls
}

// An absent right context matches with score 0 and no delFwd.
func ruleRightScore(_ rs: Ruleset, _ rule: PhonemeRule,
                    _ word: [UInt8], _ wn: Int, _ after: Int,
                    _ lastMatchChar: UInt8, _ wordAltFlags: Int,
                    _ replacedEArr: [Bool]?, _ reN: Int,
                    _ suffixRemoved: Bool) -> RightCtxScore {
    var rr = RightCtxScore()
    rr.matched = true
    if rule.rightCtx.count > 0 {
        rr = matchRightContextScore(rule.rightCtx,
                                    rule.rightCtx.count,
                                    word, wn, after, rs,
                                    lastMatchChar, wordAltFlags,
                                    replacedEArr, reN,
                                    suffixRemoved)
    }
    return rr
}

// Base 1, plus 21 for every matched character beyond the group, plus
// both context scores and the dialect bonus.
func ruleTotalScore(_ rule: PhonemeRule, _ mlen: Int,
                    _ groupLength: Int, _ lscore: Int,
                    _ rscore: Int) -> Int {
    var additional = mlen - groupLength
    if additional < 0 { additional = 0 }
    let dialectBonus = (rule.condition != 0) ? 1 : 0
    return 1 + additional * 21 + lscore + rscore + dialectBonus
}

func matchRule(_ rs: Ruleset, _ rule: PhonemeRule,
               _ word: [UInt8], _ wn: Int, _ pos: Int,
               _ groupLength: Int,
               _ phSoFar: [UInt8], _ phN: Int,
               _ wordAltFlags: Int,
               _ replacedEArr: [Bool]?, _ reN: Int,
               _ suffixRemoved: Bool) -> MatchRuleResult {
    var result = MatchRuleResult()
    let match = rule.match
    let mlen = match.count
    if pos + mlen <= wn {
        if ruleKeyMatches(match, word, pos, mlen) {
            let ls = ruleLeftScore(rs, rule, word, wn, pos,
                                   phSoFar, phN)
            if ls.matched {
                let lastMatchChar: UInt8 = (mlen > 0)
                    ? word[pos + mlen - 1] : 0
                let rresult = ruleRightScore(
                    rs, rule, word, wn, pos + mlen, lastMatchChar,
                    wordAltFlags, replacedEArr, reN, suffixRemoved)
                if rresult.matched {
                    let total = ruleTotalScore(rule, mlen,
                                               groupLength, ls.score,
                                               rresult.score)
                    result.phonemes = ruleTrimmedPhonemes(rule)
                    result.advance = mlen
                    result.delFwdStart = rresult.delFwdStart
                    result.delFwdCount = rresult.delFwdCount
                    result.score = total
                }
            }
        }
    }
    return result
}

// Ties go to the last rule via the `>=`.
func tryGroup(_ rs: Ruleset, _ key: [UInt8], _ bonus: Int,
              _ groupLength: Int,
              _ word: [UInt8], _ wn: Int, _ pos: Int,
              _ len: Int, _ wordAltFlags: Int,
              _ replacedE: [Bool]?, _ reN: Int,
              _ allowSuffixStrip: Bool,
              _ suffixPhonemeOnly: Bool,
              _ suffixRemoved: Bool,
              _ phSoFar: [UInt8], _ phN: Int,
              _ best: inout RuleMatch) {
    if let rv = rs.ruleGroups[key] {
        for i in 0..<rv.count {
            let rule = rv[i]
            let m = matchRule(rs, rule, word, wn, pos,
                              groupLength, phSoFar, phN,
                              wordAltFlags,
                              replacedE, reN, suffixRemoved)
            let valid = m.score >= 0
            let isEndingSkip = !allowSuffixStrip
                && !suffixPhonemeOnly && rule.isSuffix
                && pos + m.advance == len
            if valid && !isEndingSkip && m.score + bonus >= best.score {
                best.score = m.score + bonus
                best.phonemes = m.phonemes
                best.advance = m.advance
                best.delStart = m.delFwdStart
                best.delCount = m.delFwdCount
                best.isPrefix = rule.isPrefix
                best.isSuffix = rule.isSuffix
                best.suffixStripLen = rule.suffixStripLen
                best.suffixFlags = rule.suffixFlags
            }
        }
    }
}

// The 2-char group (with a +35 bonus) then the 1-char group.
func findBestRule(_ rs: Ruleset, _ word: [UInt8], _ wn: Int,
                  _ pos: Int, _ len: Int, _ posChar: UInt8,
                  _ wordAltFlags: Int,
                  _ replacedE: [Bool]?, _ reN: Int,
                  _ allowSuffixStrip: Bool,
                  _ suffixPhonemeOnly: Bool,
                  _ suffixRemoved: Bool,
                  _ phSoFar: [UInt8], _ phN: Int) -> RuleMatch {
    var best = RuleMatch()
    if pos + 1 < len {
        let k2 = [posChar, toLowerC(word[pos + 1])]
        tryGroup(rs, k2, 35, 2, word, wn, pos, len, wordAltFlags,
                 replacedE, reN, allowSuffixStrip,
                 suffixPhonemeOnly, suffixRemoved,
                 phSoFar, phN, &best)
    }
    tryGroup(rs, [posChar], 0, 1, word, wn, pos, len, wordAltFlags,
             replacedE, reN, allowSuffixStrip,
             suffixPhonemeOnly, suffixRemoved,
             phSoFar, phN, &best)
    return best
}

struct PrevNonBnd {
    var ch: UInt8 = 0
    var pos = -1
}

func prevNonBnd(_ ph: [UInt8], _ from: Int) -> PrevNonBnd {
    var r = PrevNonBnd()
    var pi = from - 1
    while pi >= 0 && r.pos < 0 {
        if ph[pi] != phBnd {
            r.ch = ph[pi]
            r.pos = pi
        }
        pi -= 1
    }
    return r
}

let spVowels = bs("aAeEiIoOuUV03@")
let diphSecond = bs("IU")
let diphStart = bs("eaOoUAE")
let atStarters = bs("AeioOU")

// Byte index of the last fully-stressable vowel code's FIRST char, or -1.

// A diphthong, "aa" or a schwa-tail code starts one position earlier.
func stepBackVowelCode(_ phonemes: [UInt8], _ sc: UInt8,
                       _ si: inout Int,
                       _ p: inout PrevNonBnd) {
    if (strchrHit(diphSecond, sc) && strchrHit(diphStart, p.ch))
        || (sc == asc("a") && p.ch == asc("a"))
        || (sc == asc("@") && strchrHit(atStarters, p.ch)) {
        si = p.pos
        p = prevNonBnd(phonemes, si)
    }
}

func findLastStressableVowel(_ phonemes: [UInt8], _ pn: Int) -> Int {
    var insertAt = -1
    let slen = pn
    var si = slen - 1
    while si >= 0 && insertAt < 0 {
        let sc = phonemes[si]
        if sc != phBnd && strchrHit(spVowels, sc) {
            var ni = si + 1
            while ni < slen && phonemes[ni] == phBnd { ni += 1 }
            let isReduced = ni < slen
                && (phonemes[ni] == asc("2") || phonemes[ni] == asc("#"))
            if !isReduced {
                var p = prevNonBnd(phonemes, si)
                stepBackVowelCode(phonemes, sc, &si, &p)
                if p.ch != asc("%") && p.ch != asc("=") {
                    insertAt = si
                }
            }
        }
        si -= 1
    }
    return insertAt
}

// A leading '=' retroactively promotes the last stressable vowel to primary;
// earlier primaries become 0x02 so the prosody step still sees a secondary.
func applyStressPrev(_ emit: inout [UInt8],
                     _ phonemes: inout [UInt8]) {
    if emit.count > 0 && emit[0] == asc("=") {
        emit.removeFirst()
        let insertAt = findLastStressableVowel(phonemes, phonemes.count)
        if insertAt >= 0 {
            var beforeVowel: UInt8 = 0
            var pi = insertAt - 1
            while pi >= 0 && beforeVowel == 0 {
                if phonemes[pi] != phBnd { beforeVowel = phonemes[pi] }
                pi -= 1
            }
            if beforeVowel != asc("'") {
                phonemes.insert(asc("'"), at: insertAt)
                for di in 0..<insertAt where phonemes[di] == asc("'") {
                    phonemes[di] = phSec2
                }
            }
        }
    }
}

let sMC2 = bsv([
    "aI@3", "aU@r", "i@3r", "aI@", "aI3", "aU@", "i@3", "3:r", "A:r",
    "o@r", "A@r", "e@r",
    "eI", "aI", "aU", "OI", "oU", "tS", "dZ", "IR", "VR",
    "e@", "i@", "U@", "A@", "O@", "o@",
    "3:", "A:", "i:", "u:", "O:", "e:", "a:", "aa",
    "@L", "@2", "@5",
    "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02", "O2",
    "A~", "O~", "A#",
    "r-", "w#", "t#", "d#", "z#", "t2", "d2", "n-", "m-", "l/", "z/",
])

struct PhonemeCode {
    var code = [UInt8]()
    var len = 0
}

func findPhonemeCode(_ ph: [UInt8], _ pn: Int,
                     _ pos: Int) -> PhonemeCode {
    var r = PhonemeCode()
    var found = false
    var mi = 0
    while mi < sMC2.count && !found {
        let mc = sMC2[mi]
        if pos + mc.count <= pn && equalRange(ph, pos, mc, mc.count) {
            r.code = mc
            r.len = mc.count
            found = true
        }
        mi += 1
    }
    if !found {
        r.code = [ph[pos]]
        r.len = 1
    }
    return r
}

// Strips every existing stress mark, then '\'' before the n-th vowel code.
func applyStressPosition(_ raw: [UInt8], _ rn: Int,
                         _ n: Int) -> [UInt8] {
    var out = [UInt8]()
    out.reserveCapacity(rn + 1)
    for i in 0..<rn where raw[i] != asc("'") && raw[i] != asc(",") {
        out.append(raw[i])
    }
    var vowelCount = 0
    var pi = 0
    var inserted = false
    while pi < out.count && !inserted {
        let c = out[pi]
        if c == asc("%") || c == asc("=") || c == asc("|") {
            pi += 1
        } else {
            let pc = findPhonemeCode(out, out.count, pi)
            if isVowelCode(pc.code, 0, pc.len) {
                vowelCount += 1
                if vowelCount == n {
                    out.insert(asc("'"), at: pi)
                    inserted = true
                }
            }
            if !inserted { pi += pc.len }
        }
    }
    return out
}
