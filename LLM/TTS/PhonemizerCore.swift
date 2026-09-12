// Port of phonemizer.c from github.com/leok7v/tts.cli, in the C's own order.
// Every buffer is [UInt8]: 0x01 and 0x02 are in-band sentinels.
import Foundation

@inline(__always) func asc(_ s: Unicode.Scalar) -> UInt8 {
    UInt8(ascii: s)
}

func bs(_ s: String) -> [UInt8] { Array(s.utf8) }

func bsv(_ s: [String]) -> [[UInt8]] { s.map { e in Array(e.utf8) } }

let phBnd: UInt8 = 0x01
let phSec2: UInt8 = 0x02

// C strchr searches the terminator too, so NUL always hits; devoice_ed_suffix
// relies on it for an all-marker stem.
@inline(__always) func strchrHit(_ set: [UInt8], _ c: UInt8) -> Bool {
    c == 0 || set.contains(c)
}

@inline(__always) func isSpaceC(_ c: UInt8) -> Bool {
    c == 0x20 || (c >= 0x09 && c <= 0x0D)
}

@inline(__always) func isDigitC(_ c: UInt8) -> Bool {
    c >= asc("0") && c <= asc("9")
}

@inline(__always) func isUpperC(_ c: UInt8) -> Bool {
    c >= asc("A") && c <= asc("Z")
}

@inline(__always) func isLowerC(_ c: UInt8) -> Bool {
    c >= asc("a") && c <= asc("z")
}

@inline(__always) func isAlphaC(_ c: UInt8) -> Bool {
    isUpperC(c) || isLowerC(c)
}

@inline(__always) func toLowerC(_ c: UInt8) -> UInt8 {
    isUpperC(c) ? c + 32 : c
}

@inline(__always)
func matchAt(_ a: [UInt8], _ at: Int, _ b: [UInt8]) -> Bool {
    var same = at >= 0 && at + b.count <= a.count
    var i = 0
    while same && i < b.count {
        if a[at + i] != b[i] { same = false }
        i += 1
    }
    return same
}

@inline(__always)
func equalRange(_ a: [UInt8], _ at: Int, _ b: [UInt8], _ n: Int) -> Bool {
    var same = at >= 0 && at + n <= a.count && n <= b.count
    var i = 0
    while same && i < n {
        if a[at + i] != b[i] { same = false }
        i += 1
    }
    return same
}

@inline(__always)
func firstIndexOf(_ a: [UInt8], _ c: UInt8) -> Int {
    var found = -1
    var i = 0
    while i < a.count && found < 0 {
        if a[i] == c { found = i }
        i += 1
    }
    return found
}

@inline(__always)
func firstIndexOf(_ a: [UInt8], _ c: UInt8, from: Int) -> Int {
    var found = -1
    var i = from
    while i < a.count && found < 0 {
        if a[i] == c { found = i }
        i += 1
    }
    return found
}

@inline(__always)
func containsByte(_ a: [UInt8], _ c: UInt8) -> Bool {
    firstIndexOf(a, c) >= 0
}

func containsSub(_ a: [UInt8], _ b: [UInt8]) -> Bool {
    var found = false
    var i = 0
    while !found && i + b.count <= a.count {
        if matchAt(a, i, b) { found = true }
        i += 1
    }
    return found
}

func isVowelLetter(_ c0: UInt8) -> Bool {
    let c = toLowerC(c0)
    return c == asc("a") || c == asc("e") || c == asc("i")
        || c == asc("o") || c == asc("u") || c == asc("y")
}

func hasAnyVowelLetter(_ s: [UInt8], _ from: Int, _ n: Int) -> Bool {
    var found = false
    for i in 0..<n where !found {
        if isVowelLetter(s[from + i]) { found = true }
    }
    return found
}

let vowelCodeChars = bs("aAeEiIoOuUV03@")

func hasAnyVowelCode(_ s: [UInt8]) -> Bool {
    var found = false
    for i in 0..<s.count where !found {
        if strchrHit(vowelCodeChars, s[i]) { found = true }
    }
    return found
}

func toLower(_ s: [UInt8], _ from: Int, _ n: Int) -> [UInt8] {
    var out = [UInt8]()
    out.reserveCapacity(n)
    for i in 0..<n { out.append(toLowerC(s[from + i])) }
    return out
}

func toLower(_ s: [UInt8]) -> [UInt8] { toLower(s, 0, s.count) }

func trim(_ s: [UInt8], _ from: Int, _ n: Int) -> [UInt8] {
    var start = 0
    while start < n && (s[from + start] == asc(" ")
                        || s[from + start] == asc("\t")
                        || s[from + start] == asc("\r")
                        || s[from + start] == asc("\n")) {
        start += 1
    }
    var end = n
    while end > start && (s[from + end - 1] == asc(" ")
                          || s[from + end - 1] == asc("\t")
                          || s[from + end - 1] == asc("\r")
                          || s[from + end - 1] == asc("\n")) {
        end -= 1
    }
    var out = [UInt8]()
    if end > start { out = Array(s[(from + start)..<(from + end)]) }
    return out
}

func replaceFirstChar(_ s: inout [UInt8], _ from: UInt8, _ to: UInt8) {
    var i = 0
    while i < s.count && s[i] != from { i += 1 }
    if i < s.count { s[i] = to }
}

func parseStressN(_ flag: [UInt8], _ from: Int, _ len: Int) -> Int {
    var n = 0
    if len == 2 && flag[from] == asc("$")
        && flag[from + 1] >= asc("1") && flag[from + 1] <= asc("6") {
        n = Int(flag[from + 1] - asc("0"))
    }
    return n
}

func parseStressN(_ flag: [UInt8]) -> Int {
    parseStressN(flag, 0, flag.count)
}

struct EntryFlags {
    var noun = false, verb = false, past = false
    var pastf = false, nounf = false, verbf = false
    var atend = false, capital = false, atstart = false
    var onlys = false, only = false
    var grammar = false
    var strend2 = false, u2 = false
    var stressN = 0
}

func splitWS(_ s: [UInt8], _ from: Int, _ n: Int) -> [[UInt8]] {
    var out = [[UInt8]]()
    var tok = [UInt8]()
    for i in 0..<n {
        if isSpaceC(s[from + i]) {
            if tok.count > 0 {
                out.append(tok)
                tok = []
            }
        } else {
            tok.append(s[from + i])
        }
    }
    if tok.count > 0 { out.append(tok) }
    return out
}

func splitWS(_ s: [UInt8]) -> [[UInt8]] { splitWS(s, 0, s.count) }

struct LeadingDialect {
    var cond = 0
    var negated = false
    var rest = [UInt8]()
}

// The .cpp swallows a stoi throw (cond stays 0) but ALWAYS strips the
// "?<x>" prefix, so the cond is gated on a digit run while the trim is not.
func parseLeadingDialect(_ s: [UInt8], _ n: Int) -> LeadingDialect {
    var r = LeadingDialect()
    var matched = false
    if n > 0 && s[0] == asc("?") {
        var space = 0
        while space < n && s[space] != asc(" ") && s[space] != asc("\t") {
            space += 1
        }
        if space < n {
            var cs = 1
            let ce = space
            if cs < ce && s[cs] == asc("!") {
                r.negated = true
                cs += 1
            }
            var v = 0
            var anyDigit = false
            while cs < ce && isDigitC(s[cs]) {
                v = v * 10 + Int(s[cs] - asc("0"))
                cs += 1
                anyDigit = true
            }
            if anyDigit && cs == ce { r.cond = v }
            r.rest = trim(s, space, n - space)
            matched = true
        }
    }
    if !matched { r.rest = Array(s[0..<n]) }
    return r
}

func stripCommentAndTrim(_ raw: [UInt8], _ n: Int) -> [UInt8] {
    var cut = n
    var i = 0
    while i + 1 < n {
        if cut == n && raw[i] == asc("/") && raw[i + 1] == asc("/") {
            cut = i
        }
        i += 1
    }
    return trim(raw, 0, cut)
}

// Classified by the first byte only, as the C++ original does.
func isVowelCode(_ code: [UInt8], _ at: Int, _ n: Int) -> Bool {
    var r = false
    if n > 0 {
        let c = code[at]
        r = c == asc("@") || c == asc("a") || c == asc("A")
         || c == asc("E") || c == asc("I") || c == asc("i")
         || c == asc("O") || c == asc("U") || c == asc("V")
         || c == asc("0") || c == asc("3") || c == asc("e")
         || c == asc("o") || c == asc("u")
    }
    return r
}

func isVowelCode(_ code: [UInt8]) -> Bool {
    isVowelCode(code, 0, code.count)
}

let vowLet = bs("aeiouAEIOU")

// A silent trailing magic-e adds no syllable.
func countSuffixSyllables(_ s: [UInt8], _ from: Int, _ n: Int) -> Int {
    var r = 0
    var inVowel = false
    for i in 0..<n {
        if strchrHit(vowLet, s[from + i]) {
            if !inVowel { r += 1; inVowel = true }
        } else {
            inVowel = false
        }
    }
    if n > 0 && (s[from + n - 1] == asc("e") || s[from + n - 1] == asc("E"))
        && n >= 2 && !strchrHit(vowLet, s[from + n - 2]) {
        r = r - 1 > 1 ? r - 1 : 1
    }
    return r
}

// The three local-static tables in the .cpp are byte-identical, merged here.
let vowelCodesMulti = bsv([
    "aI@3", "aU@r", "i@3r",
    "aI@",  "aI3",  "aU@",  "i@3", "3:r", "A:r",
    "o@r",  "A@r",  "e@r",
    "eI", "aI", "aU", "OI", "oU", "IR", "VR", "U@",
    "A@", "e@", "i@", "O@", "o@", "3:", "A:", "i:", "u:",
    "O:", "e:", "a:",
    "aa", "@L", "@2", "@5", "I2", "I#", "E2", "E#", "e#",
    "a#", "a2", "0#", "02", "O2", "A#",
])

let fullVowels = bsv([
    "eI", "aI", "aU", "OI", "oU", "3:", "A:", "i:", "u:",
    "O:", "e:", "a:",
    "aI@", "aU@", "oU#", "i@", "e@", "A@", "U@", "O@",
    "a", "A", "E", "V", "0", "o",
])

func prefixHasFullVowel(_ phonemes: [UInt8]) -> Bool {
    var found = false
    var pi = 0
    while pi < phonemes.count && !found {
        var fi = 0
        while fi < fullVowels.count && !found {
            found = matchAt(phonemes, pi, fullVowels[fi])
            fi += 1
        }
        pi += 1
    }
    return found
}

func countPrefixVowels(_ phonemes: [UInt8]) -> Int {
    var r = 0
    var pi = 0
    while pi < phonemes.count {
        let c = phonemes[pi]
        if c == asc("'") || c == asc(",") || c == asc("%")
            || c == asc("=") {
            pi += 1
        } else {
            var mch = false
            var mi = 0
            while mi < vowelCodesMulti.count && !mch {
                let mc = vowelCodesMulti[mi]
                mch = matchAt(phonemes, pi, mc)
                if mch {
                    if isVowelCode(mc) { r += 1 }
                    pi += mc.count
                }
                mi += 1
            }
            if !mch {
                if isVowelCode(phonemes, pi, 1) { r += 1 }
                pi += 1
            }
        }
    }
    return r
}

func lastVowelCode(_ phonemes: [UInt8]) -> [UInt8] {
    var lastV = [UInt8]()
    var pi = 0
    while pi < phonemes.count {
        let c = phonemes[pi]
        if c == asc("'") || c == asc(",") || c == asc("%")
            || c == asc("=") {
            pi += 1
        } else {
            var mch = false
            var mi = 0
            while mi < vowelCodesMulti.count && !mch {
                let mc = vowelCodesMulti[mi]
                mch = matchAt(phonemes, pi, mc)
                if mch {
                    if isVowelCode(mc) { lastV = mc }
                    pi += mc.count
                }
                mi += 1
            }
            if !mch {
                if isVowelCode(phonemes, pi, 1) { lastV = [phonemes[pi]] }
                pi += 1
            }
        }
    }
    return lastV
}

// 3, 3:, @, @2, @5, @L, I2, I#, a#. An empty code is no vowel at all.
func isSchwaCode(_ v: [UInt8]) -> Bool {
    var r = false
    if v.count == 1 {
        r = v[0] == asc("3") || v[0] == asc("@")
    } else if v.count == 2 {
        r = (v[0] == asc("3") && v[1] == asc(":"))
         || (v[0] == asc("@") && (v[1] == asc("2") || v[1] == asc("5")
                                  || v[1] == asc("L")))
         || (v[0] == asc("I") && (v[1] == asc("2") || v[1] == asc("#")))
         || (v[0] == asc("a") && v[1] == asc("#"))
    }
    return r
}

// Drives the prefix-stress heuristic ("super" + "nova" keeps the prefix).
func prefixEndsInSchwa(_ phonemes: [UInt8]) -> Bool {
    isSchwaCode(lastVowelCode(phonemes))
}

let voicelessLast = bs("ptkfTSCxXhs")

func devoiceEdSuffix(_ stemPh: [UInt8], _ suffix: inout [UInt8]) {
    let isDhash = suffix.count == 2 && suffix[0] == asc("d")
                                   && suffix[1] == asc("#")
    if isDhash && stemPh.count > 0 {
        var lastPh: UInt8 = 0
        var sj = stemPh.count - 1
        while sj >= 0 && lastPh == 0 {
            let c = stemPh[sj]
            if c != asc("'") && c != asc(",") && c != asc("%")
                && c != asc("=") && c != phBnd {
                lastPh = c
            }
            sj -= 1
        }
        if lastPh == asc("t") || lastPh == asc("d") {
            suffix = bs("I#d")
        } else if strchrHit(voicelessLast, lastPh) {
            suffix = bs("t")
        }
    }
}

// Empty base defaults to true; the caller's outer flow decides.
func shouldUseMagicEForCVCStem(_ basePh: [UInt8],
                               _ weakVowels: [UInt8]) -> Bool {
    var useMagicE = true
    if basePh.count > 0 {
        var lastV = -1
        var k = basePh.count - 1
        while k >= 0 && lastV < 0 {
            if strchrHit(vowelCodeChars, basePh[k]) { lastV = k }
            k -= 1
        }
        if lastV > 0 {
            var vstart = lastV
            while vstart > 0
                && (strchrHit(vowelCodeChars, basePh[vstart - 1])
                    || basePh[vstart - 1] == asc(":")
                    || basePh[vstart - 1] == asc("#")) {
                vstart -= 1
            }
            let stressedAtEnd = vstart > 0
                && (basePh[vstart - 1] == asc("'")
                    || basePh[vstart - 1] == asc(","))
            let noExplicitStress = !containsByte(basePh, asc("'"))
            let lastVowelIsFull = !weakVowels.contains(basePh[lastV])
            useMagicE = stressedAtEnd || noExplicitStress
                     || lastVowelIsFull
        }
    }
    return useMagicE
}

// The syllabic context is lost when -ing follows, so "@L" drops to "@l"/"l".

// "-tl" and "-ngl" keep their syllabic l.
func syllabicLBlocked(_ base: [UInt8]) -> Bool {
    let bn = base.count
    let basePrevIsT = bn >= 2 && base[bn - 2] == asc("t")
    let baseEndsInNgl = bn >= 3 && base[bn - 3] == asc("n")
                     && base[bn - 2] == asc("g")
                     && base[bn - 1] == asc("l")
    return basePrevIsT || baseEndsInNgl
}

func simplifySyllabicLForIng(_ base: [UInt8], _ sph: inout [UInt8]) {
    let bn = base.count
    let endsInAtL = sph.count >= 2
                 && sph[sph.count - 2] == asc("@")
                 && sph[sph.count - 1] == asc("L")
    let baseEndsInL = bn >= 2 && base[bn - 1] == asc("l")
    if endsInAtL && baseEndsInL && !syllabicLBlocked(base) {
        let penult = base[bn - 2]
        let vowelBeforeL = penult == asc("a") || penult == asc("e")
                        || penult == asc("i") || penult == asc("o")
                        || penult == asc("u")
        sph.removeLast(2)
        sph.append(contentsOf: vowelBeforeL ? bs("@l") : bs("l"))
    }
}

// SetLetterBits semantics from tr_languages.c.
struct LetterGroups {
    var groupA = [Bool](repeating: false, count: 256)
    var groupB = [Bool](repeating: false, count: 256)
    var groupC = [Bool](repeating: false, count: 256)
    var groupF = [Bool](repeating: false, count: 256)
    var groupG = [Bool](repeating: false, count: 256)
    var groupH = [Bool](repeating: false, count: 256)
    var groupY = [Bool](repeating: false, count: 256)
    var groupK = [Bool](repeating: false, count: 256)
    var lgroups = [[[UInt8]]](repeating: [], count: 100)
}

struct PhonemeRule {
    var condition = 0
    var conditionNegated = false
    var leftCtx = [UInt8]()
    var match = [UInt8]()
    var rightCtx = [UInt8]()
    var phonemes = [UInt8]()
    var delFwd = 0
    var isPrefix = false
    var isSuffix = false
    var suffixStripLen = 0
    var suffixFlags = 0
}

struct ReplaceRule {
    var from = [UInt8]()
    var to = [UInt8]()
}

struct Ruleset {
    var groups = LetterGroups()
    var replacements = [ReplaceRule]()
    var ruleGroups = [[UInt8]: [PhonemeRule]]()
}

func letterBits(_ mask: inout [Bool], _ letters: String) {
    for c in Array(letters.utf8) { mask[Int(c)] = true }
}

func rulesetInit(_ rs: inout Ruleset) {
    letterBits(&rs.groups.groupA, "aeiou")
    letterBits(&rs.groups.groupB, "bcdfgjklmnpqstvxz")
    letterBits(&rs.groups.groupC, "bcdfghjklmnpqrstvwxz")
    letterBits(&rs.groups.groupF, "cfhkpqstx")
    letterBits(&rs.groups.groupG, "bdgjlmnrvwyz")
    letterBits(&rs.groups.groupH, "hlmnr")
    letterBits(&rs.groups.groupY, "aeiouy")
    letterBits(&rs.groups.groupK, "bcdfghjklmnpqrstvwxyz")
}

// SUFX bits from the 'S<N>[flags]' RULE_ENDING marker.
let sufxIBit = 0x200
let sufxMBit = 0x80000
let sufxVBit = 0x800
let sufxEBit = 0x100
let sufxDBit = 0x1000
let sufxQBit = 0x4000
let sufxPBit = 0x400
