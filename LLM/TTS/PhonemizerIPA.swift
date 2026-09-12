// Tokenization, phoneme code to IPA, and the IPA-level utilities; no state.
import Foundation

struct Token {
    var text = [UInt8]()
    var isWord = false
    var needsSpace = false
}

func flushWord(_ out: inout [Token], _ cur: inout [UInt8]) {
    if cur.count > 0 {
        var tk = Token()
        tk.isWord = true
        tk.needsSpace = out.count > 0
        tk.text = cur
        out.append(tk)
        cur = []
    }
}

func emitPunct(_ out: inout [Token], _ cur: inout [UInt8], _ c: UInt8) {
    flushWord(&out, &cur)
    var tk = Token()
    tk.text = [c]
    out.append(tk)
}

func takeUtf8Sequence(_ text: [UInt8], _ n: Int, _ start: Int,
                      _ cur: inout [UInt8]) -> Int {
    var i = start
    cur.append(text[i])
    i += 1
    while i < n && (text[i] & 0xC0) == 0x80 {
        cur.append(text[i])
        i += 1
    }
    return i
}

// "U." starts an abbreviation: a single capital, the period, a capital.
func startsAbbreviation(_ text: [UInt8], _ n: Int, _ i: Int,
                        _ cur: [UInt8]) -> Bool {
    return cur.count == 1 && isUpperC(cur[0])
        && i + 1 < n && isUpperC(text[i + 1])
}

// A period inside an accumulating abbreviation ("U.S") closes the word.
func takeTrailingPeriod(_ out: inout [Token],
                        _ cur: inout [UInt8]) {
    if containsByte(cur, asc(".")) {
        cur.append(asc("."))
        flushWord(&out, &cur)
    } else {
        emitPunct(&out, &cur, asc("."))
    }
}

func hyphenJoinsWord(_ text: [UInt8], _ n: Int, _ i: Int,
                     _ cur: [UInt8]) -> Bool {
    return cur.count > 0 && i + 1 < n && isAlphaC(text[i + 1])
}

func tokenize(_ text: [UInt8], _ n: Int) -> [Token] {
    var out = [Token]()
    var cur = [UInt8]()
    var i = 0
    while i < n {
        let c = text[i]
        if c >= 0x80 {
            i = takeUtf8Sequence(text, n, i, &cur)
        } else if isAlphaC(c) || c == asc("'") {
            cur.append(c)
            i += 1
        } else if c == asc(".")
                  && startsAbbreviation(text, n, i, cur) {
            cur.append(asc("."))
            i += 1
        } else if c == asc(".") && cur.count > 0 {
            takeTrailingPeriod(&out, &cur)
            i += 1
        } else if c == asc("-") && hyphenJoinsWord(text, n, i, cur) {
            cur.append(c)
            i += 1
        } else if isDigitC(c) {
            cur.append(c)
            i += 1
        } else if isSpaceC(c) {
            flushWord(&out, &cur)
            i += 1
        } else {
            emitPunct(&out, &cur, c)
            i += 1
        }
    }
    flushWord(&out, &cur)
    return out
}

// ipa1[] mirror: ASCII 0x20..0x7F to a code point; an X-SAMPA-like alphabet.
let asciiToIpa: [UInt32] = [
    0x0020, 0x0021, 0x0022, 0x02b0, 0x0024, 0x0025, 0x00e6, 0x02c8,
    0x0028, 0x0029, 0x027e, 0x002b, 0x02cc, 0x002d, 0x002e, 0x002f,
    0x0252, 0x0031, 0x0032, 0x025c, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0275, 0x0039, 0x02d0, 0x02b2, 0x003c, 0x003d, 0x003e, 0x0294,
    0x0259, 0x0251, 0x03b2, 0x00e7, 0x00f0, 0x025b, 0x0046, 0x0262,
    0x0127, 0x026a, 0x025f, 0x004b, 0x026b, 0x0271, 0x014b, 0x0254,
    0x03a6, 0x0263, 0x0280, 0x0283, 0x03b8, 0x028a, 0x028c, 0x0153,
    0x03c7, 0x00f8, 0x0292, 0x032a, 0x005c, 0x005d, 0x005e, 0x005f,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0261,
    0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x0303, 0x007f,
]

func appendUtf8(_ out: inout [UInt8], _ cp: UInt32) {
    if cp < 0x80 {
        out.append(UInt8(cp))
    } else if cp < 0x800 {
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    } else if cp < 0x10000 {
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    } else {
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    }
}

// `isVowel` is unused, kept for signature parity.
func phonemeCodeToIpaTable(_ code: [UInt8], _ at: Int, _ cn: Int,
                           _ isVowel: Bool, _ out: inout [UInt8]) {
    var first = true
    var stop = false
    var i = 0
    while i < cn && !stop {
        let c = code[at + i]
        if c == asc("/") {
            stop = true
        } else if !first && c >= asc("0") && c <= asc("9") {
            // variant digit after the first char
        } else if c == asc("#") {
            stop = true
        } else if c == asc("|") {
            // pipe separator
        } else if c == asc("_") && first {
            stop = true
        } else {
            if c >= 0x20 && c < 0x80 {
                appendUtf8(&out, asciiToIpa[Int(c) - 0x20])
            } else if c >= 0x80 {
                out.append(c)
            }
            first = false
        }
        i += 1
    }
}

// Longest-first, so the greedy match prefers "aU@r" over "aU@" or "aU".
let multiCodes = bsv([
    "aI@3", "aU@r", "i@3r",
    "aI@", "aI3", "aU@", "i@3", "3:r", "A:r", "I2#",
    "o@r", "O@r", "e@r",
    "eI", "aI", "aU", "OI", "oU", "tS", "dZ", "IR", "VR",
    "e@", "i@", "U@", "A@", "O@", "o@",
    "3:", "A:", "i:", "u:", "O:", "e:", "a:",
    "aa",
    "@L", "@2", "@5",
    "r-", "w#", "t#", "d#", "z#", "t2", "d2", "n-", "m-",
    "l/", "z/",
    "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02",
    "O2", "A~", "O~", "A#",
])

struct IpaOverrideEntry {
    var code: String
    var ipa: [UInt8]
}

func ipaEntry(_ code: String, _ bytes: [UInt8]) -> IpaOverrideEntry {
    IpaOverrideEntry(code: code, ipa: bytes)
}

let ipaCommon: [IpaOverrideEntry] = [
    ipaEntry("r", [0xc9, 0xb9]),
    ipaEntry("r-", [0xc9, 0xb9]),
    ipaEntry("n-", [0x6e, 0xcc, 0xa9]),
    ipaEntry("m-", [0x6d, 0xcc, 0xa9]),
    ipaEntry("3:r", [0xc9, 0x9c, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("3:", [0xc9, 0x9c, 0xcb, 0x90]),
    ipaEntry("@L", [0xc9, 0x99, 0x6c]),
    ipaEntry("a#", [0xc9, 0x90]),
    ipaEntry("e#", [0xc9, 0x9b]),
    ipaEntry("I#", [0xe1, 0xb5, 0xbb]),
    ipaEntry("I2#", [0xe1, 0xb5, 0xbb]),
    ipaEntry("w#", [0xca, 0x8d]),
    ipaEntry("@2", [0xc9, 0x99]),
    ipaEntry("@5", [0xc9, 0x99]),
    ipaEntry("I2", [0xc9, 0xaa]),
]

let ipaEnUs: [IpaOverrideEntry] = [
    ipaEntry("3", [0xc9, 0x9a]),
    ipaEntry("a", [0xc3, 0xa6]),
    ipaEntry("aa", [0xc3, 0xa6]),
    ipaEntry("0", [0xc9, 0x91, 0xcb, 0x90]),
    ipaEntry("0#", [0xc9, 0x91, 0xcb, 0x90]),
    ipaEntry("A#", [0xc9, 0x91, 0xcb, 0x90]),
    ipaEntry("A@", [0xc9, 0x91, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("A:r", [0xc9, 0x91, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("e@", [0xc9, 0x9b, 0xc9, 0xb9]),
    ipaEntry("e@r", [0xc9, 0x9b, 0xc9, 0xb9]),
    ipaEntry("U@", [0xca, 0x8a, 0xc9, 0xb9]),
    ipaEntry("O@", [0xc9, 0x94, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("O@r", [0xc9, 0x94, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("o@", [0x6f, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("o@r", [0x6f, 0xcb, 0x90, 0xc9, 0xb9]),
    ipaEntry("i@", [0x69, 0xc9, 0x99]),
    ipaEntry("i@3", [0xc9, 0xaa, 0xc9, 0xb9]),
    ipaEntry("i@3r", [0xc9, 0xaa, 0xc9, 0xb9]),
    ipaEntry("aI@", [0x61, 0xc9, 0xaa, 0xc9, 0x99]),
    ipaEntry("aI3", [0x61, 0xc9, 0xaa, 0xc9, 0x9a]),
    ipaEntry("aU@", [0x61, 0xc9, 0xaa, 0xca, 0x8a, 0xc9, 0xb9]),
    ipaEntry("IR", [0xc9, 0x99, 0xc9, 0xb9]),
    ipaEntry("VR", [0xca, 0x8c, 0xc9, 0xb9]),
    ipaEntry("02", [0xca, 0x8c]),
    ipaEntry("i", [0x69]),
]

let ipaEnGb: [IpaOverrideEntry] = [
    ipaEntry("3", [0xc9, 0x9a]),
    ipaEntry("a", [0x61]),
    ipaEntry("aa", [0x61]),
    ipaEntry("0", [0xc9, 0x92]),
    ipaEntry("oU", [0xc9, 0x99, 0xca, 0x8a]),
    ipaEntry("A@", [0xc9, 0x91, 0xcb, 0x90]),
    ipaEntry("IR", [0xc9, 0x99, 0xc9, 0xb9]),
]

func containsStressMarker(_ ipa: [UInt8], _ n: Int) -> Bool {
    var found = false
    var i = 0
    while i + 1 < n && !found {
        if ipa[i] == 0xCB && (ipa[i + 1] == 0x88 || ipa[i + 1] == 0x8C) {
            found = true
        }
        i += 1
    }
    return found
}

func isAsciiVowelChar(_ c: UInt8) -> Bool {
    let lc = toLowerC(c)
    return lc == asc("a") || lc == asc("e") || lc == asc("i")
        || lc == asc("o") || lc == asc("u")
}

func isIpaVowelC9(_ c2: UInt8) -> Bool {
    return c2 == 0x90 || c2 == 0x91 || c2 == 0x92
        || c2 == 0x94 || c2 == 0x99 || c2 == 0x9A
        || c2 == 0x9B || c2 == 0x9C || c2 == 0x9E
        || c2 == 0xAA || c2 == 0xB5
}

// Two-byte IPA vowels: C3 (ae, o-slash), C5 (oe), the C9 block, CA (U, V).
// U+0153 oe encodes as C5 93, not C3 93.
func isIpaVowelPair(_ c: UInt8, _ c2: UInt8) -> Bool {
    var r = false
    if c == 0xC3 {
        r = c2 == 0xA6 || c2 == 0xB8
    } else if c == 0xC5 {
        r = c2 == 0x93
    } else if c == 0xC9 {
        r = isIpaVowelC9(c2)
    } else if c == 0xCA {
        r = c2 == 0x8A || c2 == 0x8C
    }
    return r
}

func isIpaVowelStart(_ s: [UInt8], _ n: Int, _ i: Int) -> Bool {
    var r = false
    if i < n {
        let c = s[i]
        if c < 0x80 {
            r = isAsciiVowelChar(c)
        } else if i + 1 < n {
            let c2 = s[i + 1]
            if c == 0xE1 && i + 2 < n {
                r = c2 == 0xB5 && s[i + 2] == 0xBB
            } else {
                r = isIpaVowelPair(c, c2)
            }
        }
    }
    return r
}

func isIpaSchwa(_ s: [UInt8], _ n: Int, _ i: Int) -> Bool {
    i + 1 < n && s[i] == 0xC9 && s[i + 1] == 0x99
}

func utf8Step(_ c: UInt8) -> Int {
    var step = 4
    if c < 0x80 { step = 1 }
    else if c < 0xE0 { step = 2 }
    else if c < 0xF0 { step = 3 }
    return step
}

struct IpaVowelHit {
    var pos = -1
    var isSchwa = false
}

func findFirstIpaVowel(_ s: [UInt8], _ n: Int,
                       _ start: Int) -> IpaVowelHit {
    var r = IpaVowelHit()
    var i = start
    while i < n && r.pos < 0 {
        if isIpaVowelStart(s, n, i) {
            r.pos = i
            r.isSchwa = isIpaSchwa(s, n, i)
        } else {
            i += utf8Step(s[i])
        }
    }
    return r
}

func findFirstNonSchwaVowel(_ s: [UInt8], _ n: Int,
                            _ start: Int) -> Int {
    var pos = -1
    var j = start
    while j < n && pos < 0 {
        if isIpaVowelStart(s, n, j) && !isIpaSchwa(s, n, j) {
            pos = j
        } else {
            j += utf8Step(s[j])
        }
    }
    return pos
}

let ipaStress: [UInt8] = [0xcb, 0x88]

// Skips an initial schwa when a non-schwa vowel follows.
func addDefaultStress(_ ipa: [UInt8], _ n: Int) -> [UInt8] {
    var out = [UInt8]()
    if n == 0 {
        // empty stays empty
    } else if containsStressMarker(ipa, n) {
        out = Array(ipa[0..<n])
    } else {
        let first = findFirstIpaVowel(ipa, n, 0)
        if first.pos < 0 {
            out = Array(ipa[0..<n])
        } else {
            var target = first.pos
            if first.isSchwa {
                let nonSchwa = findFirstNonSchwaVowel(ipa, n,
                                                      first.pos + 2)
                if nonSchwa >= 0 { target = nonSchwa }
            }
            out.append(contentsOf: ipa[0..<target])
            out.append(contentsOf: ipaStress)
            out.append(contentsOf: ipa[target..<n])
        }
    }
    return out
}

let numOnes = [
    "", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]

let numTens = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]

func intToWords(_ n0: Int, _ out: inout [UInt8]) {
    var n = n0
    if n == 0 {
        out.append(contentsOf: bs("zero"))
    } else {
        if n >= 1000000 {
            intToWords(n / 1000000, &out)
            out.append(contentsOf: bs(" million"))
            n %= 1000000
            if n > 0 { out.append(asc(" ")) }
        }
        if n >= 1000 {
            intToWords(n / 1000, &out)
            out.append(contentsOf: bs(" thousand"))
            n %= 1000
            if n > 0 { out.append(asc(" ")) }
        }
        if n >= 100 {
            out.append(contentsOf: bs(numOnes[n / 100]))
            out.append(contentsOf: bs(" hundred"))
            n %= 100
            if n > 0 { out.append(asc(" ")) }
        }
        if n >= 20 {
            out.append(contentsOf: bs(numTens[n / 10]))
            n %= 10
            if n > 0 {
                out.append(asc(" "))
                out.append(contentsOf: bs(numOnes[n]))
            }
        } else if n > 0 {
            out.append(contentsOf: bs(numOnes[n]))
        }
    }
}

// The rhotic schwa or the long 3: are the r-sandhi candidates.
func endsInRhoticR(_ ipa: [UInt8], _ n: Int) -> Bool {
    let endsInSchwar = n >= 2 && ipa[n - 2] == 0xC9 && ipa[n - 1] == 0x9A
    let endsInLongEr = n >= 4 && ipa[n - 4] == 0xC9 && ipa[n - 3] == 0x9C
                    && ipa[n - 2] == 0xCB && ipa[n - 1] == 0x90
    return endsInSchwar || endsInLongEr
}

func findNextWordStopOnPunct(_ ts: [Token], _ start: Int) -> Int {
    var found = -1
    var stop = false
    var tj = start
    while tj < ts.count && found < 0 && !stop {
        if ts[tj].isWord {
            if ts[tj].text.count > 0 { found = tj }
        } else {
            stop = true
        }
        tj += 1
    }
    return found
}

func findNextNonEmptyWord(_ ts: [Token], _ start: Int) -> Int {
    var found = -1
    var tk = start
    while tk < ts.count && found < 0 {
        if ts[tk].isWord && ts[tk].text.count > 0 { found = tk }
        tk += 1
    }
    return found
}

// "or"/"and" between vowels suppresses linking-r.
func conjunctionSuppressesLinking(_ ts: [Token], _ tj: Int,
                                  _ firstWord: Bool) -> Bool {
    var suppress = false
    let lc = toLower(ts[tj].text)
    let isConjunction = (lc.count == 2 && lc == bs("or"))
        || (lc.count == 3 && lc == bs("and"))
    if isConjunction && !firstWord {
        suppress = findNextWordStopOnPunct(ts, tj + 1) >= 0
    }
    return suppress
}

let twoCharCodes = bsv([
    "tS", "dZ", "O:", "A:", "i:", "u:", "e:", "I#", "I2", "@L",
    "3:", "eI", "aI", "aU", "oU", "OI",
])

func lastPhonemeCode(_ rc: [UInt8]) -> [UInt8] {
    var out = [UInt8]()
    var ri = rc.count
    var found = false
    while ri > 0 && !found {
        ri -= 1
        let c = rc[ri]
        let isMarker = c == asc("'") || c == asc(",") || c == asc("%")
                    || c == asc("=") || c == asc("-")
        if !isMarker {
            out = [c]
            if ri >= 1 {
                let two: [UInt8] = [rc[ri - 1], c]
                if wordListHas(twoCharCodes, two) { out = two }
            }
            found = true
        }
    }
    return out
}

let diphs = bsv(["eI", "aI", "aU", "OI", "oU"])

// The whole diphthong digraph carries the primary, not its second half.
func fixDiphthongStressPosition(_ phCodes: inout [UInt8]) {
    var si = 1
    while si + 1 < phCodes.count {
        if phCodes[si] == asc("'") {
            let prev = phCodes[si - 1]
            let next = phCodes[si + 1]
            var matched = false
            var di = 0
            while di < diphs.count && !matched {
                matched = diphs[di][0] == prev && diphs[di][1] == next
                if matched {
                    phCodes.remove(at: si)
                    phCodes.insert(asc("'"), at: si - 1)
                }
                di += 1
            }
        }
        si += 1
    }
}

// Pre-vowel "the": the trailing schwa becomes a short i.
func applyPreVowelTheFixup(_ phrasePreVowelThe: Bool,
                           _ ipa: inout [UInt8]) {
    if phrasePreVowelThe && ipa.count >= 2
        && ipa[ipa.count - 2] == 0xc9 && ipa[ipa.count - 1] == 0x99 {
        ipa[ipa.count - 1] = 0xaa
    }
}

let stepBKeepSecondaryWords = bsv([
    "within", "without", "about", "across", "above", "among",
    "amongst", "before", "upon", "below", "beside", "between",
    "beyond", "despite", "except", "inside", "outside", "toward",
    "towards", "along", "around", "behind", "beneath",
    "underneath", "over", "under",
])

let stepCKeepSecondary = bsv([
    "on", "onto", "multiple", "multiples", "going",
    "into", "any", "how", "where", "why",
    "being", "while", "but",
    "across", "above", "among", "amongst", "before",
    "within", "without", "upon", "below", "beside", "between",
    "beyond", "underneath", "behind", "beneath",
    "over", "under",
    "about",
    "make", "makes",
])

let stepCNeedsSecondary = bsv(["our"])

func wordListHas(_ list: [[UInt8]], _ w: [UInt8]) -> Bool {
    var found = false
    var i = 0
    while i < list.count && !found {
        if list[i] == w { found = true }
        i += 1
    }
    return found
}

let strongVowelsC = bs("aAeEiIoOuUV3")
let vowelCharsC = bs("aAeEiIoOuUV03@")

func firstCommaPos(_ phCodes: [UInt8]) -> Int {
    var commaPos = phCodes.count
    var i = 0
    while i < phCodes.count && commaPos == phCodes.count {
        if phCodes[i] == asc(",") { commaPos = i }
        i += 1
    }
    return commaPos
}

// A '%' may stand between the string start and the comma.
func commaAtStringStart(_ phCodes: [UInt8], _ commaPos: Int) -> Bool {
    return commaPos == 0
        || (commaPos == 1 && phCodes.count > 0
            && phCodes[0] == asc("%"))
}

// Stress markers do not count as vowels for this scan.
func noRealVowelBeforeComma(_ phCodes: [UInt8],
                            _ commaPos: Int) -> Bool {
    var none = true
    var k = 0
    while k < commaPos && none {
        let cc = phCodes[k]
        if cc != asc("'") && cc != asc("%") && cc != asc("=")
            && strchrHit(vowelCharsC, cc) {
            none = false
        }
        k += 1
    }
    return none
}

// A comma with no vowel before it leads the word.
func commaIsLeading(_ phCodes: [UInt8], _ commaPos: Int) -> Bool {
    let atStart = commaAtStringStart(phCodes, commaPos)
    let effectivelyLeading = !atStart && commaPos < phCodes.count
        && noRealVowelBeforeComma(phCodes, commaPos)
    return atStart || effectivelyLeading
}

// A phrase-level '%' prefix owns the stress, so the comma stays.
func isPctPhrasePrefix(_ phCodes: [UInt8], _ phraseMatched: Bool,
                       _ isIsolatedWord: Bool,
                       _ hasPrime: Bool) -> Bool {
    return phraseMatched && phCodes.count > 0
        && phCodes[0] == asc("%") && !isIsolatedWord && !hasPrime
}

func applyCommaToPrimaryPromotion(_ keepSec: Bool, _ needsSec: Bool,
                                  _ phraseMatched: Bool,
                                  _ isIsolatedWord: Bool,
                                  _ phCodes: inout [UInt8]) {
    let commaPos = firstCommaPos(phCodes)
    let leadingComma = commaIsLeading(phCodes, commaPos)
    let hasPrime = containsByte(phCodes, asc("'"))
    let shouldPromote = !keepSec && !needsSec && !hasPrime
        && commaPos < phCodes.count
        && !isPctPhrasePrefix(phCodes, phraseMatched,
                              isIsolatedWord, hasPrime)
        && (isIsolatedWord || !leadingComma)
    if shouldPromote { replaceFirstChar(&phCodes, asc(","), asc("'")) }
}

let cliticIpa: [IpaOverrideEntry] = [
    ipaEntry("of a", [0xc9, 0x99, 0x76, 0xc9, 0x99]),
    ipaEntry("of the", [0xca, 0x8c, 0x76, 0xc3, 0xb0, 0xc9, 0x99]),
    ipaEntry("in the", [0xc9, 0xaa, 0x6e, 0xc3, 0xb0, 0xc9, 0x99]),
    ipaEntry("on the", [0xc9, 0x94, 0x6e, 0xc3, 0xb0, 0xc9, 0x99]),
    ipaEntry("from the",
             [0x66, 0xc9, 0xb9, 0xca, 0x8c, 0x6d, 0xc3, 0xb0,
              0xc9, 0x99]),
    ipaEntry("that a",
             [0xc3, 0xb0, 0xcb, 0x8c, 0xc3, 0xa6, 0xc9, 0xbe,
              0xc9, 0x99]),
    ipaEntry("i am", [0x61, 0xc9, 0xaa, 0xc9, 0x90, 0x6d]),
    ipaEntry("was a", [0x77, 0xca, 0x8c, 0x7a, 0xc9, 0x90]),
    ipaEntry("to be", [0x74, 0xc9, 0x99, 0x62, 0x69]),
    ipaEntry("out of",
             [0xcb, 0x8c, 0x61, 0xca, 0x8a, 0xc9, 0xbe, 0xc9, 0x99,
              0x76]),
]

let staticPhraseCodes: [IpaOverrideEntry] = [
    ipaEntry("has been", bs("h'azbi:n")),
]

func lookupConstTable(_ t: [IpaOverrideEntry],
                      _ k: [UInt8]) -> [UInt8]? {
    var result: [UInt8]?
    var i = 0
    while i < t.count && result == nil {
        if bs(t[i].code) == k { result = t[i].ipa }
        i += 1
    }
    return result
}

func buildLowerBigram(_ t1: Token, _ t2: Token) -> [UInt8] {
    var out = toLower(t1.text)
    out.append(asc(" "))
    out.append(contentsOf: toLower(t2.text))
    return out
}

let pronounsUse = bsv(["i", "we", "you", "they", "he", "she", "who"])
