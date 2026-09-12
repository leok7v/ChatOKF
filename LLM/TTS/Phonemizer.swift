// The stateful half of the port: the C's `struct phonemizer` and its functions.
import Foundation

struct StrPair {
    var a = [UInt8]()
    var b = [UInt8]()
}

@inline(__always) func pre(_ a: [UInt8], _ n: Int) -> [UInt8] {
    Array(a[0..<n])
}

func lineLenWithoutEol(_ a: [UInt8], _ n0: Int) -> Int {
    var n = n0
    if n > 0 && a[n - 1] == asc("\n") { n -= 1 }
    if n > 0 && a[n - 1] == asc("\r") { n -= 1 }
    return n
}

// Insert-if-absent: an entry already present wins and is left alone.
@discardableResult
func smapEmplace(_ m: inout [[UInt8]: [UInt8]], _ k: [UInt8],
                 _ v: [UInt8]) -> Bool {
    let inserted = m[k] == nil
    if inserted { m[k] = v }
    return inserted
}

// compound_prefixes is sorted by a comparator that returns 0 on ties, so the
// tie order is qsort's own and load-bearing; the same qsort reproduces it.
func sortCompoundPrefixesLikeQsort(_ arr: inout [StrPair]) {
    let n = arr.count
    if n > 1 {
        // sizeof(struct strpair) == 2 * sizeof(struct chars) == 48.
        let es = 48
        let buf = UnsafeMutableRawPointer.allocate(
            byteCount: n * es, alignment: 16)
        buf.initializeMemory(as: UInt8.self, repeating: 0,
                             count: n * es)
        for i in 0..<n {
            let key = Int64(arr[i].a.count) << 32 | Int64(i)
            buf.advanced(by: i * es).storeBytes(of: key, as: Int64.self)
        }
        qsort(buf, n, es) { pa, pb in
            let x = pa!.loadUnaligned(as: Int64.self) >> 32
            let y = pb!.loadUnaligned(as: Int64.self) >> 32
            var r: Int32 = 0
            if x < y { r = 1 } else if x > y { r = -1 }
            return r
        }
        var sorted = [StrPair]()
        sorted.reserveCapacity(n)
        for i in 0..<n {
            let key = buf.advanced(by: i * es).loadUnaligned(
                as: Int64.self)
            sorted.append(arr[Int(key & 0xFFFFFFFF)])
        }
        buf.deallocate()
        arr = sorted
    }
}

final class Phonemizer {
    var err = ""
    var dialect = [UInt8]()
    var loaded = false

    var dict = [[UInt8]: [UInt8]]()
    var verbDict = [[UInt8]: [UInt8]]()
    var pastDict = [[UInt8]: [UInt8]]()
    var nounDict = [[UInt8]: [UInt8]]()
    var ipaOverrides = [[UInt8]: [UInt8]]()
    var atstartDict = [[UInt8]: [UInt8]]()
    var atendDict = [[UInt8]: [UInt8]]()
    var capitalDict = [[UInt8]: [UInt8]]()
    var onlysBareDict = [[UInt8]: [UInt8]]()
    var phraseDict = [[UInt8]: [UInt8]]()

    var pastfWords = Set<[UInt8]>()
    var nounfWords = Set<[UInt8]>()
    var verbfWords = Set<[UInt8]>()
    var unstressedWords = Set<[UInt8]>()
    var unstressendWords = Set<[UInt8]>()
    var abbrevWords = Set<[UInt8]>()
    var onlysWords = Set<[UInt8]>()
    var onlyWords = Set<[UInt8]>()
    var nounFormStress = Set<[UInt8]>()
    var verbFlagWords = Set<[UInt8]>()
    var strendWords = Set<[UInt8]>()
    var u2Strend2Words = Set<[UInt8]>()
    var commaStrend2Words = Set<[UInt8]>()
    var uPlusSecondaryWords = Set<[UInt8]>()
    var keepSecPhraseKeys = Set<[UInt8]>()

    var stressPos = [[UInt8]: Int]()
    var wordAltFlags = [[UInt8]: Int]()

    var phraseSplitDict = [[UInt8]: StrPair]()
    var compoundPrefixes = [StrPair]()

    var rules = Ruleset()

    func determineAltFlags(_ word: [UInt8], _ explicitFlags: Int) -> Int {
        var r = 0
        if explicitFlags >= 0 {
            r = explicitFlags
        } else {
            let wl = toLower(word)
            if let v = wordAltFlags[wl] { r = v }
        }
        return r
    }

    func storeEntryPosSets(_ w: [UInt8], _ f: EntryFlags) {
        if f.pastf { pastfWords.insert(w) }
        if f.nounf { nounfWords.insert(w) }
        if f.verbf { verbfWords.insert(w) }
    }

    func storeEntryStressPos(_ w: [UInt8], _ isFlagOnly: Bool,
                             _ f: EntryFlags) {
        if f.stressN > 0 && !f.noun && !f.verb
            && (!f.grammar || isFlagOnly) {
            if stressPos[w] == nil { stressPos[w] = f.stressN }
            if isFlagOnly && f.onlys { nounFormStress.insert(w) }
        }
    }

    // An "$altN" in the phonemes column retracts an earlier plain entry.
    func storeFlagOnlyEntry(_ w: [UInt8], _ ph: [UInt8],
                            _ f: EntryFlags) {
        if flagIsAltN(ph) { dict.removeValue(forKey: w) }
        if f.verb { verbFlagWords.insert(w) }
    }

    // A dialect-conditional $onlys overwrites; an unconditional one yields to
    // what is there and parks its own form in the bare dict.
    func storeOnlysEntry(_ w: [UInt8], _ ph: [UInt8],
                         _ dialectCond: Int) {
        if dialectCond != 0 {
            dict[w] = ph
            onlysWords.insert(w)
        } else {
            let inserted = smapEmplace(&dict, w, ph)
            if inserted {
                onlysWords.insert(w)
            } else if ph[0] != asc("$") {
                onlysBareDict[w] = ph
            }
        }
    }

    func storePlainEntry(_ w: [UInt8], _ ph: [UInt8], _ f: EntryFlags) {
        dict[w] = ph
        if f.only { onlyWords.insert(w) }
        if f.strend2 && w.count >= 2 && ph.count > 0
            && ph[0] != asc(",") && ph[0] != asc("'")
            && ph[0] != asc("%") {
            var sp = StrPair()
            sp.a = w
            sp.b = ph
            compoundPrefixes.append(sp)
            strendWords.insert(w)
        }
        if f.strend2 && ph.count > 0 && ph[0] == asc(",") {
            commaStrend2Words.insert(w)
        }
        if f.u2 && f.strend2 { u2Strend2Words.insert(w) }
    }

    func storeDictionaryEntry(_ w: [UInt8], _ ph: [UInt8],
                              _ dialectCond: Int,
                              _ f: EntryFlags) {
        storeEntryPosSets(w, f)
        let isFlagOnly = ph.count > 0 && ph[0] == asc("$")
        storeEntryStressPos(w, isFlagOnly, f)
        if isFlagOnly {
            storeFlagOnlyEntry(w, ph, f)
        } else if f.noun {
            smapEmplace(&nounDict, w, ph)
        } else if f.verb {
            smapEmplace(&verbDict, w, ph)
        } else if f.atend {
            if !f.atstart && ph[0] != asc("$") { atendDict[w] = ph }
        } else if f.capital {
            if ph[0] != asc("$") { capitalDict[w] = ph }
        } else if f.atstart {
            atstartDict[w] = ph
        } else if f.past {
            smapEmplace(&pastDict, w, ph)
        } else if f.onlys {
            storeOnlysEntry(w, ph, dialectCond)
        } else {
            storePlainEntry(w, ph, f)
        }
    }

    // "(word1 word2) phonemes [flags]".
    struct PhraseFlags {
        var atend = false
        var pause = false
        var u2Plus = false
    }

    func parsePhraseFlags(_ rp: [[UInt8]]) -> PhraseFlags {
        var f = PhraseFlags()
        var ri = 1
        while ri < rp.count {
            let r = rp[ri]
            if r == bs("$atend") { f.atend = true }
            if r == bs("$pause") { f.pause = true }
            if r == bs("$u2+") { f.u2Plus = true }
            ri += 1
        }
        return f
    }

    // Only a two-word phrase without ".", $atend or $pause is storable.
    func phraseIsStorable(_ words: [[UInt8]],
                          _ f: PhraseFlags) -> Bool {
        words.count == 2 && !f.atend && !f.pause
            && !containsByte(words[0], asc("."))
            && !containsByte(words[1], asc("."))
    }

    func phraseKey(_ words: [[UInt8]]) -> [UInt8] {
        var key = toLower(words[0])
        key.append(asc(" "))
        key.append(contentsOf: toLower(words[1]))
        return key
    }

    // Offset of the "||" that splits a phrase across its two words, or -1.
    func phraseSplitAt(_ phs: [UInt8]) -> Int {
        var k = 0
        while k + 1 < phs.count
            && !(phs[k] == asc("|") && phs[k + 1] == asc("|")) {
            k += 1
        }
        return k + 1 < phs.count ? k : -1
    }

    // A phrase carrying no stress of its own is marked unstressed.
    func storePhrase(_ key: [UInt8], _ phs: [UInt8],
                     _ hasU2Plus: Bool) {
        let hasPrime = containsByte(phs, asc("'"))
        let startsPct = phs.count > 0 && phs[0] == asc("%")
        var phon = [UInt8]()
        if !hasPrime && !startsPct { phon.append(asc("%")) }
        phon.append(contentsOf: phs)
        smapEmplace(&phraseDict, key, phon)
        if hasU2Plus { keepSecPhraseKeys.insert(key) }
    }

    func storePhraseEntry(_ wordsStr: [UInt8], _ rp: [[UInt8]]) {
        let words = splitWS(wordsStr)
        let f = parsePhraseFlags(rp)
        if phraseIsStorable(words, f) {
            let key = phraseKey(words)
            let phs = rp[0]
            let pa = phraseSplitAt(phs)
            if pa >= 0 {
                if phraseSplitDict[key] == nil {
                    var sp = StrPair()
                    sp.a = pre(phs, pa)
                    sp.b = Array(phs[(pa + 2)...])
                    phraseSplitDict[key] = sp
                }
            } else {
                storePhrase(key, phs, f.u2Plus)
            }
        }
    }

    func parsePhraseEntry(_ line: [UInt8]) -> Bool {
        let n = line.count
        let isPhrase = n > 0 && line[0] == asc("(")
        if isPhrase {
            let close = firstIndexOf(line, asc(")"))
            if close > 1 {
                let wordsStr = trim(line, 1, close - 1)
                let restStr = trim(line, close + 1, n - close - 1)
                if restStr.count > 0 && restStr[0] != asc("$") {
                    let rp = splitWS(restStr)
                    if rp.count > 0 && rp[0].count > 0
                        && rp[0][0] != asc("$") {
                        storePhraseEntry(wordsStr, rp)
                    }
                }
            }
        }
        return isPhrase
    }

    // "$alt1".."$alt6"; the digit selects one bit of wordAltFlags.
    func flagIsAltN(_ fd: [UInt8]) -> Bool {
        fd.count == 5 && fd[0] == asc("$") && fd[1] == asc("a")
            && fd[2] == asc("l") && fd[3] == asc("t")
            && fd[4] >= asc("1") && fd[4] <= asc("6")
    }

    func entryFlagPosBits(_ fd: [UInt8], _ flags: inout EntryFlags) {
        if fd == bs("$noun") { flags.noun = true }
        if fd == bs("$verb") { flags.verb = true; flags.grammar = true }
        if fd == bs("$past") { flags.past = true }
        if fd == bs("$pastf") { flags.pastf = true }
        if fd == bs("$nounf") { flags.nounf = true; flags.grammar = true }
        if fd == bs("$verbf") { flags.verbf = true; flags.grammar = true }
    }

    func entryFlagContextBits(_ fd: [UInt8], _ flags: inout EntryFlags) {
        if fd == bs("$atend") || fd == bs("$allcaps")
            || fd == bs("$sentence") {
            flags.atend = true
        }
        if fd == bs("$capital") { flags.capital = true }
        if fd == bs("$atstart") { flags.atstart = true }
    }

    func entryFlagVariantBits(_ fd: [UInt8], _ flags: inout EntryFlags) {
        let eqVerbf = fd == bs("$verbf")
        let eqStrend2 = fd == bs("$strend2")
        let eqAlt2 = fd == bs("$alt2")
        let eqAlt3 = fd == bs("$alt3")
        let eqOnly = fd == bs("$only")
        if eqVerbf || eqStrend2 || eqAlt2 || eqAlt3 || eqOnly {
            flags.grammar = true
        }
        if eqOnly { flags.only = true }
        if fd == bs("$onlys") { flags.onlys = true }
        if eqStrend2 { flags.strend2 = true }
        if fd == bs("$u2") { flags.u2 = true }
        if flags.stressN == 0 {
            flags.stressN = parseStressN(fd, 0, fd.count)
        }
    }

    func entryFlagSets(_ fd: [UInt8], _ w: [UInt8], _ ph: [UInt8]) {
        if fd == bs("$u+") {
            unstressedWords.insert(w)
            let hasComma = containsByte(ph, asc(","))
            let hasPrime = containsByte(ph, asc("'"))
            if hasComma && !hasPrime { uPlusSecondaryWords.insert(w) }
        }
        if fd == bs("$u") { unstressedWords.insert(w) }
        if fd == bs("$unstressend") { unstressendWords.insert(w) }
        if fd == bs("$abbrev") { abbrevWords.insert(w) }
        if flagIsAltN(fd) {
            let bit = 1 << Int(fd[4] - asc("1"))
            wordAltFlags[w] = (wordAltFlags[w] ?? 0) | bit
        }
    }

    // A flag may sit in the phonemes column instead of a phoneme string: "gi
    // $abbrev".
    func phonemeFieldAsFlag(_ w: [UInt8], _ ph: [UInt8],
                            _ flags: inout EntryFlags) {
        let phEqVerb = ph == bs("$verb")
        let phEqVerbf = ph == bs("$verbf")
        let phEqNounf = ph == bs("$nounf")
        let phEqPastf = ph == bs("$pastf")
        let phEqU = ph == bs("$u")
        if ph == bs("$abbrev") { abbrevWords.insert(w) }
        if flagIsAltN(ph) {
            let bit = 1 << Int(ph[4] - asc("1"))
            wordAltFlags[w] = (wordAltFlags[w] ?? 0) | bit
        }
        if phEqVerb || phEqVerbf || phEqNounf || phEqPastf
            || ph == bs("$only") {
            flags.grammar = true
        }
        if phEqPastf { flags.pastf = true }
        if phEqNounf { flags.nounf = true }
        if phEqVerbf { flags.verbf = true }
        if phEqU || ph == bs("$u+") { unstressedWords.insert(w) }
        if phEqU { flags.grammar = true }
        if phEqVerb { flags.verb = true }
    }

    func parseEntryFlags(_ parts: [[UInt8]], _ w: [UInt8],
                         _ ph: [UInt8]) -> EntryFlags {
        var flags = EntryFlags()
        let pn = ph.count
        if pn > 0 && ph[0] == asc("$") {
            flags.stressN = parseStressN(ph)
        }
        var fi = 2
        while fi < parts.count {
            let fd = parts[fi]
            entryFlagPosBits(fd, &flags)
            entryFlagContextBits(fd, &flags)
            entryFlagVariantBits(fd, &flags)
            entryFlagSets(fd, w, ph)
            fi += 1
        }
        phonemeFieldAsFlag(w, ph, &flags)
        return flags
    }

    func dialectConditionApplies(_ line: inout [UInt8],
                                 _ isEnUs: Bool,
                                 _ dialectCond: inout Int) -> Bool {
        let d = parseLeadingDialect(line, line.count)
        dialectCond = d.cond
        line = d.rest
        var applies = true
        if dialectCond != 0 {
            let match = (dialectCond == 3 || dialectCond == 6)
                     && isEnUs
            applies = d.negated ? !match : match
        }
        return applies
    }

    func storeDictionaryLine(_ line: [UInt8], _ dialectCond: Int) {
        let parts = splitWS(line)
        if parts.count >= 2 {
            let norm = toLower(parts[0])
            let flags = parseEntryFlags(parts, norm, parts[1])
            storeDictionaryEntry(norm, parts[1], dialectCond, flags)
        }
    }

    func loadDictionaryLine(_ line0: [UInt8], _ isEnUs: Bool) {
        var line = stripCommentAndTrim(line0, line0.count)
        var live = line.count > 0
        if live && parsePhraseEntry(line) { live = false }
        var dialectCond = 0
        if live {
            live = dialectConditionApplies(&line, isEnUs,
                                           &dialectCond)
        }
        if live { storeDictionaryLine(line, dialectCond) }
    }

    func loadDictionary(_ path: String) -> Bool {
        var ok = true
        if let raw = FileManager.default.contents(atPath: path) {
            let isEnUs = dialect == bs("en-us") || dialect == bs("en_us")
            for var line0 in getlines([UInt8](raw)) {
                var n = line0.count
                if n > 0 && line0[n - 1] == asc("\n") { n -= 1 }
                if n > 0 && line0[n - 1] == asc("\r") { n -= 1 }
                line0 = pre(line0, n)
                loadDictionaryLine(line0, isEnUs)
            }
            sortCompoundPrefixesLikeQsort(&compoundPrefixes)
            // "made" is a content word the reference still stresses in
            // sentence context, so its $u+ entry is dropped after load.
            unstressedWords.remove(bs("made"))
        } else {
            err = "Cannot open dictionary file: \(path)"
            ok = false
        }
        return ok
    }

    func parseLgroupDef(_ line: [UInt8], _ n: Int) {
        if n >= 3 && line[1] == asc("L") {
            var id = 0
            var i = 2
            while i < n && isDigitC(line[i]) {
                id = id * 10 + Int(line[i] - asc("0"))
                i += 1
            }
            if id > 0 && id < 100 {
                let items = splitWS(line, i, n - i)
                var taken = items.count
                for k in 0..<items.count {
                    let s = items[k]
                    if taken == items.count && s.count >= 2
                        && s[0] == asc("/") && s[1] == asc("/") {
                        taken = k
                    }
                }
                for k in 0..<taken {
                    rules.groups.lgroups[id].append(items[k])
                }
            }
        }
    }

    // 'P' is the prefix marker when followed by a digit / end / '_' / '+'
    // / '<'. A 'P' after 'L' is an L-group reference instead.
    func detectPrefixMarker(_ rule: inout PhonemeRule) {
        let rc = rule.rightCtx
        var k = 0
        while k < rc.count && !rule.isPrefix {
            if rc[k] == asc("P") {
                let followedByMarker = (k + 1 >= rc.count)
                    || (rc[k + 1] >= asc("1") && rc[k + 1] <= asc("9"))
                    || rc[k + 1] == asc("_") || rc[k + 1] == asc("+")
                    || rc[k + 1] == asc("<")
                let precededByL = k > 0 && rc[k - 1] == asc("L")
                if followedByMarker && !precededByL {
                    rule.isPrefix = true
                }
            }
            k += 1
        }
    }

    func suffixFlagBit(_ fc: UInt8) -> Int {
        var bit = 0
        if fc == asc("i") { bit = sufxIBit }
        else if fc == asc("m") { bit = sufxMBit }
        else if fc == asc("v") { bit = sufxVBit }
        else if fc == asc("e") { bit = sufxEBit }
        else if fc == asc("d") { bit = sufxDBit }
        else if fc == asc("q") { bit = sufxQBit }
        else if fc == asc("p") { bit = sufxPBit }
        return bit
    }

    struct SuffixMarker {
        var stripLen = 0
        var flags = 0
    }

    func parseSuffixMarker(_ rc: [UInt8], _ k: Int) -> SuffixMarker {
        var r = SuffixMarker()
        var k2 = k + 1
        while k2 < rc.count && isDigitC(rc[k2]) {
            r.stripLen = r.stripLen * 10 + Int(rc[k2] - asc("0"))
            k2 += 1
        }
        while k2 < rc.count && isAlphaC(rc[k2]) {
            r.flags |= suffixFlagBit(rc[k2])
            k2 += 1
        }
        return r
    }

    func detectSuffixMarker(_ rule: inout PhonemeRule) {
        let rc = rule.rightCtx
        var k = 0
        while k < rc.count && !rule.isSuffix {
            if rc[k] == asc("S") && (k == 0 || rc[k - 1] != asc("L")) {
                let m = parseSuffixMarker(rc, k)
                if m.stripLen > 0 {
                    rule.isSuffix = true
                    rule.suffixStripLen = m.stripLen
                    rule.suffixFlags = m.flags
                }
            }
            k += 1
        }
    }

    // ".LNN" / ".replace" / ".group X" directives.
    func parseRulesDirective(_ line: [UInt8],
                             _ currentGroup: inout [UInt8],
                             _ inReplaceSection: inout Bool) -> Bool {
        let consumed: Bool
        if line[0] != asc(".") {
            consumed = false
        } else if line.count >= 2 && line[1] == asc("L") {
            parseLgroupDef(line, line.count)
            consumed = true
        } else if line == bs(".replace") {
            inReplaceSection = true
            currentGroup = []
            consumed = true
        } else if line.count >= 6 && equalRange(line, 0, bs(".group"), 6) {
            inReplaceSection = false
            currentGroup = trim(line, 6, line.count - 6)
            consumed = true
        } else {
            consumed = false
        }
        return consumed
    }

    // A ".replace" body line: "from to", extra fields ignored.
    func parseReplaceLine(_ line: [UInt8]) {
        let parts = splitWS(line)
        if parts.count >= 2 {
            var rr = ReplaceRule()
            rr.from = parts[0]
            rr.to = parts[1]
            rules.replacements.append(rr)
        }
    }

    // "left) match (right phonemes", each part optional. An absent match
    // string means the rule matches the group letter itself.
    func buildRuleFields(_ tokens: [[UInt8]], _ currentGroup: [UInt8],
                         _ rule: inout PhonemeRule) {
        var ti = 0
        if ti < tokens.count && tokens[ti].count > 0
            && tokens[ti][tokens[ti].count - 1] == asc(")") {
            rule.leftCtx = pre(tokens[ti], tokens[ti].count - 1)
            ti += 1
        }
        if ti < tokens.count && tokens[ti].count > 0
            && tokens[ti][0] != asc("(") {
            rule.match = tokens[ti]
            ti += 1
        } else {
            rule.match = currentGroup
        }
        if ti < tokens.count && tokens[ti].count > 0
            && tokens[ti][0] == asc("(") {
            rule.rightCtx = Array(tokens[ti][1...])
            ti += 1
        }
        var j = ti
        while j < tokens.count {
            rule.phonemes.append(contentsOf: tokens[j])
            j += 1
        }
    }

    // A bare-flag rule, or one with nothing to match on, is dropped.
    func storeRuleInGroup(_ currentGroup: [UInt8],
                          _ rule: inout PhonemeRule) {
        let flagOnly = rule.phonemes.count > 0
            && rule.phonemes[0] == asc("$")
        let matchEmpty = rule.match.count == 0
        if !flagOnly && !matchEmpty {
            detectPrefixMarker(&rule)
            detectSuffixMarker(&rule)
            rules.ruleGroups[currentGroup, default: []].append(rule)
        }
    }

    func parseRuleLine(_ line: [UInt8], _ currentGroup: [UInt8],
                       _ isEnUs: Bool) {
        let d = parseLeadingDialect(line, line.count)
        var applies = true
        if d.cond != 0 {
            let match = (d.cond == 3) && isEnUs
            applies = d.negated ? !match : match
        }
        if applies {
            let tokens = splitWS(d.rest)
            if tokens.count > 0 {
                var rule = PhonemeRule()
                rule.condition = d.cond
                rule.conditionNegated = d.negated
                buildRuleFields(tokens, currentGroup, &rule)
                storeRuleInGroup(currentGroup, &rule)
            }
        }
    }

    func loadRules(_ path: String) -> Bool {
        var ok = true
        if let raw = FileManager.default.contents(atPath: path) {
            let isEnUs = dialect == bs("en-us") || dialect == bs("en_us")
            var currentGroup = [UInt8]()
            var inReplaceSection = false
            for var line0 in getlines([UInt8](raw)) {
                line0 = pre(line0, lineLenWithoutEol(line0, line0.count))
                let line = stripCommentAndTrim(line0, line0.count)
                var live = line.count > 0
                if live && parseRulesDirective(line, &currentGroup,
                                               &inReplaceSection) {
                    live = false
                }
                if live && inReplaceSection {
                    parseReplaceLine(line)
                    live = false
                }
                if live && currentGroup.count == 0 { live = false }
                if live {
                    parseRuleLine(line, currentGroup, isEnUs)
                }
            }
        } else {
            err = "Cannot open rules file: \(path)"
            ok = false
        }
        return ok
    }

    let addEEndings = bsv([
        "c", "rs", "ir", "ur", "ath", "ns", "u",
        "spong", "rang", "larg",
    ])

    // The plural / 3rd-person allomorphs never restore a magic e.
    func suffixIsBareS(_ suffixPh: [UInt8]) -> Bool {
        let sn = suffixPh.count
        return (sn == 1 && suffixPh[0] == asc("s"))
            || (sn == 1 && suffixPh[0] == asc("z"))
            || (sn == 3 && suffixPh == bs("I#z"))
            || (sn == 4 && suffixPh == bs("%I#z"))
    }

    func verbDictHasStemE(_ stemNorm: [UInt8]) -> Bool {
        var stemE = stemNorm
        stemE.append(asc("e"))
        return verbDict[stemE] != nil
    }

    // A stem the dictionaries already spell keeps its own spelling.
    func stemBareInDicts(_ stemNorm: [UInt8],
                         _ sfxIsSBare: Bool) -> Bool {
        var inDict = false
        if !sfxIsSBare { inDict = verbDict[stemNorm] != nil }
        let blockedByOnlys = onlysWords.contains(stemNorm)
            || onlyWords.contains(stemNorm)
        if !inDict && !blockedByOnlys {
            inDict = dict[stemNorm] != nil
        }
        return inDict
    }

    // Endings whose dropped 'e' the consonant shape below cannot recover.
    func stemEndingWantsE(_ stem: [UInt8]) -> Bool {
        var addE = false
        var ai = 0
        while ai < addEEndings.count && !addE {
            let e = addEEndings[ai]
            if stem.count >= e.count
                && equalRange(stem, stem.count - e.count,
                              e, e.count) {
                addE = true
            }
            ai += 1
        }
        return addE
    }

    // Vowel plus hard consonant is the magic-e shape; "-ion" is not.
    func stemShapeWantsE(_ stem: [UInt8]) -> Bool {
        let vowelsInclY = bs("aeiouy")
        let hardCons = bs("bcdfgjklmnpqstvxz")
        var addE = false
        if stem.count >= 2 {
            let last = toLowerC(stem[stem.count - 1])
            let prev = toLowerC(stem[stem.count - 2])
            let lastHard = strchrHit(hardCons, last)
            let prevVowel = strchrHit(vowelsInclY, prev)
            let ionExc = stem.count >= 3
                && equalRange(stem, stem.count - 3, bs("ion"), 3)
            addE = lastHard && prevVowel && !ionExc
        }
        return addE
    }

    // SUFX_E: append 'e' so the dict / magic-e rules fire on the original verb.
    func appendMagicEIfNeeded(_ stem: inout [UInt8],
                              _ suffixPh: [UInt8],
                              _ suffixFlags: Int) {
        let entering = (suffixFlags & sufxEBit) != 0
                    && stem.count > 0
                    && stem[stem.count - 1] != asc("e")
        if entering {
            let stemNorm = toLower(stem)
            let sfxIsSBare = suffixIsBareS(suffixPh)
            if !sfxIsSBare && (suffixFlags & sufxVBit) != 0
                && verbDictHasStemE(stemNorm) {
                stem.append(asc("e"))
            }
            if stem[stem.count - 1] != asc("e")
                && !stemBareInDicts(stemNorm, sfxIsSBare)
                && (stemEndingWantsE(stem)
                    || stemShapeWantsE(stem)) {
                stem.append(asc("e"))
            }
        }
    }

    struct StemLookup {
        var stemPh = [UInt8]()
        var matchedStem = [UInt8]()
        var foundDictEntry = false
        var usedOnlysBare = false
    }

    func lookupStemInDicts(_ stemNorm: [UInt8],
                           _ stemIsOnlys: Bool, _ suffixIsS: Bool,
                           _ suffixFlags: Int) -> StemLookup {
        var r = StemLookup()
        r.matchedStem = stemNorm
        if !suffixIsS && (suffixFlags & sufxVBit) != 0 {
            if let v = verbDict[stemNorm] {
                r.stemPh = v
                r.foundDictEntry = true
            }
        }
        if !r.foundDictEntry {
            if suffixIsS {
                if let o = onlysBareDict[stemNorm] {
                    r.stemPh = o
                    r.usedOnlysBare = true
                }
            }
            if !r.usedOnlysBare {
                if let d = dict[stemNorm], !stemIsOnlys {
                    r.stemPh = d
                    r.foundDictEntry = true
                }
            }
        }
        let snn = stemNorm.count
        if !r.foundDictEntry && !r.usedOnlysBare && snn > 1
            && stemNorm[snn - 1] == asc("e") {
            let short = pre(stemNorm, snn - 1)
            let blocked = onlysWords.contains(short)
                || onlyWords.contains(short)
            if let d = dict[short], !blocked {
                r.stemPh = d
                r.matchedStem = short
                r.foundDictEntry = true
            }
        }
        return r
    }

    func applyStemStressOrRulesFallback(_ stem: [UInt8],
                                        _ stemNorm: [UInt8],
                                        _ lookup: StemLookup,
                                        _ wordAltFlagsIn: Int,
                                        _ match: RuleMatch) -> [UInt8] {
        var out = [UInt8]()
        if lookup.usedOnlysBare || lookup.foundDictEntry {
            let key = lookup.matchedStem
            out = lookup.stemPh
            if let sp = stressPos[key] {
                out = applyStressPosition(out, out.count, sp)
            }
        } else {
            // The stem's own $altN combines with the parent word's; a
            // stem+"e" lookup carries "fertil" -> "fertile" inheritance.
            var stemOwn = wordAltFlags[stemNorm]
            if stemOwn == nil {
                var tryE = stemNorm
                tryE.append(asc("e"))
                stemOwn = wordAltFlags[tryE]
            }
            let stemAlt = stemOwn ?? 0
            let combined = stemAlt | wordAltFlagsIn
            out = applyRules(stem, true, combined, false, true).out
            let sp2 = stressPos[stemNorm]
            let nounForm = nounFormStress.contains(stemNorm)
            let verbFlag = verbFlagWords.contains(stemNorm)
            let applyStress = sp2 != nil
                && (!nounForm || (match.suffixFlags & sufxVBit) == 0)
                && !verbFlag
            if applyStress {
                out = applyStressPosition(out, out.count, sp2!)
            }
        }
        return out
    }

    func stemPhonemeFromDict(_ stem: [UInt8],
                             _ match: RuleMatch,
                             _ wordAltFlagsIn: Int) -> [UInt8] {
        var out = [UInt8]()
        if stem.count > 0 {
            let stemNorm = toLower(stem)
            var stemIsOnlys = onlysWords.contains(stemNorm)
            let mp = match.phonemes
            let suffixIsS =
                (mp.count == 1 && (mp[0] == asc("s") || mp[0] == asc("z")))
                || (mp.count == 3 && mp == bs("I#z"))
                || (mp.count == 4 && mp == bs("%I#z"))
            if suffixIsS { stemIsOnlys = false }
            if onlyWords.contains(stemNorm) && !suffixIsS {
                stemIsOnlys = true
            }
            let lookup = lookupStemInDicts(stemNorm, stemIsOnlys,
                                           suffixIsS, match.suffixFlags)
            out = applyStemStressOrRulesFallback(stem, stemNorm, lookup,
                                                 wordAltFlagsIn, match)
        }
        return out
    }

    func processSuffixRule(_ word: [UInt8], _ wordAltFlagsIn: Int,
                           _ match: RuleMatch) -> [UInt8] {
        let wn = word.count
        var strip = match.suffixStripLen
        if strip <= 0 || strip > wn { strip = match.advance }
        let stemN = wn - strip
        var stem = pre(word, stemN)
        if (match.suffixFlags & sufxIBit) != 0 && stem.count > 0
            && stem[stem.count - 1] == asc("i") {
            stem[stem.count - 1] = asc("y")
        }
        appendMagicEIfNeeded(&stem, match.phonemes, match.suffixFlags)
        let stemPh = stemPhonemeFromDict(stem, match, wordAltFlagsIn)
        var suffixPh = match.phonemes
        devoiceEdSuffix(stemPh, &suffixPh)
        var out = stemPh
        out.append(contentsOf: suffixPh)
        return out
    }

    // Dict-lookup arms: each returns non-empty iff it claimed the word.
    func checkCapitalDict(_ word: [UInt8], _ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let isCapital = word.count > 0 && word[0] >= asc("A")
                     && word[0] <= asc("Z")
        if isCapital {
            if let v = capitalDict[norm] {
                out = processPhonemeString(v, false)
            }
        }
        return out
    }

    func checkMainDict(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        var raw = onlysBareDict[norm]
        if raw == nil { raw = dict[norm] }
        if let r = raw {
            var temp = r
            if let sp = stressPos[norm] {
                temp = applyStressPosition(temp, temp.count, sp)
            }
            let isStrend = strendWords.contains(norm)
            out = processPhonemeString(temp, isStrend)
        }
        return out
    }

    func segmentHasLetter(_ norm: [UInt8], _ segStart: Int,
                          _ segLen: Int) -> Bool {
        var hasLetter = false
        var k = 0
        while k < segLen && !hasLetter {
            if isAlphaC(norm[segStart + k]) { hasLetter = true }
            k += 1
        }
        return hasLetter
    }

    // A segment that phonemizes to nothing aborts the whole word.
    func appendHyphenSegment(_ norm: [UInt8], _ segStart: Int,
                             _ segEnd: Int,
                             _ accum: inout [UInt8]) -> Bool {
        let segIpa = wordToPhonemes(Array(norm[segStart..<segEnd]))
        let got = segIpa.count > 0
        if got { accum.append(contentsOf: segIpa) }
        return got
    }

    func checkHyphenated(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        let firstHyphen = firstIndexOf(norm, asc("-"))
        let hasHyphen = firstHyphen > 0 && firstHyphen + 1 < nn
        if hasHyphen {
            var accum = [UInt8]()
            var segStart = 0
            var processing = true
            while segStart < nn && processing {
                let nextH = firstIndexOf(norm, asc("-"), from: segStart)
                let segEnd = nextH >= 0 ? nextH : nn
                let segLen = segEnd - segStart
                processing = segLen > 0
                    && segmentHasLetter(norm, segStart, segLen)
                    && appendHyphenSegment(norm, segStart, segEnd,
                                           &accum)
                if processing {
                    segStart = nextH >= 0 ? nextH + 1 : nn
                }
            }
            if processing && accum.count > 0 { out = accum }
        }
        return out
    }

    func possessiveBaseCodes(_ base: [UInt8]) -> [UInt8] {
        var rawCode = [UInt8]()
        if let d = dict[base] {
            rawCode = processPhonemeString(d, false)
        } else {
            let rulesOut = applyRules(base, true, -1, false, false).out
            rawCode = processPhonemeString(rulesOut, false)
        }
        return rawCode
    }

    // tS, dZ, s, z, S, Z.
    func isSibilantCode(_ ph: [UInt8]) -> Bool {
        ph == bs("tS") || ph == bs("dZ")
            || (ph.count == 1 && (ph[0] == asc("s") || ph[0] == asc("z")
                                  || ph[0] == asc("S")
                                  || ph[0] == asc("Z")))
    }

    func possessiveAfterCh(_ lastPh: [UInt8]) -> [UInt8] {
        let unvoiced = bs("ptkfsTSx")
        var poss = bs("z")
        if isSibilantCode(lastPh) {
            poss = bs("I2z")
        } else if lastPh.count > 0 && strchrHit(unvoiced, lastPh[0]) {
            poss = bs("s")
        }
        return poss
    }

    func possessiveByLastLetter(_ lc: UInt8) -> [UInt8] {
        var poss = bs("z")
        if lc == asc("s") || lc == asc("z") || lc == asc("x") {
            poss = bs("I#z")
        } else if lc == asc("f") || lc == asc("p") || lc == asc("t")
                  || lc == asc("k") {
            poss = bs("s")
        }
        return poss
    }

    // The spelling of the base decides first and its last phoneme second.
    func possessiveAllomorph(_ base: [UInt8],
                             _ lastPh: [UInt8]) -> [UInt8] {
        let baseN = base.count
        var poss = bs("z")
        let endsOch = baseN >= 3
            && equalRange(base, baseN - 3, bs("och"), 3)
        let endsCh = baseN >= 2
            && equalRange(base, baseN - 2, bs("ch"), 2)
        let endsSe = baseN >= 2
            && equalRange(base, baseN - 2, bs("se"), 2)
        let endsCe = baseN >= 2
            && equalRange(base, baseN - 2, bs("ce"), 2)
        let endsSh = baseN >= 2
            && equalRange(base, baseN - 2, bs("sh"), 2)
        if endsOch {
            poss = bs("s")
        } else if endsCh {
            poss = possessiveAfterCh(lastPh)
        } else if endsSe || endsCe || endsSh {
            poss = bs("I#z")
        } else if baseN > 0 {
            poss = possessiveByLastLetter(base[baseN - 1])
        }
        return poss
    }

    func checkPossessive(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        let isPossessive = nn >= 3 && norm[nn - 2] == asc("'")
                        && norm[nn - 1] == asc("s")
        if isPossessive {
            let base = pre(norm, nn - 2)
            let baseIpa = wordToPhonemes(base)
            if baseIpa.count > 0 {
                let rawCode = possessiveBaseCodes(base)
                let lastPh = lastPhonemeCode(rawCode)
                var combined = rawCode
                combined.append(contentsOf:
                    possessiveAllomorph(base, lastPh))
                out = processPhonemeString(combined, false)
            }
        }
        return out
    }

    func checkSingleLetter(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let isSingle = norm.count == 1 && norm[0] != asc("a")
        if isSingle {
            let key: [UInt8] = [asc("_"), norm[0]]
            if let v = dict[key] {
                out = processPhonemeString(v, false)
            }
        }
        return out
    }

    let mcVowels = bsv([
        "O@", "o@", "U@", "A@", "e@", "i@", "aI@3", "aI3",
        "aU@", "aI@", "i@3", "3:r", "A:r", "o@r", "A@r", "e@r",
        "eI", "aI", "aU", "OI", "oU", "IR", "VR", "3:", "A:",
        "i:", "u:", "O:", "e:", "a:", "aa", "@L", "@2", "@5",
        "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02",
        "O2", "A#",
    ])

    // A prefix needs 4+ letters and a suffix of 2+ with a vowel, the suffix
    // either 4 letters long or a dictionary word in its own right.
    func compoundPrefixFits(_ norm: [UInt8], _ pref: [UInt8]) -> Bool {
        let nn = norm.count
        var live = pref.count >= 4 && pref.count < nn
        if live {
            let sfxLen = nn - pref.count
            live = sfxLen >= 2 && equalRange(norm, 0, pref, pref.count)
                && hasAnyVowelLetter(norm, pref.count, sfxLen)
            if live && sfxLen < 4 {
                let sfx = Array(norm[pref.count...])
                live = dict[sfx] != nil || verbDict[sfx] != nil
            }
        }
        return live
    }

    func countPrefixPhonemeVowels(_ pfxPh: [UInt8]) -> Int {
        var nvowels = 0
        var pi = 0
        while pi < pfxPh.count {
            let c = pfxPh[pi]
            if c == asc("'") || c == asc(",") || c == asc("%")
                || c == asc("=") {
                pi += 1
            } else {
                var matched = false
                var mi = 0
                while mi < mcVowels.count && !matched {
                    let mv = mcVowels[mi]
                    if matchAt(pfxPh, pi, mv) {
                        nvowels += 1
                        pi += mv.count
                        matched = true
                    }
                    mi += 1
                }
                if !matched {
                    if isVowelCode(pfxPh, pi, 1) { nvowels += 1 }
                    pi += 1
                }
            }
        }
        return nvowels
    }

    // A two-syllable prefix keeps a secondary; a one-syllable one loses stress.
    func demotePrefixStress(_ pfxPh: inout [UInt8], _ nvowels: Int) {
        if nvowels >= 2 {
            replaceFirstChar(&pfxPh, asc("'"), asc(","))
        } else {
            var stripped = [UInt8]()
            for r in 0..<pfxPh.count
            where pfxPh[r] != asc("'") && pfxPh[r] != asc(",") {
                stripped.append(pfxPh[r])
            }
            pfxPh = stripped
        }
    }

    // A "$N" entry for the whole compound overrides the joined stress.
    func applyStressPosOverride(_ norm: [UInt8],
                                _ combined: inout [UInt8]) {
        if let sp = stressPos[norm] {
            let stressed = applyStressPosition(combined,
                                               combined.count, sp)
            combined = processPhonemeString(stressed, false)
        }
    }

    func checkCompoundPrefixes(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 5 {
            var found = false
            var i = 0
            while i < compoundPrefixes.count && !found {
                let cp = compoundPrefixes[i]
                if compoundPrefixFits(norm, cp.a) {
                    var pfxPh = processPhonemeString(cp.b, false)
                    demotePrefixStress(&pfxPh,
                        countPrefixPhonemeVowels(pfxPh))
                    let sfxPh = wordToPhonemes(
                        Array(norm[cp.a.count...]))
                    var combined = pfxPh
                    combined.append(contentsOf: sfxPh)
                    applyStressPosOverride(norm, &combined)
                    out = combined
                    found = true
                }
                i += 1
            }
        }
        return out
    }

    func applyRulesFallback(_ norm: [UInt8]) -> [UInt8] {
        var rawPh = applyRules(norm, true, -1, false, false).out
        if let sp = stressPos[norm] {
            rawPh = applyStressPosition(rawPh, rawPh.count, sp)
        }
        return processPhonemeString(rawPh, false)
    }

    // Morphological-suffix arms: each writes non-empty iff it claimed the word.

    // Prefers the plain dictionary over the verb one, skipping "only" entries.
    func stemEDictPhonemes(_ stem: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        var sx = stem
        sx.append(asc("e"))
        let je = dict[sx]
        let jeOnlys = je != nil && (onlysWords.contains(sx)
                                    || onlyWords.contains(sx))
        if je != nil && !jeOnlys {
            out = je!
        } else if let ve = verbDict[sx] {
            out = ve
        }
        return out
    }

    // A noun-form or verb-flagged stem keeps the stress the rules
    // gave it; anything else takes the stress-table position.
    func stemRulesPhonemes(_ stem: [UInt8]) -> [UInt8] {
        let stemAltFlags = verbFlagWords.contains(stem) ? 1 : -1
        var raw = applyRules(stem, true, stemAltFlags,
                             false, false).out
        if let sp = stressPos[stem],
            !nounFormStress.contains(stem),
            !verbFlagWords.contains(stem) {
            raw = applyStressPosition(raw, raw.count, sp)
        }
        return processPhonemeString(raw, false)
    }

    func getStemPhonemes(_ stem: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let sn = stem.count
        if sn >= 2 && hasAnyVowelLetter(stem, 0, sn) {
            if let vt = verbDict[stem] {
                out = vt
            } else {
                let isOnlys = onlysWords.contains(stem)
                           || onlyWords.contains(stem)
                let jt = dict[stem]
                if jt == nil || isOnlys {
                    out = stemEDictPhonemes(stem)
                }
                if out.count == 0 && jt != nil && !isOnlys {
                    out = jt!
                }
                if out.count == 0 { out = stemRulesPhonemes(stem) }
            }
            if out.count > 0 && !hasAnyVowelCode(out) { out = [] }
        }
        return out
    }

    // -ns/-rs stems prefer magic-e because the bare-stem 's' fires as 'z'.
    func tryIngNonMagicEStemFallbacks(_ base: [UInt8],
                                      _ baseHasStressOverride: Bool,
                                      _ sph: inout [UInt8]) {
        let bn = base.count
        let tryMagicEFirst = !baseHasStressOverride && bn >= 2
            && (equalRange(base, bn - 2, bs("ns"), 2)
                || equalRange(base, bn - 2, bs("rs"), 2))
        if tryMagicEFirst {
            var sx = base
            sx.append(asc("e"))
            sph = getStemPhonemes(sx)
        }
        if sph.count == 0 { sph = getStemPhonemes(base) }
        if sph.count == 0 && bn > 0 && !isVowelLetter(base[bn - 1])
            && hasAnyVowelLetter(base, 0, bn) {
            var sx = base
            sx.append(asc("e"))
            sph = getStemPhonemes(sx)
        }
    }

    func stemPhonemesWithE(_ base: [UInt8]) -> [UInt8] {
        var sx = base
        sx.append(asc("e"))
        return getStemPhonemes(sx)
    }

    func countVowelGroups(_ s: [UInt8], _ n: Int) -> Int {
        var vowelGroups = 0
        var inV = false
        for i in 0..<n {
            if isVowelLetter(s[i]) {
                if !inV { vowelGroups += 1; inV = true }
            } else {
                inV = false
            }
        }
        return vowelGroups
    }

    // A doubled final consonant over a two-syllable prefix undoubles
    // ("controlled").
    func prefersUndoubledStem(_ base: [UInt8],
                              _ baseHasDouble: Bool) -> Bool {
        return baseHasDouble
            && countVowelGroups(base, base.count - 2) >= 2
    }

    // "-ns" / "-rs" stems restore the magic e before anything else.
    func edNsRsWantsE(_ base: [UInt8],
                      _ baseHasStressOverride: Bool) -> Bool {
        let bn = base.count
        return !baseHasStressOverride && bn >= 2
            && (equalRange(base, bn - 2, bs("ns"), 2)
                || equalRange(base, bn - 2, bs("rs"), 2))
    }

    // A consonant-final stem with a vowel somewhere gets one last try with
    // magic e.
    func edConsonantFinalMagicE(_ base: [UInt8],
                                _ sph: inout [UInt8]) {
        let bn = base.count
        if sph.count == 0 && bn > 0 && !isVowelLetter(base[bn - 1])
            && hasAnyVowelLetter(base, 0, bn) {
            sph = stemPhonemesWithE(base)
        }
    }

    func tryEdNonMagicEStemFallbacks(_ base: [UInt8],
                                     _ baseHasStressOverride: Bool,
                                     _ sph: inout [UInt8]) {
        let bn = base.count
        let cvcDoubleCons = bs("lptmnrgdb")
        let baseHasDouble = bn >= 2 && base[bn - 1] == base[bn - 2]
            && strchrHit(cvcDoubleCons, base[bn - 1])
        let preferUndoubled = prefersUndoubledStem(base,
                                                   baseHasDouble)
        if sph.count == 0 && baseHasDouble && preferUndoubled {
            sph = getStemPhonemes(pre(base, bn - 1))
        }
        let tryEFirst = sph.count == 0
            && edNsRsWantsE(base, baseHasStressOverride)
        if tryEFirst { sph = stemPhonemesWithE(base) }
        if sph.count == 0 { sph = getStemPhonemes(base) }
        if sph.count == 0 && baseHasDouble && !preferUndoubled {
            sph = getStemPhonemes(pre(base, bn - 1))
        }
        edConsonantFinalMagicE(base, &sph)
    }

    func stemInDicts(_ base: [UInt8]) -> Bool {
        dict[base] != nil || verbDict[base] != nil
    }

    // Soft c/g before "-ed" needs a dictionary entry to count as a
    // regular suffix; "-nged" is exempt.
    func edSoftCGBlocked(_ norm: [UInt8], _ base: [UInt8]) -> Bool {
        let nn = norm.count
        var blocked = false
        if nn >= 4 {
            let penult = norm[nn - 3]
            let isSoftC = penult == asc("c") || penult == asc("g")
            let isNgGed = penult == asc("g") && nn >= 6
                       && norm[nn - 4] == asc("n")
            blocked = isSoftC && !isNgGed && !stemInDicts(base)
        }
        return blocked
    }

    func edEndingNeedsDict(_ norm: [UInt8], _ base: [UInt8],
                           _ ending: [UInt8]) -> Bool {
        let nn = norm.count
        return nn >= 5 && equalRange(norm, nn - 4, ending, 4)
            && !stemInDicts(base)
    }

    // "-mented" on an unstressed consonant-final stem is a morpheme, not -ed.
    func edMentedBlocked(_ norm: [UInt8]) -> Bool {
        let nn = norm.count
        var blocked = false
        if nn >= 7 && equalRange(norm, nn - 6, bs("mented"), 6) {
            let beforeM = norm[nn - 7]
            let stemHasStress = stressPos[pre(norm, nn - 2)] != nil
            blocked = !isVowelLetter(beforeM) && !stemHasStress
        }
        return blocked
    }

    func trailingConsonants(_ base: [UInt8]) -> Int {
        var trailCons = 0
        var bi = base.count - 1
        while bi >= 0 && !isVowelLetter(base[bi]) {
            trailCons += 1
            bi -= 1
        }
        return trailCons
    }

    // Pre-flight: silent-e or "u" stem, soft-c/g or "nged"/"eted" without dict,
    // "mented" without prefix or stress, or 3+ trailing consonants.
    func isEdSuffixCandidate(_ norm: [UInt8], _ base: [UInt8]) -> Bool {
        let bn = base.count
        let blocked = (bn > 0 && base[bn - 1] == asc("e"))
            || (bn > 0 && base[bn - 1] == asc("u"))
            || edSoftCGBlocked(norm, base)
            || edEndingNeedsDict(norm, base, bs("nged"))
            || edEndingNeedsDict(norm, base, bs("eted"))
            || edMentedBlocked(norm)
            || trailingConsonants(base) >= 3
        return !blocked
    }

    // t/d -> I#d, unvoiced -> t, else d. When the regular choice is 't' but a
    // full-word rule fires a 'd' ("tied"), the full-word phonemes win.
    func computeEdSuffixVoicing(_ sph: [UInt8],
                                _ norm: [UInt8]) -> [UInt8] {
        let unvoiced = bs("ptkfTSshx")
        let sn = sph.count
        let nn = norm.count
        let last = sph[sn - 1]
        var edPh = bs("d")
        if last == asc("t") || last == asc("d") {
            edPh = bs("I#d")
        } else if strchrHit(unvoiced, last) {
            edPh = bs("t")
        }
        var out = sph
        out.append(contentsOf: edPh)
        if edPh.count == 1 && edPh[0] == asc("t") && nn >= 2 {
            let fw = applyRules(norm, false, -1, false, false)
            var fwPv = [Bool](repeating: false, count: nn)
            let copyN = min(fw.posVisited.count, nn)
            for i in 0..<copyN { fwPv[i] = fw.posVisited[i] }
            let ePos = nn - 2
            let eWasVisited = ePos >= 0 && ePos < nn && fwPv[ePos]
            var fwLast: UInt8 = 0
            var ri = fw.out.count - 1
            while ri >= 0 && fwLast == 0 {
                if fw.out[ri] != phBnd { fwLast = fw.out[ri] }
                ri -= 1
            }
            if !eWasVisited && fwLast == asc("d") { out = fw.out }
        }
        return out
    }

    func prefixMatchInGroup(_ key: [UInt8], _ w: [UInt8],
                            _ glen: Int) -> Bool {
        var matched = false
        if let rv = rules.ruleGroups[key] {
            var i = 0
            while i < rv.count && !matched {
                let rule = rv[i]
                if rule.isPrefix {
                    let m = matchRule(rules, rule, w, w.count, 0,
                                      glen, [], 0, 0, nil, 0, false)
                    matched = m.score >= 0 && m.advance > 0
                           && m.advance < w.count
                }
                i += 1
            }
        }
        return matched
    }

    // A PREFIX rule covering > 0 and < wn characters: a real compound prefix.
    func hasPrefixAtStart(_ w: [UInt8]) -> Bool {
        var r = false
        let wn = w.count
        if wn > 0 {
            let c0 = toLowerC(w[0])
            if wn >= 2 {
                let k2 = [c0, toLowerC(w[1])]
                r = prefixMatchInGroup(k2, w, 2)
            }
            if !r { r = prefixMatchInGroup([c0], w, 1) }
        }
        return r
    }

    struct StemPattern {
        var cvc = false
        var nc = false
    }

    // A stem whose phonemes end in schwa+n or a syllabic n is not a
    // doubling CVC even though its spelling looks like one.
    func stemEndsInNasalSyllable(_ base: [UInt8]) -> Bool {
        let basePh = getStemPhonemes(base)
        var nasal = false
        if basePh.count >= 2 {
            let last2 = basePh[basePh.count - 1]
            let last1 = basePh[basePh.count - 2]
            let endsInSchwaN = last1 == asc("@")
                && (last2 == asc("n") || last2 == asc("N"))
            let endsInSyllabicN = last1 == asc("n")
                && last2 == asc("-")
            nasal = endsInSchwaN || endsInSyllabicN
        }
        return nasal
    }

    // "-rg" / "-rc" after a vowel doubles like a CVC stem.
    func isCvrcStem(_ base: [UInt8]) -> Bool {
        let bn = base.count
        return bn >= 3
            && (base[bn - 1] == asc("g") || base[bn - 1] == asc("c"))
            && base[bn - 2] == asc("r") && isVowelLetter(base[bn - 3])
    }

    // "-le", "-w" and "-er" stems never double.
    func blocksCvc(_ base: [UInt8]) -> Bool {
        let bn = base.count
        return (bn >= 2 && base[bn - 1] == asc("l")
                && base[bn - 2] == asc("e"))
            || (bn > 0 && base[bn - 1] == asc("w"))
            || (bn >= 2 && base[bn - 1] == asc("r")
                && toLowerC(base[bn - 2]) == asc("e"))
    }

    func detectStemPatternForSuffix(_ base: [UInt8],
                                    _ includeCvrc: Bool) -> StemPattern {
        var r = StemPattern()
        let bn = base.count
        var cvc = bn >= 2 && !isVowelLetter(base[bn - 1])
                  && isVowelLetter(base[bn - 2])
        if includeCvrc && !cvc && isCvrcStem(base) { cvc = true }
        if cvc && bn >= 2 && base[bn - 1] == asc("n")
            && isPlainVowel(base[bn - 2])
            && stemEndsInNasalSyllable(base) {
            cvc = false
        }
        if cvc && blocksCvc(base) { cvc = false }
        r.cvc = cvc
        r.nc = bn >= 2 && base[bn - 1] == asc("c")
            && base[bn - 2] == asc("n")
        return r
    }

    // A stem carries its own stress when the stress table or a dictionary pins
    // it; the "only" sets mark entries that do not count as such a pin.
    func stemHasStressOverride(_ base: [UInt8]) -> Bool {
        return stressPos[base] != nil
            || (dict[base] != nil && !onlysWords.contains(base)
                && !onlyWords.contains(base))
            || verbDict[base] != nil
    }

    // A "-ng" base no dictionary knows is a word like "thing", not an -ing
    // form.
    func ingBaseIsNgNonword(_ base: [UInt8]) -> Bool {
        let bn = base.count
        return bn >= 2 && equalRange(base, bn - 2, bs("ng"), 2)
            && dict[base] == nil && verbDict[base] == nil
    }

    // $strend stems take their phonemes from the rule engine, not the
    // dictionaries.
    func ingStrendStemPhonemes(_ mw: [UInt8],
                               _ sph: inout [UInt8]) {
        if strendWords.contains(mw) {
            let rulesPh = applyRules(mw, true, -1, false, false).out
            if rulesPh.count > 0 {
                sph = processPhonemeString(rulesPh, false)
            }
        }
    }

    func ingMagicEStem(_ base: [UInt8], _ pat: StemPattern,
                       _ sph: inout [UInt8]) {
        var magicEIng = base
        magicEIng.append(asc("e"))
        var useMagicE = true
        var baseOnlyPh = [UInt8]()
        if pat.cvc && !pat.nc {
            baseOnlyPh = getStemPhonemes(base)
            useMagicE = shouldUseMagicEForCVCStem(baseOnlyPh, bs("I@"))
        }
        if useMagicE {
            ingStrendStemPhonemes(magicEIng, &sph)
            if sph.count == 0 {
                sph = getStemPhonemes(magicEIng)
            }
            if sph.count == 0 {
                if baseOnlyPh.count > 0 {
                    sph = baseOnlyPh
                } else {
                    sph = getStemPhonemes(base)
                }
            }
        }
    }

    // Last resorts: undo a doubled final consonant, then an 'i' that was a 'y'.
    func ingStemSpellingFallbacks(_ base: [UInt8],
                                  _ sph: inout [UInt8]) {
        let bn = base.count
        if sph.count == 0 && bn >= 2 && base[bn - 1] == base[bn - 2] {
            sph = getStemPhonemes(pre(base, bn - 1))
        }
        if sph.count == 0 && bn > 0 && base[bn - 1] == asc("i") {
            var sx = pre(base, bn - 1)
            sx.append(asc("y"))
            sph = getStemPhonemes(sx)
        }
    }

    // Either spelling of the stem may carry the $strend flag.
    func ingStemIsStrend(_ base: [UInt8]) -> Bool {
        var strend = strendWords.contains(base)
        if !strend {
            var sx = base
            sx.append(asc("e"))
            strend = strendWords.contains(sx)
        }
        return strend
    }

    func checkSuffixIng(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        let isIng = nn >= 3 && equalRange(norm, nn - 3, bs("ing"), 3)
        if isIng {
            let bn = nn - 3
            let base = pre(norm, bn)
            var sph = [UInt8]()
            if !ingBaseIsNgNonword(base) {
                let pat = detectStemPatternForSuffix(base, true)
                let hasOverride = stemHasStressOverride(base)
                if (pat.cvc || pat.nc) && !hasOverride {
                    ingMagicEStem(base, pat, &sph)
                } else {
                    tryIngNonMagicEStemFallbacks(base, hasOverride,
                                                 &sph)
                }
                ingStemSpellingFallbacks(base, &sph)
            }
            if sph.count > 0 {
                simplifySyllabicLForIng(base, &sph)
                let stemIsStrend = ingStemIsStrend(base)
                sph.append(contentsOf: bs("%IN"))
                out = processPhonemeString(sph, stemIsStrend)
            }
        }
        return out
    }

    // A stem-final 'i' stands for the 'y' the suffix replaced.
    func edYStemPhonemes(_ base: [UInt8], _ sph: inout [UInt8]) {
        let bn = base.count
        if bn > 0 && base[bn - 1] == asc("i") && bn >= 2 {
            var sx = pre(base, bn - 1)
            sx.append(asc("y"))
            sph = getStemPhonemes(sx)
        }
    }

    // True when the magic-e spelling starts with a compound prefix: then the
    // word is not an -ed form at all ("infrared" is not "infrare" + d).
    func edMagicEStem(_ base: [UInt8], _ pat: StemPattern,
                      _ sph: inout [UInt8]) -> Bool {
        var magicE = base
        magicE.append(asc("e"))
        let blocked = hasPrefixAtStart(magicE)
        if !blocked {
            var useMagicEEd = true
            if pat.cvc && !pat.nc {
                let basePhEd = getStemPhonemes(base)
                useMagicEEd = shouldUseMagicEForCVCStem(basePhEd,
                                                        bs("I@3"))
            }
            if useMagicEEd {
                if sph.count == 0 { sph = getStemPhonemes(magicE) }
                if sph.count == 0 { sph = getStemPhonemes(base) }
            }
        }
        return blocked
    }

    // Undo a doubled final consonant, then fall back to the stem as it stands.
    func edDoubledConsonantStem(_ base: [UInt8],
                                _ sph: inout [UInt8]) {
        let bn = base.count
        if sph.count == 0 && bn >= 2
            && base[bn - 1] == base[bn - 2] {
            sph = getStemPhonemes(pre(base, bn - 1))
            if sph.count == 0 { sph = getStemPhonemes(base) }
        }
    }

    func checkSuffixEd(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        var processing = nn >= 4 && equalRange(norm, nn - 2, bs("ed"), 2)
        var base = [UInt8]()
        if processing {
            base = pre(norm, nn - 2)
            processing = isEdSuffixCandidate(norm, base)
        }
        if processing {
            var sph = [UInt8]()
            let pat = detectStemPatternForSuffix(base, false)
            let hasOverride = stemHasStressOverride(base)
            edYStemPhonemes(base, &sph)
            if (pat.cvc || pat.nc) && !hasOverride {
                processing = !edMagicEStem(base, pat, &sph)
            } else {
                tryEdNonMagicEStemFallbacks(base, hasOverride, &sph)
            }
            if processing { edDoubledConsonantStem(base, &sph) }
            if processing && sph.count > 0 {
                let finalPh = computeEdSuffixVoicing(sph, norm)
                out = processPhonemeString(finalPh, false)
            }
        }
        return out
    }

    func doStemPhS(_ stem: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let sn = stem.count
        if sn >= 2 && hasAnyVowelLetter(stem, 0, sn) {
            var ph2 = [UInt8]()
            if let obit = onlysBareDict[stem] {
                ph2 = processPhonemeString(obit, false)
            } else if onlyWords.contains(stem) {
                if let vt = verbDict[stem] {
                    ph2 = processPhonemeString(vt, false)
                }
            } else if let jt = dict[stem] {
                ph2 = processPhonemeString(jt, false)
            } else if let sp = stressPos[stem] {
                let raw = applyRules(stem, true, 0, false, false).out
                if raw.count > 0 {
                    let stressed = applyStressPosition(raw, raw.count, sp)
                    ph2 = processPhonemeString(stressed, false)
                }
            }
            if ph2.count > 0 && hasAnyVowelCode(ph2) { out = ph2 }
        }
        return out
    }

    func checkSuffixDictS(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 3 && norm[nn - 1] == asc("s")
            && !(nn >= 2 && norm[nn - 2] == asc("s")) {
            let unvoicedS = bs("ptkfTSCxXhs")
            let ssn = nn - 1
            let stemS = pre(norm, ssn)
            let skipSStrip = ssn >= 3 && stemS[ssn - 1] == asc("u")
            if !skipSStrip {
                var sphS = doStemPhS(stemS)
                if sphS.count > 0 {
                    let sibilantsPh = bs("SZsz")
                    let last = sphS[sphS.count - 1]
                    let lastSibilant = strchrHit(sibilantsPh, last)
                    var sPh = bs("z")
                    if lastSibilant {
                        sPh = bs("I#z")
                    } else if strchrHit(unvoicedS, last) {
                        sPh = bs("s")
                    }
                    sphS.append(contentsOf: sPh)
                    out = processPhonemeString(sphS, false)
                }
            }
            if out.count == 0 && nn >= 4 && norm[nn - 2] == asc("e") {
                let esn = nn - 2
                var sphEs = doStemPhS(pre(norm, esn))
                if sphEs.count > 0 {
                    let sibilantsEs = bs("SZszC")
                    let last = sphEs[sphEs.count - 1]
                    if strchrHit(sibilantsEs, last) {
                        sphEs.append(contentsOf: bs("I#z"))
                        out = processPhonemeString(sphEs, false)
                    }
                }
            }
        }
        return out
    }

    let vowelsIes = bs("aAeEIiOUVu03@o")

    func firstIesVowelPos(_ s: [UInt8]) -> Int {
        var at = -1
        var k = 0
        while k < s.count && at < 0 {
            if strchrHit(vowelsIes, s[k]) { at = k }
            k += 1
        }
        return at
    }

    // The whole word through the rules wins when its first vowel
    // disagrees with the stem's ("species", not "specy" + z).
    func iesDirectRulesOverride(_ norm: [UInt8], _ sv: UInt8,
                                _ out: inout [UInt8]) -> Bool {
        let fullRaw = applyRules(norm, false, 0, false, false).out
        let dvPos = firstIesVowelPos(fullRaw)
        let dv: UInt8 = dvPos >= 0 ? fullRaw[dvPos] : 0
        let directFvUnstressed = dvPos > 0
            && fullRaw[dvPos - 1] == asc("%")
        let magicEStrut = sv == asc("V")
            && (dv == asc("u") || dv == asc("U"))
        let emitted = dv != 0 && sv != dv && !directFvUnstressed
            && !magicEStrut
        if emitted { out = processPhonemeString(fullRaw, false) }
        return emitted
    }

    func checkSuffixIes(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 4 && equalRange(norm, nn - 3, bs("ies"), 3) {
            var base = pre(norm, nn - 3)
            base.append(asc("y"))
            var sph = getStemPhonemes(base)
            if sph.count > 0 {
                let svPos = firstIesVowelPos(sph)
                let stemHasDiphthong = svPos >= 0 && svPos + 1 < sph.count
                    && (sph[svPos + 1] == asc("I")
                        || sph[svPos + 1] == asc("U"))
                let sv: UInt8 = svPos >= 0 ? sph[svPos] : 0
                let emittedDirect = !stemHasDiphthong
                    && iesDirectRulesOverride(norm, sv, &out)
                if !emittedDirect {
                    sph.append(asc("z"))
                    out = processPhonemeString(sph, false)
                }
            }
        }
        return out
    }

    func applyStressPosIfSet(_ base: [UInt8],
                             _ ph: [UInt8]) -> [UInt8] {
        var r = ph
        if let sp = stressPos[base] {
            r = applyStressPosition(ph, ph.count, sp)
        }
        return r
    }

    // Consonant + 'e', where the consonant is neither a sibilant nor
    // the second half of a "ch" / "sh" digraph.
    func magicEsBaseIsCandidate(_ base: [UInt8], _ bn: Int) -> Bool {
        let cBeforeE = base[bn - 2]
        let isSibilant = cBeforeE == asc("s") || cBeforeE == asc("z")
            || cBeforeE == asc("x") || cBeforeE == asc("c")
        let isDigraphSibilant = cBeforeE == asc("h") && bn >= 3
            && (base[bn - 3] == asc("c")
                || base[bn - 3] == asc("s"))
        return bn >= 2 && base[bn - 1] == asc("e")
            && !isVowelLetter(cBeforeE)
            && !isSibilant && !isDigraphSibilant
    }

    // A group-B consonant after a, i, o or y takes the suffix-
    // phoneme-only path through the rules.
    func magicEsWantsSuffixPhonemeOnly(_ base: [UInt8]) -> Bool {
        let groupBChars = bs("bcdfgjklmnpqstvxz")
        let delfwdVowels = bs("aioy")
        let bn = base.count
        var useSpo = false
        if bn >= 3 && base[bn - 1] == asc("e") {
            let cCons = toLowerC(base[bn - 2])
            let cPrev = toLowerC(base[bn - 3])
            useSpo = strchrHit(groupBChars, cCons)
                && strchrHit(delfwdVowels, cPrev)
        }
        return useSpo
    }

    func magicEsStemPhonemes(_ base: [UInt8]) -> [UInt8] {
        var ph2 = [UInt8]()
        if let jt = dict[base] {
            ph2 = applyStressPosIfSet(base, jt)
        } else {
            let useSpo = magicEsWantsSuffixPhonemeOnly(base)
            let raw = applyRules(base, true, -1, useSpo, false).out
            ph2 = processPhonemeString(applyStressPosIfSet(base, raw),
                                       false)
        }
        return ph2
    }

    // The plural allomorph: /I#z/ after a sibilant, /s/ after unvoiced, /z/
    // else.
    func appendSAllomorph(_ sph: inout [UInt8]) {
        let unvoiced = bs("ptkfTSCxXhs")
        let sibilantsPh = bs("SZsz")
        let last = sph[sph.count - 1]
        var sPh = bs("z")
        if strchrHit(sibilantsPh, last) {
            sPh = bs("I#z")
        } else if strchrHit(unvoiced, last) {
            sPh = bs("s")
        }
        sph.append(contentsOf: sPh)
    }

    func checkSuffixMagicEs(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        let esShape = nn >= 4 && norm[nn - 1] == asc("s")
            && !(nn >= 2 && norm[nn - 2] == asc("s"))
        if esShape && magicEsBaseIsCandidate(norm, nn - 1) {
            let bn = nn - 1
            let base = pre(norm, bn)
            var sph = [UInt8]()
            if hasAnyVowelLetter(base, 0, bn) {
                let ph2 = magicEsStemPhonemes(base)
                if ph2.count > 0 && hasAnyVowelCode(ph2) {
                    sph = ph2
                }
            }
            if sph.count > 0 {
                appendSAllomorph(&sph)
                out = processPhonemeString(sph, false)
            }
        }
        return out
    }

    func checkSuffixChShEs(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 5 && norm[nn - 1] == asc("s") && norm[nn - 2] == asc("e")
            && norm[nn - 3] == asc("h")
            && (norm[nn - 4] == asc("c") || norm[nn - 4] == asc("s")) {
            let sn2 = nn - 2
            let stem = pre(norm, sn2)
            if hasAnyVowelLetter(stem, 0, sn2) && sn2 >= 2 {
                var sph = [UInt8]()
                if let jt = dict[stem] {
                    sph = jt
                } else {
                    let raw = applyRules(stem, true, -1, false, false).out
                    sph = processPhonemeString(raw, false)
                }
                if hasAnyVowelCode(sph) {
                    sph.append(contentsOf: bs("I#z"))
                    out = processPhonemeString(sph, false)
                }
            }
        }
        return out
    }

    func checkSuffixXes(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 4 && equalRange(norm, nn - 3, bs("xes"), 3) {
            let sn2 = nn - 2
            let stem = pre(norm, sn2)
            if hasAnyVowelLetter(stem, 0, sn2) && sn2 >= 2 {
                var sph = [UInt8]()
                if let jt = dict[stem] {
                    sph = jt
                } else {
                    let raw = applyRules(stem, true, -1, false, false).out
                    sph = processPhonemeString(raw, false)
                }
                if hasAnyVowelCode(sph) {
                    sph.append(contentsOf: bs("I#z"))
                    out = processPhonemeString(sph, false)
                }
            }
        }
        return out
    }

    func checkSuffixArily(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let nn = norm.count
        if nn >= 8 && equalRange(norm, nn - 5, bs("arily"), 5) {
            let stemArilyN = nn - 5
            let stemArily = pre(norm, stemArilyN)
            if hasAnyVowelLetter(stemArily, 0, stemArilyN)
                && stemArilyN >= 2 {
                var stemWithAri = stemArily
                stemWithAri.append(contentsOf: bs("ari"))
                var sphArily = [UInt8]()
                if let jtAr = dict[stemWithAri] {
                    sphArily = jtAr
                } else {
                    sphArily = applyRules(stemWithAri, true, -1,
                                          false, false).out
                }
                if sphArily.count > 0 {
                    var sphStem = [UInt8]()
                    let an = sphArily.count
                    let endsInStressed = an >= 5
                        && equalRange(sphArily, an - 5, bs("'A@ri"), 5)
                    let endsInSchwaR = !endsInStressed && an >= 3
                        && (equalRange(sphArily, an - 3, bs("3ri"), 3)
                            || equalRange(sphArily, an - 3, bs("@ri"), 3))
                    if endsInStressed {
                        sphStem = pre(sphArily, an - 5)
                    } else if endsInSchwaR {
                        sphStem = pre(sphArily, an - 3)
                        replaceFirstChar(&sphStem, asc("'"), asc(","))
                    }
                    if sphStem.count > 0 {
                        sphStem.append(contentsOf: bs("'e@rI#l%i"))
                        out = processPhonemeString(sphStem, false)
                    }
                }
            }
        }
        return out
    }

    func checkMorphologicalSuffixes(_ norm: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        if norm.count >= 5 {
            out = checkSuffixIng(norm)
            if out.count == 0 { out = checkSuffixEd(norm) }
        }
        if out.count == 0 { out = checkSuffixIes(norm) }
        if out.count == 0 { out = checkSuffixDictS(norm) }
        if out.count == 0 { out = checkSuffixMagicEs(norm) }
        if out.count == 0 { out = checkSuffixChShEs(norm) }
        if out.count == 0 { out = checkSuffixXes(norm) }
        if out.count == 0 { out = checkSuffixArily(norm) }
        return out
    }

    // Dispatch chain: the first non-empty arm wins.
    func wordToPhonemes(_ word: [UInt8]) -> [UInt8] {
        let norm = toLower(word)
        var out = checkCapitalDict(word, norm)
        if out.count == 0 { out = checkMainDict(norm) }
        if out.count == 0 { out = checkHyphenated(norm) }
        if out.count == 0 { out = checkPossessive(norm) }
        if out.count == 0 { out = checkSingleLetter(norm) }
        if out.count == 0 { out = checkMorphologicalSuffixes(norm) }
        if out.count == 0 { out = checkCompoundPrefixes(norm) }
        if out.count == 0 { out = applyRulesFallback(norm) }
        return out
    }

    // $onlys / noun-form / verb-flag suffixes take the rule path
    // rather than their dictionary entry.
    func prefixRuleSuffixPhonemes(_ suffix: [UInt8]) -> [UInt8] {
        var sfxPh = [UInt8]()
        let sfxLo = toLower(suffix)
        let onlys = onlysWords.contains(suffix)
        let nounForm = nounFormStress.contains(sfxLo)
        let verbFlag = verbFlagWords.contains(sfxLo)
        if onlys || nounForm || verbFlag {
            let rulesOut = applyRules(suffix, true, -1,
                                      false, false).out
            sfxPh = processPhonemeString(rulesOut, false)
        } else {
            sfxPh = wordToPhonemes(suffix)
        }
        return sfxPh
    }

    // Two primaries cannot stand: a long suffix demotes the prefix's, a
    // one-vowel prefix deletes the suffix's, else the suffix's is demoted.
    func demoteDoublePrimary(_ phonemes: inout [UInt8],
                             _ sfxPh: inout [UInt8],
                             _ suffix: [UInt8]) {
        let sfxSyllables = countSuffixSyllables(suffix, 0,
                                                suffix.count)
        let pfxVowels = countPrefixVowels(phonemes)
        let pfxEndsSchwa = sfxSyllables == 2 && pfxVowels >= 2
            && prefixEndsInSchwa(phonemes)
        if sfxSyllables >= 2 && pfxVowels >= 2 && !pfxEndsSchwa {
            let pp = firstIndexOf(phonemes, asc("'"))
            if pp >= 0 { phonemes[pp] = asc(",") }
        } else if pfxVowels == 1 {
            let sp = firstIndexOf(sfxPh, asc("'"))
            if sp >= 0 { sfxPh.remove(at: sp) }
        } else {
            let sp = firstIndexOf(sfxPh, asc("'"))
            if sp >= 0 { sfxPh[sp] = asc(",") }
        }
    }

    // A secondary right before a reduced vowel (@, I#, I2, a#) is dropped.
    func dropSecondaryBeforeReduced(_ sfxPh: inout [UInt8]) {
        let cp = firstIndexOf(sfxPh, asc(","))
        if cp >= 0 && cp + 1 < sfxPh.count {
            let nc = sfxPh[cp + 1]
            let phU = nc == asc("@")
                || (nc == asc("I") && cp + 2 < sfxPh.count
                    && (sfxPh[cp + 2] == asc("#")
                        || sfxPh[cp + 2] == asc("2")))
                || (nc == asc("a") && cp + 2 < sfxPh.count
                    && sfxPh[cp + 2] == asc("#"))
            if phU { sfxPh.remove(at: cp) }
        }
    }

    // The rule boundaries have done their job by this point.
    func stripRuleBoundaryBytes(_ phonemes: inout [UInt8]) {
        var stripped = [UInt8]()
        for r in 0..<phonemes.count where phonemes[r] != phBnd {
            stripped.append(phonemes[r])
        }
        phonemes = stripped
    }

    func processPrefixRule(_ word: [UInt8], _ pos: Int,
                           _ match: RuleMatch,
                           _ phonemes: inout [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let suffix = Array(word[(pos + match.advance)..<word.count])
        let prefixHasStress = containsByte(phonemes, asc("'"))
                           || containsByte(phonemes, asc(","))
                           || containsByte(phonemes, asc("%"))
        let skip = !prefixHasStress && prefixHasFullVowel(phonemes)
        if !skip {
            var sfxPh = prefixRuleSuffixPhonemes(suffix)
            let pfxHasPrime = containsByte(phonemes, asc("'"))
            let sfxHasPrime = containsByte(sfxPh, asc("'"))
            if pfxHasPrime && sfxHasPrime {
                demoteDoublePrimary(&phonemes, &sfxPh, suffix)
                dropSecondaryBeforeReduced(&sfxPh)
            }
            stripRuleBoundaryBytes(&phonemes)
            out = phonemes
            out.append(contentsOf: sfxPh)
        }
        return out
    }

    struct ApplyRulesResult {
        var out = [UInt8]()
        var replacedE = [Bool]()
        var posVisited = [Bool]()
    }

    // The applyRules scan: the cursor, the phonemes so far, and the result a
    // terminal suffix or prefix rule substitutes when it ends the scan early.
    struct RuleScan {
        var phonemes = [UInt8]()
        var finalResult = [UInt8]()
        var replacedE = [Bool]()
        var posVisited = [Bool]()
        var len = 0
        var pos = 0
        var finished = false
    }

    // A deleted span phonemizes as 'E' when the scan reaches it.
    func applyMatchDeletions(_ sc: inout RuleScan,
                             _ match: RuleMatch) {
        if match.delCount > 0 && match.delStart >= 0 {
            var d = 0
            while d < match.delCount && match.delStart + d < sc.len {
                sc.replacedE[match.delStart + d] = true
                d += 1
            }
        }
    }

    // A prefix rule that produced phonemes ends the scan; one that
    // produced none only advances the cursor.
    func applyPrefixMatch(_ word: [UInt8], _ match: RuleMatch,
                          _ sc: inout RuleScan) {
        let pos = sc.pos
        let pfxRes = processPrefixRule(word, pos, match,
                                       &sc.phonemes)
        if pfxRes.count > 0 {
            sc.finalResult = pfxRes
            sc.finished = true
        } else {
            sc.pos += match.advance
        }
    }

    func applyMatchedRule(_ word: [UInt8], _ wordAltFlagsV: Int,
                          _ allowSuffixStrip: Bool,
                          _ suffixPhonemeOnly: Bool,
                          _ match: inout RuleMatch,
                          _ sc: inout RuleScan) {
        applyStressPrev(&match.phonemes, &sc.phonemes)
        sc.phonemes.append(contentsOf: match.phonemes)
        sc.phonemes.append(phBnd)
        let terminalSuffix = (allowSuffixStrip || suffixPhonemeOnly)
            && match.isSuffix && sc.pos + match.advance == sc.len
        if terminalSuffix && !suffixPhonemeOnly {
            sc.finalResult = processSuffixRule(word, wordAltFlagsV,
                                               match)
            sc.finished = true
        } else if terminalSuffix && suffixPhonemeOnly {
            sc.pos += match.advance
        } else if match.isPrefix
                  && sc.pos + match.advance < sc.len {
            applyPrefixMatch(word, match, &sc)
        } else {
            applyMatchDeletions(&sc, match)
            sc.pos += match.advance
        }
    }

    func runRuleScan(_ word: [UInt8], _ wordAltFlagsV: Int,
                     _ allowSuffixStrip: Bool,
                     _ suffixPhonemeOnly: Bool,
                     _ suffixRemoved: Bool,
                     _ sc: inout RuleScan) {
        while sc.pos < sc.len && !sc.finished {
            sc.posVisited[sc.pos] = true
            let posChar = sc.replacedE[sc.pos] ?
                          asc("E") : toLowerC(word[sc.pos])
            var match = findBestRule(rules, word, word.count, sc.pos,
                                     sc.len, posChar, wordAltFlagsV,
                                     sc.replacedE, sc.len,
                                     allowSuffixStrip, suffixPhonemeOnly,
                                     suffixRemoved,
                                     sc.phonemes, sc.phonemes.count)
            if match.score < 0 {
                sc.pos += 1
            } else {
                applyMatchedRule(word, wordAltFlagsV, allowSuffixStrip,
                                 suffixPhonemeOnly, &match, &sc)
            }
        }
    }

    func applyRules(_ wordOrig: [UInt8], _ allowSuffixStrip: Bool,
                    _ wordAltFlagsParam: Int,
                    _ suffixPhonemeOnly: Bool,
                    _ suffixRemoved: Bool) -> ApplyRulesResult {
        var result = ApplyRulesResult()
        var word = wordOrig
        applyReplacements(&word, rules.replacements)
        let len = word.count
        let wordAltFlagsV = determineAltFlags(word, wordAltFlagsParam)
        var sc = RuleScan()
        sc.replacedE = [Bool](repeating: false, count: max(len, 1))
        sc.posVisited = [Bool](repeating: false, count: max(len, 1))
        sc.len = len
        runRuleScan(word, wordAltFlagsV, allowSuffixStrip,
                    suffixPhonemeOnly, suffixRemoved, &sc)
        result.out = sc.finished ? sc.finalResult : sc.phonemes
        result.replacedE = len > 0 ? Array(sc.replacedE[0..<len]) : []
        result.posVisited = len > 0 ? Array(sc.posVisited[0..<len]) : []
        return result
    }

    // Unstressed / boundary / word-end markers carry no sound.
    func isSilentMarkerCode(_ code: [UInt8], _ at: Int,
                            _ cn: Int) -> Bool {
        return (cn == 1
                && (code[at] == asc("%") || code[at] == asc("=")
                    || code[at] == asc("|")))
            || (cn == 2
                && ((code[at] == asc("=") && code[at + 1] == asc("="))
                    || (code[at] == asc("|")
                        && code[at + 1] == asc("|"))))
    }

    func singleCodeToIpa(_ code: [UInt8], _ at: Int,
                         _ cn: Int) -> [UInt8] {
        var out = [UInt8]()
        if cn == 1 && code[at] == asc("'") {
            out = [0xcb, 0x88]
        } else if cn == 1 && code[at] == asc(",") {
            out = [0xcb, 0x8c]
        } else if isSilentMarkerCode(code, at, cn) {
            // nothing emitted
        } else if cn > 0 {
            let key = Array(code[at..<(at + cn)])
            if let over = ipaOverrides[key] {
                out = over
            } else {
                let isVowel = isVowelCode(code, at, cn)
                phonemeCodeToIpaTable(code, at, cn, isVowel, &out)
            }
        }
        return out
    }

    struct EmitState {
        var i = 0
        var pendingStress = [UInt8]()
        var lastWasUnstress = false
        var lastCodeWasVowel = false
    }

    // The longest multiCodes entry at st.i, else the single byte. "i@" / "U@"
    // after %/= are false diphthongs and are not matched as pairs.
    func matchMultiCode(_ pstr: [UInt8], _ plen: Int,
                        _ st: inout EmitState) -> [UInt8] {
        var code = [UInt8]()
        var found = false
        var mi = 0
        while mi < multiCodes.count && !found {
            let mc = multiCodes[mi]
            let mclen = mc.count
            let skip = st.lastWasUnstress && mclen == 2
                && (mc == bs("i@") || mc == bs("U@"))
            if !skip && st.i + mclen <= plen
                && equalRange(pstr, st.i, mc, mclen) {
                code = mc
                st.i += mclen
                found = true
            }
            mi += 1
        }
        if !found {
            code = [pstr[st.i]]
            st.i += 1
        }
        return code
    }

    // Variant-marker digits map to themselves in IPA and carry no sound.
    func skipIdentityVariantDigits(_ pstr: [UInt8], _ plen: Int,
                                   _ st: inout EmitState) {
        while st.i < plen && pstr[st.i] >= asc("0")
              && pstr[st.i] <= asc("9")
              && asciiToIpa[Int(pstr[st.i]) - 0x20] == UInt32(pstr[st.i]) {
            st.i += 1
        }
    }

    // Syllabic consonants are nuclei, so they attract a pending stress like a
    // vowel.
    func isSyllabicCode(_ code: [UInt8]) -> Bool {
        return code.count == 2
            && ((code[0] == asc("n") && code[1] == asc("-"))
                || (code[0] == asc("m") && code[1] == asc("-"))
                || (code[0] == asc("@") && code[1] == asc("L"))
                || (code[0] == asc("r") && code[1] == asc("-"))
                || (code[0] == asc("l") && code[1] == asc("/")))
    }

    func emitPhonemeCode(_ pstr: [UInt8], _ plen: Int,
                         _ st: inout EmitState,
                         _ result: inout [UInt8]) {
        let code = matchMultiCode(pstr, plen, &st)
        let codeLen = code.count
        st.lastWasUnstress = false
        skipIdentityVariantDigits(pstr, plen, &st)
        let emitStress = st.pendingStress.count > 0
            && (isVowelCode(code, 0, codeLen) || isSyllabicCode(code))
        if emitStress {
            result.append(contentsOf: st.pendingStress)
            st.pendingStress = []
        }
        result.append(contentsOf: singleCodeToIpa(code, 0, codeLen))
        st.lastCodeWasVowel = isVowelCode(code, 0, codeLen)
    }

    func phonemesToIpa(_ ph: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let pn = ph.count
        if pn > 0 {
            var dollar = pn
            var i0 = 0
            while i0 < pn && dollar == pn {
                if ph[i0] == asc("$") { dollar = i0 }
                i0 += 1
            }
            let pstr = trim(ph, 0, dollar)
            let len = pstr.count
            var st = EmitState()
            while st.i < len {
                let c = pstr[st.i]
                if isSpaceC(c) {
                    st.i += 1
                } else if c == asc("|") || c == asc("-") {
                    st.i += 1
                } else if c == asc(";") {
                    if !st.lastCodeWasVowel {
                        out.append(contentsOf: [0xca, 0xb2])
                    }
                    st.i += 1
                } else if c == asc("'") || c == asc(",") {
                    st.pendingStress = singleCodeToIpa(pstr, st.i, 1)
                    st.lastWasUnstress = false
                    st.i += 1
                } else if c == asc("%") || c == asc("=") {
                    st.pendingStress = []
                    st.lastWasUnstress = true
                    st.i += 1
                } else {
                    emitPhonemeCode(pstr, len, &st, &out)
                }
            }
        }
        return out
    }

    func loadOverrideTable(_ t: [IpaOverrideEntry]) {
        for e in t { ipaOverrides[bs(e.code)] = e.ipa }
    }

    func buildIpaOverrides() {
        let isEnUs = dialect == bs("en-us") || dialect == bs("en_us")
        loadOverrideTable(ipaCommon)
        loadOverrideTable(isEnUs ? ipaEnUs : ipaEnGb)
    }

    func expandNumberToken(_ t: Token, _ result: inout [UInt8],
                           _ firstWord: inout Bool) -> Bool {
        var allDigits = t.text.count > 0
        for i in 0..<t.text.count where !isDigitC(t.text[i]) {
            allDigits = false
        }
        var consumed = false
        if allDigits {
            var numVal = 0
            for c in t.text where numVal <= 9999999 {
                numVal = numVal * 10 + Int(c - asc("0"))
            }
            if numVal >= 0 && numVal <= 9999999 {
                var numWords = [UInt8]()
                intToWords(numVal, &numWords)
                var i = 0
                var isFirstSub = true
                while i < numWords.count {
                    var j = i
                    while j < numWords.count && numWords[j] != asc(" ") {
                        j += 1
                    }
                    if j > i {
                        let wph = wordToPhonemes(Array(numWords[i..<j]))
                        let wipaRaw = phonemesToIpa(wph)
                        let wipa = addDefaultStress(wipaRaw,
                                                    wipaRaw.count)
                        if isFirstSub {
                            if t.needsSpace && !firstWord {
                                result.append(asc(" "))
                            }
                            isFirstSub = false
                        } else {
                            result.append(asc(" "))
                        }
                        result.append(contentsOf: wipa)
                        firstWord = false
                    }
                    i = j + 1
                }
                consumed = true
            }
        }
        return consumed
    }

    // Letter names: non-final codes demoted to ',', the final one promoted to
    // '\''.
    func spellGroup(_ grp: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        let gn = grp.count
        if gn > 0 {
            var codes = [UInt8]()
            for li in 0..<gn {
                let lcLower = toLowerC(grp[li])
                let uk: [UInt8] = [asc("_"), lcLower]
                var letterPh = [UInt8]()
                if let uit = dict[uk] {
                    letterPh = uit
                } else {
                    letterPh = wordToPhonemes([grp[li]])
                }
                if li + 1 < gn {
                    var first = letterPh.count
                    var k = 0
                    while k < letterPh.count && first == letterPh.count {
                        if letterPh[k] == asc("'")
                            || letterPh[k] == asc(",") {
                            first = k
                        }
                        k += 1
                    }
                    if first == letterPh.count {
                        codes.append(asc(","))
                        codes.append(contentsOf: letterPh)
                    } else {
                        codes.append(contentsOf: letterPh[0..<first])
                        codes.append(asc(","))
                        var k2 = first + 1
                        while k2 < letterPh.count {
                            let c2 = letterPh[k2]
                            if c2 != asc("'") && c2 != asc(",") {
                                codes.append(c2)
                            }
                            k2 += 1
                        }
                    }
                } else {
                    if !containsByte(letterPh, asc("'")) {
                        codes.append(asc("'"))
                    }
                    codes.append(contentsOf: letterPh)
                }
            }
            if codes.count > 0 {
                let processed = processPhonemeString(codes, false)
                out = phonemesToIpa(processed)
            }
        }
        return out
    }

    func isAllUpper(_ s: [UInt8]) -> Bool {
        var allUpper = s.count >= 2
        for i in 0..<s.count where !isUpperC(s[i]) { allUpper = false }
        return allUpper
    }

    func isPlainVowel(_ lc: UInt8) -> Bool {
        lc == asc("a") || lc == asc("e") || lc == asc("i")
            || lc == asc("o") || lc == asc("u")
    }

    func isNasalLetter(_ lc: UInt8) -> Bool {
        lc == asc("m") || lc == asc("h") || lc == asc("n")
    }

    // A token whose letters are ALL from {m, h, n} is a nasal interjection
    // ("Mmm", "Hmm"); real abbreviations ("PhD", "BSc") carry other consonants.
    func isMixedCaseNoVowel(_ s: [UInt8]) -> Bool {
        var anyUpper = false
        var anyLower = false
        var anyVowel = false
        var allNasal = true
        for i in 0..<s.count {
            let ch = s[i]
            if isUpperC(ch) { anyUpper = true }
            if isLowerC(ch) { anyLower = true }
            let lc = toLowerC(ch)
            if isPlainVowel(lc) { anyVowel = true }
            if !isNasalLetter(lc) { allNasal = false }
        }
        return anyUpper && anyLower && !anyVowel && !allNasal
    }

    // CamelCase: a new group starts where a lower letter meets an upper one.
    func spellAcronymGroups(_ t: Token, _ splitCamel: Bool,
                            _ result: inout [UInt8],
                            _ firstWord: inout Bool) {
        var firstGrp = true
        var gs = 0
        var ci = 0
        while ci <= t.text.count {
            let boundary = splitCamel && ci > 0 && ci < t.text.count
                && isLowerC(t.text[ci - 1]) && isUpperC(t.text[ci])
            if boundary || ci == t.text.count {
                if ci > gs {
                    let ipa = spellGroup(Array(t.text[gs..<ci]))
                    if ipa.count > 0 {
                        let leadSpace = firstGrp
                            ? (t.needsSpace && !firstWord) : true
                        if leadSpace { result.append(asc(" ")) }
                        result.append(contentsOf: ipa)
                        firstWord = false
                        firstGrp = false
                    }
                }
                gs = ci
            }
            ci += 1
        }
    }

    func spellAcronymToken(_ t: Token, _ result: inout [UInt8],
                           _ firstWord: inout Bool) -> Bool {
        let lower = toLower(t.text)
        let allUpper = isAllUpper(t.text)
        let unknownWord = dict[lower] == nil
        let mixedCaseNoVowelAbbrev = !allUpper && t.text.count >= 2
            && unknownWord && isMixedCaseNoVowel(t.text)
        let consumed = (allUpper || mixedCaseNoVowelAbbrev)
            && (abbrevWords.contains(lower) || unknownWord)
        if consumed {
            spellAcronymGroups(t, mixedCaseNoVowelAbbrev, &result,
                               &firstWord)
        }
        return consumed
    }

    func wordIsVowelInitialNonYod(_ ts: [Token], _ idx: Int) -> Bool {
        var result = false
        if ts[idx].text.count > 0 {
            let fc = toLowerC(ts[idx].text[0])
            let isVowel = fc == asc("a") || fc == asc("e") || fc == asc("i")
                || fc == asc("o") || fc == asc("u")
            if isVowel {
                var jOnset = false
                if fc == asc("u") || fc == asc("e") {
                    let lower = toLower(ts[idx].text)
                    let nph = wordToPhonemes(lower)
                    var npi = 0
                    while npi < nph.count
                        && (nph[npi] == asc("'") || nph[npi] == asc(",")
                            || nph[npi] == asc("%")
                            || nph[npi] == asc("=")) {
                        npi += 1
                    }
                    jOnset = npi < nph.count && nph[npi] == asc("j")
                }
                result = !jOnset
            }
        }
        return result
    }

    func tokenIsAllUpper(_ text: [UInt8]) -> Bool {
        var allUpper = text.count > 0
        for i in 0..<text.count where !isUpperC(text[i]) {
            allUpper = false
        }
        return allUpper
    }

    // An all-caps token unknown to the dictionary, or flagged $abbrev, is
    // spelled out.
    func spellsOutLetterByLetter(_ text: [UInt8]) -> Bool {
        var spellsOut = false
        if tokenIsAllUpper(text) {
            let lowerNxt = toLower(text)
            let unknown = dict[lowerNxt] == nil
            let isAbbrev = abbrevWords.contains(lowerNxt)
            spellsOut = unknown || isAbbrev
        }
        return spellsOut
    }

    func firstLetterPhonemes(_ text: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        var found = false
        var i = 0
        while i < text.count && !found {
            let lc = text[i]
            if isAlphaC(lc) {
                out = wordToPhonemes([lc])
                found = true
            }
            i += 1
        }
        return out
    }

    func nextWordIsVowelInitial(_ ts: [Token], _ tj: Int) -> Bool {
        let text = ts[tj].text
        var useFirstLetter = containsByte(text, asc("."))
        if !useFirstLetter && text.count >= 2 {
            useFirstLetter = spellsOutLetterByLetter(text)
        }
        var nextPh = [UInt8]()
        if useFirstLetter {
            nextPh = firstLetterPhonemes(text)
        } else {
            nextPh = wordToPhonemes(text)
        }
        var pi = 0
        while pi < nextPh.count
            && (nextPh[pi] == asc("'") || nextPh[pi] == asc(",")
                || nextPh[pi] == asc("%") || nextPh[pi] == asc("=")) {
            pi += 1
        }
        return pi < nextPh.count && isVowelCode(nextPh, pi, 1)
    }

    // R-sandhi, suppressed before a mid-utterance conjunction and across
    // punctuation.
    func applyRLinking(_ firstWord: Bool, _ ti: Int,
                       _ ts: [Token], _ ipa: inout [UInt8]) {
        if endsInRhoticR(ipa, ipa.count) {
            let tj = findNextWordStopOnPunct(ts, ti + 1)
            if tj >= 0 && !conjunctionSuppressesLinking(ts, tj, firstWord)
                && nextWordIsVowelInitial(ts, tj) {
                ipa.append(contentsOf: [0xC9, 0xB9])
            }
        }
    }

    let vowelStartsFlap = bs("aAeEIiOUVu03@oY")

    func applyInterWordTFlap(_ ti: Int, _ ts: [Token],
                             _ phCodes: inout [UInt8]) {
        if phCodes.count >= 2
            && phCodes[phCodes.count - 2] == asc("t")
            && phCodes[phCodes.count - 1] == asc("#") {
            var tnext = ti + 1
            while tnext < ts.count && !ts[tnext].isWord { tnext += 1 }
            if tnext < ts.count && ts[tnext].text.count > 0 {
                let lower = toLower(ts[tnext].text)
                let nph = wordToPhonemes(lower)
                var npi = 0
                while npi < nph.count
                    && (nph[npi] == asc("'") || nph[npi] == asc(",")
                        || nph[npi] == asc("%") || nph[npi] == asc("=")) {
                    npi += 1
                }
                let nextVowelOnset = npi < nph.count
                    && strchrHit(vowelStartsFlap, nph[npi])
                if nextVowelOnset {
                    phCodes[phCodes.count - 2] = asc("*")
                    phCodes.removeLast()
                }
            }
        }
    }

    // A schwa after another vowel is the tail of a diphthong, not standalone.
    func trailingSchwaIsStandalone(_ ph: [UInt8]) -> Bool {
        let prev: UInt8 = ph.count >= 2 ? ph[ph.count - 2] : 0
        return !(prev == asc("A") || prev == asc("e")
                 || prev == asc("O") || prev == asc("o")
                 || prev == asc("U") || prev == asc("i"))
    }

    func nextWordStartsWithR(_ ts: [Token], _ tj: Int) -> Bool {
        var startsR = false
        if tj < ts.count && ts[tj].isWord && ts[tj].text.count > 0 {
            let nph = wordToPhonemes(ts[tj].text)
            var pi = 0
            while pi < nph.count
                && (nph[pi] == asc("'") || nph[pi] == asc(",")
                    || nph[pi] == asc("%") || nph[pi] == asc("=")) {
                pi += 1
            }
            startsR = pi < nph.count && nph[pi] == asc("r")
        }
        return startsR
    }

    func applyCrossWordSchwaRhotic(_ ti: Int, _ isEnUs: Bool,
                                   _ isIsolatedWord: Bool,
                                   _ ts: [Token],
                                   _ phCodes: inout [UInt8]) {
        if isEnUs && !isIsolatedWord && phCodes.count > 0
            && phCodes[phCodes.count - 1] == asc("@")
            && trailingSchwaIsStandalone(phCodes)
            && nextWordStartsWithR(ts, ti + 1) {
            phCodes[phCodes.count - 1] = asc("3")
        }
    }

    // Cross-word /t/ flap, restricted to "it".
    func nextWordIndex(_ ts: [Token], _ from: Int) -> Int {
        var tj = from
        while tj < ts.count && !ts[tj].isWord { tj += 1 }
        return tj
    }

    func nextWordPhonemeIsVowel(_ ts: [Token], _ tj: Int) -> Bool {
        let nextVowels = bs("aAeEiIoOuUV03@")
        var isVowel = false
        if tj < ts.count && ts[tj].text.count > 0 {
            let nph = wordToPhonemes(ts[tj].text)
            var pi2 = 0
            while pi2 < nph.count
                && (nph[pi2] == asc("'") || nph[pi2] == asc(",")
                    || nph[pi2] == asc("%") || nph[pi2] == asc("=")) {
                pi2 += 1
            }
            isVowel = pi2 < nph.count
                && strchrHit(nextVowels, nph[pi2])
        }
        return isVowel
    }

    func applyCrossWordTFlap(_ t: Token, _ ti: Int, _ isEnUs: Bool,
                             _ isIsolatedWord: Bool, _ ts: [Token],
                             _ ipa: inout [UInt8]) {
        let tokenIsIt = t.text.count == 2
            && toLowerC(t.text[0]) == asc("i")
            && toLowerC(t.text[1]) == asc("t")
        if isEnUs && !isIsolatedWord && ipa.count > 0
            && ipa[ipa.count - 1] == asc("t") && tokenIsIt
            && nextWordPhonemeIsVowel(ts, nextWordIndex(ts, ti + 1)) {
            ipa.removeLast()
            ipa.append(contentsOf: [0xC9, 0xBE])
        }
    }

    // A '%' word with no primary is unstressed, unless it carries a
    // secondary outside a matched phrase.
    func pctWordUnstressed(_ phCodes: [UInt8],
                           _ phraseMatched: Bool,
                           _ isIsolatedWord: Bool) -> Bool {
        let hasPct = containsByte(phCodes, asc("%"))
        let hasPrime = containsByte(phCodes, asc("'"))
        let hasComma = containsByte(phCodes, asc(","))
        return hasPct && !hasPrime
            && (!hasComma || phraseMatched) && !isIsolatedWord
    }

    // @2 / @5 with no stress marker of its own stays weak.
    func hasWeakSchwaOnly(_ phCodes: [UInt8]) -> Bool {
        let pn = phCodes.count
        let hasPrime = containsByte(phCodes, asc("'"))
        let hasComma = containsByte(phCodes, asc(","))
        let hasAt2 = pn >= 2 && containsSub(phCodes, bs("@2"))
        let hasAt5 = pn >= 2 && containsSub(phCodes, bs("@5"))
        return (hasAt2 || hasAt5) && !hasPrime && !hasComma
    }

    // The articles "a" / "an" (a# / a#n) never take stress in a sentence.
    func isUnstressedArticle(_ t: Token, _ phCodes: [UInt8],
                             _ isIsolatedWord: Bool) -> Bool {
        let tl = toLower(t.text)
        let isA = tl.count == 1 && tl[0] == asc("a")
        let isAn = tl == bs("an")
        let isASharp = phCodes == bs("a#")
        let isASharpN = phCodes == bs("a#n")
        return !isIsolatedWord && (isA || isAn)
            && (isASharp || isASharpN)
    }

    func maybeAddDefaultStress(_ t: Token, _ phraseMatched: Bool,
                               _ isIsolatedWord: Bool,
                               _ isUnstressedWord: Bool,
                               _ phCodes: [UInt8],
                               _ ipa: inout [UInt8]) {
        let noStress = pctWordUnstressed(phCodes, phraseMatched,
                                         isIsolatedWord)
            || hasWeakSchwaOnly(phCodes)
            || (isUnstressedWord && !isIsolatedWord)
            || isUnstressedArticle(t, phCodes, isIsolatedWord)
        if !noStress {
            ipa = addDefaultStress(ipa, ipa.count)
        }
    }

    // $pastf/$nounf/$verbf open a window; the tail decrement gives the 2/1
    // schedule.
    func updatePosContextCounters(_ t: Token, _ expectPast: inout Int,
                                  _ expectNoun: inout Int,
                                  _ expectVerb: inout Int) {
        let lw = toLower(t.text)
        if pastfWords.contains(lw) {
            expectPast = 3
            expectNoun = 0
            expectVerb = 0
        } else if nounfWords.contains(lw) {
            expectNoun = 2
            expectPast = 0
            expectVerb = 0
        } else if verbfWords.contains(lw) {
            expectVerb = 2
            expectPast = 0
            expectNoun = 0
        }
        if expectPast > 0 { expectPast -= 1 }
        if expectNoun > 0 { expectNoun -= 1 }
        if expectVerb > 0 { expectVerb -= 1 }
    }

    func applyPosContextOverride(_ t: Token, _ isIsolatedWord: Bool,
                                 _ phraseMatched: Bool,
                                 _ expectPast: Int, _ expectNoun: Int,
                                 _ expectVerb: Int,
                                 _ phCodes: inout [UInt8]) {
        if !isIsolatedWord && !phraseMatched {
            let lw = toLower(t.text)
            var hit: [UInt8]?
            if expectPast > 0 {
                hit = pastDict[lw]
            } else if expectNoun > 0 {
                hit = nounDict[lw]
            } else if expectVerb > 0 {
                hit = verbDict[lw]
            }
            if let h = hit {
                phCodes = processPhonemeString(h, false)
            }
        }
    }

    func applyAtstartOverride(_ t: Token, _ firstWord: Bool,
                              _ phraseMatched: Bool,
                              _ phCodes: inout [UInt8]) {
        if firstWord && !phraseMatched {
            let lw = toLower(t.text)
            if let ait = atstartDict[lw] { phCodes = ait }
        }
    }

    func applyAtendOverride(_ t: Token, _ ti: Int, _ lastWordTi: Int,
                            _ isIsolatedWord: Bool,
                            _ phraseMatched: Bool,
                            _ phCodes: inout [UInt8]) {
        if ti == lastWordTi && !isIsolatedWord && !phraseMatched {
            let lw = toLower(t.text)
            if let aeit = atendDict[lw] { phCodes = aeit }
        }
    }

    // Hand-coded function-word allophone overrides, each gated on lemma
    // equality.
    func lemmaOverrideThe(_ ti: Int, _ ts: [Token],
                          _ phCodes: inout [UInt8]) {
        let tj = findNextWordStopOnPunct(ts, ti + 1)
        if tj >= 0 && wordIsVowelInitialNonYod(ts, tj) {
            phCodes = bs("%DI")
        }
    }

    func lemmaOverrideA(_ isIsolatedWord: Bool,
                        _ phCodes: inout [UInt8]) {
        phCodes = isIsolatedWord ? bs("eI") : bs("a#")
    }

    func lemmaOverrideAn(_ ti: Int, _ ts: [Token],
                         _ phCodes: inout [UInt8]) {
        let tj2 = findNextNonEmptyWord(ts, ti + 1)
        var nextVowelInitial = false
        if tj2 >= 0 {
            nextVowelInitial = isPlainVowel(toLowerC(ts[tj2].text[0]))
        }
        phCodes = nextVowelInitial ? bs("a#n") : bs("an")
    }

    func lemmaOverrideTo(_ ti: Int, _ ts: [Token],
                         _ isIsolatedWord: Bool, _ lastWordTi: Int,
                         _ phCodes: inout [UInt8]) {
        if isIsolatedWord || ti == lastWordTi {
            phCodes = bs("tu:")
        } else {
            let tj2 = findNextNonEmptyWord(ts, ti + 1)
            let useTU = tj2 >= 0 && wordIsVowelInitialNonYod(ts, tj2)
            phCodes = useTU ? bs("tU") : bs("t@5")
        }
    }

    // "use" after a pronoun subject is the verb (voiced z), not the noun.
    func lemmaOverrideUse(_ ti: Int, _ ts: [Token],
                          _ phCodes: inout [UInt8]) {
        var prevWord = [UInt8]()
        var found = false
        var tj = ti - 1
        while tj >= 0 && !found {
            if ts[tj].isWord {
                prevWord = toLower(ts[tj].text)
                found = true
            }
            tj -= 1
        }
        if wordListHas(pronounsUse, prevWord) {
            phCodes = bs("ju:z")
        }
    }

    func applyLemmaOverride(_ t: Token, _ ti: Int, _ ts: [Token],
                            _ isIsolatedWord: Bool, _ lastWordTi: Int,
                            _ phCodes: inout [UInt8]) {
        let lw = toLower(t.text)
        let eqThe = lw == bs("the")
        let eqA = lw.count == 1 && lw[0] == asc("a")
        let eqAn = lw == bs("an")
        let eqTo = lw == bs("to")
        let eqUse = lw == bs("use")
        if eqThe && phCodes.count > 0 {
            lemmaOverrideThe(ti, ts, &phCodes)
        }
        if eqA { lemmaOverrideA(isIsolatedWord, &phCodes) }
        if eqAn && !isIsolatedWord {
            lemmaOverrideAn(ti, ts, &phCodes)
        }
        if eqTo {
            lemmaOverrideTo(ti, ts, isIsolatedWord, lastWordTi,
                            &phCodes)
        }
        if eqUse && !isIsolatedWord {
            lemmaOverrideUse(ti, ts, &phCodes)
        }
    }

    // "-ing forms of $strend2 stems whose dict entry carries a secondary"
    // -- e.g. "making" from "make" (m,eIk).
    func isIngOfStrendSecondary(_ w: [UInt8]) -> Bool {
        var result = false
        let wn = w.count
        if wn > 3 && equalRange(w, wn - 3, bs("ing"), 3) {
            let bn = wn - 3
            let base = pre(w, bn)
            var sk = [UInt8]()
            if strendWords.contains(base) {
                sk = base
            } else {
                var magicE = base
                magicE.append(asc("e"))
                if strendWords.contains(magicE) { sk = magicE }
            }
            if sk.count > 0 {
                if let sit = dict[sk] {
                    result = containsByte(sit, asc(","))
                }
            }
        }
        return result
    }

    // A following word is weak when it is not itself a keep-secondary
    // word and its dictionary entry opens with ',' or '%'.
    func followingWordIsWeak(_ fw: [UInt8]) -> Bool {
        let fwIsStrendSec = commaStrend2Words.contains(fw)
        let fwIsIngStrend = isIngOfStrendSecondary(fw)
        let fwInSecondarySet =
            wordListHas(stepCKeepSecondary, fw)
            || u2Strend2Words.contains(fw)
            || fwIsStrendSec || fwIsIngStrend
        let dit = dict[fw]
        return !fwInSecondarySet && dit != nil && dit!.count > 0
            && (dit![0] == asc(",") || dit![0] == asc("%"))
    }

    func tryPromoteKeepSecToPrimary(_ tl: [UInt8], _ ti: Int,
                                    _ ts: [Token], _ keepSec: inout Bool,
                                    _ phCodes: inout [UInt8]) {
        if keepSec && !uPlusSecondaryWords.contains(tl) {
            var hasFollowingStressed = false
            var tj = ti + 1
            while tj < ts.count && !hasFollowingStressed {
                if ts[tj].isWord {
                    let fw = toLower(ts[tj].text)
                    if !unstressedWords.contains(fw)
                        && !followingWordIsWeak(fw) {
                        hasFollowingStressed = true
                    }
                }
                tj += 1
            }
            if !hasFollowingStressed {
                keepSec = false
                if !containsByte(phCodes, asc("'")) {
                    replaceFirstChar(&phCodes, asc(","), asc("'"))
                }
            }
        }
    }

    func firstBytePos(_ s: [UInt8], _ c: UInt8) -> Int {
        var at = s.count
        var i = 0
        while i < s.count && at == s.count {
            if s[i] == c { at = i }
            i += 1
        }
        return at
    }

    // A trailing 's clitic does not change whether the word is unstressed.
    func tokenIsUnstressedWord(_ tl: [UInt8],
                               _ phraseMatched: Bool) -> Bool {
        let aposPos = firstBytePos(tl, asc("'"))
        return !phraseMatched
            && (unstressedWords.contains(tl)
                || (aposPos != tl.count
                    && unstressedWords.contains(pre(tl, aposPos))))
    }

    // Position of the "'a#" the step-5 last resort leaves behind.
    func primeBeforeHashAPos(_ phCodes: [UInt8]) -> Int {
        var at = phCodes.count
        if phCodes.count >= 3 {
            var j = 0
            while j + 2 < phCodes.count && at == phCodes.count {
                if phCodes[j] == asc("'") && phCodes[j + 1] == asc("a")
                    && phCodes[j + 2] == asc("#") {
                    at = j
                }
                j += 1
            }
        }
        return at
    }

    // A %-prefix word whose only primary is that last resort.
    func isPctLastResortWord(_ phCodes: [UInt8],
                             _ isIsolatedWord: Bool) -> Bool {
        let primePos = firstBytePos(phCodes, asc("'"))
        let hashAPos = primeBeforeHashAPos(phCodes)
        return !isIsolatedWord && phCodes.count > 0
            && phCodes[0] == asc("%") && primePos < phCodes.count
            && hashAPos < phCodes.count && primePos == hashAPos
    }

    func stripAllPrimes(_ phCodes: inout [UInt8]) {
        var stripped = [UInt8]()
        for ri in 0..<phCodes.count where phCodes[ri] != asc("'") {
            stripped.append(phCodes[ri])
        }
        phCodes = stripped
    }

    // Step B: strip '\'' from $u-flagged words and from %-prefix words whose
    // only primary is the step-5 last resort before 'a#'.
    func applyStepB(_ t: Token, _ phraseMatched: Bool,
                    _ isIsolatedWord: Bool,
                    _ phCodes: inout [UInt8]) -> Bool {
        let tokenLower = toLower(t.text)
        let isUnstressedWord = tokenIsUnstressedWord(tokenLower,
                                                     phraseMatched)
        let isStepBKeepSec = wordListHas(stepBKeepSecondaryWords,
                                         tokenLower)
        if (isUnstressedWord
            || isPctLastResortWord(phCodes, isIsolatedWord))
            && !isIsolatedWord && !isStepBKeepSec {
            stripAllPrimes(&phCodes)
        }
        return isUnstressedWord
    }

    // The word keeps a secondary instead of taking the primary.
    func stepCKeepsSecondary(_ tl: [UInt8], _ isIsolatedWord: Bool,
                             _ phCodes: [UInt8],
                             _ matchedPhraseKey: [UInt8]) -> Bool {
        let hasPrime = containsByte(phCodes, asc("'"))
        let isStrendSecondary = commaStrend2Words.contains(tl)
            && phCodes.count > 0 && phCodes[0] == asc(",") && !hasPrime
        let isIngOfStrend = isIngOfStrendSecondary(tl)
        let isKeepSecPhrase = matchedPhraseKey.count > 0
            && keepSecPhraseKeys.contains(matchedPhraseKey)
        return !isIsolatedWord
            && (wordListHas(stepCKeepSecondary, tl)
                || u2Strend2Words.contains(tl)
                || uPlusSecondaryWords.contains(tl)
                || isStrendSecondary || isIngOfStrend
                || isKeepSecPhrase)
    }

    // Needs a secondary and carries none: one goes on the first strong vowel.
    func insertSecondaryAtStrongVowel(_ phCodes: inout [UInt8]) {
        var pos = phCodes.count
        var i = 0
        while i < phCodes.count && pos == phCodes.count {
            if strchrHit(strongVowelsC, phCodes[i]) { pos = i }
            i += 1
        }
        if pos < phCodes.count { phCodes.insert(asc(","), at: pos) }
    }

    func applyStepC(_ t: Token, _ ti: Int, _ ts: [Token],
                    _ isIsolatedWord: Bool, _ phraseMatched: Bool,
                    _ lastWordTi: Int,
                    _ matchedPhraseKey: [UInt8],
                    _ phCodes: inout [UInt8]) {
        let tl = toLower(t.text)
        var keepSec = stepCKeepsSecondary(tl, isIsolatedWord, phCodes,
                                          matchedPhraseKey)
        let needsSec = !isIsolatedWord
            && wordListHas(stepCNeedsSecondary, tl)
        tryPromoteKeepSecToPrimary(tl, ti, ts, &keepSec, &phCodes)
        if keepSec { replaceFirstChar(&phCodes, asc("'"), asc(",")) }
        let hasComma = containsByte(phCodes, asc(","))
        let hasPrime = containsByte(phCodes, asc("'"))
        if needsSec && !hasPrime && !hasComma {
            insertSecondaryAtStrongVowel(&phCodes)
        }
        applyCommaToPrimaryPromotion(keepSec, needsSec, phraseMatched,
                                     isIsolatedWord, &phCodes)
        if !isIsolatedWord && ti == lastWordTi
            && unstressendWords.contains(tl) {
            replaceFirstChar(&phCodes, asc("'"), asc(","))
        }
    }

    struct CliticOrPhraseResult {
        var phCodes = [UInt8]()
        var phraseMatched = false
        var cliticMatched = false
        var phrasePreVowelThe = false
        var matchedPhraseKey = [UInt8]()
        var advanceTo = 0
    }

    func tryEmitDirectClitic(_ bigram: [UInt8], _ t: Token,
                             _ ts: [Token], _ tj: Int,
                             _ result: inout [UInt8],
                             _ firstWord: inout Bool) -> Bool {
        let raw = lookupConstTable(cliticIpa, bigram)
        let matched = raw != nil
        if let r = raw {
            var clitic = r
            let t2lo = toLower(ts[tj].text)
            if t2lo == bs("the") {
                let tk = findNextNonEmptyWord(ts, tj + 1)
                if tk >= 0 && wordIsVowelInitialNonYod(ts, tk)
                    && clitic.count >= 2
                    && clitic[clitic.count - 2] == 0xc9
                    && clitic[clitic.count - 1] == 0x99 {
                    clitic[clitic.count - 1] = 0xaa
                }
            }
            if t.needsSpace && !firstWord { result.append(asc(" ")) }
            result.append(contentsOf: clitic)
            firstWord = false
        }
        return matched
    }

    func tryMatchStaticPhrase(_ bigram: [UInt8],
                              _ r: inout CliticOrPhraseResult) -> Bool {
        let raw = lookupConstTable(staticPhraseCodes, bigram)
        let matched = raw != nil
        if let v = raw {
            r.phCodes = processPhonemeString(v, false)
            r.phraseMatched = true
        }
        return matched
    }

    func tryMatchLoadedPhrase(_ bigram: [UInt8], _ tj: Int,
                              _ ts: [Token],
                              _ r: inout CliticOrPhraseResult) -> Bool {
        let pit = phraseDict[bigram]
        let matched = pit != nil
        if let v = pit {
            let phraseEndsThe = v.count >= 3
                && equalRange(v, v.count - 3, bs("D@2"), 3)
            if phraseEndsThe {
                let tk = findNextNonEmptyWord(ts, tj + 1)
                r.phrasePreVowelThe = tk >= 0
                    && wordIsVowelInitialNonYod(ts, tk)
            }
            r.phCodes = processPhonemeString(v, false)
            r.phraseMatched = true
            r.matchedPhraseKey = bigram
        }
        return matched
    }

    // One half of a split phrase: prefixed '%' when it carries no stress mark.
    func renderSplitPart(_ ph: [UInt8], _ isFirst: Bool,
                         _ phraseHasPrimary: Bool) -> [UInt8] {
        let hasStress = containsByte(ph, asc("'"))
            || containsByte(ph, asc(","))
        var phProc = [UInt8]()
        if !hasStress && !(isFirst && !phraseHasPrimary) {
            phProc.append(asc("%"))
        }
        phProc.append(contentsOf: ph)
        let processed = processPhonemeString(phProc, false)
        return phonemesToIpa(processed)
    }

    func tryEmitSplitPhrase(_ bigram: [UInt8], _ t: Token,
                            _ result: inout [UInt8],
                            _ firstWord: inout Bool) -> Bool {
        let psit = phraseSplitDict[bigram]
        let matched = psit != nil
        if let p = psit {
            let phraseHasPrimary = containsByte(p.a, asc("'"))
                || containsByte(p.b, asc("'"))
            let ipa1 = renderSplitPart(p.a, true, phraseHasPrimary)
            if t.needsSpace && !firstWord { result.append(asc(" ")) }
            result.append(contentsOf: ipa1)
            firstWord = false
            let ipa2 = renderSplitPart(p.b, false, phraseHasPrimary)
            result.append(asc(" "))
            result.append(contentsOf: ipa2)
        }
        return matched
    }

    func tryCliticOrPhrase(_ t: Token, _ ti: Int, _ ts: [Token],
                           _ result: inout [UInt8],
                           _ firstWord: inout Bool)
            -> CliticOrPhraseResult {
        var r = CliticOrPhraseResult()
        r.advanceTo = ti
        var tj = ti + 1
        while tj < ts.count && !ts[tj].isWord { tj += 1 }
        if tj < ts.count && ts[tj].isWord {
            let bigram = buildLowerBigram(t, ts[tj])
            if tryEmitDirectClitic(bigram, t, ts, tj, &result,
                                   &firstWord) {
                r.cliticMatched = true
                r.advanceTo = tj
            } else if tryMatchStaticPhrase(bigram, &r) {
                r.advanceTo = tj
            } else if tryMatchLoadedPhrase(bigram, tj, ts, &r) {
                r.advanceTo = tj
            } else if tryEmitSplitPhrase(bigram, t, &result,
                                         &firstWord) {
                r.cliticMatched = true
                r.advanceTo = tj
            }
        }
        return r
    }

    // "U.S." / "U.K.": letter-spell with a secondary on all but the last
    // letter.
    func collectLetterPhonemes(_ text: [UInt8]) -> [[UInt8]] {
        var letterIpa = [[UInt8]]()
        for ci in 0..<text.count {
            let lc = text[ci]
            if isAlphaC(lc) {
                let lcLower = toLowerC(lc)
                let uk: [UInt8] = [asc("_"), lcLower]
                var entry = [UInt8]()
                if let uit = dict[uk] {
                    entry = uit
                } else {
                    entry = wordToPhonemes([lc])
                }
                letterIpa.append(entry)
            }
        }
        return letterIpa
    }

    // Every letter but the last carries a secondary: its own first
    // marker becomes that secondary and any later marker is dropped.
    func appendSecondaryLetter(_ combined: inout [UInt8],
                               _ code: [UInt8]) {
        var first = code.count
        var k = 0
        while k < code.count && first == code.count {
            if code[k] == asc("'") || code[k] == asc(",") {
                first = k
            }
            k += 1
        }
        if first == code.count {
            combined.append(asc(","))
            combined.append(contentsOf: code)
        } else {
            combined.append(contentsOf: code[0..<first])
            combined.append(asc(","))
            var k2 = first + 1
            while k2 < code.count {
                let c2 = code[k2]
                if c2 != asc("'") && c2 != asc(",") {
                    combined.append(c2)
                }
                k2 += 1
            }
        }
    }

    // The last letter carries the primary.
    func appendPrimaryLetter(_ combined: inout [UInt8],
                             _ code: [UInt8]) {
        if !containsByte(code, asc("'")) {
            combined.append(asc("'"))
        }
        combined.append(contentsOf: code)
    }

    func combineLetterCodes(_ letterIpa: [[UInt8]]) -> [UInt8] {
        var combined = [UInt8]()
        for li in 0..<letterIpa.count {
            let code = letterIpa[li]
            if li + 1 < letterIpa.count {
                appendSecondaryLetter(&combined, code)
            } else {
                appendPrimaryLetter(&combined, code)
            }
        }
        return combined
    }

    func emitSpelledAbbreviation(_ t: Token, _ ipa: [[UInt8]],
                                 _ result: inout [UInt8],
                                 _ firstWord: inout Bool) {
        let combinedIpa = phonemesToIpa(combineLetterCodes(ipa))
        if t.needsSpace && !firstWord { result.append(asc(" ")) }
        result.append(contentsOf: combinedIpa)
        firstWord = false
    }

    func expandPeriodAbbreviation(_ t: Token, _ result: inout [UInt8],
                                  _ firstWord: inout Bool,
                                  _ phCodes: inout [UInt8]) -> Bool {
        var consumed = false
        let hasDot = containsByte(t.text, asc("."))
        if hasDot {
            let letterIpa = collectLetterPhonemes(t.text)
            if letterIpa.count >= 2 {
                emitSpelledAbbreviation(t, letterIpa, &result,
                                        &firstWord)
                consumed = true
            } else if letterIpa.count == 1 {
                phCodes = letterIpa[0]
            } else {
                phCodes = wordToPhonemes(t.text)
            }
        } else {
            phCodes = wordToPhonemes(t.text)
        }
        return consumed
    }

    func processWordToken(_ t: Token, _ ti: inout Int, _ ts: [Token],
                          _ isIsolatedWord: Bool, _ lastWordTi: Int,
                          _ isEnUs: Bool,
                          _ expectPast: inout Int,
                          _ expectNoun: inout Int,
                          _ expectVerb: inout Int,
                          _ result: inout [UInt8],
                          _ firstWord: inout Bool) {
        var consumed = false
        if !consumed && expandNumberToken(t, &result, &firstWord) {
            consumed = true
        }
        if !consumed && spellAcronymToken(t, &result, &firstWord) {
            consumed = true
        }
        var phCodes = [UInt8]()
        var phraseMatched = false
        var phrasePreVowelThe = false
        var matchedPhraseKey = [UInt8]()
        if !consumed {
            let cr = tryCliticOrPhrase(t, ti, ts, &result, &firstWord)
            phCodes = cr.phCodes
            phraseMatched = cr.phraseMatched
            phrasePreVowelThe = cr.phrasePreVowelThe
            matchedPhraseKey = cr.matchedPhraseKey
            ti = cr.advanceTo
            if cr.cliticMatched { consumed = true }
        }
        if !consumed && !phraseMatched {
            if expandPeriodAbbreviation(t, &result, &firstWord,
                                        &phCodes) {
                consumed = true
            }
        }
        if !consumed {
            applyPosContextOverride(t, isIsolatedWord, phraseMatched,
                                    expectPast, expectNoun, expectVerb,
                                    &phCodes)
            applyAtstartOverride(t, firstWord, phraseMatched, &phCodes)
            applyAtendOverride(t, ti, lastWordTi, isIsolatedWord,
                               phraseMatched, &phCodes)
            applyLemmaOverride(t, ti, ts, isIsolatedWord, lastWordTi,
                               &phCodes)
            let isUnstressedWord = applyStepB(t, phraseMatched,
                                              isIsolatedWord, &phCodes)
            applyStepC(t, ti, ts, isIsolatedWord, phraseMatched,
                       lastWordTi, matchedPhraseKey, &phCodes)
            fixDiphthongStressPosition(&phCodes)
            applyInterWordTFlap(ti, ts, &phCodes)
            applyCrossWordSchwaRhotic(ti, isEnUs, isIsolatedWord, ts,
                                      &phCodes)
            var ipa = phonemesToIpa(phCodes)
            maybeAddDefaultStress(t, phraseMatched, isIsolatedWord,
                                  isUnstressedWord, phCodes, &ipa)
            applyPreVowelTheFixup(phrasePreVowelThe, &ipa)
            applyRLinking(firstWord, ti, ts, &ipa)
            applyCrossWordTFlap(t, ti, isEnUs, isIsolatedWord, ts, &ipa)
            if ipa.count > 0 {
                if !firstWord { result.append(asc(" ")) }
                result.append(contentsOf: ipa)
                firstWord = false
            }
        }
        updatePosContextCounters(t, &expectPast, &expectNoun,
                                 &expectVerb)
    }

    func phonemizeText(_ text: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        if loaded {
            let isEnUs = dialect == bs("en-us") || dialect == bs("en_us")
            let ts = tokenize(text, text.count)
            var firstWord = true
            var wordTokenCount = 0
            var lastWordTi = -1
            for tii in 0..<ts.count where ts[tii].isWord {
                wordTokenCount += 1
                lastWordTi = tii
            }
            let isIsolatedWord = wordTokenCount == 1
            var expectPast = 0
            var expectNoun = 0
            var expectVerb = 0
            var ti = 0
            while ti < ts.count {
                if ts[ti].isWord {
                    processWordToken(ts[ti], &ti, ts, isIsolatedWord,
                                     lastWordTi, isEnUs, &expectPast,
                                     &expectNoun, &expectVerb,
                                     &out, &firstWord)
                }
                ti += 1
            }
        }
        return out
    }

    init?(rulesPath: String, listPath: String, dialect d: String) {
        rulesetInit(&rules)
        dialect = bs(d)
        buildIpaOverrides()
        loaded = loadDictionary(listPath) && loadRules(rulesPath)
        if !loaded { return nil }
    }

    func phonemize(_ text: String) -> [UInt8] {
        phonemizeText(bs(text))
    }

    func phonemize(_ text: [UInt8]) -> [UInt8] {
        phonemizeText(text)
    }

    var errorMessage: String { err }
}

// getline semantics: a file not ending in a newline still yields its tail.
func getlines(_ bytes: [UInt8]) -> [[UInt8]] {
    var out = [[UInt8]]()
    var start = 0
    var i = 0
    while i < bytes.count {
        if bytes[i] == 0x0A {
            out.append(Array(bytes[start...i]))
            start = i + 1
        }
        i += 1
    }
    if start < bytes.count { out.append(Array(bytes[start...])) }
    return out
}
