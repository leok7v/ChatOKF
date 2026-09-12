public enum GrammarOp: Equatable, Sendable {
    case char
    case range
    case any
    case not
    case split
    case jmp
    case match
}

// `next` < 0 means "fall through to the following instruction"; `alt` is the
// SPLIT secondary successor.
public struct GrammarInst: Sendable {
    let op: GrammarOp
    let a: UInt8
    let b: UInt8
    let next: Int
    let alt: Int
}

public final class Grammar {
    public static let structuralMaxLive = 16

    private let prog: [GrammarInst]
    private var active: [Int]
    private var seen: [Int]
    private var gen: Int

    public init(_ program: [GrammarInst]) {
        prog = program
        seen = [Int](repeating: 0, count: max(program.count, 1))
        gen = 0
        active = []
        active = closure([0])
    }

    private func resolveNext(_ idx: Int) -> Int {
        prog[idx].next < 0 ? idx + 1 : prog[idx].next
    }

    private func isConsumer(_ op: GrammarOp) -> Bool {
        op == .char || op == .range || op == .any || op == .not
    }

    private func markPush(_ idx: Int, _ g: Int, _ stack: inout [Int]) {
        if idx >= 0 && idx < prog.count && seen[idx] != g {
            seen[idx] = g
            stack.append(idx)
        }
    }

    private func closure(_ seeds: [Int]) -> [Int] {
        if gen == Int.max {
            for i in 0..<seen.count { seen[i] = 0 }
            gen = 0
        }
        gen += 1
        let g = gen
        var stack: [Int] = []
        for idx in seeds { markPush(idx, g, &stack) }
        var out: [Int] = []
        while !stack.isEmpty {
            let idx = stack.removeLast()
            let inst = prog[idx]
            if inst.op == .split {
                markPush(resolveNext(idx), g, &stack)
                markPush(inst.alt, g, &stack)
            } else if inst.op == .jmp {
                markPush(resolveNext(idx), g, &stack)
            } else if isConsumer(inst.op) || inst.op == .match {
                out.append(idx)
            }
        }
        return out
    }

    private func matchByte(_ inst: GrammarInst, _ c: UInt8) -> Bool {
        switch inst.op {
        case .char: return c == inst.a
        case .range: return c >= inst.a && c <= inst.b
        case .any: return c != 0
        case .not: return c != inst.a
        default: return false
        }
    }

    // MATCH is absorbing: once the grammar is satisfied the remaining token
    // bytes stay unconstrained and the masking state stack stays complete.
    private func step(_ cur: [Int], _ c: UInt8) -> [Int] {
        var seeds: [Int] = []
        var matchIdx = -1
        for idx in cur {
            let inst = prog[idx]
            if inst.op == .match {
                matchIdx = idx
            } else if isConsumer(inst.op) && matchByte(inst, c) {
                seeds.append(resolveNext(idx))
            }
        }
        var out = closure(seeds)
        if matchIdx >= 0 && !out.contains(matchIdx) {
            out.append(matchIdx)
        }
        return out
    }

    private func simulate(_ s: [UInt8], commit: Bool) -> Bool {
        var cur = active
        var i = 0
        while i < s.count && !cur.isEmpty {
            cur = step(cur, s[i])
            i += 1
        }
        if commit { active = cur }
        return !s.isEmpty && i == s.count && !cur.isEmpty
    }

    public func accepts(_ s: [UInt8]) -> Bool {
        simulate(s, commit: false)
    }

    public func advance(_ s: [UInt8]) {
        _ = simulate(s, commit: true)
    }

    public var canStop: Bool {
        active.contains(where: { idx in prog[idx].op == .match })
    }

    public var dead: Bool {
        active.isEmpty
    }

    public var permissive: Bool {
        let probes: [UInt8] = [0x61, 0x42, 0x37, 0x7A, 0x51, 0x6D, 0x2E]
        return probes.allSatisfy { probe in
            simulate([probe], commit: false)
        }
    }

    private func commonPrefix(_ a: [UInt8], _ b: [UInt8]) -> Int {
        var i = 0
        while i < a.count && i < b.count && a[i] == b[i] { i += 1 }
        return i
    }

    private func maskBucket(_ vocab: GrammarVocab, _ lo: Int, _ hi: Int,
                            _ st: inout [[Int]],
                            _ logits: inout [Float]) {
        var prev: [UInt8] = []
        var valid = 0
        var k = lo
        while k < hi {
            let s = vocab.strs[k]
            let lcp = commonPrefix(prev, s)
            var d = min(lcp, valid)
            while d < s.count && !st[d].isEmpty {
                st[d + 1] = step(st[d], s[d])
                d += 1
            }
            let legal = d == s.count && !st[d].isEmpty
            if !legal { logits[vocab.ids[k]] = -Float.infinity }
            valid = d
            prev = s
            k += 1
        }
    }

    public func maskLogits(_ vocab: GrammarVocab,
                           _ logits: inout [Float],
                           structuralOnly: Bool = false) {
        let seed = active
        var live = [Bool](repeating: false, count: 256)
        var liveCount = 0
        var b = 0
        while b < 256 {
            let ok = !step(seed, UInt8(b)).isEmpty
            live[b] = ok
            if ok { liveCount += 1 }
            b += 1
        }
        if structuralOnly && liveCount > Grammar.structuralMaxLive { return }
        var st = [[Int]](repeating: [], count: vocab.maxLen + 1)
        st[0] = seed
        var c = 0
        while c < 256 {
            let lo = vocab.bucket[c]
            let hi = vocab.bucket[c + 1]
            // Byte 0 is the empty-token bucket: it stays on the walk so a
            // zero-length token, which leaves the state unchanged, is allowed.
            let deadRun = c != 0 && !live[c]
            if lo < hi && deadRun {
                var k = lo
                while k < hi {
                    logits[vocab.ids[k]] = -Float.infinity
                    k += 1
                }
            } else if lo < hi {
                maskBucket(vocab, lo, hi, &st, &logits)
            }
            c += 1
        }
    }

    public static func toolCall() -> Grammar {
        let b = GrammarBuilder()
        b.buildToolCall()
        return Grammar(b.program)
    }

    public static func toolCallNamed(_ names: [String]) -> Grammar {
        let b = GrammarBuilder()
        b.buildToolCallNamed(names)
        return Grammar(b.program)
    }

    public static func toolCallTyped(_ params: [GrammarParam]) -> Grammar {
        let b = GrammarBuilder()
        b.buildToolCallTyped(params)
        return Grammar(b.program)
    }
}

public enum GrammarParamType: Sendable {
    case string
    case integer
    case number
    case boolean
    case enumeration
    case free
}

public struct GrammarParam: Sendable {
    public let name: String
    public let type: GrammarParamType
    public let enumValues: [String]

    public init(name: String, type: GrammarParamType,
                enumValues: [String] = []) {
        self.name = name
        self.type = type
        self.enumValues = enumValues
    }
}

public final class GrammarBuilder {
    private var inst: [GrammarInst] = []

    public init() {}

    public var program: [GrammarInst] { inst }

    @discardableResult
    public func emit(_ op: GrammarOp, _ a: UInt8, _ b: UInt8,
                     _ next: Int, _ alt: Int) -> Int {
        let i = inst.count
        inst.append(GrammarInst(op: op, a: a, b: b, next: next, alt: alt))
        return i
    }

    // Patch a previously-emitted instruction's next / alt; -1 leaves a field
    // unchanged (so a -1 "fall through" successor is never overwritten).
    public func patch(_ idx: Int, _ next: Int, _ alt: Int) {
        if idx >= 0 && idx < inst.count {
            let cur = inst[idx]
            inst[idx] = GrammarInst(
                op: cur.op, a: cur.a, b: cur.b,
                next: next >= 0 ? next : cur.next,
                alt: alt >= 0 ? alt : cur.alt)
        }
    }

    @discardableResult
    public func lit(_ s: String) -> Int {
        let first = inst.count
        for byte in Array(s.utf8) { emit(.char, byte, 0, -1, -1) }
        return first
    }

    @discardableResult
    public func match() -> Int {
        emit(.match, 0, 0, -1, -1)
    }

    @discardableResult
    public func untilChar(_ c: UInt8) -> Int {
        let i = inst.count
        emit(.split, 0, 0, i + 1, i + 2)
        emit(.not, c, 0, i, -1)
        return i
    }

    @discardableResult
    public func wsStar() -> Int {
        let loop = emit(.split, 0, 0, -1, 0)
        let a1 = emit(.split, 0, 0, -1, 0)
        emit(.char, 0x20, 0, loop, -1)
        let a2 = emit(.split, 0, 0, -1, 0)
        emit(.char, 0x09, 0, loop, -1)
        let a3 = emit(.split, 0, 0, -1, 0)
        emit(.char, 0x0A, 0, loop, -1)
        emit(.char, 0x0D, 0, loop, -1)
        let after = inst.count
        patch(loop, -1, after)
        patch(a1, -1, a1 + 2)
        patch(a2, -1, a2 + 2)
        patch(a3, -1, a3 + 2)
        return loop
    }

    // A '<' is legal inside a value unless it starts '</', so code like a < b
    // survives while the first '</' still ends the value.
    @discardableResult
    public func value() -> Int {
        let i0 = emit(.split, 0, 0, -1, -1)
        let i1 = emit(.split, 0, 0, -1, -1)
        let i2 = emit(.not, 0x3C, 0, i0, -1)
        let i3 = emit(.char, 0x3C, 0, -1, -1)
        let i4 = emit(.not, 0x2F, 0, i0, -1)
        let exit = inst.count
        patch(i0, i1, exit)
        patch(i1, i2, i3)
        patch(i3, i4, -1)
        return i0
    }

    @discardableResult
    private func digits() -> Int {
        let d = emit(.range, 0x30, 0x39, -1, -1)
        let sp = emit(.split, 0, 0, d, -1)
        patch(sp, -1, sp + 1)
        return d
    }

    @discardableResult
    public func intCore() -> Int {
        let sp = emit(.split, 0, 0, -1, 0)
        emit(.char, 0x2D, 0, -1, -1)
        let digitsStart = inst.count
        patch(sp, -1, digitsStart)
        digits()
        return sp
    }

    private func oneOf2(_ lo: UInt8, _ hi: UInt8) {
        let sp = emit(.split, 0, 0, -1, -1)
        let a = emit(.char, lo, 0, -1, -1)
        emit(.char, hi, 0, -1, -1)
        let after = inst.count
        patch(sp, a, a + 1)
        patch(a, after, -1)
    }

    @discardableResult
    public func numberCore() -> Int {
        let start = intCore()
        let fsp = emit(.split, 0, 0, -1, 0)
        emit(.char, 0x2E, 0, -1, -1)
        digits()
        let afterFrac = inst.count
        patch(fsp, -1, afterFrac)
        let esp = emit(.split, 0, 0, -1, 0)
        oneOf2(0x65, 0x45)
        let ssp = emit(.split, 0, 0, -1, 0)
        oneOf2(0x2B, 0x2D)
        let expDigits = inst.count
        patch(ssp, -1, expDigits)
        digits()
        let afterExp = inst.count
        patch(esp, -1, afterExp)
        return start
    }

    @discardableResult
    public func boolCore() -> Int {
        let sp = emit(.split, 0, 0, -1, -1)
        let t = lit("true")
        let jmp = emit(.jmp, 0, 0, -1, -1)
        let f = lit("false")
        let after = inst.count
        patch(sp, t, f)
        patch(jmp, after, -1)
        return sp
    }

    private func alternation(_ count: Int,
                             _ emitArm: (Int) -> Void) -> Int {
        var first = inst.count
        if count > 0 {
            var jmps: [Int] = []
            first = -1
            var i = 0
            while i < count {
                var sp = -1
                if i < count - 1 { sp = emit(.split, 0, 0, -1, 0) }
                if first < 0 { first = sp >= 0 ? sp : inst.count }
                let arm = inst.count
                emitArm(i)
                jmps.append(emit(.jmp, 0, 0, -1, -1))
                if sp >= 0 {
                    patch(sp, arm, -1)
                    patch(sp, -1, inst.count)
                }
                i += 1
            }
            let merge = inst.count
            for j in jmps { patch(j, merge, -1) }
        }
        return first
    }

    @discardableResult
    public func altLits(_ lits: [String]) -> Int {
        alternation(lits.count) { index in self.lit(lits[index]) }
    }

    @discardableResult
    public func utf8Range(_ lo: UInt32, _ hi: UInt32) -> Int {
        let arms = utf8RangeArms(lo, hi)
        return alternation(arms.count) { index in
            for r in arms[index] { self.emit(.range, r.0, r.1, -1, -1) }
        }
    }

    private func utf8Encode(_ cp: UInt32) -> [UInt8] {
        var bytes: [UInt8] = []
        if cp <= 0x7F {
            bytes = [UInt8(cp)]
        } else if cp <= 0x7FF {
            bytes = [UInt8(0xC0 | (cp >> 6)),
                     UInt8(0x80 | (cp & 0x3F))]
        } else if cp <= 0xFFFF {
            bytes = [UInt8(0xE0 | (cp >> 12)),
                     UInt8(0x80 | ((cp >> 6) & 0x3F)),
                     UInt8(0x80 | (cp & 0x3F))]
        } else {
            bytes = [UInt8(0xF0 | (cp >> 18)),
                     UInt8(0x80 | ((cp >> 12) & 0x3F)),
                     UInt8(0x80 | ((cp >> 6) & 0x3F)),
                     UInt8(0x80 | (cp & 0x3F))]
        }
        return bytes
    }

    private func utf8RangeArms(_ lo: UInt32,
                               _ hi: UInt32) -> [[(UInt8, UInt8)]] {
        let bounds: [UInt32] = [0x7F, 0x7FF, 0xFFFF, 0x10FFFF]
        var arms: [[(UInt8, UInt8)]] = []
        var start = lo
        for maxv in bounds {
            if start <= hi && start <= maxv {
                let end = min(hi, maxv)
                arms += splitRanges(utf8Encode(start), utf8Encode(end))
                start = maxv + 1
            }
        }
        return arms
    }

    private func splitRanges(_ start: [UInt8],
                             _ end: [UInt8]) -> [[(UInt8, UInt8)]] {
        var arms: [[(UInt8, UInt8)]] = []
        if start.count == 1 {
            arms = [[(start[0], end[0])]]
        } else if start[0] == end[0] {
            let rest = splitRanges(Array(start[1...]), Array(end[1...]))
            arms = rest.map { tail in [(start[0], start[0])] + tail }
        } else {
            let n = start.count
            let maxTail = [UInt8](repeating: 0xBF, count: n - 1)
            let minTail = [UInt8](repeating: 0x80, count: n - 1)
            let armsA = splitRanges(Array(start[1...]), maxTail)
                .map { tail in [(start[0], start[0])] + tail }
            let armsC = splitRanges(minTail, Array(end[1...]))
                .map { tail in [(end[0], end[0])] + tail }
            var middle: [[(UInt8, UInt8)]] = []
            if start[0] + 1 <= end[0] - 1 {
                let full = [(UInt8, UInt8)](
                    repeating: (0x80, 0xBF), count: n - 1)
                middle = [[(start[0] + 1, end[0] - 1)] + full]
            }
            arms = armsA + middle + armsC
        }
        return arms
    }

    private func typedValue(_ prm: GrammarParam) {
        switch prm.type {
        case .integer:
            wsStar()
            intCore()
            wsStar()
        case .number:
            wsStar()
            numberCore()
            wsStar()
        case .boolean:
            wsStar()
            boolCore()
            wsStar()
        case .enumeration:
            wsStar()
            altLits(prm.enumValues)
            wsStar()
        case .string, .free:
            value()
        }
    }

    @discardableResult
    public func buildToolCallNamed(_ names: [String]) -> Int {
        inst.removeAll(keepingCapacity: true)
        lit("<tool_call>")
        wsStar()
        lit("<function=")
        if names.isEmpty { untilChar(0x3E) } else { altLits(names) }
        lit(">")
        let loop = wsStar()
        let sp = emit(.split, 0, 0, -1, 0)
        lit("<parameter=")
        untilChar(0x3E)
        lit(">")
        value()
        lit("</parameter>")
        emit(.jmp, 0, 0, loop, -1)
        let endb = inst.count
        patch(sp, -1, endb)
        lit("</function>")
        wsStar()
        lit("</tool_call>")
        return match()
    }

    @discardableResult
    public func buildToolCall() -> Int {
        buildToolCallNamed([])
    }

    @discardableResult
    public func buildToolCallTyped(_ params: [GrammarParam]) -> Int {
        inst.removeAll(keepingCapacity: true)
        lit("<tool_call>")
        wsStar()
        lit("<function=")
        untilChar(0x3E)
        lit(">")
        let loop = wsStar()
        let sp = emit(.split, 0, 0, -1, 0)
        if params.isEmpty {
            emitFreeParam(loop)
        } else {
            emitTypedParams(params, loop)
        }
        let endb = inst.count
        patch(sp, -1, endb)
        lit("</function>")
        wsStar()
        lit("</tool_call>")
        return match()
    }

    private func emitFreeParam(_ loop: Int) {
        lit("<parameter=")
        untilChar(0x3E)
        lit(">")
        value()
        lit("</parameter>")
        emit(.jmp, 0, 0, loop, -1)
    }

    private func emitTypedParams(_ params: [GrammarParam], _ loop: Int) {
        var i = 0
        while i < params.count {
            var asp = -1
            if i < params.count - 1 { asp = emit(.split, 0, 0, -1, 0) }
            let arm = inst.count
            lit("<parameter=")
            lit(params[i].name)
            lit(">")
            typedValue(params[i])
            lit("</parameter>")
            emit(.jmp, 0, 0, loop, -1)
            if asp >= 0 {
                patch(asp, arm, -1)
                patch(asp, -1, inst.count)
            }
            i += 1
        }
    }
}

public final class GrammarVocab: Sendable {
    let strs: [[UInt8]]
    let ids: [Int]
    let maxLen: Int
    let bucket: [Int]

    public init(_ tokens: [[UInt8]]) {
        var pairs = tokens.enumerated().map { pair in
            (bytes: pair.element, id: pair.offset)
        }
        pairs.sort { lhs, rhs in
            lhs.bytes.lexicographicallyPrecedes(rhs.bytes)
        }
        var s: [[UInt8]] = []
        var idArr: [Int] = []
        var maxL = 1
        for p in pairs {
            s.append(p.bytes)
            idArr.append(p.id)
            if p.bytes.count > maxL { maxL = p.bytes.count }
        }
        var bkt = [Int](repeating: 0, count: 257)
        for str in s {
            let fb = str.isEmpty ? 0 : Int(str[0])
            bkt[fb + 1] += 1
        }
        var c = 1
        while c <= 256 {
            bkt[c] += bkt[c - 1]
            c += 1
        }
        strs = s
        ids = idArr
        maxLen = maxL
        bucket = bkt
    }
}
