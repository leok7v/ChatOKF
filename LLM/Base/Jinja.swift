import Foundation

// `.undefined` is distinct from `.none`. `.node` dispatches on `tag`; the
// engine reserves two negative tags a host MUST NOT use.
public enum JinjaValue: Sendable {
    case none
    case undefined
    case bool(Bool)
    case int(Int)
    case str(String)
    case node(handle: Int, tag: Int)
    case list([JinjaValue])
}

public enum JinjaError: Error {
    case raise(String)
    case unsupported(String)
}

public protocol JinjaHost {
    func get(_ obj: JinjaValue, _ name: String) -> JinjaValue
    func index(_ obj: JinjaValue, _ i: Int) -> JinjaValue
    func len(_ obj: JinjaValue) -> Int
    func truthy(_ v: JinjaValue) -> Bool
    func test(_ v: JinjaValue, _ name: String) -> Bool
    func method(_ obj: JinjaValue, _ name: String,
                _ arg: JinjaValue) -> JinjaValue
}

public func jinjaRender(_ tmpl: String, host: JinjaHost,
                        vars: [(String, JinjaValue)]) throws -> String {
    let engine = JinjaEngine(tmpl, host)
    for (name, value) in vars {
        engine.vars[name] = value
    }
    return try engine.render()
}

private enum TokKind {
    case end
    case id
    case str
    case int
    case op
}

private enum Stop {
    case none
    case elif
    case elseArm
    case endif
    case endfor
    case endmacro
    case endset
    case endfilter
    case eof
}

private final class Box {
    var entries: [(String, JinjaValue)] = []
}

private struct Param {
    let name: String
    let defPos: Int?
}

private struct Macro {
    let name: String
    let params: [Param]
    let body: Int
}

private struct CallArgs {
    var pos: [JinjaValue] = []
    var kw: [(String, JinjaValue)] = []
}

private struct Lexeme {
    let kind: TokKind
    let text: String
    let iv: Int
    let p: Int
    let ts: Int
}

// A backstop for a recursive macro an unsupported feature would not terminate.
private let maxDepth = 512
private let dictTag = -31337
private let builtinTag = -31338

private let bLbrace: UInt8 = 0x7B
private let bRbrace: UInt8 = 0x7D
private let bPercent: UInt8 = 0x25
private let bHash: UInt8 = 0x23
private let bDash: UInt8 = 0x2D
private let bSquote: UInt8 = 0x27
private let bDquote: UInt8 = 0x22
private let bBackslash: UInt8 = 0x5C
private let bEq: UInt8 = 0x3D
private let bBang: UInt8 = 0x21
private let bLt: UInt8 = 0x3C
private let bGt: UInt8 = 0x3E
private let bStar: UInt8 = 0x2A
private let bSlash: UInt8 = 0x2F

private final class JinjaEngine {
    private let host: JinjaHost
    private let src: [UInt8]
    var vars: [String: JinjaValue] = [:]
    private var dicts: [Box] = []
    private var macros: [Macro] = []
    private var out: [UInt8] = []
    private var kind = TokKind.end
    private var text = ""
    private var iv = 0
    private var tokenStart = 0
    private var p = 0
    private var trimLeft = false
    private var brace = 0
    private var mute = 0
    private var depth = 0

    init(_ tmpl: String, _ host: JinjaHost) {
        self.host = host
        var bytes = Array(tmpl.utf8)
        // Jinja strips one trailing template newline by default.
        if bytes.last == 0x0A {
            bytes.removeLast()
        }
        self.src = bytes
    }

    func render() throws -> String {
        p = 0
        _ = try renderBlock()
        return String(decoding: out, as: UTF8.self)
    }

    private func at(_ i: Int) -> UInt8 {
        i >= 0 && i < src.count ? src[i] : 0
    }

    private func isSpace(_ c: UInt8) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D
    }

    private func isDigit(_ c: UInt8) -> Bool {
        c >= 0x30 && c <= 0x39
    }

    private func isIdentStart(_ c: UInt8) -> Bool {
        (c >= 0x61 && c <= 0x7A) || (c >= 0x41 && c <= 0x5A) || c == 0x5F
    }

    private func isIdentCont(_ c: UInt8) -> Bool {
        isIdentStart(c) || isDigit(c)
    }

    private func fail(_ msg: String) throws -> Never {
        throw JinjaError.raise(msg)
    }

    // A no-op in a muted branch: Jinja resolves filters at run time, so one
    // that never executes is not an error. On a live path it refuses loudly.

    private func unsupported(_ what: String, _ name: String) throws {
        if mute == 0 {
            throw JinjaError.unsupported("\(what): \(name)")
        }
    }

    private func emit(_ s: String) {
        if mute == 0 {
            out.append(contentsOf: s.utf8)
        }
    }

    private func emitBytes(_ lo: Int, _ hi: Int) {
        if mute == 0 && hi > lo {
            out.append(contentsOf: src[lo..<hi])
        }
    }

    private func skipWs() {
        while isSpace(at(p)) {
            p += 1
        }
    }

    private func lex() {
        skipWs()
        tokenStart = p
        let c = at(p)
        let c1 = at(p + 1)
        if isCloseMarker(c, c1) {
            kind = .end
            text = ""
        } else if c == bSquote || c == bDquote {
            lexString(c)
        } else if isDigit(c) {
            lexInt()
        } else if isIdentStart(c) {
            lexIdent()
        } else {
            lexOp(c, c1)
        }
    }

    private func isCloseMarker(_ c: UInt8, _ c1: UInt8) -> Bool {
        (c == bDash && (c1 == bRbrace ||
                        (c1 == bPercent && at(p + 2) == bRbrace))) ||
        (c == bRbrace && brace == 0) ||
        (c == bPercent && c1 == bRbrace)
    }

    private func lexString(_ quote: UInt8) {
        var e = p + 1
        while at(e) != 0 && at(e) != quote {
            e += (at(e) == bBackslash && at(e + 1) != 0) ? 2 : 1
        }
        text = unescape(p + 1, e)
        kind = .str
        p = at(e) != 0 ? e + 1 : e
    }

    private func unescape(_ lo: Int, _ hi: Int) -> String {
        var bytes: [UInt8] = []
        var i = lo
        while i < hi {
            var c = src[i]
            if c == bBackslash && i + 1 < hi {
                i += 1
                let e = src[i]
                c = e == 0x6E ? 0x0A : e == 0x74 ? 0x09
                  : e == 0x72 ? 0x0D : e
            }
            bytes.append(c)
            i += 1
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func lexInt() {
        var e = p
        var val = 0
        while isDigit(at(e)) {
            val = val * 10 + Int(at(e) - 0x30)
            e += 1
        }
        iv = val
        kind = .int
        p = e
    }

    private func lexIdent() {
        var e = p
        while isIdentCont(at(e)) {
            e += 1
        }
        text = String(decoding: src[p..<e], as: UTF8.self)
        kind = .id
        p = e
    }

    private func lexOp(_ c: UInt8, _ c1: UInt8) {
        let two = (c1 == bEq && (c == bEq || c == bBang ||
                                 c == bLt || c == bGt)) ||
                  (c == bStar && c1 == bStar) ||
                  (c == bSlash && c1 == bSlash)
        let n = two ? 2 : 1
        let hi = min(p + n, src.count)
        text = String(decoding: src[p..<hi], as: UTF8.self)
        kind = c != 0 ? .op : .end
        p += n
    }

    private func isId(_ w: String) -> Bool {
        kind == .id && text == w
    }

    private func isOp(_ w: String) -> Bool {
        kind == .op && text == w
    }

    private func snapshot() -> Lexeme {
        Lexeme(kind: kind, text: text, iv: iv, p: p, ts: tokenStart)
    }

    private func restore(_ s: Lexeme) {
        kind = s.kind
        text = s.text
        iv = s.iv
        p = s.p
        tokenStart = s.ts
    }

    private func prime(_ after: Int) {
        p = after
        lex()
    }

    private func consumeClose() {
        let dash = at(tokenStart) == bDash
        trimLeft = dash
        p = tokenStart + (dash ? 3 : 2)
    }

    private func isStr(_ v: JinjaValue) -> Bool {
        var result = false
        if case .str = v { result = true }
        return result
    }

    private func isNone(_ v: JinjaValue) -> Bool {
        var result = false
        if case .none = v { result = true }
        return result
    }

    private func isUndef(_ v: JinjaValue) -> Bool {
        var result = false
        if case .undefined = v { result = true }
        return result
    }

    private func isInt(_ v: JinjaValue) -> Bool {
        var result = false
        if case .int = v { result = true }
        return result
    }

    private func isDict(_ v: JinjaValue) -> Bool {
        var result = false
        if case .node(_, dictTag) = v { result = true }
        return result
    }

    private func isHostNode(_ v: JinjaValue) -> Bool {
        var result = false
        if case .node(_, let t) = v {
            result = t != dictTag && t != builtinTag
        }
        return result
    }

    private func boolValue(_ v: JinjaValue) -> Bool? {
        var result: Bool? = nil
        if case .bool(let b) = v { result = b }
        return result
    }

    private func intOf(_ v: JinjaValue) -> Int {
        switch v {
        case .int(let n): return n
        case .bool(let b): return b ? 1 : 0
        default: return 0
        }
    }

    private func asStr(_ v: JinjaValue) -> String {
        if case .str(let s) = v { return s }
        return ""
    }

    private func toStr(_ v: JinjaValue) -> String {
        switch v {
        case .str(let s): return s
        case .int(let n): return String(n)
        case .bool(let b): return b ? "True" : "False"
        case .none: return "None"
        default: return ""
        }
    }

    private func truthy(_ v: JinjaValue) -> Bool {
        switch v {
        case .bool(let b): return b
        case .int(let n): return n != 0
        case .str(let s): return !s.isEmpty
        case .list(let a): return !a.isEmpty
        case .node(let h, let t):
            return t == dictTag ? !dicts[h].entries.isEmpty
                 : t == builtinTag ? true : host.truthy(v)
        default: return false
        }
    }

    private func sameKind(_ a: JinjaValue, _ b: JinjaValue) -> Bool {
        switch (a, b) {
        case (.none, .none): return true
        case (.undefined, .undefined): return true
        default: return false
        }
    }

    private func jvEq(_ a: JinjaValue, _ b: JinjaValue) -> Bool {
        var result = false
        if isStr(a) || isStr(b) {
            result = isStr(a) && isStr(b) && asStr(a) == asStr(b)
        } else if isNone(a) || isUndef(a) || isNone(b) || isUndef(b) {
            result = sameKind(a, b)
        } else {
            result = intOf(a) == intOf(b)
        }
        return result
    }

    private func cmpJv(_ a: JinjaValue, _ b: JinjaValue) -> Int {
        var result = 0
        if case .int(let x) = a, case .int(let y) = b {
            result = x < y ? -1 : (x > y ? 1 : 0)
        } else {
            let sa = toStr(a)
            let sb = toStr(b)
            result = sa < sb ? -1 : (sa > sb ? 1 : 0)
        }
        return result
    }

    private func newDict() -> JinjaValue {
        dicts.append(Box())
        return .node(handle: dicts.count - 1, tag: dictTag)
    }

    private func dictPut(_ dv: JinjaValue, _ key: String,
                         _ val: JinjaValue) {
        if case .node(let h, _) = dv {
            let hit = dicts[h].entries.firstIndex(where: { e in
                e.0 == key
            })
            if let hit {
                dicts[h].entries[hit].1 = val
            } else {
                dicts[h].entries.append((key, val))
            }
        }
    }

    private func dictGet(_ dv: JinjaValue, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        if case .node(let h, _) = dv {
            let hit = dicts[h].entries.lastIndex(where: { e in
                e.0 == name
            })
            if let hit {
                result = dicts[h].entries[hit].1
            }
        }
        return result
    }

    private func getAttr(_ obj: JinjaValue, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        if isDict(obj) {
            result = dictGet(obj, name)
        } else if isHostNode(obj) {
            result = host.get(obj, name)
        }
        return result
    }

    private func seqLen(_ v: JinjaValue) -> Int {
        switch v {
        case .list(let a): return a.count
        case .str(let s): return s.utf8.count
        case .node(let h, let t):
            return t == dictTag ? dicts[h].entries.count
                 : t == builtinTag ? 0 : host.len(v)
        default: return 0
        }
    }

    private func byteStr(_ s: String, _ i: Int) -> String {
        let bytes = Array(s.utf8)
        return String(decoding: [bytes[i]], as: UTF8.self)
    }

    private func seqIndex(_ v: JinjaValue, _ i: Int) -> JinjaValue {
        var result = JinjaValue.undefined
        let n = seqLen(v)
        var idx = i
        if idx < 0 {
            idx += n
        }
        if idx >= 0 && idx < n {
            switch v {
            case .list(let a): result = a[idx]
            case .str(let s): result = .str(byteStr(s, idx))
            case .node(let h, let t):
                result = t == dictTag
                    ? .str(dicts[h].entries[idx].0) : host.index(v, idx)
            default: break
            }
        }
        return result
    }

    private func listItems(_ v: JinjaValue) -> [JinjaValue] {
        var result: [JinjaValue] = []
        if case .list(let a) = v { result = a }
        return result
    }

    private func contains(_ c: JinjaValue, _ item: JinjaValue) -> Bool {
        var result = false
        if case .str(let s) = c {
            result = s.range(of: toStr(item)) != nil
        } else {
            let n = seqLen(c)
            var i = 0
            while i < n && !result {
                result = jvEq(seqIndex(c, i), item)
                i += 1
            }
        }
        return result
    }

    private func materialize(_ v: JinjaValue) -> JinjaValue {
        var result = v
        if case .list = v {
        } else {
            let n = seqLen(v)
            var items: [JinjaValue] = []
            var i = 0
            while i < n {
                items.append(seqIndex(v, i))
                i += 1
            }
            result = .list(items)
        }
        return result
    }

    private func reverseSeq(_ v: JinjaValue) -> JinjaValue {
        let n = seqLen(v)
        var items: [JinjaValue] = []
        var k = 0
        while k < n {
            items.append(seqIndex(v, n - 1 - k))
            k += 1
        }
        return .list(items)
    }

    private func pairKey(_ v: JinjaValue) -> String {
        let items = listItems(v)
        return items.isEmpty ? "" : asStr(items[0])
    }

    private func toPairs(_ v: JinjaValue, _ sorted: Bool) -> JinjaValue {
        var pairs: [JinjaValue] = []
        if isDict(v), case .node(let h, _) = v {
            for entry in dicts[h].entries {
                pairs.append(.list([.str(entry.0), entry.1]))
            }
        } else if isHostNode(v) {
            let m = host.len(v)
            var i = 0
            while i < m {
                let key = host.index(v, i)
                pairs.append(.list([key, getAttr(v, asStr(key))]))
                i += 1
            }
        }
        return .list(sorted ? sortPairsByKey(pairs) : pairs)
    }

    private func sortPairsByKey(_ input: [JinjaValue]) -> [JinjaValue] {
        var pairs = input
        var i = 1
        while i < pairs.count {
            let cur = pairs[i]
            let key = pairKey(cur)
            var j = i - 1
            while j >= 0 && pairKey(pairs[j]) > key {
                pairs[j + 1] = pairs[j]
                j -= 1
            }
            pairs[j + 1] = cur
            i += 1
        }
        return pairs
    }

    private func sortKey(_ it: JinjaValue,
                         _ attr: JinjaValue) -> JinjaValue {
        if case .str(let s) = attr { return getAttr(it, s) }
        return it
    }

    private func filterSort(_ v: JinjaValue, _ attr: JinjaValue,
                            _ rev: Bool) -> JinjaValue {
        var items = listItems(materialize(v))
        var i = 1
        while i < items.count {
            let cur = items[i]
            let ck = sortKey(cur, attr)
            var j = i - 1
            while j >= 0 && sortShift(sortKey(items[j], attr), ck, rev) {
                items[j + 1] = items[j]
                j -= 1
            }
            items[j + 1] = cur
            i += 1
        }
        return .list(items)
    }

    private func sortShift(_ a: JinjaValue, _ b: JinjaValue,
                           _ rev: Bool) -> Bool {
        rev ? cmpJv(a, b) < 0 : cmpJv(a, b) > 0
    }

    private func filterUnique(_ v: JinjaValue) -> JinjaValue {
        let n = seqLen(v)
        var items: [JinjaValue] = []
        var i = 0
        while i < n {
            let e = seqIndex(v, i)
            let seen = items.contains(where: { x in jvEq(x, e) })
            if !seen { items.append(e) }
            i += 1
        }
        return .list(items)
    }

    private func filterMinMax(_ v: JinjaValue,
                              _ wantMax: Bool) -> JinjaValue {
        let n = seqLen(v)
        var result = n > 0 ? seqIndex(v, 0) : JinjaValue.undefined
        var i = 1
        while i < n {
            let e = seqIndex(v, i)
            let c = cmpJv(e, result)
            if wantMax ? c > 0 : c < 0 { result = e }
            i += 1
        }
        return result
    }

    private func filterSum(_ v: JinjaValue) -> Int {
        let n = seqLen(v)
        var s = 0
        var i = 0
        while i < n {
            if case .int(let x) = seqIndex(v, i) { s += x }
            i += 1
        }
        return s
    }

    private func joinSeq(_ v: JinjaValue, _ sep: String) -> JinjaValue {
        let n = seqLen(v)
        var parts: [String] = []
        var i = 0
        while i < n {
            parts.append(toStr(seqIndex(v, i)))
            i += 1
        }
        return .str(parts.joined(separator: sep))
    }

    private func mapSeq(_ v: JinjaValue,
                        _ ca: CallArgs) throws -> JinjaValue {
        let attr = kwGet(ca, "attribute")
        let fn = ca.pos.count > 0 ? asStr(ca.pos[0]) : ""
        let n = seqLen(v)
        var items: [JinjaValue] = []
        var i = 0
        while i < n {
            let e = seqIndex(v, i)
            if case .str(let a) = attr {
                items.append(getAttr(e, a))
            } else {
                items.append(try applyFilter(e, fn, CallArgs()))
            }
            i += 1
        }
        return .list(items)
    }

    private func filterSelect(_ v: JinjaValue, _ ca: CallArgs,
                              _ byattr: Bool,
                              _ rej: Bool) throws -> JinjaValue {
        let attr = byattr && ca.pos.count > 0
            ? ca.pos[0] : JinjaValue.undefined
        let ti = byattr ? 1 : 0
        let hasTst = ca.pos.count > ti
        let tst = hasTst ? ca.pos[ti] : JinjaValue.undefined
        let hasArg = ca.pos.count > ti + 1
        let targ = hasArg ? ca.pos[ti + 1] : JinjaValue.undefined
        let n = seqLen(v)
        var items: [JinjaValue] = []
        var i = 0
        while i < n {
            let e = seqIndex(v, i)
            let t = sortKey(e, attr)
            let keep = hasTst
                ? try doTest(t, asStr(tst), targ, hasArg) : truthy(t)
            if keep != rej { items.append(e) }
            i += 1
        }
        return .list(items)
    }

    private func filterGroupby(_ v: JinjaValue,
                               _ attr: JinjaValue) -> JinjaValue {
        let sorted = listItems(filterSort(v, attr, false))
        var groups: [JinjaValue] = []
        var i = 0
        while i < sorted.count {
            let k = sortKey(sorted[i], attr)
            var grp: [JinjaValue] = []
            while i < sorted.count &&
                  cmpJv(sortKey(sorted[i], attr), k) == 0 {
                grp.append(sorted[i])
                i += 1
            }
            groups.append(.list([k, .list(grp)]))
        }
        return .list(groups)
    }

    private func stripStr(_ s: String, _ set: String, _ left: Bool,
                          _ right: Bool) -> String {
        let setChars = Set(set)
        let chars = Array(s)
        var a = 0
        var b = chars.count
        if left {
            while a < b && setChars.contains(chars[a]) { a += 1 }
        }
        if right {
            while b > a && setChars.contains(chars[b - 1]) { b -= 1 }
        }
        return String(chars[a..<b])
    }

    private func replaceStr(_ s: String, _ from: String,
                            _ to: String) -> String {
        from.isEmpty ? s : s.replacingOccurrences(of: from, with: to)
    }

    private func splitStr(_ s: String, _ sep: String) -> JinjaValue {
        var items: [JinjaValue] = []
        if !sep.isEmpty {
            var rest = Substring(s)
            var more = true
            while more {
                if let r = rest.range(of: sep) {
                    let head = rest[rest.startIndex..<r.lowerBound]
                    items.append(.str(String(head)))
                    rest = rest[r.upperBound...]
                } else {
                    items.append(.str(String(rest)))
                    more = false
                }
            }
        }
        return .list(items)
    }

    private func upcase(_ c: Character) -> Character {
        String(c).uppercased().first ?? c
    }

    private func downcase(_ c: Character) -> Character {
        String(c).lowercased().first ?? c
    }

    private func isAlnumChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber
    }

    private func strTitle(_ s: String) -> String {
        var chars = Array(s)
        var start = true
        var i = 0
        while i < chars.count {
            chars[i] = start ? upcase(chars[i]) : downcase(chars[i])
            start = !isAlnumChar(chars[i])
            i += 1
        }
        return String(chars)
    }

    private func capitalizeStr(_ s: String) -> String {
        var chars = Array(s)
        if !chars.isEmpty { chars[0] = upcase(chars[0]) }
        return String(chars)
    }

    private func parseLeadingInt(_ s: String) -> Int {
        let chars = Array(s.utf8)
        var i = 0
        var sign = 1
        if i < chars.count && (chars[i] == bDash || chars[i] == 0x2B) {
            sign = chars[i] == bDash ? -1 : 1
            i += 1
        }
        var val = 0
        while i < chars.count && isDigit(chars[i]) {
            val = val * 10 + Int(chars[i] - 0x30)
            i += 1
        }
        return sign * val
    }

    private func toInt(_ v: JinjaValue) -> Int {
        var result = 0
        if case .int(let n) = v {
            result = n
        } else {
            result = parseLeadingInt(toStr(v))
        }
        return result
    }

    private func defaultFilter(_ v: JinjaValue,
                               _ ca: CallArgs) -> JinjaValue {
        let a0 = ca.pos.count > 0 ? ca.pos[0] : JinjaValue.undefined
        let falsy = ca.pos.count > 1 && truthy(ca.pos[1])
        let rep = falsy ? !truthy(v) : isUndef(v)
        return (rep && ca.pos.count > 0) ? a0 : v
    }

    private func jsonStr(_ s: String) {
        emit("\"")
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": emit("\\\"")
            case "\\": emit("\\\\")
            case "\n": emit("\\n")
            case "\r": emit("\\r")
            case "\t": emit("\\t")
            default: emit(String(ch))
            }
        }
        emit("\"")
    }

    private func jsonVal(_ v: JinjaValue) {
        switch v {
        case .str(let s): jsonStr(s)
        case .int(let n): emit(String(n))
        case .bool(let b): emit(b ? "true" : "false")
        case .none, .undefined: emit("null")
        case .list(let a):
            emit("[")
            var i = 0
            while i < a.count {
                if i > 0 { emit(", ") }
                jsonVal(a[i])
                i += 1
            }
            emit("]")
        case .node: emit("")
        }
    }

    private func toJson(_ v: JinjaValue) -> JinjaValue {
        var result = JinjaValue.str("")
        if isHostNode(v) {
            result = host.method(v, "tojson", .none)
        } else {
            let mark = out.count
            jsonVal(v)
            result = .str(String(decoding: out[mark...], as: UTF8.self))
            out.removeLast(out.count - mark)
        }
        return result
    }

    private func applyFilter(_ v: JinjaValue, _ nm: String,
                             _ ca: CallArgs) throws -> JinjaValue {
        var result = v
        let a0 = ca.pos.count > 0 ? ca.pos[0] : JinjaValue.undefined
        switch nm {
        case "trim": result = .str(stripStr(toStr(v), " \t\n\r", true, true))
        case "length", "count": result = .int(seqLen(v))
        case "tojson": result = toJson(v)
        case "safe": result = v
        case "string": result = .str(toStr(v))
        case "upper": result = .str(toStr(v).uppercased())
        case "lower": result = .str(toStr(v).lowercased())
        case "title": result = .str(strTitle(toStr(v)))
        case "capitalize": result = .str(capitalizeStr(toStr(v)))
        case "int": result = .int(toInt(v))
        case "abs": result = .int(abs(intOf(v)))
        case "default": result = defaultFilter(v, ca)
        case "first": result = seqIndex(v, 0)
        case "last": result = seqIndex(v, seqLen(v) - 1)
        case "dictsort": result = toPairs(v, true)
        case "items": result = toPairs(v, false)
        case "list": result = materialize(v)
        case "join":
            result = joinSeq(v, ca.pos.count > 0 ? asStr(a0) : "")
        case "map": result = try mapSeq(v, ca)
        case "replace":
            result = ca.pos.count > 1
                ? .str(replaceStr(toStr(v), asStr(a0), asStr(ca.pos[1])))
                : v
        case "reverse": result = reverseSeq(v)
        case "sort":
            result = filterSort(v, kwGet(ca, "attribute"),
                                truthy(kwGet(ca, "reverse")))
        case "unique": result = filterUnique(v)
        case "min": result = filterMinMax(v, false)
        case "max": result = filterMinMax(v, true)
        case "sum": result = .int(filterSum(v))
        case "select": result = try filterSelect(v, ca, false, false)
        case "reject": result = try filterSelect(v, ca, false, true)
        case "selectattr": result = try filterSelect(v, ca, true, false)
        case "rejectattr": result = try filterSelect(v, ca, true, true)
        case "groupby": result = filterGroupby(v, a0)
        default: try unsupported("filter", nm)
        }
        return result
    }

    private func callMethod(_ obj: JinjaValue, _ nm: String,
                            _ args: [JinjaValue]) throws -> JinjaValue {
        var result = JinjaValue.undefined
        let s = asStr(obj)
        let a0 = args.count > 0 ? asStr(args[0]) : ""
        switch nm {
        case "startswith": result = .bool(s.hasPrefix(a0))
        case "endswith": result = .bool(s.hasSuffix(a0))
        case "split": result = splitStr(s, args.count > 0 ? a0 : " ")
        case "lstrip":
            result = .str(stripStr(s, methodSet(args), true, false))
        case "rstrip":
            result = .str(stripStr(s, methodSet(args), false, true))
        case "strip":
            result = .str(stripStr(s, methodSet(args), true, true))
        case "replace":
            result = args.count > 1
                ? .str(replaceStr(s, a0, asStr(args[1]))) : obj
        case "items": result = toPairs(obj, false)
        case "get":
            let g = getAttr(obj, a0)
            result = isUndef(g) ? (args.count > 1 ? args[1] : .none) : g
        case "keys": result = materialize(obj)
        default:
            if isHostNode(obj) {
                result = host.method(obj, nm,
                                     args.count > 0 ? args[0] : .none)
            } else {
                try unsupported("method", nm)
            }
        }
        return result
    }

    private func methodSet(_ args: [JinjaValue]) -> String {
        args.count > 0 ? asStr(args[0]) : " \t\n\r"
    }

    private func isMapping(_ v: JinjaValue) -> Bool {
        var result = isDict(v)
        if !result && isHostNode(v) {
            result = host.test(v, "mapping")
        }
        return result
    }

    private func isSequence(_ v: JinjaValue) -> Bool {
        var result = false
        switch v {
        case .list: result = true
        case .str: result = true
        default:
            if isHostNode(v) { result = host.test(v, "sequence") }
        }
        return result
    }

    private func typeTest(_ v: JinjaValue, _ name: String) -> Bool? {
        switch name {
        case "string": return isStr(v)
        case "none": return isNone(v)
        case "undefined": return isUndef(v)
        case "defined": return !isUndef(v)
        case "true": return boolValue(v) == true
        case "false": return boolValue(v) == false
        case "boolean": return boolValue(v) != nil
        case "number", "integer": return isInt(v)
        case "mapping": return isMapping(v)
        case "iterable", "sequence": return isSequence(v)
        case "even": return isInt(v) && intOf(v) % 2 == 0
        case "odd": return isInt(v) && intOf(v) % 2 != 0
        default: return nil
        }
    }

    private func isComparisonName(_ name: String) -> Bool {
        switch name {
        case "eq", "ne", "lt", "gt", "le", "ge", "equalto", "lessthan",
             "greaterthan", "lessthanorequalto", "greaterthanorequalto":
            return true
        default:
            return false
        }
    }

    private func compareTest(_ v: JinjaValue, _ name: String,
                             _ arg: JinjaValue,
                             _ hasArg: Bool) throws -> Bool {
        var result = false
        if !hasArg && isComparisonName(name) {
            try unsupported("test", name)
        } else if name == "equalto" || name == "eq" {
            result = jvEq(v, arg)
        } else if name == "ne" {
            result = !jvEq(v, arg)
        } else if name == "lessthan" || name == "lt" {
            result = cmpJv(v, arg) < 0
        } else if name == "greaterthan" || name == "gt" {
            result = cmpJv(v, arg) > 0
        } else if name == "lessthanorequalto" || name == "le" {
            result = cmpJv(v, arg) <= 0
        } else if name == "greaterthanorequalto" || name == "ge" {
            result = cmpJv(v, arg) >= 0
        } else if isHostNode(v) {
            result = host.test(v, name)
        } else {
            try unsupported("test", name)
        }
        return result
    }

    private func doTest(_ v: JinjaValue, _ name: String,
                        _ arg: JinjaValue,
                        _ hasArg: Bool) throws -> Bool {
        var result = false
        if let r = typeTest(v, name) {
            result = r
        } else {
            result = try compareTest(v, name, arg, hasArg)
        }
        return result
    }

    private func kwGet(_ ca: CallArgs, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        for pair in ca.kw where pair.0 == name {
            result = pair.1
        }
        return result
    }

    private func isBuiltinGlobal(_ nm: String) -> Bool {
        nm == "strftime_now" || nm == "range" || nm == "namespace" ||
        nm == "raise_exception"
    }

    private func findMacro(_ nm: String) -> Macro? {
        macros.last(where: { m in m.name == nm })
    }

    private func takeKeyword() -> String? {
        var name: String? = nil
        if kind == .id {
            let saved = snapshot()
            let nm = text
            lex()
            if isOp("=") {
                lex()
                name = nm
            } else {
                restore(saved)
            }
        }
        return name
    }

    private func parseCall() throws -> CallArgs {
        var ca = CallArgs()
        lex()
        while !isOp(")") && kind != .end {
            if let name = takeKeyword() {
                ca.kw.append((name, try evalExpr()))
            } else {
                ca.pos.append(try evalExpr())
            }
            if isOp(",") { lex() }
        }
        if isOp(")") { lex() }
        return ca
    }

    private func parseArgs() throws -> [JinjaValue] {
        var args: [JinjaValue] = []
        lex()
        while !isOp(")") && kind != .end {
            args.append(try evalExpr())
            if isOp(",") { lex() }
        }
        if isOp(")") { lex() }
        return args
    }

    private func newNamespace(_ ca: CallArgs) -> JinjaValue {
        let d = newDict()
        for pair in ca.kw {
            dictPut(d, pair.0, pair.1)
        }
        return d
    }

    private func rangeFn(_ ca: CallArgs) throws -> JinjaValue {
        let a = ca.pos.count >= 2 ? intOf(ca.pos[0]) : 0
        let b = ca.pos.count >= 2 ? intOf(ca.pos[1])
              : (ca.pos.count >= 1 ? intOf(ca.pos[0]) : 0)
        let step = ca.pos.count >= 3 ? intOf(ca.pos[2]) : 1
        if step == 0 { try fail("range() step cannot be zero") }
        var cnt = 0
        if step > 0 && b > a { cnt = (b - a + step - 1) / step }
        if step < 0 && b < a { cnt = (a - b - step - 1) / -step }
        var items: [JinjaValue] = []
        var x = a
        var w = 0
        while w < cnt {
            items.append(.int(x))
            x += step
            w += 1
        }
        return .list(items)
    }

    private func strftimeNow(_ fmt: String) -> String {
        let cal = Calendar.current
        let c = cal.dateComponents(
            [.year, .month, .day, .hour, .minute, .second], from: Date())
        var out = ""
        let chars = Array(fmt)
        var i = 0
        while i < chars.count {
            if chars[i] == "%" && i + 1 < chars.count {
                out += strftimeField(chars[i + 1], c)
                i += 2
            } else {
                out.append(chars[i])
                i += 1
            }
        }
        return out
    }

    private func pad2(_ n: Int) -> String {
        n < 10 ? "0\(n)" : "\(n)"
    }

    private func strftimeField(_ code: Character,
                               _ c: DateComponents) -> String {
        switch code {
        case "Y": return "\(c.year ?? 0)"
        case "y": return pad2((c.year ?? 0) % 100)
        case "m": return pad2(c.month ?? 0)
        case "d": return pad2(c.day ?? 0)
        case "H": return pad2(c.hour ?? 0)
        case "M": return pad2(c.minute ?? 0)
        case "S": return pad2(c.second ?? 0)
        default: return "%\(code)"
        }
    }

    private func callNamed(_ nm: String) throws -> JinjaValue {
        let ca = try parseCall()
        var result = JinjaValue.undefined
        if nm == "namespace" {
            result = newNamespace(ca)
        } else if let m = findMacro(nm) {
            result = try callMacro(m, ca)
        } else if nm == "raise_exception" {
            if mute == 0 {
                try fail(ca.pos.count > 0
                         ? asStr(ca.pos[0]) : "raise_exception")
            }
        } else if nm == "range" {
            result = try rangeFn(ca)
        } else if nm == "strftime_now" {
            result = .str(strftimeNow(ca.pos.count > 0
                                      ? asStr(ca.pos[0]) : "%Y-%m-%d"))
        } else {
            try unsupported("function", nm)
        }
        return result
    }

    // In a muted branch running the body would recurse without bound in a
    // self-recursive macro; the arguments were already evaluated by parseCall.

    private func callMacro(_ m: Macro,
                           _ ca: CallArgs) throws -> JinjaValue {
        var result = JinjaValue.str("")
        if mute == 0 {
            result = try runMacro(m, ca)
        }
        return result
    }

    private func bindParam(_ pr: Param, _ i: Int,
                           _ ca: CallArgs) throws -> JinjaValue {
        var result = JinjaValue.undefined
        if i < ca.pos.count {
            result = ca.pos[i]
        } else if let kw = ca.kw.first(where: { pair in
            pair.0 == pr.name
        }) {
            result = kw.1
        } else if let def = pr.defPos {
            let back = p
            p = def
            lex()
            result = try evalExpr()
            p = back
        }
        return result
    }

    private func runMacro(_ m: Macro,
                          _ ca: CallArgs) throws -> JinjaValue {
        let savedLoop = vars["loop"]
        let savedLex = snapshot()
        let savedTrim = trimLeft
        var savedParams: [JinjaValue?] = []
        for i in 0..<m.params.count {
            savedParams.append(vars[m.params[i].name])
            vars[m.params[i].name] = try bindParam(m.params[i], i, ca)
        }
        let mark = out.count
        p = m.body
        trimLeft = false
        _ = try renderBlock()
        let cap = String(decoding: out[mark...], as: UTF8.self)
        out.removeLast(out.count - mark)
        restore(savedLex)
        trimLeft = savedTrim
        for i in 0..<m.params.count {
            vars[m.params[i].name] = savedParams[i]
        }
        vars["loop"] = savedLoop
        return .str(cap)
    }

    private func listLiteral() throws -> JinjaValue {
        lex()
        var items: [JinjaValue] = []
        while !isOp("]") && kind != .end {
            items.append(try evalExpr())
            if isOp(",") { lex() }
        }
        if isOp("]") { lex() }
        return .list(items)
    }

    private func dictLiteral() throws -> JinjaValue {
        let d = newDict()
        brace += 1
        lex()
        while !isOp("}") && kind != .end {
            let key = toStr(try evalExpr())
            if isOp(":") { lex() }
            dictPut(d, key, try evalExpr())
            if isOp(",") { lex() }
        }
        brace -= 1
        if isOp("}") { lex() }
        return d
    }

    private func tupleLiteral(_ first: JinjaValue) throws -> JinjaValue {
        var items = [first]
        while isOp(",") {
            lex()
            if !isOp(")") && kind != .end {
                items.append(try evalExpr())
            }
        }
        return .list(items)
    }

    // A bare reference to a function-style builtin is "defined"; a sentinel
    // node carries that fact since no variable is stored for it.

    private func identifier() throws -> JinjaValue {
        let nm = text
        lex()
        var v = JinjaValue.undefined
        if isOp("(") {
            v = try callNamed(nm)
        } else if let bound = vars[nm] {
            v = bound
        } else if isBuiltinGlobal(nm) {
            v = .node(handle: 0, tag: builtinTag)
        }
        return v
    }

    private func evalPrimary() throws -> JinjaValue {
        var v = JinjaValue.undefined
        if kind == .str {
            v = .str(text)
            lex()
        } else if kind == .int {
            v = .int(iv)
            lex()
        } else if isId("true") || isId("True") {
            v = .bool(true)
            lex()
        } else if isId("false") || isId("False") {
            v = .bool(false)
            lex()
        } else if isId("none") || isId("None") {
            v = .none
            lex()
        } else if isOp("(") {
            lex()
            v = try evalExpr()
            if isOp(",") { v = try tupleLiteral(v) }
            if isOp(")") { lex() }
        } else if isOp("[") {
            v = try listLiteral()
        } else if isOp("{") {
            v = try dictLiteral()
        } else if kind == .id {
            v = try identifier()
        }
        return v
    }

    private func postfixDot(_ obj: JinjaValue) throws -> JinjaValue {
        lex()
        let nm = text
        lex()
        var result = JinjaValue.undefined
        if isOp("(") {
            result = try callMethod(obj, nm, try parseArgs())
        } else {
            result = getAttr(obj, nm)
        }
        return result
    }

    private func postfixFilter(_ v: JinjaValue) throws -> JinjaValue {
        lex()
        let nm = text
        lex()
        var ca = CallArgs()
        if isOp("(") { ca = try parseCall() }
        return try applyFilter(v, nm, ca)
    }

    private func sliceSeq(_ v: JinjaValue, _ hStart: Bool, _ start: Int,
                          _ hStop: Bool, _ stop: Int, _ hStep: Bool,
                          _ step: Int) throws -> JinjaValue {
        let len = seqLen(v)
        let stepv = hStep ? step : 1
        if stepv == 0 { try fail("slice step cannot be zero") }
        let lo = stepv < 0 ? -1 : 0
        let hi = stepv < 0 ? len - 1 : len
        var s = hStart ? (start < 0 ? start + len : start)
                       : (stepv < 0 ? len - 1 : 0)
        var e = hStop ? (stop < 0 ? stop + len : stop)
                      : (stepv < 0 ? -1 : len)
        s = min(max(s, lo), hi)
        e = min(max(e, lo), hi)
        var items: [JinjaValue] = []
        var i = s
        while stepv < 0 ? i > e : i < e {
            if i >= 0 && i < len { items.append(seqIndex(v, i)) }
            i += stepv
        }
        return .list(items)
    }

    private func postfixIndex(_ v: JinjaValue) throws -> JinjaValue {
        lex()
        var idx = JinjaValue.undefined
        var start = 0
        var stop = 0
        var step = 0
        var hStart = false
        var hStop = false
        var hStep = false
        var slice = false
        if !isOp(":") {
            idx = try evalExpr()
            hStart = true
            start = intOf(idx)
        }
        if isOp(":") {
            slice = true
            lex()
            if !isOp(":") && !isOp("]") {
                hStop = true
                stop = intOf(try evalExpr())
            }
            if isOp(":") {
                lex()
                if !isOp("]") {
                    hStep = true
                    step = intOf(try evalExpr())
                }
            }
        }
        var result = JinjaValue.undefined
        if slice {
            result = try sliceSeq(v, hStart, start, hStop, stop, hStep,
                                  step)
        } else if case .str(let k) = idx {
            result = getAttr(v, k)
        } else {
            result = seqIndex(v, intOf(idx))
        }
        if isOp("]") { lex() }
        return result
    }

    private func evalPostfix() throws -> JinjaValue {
        var v = try evalPrimary()
        while isOp(".") || isOp("[") || isOp("|") {
            if isOp(".") {
                v = try postfixDot(v)
            } else if isOp("[") {
                v = try postfixIndex(v)
            } else {
                v = try postfixFilter(v)
            }
        }
        return v
    }

    private func evalUnary() throws -> JinjaValue {
        var v = JinjaValue.undefined
        if isOp("-") {
            lex()
            v = .int(-intOf(try evalUnary()))
        } else if isOp("+") {
            lex()
            v = try evalUnary()
        } else {
            v = try evalPostfix()
        }
        return v
    }

    // Unary minus binds into the base of `**` (-2**2 is 4) and `**` is
    // left-associative; integer only, a negative exponent is 1.

    private func evalPow() throws -> JinjaValue {
        var v = try evalUnary()
        while isOp("**") {
            lex()
            let base = intOf(v)
            var e = intOf(try evalUnary())
            var res = 1
            while e > 0 {
                res *= base
                e -= 1
            }
            v = .int(res)
        }
        return v
    }

    private func pyFloorDiv(_ a: Int, _ b: Int) -> Int {
        var q = a / b
        let r = a % b
        if r != 0 && ((r < 0) != (b < 0)) { q -= 1 }
        return q
    }

    private func pyMod(_ a: Int, _ b: Int) -> Int {
        var m = a % b
        if m != 0 && ((m < 0) != (b < 0)) { m += b }
        return m
    }

    // `/` is integer division, a documented divergence from Jinja; `//` and `%`
    // follow Python's floor and sign rules.

    private func evalMul() throws -> JinjaValue {
        var v = try evalPow()
        while isOp("*") || isOp("/") || isOp("//") || isOp("%") {
            let mul = isOp("*")
            let mod = isOp("%")
            lex()
            let a = intOf(v)
            let b = intOf(try evalPow())
            if !mul && b == 0 { try fail("division by zero") }
            v = .int(mul ? a * b : (mod ? pyMod(a, b) : pyFloorDiv(a, b)))
        }
        return v
    }

    private func evalConcat() throws -> JinjaValue {
        var v = try evalMul()
        while isOp("~") || isOp("+") || isOp("-") {
            let sub = isOp("-")
            let tilde = isOp("~")
            lex()
            let r = try evalMul()
            if !sub && (isStr(v) || isStr(r) || tilde) {
                v = .str(toStr(v) + toStr(r))
            } else {
                v = .int(sub ? intOf(v) - intOf(r) : intOf(v) + intOf(r))
            }
        }
        return v
    }

    private func relational(_ v: JinjaValue) throws -> JinjaValue {
        let ne = isOp("!=")
        let lt = isOp("<")
        let gt = isOp(">")
        let le = isOp("<=")
        let ge = isOp(">=")
        lex()
        let r = try evalConcat()
        var result = false
        if lt {
            result = intOf(v) < intOf(r)
        } else if gt {
            result = intOf(v) > intOf(r)
        } else if le {
            result = intOf(v) <= intOf(r)
        } else if ge {
            result = intOf(v) >= intOf(r)
        } else {
            result = jvEq(v, r) != ne
        }
        return .bool(result)
    }

    private func isTest(_ v: JinjaValue) throws -> JinjaValue {
        lex()
        let neg = isId("not")
        if neg { lex() }
        let nm = text
        lex()
        let r = try doTest(v, nm, .undefined, false)
        return .bool(r != neg)
    }

    private func atCmpOp() -> Bool {
        isOp("==") || isOp("!=") || isOp("<") || isOp(">") ||
        isOp("<=") || isOp(">=") || isId("in") || isId("not") ||
        isId("is")
    }

    private func evalCmp() throws -> JinjaValue {
        var v = try evalConcat()
        while atCmpOp() {
            if isId("in") {
                lex()
                v = .bool(contains(try evalConcat(), v))
            } else if isId("not") {
                lex()
                lex()
                v = .bool(!contains(try evalConcat(), v))
            } else if isId("is") {
                v = try isTest(v)
            } else {
                v = try relational(v)
            }
        }
        return v
    }

    private func evalNot() throws -> JinjaValue {
        var v = JinjaValue.undefined
        if isId("not") {
            lex()
            v = .bool(!truthy(try evalNot()))
        } else {
            v = try evalCmp()
        }
        return v
    }

    private func evalAnd() throws -> JinjaValue {
        var v = try evalNot()
        while isId("and") {
            lex()
            let r = try evalNot()
            v = truthy(v) ? r : v
        }
        return v
    }

    private func evalOr() throws -> JinjaValue {
        var v = try evalAnd()
        while isId("or") {
            lex()
            let r = try evalAnd()
            v = truthy(v) ? v : r
        }
        return v
    }

    private func evalExpr() throws -> JinjaValue {
        var v = try evalOr()
        if isId("if") {
            lex()
            let cond = truthy(try evalOr())
            var alt = JinjaValue.undefined
            if isId("else") {
                lex()
                alt = try evalExpr()
            }
            v = cond ? v : alt
        }
        return v
    }

    private func bindLoop(_ v1: String, _ v2: String?,
                          _ base: JinjaValue, _ n: Int, _ k: Int) {
        let cur = seqIndex(base, k)
        let lp = newDict()
        dictPut(lp, "index", .int(k + 1))
        dictPut(lp, "index0", .int(k))
        dictPut(lp, "first", .bool(k == 0))
        dictPut(lp, "last", .bool(k == n - 1))
        dictPut(lp, "length", .int(n))
        dictPut(lp, "previtem", k > 0 ? seqIndex(base, k - 1) : .undefined)
        dictPut(lp, "nextitem",
                k < n - 1 ? seqIndex(base, k + 1) : .undefined)
        if let v2 {
            let items = listItems(cur)
            let pair = items.count >= 2
            vars[v1] = pair ? items[0] : cur
            vars[v2] = pair ? items[1] : .undefined
        } else {
            vars[v1] = cur
        }
        vars["loop"] = lp
    }

    private func forElse(_ n: Int) throws {
        let show = mute == 0 && n == 0
        consumeClose()
        mute += show ? 0 : 1
        _ = try renderBlock()
        mute -= show ? 0 : 1
    }

    private func doFor() throws {
        let v1 = text
        var v2: String? = nil
        lex()
        if isOp(",") {
            lex()
            v2 = text
            lex()
        }
        lex()
        let seq = try evalExpr()
        consumeClose()
        let savedLoop = vars["loop"]
        let body = p
        let trim0 = trimLeft
        let n = seqLen(seq)
        let real = mute == 0 && n > 0
        let iters = real ? n : 1
        var stop = Stop.endfor
        mute += real ? 0 : 1
        var k = 0
        while k < iters {
            p = body
            trimLeft = trim0
            if real { bindLoop(v1, v2, seq, n, k) }
            stop = try renderBlock()
            k += 1
        }
        mute -= real ? 0 : 1
        if stop == .elseArm { try forElse(n) }
        consumeClose()
        vars["loop"] = savedLoop
    }

    private func doIf() throws {
        var cond = truthy(try evalExpr())
        consumeClose()
        var takenAny = false
        var stop = Stop.none
        repeat {
            let take = cond && !takenAny
            mute += take ? 0 : 1
            stop = try renderBlock()
            mute -= take ? 0 : 1
            takenAny = takenAny || take
            if stop == .elif {
                cond = truthy(try evalExpr())
                consumeClose()
            } else if stop == .elseArm {
                cond = true
                consumeClose()
            } else if stop == .endif {
                consumeClose()
            }
        } while stop == .elif || stop == .elseArm
    }

    private func setValue(_ name: String, _ key: String?,
                          _ val: JinjaValue) throws {
        if mute == 0 {
            if let key {
                let ns = vars[name]
                if let ns, isDict(ns) {
                    dictPut(ns, key, val)
                } else {
                    try fail("set member on undeclared namespace")
                }
            } else {
                vars[name] = val
            }
        }
    }

    private func doSet() throws {
        let name = text
        var key: String? = nil
        lex()
        if isOp(".") {
            lex()
            key = text
            lex()
        }
        var val = JinjaValue.str("")
        if isOp("=") {
            lex()
            val = try evalExpr()
            consumeClose()
        } else {
            consumeClose()
            let mark = out.count
            _ = try renderBlock()
            val = .str(String(decoding: out[mark...], as: UTF8.self))
            out.removeLast(out.count - mark)
            consumeClose()
        }
        try setValue(name, key, val)
    }

    private func doFilter() throws {
        let nm = text
        lex()
        var ca = CallArgs()
        if isOp("(") { ca = try parseCall() }
        consumeClose()
        let mark = out.count
        _ = try renderBlock()
        let body = JinjaValue.str(
            String(decoding: out[mark...], as: UTF8.self))
        out.removeLast(out.count - mark)
        emit(toStr(try applyFilter(body, nm, ca)))
        consumeClose()
    }

    // Rendering the body here, even muted, would drift the mute counter, so it
    // is skipped by raw scan and rendered only when called.

    private func skipMacroBody() {
        var pos = p
        var level = 1
        while level > 0 && at(pos) != 0 {
            if at(pos) == bLbrace && at(pos + 1) == bHash {
                pos = skipComment(from: pos + 2)
            } else if at(pos) == bLbrace && at(pos + 1) == bPercent {
                pos = scanStatement(from: pos, level: &level)
            } else {
                pos += 1
            }
        }
        p = pos
    }

    private func skipComment(from: Int) -> Int {
        var q = from
        while at(q) != 0 && !(at(q) == bHash && at(q + 1) == bRbrace) {
            q += 1
        }
        return at(q) != 0 ? q + 2 : q
    }

    private func matches(_ pos: Int, _ word: String) -> Bool {
        let bytes = Array(word.utf8)
        var i = 0
        while i < bytes.count && at(pos + i) == bytes[i] {
            i += 1
        }
        return i == bytes.count
    }

    private func scanStatement(from: Int, level: inout Int) -> Int {
        var q = from + 2 + (at(from + 2) == bDash ? 1 : 0)
        while isSpace(at(q)) { q += 1 }
        level += matches(q, "macro") ? 1 : 0
        level -= matches(q, "endmacro") ? 1 : 0
        while at(q) != 0 && !(at(q) == bPercent && at(q + 1) == bRbrace) {
            let c = at(q)
            q += (c == bSquote || c == bDquote) ? 1 : 0
            while (c == bSquote || c == bDquote) && at(q) != 0 &&
                  at(q) != c {
                q += (at(q) == bBackslash && at(q + 1) != 0) ? 2 : 1
            }
            q += 1
        }
        return at(q) != 0 ? q + 2 : q
    }

    private func doMacro() throws {
        let name = text
        var params: [Param] = []
        lex()
        lex()
        while kind == .id {
            let pname = text
            var defPos: Int? = nil
            lex()
            if isOp("=") {
                lex()
                defPos = tokenStart
                _ = try evalExpr()
            }
            params.append(Param(name: pname, defPos: defPos))
            if isOp(",") { lex() }
        }
        if isOp(")") { lex() }
        consumeClose()
        macros.append(Macro(name: name, params: params, body: p))
        skipMacroBody()
    }

    private func handleStatement() throws -> Stop {
        var stop = Stop.none
        if isId("if") {
            lex()
            try doIf()
        } else if isId("for") {
            lex()
            try doFor()
        } else if isId("set") {
            lex()
            try doSet()
        } else if isId("macro") {
            lex()
            try doMacro()
        } else if isId("filter") {
            lex()
            try doFilter()
        } else if isId("generation") || isId("endgeneration") {
            lex()
            consumeClose()
        } else if isId("elif") {
            lex()
            stop = .elif
        } else if isId("else") {
            lex()
            stop = .elseArm
        } else if isId("endif") {
            lex()
            stop = .endif
        } else if isId("endfor") {
            lex()
            stop = .endfor
        } else if isId("endmacro") {
            lex()
            stop = .endmacro
        } else if isId("endset") {
            lex()
            stop = .endset
        } else if isId("endfilter") {
            lex()
            stop = .endfilter
        } else {
            try fail("unsupported tag: \(text)")
        }
        return stop
    }

    private func emitLit(_ lo: Int, _ hi: Int, _ lead: Bool) {
        var a = lo
        var b = hi
        if trimLeft {
            while a < b && isSpace(src[a]) { a += 1 }
        }
        if lead {
            while b > a && isSpace(src[b - 1]) { b -= 1 }
        }
        emitBytes(a, b)
        trimLeft = false
    }

    private func renderBlock() throws -> Stop {
        depth += 1
        if depth > maxDepth { try fail("max recursion depth exceeded") }
        var stop = Stop.none
        while stop == .none {
            let litStart = p
            while at(p) != 0 && !(at(p) == bLbrace &&
                  (at(p + 1) == bLbrace || at(p + 1) == bPercent ||
                   at(p + 1) == bHash)) {
                p += 1
            }
            let two = at(p) == bLbrace
            let comment = two && at(p + 1) == bHash
            let stmt = two && at(p + 1) == bPercent
            let lead = two && at(p + 2) == bDash
            emitLit(litStart, p, lead)
            if at(p) == 0 {
                stop = .eof
            } else if comment {
                p = skipCommentTag()
            } else if !stmt {
                prime(p + 2 + (lead ? 1 : 0))
                emit(toStr(try evalExpr()))
                consumeClose()
            } else {
                prime(p + 2 + (lead ? 1 : 0))
                stop = try handleStatement()
            }
        }
        depth -= 1
        return stop
    }

    private func skipCommentTag() -> Int {
        var q = p
        while at(q) != 0 && !(at(q) == bHash && at(q + 1) == bRbrace) {
            q += 1
        }
        trimLeft = at(q - 1) == bDash
        return at(q) != 0 ? q + 2 : q
    }
}
