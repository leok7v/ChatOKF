import Foundation

public struct ToolSpec: Sendable {
    public let name: String
    public let description: String
    public let parametersJSON: String

    public init(name: String, description: String,
                parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

public struct ToolArg: Sendable {
    public let name: String
    public let value: String

    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }
}

public struct ToolCall: Sendable {
    public var functionName: String
    public var params: [ToolArg]
    public var rawBlock: String

    public init(functionName: String, params: [ToolArg],
                rawBlock: String) {
        self.functionName = functionName
        self.params = params
        self.rawBlock = rawBlock
    }
}

public protocol ToolRunner: Sendable {
    var tools: [ToolSpec] { get }
    func execute(_ name: String, _ args: [ToolArg]) async -> String
    func beginTurn()
}

public extension ToolRunner {
    func beginTurn() {}
}

public final class TurnMemo: @unchecked Sendable {
    static let maxPages = 4
    static let maxPageBytes = 2 << 20

    private let lock = NSLock()
    private var titles: [String: String] = [:]
    private var pages: [(url: String, text: String)] = []
    private var slices: Set<String> = []

    public init() {}

    public func reset() {
        lock.lock()
        titles.removeAll()
        pages.removeAll()
        slices.removeAll()
        lock.unlock()
    }

    func title(for id: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return titles[id]
    }

    func note(_ id: String, _ title: String) {
        lock.lock()
        titles[id] = title
        lock.unlock()
    }

    func page(for url: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return pages.first { entry in entry.url == url }?.text
    }

    func notePage(_ url: String, _ text: String) {
        lock.lock()
        pages.removeAll { entry in entry.url == url }
        pages.append((url, text))
        var total = pages.reduce(0) { sum, entry in
            sum + entry.text.utf8.count
        }
        while pages.count > TurnMemo.maxPages
            || (total > TurnMemo.maxPageBytes && pages.count > 1) {
            total -= pages[0].text.utf8.count
            pages.removeFirst()
        }
        lock.unlock()
    }

    func sliceDelivered(_ url: String, _ offset: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return slices.contains("\(url)#\(offset)")
    }

    func noteSlice(_ url: String, _ offset: Int) {
        lock.lock()
        slices.insert("\(url)#\(offset)")
        lock.unlock()
    }
}

public struct SafeToolRunner: ToolRunner {
    let slugsPath: String?
    let wikipedia: Bool
    let network: Bool
    let calculator = Calculator()
    let memo = TurnMemo()

    public init(slugsPath: String? = nil, wikipedia: Bool = true,
                network: Bool = true) {
        self.slugsPath = slugsPath
        self.wikipedia = wikipedia
        self.network = network
    }

    public var tools: [ToolSpec] {
        var t = [Tools.getCurrentTimeSpec, Tools.calculatorSpec]
        if network {
            if SearchProvider.any {
                t.append(Tools.webSearchSpec(
                    wikiAdvertised: wikipedia && slugsPath != nil))
            }
            t.append(Tools.fetchUrlSpec)
            t.append(Tools.getWeatherSpec)
        }
        if wikipedia {
            t.append(Tools.getNewsSpec)
            if slugsPath != nil { t.append(Tools.wikipediaSpec) }
        }
        return t
    }

    public func execute(_ name: String,
                        _ args: [ToolArg]) async -> String {
        let webOff = !network && (name == "web_search"
            || name == "fetch_url" || name == "get_weather")
        let searchOff = name == "web_search" && !SearchProvider.any
        let wikiOff = !wikipedia && (name == "wikipedia_query"
            || name == "get_news")
        let result: String
        if webOff {
            result = "error: \(name) is disabled; web access is off"
        } else if searchOff {
            result = "error: web_search is disabled; no search provider "
                + "is switched on"
        } else if wikiOff {
            result = "error: \(name) is disabled; Wikipedia access is off"
        } else if name == "calculator" {
            let expr = args.first { arg in arg.name == "expression" }?.value
            result = await calculator.evaluate(expr ?? "")
        } else if name == "wikipedia_query", let path = slugsPath {
            result = await Tools.runWikipedia(args, slugsPath: path,
                                              memo: memo)
        } else {
            result = await Tools.executeSafe(name, args, memo: memo)
        }
        return result
    }

    public func beginTurn() {
        memo.reset()
    }
}

public enum Tools {
    private static let maxParams = 8

    private static let diagSinkLock = NSLock()
    nonisolated(unsafe) private static var diagSinkBox:
        (@Sendable (String) -> Void)?

    public static func setDiagSink(_ sink: (@Sendable (String) -> Void)?) {
        diagSinkLock.lock()
        diagSinkBox = sink
        diagSinkLock.unlock()
    }

    static func diag(_ s: String) {
        Diag.shared.report(.tools, s)
        diagSinkLock.lock()
        let sink = diagSinkBox
        diagSinkLock.unlock()
        sink?(s)
    }

    public static func parameterNames(_ parametersJSON: String)
        -> Set<String> {
        var out: Set<String> = []
        if let obj = try? JSONSerialization.jsonObject(
               with: Data(parametersJSON.utf8)) as? [String: Any],
           let props = obj["properties"] as? [String: Any] {
            out = Set(props.keys)
        }
        return out
    }

    public static func canonicalArgName(_ name: String,
                                        _ known: Set<String>) -> String? {
        var result: String? = known.contains(name) ? name : nil
        var k = 0
        while result == nil && k < argAliases.count {
            if argAliases[k].alias == name
                && known.contains(argAliases[k].canon) {
                result = argAliases[k].canon
            }
            k += 1
        }
        return result
    }

    static let getCurrentTimeSpec = ToolSpec(
        name: "get_current_time",
        description: "Returns the current date and time. Use for any "
            + "'what is the date / time / today / now' question.",
        parametersJSON:
            "{\"type\":\"object\",\"properties\":{},\"required\":[]}")

    static let calculatorSpec = ToolSpec(
        name: "calculator",
        description: "Evaluate a math expression and return the number. Use it "
            + "for ANY arithmetic instead of computing by hand. Supports "
            + "+ - * / ^, parentheses, the constants pi/e/tau/i, and functions "
            + "sin cos tan asin acos atan exp ln log log2 sqrt cbrt abs floor "
            + "ceil round min max pow mod hypot atan2 (ln = natural log, log = "
            + "base 10, log(x,b) = base b, round(x,n) = n decimals). Function "
            + "arguments go in "
            + "parentheses separated by commas: pow(1.05, 22). Names are not "
            + "case-sensitive. Complex numbers work: i is the "
            + "imaginary unit, e.g. 'exp(i*pi)' -> -1, 'i^i', 'sqrt(-1)' -> i, "
            + "'ln(-1)'; abs = modulus, plus re im conj arg. Has variable "
            + "memory: send 'x = 5' to store x, then reuse it ('x * 3'), or "
            + "update it ('x = x + 1'). Write multiplication explicitly: "
            + "'2 * x', never '2x'. Send plain numbers: no $ signs, no "
            + "thousands separators. It EVALUATES, it does not solve: send "
            + "'(1.10 - 1) / 2', not 'b + (b + 1) = 1.10'. Send ONE "
            + "expression, not a list of equations.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"expression\":{\"type\":\"string\",\"description\":\"A math "
            + "expression or an assignment, e.g. 'x = 5' or 'sqrt(x) * pi'. "
            + "Multiplication must be explicit: '2 * x', not '2x'. No $ "
            + "signs.\"}},"
            + "\"required\":[\"expression\"]}")

    static func webSearchSpec(wikiAdvertised: Bool) -> ToolSpec {
        let routing = wikiAdvertised
            ? "For encyclopedic facts (people, places, science, history, "
                + "definitions) use wikipedia_query FIRST; use web_search "
                + "for current events, news, prices, or when wikipedia_query "
                + "finds no match."
            : "Use it for current events, prices, or facts you are unsure "
                + "of."
        return ToolSpec(
            name: "web_search",
            description: "Search the web; returns ranked results with "
                + "snippets. To read a result page, pass its URL to "
                + "fetch_url. " + routing,
            parametersJSON: "{\"type\":\"object\",\"properties\":{"
                + "\"query\":{\"type\":\"string\"}},"
                + "\"required\":[\"query\"]}")
    }

    static let wikipediaSpec = ToolSpec(
        name: "wikipedia_query",
        description: "Look a topic up in Wikipedia (people, places, science, "
            + "history, definitions). Use this FIRST for any 'what does "
            + "Wikipedia say about X' or factual/encyclopedic question, BEFORE "
            + "web_search: it runs a LOCAL semantic search (your query is NOT "
            + "sent to any search engine; only the matched article id is "
            + "fetched, so it is private) and returns the best article's text, "
            + "ending with a 'Related articles' list you can look up with "
            + "further wikipedia_query calls. Ask a FULL question or "
            + "descriptive phrase, not a bare keyword -- e.g. 'What is a red "
            + "dwarf star?'. Only if it reports no confident match, fall back "
            + "to web_search or your own knowledge.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"query\":{\"type\":\"string\",\"description\":\"A full "
            + "question or descriptive phrase.\"}},"
            + "\"required\":[\"query\"]}")

    static let fetchUrlSpec = ToolSpec(
        name: "fetch_url",
        description: "Fetch the readable text content of a web page at an "
            + "http(s):// URL (a bare domain like example.com also works); "
            + "HTML markup, scripts, and styles are removed. Capped to "
            + "`limit` bytes (default 16384; -1 = no cap); use `offset` to "
            + "page through long content.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"url\":{\"type\":\"string\",\"description\":\"Full "
            + "http(s):// URL or a bare domain.\"},"
            + "\"limit\":{\"type\":\"integer\",\"description\":\"Max "
            + "bytes to return (default 16384; -1 = no cap)\"},"
            + "\"offset\":{\"type\":\"integer\",\"description\":\"Byte "
            + "offset for paging (default 0)\"}},"
            + "\"required\":[\"url\"]}")

    static let getNewsSpec = ToolSpec(
        name: "get_news",
        description: "Get today's top news headlines (current events, 'in the "
            + "news', what's happening now). Use this FIRST for any 'what is in "
            + "the news / latest headlines / current events' question, before "
            + "web_search. Call with NO arguments for general / top news; pass a "
            + "topic ONLY to narrow to a specific subject (e.g. 'science'). Do "
            + "not pass a filler topic like 'general', 'today', or 'news'.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"topic\":{\"type\":\"string\",\"description\":\"Optional "
            + "keyword to filter the headlines, e.g. 'science'.\"}},"
            + "\"required\":[]}")

    static let getWeatherSpec = ToolSpec(
        name: "get_weather",
        description: "Get current weather plus a multi-day forecast (today and "
            + "the next few days). The location is OPTIONAL: when the user does "
            + "not name a place (e.g. 'weather here', 'my weather', 'weather "
            + "tomorrow'), call this with NO arguments -- it resolves the "
            + "caller's own location automatically. NEVER ask the user where "
            + "they are. Pass `location` only when they name a specific city.",
        parametersJSON: "{\"type\":\"object\",\"properties\":{"
            + "\"location\":{\"type\":\"string\",\"description\":\"City or "
            + "place, e.g. 'Tokyo' or 'Paris, France'. Omit for the caller's "
            + "own location.\"}},\"required\":[]}")

    public static func findToolCall(
        in stream: String, from: Int,
        open openTag: String = "<tool_call>",
        close closeTag: String = "</tool_call>"
    ) -> Range<String.Index>? {
        var result: Range<String.Index>? = nil
        let start = stream.index(stream.startIndex, offsetBy: from,
                                 limitedBy: stream.endIndex)
        if let start {
            let tail = start..<stream.endIndex
            if let open = stream.range(of: openTag, range: tail),
               let close = stream.range(
                    of: closeTag,
                    range: open.upperBound..<stream.endIndex) {
                result = open.upperBound..<close.lowerBound
            }
        }
        return result
    }

    public static func parse(_ block: Substring) -> ToolCall? {
        let cs = Array(block)
        var result: ToolCall? = nil
        if firstNonSpace(cs) == "{" {
            result = parseJSON(block)
        } else if let gemma = parseGemma(block) {
            result = gemma
        } else if let fn = functionName(cs) {
            result = ToolCall(functionName: fn.name,
                              params: foldArgPairs(params(cs, fn.next)),
                              rawBlock: String(block))
        }
        return result
    }

    static let gemmaQuote = Array("<|\"|>")
    private static let gemmaOpener = "call:"

    static func parseGemma(_ block: Substring) -> ToolCall? {
        let s = block.trimmingCharacters(in: .whitespacesAndNewlines)
        var result: ToolCall? = nil
        if s.hasPrefix(gemmaOpener) {
            let rest = s.dropFirst(gemmaOpener.count)
            var name = String(rest)
            var args: [ToolArg] = []
            if let brace = rest.firstIndex(of: "{") {
                name = String(rest[..<brace])
                var body = rest[rest.index(after: brace)...]
                if body.hasSuffix("}") { body = body.dropLast() }
                args = gemmaArgs(Array(body))
            }
            name = name.trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty {
                result = ToolCall(functionName: name, params: args,
                                  rawBlock: String(block))
            }
        }
        return result
    }

    static func gemmaArgs(_ c: [Character]) -> [ToolArg] {
        var out: [ToolArg] = []
        var depth = 0
        var quoted = false
        var key: String? = nil
        var buf = ""
        var i = 0
        func flush() {
            let n = (key ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if key != nil && !n.isEmpty && out.count < maxParams {
                out.append(ToolArg(name: n, value: buf))
            }
            key = nil
            buf = ""
        }
        while i < c.count {
            var step = 1
            if startsAt(c, i, gemmaQuote) {
                quoted = !quoted
                step = gemmaQuote.count
            } else if quoted {
                buf.append(c[i])
            } else if c[i] == "{" || c[i] == "[" {
                depth += 1
                buf.append(c[i])
            } else if c[i] == "}" || c[i] == "]" {
                depth -= 1
                buf.append(c[i])
            } else if c[i] == ":" && depth == 0 && key == nil {
                key = buf
                buf = ""
            } else if c[i] == "," && depth == 0 {
                flush()
            } else {
                buf.append(c[i])
            }
            i += step
        }
        flush()
        return out
    }

    private static func firstNonSpace(_ cs: [Character]) -> Character? {
        let ws: Set<Character> = [" ", "\t", "\n", "\r"]
        var result: Character? = nil
        var i = 0
        while result == nil && i < cs.count {
            if !ws.contains(cs[i]) { result = cs[i] }
            i += 1
        }
        return result
    }

    static func parseJSON(_ block: Substring) -> ToolCall? {
        var result: ToolCall? = nil
        if let obj = try? JSONSerialization.jsonObject(
               with: Data(block.utf8)) as? [String: Any],
           let name = obj["name"] as? String, !name.isEmpty {
            result = ToolCall(functionName: name,
                              params: jsonArgs(obj["arguments"]),
                              rawBlock: String(block))
        }
        return result
    }

    static func jsonArgs(_ value: Any?) -> [ToolArg] {
        var dict = value as? [String: Any]
        if dict == nil, let s = value as? String {
            dict = (try? JSONSerialization.jsonObject(
                with: Data(s.utf8))) as? [String: Any]
        }
        var out: [ToolArg] = []
        for (name, v) in dict ?? [:] {
            out.append(ToolArg(name: name, value: jsonScalar(v)))
        }
        return out
    }

    static func jsonScalar(_ v: Any) -> String {
        var result = ""
        if let s = v as? String {
            result = s
        } else if let n = v as? NSNumber {
            result = numberString(n)
        } else if let data = try? JSONSerialization.data(withJSONObject: v),
                  let s = String(data: data, encoding: .utf8) {
            result = s
        }
        return result
    }

    private static func numberString(_ n: NSNumber) -> String {
        var result: String
        if CFGetTypeID(n) == CFBooleanGetTypeID() {
            result = n.boolValue ? "true" : "false"
        } else {
            let d = n.doubleValue
            result = d == d.rounded() && abs(d) < 1e15
                ? String(n.int64Value) : "\(d)"
        }
        return result
    }

    static func foldArgPairs(_ args: [ToolArg]) -> [ToolArg] {
        var out: [ToolArg] = []
        var pendingKey: String? = nil
        for arg in args {
            if arg.name == "arg_key" {
                pendingKey = arg.value
            } else if arg.name == "arg_value", let key = pendingKey {
                out.append(ToolArg(name: key, value: arg.value))
                pendingKey = nil
            } else {
                out.append(arg)
            }
        }
        return out
    }

    private static func functionName(_ cs: [Character])
        -> (name: String, next: Int)? {
        let ws: Set<Character> = [" ", "\t", "\n", "\r"]
        var result: (name: String, next: Int)? = nil
        var i = 0
        while result == nil && i < cs.count {
            if cs[i] == "<" {
                var j = i + 1
                while j < cs.count && ws.contains(cs[j]) { j += 1 }
                var s: Int? = nil
                for opener in ["function=", "fuction="] where s == nil {
                    if ciStarts(cs, j, Array(opener)) {
                        s = j + opener.count
                    }
                }
                if let s, let close = find(cs, [">"], s) {
                    var e = close
                    while e > s && (cs[e - 1] == "/" || cs[e - 1] == " ") {
                        e -= 1
                    }
                    result = (String(cs[s ..< e]), close + 1)
                }
            }
            i += 1
        }
        return result
    }

    private static func params(_ cs: [Character],
                               _ from: Int) -> [ToolArg] {
        var out: [ToolArg] = []
        var p = from
        let n = cs.count
        while p < n && out.count < maxParams {
            if let lt = find(cs, ["<"], p),
               let gt = find(cs, [">"], lt) {
                let tag = Array(cs[(lt + 1)..<gt])
                if isStructuralTag(tag) {
                    p = gt + 1
                } else {
                    let parsed = parseParam(cs, tag, gt)
                    if let arg = parsed.arg { out.append(arg) }
                    p = parsed.next
                }
            } else {
                p = n
            }
        }
        return out
    }

    private static func isStructuralTag(_ tag: [Character]) -> Bool {
        tag.isEmpty || tag[0] == "/"
            || starts(tag, Array("function"))
            || starts(tag, Array("fuction"))
            || starts(tag, Array("tool_call"))
    }

    private static func parseParam(_ cs: [Character], _ tag: [Character],
                                   _ gt: Int) -> (arg: ToolArg?, next: Int) {
        var key = tag
        let prefix = Array("parameter=")
        if starts(key, prefix) { key = Array(key[prefix.count...]) }
        var result: (arg: ToolArg?, next: Int)
        if let eq = key.firstIndex(of: "=") {
            let name = String(key[..<eq])
            let value = unquote(String(key[(eq + 1)...]))
            let arg = name.isEmpty
                ? nil : ToolArg(name: name, value: value)
            result = (arg, gt + 1)
        } else {
            result = bareParam(cs, key, gt)
        }
        return result
    }

    private static func bareParam(_ cs: [Character], _ key: [Character],
                                  _ gt: Int) -> (arg: ToolArg?, next: Int) {
        let n = cs.count
        var vstart = gt + 1
        if vstart < n && cs[vstart] == "\n" { vstart += 1 }
        let closeKey = Array("</") + key + Array(">")
        let close = find(cs, closeKey, vstart)
            ?? find(cs, Array("</parameter>"), vstart)
        let vend = close ?? find(cs, ["<"], vstart) ?? n
        var ve = vend
        while ve > vstart && (cs[ve - 1] == "\n" || cs[ve - 1] == " ") {
            ve -= 1
        }
        let name = String(key)
        let arg = name.isEmpty
            ? nil : ToolArg(name: name, value: String(cs[vstart..<ve]))
        var p = vend
        if p < n && cs[p] == "<" {
            if let cgt = find(cs, [">"], p), p + 1 < n && cs[p + 1] == "/" {
                p = cgt + 1
            }
        }
        return (arg, p)
    }

    private static func unquote(_ s: String) -> String {
        let c = Array(s)
        var result = s
        let quoted = c.count >= 2 && (c[0] == "\"" || c[0] == "'")
            && c[c.count - 1] == c[0]
        if quoted { result = String(c[1..<(c.count - 1)]) }
        return result
    }

    public static func pathEscapesWorkdir(_ path: String) -> Bool {
        var result = false
        if path.isEmpty {
            result = false
        } else if path.hasPrefix("~") {
            result = true
        } else {
            let wd = FileManager.default.currentDirectoryPath
            let cand = path.hasPrefix("/") ? path : wd + "/" + path
            let norm = lexnormAbs(cand)
            let inside = norm == wd || norm.hasPrefix(wd + "/")
            result = !inside
        }
        return result
    }

    private static func lexnormAbs(_ p: String) -> String {
        var comps: [Substring] = []
        for seg in p.split(separator: "/") {
            if seg == "." {
            } else if seg == ".." {
                if !comps.isEmpty { comps.removeLast() }
            } else {
                comps.append(seg)
            }
        }
        return comps.isEmpty ? "/" : "/" + comps.joined(separator: "/")
    }

    public static func outboundSecret(_ s: String) -> String? {
        let b = Array(s.utf8)
        var result: String? = nil
        if s.contains("PRIVATE KEY-----") {
            result = "a private key"
        } else {
            result = tokenSecret(b) ?? entropySecret(b)
        }
        return result
    }

    private static let secretPrefixes: [[UInt8]] = [
        Array("sk-".utf8), Array("ghp_".utf8), Array("gho_".utf8),
        Array("ghs_".utf8), Array("github_pat_".utf8),
        Array("xoxb-".utf8), Array("xoxp-".utf8), Array("AKIA".utf8),
    ]
    private static let idsetBytes = Set(Array(
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            + "0123456789_-").utf8))
    private static let b64Bytes = Set(Array(
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            + "0123456789+/=").utf8))

    private static func tokenSecret(_ b: [UInt8]) -> String? {
        var result: String? = nil
        var p = 0
        while result == nil && p < b.count {
            var k = 0
            while result == nil && k < secretPrefixes.count {
                let pat = secretPrefixes[k]
                if matches(b, p, pat)
                    && runLen(b, p + pat.count, idsetBytes) >= 16 {
                    result = "an API key or access token"
                }
                k += 1
            }
            p += 1
        }
        return result
    }

    private static func entropySecret(_ b: [UInt8]) -> String? {
        var result: String? = nil
        var p = 0
        while result == nil && p < b.count {
            let n = runLen(b, p, b64Bytes)
            if n >= 40 { result = "a long high-entropy token" }
            p += n > 0 ? n : 1
        }
        return result
    }

    private static func runLen(_ b: [UInt8], _ from: Int,
                               _ set: Set<UInt8>) -> Int {
        var n = 0
        while from + n < b.count && set.contains(b[from + n]) { n += 1 }
        return n
    }

    private static func matches(_ b: [UInt8], _ at: Int,
                                _ pat: [UInt8]) -> Bool {
        var j = 0
        while j < pat.count && at + j < b.count && b[at + j] == pat[j] {
            j += 1
        }
        return j == pat.count
    }

    public static func sanitize(_ name: String,
                                _ args: [ToolArg]) -> String? {
        var result: String? = nil
        if name == "execute_shell_command" || name == "exec_shell_command" {
            if let cmd = findArg(args, "command") {
                result = shellCommandEscapes(cmd)
            }
        } else if name == "web_search" {
            if let what = outboundSecret(findArg(args, "query") ?? "") {
                result = outboundMessage(what)
            }
        } else if name == "fetch_url" || name == "read_file"
            || name == "view_text_file" {
            result = sanitizeSource(name, args)
        } else if let path = findArg(args, "path"),
            pathEscapesWorkdir(path) {
            result = pathOutsideMessage(path)
        }
        return result
    }

    private static func sanitizeSource(_ name: String,
                                       _ args: [ToolArg]) -> String? {
        let argName = name == "fetch_url" ? "url" : "path"
        var result: String? = nil
        if let arg = findArg(args, argName) {
            switch resolveSource(arg) {
            case .url(let href):
                if let what = outboundSecret(href) {
                    result = outboundMessage(what)
                }
            case .file(let path):
                if pathEscapesWorkdir(path) {
                    result = pathOutsideMessage(path)
                }
            case .unknown:
                result = nil
            }
        }
        return result
    }

    private static func outboundMessage(_ what: String) -> String {
        "outbound request appears to contain \(what); not sending it"
    }

    private static func pathOutsideMessage(_ path: String) -> String {
        "path '\(path)' is outside the working directory; "
            + "use a path under '.'"
    }

    private enum Source {
        case url(String)
        case file(String)
        case unknown
    }

    private static func resolveSource(_ arg: String) -> Source {
        var result: Source = .unknown
        if arg.isEmpty {
            result = .unknown
        } else if arg.hasPrefix("http://") || arg.hasPrefix("https://") {
            result = .url(arg)
        } else if arg.hasPrefix("file://") {
            result = .file(String(arg.dropFirst(7)))
        } else if arg.hasPrefix("/") || arg.hasPrefix("~")
            || arg.hasPrefix("./") || arg.hasPrefix("../") {
            result = .file(arg)
        } else if isDomainShaped(arg) {
            result = .url("https://" + arg)
        } else {
            result = .unknown
        }
        return result
    }

    private static let commonTLDs: Set<String> = [
        "com", "net", "org", "io", "dev", "ai", "gov", "edu", "co",
        "app", "xyz", "info", "biz", "me", "us", "uk", "de", "fr", "jp",
        "cn", "ca", "au", "in", "br", "ru", "nl", "it", "es", "se", "no",
        "ch", "eu", "tv", "gg", "sh", "to", "ly", "cc", "tech", "online",
    ]

    private static func isDomainShaped(_ arg: String) -> Bool {
        let host = String(arg.prefix(while: { c in c != "/" }))
        let bare = String(host.prefix(while: { c in c != ":" }))
        var result = false
        if let dot = bare.lastIndex(of: "."), bare.first != "." {
            let tld = bare[bare.index(after: dot)...]
            result = !tld.isEmpty
                && commonTLDs.contains(tld.lowercased())
        }
        return result
    }

    private static func shellCommandEscapes(_ cmd: String) -> String? {
        let seps: Set<Character> = [";", "|", "&", "(", ")", "<", ">"]
        let cs = Array(cmd)
        let n = cs.count
        var result: String? = nil
        var i = 0
        while result == nil && i < n {
            while i < n && (cs[i].isWhitespace || seps.contains(cs[i])) {
                i += 1
            }
            if i < n {
                let start = i
                while i < n && !cs[i].isWhitespace
                    && !seps.contains(cs[i]) {
                    i += 1
                }
                let tok = unquote(String(cs[start..<i]))
                if shellTokenEscapes(tok) {
                    result = "command references '\(tok)', outside the "
                        + "working directory"
                }
            }
        }
        return result
    }

    private static let shellRoots = [
        "/Users", "/home", "/etc", "/var", "/root", "/private",
        "/System", "/Library", "/Volumes",
    ]

    private static func shellTokenEscapes(_ t: String) -> Bool {
        let c = Array(t)
        var result = false
        if c.first == "~" {
            result = true
        } else if t == ".." || t.hasPrefix("../") {
            result = true
        } else if c.first != "/" {
            result = false
        } else if c.count == 1 {
            result = true
        } else {
            result = shellRoots.contains(where: { root in
                t == root || t.hasPrefix(root + "/")
            })
        }
        return result
    }

    private static let argAliases: [(canon: String, alias: String)] = [
        ("content", "pcontent"), ("content", "text"),
        ("content", "body"), ("content", "data"),
        ("content", "file_content"),
        ("path", "file"), ("path", "filename"), ("path", "filepath"),
        ("path", "file_path"), ("path", "filimame"),
        ("pattern", "regex"), ("pattern", "query"), ("pattern", "search"),
        ("command", "cmd"), ("command", "cmdline"),
        ("diff", "patch"), ("changes", "edits"),
        ("query", "q"), ("url", "link"), ("url", "u"),
        ("count", "limit"), ("count", "n"),
    ]

    private static func firstValue(_ args: [ToolArg],
                                   _ name: String) -> String? {
        var result: String? = nil
        var i = 0
        while result == nil && i < args.count {
            if args[i].name == name { result = args[i].value }
            i += 1
        }
        return result
    }

    private static func findArg(_ args: [ToolArg],
                                _ name: String) -> String? {
        var result = firstValue(args, name)
        var k = 0
        while result == nil && k < argAliases.count {
            if argAliases[k].canon == name {
                result = firstValue(args, argAliases[k].alias)
            }
            k += 1
        }
        return result
    }

    static func executeSafe(_ name: String, _ args: [ToolArg],
                            memo: TurnMemo? = nil) async -> String {
        let result: String
        switch name {
        case "get_current_time", "get_datetime":
            result = await getCurrentTime()
        case "web_search":
            result = await runWebsearch(args)
        case "fetch_url":
            result = await runFetch(args, memo: memo)
        case "get_news":
            result = await runNews(args)
        case "get_weather":
            result = await runWeather(args)
        default:
            result = "error: tool \(name) unavailable in this environment"
        }
        return result
    }

    public static func getCurrentTime() async -> String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "EEE MMM dd HH:mm:ss yyyy zzz"
        return fmt.string(from: Date())
    }

    private static func runWebsearch(_ args: [ToolArg]) async -> String {
        var result = "error: missing 'query' argument"
        if let query = findArg(args, "query"), !query.isEmpty {
            result = await websearch(query, count: 5)
        }
        return result
    }

    private static func runFetch(_ args: [ToolArg],
                                 memo: TurnMemo? = nil) async -> String {
        var result = "error: missing 'url' argument"
        if let url = findArg(args, "url"), !url.isEmpty {
            let limStr = findArg(args, "limit") ?? firstValue(args, "cap")
            let limit = limStr.flatMap { s in Int(s) } ?? 16_384
            let offset = findArg(args, "offset")
                .flatMap { s in Int(s) } ?? 0
            result = await fetch(url, limit: limit, offset: offset,
                                 memo: memo)
        }
        return result
    }

    // Mwmbl indexes no search operators: a site:/inurl:/intitle:/filetype:
    // term matches nothing and empties the whole result set.
    static func stripOperators(_ query: String) -> String {
        var words: [String] = []
        for word in query.split(separator: " ") {
            let lower = word.lowercased()
            let op = ["site:", "inurl:", "intitle:", "filetype:"]
                .contains { prefix in lower.hasPrefix(prefix) }
            if !op { words.append(String(word)) }
        }
        let cleaned = unquote(words.joined(separator: " "))
            .trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? query : cleaned
    }

    public static func websearch(_ query: String,
                                 count: Int) async -> String {
        var result = "error: missing 'query' argument"
        if !query.isEmpty {
            let raw = query
            let query = stripOperators(raw)
            if query != raw {
                diag("web_search searching \"\(query)\" "
                    + "(operators stripped from \"\(raw)\")")
            }
            var topk = count
            if topk < 1 { topk = 5 }
            if topk > 20 { topk = 20 }
            result = await WebSearch.run(query, count: topk)
        }
        return result
    }

    public static func fetch(_ url: String, limit: Int, offset: Int,
                             memo: TurnMemo? = nil) async -> String {
        var result = "error: missing 'url' argument"
        if !url.isEmpty {
            var lim = limit
            if lim == 0 { lim = 16_384 }
            switch resolveSource(url) {
            case .url(let href):
                result = await fetchURL(href, lim, offset, memo)
            case .file, .unknown:
                result = "error: '\(url)' is not a recognizable URL"
            }
        }
        return result
    }

    private static func fetchURL(_ href: String, _ limit: Int,
                                 _ offset: Int,
                                 _ memo: TurnMemo?) async -> String {
        var result = "error: fetch failed (network error)"
        if memo?.sliceDelivered(href, offset) == true {
            // Slices note only on successful delivery, so a failed fetch stays
            // retryable.
            diag("fetch_url repeat \(href) offset=\(offset) grounded")
            result = "This exact page (offset \(offset)) was already "
                + "fetched in this turn; its text is above. Do not fetch "
                + "it again: continue from a different offset, fetch a "
                + "DIFFERENT page, or give your final answer now."
        } else if let cached = memo?.page(for: href) {
            memo?.noteSlice(href, offset)
            result = deliverText(cached, limit, offset)
        } else if let url = URL(string: href) {
            var req = URLRequest(url: url)
            req.setValue("Mozilla/5.0 (compatible; chatokf-agent/1.0)",
                         forHTTPHeaderField: "User-Agent")
            let fetched = try? await URLSession.shared.data(for: req)
            if let (data, _) = fetched {
                let text = htmlToText(String(decoding: data, as: UTF8.self))
                memo?.notePage(href, text)
                memo?.noteSlice(href, offset)
                result = deliverText(text, limit, offset)
            }
        }
        return result
    }

    static func deliverText(_ text: String, _ limit: Int,
                            _ offset: Int) -> String {
        let b = Array(text.utf8)
        let tlen = b.count
        let start = max(offset, 0)
        var result = ""
        if tlen == 0 {
            result = "(empty)"
        } else if start >= tlen {
            result = "(offset \(start) is beyond the end; "
                + "\(tlen) bytes total)"
        } else {
            let avail = tlen - start
            var want = limit < 0 ? avail : min(limit, avail)
            if start + want < tlen {
                while want > 0 && (b[start + want] & 0xC0) == 0x80 {
                    want -= 1
                }
            }
            var out = String(
                decoding: b[start..<(start + want)], as: UTF8.self)
            if start + want < tlen {
                let next = start + want
                let more = want > 0 ? (tlen - next + want - 1) / want : 1
                out += more > 2
                    ? "\n...[truncated: bytes \(start)-\(next) of \(tlen) "
                        + "shown; ~\(more) more calls of this size remain. "
                        + "Re-call with offset=\(next) ONLY if you truly "
                        + "need more of this page; do not read whole long "
                        + "pages]"
                    : "\n...[truncated: bytes \(start)-\(next) of \(tlen) "
                        + "shown; re-call with offset=\(next) for more]"
            }
            result = out
        }
        return result
    }

    private static func runNews(_ args: [ToolArg]) async -> String {
        await news(findArg(args, "topic"))
    }

    public static func news(_ topic: String?) async -> String {
        var result = "error: news request failed"
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone(identifier: "UTC")
        fmt.dateFormat = "yyyy/MM/dd"
        let day = fmt.string(from: Date())
        let base = "https://en.wikipedia.org/api/rest_v1/feed/featured/"
        if let url = URL(string: base + day) {
            var req = URLRequest(url: url)
            req.setValue("Mozilla/5.0 (compatible; chatokf-agent/1.0)",
                         forHTTPHeaderField: "User-Agent")
            let fetched = try? await URLSession.shared.data(for: req)
            if let (data, _) = fetched {
                result = newsFormat(data, topic)
            }
        }
        return result
    }

    private static func newsFormat(_ data: Data, _ topic: String?) -> String {
        var result = "(no current headlines)"
        let obj = try? JSONSerialization.jsonObject(with: data)
        if let root = obj as? [String: Any],
           let items = root["news"] as? [[String: Any]] {
            let want = topic?.lowercased()
            var all: [String] = []
            var hits: [String] = []
            for item in items {
                let story = htmlToText(item["story"] as? String ?? "")
                    .replacingOccurrences(of: "\n", with: " ")
                if !story.isEmpty {
                    all.append("- " + story)
                    if let want, !want.isEmpty,
                       story.lowercased().contains(want) {
                        hits.append("- " + story)
                    }
                }
            }
            let filtered = want == nil || want!.isEmpty
            if !filtered && !hits.isEmpty {
                result = "In the news (Wikipedia):\n"
                    + hits.joined(separator: "\n")
            } else if !filtered && !all.isEmpty {
                result = "No headline specifically about \"\(topic!)\"; "
                    + "today's top stories:\n" + all.joined(separator: "\n")
            } else if !all.isEmpty {
                result = "In the news (Wikipedia):\n"
                    + all.joined(separator: "\n")
            }
        }
        return result
    }

    private static func runWeather(_ args: [ToolArg]) async -> String {
        await weather(findArg(args, "location") ?? "")
    }

    public static func weather(_ location: String) async -> String {
        var coord: String? = nil
        var place: String? = nil
        if isLatLon(location) {
            coord = location
        } else if location.isEmpty {
            let ip = await ipInfo()
            coord = ip?.coord; place = ip?.city
        } else {
            let geo = await geocode(location)
            coord = geo?.coord; place = geo?.place
        }
        var result: String? = nil
        if let coord {
            result = await weatherNWS(coord)
            if result == nil { result = await weatherOpenMeteo(coord, place) }
        }
        let what = location.isEmpty ? "your location" : location
        return result ?? "error: could not get weather for \(what)"
    }

    static func isLatLon(_ s: String) -> Bool {
        let parts = s.split(separator: ",")
        var result = false
        if parts.count == 2, let lat = Double(parts[0]),
           let lon = Double(parts[1]) {
            result = lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180
        }
        return result
    }

    private static func ipInfo() async -> (coord: String, city: String?)? {
        var result: (String, String?)? = nil
        if let url = URL(string: "https://ipinfo.io/json") {
            let fetched = try? await URLSession.shared.data(from: url)
            if let (data, _) = fetched,
               let obj = try? JSONSerialization.jsonObject(with: data)
                   as? [String: Any],
               let loc = obj["loc"] as? String, isLatLon(loc) {
                result = (loc, obj["city"] as? String)
            }
        }
        return result
    }

    private static func geocode(_ name: String)
        async -> (coord: String, place: String)? {
        var result: (String, String)? = nil
        let enc = name.addingPercentEncoding(
            withAllowedCharacters: .urlQueryAllowed) ?? name
        let url = "https://geocoding-api.open-meteo.com/v1/search?name="
            + "\(enc)&count=1"
        if let u = URL(string: url),
           let (data, _) = try? await URLSession.shared.data(from: u),
           let obj = try? JSONSerialization.jsonObject(with: data)
               as? [String: Any],
           let r = (obj["results"] as? [[String: Any]])?.first,
           let lat = r["latitude"] as? Double,
           let lon = r["longitude"] as? Double {
            let nm = r["name"] as? String ?? name
            let country = r["country"] as? String ?? ""
            let place = country.isEmpty ? nm : "\(nm), \(country)"
            result = (String(format: "%.4f,%.4f", lat, lon), place)
        }
        return result
    }

    // NWS requires a contact User-Agent. It names the PUBLIC repo so the
    // contact resolves for whoever reads a log at the other end.
    static let nwsAgent =
        "(github.com/leok7v/ChatOKF, leo.kuznetsov@gmail.com)"

    public static func weatherNWS(_ coord: String) async -> String? {
        var result: String? = nil
        let pts = "https://api.weather.gov/points/\(nwsRound(coord))"
        if let (props, _) = await nwsJSON(pts),
           let furl = props["forecast"] as? String,
           let (fprops, _) = await nwsJSON(furl),
           let periods = fprops["periods"] as? [[String: Any]], !periods.isEmpty {
            var parts: [String] = []
            for p in periods.prefix(6) {
                let name = p["name"] as? String ?? "?"
                let temp = p["temperature"] as? Int ?? 0
                let unit = p["temperatureUnit"] as? String ?? "F"
                let sf = p["shortForecast"] as? String ?? ""
                parts.append("\(name): \(temp)\(unit) \(sf)")
            }
            result = "Weather forecast for \(nwsPlace(props)): "
                + parts.joined(separator: "; ")
        }
        return result
    }

    private static func nwsJSON(_ url: String) async
        -> (props: [String: Any], relative: [String: Any]?)? {
        var result: (props: [String: Any], relative: [String: Any]?)? = nil
        if let u = URL(string: url) {
            var req = URLRequest(url: u)
            req.setValue(nwsAgent, forHTTPHeaderField: "User-Agent")
            let fetched = try? await URLSession.shared.data(for: req)
            if let (data, resp) = fetched,
               (resp as? HTTPURLResponse)?.statusCode == 200,
               let obj = try? JSONSerialization.jsonObject(with: data)
                   as? [String: Any],
               let props = obj["properties"] as? [String: Any] {
                result = (props, nil)
            }
        }
        return result
    }

    private static func nwsPlace(_ props: [String: Any]) -> String {
        var result = "your area"
        if let rel = (props["relativeLocation"] as? [String: Any])?["properties"]
            as? [String: Any], let city = rel["city"] as? String {
            let state = rel["state"] as? String ?? ""
            result = state.isEmpty ? city : "\(city), \(state)"
        }
        return result
    }

    // NWS 301-redirects when a coordinate carries more than 4 decimal places.
    private static func nwsRound(_ coord: String) -> String {
        let parts = coord.split(separator: ",")
        var result = coord
        if parts.count == 2, let a = Double(parts[0]), let b = Double(parts[1]) {
            result = String(format: "%.4f,%.4f", a, b)
        }
        return result
    }

    // Attribution rides in the output, per Open-Meteo's CC BY 4.0 licence.
    static func weatherOpenMeteo(_ coord: String, _ place: String?)
        async -> String? {
        var result: String? = nil
        let parts = coord.split(separator: ",")
        if parts.count == 2 {
            let url = "https://api.open-meteo.com/v1/forecast?latitude="
                + "\(parts[0])&longitude=\(parts[1])&current=temperature_2m,"
                + "apparent_temperature,relative_humidity_2m,weather_code,"
                + "wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,"
                + "weather_code&timezone=auto&forecast_days=4"
            if let u = URL(string: url),
               let (data, _) = try? await URLSession.shared.data(from: u),
               let obj = try? JSONSerialization.jsonObject(with: data)
                   as? [String: Any],
               let cur = obj["current"] as? [String: Any] {
                result = openMeteoFormat(cur, obj["daily"] as? [String: Any],
                                         place ?? "your area")
            }
        }
        return result
    }

    private static func openMeteoFormat(_ cur: [String: Any],
                                        _ daily: [String: Any]?,
                                        _ place: String) -> String {
        let t = cur["temperature_2m"] as? Double ?? 0
        let feels = cur["apparent_temperature"] as? Double ?? t
        let hum = cur["relative_humidity_2m"] as? Int ?? 0
        let wind = cur["wind_speed_10m"] as? Double ?? 0
        let cond = wmoText(cur["weather_code"] as? Int ?? 0)
        var line = "Weather now in \(place): \(cond), \(cf(t)) "
            + "(feels \(cf(feels))), humidity \(hum)%, wind \(Int(wind)) km/h"
        if let daily, let hi = daily["temperature_2m_max"] as? [Double],
           let lo = daily["temperature_2m_min"] as? [Double],
           let codes = daily["weather_code"] as? [Int] {
            let labels = ["today", "tomorrow"]
            var parts: [String] = []
            var i = 0
            while i < hi.count && i < lo.count && i < codes.count && i < 3 {
                let label = i < labels.count ? labels[i] : "day \(i + 1)"
                parts.append("\(label) \(cfRange(lo[i], hi[i])) "
                    + wmoText(codes[i]))
                i += 1
            }
            if !parts.isEmpty { line += ". Forecast: " + parts.joined(
                separator: "; ") }
        }
        return line + " (Weather data by Open-Meteo.com)"
    }

    private static func cf(_ c: Double) -> String {
        String(format: "%.0fC/%.0fF", c, c * 9 / 5 + 32)
    }
    private static func cfRange(_ loC: Double, _ hiC: Double) -> String {
        String(format: "%.0f-%.0fC (%.0f-%.0fF)",
               loC, hiC, loC * 9 / 5 + 32, hiC * 9 / 5 + 32)
    }

    private static let wmoCodes: [Int: String] = [
        0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
        61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
        67: "Freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Light showers", 81: "Showers",
        82: "Violent showers", 85: "Snow showers", 86: "Snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail",
        99: "Thunderstorm with hail",
    ]
    private static func wmoText(_ code: Int) -> String {
        wmoCodes[code] ?? "code \(code)"
    }

    static func runWikipedia(_ args: [ToolArg], slugsPath: String,
                             memo: TurnMemo? = nil) async -> String {
        var result = "error: missing 'query' argument"
        if let query = findArg(args, "query"), !query.isEmpty {
            result = await wikipediaQuery(query, slugsPath: slugsPath,
                                          memo: memo)
        }
        return result
    }

    static func stripMetaWords(_ query: String) -> String {
        var s = " " + query.lowercased() + " "
        for phrase in ["simple english wikipedia", "simple english",
                       "wikipedia article", "wikipedia articles",
                       "wikipedia", "encyclopedia", "according to"] {
            s = s.replacingOccurrences(of: " " + phrase + " ", with: " ")
        }
        let cleaned = s.trimmingCharacters(in: .whitespaces)
        return cleaned.count >= 3 ? cleaned : query
    }

    static func needsTitleRescue(_ best: SlugHit?,
                                 _ query: String) -> Bool {
        var rescue = true
        if let best, best.isConfident {
            let hay = " " + WikiSlugs.foldWords(query) + " "
            rescue = !hay.contains(
                " " + WikiSlugs.foldWords(best.title) + " ")
        }
        return rescue
    }

    public static func wikipediaQuery(_ query: String, slugsPath: String,
                                      topK: Int = 5, limit: Int = 4000,
                                      memo: TurnMemo? = nil) async -> String {
        var result = "error: wikipedia index unavailable"
        if let w = WikiSlugs(ggufPath: slugsPath) {
            let cleaned = Tools.stripMetaWords(query)
            if cleaned != query.lowercased()
                .trimmingCharacters(in: .whitespaces) {
                diag("wikipedia_query embedding \"\(cleaned)\" "
                    + "(meta words stripped)")
            }
            var best = w.query(cleaned, topK: max(1, topK)).first
            if Tools.needsTitleRescue(best, cleaned),
               let hit = w.titleMatch(cleaned) {
                diag("wikipedia_query title match \"\(hit.title)\""
                    + (best.map { b in
                        " (embedding picked \"\(b.title)\" d=\(b.distance))"
                    } ?? ""))
                best = hit
            }
            let dup = best.flatMap { b in
                b.isConfident ? memo?.title(for: b.id) : nil
            }
            var body = ""
            var related: [String] = []
            if let best, best.isConfident, dup == nil {
                let page = await wikipediaExtract(best.id)
                body = clampText(page?.text ?? "", limit)
                related = relatedIn(body, links: page?.links ?? [])
            }
            if let dup, let best {
                diag("wikipedia_query dedupe \"\(dup)\" id=\(best.id)")
                result = "Article \"\(dup)\" was already retrieved in this "
                    + "turn; its full text is above. Do not request it "
                    + "again: use it and its Related articles list, query a "
                    + "DIFFERENT topic, or give your final answer now."
            } else if let best, best.isConfident, !body.isEmpty {
                memo?.note(best.id, best.title)
                result = "Simple English Wikipedia -- article "
                    + "\"\(best.title)\":\n\(body)"
                if !related.isEmpty {
                    result += "\n\nRelated articles: "
                        + related.joined(separator: "; ")
                }
            } else {
                let near = best.map { " (nearest \"\($0.title)\" is weak)" }
                    ?? ""
                if let best {
                    diag("wikipedia_query \"\(query)\": nearest "
                        + "\"\(best.title)\" id=\(best.id) "
                        + "d=\(best.distance) (cutoff "
                        + "\(SlugHit.lowConfidenceDistance))"
                        + (best.isConfident ? ", extract EMPTY" : ""))
                }
                result = "No confident Simple English Wikipedia match for "
                    + "\"\(query)\"\(near). Answer from your own knowledge."
            }
        }
        return result
    }

    static func wikipediaExtract(_ id: String) async
        -> (text: String, links: [String])? {
        var result: (text: String, links: [String])? = nil
        var comps = URLComponents(
            string: "https://simple.wikipedia.org/w/api.php")
        comps?.queryItems = [
            URLQueryItem(name: "action", value: "query"),
            URLQueryItem(name: "prop", value: "extracts|links"),
            URLQueryItem(name: "explaintext", value: "1"),
            URLQueryItem(name: "plnamespace", value: "0"),
            URLQueryItem(name: "pllimit", value: "max"),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "pageids", value: id),
        ]
        if let url = comps?.url {
            var req = URLRequest(url: url)
            req.setValue("Mozilla/5.0 (compatible; chatokf-agent/1.0)",
                         forHTTPHeaderField: "User-Agent")
            do {
                let (data, resp) = try await URLSession.shared.data(for: req)
                if let obj = try? JSONSerialization.jsonObject(with: data)
                       as? [String: Any],
                   let query = obj["query"] as? [String: Any],
                   let pages = query["pages"] as? [String: Any],
                   let page = pages.values.first as? [String: Any],
                   let extract = page["extract"] as? String {
                    let links = (page["links"] as? [[String: Any]] ?? [])
                        .compactMap { link in link["title"] as? String }
                    result = (extract, links)
                } else {
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                    diag("wikipedia extract id=\(id) -> HTTP \(code), "
                        + "\(data.count) bytes: "
                        + String(decoding: data.prefix(300), as: UTF8.self))
                }
            } catch {
                diag("wikipedia extract id=\(id) transport error: "
                    + error.localizedDescription)
            }
        }
        return result
    }

    static func relatedIn(_ body: String, links: [String],
                          cap: Int = 25) -> [String] {
        let hay = body.lowercased()
        var out: [String] = []
        for title in links where out.count < cap {
            let numeric = title.allSatisfy { c in c.isNumber }
            if title.count >= 3, !numeric,
               hay.contains(title.lowercased()) {
                out.append(title)
            }
        }
        return out
    }

    static func clampText(_ text: String, _ limit: Int) -> String {
        var s = text
        while s.count > limit,
              let r = s.range(of: "\n\n", options: .backwards) {
            s = String(s[..<r.lowerBound])
        }
        while s.count > limit, let r = s.range(of: ". ", options: .backwards) {
            s = String(s[..<r.upperBound])
                .trimmingCharacters(in: .whitespaces)
        }
        while s.count > limit, let r = s.range(of: " ", options: .backwards) {
            s = String(s[..<r.lowerBound])
        }
        if s.count > limit { s = String(s.prefix(limit)) }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func htmlToText(_ html: String) -> String {
        collapseWhitespace(htmlStripped(mainContent(Array(html))))
    }

    private static func htmlStripped(_ cs: [Character]) -> [Character] {
        var raw: [Character] = []
        var p = 0
        let n = cs.count
        while p < n {
            let c = cs[p]
            if c == "&" {
                if let dec = decodeEntity(cs, p) {
                    raw.append(contentsOf: dec.text)
                    p += dec.consumed
                } else {
                    raw.append(c)
                    p += 1
                }
            } else if c != "<" {
                raw.append(c)
                p += 1
            } else {
                p = consumeTag(cs, p, &raw)
            }
        }
        return raw
    }

    private static func consumeTag(_ cs: [Character], _ p: Int,
                                   _ raw: inout [Character]) -> Int {
        let n = cs.count
        var next = n
        if ciStarts(cs, p, Array("<!--")) {
            next = find(cs, Array("-->"), p + 4).map { e in e + 3 } ?? n
        } else {
            let closing = p + 1 < n && cs[p + 1] == "/"
            let nameAt = closing ? p + 2 : p + 1
            let drop = closing ? nil : dropTagAt(cs, nameAt)
            if let drop {
                next = skipElement(cs, p, drop, &raw)
            } else {
                if isBlockTag(cs, nameAt) { raw.append("\n") }
                next = find(cs, [">"], p).map { g in g + 1 } ?? n
            }
        }
        return next
    }

    private static let dropTags = ["script", "style", "nav", "header",
                                   "footer", "aside", "form", "math"]

    // The delimiter check stops <form> matching <format> and <nav> <navbar>.
    private static func dropTagAt(_ cs: [Character], _ at: Int) -> [Character]? {
        let delims: Set<Character> = [">", " ", "/", "\t", "\n", "\r"]
        var result: [Character]? = nil
        var k = 0
        while result == nil && k < dropTags.count {
            let tag = Array(dropTags[k])
            let after = at + tag.count
            if ciStarts(cs, at, tag) && after < cs.count
                && delims.contains(cs[after]) {
                result = tag
            }
            k += 1
        }
        return result
    }

    private static func skipElement(_ cs: [Character], _ p: Int,
                                    _ tag: [Character],
                                    _ raw: inout [Character]) -> Int {
        let close = Array("</") + tag
        var next = cs.count
        if let e = find(cs, close, p + 1, ci: true) {
            next = find(cs, [">"], e).map { g in g + 1 } ?? cs.count
        }
        raw.append("\n")
        return next
    }

    private static func mainContent(_ cs: [Character]) -> [Character] {
        let inner = elementInner(cs, Array("main"))
            ?? elementInner(cs, Array("article"))
        return inner ?? cs
    }

    private static func elementInner(_ cs: [Character],
                                     _ tag: [Character]) -> [Character]? {
        let delims: Set<Character> = [">", " ", "/", "\t", "\n", "\r"]
        var result: [Character]? = nil
        let open = Array("<") + tag
        if let start = find(cs, open, 0, ci: true) {
            let after = start + open.count
            if after < cs.count && delims.contains(cs[after]),
               let gt = find(cs, [">"], after),
               let close = lastFind(cs, Array("</") + tag, gt + 1, ci: true) {
                result = Array(cs[(gt + 1)..<close])
            }
        }
        return result
    }

    private static func lastFind(_ cs: [Character], _ needle: [Character],
                                 _ from: Int, ci: Bool = false) -> Int? {
        var result: Int? = nil
        var i = cs.count - needle.count
        while result == nil && i >= from {
            var j = 0
            while j < needle.count && charEq(cs[i + j], needle[j], ci) {
                j += 1
            }
            if j == needle.count { result = i }
            i -= 1
        }
        return result
    }

    private static let blockTags = [
        "p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "table", "section", "article", "header", "footer",
        "blockquote", "pre", "hr",
    ]

    private static func isBlockTag(_ cs: [Character], _ at: Int) -> Bool {
        let delims: Set<Character> = [">", " ", "/", "\t", "\n", "\r"]
        var result = false
        var k = 0
        while !result && k < blockTags.count {
            let tag = Array(blockTags[k])
            let after = at + tag.count
            if ciStarts(cs, at, tag) && after < cs.count
                && delims.contains(cs[after]) {
                result = true
            }
            k += 1
        }
        return result
    }

    private static func decodeEntity(_ cs: [Character], _ p: Int)
        -> (text: [Character], consumed: Int)? {
        let n = cs.count
        var result: (text: [Character], consumed: Int)? = nil
        if p + 1 < n, let semi = find(cs, [";"], p + 1),
           semi - p <= 12, semi > p + 1 {
            if cs[p + 1] == "#" {
                result = decodeNumericEntity(cs, p, semi)
            } else if let cp = namedEntities[String(cs[(p + 1)..<semi])] {
                result = (utf8Chars(cp), semi - p + 1)
            }
        }
        return result
    }

    private static func decodeNumericEntity(_ cs: [Character], _ p: Int,
                                            _ semi: Int)
        -> (text: [Character], consumed: Int)? {
        let hex = p + 2 < cs.count
            && (cs[p + 2] == "x" || cs[p + 2] == "X")
        let from = hex ? p + 3 : p + 2
        var result: (text: [Character], consumed: Int)? = nil
        if let cp = parseCodepoint(cs, from, semi, hex), cp != 0 {
            result = (utf8Chars(cp), semi - p + 1)
        }
        return result
    }

    private static func parseCodepoint(_ cs: [Character], _ from: Int,
                                       _ to: Int, _ hex: Bool) -> UInt32? {
        var cp: UInt32? = from < to ? 0 : nil
        let base: UInt32 = hex ? 16 : 10
        var q = from
        while q < to {
            if let acc = cp {
                let d = digitValue(cs[q], hex)
                cp = d >= 0 ? acc &* base &+ UInt32(d) : nil
            }
            q += 1
        }
        return cp
    }

    private static func digitValue(_ c: Character, _ hex: Bool) -> Int {
        var result = -1
        if let a = c.asciiValue {
            if a >= 48 && a <= 57 {
                result = Int(a - 48)
            } else if hex && a >= 97 && a <= 102 {
                result = Int(a - 97 + 10)
            } else if hex && a >= 65 && a <= 70 {
                result = Int(a - 65 + 10)
            }
        }
        return result
    }

    private static let namedEntities: [String: UInt32] = [
        "amp": 0x26, "lt": 0x3c, "gt": 0x3e, "quot": 0x22, "apos": 0x27,
        "nbsp": 0x20, "copy": 0xa9, "reg": 0xae, "mdash": 0x2014,
        "ndash": 0x2013, "hellip": 0x2026, "rsquo": 0x2019,
        "lsquo": 0x2018, "ldquo": 0x201c, "rdquo": 0x201d,
        "trade": 0x2122, "deg": 0xb0,
    ]

    private static func utf8Chars(_ cp: UInt32) -> [Character] {
        Unicode.Scalar(cp).map { s in [Character(s)] } ?? []
    }

    private static func collapseWhitespace(_ raw: [Character]) -> String {
        var clean: [Character] = []
        var nl = 0
        var sp = false
        for c in raw {
            if c == "\n" {
                nl += 1
                sp = false
            } else if c == " " || c == "\t" || c == "\r" {
                sp = true
            } else {
                if !clean.isEmpty {
                    if nl >= 2 {
                        clean.append(contentsOf: "\n\n")
                    } else if nl == 1 {
                        clean.append("\n")
                    } else if sp {
                        clean.append(" ")
                    }
                }
                nl = 0
                sp = false
                clean.append(c)
            }
        }
        return String(clean)
    }

    private static func starts(_ cs: [Character],
                               _ prefix: [Character]) -> Bool {
        var j = 0
        while j < prefix.count && j < cs.count && cs[j] == prefix[j] {
            j += 1
        }
        return j == prefix.count
    }

    private static func startsAt(_ cs: [Character], _ at: Int,
                                 _ prefix: [Character]) -> Bool {
        var j = 0
        while j < prefix.count && at + j < cs.count
            && cs[at + j] == prefix[j] {
            j += 1
        }
        return j == prefix.count
    }

    private static func ciStarts(_ cs: [Character], _ at: Int,
                                 _ kw: [Character]) -> Bool {
        var j = 0
        while j < kw.count && at + j < cs.count
            && lower(cs[at + j]) == kw[j] {
            j += 1
        }
        return j == kw.count
    }

    private static func lower(_ c: Character) -> Character {
        var result = c
        if c >= "A" && c <= "Z", let a = c.asciiValue {
            result = Character(Unicode.Scalar(a + 32))
        }
        return result
    }

    private static func find(_ cs: [Character], _ needle: [Character],
                             _ from: Int, ci: Bool = false) -> Int? {
        var result: Int? = nil
        var i = max(from, 0)
        let last = cs.count - needle.count
        while result == nil && i <= last {
            var j = 0
            while j < needle.count && charEq(cs[i + j], needle[j], ci) {
                j += 1
            }
            if j == needle.count { result = i }
            i += 1
        }
        return result
    }

    private static func charEq(_ a: Character, _ b: Character,
                               _ ci: Bool) -> Bool {
        ci ? lower(a) == lower(b) : a == b
    }
}
