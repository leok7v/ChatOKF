import Foundation
import Testing
@testable import LLM

private let wikipediaExcerpt = """
move to sidebar hide
* (Top)
* 1 Track listing
Toggle the table of contents
Hello World (Scandal album)
7 languages
From Wikipedia, the free encyclopedia
Hello World is the sixth studio album by Japanese pop rock band, Scandal .
The album was released on December 3, 2014, in Japan by Epic and being \
distributed in Europe through JPU Records.
Hidden categories:
* Articles with short description
"""

private let parallelBody = """
{"search_id": "s1", "results": [
  {"url": "https://en.wikipedia.org/wiki/Hello_World",
   "title": "Hello World (Scandal album) - Wikipedia",
   "publish_date": "2025-12-06",
   "excerpts": ["\(wikipediaExcerpt.replacingOccurrences(of: "\n",
                                                        with: "\\n"))"]},
  {"url": "https://example.com/b", "title": "Second",
   "excerpts": ["A second result whose only line is long enough to read."]}
]}
"""

private let mwmblBody = """
[{"url": "https://example.com/a",
  "title": [{"value": "Hello ", "is_bold": false},
            {"value": "World", "is_bold": true}],
  "extract": [{"value": "An extract in spans.", "is_bold": false}]}]
"""

@Suite struct WebSearchTests {

    @Test func excerptKeepsProseAndDropsNavigationChrome() {
        let out = WebSearch.excerpt([wikipediaExcerpt])
        #expect(out.contains("sixth studio album"))
        #expect(out.contains("JPU Records"))
        #expect(!out.contains("move to sidebar"))
        #expect(!out.contains("Toggle the table"))
        #expect(!out.contains("Hidden categories"))
        #expect(!out.contains("7 languages"))
    }

    @Test func excerptIsClampedAndCarriesNoNewlines() {
        let long = String(repeating: "Every line here is prose, long "
                          + "enough to be kept by the filter.\n", count: 40)
        let out = WebSearch.excerpt([long])
        #expect(out.count <= WebSearch.excerptLimit)
        #expect(!out.contains("\n"))
    }

    @Test func clampingTerminatesOnAParagraphlessRunOfSentences() {
        let run = String(repeating: "One sentence here. ", count: 200)
        let out = Tools.clampText(run, 100)
        #expect(out.count <= 100)
        #expect(out.hasPrefix("One sentence here."))
    }

    @Test func excerptFallsBackWhenEveryLineIsShort() {
        let out = WebSearch.excerpt(["tiny\nbits\nonly"])
        #expect(out == "tiny bits only")
    }

    @Test func parallelResultsBecomeRankedHits() {
        let hits = WebSearch.parallelHits(parallelBody, 5)
        #expect(hits.count == 2)
        #expect(hits[0].url == "https://en.wikipedia.org/wiki/Hello_World")
        #expect(hits[0].title.hasPrefix("Hello World"))
        #expect(hits[0].extract.contains("sixth studio album"))
        #expect(hits[1].title == "Second")
    }

    @Test func parallelResultsHonourTheRequestedCount() {
        #expect(WebSearch.parallelHits(parallelBody, 1).count == 1)
    }

    @Test func parallelJunkYieldsNothingRatherThanThrowing() {
        #expect(WebSearch.parallelHits("not json at all", 5).isEmpty)
        #expect(WebSearch.parallelHits("{}", 5).isEmpty)
    }

    @Test func mwmblSpansAreJoinedIntoOneLine() {
        let hits = WebSearch.mwmblHits(Data(mwmblBody.utf8), 5)
        #expect(hits.count == 1)
        #expect(hits[0].title == "Hello World")
        #expect(hits[0].extract == "An extract in spans.")
    }

    @Test func anEmptyResultSetGroundsTheModel() {
        let out = WebSearch.format([], "dark matter")
        #expect(out.contains("No web results for \"dark matter\""))
        #expect(out.contains("your own knowledge"))
    }

    @Test func formattedHitsAreNumberedWithUrlAndExtract() {
        let out = WebSearch.format(
            [SearchHit(title: "One", url: "https://a", extract: "First."),
             SearchHit(title: "Two", url: "https://b", extract: "Second.")],
            "q")
        #expect(out == "1. One\n   https://a\n   First.\n"
                + "2. Two\n   https://b\n   Second.\n")
    }

    @Test func aHitWithNoTitleStillReads() {
        let out = WebSearch.format(
            [SearchHit(title: "", url: "https://a", extract: "")], "q")
        #expect(out == "1. (no title)\n   https://a\n")
    }

    @Test func anEventStreamReplyIsDecoded() {
        let sse = "event: message\ndata: {\"result\":{\"ok\":true}}\n\n"
        let out = ParallelSearch.decode(Data(sse.utf8))
        #expect(out["result"] != nil)
    }

    @Test func aPlainJsonReplyIsDecoded() {
        let out = ParallelSearch.decode(Data("{\"id\":1}".utf8))
        #expect(out["id"] != nil)
    }

    @Test func toolTextIsJoinedFromContentBlocks() {
        let result: [String: Any] = [
            "content": [["type": "text", "text": "abc"],
                        ["type": "text", "text": "def"]]]
        #expect(ParallelSearch.text(result) == "abcdef")
    }

    @Test func toolArgumentsFollowTheAdvertisedSchema() {
        let tools: [[String: Any]] = [[
            "name": "web_search",
            "inputSchema": [
                "required": ["objective", "search_queries"],
                "properties": [
                    "objective": ["type": "string"],
                    "search_queries": ["type": "array"]]]]]
        let tool = MCPTool(tools, preferring: "web_search")
        let args = tool.arguments("dark matter")
        #expect(tool.name == "web_search")
        #expect(args["objective"] as? String == "dark matter")
        #expect(args["search_queries"] as? [String] == ["dark matter"])
    }

    @Test func anUnknownSchemaStillNamesTheWantedTool() {
        let tool = MCPTool([], preferring: "web_search")
        #expect(tool.name == "web_search")
        #expect(tool.arguments("x").isEmpty)
    }

    @Test func bothProvidersAreOnByDefault() {
        #expect(SearchProvider.parallel.on)
        #expect(SearchProvider.mwmbl.on)
        #expect(SearchProvider.any)
    }

    @Test func everyProviderNamesAPolicyAndAHome() {
        for provider in SearchProvider.allCases {
            #expect(URL(string: provider.home) != nil)
            #expect(URL(string: provider.privacy) != nil)
            #expect(!provider.label.isEmpty)
            #expect(!provider.detail.isEmpty)
        }
    }
}
