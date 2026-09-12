import Foundation
import Testing
@testable import LLM

struct Docs2mdTests {

    private static func crc32(_ data: Data) -> UInt32 {
        var result: UInt32 = 0xFFFF_FFFF
        for byte in data {
            result ^= UInt32(byte)
            for _ in 0 ..< 8 {
                let low = result & 1
                result >>= 1
                if low == 1 { result ^= 0xEDB8_8320 }
            }
        }
        return result ^ 0xFFFF_FFFF
    }

    private static func le(_ value: Int, _ width: Int) -> Data {
        var out = Data()
        for step in 0 ..< width {
            out.append(UInt8((value >> (8 * step)) & 0xFF))
        }
        return out
    }

    // Stored entries only, which is all the reader needs and all a fixture
    // has any reason to be.
    private static func archive(_ parts: [(String, String)]) -> Data {
        var body = Data()
        var catalogue = Data()
        for (name, text) in parts {
            let content = Data(text.utf8)
            let named = Data(name.utf8)
            let sum = Int(crc32(content))
            let offset = body.count
            body.append(le(0x0403_4b50, 4))
            body.append(le(20, 2))
            body.append(le(0, 2))
            body.append(le(0, 2))
            body.append(le(0, 2))
            body.append(le(0, 2))
            body.append(le(sum, 4))
            body.append(le(content.count, 4))
            body.append(le(content.count, 4))
            body.append(le(named.count, 2))
            body.append(le(0, 2))
            body.append(named)
            body.append(content)
            catalogue.append(le(0x0201_4b50, 4))
            catalogue.append(le(20, 2))
            catalogue.append(le(20, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(sum, 4))
            catalogue.append(le(content.count, 4))
            catalogue.append(le(content.count, 4))
            catalogue.append(le(named.count, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 2))
            catalogue.append(le(0, 4))
            catalogue.append(le(offset, 4))
            catalogue.append(named)
        }
        var out = body
        out.append(catalogue)
        out.append(le(0x0605_4b50, 4))
        out.append(le(0, 2))
        out.append(le(0, 2))
        out.append(le(parts.count, 2))
        out.append(le(parts.count, 2))
        out.append(le(catalogue.count, 4))
        out.append(le(body.count, 4))
        out.append(le(0, 2))
        return out
    }

    private static func file(_ name: String, _ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "-" + name)
        try data.write(to: url)
        return url
    }

    private static func read(_ url: URL) throws -> String {
        defer { try? FileManager.default.removeItem(at: url) }
        return try Docs2md.markdown(of: url)
    }

    private static let head = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    private static let wns =
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""
    private static let rns =
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
    private static let pns =
        "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\""
    private static let ans =
        "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
    private static let relns =
        "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\""

    private static func run(_ text: String) -> String {
        "<w:r><w:t>\(text)</w:t></w:r>"
    }

    private static func cell(_ text: String) -> String {
        "<w:tc><w:p>\(run(text))</w:p></w:tc>"
    }

    private static func docx() -> Data {
        let document = """
        \(head)
        <w:document \(wns)><w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>\
        \(run("Harvest Report"))</w:p>
        <w:p>\(run("Rain arrived late in the valley this year."))</w:p>
        <w:tbl>
        <w:tr>\(cell("Block"))\(cell("Tonnes"))</w:tr>
        <w:tr>\(cell("Eastern"))\(cell("47.2"))</w:tr>
        </w:tbl>
        </w:body></w:document>
        """
        return archive([("word/document.xml", document),
                        ("word/_rels/document.xml.rels",
                         "\(head)<Relationships \(relns)/>")])
    }

    private static func xlsx() -> Data {
        let workbook = """
        \(head)
        <workbook \(rns)><sheets>
        <sheet name="Harvest" sheetId="1" r:id="rId1"/>
        </sheets></workbook>
        """
        let rels = """
        \(head)
        <Relationships \(relns)>
        <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
        </Relationships>
        """
        func inline(_ text: String) -> String {
            "<c t=\"inlineStr\"><is><t>\(text)</t></is></c>"
        }
        let sheet = """
        \(head)
        <worksheet><sheetData>
        <row r="1">\(inline("Block"))\(inline("Tonnes"))</row>
        <row r="2">\(inline("Riverbank"))\(inline("19.4"))</row>
        </sheetData></worksheet>
        """
        return archive([("xl/workbook.xml", workbook),
                        ("xl/_rels/workbook.xml.rels", rels),
                        ("xl/worksheets/sheet1.xml", sheet)])
    }

    private static func pptx() -> Data {
        let presentation = """
        \(head)
        <p:presentation \(pns) \(rns)><p:sldIdLst>
        <p:sldId id="256" r:id="rId1"/>
        </p:sldIdLst></p:presentation>
        """
        let rels = """
        \(head)
        <Relationships \(relns)>
        <Relationship Id="rId1" Target="slides/slide1.xml"/>
        </Relationships>
        """
        func shape(_ text: String, title: Bool) -> String {
            let placeholder = title
                ? "<p:nvSpPr><p:nvPr><p:ph type=\"title\"/></p:nvPr></p:nvSpPr>"
                : ""
            return "<p:sp>\(placeholder)<p:txBody><a:p><a:r>"
                + "<a:t>\(text)</a:t></a:r></a:p></p:txBody></p:sp>"
        }
        let slide = """
        \(head)
        <p:sld \(pns) \(ans)><p:cSld><p:spTree>
        \(shape("Third Quarter", title: true))
        \(shape("The cider press ran a second shift.", title: false))
        </p:spTree></p:cSld></p:sld>
        """
        return archive([("ppt/presentation.xml", presentation),
                        ("ppt/_rels/presentation.xml.rels", rels),
                        ("ppt/slides/slide1.xml", slide)])
    }

    private static func epub() -> Data {
        let container = """
        \(head)
        <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
        <rootfiles>
        <rootfile full-path="OEBPS/content.opf"\
         media-type="application/oebps-package+xml"/>
        </rootfiles></container>
        """
        let opf = """
        \(head)
        <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
        <manifest>
        <item id="one" href="chapter.xhtml" media-type="application/xhtml+xml"/>
        </manifest>
        <spine><itemref idref="one"/></spine>
        </package>
        """
        let chapter = """
        \(head)
        <html xmlns="http://www.w3.org/1999/xhtml"><body>
        <h1>Riverbank</h1>
        <p>It carries the fewest trees and returns the most fruit.</p>
        </body></html>
        """
        return archive([("META-INF/container.xml", container),
                        ("OEBPS/content.opf", opf),
                        ("OEBPS/chapter.xhtml", chapter)])
    }

    private static let html = """
    <html><body>
    <h1>Northwind Orchards</h1>
    <p>The pickers finished the eastern blocks a week behind schedule.</p>
    <table>
    <tr><th>Block</th><th>Tonnes</th></tr>
    <tr><td>Western</td><td>22.9</td></tr>
    </table>
    </body></html>
    """

    // OOXML STATES its structure, so a heading is a heading and a table is a
    // table without anything being inferred from geometry.
    @Test func wordKeepsHeadingProseAndTable() throws {
        let url = try Docs2mdTests.file("a.docx", Docs2mdTests.docx())
        let markdown = try Docs2mdTests.read(url)
        #expect(markdown.contains("# Harvest Report"), "\(markdown)")
        #expect(markdown.contains("Rain arrived late"), "\(markdown)")
        #expect(markdown.contains("| Block | Tonnes |"), "\(markdown)")
        #expect(markdown.contains("| Eastern | 47.2 |"), "\(markdown)")
    }

    @Test func sheetKeepsItsRows() throws {
        let url = try Docs2mdTests.file("a.xlsx", Docs2mdTests.xlsx())
        let markdown = try Docs2mdTests.read(url)
        #expect(markdown.contains("Block"), "\(markdown)")
        #expect(markdown.contains("Riverbank"), "\(markdown)")
        #expect(markdown.contains("19.4"), "\(markdown)")
    }

    @Test func slideKeepsItsTitleAndBody() throws {
        let url = try Docs2mdTests.file("a.pptx", Docs2mdTests.pptx())
        let markdown = try Docs2mdTests.read(url)
        #expect(markdown.contains("Third Quarter"), "\(markdown)")
        #expect(markdown.contains("second shift"), "\(markdown)")
    }

    @Test func bookFollowsItsSpine() throws {
        let url = try Docs2mdTests.file("a.epub", Docs2mdTests.epub())
        let markdown = try Docs2mdTests.read(url)
        #expect(markdown.contains("Riverbank"), "\(markdown)")
        #expect(markdown.contains("fewest trees"), "\(markdown)")
    }

    @Test func markupKeepsHeadingProseAndTable() throws {
        let url = try Docs2mdTests.file("a.html",
                                        Data(Docs2mdTests.html.utf8))
        let markdown = try Docs2mdTests.read(url)
        #expect(markdown.contains("# Northwind Orchards"), "\(markdown)")
        #expect(markdown.contains("behind schedule"), "\(markdown)")
        #expect(markdown.contains("| Western | 22.9 |"), "\(markdown)")
    }

    // The fixtures above are the MINIMUM each reader must handle, cut to the
    // few parts it opens.
    private static let written = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures/docs")

    // The assertion is a ROUND TRIP -- Markdown out to a document and back --
    private static let table = """
        | Block | Trees | Tonnes | Yield per tree | To press |
        | --- | --- | --- | --- | --- |
        | Eastern | 840 | 47.2 | 56.2kg | 18% |
        | Western | 610 | 22.9 | 37.5kg | 41% |
        | Riverbank | 295 | 19.4 | 65.8kg | 9% |
        | Nursery | 150 | 3.1 | 20.7kg | 0% |
        """

    @Test func aWrittenWordFileRoundTrips() throws {
        let markdown = try Docs2md.markdown(
            of: Docs2mdTests.written
                .appendingPathComponent("harvest-report.docx"))
        #expect(markdown.contains("# Northwind Orchards"), "\(markdown)")
        #expect(markdown.contains("## Harvest Report, Third Quarter"),
                "\(markdown)")
        #expect(markdown.contains(Docs2mdTests.table), "\(markdown)")
        #expect(markdown.contains("Riverbank remains the puzzle"),
                "\(markdown)")
    }

    // A deck splits at its headings, so the same source becomes slides -- and
    // the table has to survive that too, sitting inside one of them.
    @Test func aWrittenDeckKeepsItsSlidesAndTable() throws {
        let markdown = try Docs2md.markdown(
            of: Docs2mdTests.written
                .appendingPathComponent("harvest-report.pptx"))
        #expect(markdown.contains("## Slide 1"), "\(markdown)")
        #expect(markdown.contains("## Slide 2"), "\(markdown)")
        #expect(markdown.contains("### Northwind Orchards"), "\(markdown)")
        #expect(markdown.contains(Docs2mdTests.table), "\(markdown)")
    }

    // An extension with no reader says so, rather than returning empty text
    // that a caller would attach to a turn as if the file had been read.
    @Test func anUnknownFormatIsRefused() throws {
        let url = try Docs2mdTests.file("a.rtf", Data("plain".utf8))
        #expect(throws: DocsError.self) {
            try Docs2md.markdown(of: url)
        }
        try? FileManager.default.removeItem(at: url)
    }
}
