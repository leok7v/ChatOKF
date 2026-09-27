import Foundation
import Testing
@testable import MD

@Suite struct ColumnLayoutTests {

    @Test func widthsThatFitAreLeftAlone() {
        let fit = TableMetrics.columnLayout(
            headers: ["a", "b"], rows: [["c", "d"]],
            natural: [100, 120], minimums: [30, 30], available: 400)
        #expect(fit.widths == [100, 120])
        #expect(!fit.wrap)
    }

    @Test func anOverflowingTableIsRedistributedAndWraps() {
        let fit = TableMetrics.columnLayout(
            headers: ["a", "b"], rows: [["c", "d"]],
            natural: [300, 300], minimums: [30, 30], available: 200)
        #expect(fit.wrap)
        #expect(abs(fit.widths.reduce(0, +) - 200) < 0.01)
        #expect(fit.widths.allSatisfy { w in w >= 30 })
    }

    @Test func aShortColumnKeepsOnlyWhatItCanHold() {
        let fit = TableMetrics.columnLayout(
            headers: ["a", "b"], rows: [["c", "d"]],
            natural: [40, 400], minimums: [20, 20], available: 200)
        #expect(fit.wrap)
        #expect(abs(fit.widths[0] - 40) < 0.01)
        #expect(abs(fit.widths[1] - 160) < 0.01)
    }
}

@MainActor @Suite struct WrapCellTests {

    private static let font = FontRole.body(15).platformFont

    private func cell(_ text: String) -> NSAttributedString {
        NSAttributedString(string: text,
                           attributes: [.font: WrapCellTests.font])
    }

    @Test func aCellIsCutIntoLinesAndLosesNothing() {
        let source = cell("one two three four five six seven")
        let lines = DocumentText.wrapCell(source, width: 60)
        #expect(lines.count > 1)
        #expect(lines.map { line in line.string }.joined() == source.string)
    }

    @Test func aCellThatFitsStaysOneLine() {
        #expect(DocumentText.wrapCell(cell("short"), width: 400).count == 1)
    }

    @Test func aColumnTooNarrowForAGlyphStillTerminates() {
        let source = cell("indivisible")
        let lines = DocumentText.wrapCell(source, width: 1)
        #expect(lines.count > 1)
        #expect(lines.map { line in line.string }.joined() == source.string)
    }

    @Test func anEmptyCellHasNoLines() {
        #expect(DocumentText.wrapCell(cell(""), width: 100).isEmpty)
    }
}

#if os(iOS)

@MainActor @Suite struct TabStopTableTests {

    private static let headers = ["Term", "Meaning"]
    private static let rows = [
        ["principal",
         "The original sum of money borrowed or invested, before any "
         + "interest is added to it"],
    ]

    private func built(width: CGFloat) -> NSAttributedString {
        DocumentText.table(headers: TabStopTableTests.headers,
                           rows: TabStopTableTests.rows,
                           alignments: [.none, .none], id: "t",
                           style: .default, images: [:], width: width)
    }

    private func lines(_ ns: NSAttributedString) -> [String] {
        ns.string.split(separator: "\n").map(String.init)
    }

    private func stops(_ ns: NSAttributedString) -> [NSTextTab] {
        let para = ns.attribute(.paragraphStyle, at: 0,
                                effectiveRange: nil) as? NSParagraphStyle
        return para?.tabStops ?? []
    }

    @Test func aCellTooWideForItsColumnWrapsOntoMoreLines() {
        #expect(lines(built(width: 320)).count > 2)
    }

    @Test func everyLineCarriesOneTabPerColumnBoundary() {
        for line in lines(built(width: 320)) {
            #expect(line.filter { c in c == "\t" }.count == 1)
        }
    }

    @Test func theColumnsAreSolvedAgainstTheWidthGiven() {
        let located = stops(built(width: 320))
        #expect(located.count == 1)
        #expect(located[0].location > 0)
        #expect(located[0].location < 320)
    }

    @Test func aWiderSurfaceMovesTheColumnsOut() {
        let sentence = "The original sum of money borrowed or invested, "
            + "before any interest is added to it"
        func stopsAt(_ width: CGFloat) -> [NSTextTab] {
            stops(DocumentText.table(
                headers: ["One", "Two", "Three"],
                rows: [[sentence, sentence, sentence]],
                alignments: [.none, .none, .none], id: "t",
                style: .default, images: [:], width: width))
        }
        let narrow = stopsAt(320)
        let wide = stopsAt(760)
        #expect(narrow.count == 2 && wide.count == 2)
        #expect(wide[0].location > narrow[0].location)
        #expect(wide[1].location > narrow[1].location)
    }

    @Test func aShortColumnIsNotPaddedPastItsContent() {
        let located = stops(built(width: 320))
        let unforced = DocumentText.tableCells(
            headers: TabStopTableTests.headers, rows: TabStopTableTests.rows,
            style: .default, images: [:]).naturals
        #expect(located[0].location <= unforced[0] + 12)
    }

    @Test func aShortHeaderWordSurvivesALongNeighbour() {
        var style = MarkdownStyle.default
        style.bodySize = 17
        let ns = DocumentText.table(
            headers: ["River", "Length", "Country"],
            rows: [["Mississippi\u{2013}Missouri River System", "6,275 km**",
                    "United States"],
                   ["Yenisei", "5,539 km", "Mongolia/Russia"]],
            alignments: [.none, .none, .none], id: "t", style: style,
            images: [:], width: 290)
        #expect(lines(ns)[0].contains("Length"))
        #expect(DocumentText.tableMinimumWidth(
            headers: ["River", "Length", "Country"],
            rows: [["Yenisei", "5,539 km", "Mongolia/Russia"]],
            style: style) < 290)
    }

    @Test func columnsThatCannotFitKeepTheirWordsAndRunPastTheRoom() {
        let ns = DocumentText.table(
            headers: ["Organism", "Instrument", "City"],
            rows: [["Hippopotamus", "Electroencephalograph",
                    "Constantinople"]],
            alignments: [.none, .none, .none], id: "t", style: .default,
            images: [:], width: 200)
        #expect(lines(ns).count == 2)
        #expect((stops(ns).last?.location ?? 0) > 200)
    }

    @Test func theRowNeverTruncates() {
        let para = built(width: 320).attribute(
            .paragraphStyle, at: 0, effectiveRange: nil) as? NSParagraphStyle
        #expect(para?.lineBreakMode == .byWordWrapping)
    }

    @Test func alignmentMovesTheStopNotJustTheText() {
        let left = DocumentText.table(
            headers: TabStopTableTests.headers, rows: TabStopTableTests.rows,
            alignments: [.none, .left], id: "t", style: .default,
            images: [:], width: 320)
        let right = DocumentText.table(
            headers: TabStopTableTests.headers, rows: TabStopTableTests.rows,
            alignments: [.none, .right], id: "t", style: .default,
            images: [:], width: 320)
        #expect(stops(left)[0].alignment == .left)
        #expect(stops(right)[0].alignment == .right)
        #expect(stops(right)[0].location > stops(left)[0].location)
    }
}

#endif

@Suite struct NarrowTableTests {

    private static let mins: [CGFloat] = [60, 40, 55, 45, 50, 52]
    private static let headers = ["Block", "Trees", "Tonnes", "Yield",
                                  "per tree", "To press"]
    private static let rows = [["Riverbank", "295", "19.4", "65.8kg",
                                "", "9%"]]

    @Test func aColumnIsNeverNarrowerThanItsLongestWord() {
        let mins = NarrowTableTests.mins
        let fit = TableMetrics.scrollingLayout(
            headers: NarrowTableTests.headers, rows: NarrowTableTests.rows,
            natural: mins.map { w in w * 3 }, minimums: mins,
            available: mins.reduce(0, +) / 2)
        #expect(fit.scrolls)
        for (i, w) in fit.widths.enumerated() {
            #expect(w >= mins[i] - 0.01)
        }
    }

    @Test func aTableThatFitsDoesNotScroll() {
        let mins = NarrowTableTests.mins
        let fit = TableMetrics.scrollingLayout(
            headers: NarrowTableTests.headers, rows: NarrowTableTests.rows,
            natural: mins, minimums: mins,
            available: mins.reduce(0, +) * 2)
        #expect(!fit.scrolls)
        #expect(!fit.wrap)
    }
}

@Suite struct FairShortfallTests {

    @Test func aColumnThatCanBeSatisfiedKeepsItsWord() {
        let fit = TableMetrics.columnLayout(
            headers: ["River", "Length", "Country"],
            rows: [["Mississippi-Missouri River System", "6,275 km",
                    "United States"]],
            natural: [400, 90, 150], minimums: [220, 60, 160],
            available: 300)
        #expect(abs(fit.widths[1] - 60) < 0.01)
        #expect(abs(fit.widths.reduce(0, +) - 300) < 0.01)
        #expect(fit.widths[0] > fit.widths[2])
    }
}

@MainActor @Suite struct UnbreakableRunTests {

    private func runs(_ text: String) -> [String] {
        let ns = text as NSString
        return DocumentText.unbreakableRuns(ns).map { r in
            ns.substring(with: r)
        }
    }

    @Test func aDashOrASlashEndsARun() {
        #expect(runs("Mississippi\u{2013}Missouri River")
                == ["Mississippi\u{2013}", "Missouri", "River"])
        #expect(runs("Mongolia/Russia") == ["Mongolia/", "Russia"])
    }

    @Test func aDigitAfterTheMarkKeepsTheRunWhole() {
        #expect(runs("24/7 on-call") == ["24/7", "on-", "call"])
    }
}

@MainActor @Suite struct WideSurfaceTests {

    private func tails(wide: Bool) -> (prose: CGFloat, table: CGFloat) {
        let doc = Markdown.parse("| A | B |\n|---|---|\n| one | two |\n\n"
                                 + "A paragraph after the table.")
        let ns = DocumentText.attributed(from: doc, style: .default,
                                         width: 300, wide: wide)
        let text = ns.string as NSString
        func tail(_ needle: String) -> CGFloat {
            let at = text.range(of: needle).location
            let para = ns.attribute(.paragraphStyle, at: at,
                                    effectiveRange: nil) as? NSParagraphStyle
            return para?.tailIndent ?? 0
        }
        return (tail("paragraph"), tail("one"))
    }

    @Test func proseWrapsAtTheVisibleWidthAndTheTableDoesNot() {
        let got = tails(wide: true)
        #expect(got.prose == 300)
        #expect(got.table == 0)
    }

    @Test func aSurfaceThatFitsIsLeftAlone() {
        #expect(tails(wide: false).prose == 0)
    }
}

@MainActor @Suite struct SharedTableCellsTests {

    static let source = """
    Intro paragraph.

    | Term | Meaning |
    |---|---|
    | principal | The original sum of money borrowed or invested |
    | `code` | x^2 and H<sub>2</sub>O |
    |  | ![alt](https://example.com/a.png) |

    > | A | B |
    > |---|---:|
    > | one | two three four |

    - item
      | X | Y | Z |
      |:-:|---|--:|
      | Mississippi\u{2013}Missouri | 24/7 | Mongolia/Russia |

    | Organism | Instrument | City |
    |---|---|---|
    | Hippopotamus | Electroencephalograph | Constantinople |
    """

    static func fingerprint(_ ns: NSAttributedString) -> String {
        var parts: [String] = [ns.string]
        let full = NSRange(location: 0, length: ns.length)
        ns.enumerateAttributes(in: full, options: []) { attrs, range, _ in
            var line = "\(range.location)+\(range.length)"
            if let f = attrs[.font] as? PlatformFont {
                line += " f=\(f.fontName)@\(f.pointSize)"
            }
            if let p = attrs[.paragraphStyle] as? NSParagraphStyle {
                line += " a=\(p.alignment.rawValue) h=\(p.headIndent)"
                    + " t=\(p.tailIndent) lb=\(p.lineBreakMode.rawValue)"
                for tab in p.tabStops {
                    line += " tab=\(tab.location):\(tab.alignment.rawValue)"
                }
                #if os(macOS)
                for case let b as NSTextTableBlock in p.textBlocks {
                    line += " blk=\(b.startingRow),\(b.startingColumn)"
                        + " w=\(b.value(for: .width))"
                }
                #endif
            }
            for key in [atomicIdKey, atomicKindKey, atomicCopyKey] {
                if let v = attrs[key] as? String { line += " \(v)" }
            }
            parts.append(line)
        }
        return parts.joined(separator: "\n")
    }

    private func tables(_ doc: Markdown.Document)
        -> [(headers: [String], rows: [[String]])] {
        var out: [(headers: [String], rows: [[String]])] = []
        for item in doc.items {
            if case .table(let h, let r, _) = item.block { out.append((h, r)) }
        }
        return out
    }

    private func reference(_ headers: [String], _ rows: [[String]],
                           style: MarkdownStyle)
        -> (minimums: [CGFloat], naturals: [CGFloat]) {
        let body = DocumentText.bodyFont(style)
        let bold = boldFont(of: body)
        let cols = max(headers.count, rows.map { r in r.count }.max() ?? 0)
        var minimums = [CGFloat](repeating: 0, count: cols)
        var naturals = [CGFloat](repeating: 0, count: cols)
        for c in 0..<cols {
            var cells: [(String, PlatformFont)] = []
            if c < headers.count { cells.append((headers[c], bold)) }
            for row in rows where c < row.count { cells.append((row[c], body)) }
            for (cell, font) in cells {
                let block = Markdown.parseCell(cell).items.first?.block
                var drawn = TeX.scriptsToUnicode(cell)
                var built = DocumentText.tableCell(cell, base: font,
                                                   style: style,
                                                   images: [:]).text
                switch block {
                    case .paragraph(let a)?: drawn = String(a.characters)
                    case .image?:
                        drawn = ""
                        built = NSAttributedString()
                    default: break
                }
                let natural = (drawn as NSString)
                    .size(withAttributes: [.font: font]).width
                if natural > naturals[c] { naturals[c] = natural }
                let text = built.string as NSString
                for run in DocumentText.unbreakableRuns(text) {
                    let line = CTLineCreateWithAttributedString(
                        built.attributedSubstring(from: run))
                    let w = CGFloat(CTLineGetTypographicBounds(line, nil, nil,
                                                               nil))
                    if w > minimums[c] { minimums[c] = w }
                }
            }
            minimums[c] = ceil(minimums[c])
            let measured = DocumentText.tableUsesNaturals
            naturals[c] = measured ? ceil(naturals[c]) : 0
        }
        return (minimums, naturals)
    }

    @Test func theColumnExtentsMatchThePerCellReference() {
        let doc = Markdown.parse(SharedTableCellsTests.source)
        for t in tables(doc) {
            let cells = DocumentText.tableCells(headers: t.headers,
                                                rows: t.rows,
                                                style: .default, images: [:])
            let want = reference(t.headers, t.rows, style: .default)
            #expect(cells.minimums == want.minimums)
            #expect(cells.naturals == want.naturals)
            #expect(cells.cols == want.minimums.count)
        }
        #expect(tables(doc).count >= 2)
    }

    @Test func theSharedCellsRenderWhatAFreshBuildRenders() {
        let doc = Markdown.parse(SharedTableCellsTests.source)
        for width: CGFloat in [0, 200, 320, 760] {
            let cache = DocumentText.RenderCache()
            let need = DocumentText.minimumWidth(of: doc, style: .default,
                                                 formulas: false, cache: cache)
            let shared = DocumentText.attributed(from: doc, style: .default,
                                                 width: width, cache: cache)
            let fresh = DocumentText.attributed(from: doc, style: .default,
                                                width: width)
            #expect(need == DocumentText.minimumWidth(of: doc, style: .default,
                                                      formulas: false))
            #expect(SharedTableCellsTests.fingerprint(shared)
                    == SharedTableCellsTests.fingerprint(fresh))
            #expect(cache.tables.count == tables(doc).count)
        }
    }

    @Test func aChangedCellRebuildsOnlyItsTable() {
        let doc = Markdown.parse(SharedTableCellsTests.source)
        let cache = DocumentText.RenderCache()
        _ = DocumentText.minimumWidth(of: doc, style: .default,
                                      formulas: false, cache: cache)
        _ = DocumentText.attributed(from: doc, style: .default, width: 320,
                                    cache: cache)
        let first = cache.tables[1]?.cells.body.first?.first?.text
        let edited = Markdown.parse(SharedTableCellsTests.source
            .replacingOccurrences(of: "Constantinople", with: "Byzantium"))
        _ = DocumentText.attributed(from: edited, style: .default,
                                    width: 320, cache: cache)
        #expect(cache.tables[1]?.cells.body.first?.first?.text === first)
        #expect(cache.tables[4]?.cells.body.first?.last?.text.string
                == "Byzantium")
    }
}

@MainActor @Suite struct TableCostTests {

    static let source: String = {
        var lines = ["| Region | Population | Area km2 | Density | Capital |"
                     + " Note |", "|---|---:|---:|---:|---|---|"]
        for i in 0 ..< 30 {
            lines.append("| Region \(i) with a long name | \(1000 + i * 37)"
                         + ",512 | \(200 + i * 13).5 | 12.\(i) | Capital-\(i)"
                         + "/City | **bold** and *italic* words here |")
        }
        return lines.joined(separator: "\n")
    }()

    private func millis(_ n: Int, _ body: () -> Void) -> Double {
        let t0 = Date()
        for _ in 0 ..< n { body() }
        return Date().timeIntervalSince(t0) * 1000 / Double(n)
    }

    @Test func aThirtyBySixTableCostsOneBuildPerCellPerChange() {
        let n = 20
        let cache = DocumentText.RenderCache()
        let docs = (0 ..< n).map { i in
            Markdown.parse(TableCostTests.source
                           + "\n| tail \(i) | 1 | 2 | 3 | 4 | 5 |")
        }
        var next = 0
        let change = millis(n) {
            _ = DocumentText.minimumWidth(of: docs[next], style: .default,
                                          formulas: false, cache: cache)
            _ = DocumentText.attributed(from: docs[next], style: .default,
                                        width: 400, cache: cache)
            next += 1
        }
        var headers: [String] = []
        var rows: [[String]] = []
        if case .table(let h, let r, _) = docs[0].items[0].block {
            headers = h
            rows = r
        }
        let cold = millis(n) {
            _ = TableMeasure.measure(headers: headers, rows: rows,
                                     style: .default)
        }
        let memo = TableMeasure()
        let hit = millis(n) {
            _ = memo.measured(headers: headers, rows: rows, style: .default)
        }
        let pdf = millis(5) { _ = MarkdownPDF.data(docs[0], title: "t") }
        print("table cost ms: change \(change) measure \(cold) hit \(hit)"
              + " pdf \(pdf)")
        #expect(cache.tables.count == 1)
        #expect(hit < cold)
    }
}
