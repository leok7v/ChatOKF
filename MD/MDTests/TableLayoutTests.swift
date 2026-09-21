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
                           alignments: [.none, .none], style: .default,
                           images: [:], width: width)
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
                alignments: [.none, .none, .none], style: .default,
                images: [:], width: width))
        }
        let narrow = stopsAt(320)
        let wide = stopsAt(760)
        #expect(narrow.count == 2 && wide.count == 2)
        #expect(wide[0].location > narrow[0].location)
        #expect(wide[1].location > narrow[1].location)
    }

    @Test func aShortColumnIsNotPaddedPastItsContent() {
        let located = stops(built(width: 320))
        let unforced = DocumentText.columnNaturals(
            headers: TabStopTableTests.headers, rows: TabStopTableTests.rows,
            cols: 2, style: .default)
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
            alignments: [.none, .none, .none], style: style, images: [:],
            width: 290)
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
            alignments: [.none, .none, .none], style: .default, images: [:],
            width: 200)
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
            alignments: [.none, .left], style: .default, images: [:],
            width: 320)
        let right = DocumentText.table(
            headers: TabStopTableTests.headers, rows: TabStopTableTests.rows,
            alignments: [.none, .right], style: .default, images: [:],
            width: 320)
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
