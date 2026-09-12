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
