import Foundation
import Testing
@testable import MD

// A delimiter row a language model actually wrote, and the paragraphs that must
// not be mistaken for one.
private func firstBlock(_ source: String) -> Markdown.Block? {
    Markdown.parse(source).items.first?.block
}

@Suite struct TableToleranceTests {

    // The reported case: three good cells and one ":**" where ":---" belonged.
    @Test func aMalformedDelimiterCellStillMakesATable() {
        let source = """
        | Era | Discovery | Key Finding | Significance |
        | :--- | :--- | :--- | :** |
        | 1930s | Missing Mass | Rotation curves | First evidence |
        """
        let block = firstBlock(source)
        var headerCount = 0
        var rowCount = 0
        if case let .table(headers, rows, _) = block {
            headerCount = headers.count
            rowCount = rows.count
        }
        #expect(headerCount == 4)
        #expect(rowCount == 1)
    }

    @Test func aWellFormedTableIsUnaffected() {
        let source = """
        | A | B |
        | :--- | ---: |
        | 1 | 2 |
        """
        var aligns: [Markdown.Alignment] = []
        if case let .table(_, _, a) = firstBlock(source) { aligns = a }
        #expect(aligns == [.left, .right])
    }

    // The guard that keeps prose out: a cell carrying letters or digits is
    // content, however dash-like the rest of the line looks.
    @Test func proseWithPipesAndDashesIsNotATable() {
        let source = """
        Item | value
        Cost | 5 - 3
        """
        var isTable = false
        if case .table = firstBlock(source) { isTable = true }
        #expect(!isTable)
    }

    @Test func aRowOfWordsIsNotADelimiter() {
        #expect(!Markdown.isTableSeparator("| Name | --- |"))
        #expect(Markdown.isTableSeparator("| :--- | :** |"))
        #expect(Markdown.isTableSeparator("|---|---|"))
        // No dash at all is never a delimiter, however punctuated.
        #expect(!Markdown.isTableSeparator("| :: | :: |"))
    }
}
