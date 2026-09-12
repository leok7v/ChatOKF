import SwiftUI

// The block model. One AST feeds every renderer (SwiftUI, single-surface,
// PDF, HTML, plain) and both the batch parser and the streaming parser.
public enum Markdown {

    public enum Alignment: Equatable, Sendable {
        case none, left, center, right
    }

    public indirect enum Block: Equatable, Sendable {
        case heading(level: Int, text: AttributedString)
        case paragraph(AttributedString)
        case code(language: String?, text: String)
        case quote([Block])
        case list(items: [ListItem], tight: Bool)
        case table(headers: [String], rows: [[String]],
                   alignments: [Alignment])
        // A $$...$$ display, carried as its TeX source.
        case math(String)
        case rule
        case image(alt: String, url: URL,
                   width: CGFloat?, height: CGFloat?)
    }

    public struct ListItem: Equatable, Sendable {
        public let marker: String       // "1." or a bullet label
        public let checked: Bool?        // nil when not a task item
        public let blocks: [Block]

        public init(marker: String, checked: Bool?, blocks: [Block]) {
            self.marker = marker
            self.checked = checked
            self.blocks = blocks
        }
    }

    public struct Document: Equatable, Sendable {

        public struct Item: Identifiable, Equatable, Sendable {
            public let id: Int           // stable across snapshots
            public let block: Block

            public init(id: Int, block: Block) {
                self.id = id
                self.block = block
            }
        }

        public let items: [Item]

        public init(items: [Item]) { self.items = items }

        public static let empty = Document(items: [])
    }
}
