import Compression
import Foundation

// see okf/decisions/ooxml-never-goes-through-html.md

enum Format: String, CaseIterable {
    case docx
    case xlsx
    case pptx
    case epub
    case html
}

enum DocsError: Error, CustomStringConvertible {
    case unreadable(URL)
    case unknownFormat(String)
    case notAnArchive(String)
    case missingPart(String)
    case malformed(String)
    case unsupported(String)

    var description: String {
        var result = ""
        switch self {
        case .unreadable(let url):
            result = "cannot read: \(url.path)"
        case .unknownFormat(let name):
            let known = Format.allCases.map { format in
                format.rawValue
            }.joined(separator: ", ")
            result = "no reader for '\(name)'; known: \(known)"
        case .notAnArchive(let name):
            result = "\(name) is not a ZIP container"
        case .missingPart(let part):
            result = "container has no part '\(part)'"
        case .malformed(let what):
            result = "malformed \(what)"
        case .unsupported(let what):
            result = "unsupported: \(what)"
        }
        return result
    }
}

// see okf/formats/zip-is-read-from-the-tail.md

struct Archive {
    struct Entry {
        let method: Int
        let stored: Int
        let expanded: Int
        let offset: Int
    }

    private let bytes: [UInt8]
    private let entries: [String: Entry]
    let names: [String]

    init(_ data: Data, _ label: String) throws {
        let raw = [UInt8](data)
        let end = Archive.directory(raw)
        let listed = end < 0 ? nil : Archive.catalogue(raw, end)
        if let listed {
            bytes = raw
            entries = listed.0
            names = listed.1
        } else {
            throw DocsError.notAnArchive(label)
        }
    }

    func data(_ name: String) -> Data? {
        var result: Data? = nil
        if let entry = entries[name] {
            let head = entry.offset
            let named = Archive.number(bytes, head + 26, 2)
            let extra = Archive.number(bytes, head + 28, 2)
            let from = head + 30 + named + extra
            let upto = from + entry.stored
            let sane = Archive.number(bytes, head, 4) == 0x0403_4b50
                && upto <= bytes.count && from <= upto
            if sane && entry.method == 0 {
                result = Data(bytes[from..<upto])
            } else if sane && entry.method == 8 {
                result = Archive.inflated(Array(bytes[from..<upto]),
                                          entry.expanded)
            }
        }
        return result
    }

    func text(_ name: String) -> String? {
        var result: String? = nil
        if let raw = data(name) {
            result = String(data: raw, encoding: .utf8)
                ?? String(data: raw, encoding: .isoLatin1)
        }
        return result
    }

    func tree(_ name: String) -> Markup? {
        var result: Markup? = nil
        if let raw = data(name) { result = MarkupParser.tree(raw) }
        return result
    }

    func has(_ name: String) -> Bool {
        entries[name] != nil
    }

    private static func number(_ bytes: [UInt8], _ at: Int,
                               _ width: Int) -> Int {
        var result = 0
        var step = 0
        while step < width && at >= 0 && at + step < bytes.count {
            result |= Int(bytes[at + step]) << (8 * step)
            step += 1
        }
        return result
    }

    // The EOCD trails a comment of up to 64K; hence the back scan.

    private static func directory(_ bytes: [UInt8]) -> Int {
        var at = bytes.count - 22
        let least = max(bytes.count - 22 - 65_535, 0)
        var found = -1
        while at >= least && found < 0 {
            if Archive.number(bytes, at, 4) == 0x0605_4b50 {
                found = at
            }
            at -= 1
        }
        return found
    }

    // An all-ones count or offset means the truth is in ZIP64.

    private static func anchor(_ bytes: [UInt8],
                               _ end: Int) -> (Int, Int) {
        var count = Archive.number(bytes, end + 10, 2)
        var start = Archive.number(bytes, end + 16, 4)
        let locator = end - 20
        let wide = count == 0xFFFF || start == 0xFFFF_FFFF
        if wide && Archive.number(bytes, locator, 4) == 0x0706_4b50 {
            let record = Archive.number(bytes, locator + 8, 8)
            if Archive.number(bytes, record, 4) == 0x0606_4b50 {
                count = Archive.number(bytes, record + 32, 8)
                start = Archive.number(bytes, record + 48, 8)
            }
        }
        return (count, start)
    }

    private static func catalogue(_ bytes: [UInt8], _ end: Int)
            -> ([String: Entry], [String])? {
        let (count, start) = Archive.anchor(bytes, end)
        var entries: [String: Entry] = [:]
        var names: [String] = []
        var at = start
        var left = count
        while left > 0 && at + 46 <= bytes.count
                && Archive.number(bytes, at, 4) == 0x0201_4b50 {
            let named = Archive.number(bytes, at + 28, 2)
            let extra = Archive.number(bytes, at + 30, 2)
            let note = Archive.number(bytes, at + 32, 2)
            let name = Archive.name(bytes, at + 46, named)
            let entry = Archive.entry(bytes, at, at + 46 + named,
                                      extra)
            if entries[name] == nil { names.append(name) }
            entries[name] = entry
            at += 46 + named + extra + note
            left -= 1
        }
        return entries.isEmpty ? nil : (entries, names)
    }

    private static func entry(_ bytes: [UInt8], _ head: Int,
                              _ extraAt: Int,
                              _ extraLength: Int) -> Entry {
        var stored = Archive.number(bytes, head + 20, 4)
        var expanded = Archive.number(bytes, head + 24, 4)
        var offset = Archive.number(bytes, head + 42, 4)
        let saturated = expanded == 0xFFFF_FFFF
            || stored == 0xFFFF_FFFF || offset == 0xFFFF_FFFF
        if saturated {
            var at = extraAt
            let upto = extraAt + extraLength
            while at + 4 <= upto {
                let tag = Archive.number(bytes, at, 2)
                let size = Archive.number(bytes, at + 2, 2)
                var field = at + 4
                if tag == 1 && expanded == 0xFFFF_FFFF {
                    expanded = Archive.number(bytes, field, 8)
                    field += 8
                }
                if tag == 1 && stored == 0xFFFF_FFFF {
                    stored = Archive.number(bytes, field, 8)
                    field += 8
                }
                if tag == 1 && offset == 0xFFFF_FFFF {
                    offset = Archive.number(bytes, field, 8)
                }
                at += 4 + size
            }
        }
        return Entry(method: Archive.number(bytes, head + 10, 2),
                     stored: stored, expanded: expanded,
                     offset: offset)
    }

    private static func name(_ bytes: [UInt8], _ at: Int,
                             _ length: Int) -> String {
        var result = ""
        if at + length <= bytes.count {
            let raw = Data(bytes[at..<(at + length)])
            result = String(data: raw, encoding: .utf8)
                ?? String(data: raw, encoding: .isoLatin1) ?? ""
        }
        return result
    }

    // COMPRESSION_ZLIB is raw DEFLATE: what a ZIP member holds.

    private static func inflated(_ source: [UInt8],
                                 _ expanded: Int) -> Data? {
        var result: Data? = nil
        if expanded == 0 {
            result = Data()
        } else if !source.isEmpty {
            var out = [UInt8](repeating: 0, count: expanded)
            let written = out.withUnsafeMutableBufferPointer { sink in
                source.withUnsafeBufferPointer { input in
                    compression_decode_buffer(
                        sink.baseAddress!, expanded,
                        input.baseAddress!, source.count,
                        nil, COMPRESSION_ZLIB)
                }
            }
            if written == expanded { result = Data(out) }
        }
        return result
    }
}

final class Markup {
    let name: String
    let attributes: [String: String]
    fileprivate(set) var children: [Markup] = []
    fileprivate(set) var text: String

    fileprivate init(_ name: String,
                     _ attributes: [String: String]) {
        self.name = name
        self.attributes = attributes
        self.text = ""
    }

    fileprivate init(text: String) {
        self.name = ""
        self.attributes = [:]
        self.text = text
    }

    fileprivate func append(_ node: Markup) {
        let merging = node.name.isEmpty
            && (children.last?.name.isEmpty ?? false)
        if merging {
            children[children.count - 1].text += node.text
        } else {
            children.append(node)
        }
    }

    func first(_ name: String) -> Markup? {
        children.first(where: { node in node.name == name })
    }

    func all(_ name: String) -> [Markup] {
        children.filter { node in node.name == name }
    }

    func find(_ name: String) -> [Markup] {
        var result: [Markup] = []
        for node in children {
            if node.name == name { result.append(node) }
            result += node.find(name)
        }
        return result
    }

    func attribute(_ name: String) -> String? {
        attributes[name]
    }

    var flattened: String {
        var result = text
        for node in children { result += node.flattened }
        return result
    }
}

final class MarkupParser: NSObject, XMLParserDelegate {
    private var stack: [Markup] = []
    private var root: Markup? = nil

    static func tree(_ data: Data) -> Markup? {
        let reader = MarkupParser()
        let parser = XMLParser(data: data)
        parser.delegate = reader
        parser.shouldProcessNamespaces = false
        parser.parse()
        return reader.root
    }

    func parser(_ parser: XMLParser, didStartElement element: String,
                namespaceURI: String?, qualifiedName: String?,
                attributes: [String: String]) {
        let node = Markup(qualifiedName ?? element, attributes)
        if let parent = stack.last {
            parent.append(node)
        } else {
            root = node
        }
        stack.append(node)
    }

    func parser(_ parser: XMLParser, didEndElement element: String,
                namespaceURI: String?, qualifiedName: String?) {
        if !stack.isEmpty { stack.removeLast() }
    }

    func parser(_ parser: XMLParser, foundCharacters text: String) {
        stack.last?.append(Markup(text: text))
    }

    func parser(_ parser: XMLParser, foundCDATA block: Data) {
        let text = String(data: block, encoding: .utf8) ?? ""
        stack.last?.append(Markup(text: text))
    }
}

// columns and rows outlive the emitter, which cannot draw a span.

struct Cell {
    let text: String
    let column: Int
    let columns: Int
    let rows: Int
}

enum DocsElement {
    case heading(Int, String)
    case paragraph(String)
    case item(Int, String, Int?)
    case quote(String)
    case code(String, String)
    case table([[Cell]])
    case rule
}

struct Span {
    var text: String
    var bold = false
    var italic = false
    var code = false
    var link: String? = nil
}

struct Emitter {
    static func markdown(_ elements: [DocsElement]) -> String {
        var blocks: [String] = []
        var listing = false
        for element in elements {
            let rendered = Emitter.render(element)
            let listed = Emitter.isItem(element)
            if listed && listing && !blocks.isEmpty {
                blocks[blocks.count - 1] += "\n" + rendered
            } else if !rendered.isEmpty {
                blocks.append(rendered)
            }
            listing = listed && !rendered.isEmpty
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }

    private static func isItem(_ element: DocsElement) -> Bool {
        var result = false
        if case .item = element { result = true }
        return result
    }

    private static func render(_ element: DocsElement) -> String {
        var result = ""
        switch element {
        case .heading(let level, let text):
            let hashes = String(repeating: "#", count: max(level, 1))
            result = "\(hashes) \(text)"
        case .paragraph(let text):
            result = text
        case .item(let depth, let text, let ordinal):
            let pad = String(repeating: "    ", count: max(depth, 0))
            let mark = ordinal.map { number in
                "\(number). "
            } ?? "* "
            result = pad + mark + text
        case .quote(let text):
            result = text.split(separator: "\n",
                                omittingEmptySubsequences: false)
                .map { line in "> " + line }
                .joined(separator: "\n")
        case .code(let language, let body):
            result = "```\(language)\n\(body)\n```"
        case .table(let rows):
            result = Emitter.table(rows)
        case .rule:
            result = "---"
        }
        return result
    }

    private static func table(_ rows: [[Cell]]) -> String {
        var width = 0
        for row in rows {
            for cell in row {
                width = max(width, cell.column + cell.columns)
            }
        }
        var lines: [String] = []
        for (index, row) in rows.enumerated() {
            var cells = [String](repeating: "", count: max(width, 1))
            for cell in row where cell.column < width {
                cells[cell.column] = Emitter.escaped(cell.text)
            }
            lines.append("| " + cells.joined(separator: " | ") + " |")
            if index == 0 {
                let rule = [String](repeating: "---",
                                    count: max(width, 1))
                lines.append("| " + rule.joined(separator: " | ")
                             + " |")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func escaped(_ text: String) -> String {
        text.replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: "<br>")
    }

    // Edge whitespace moves outside the marks: `** x **` is literal.

    static func inline(_ spans: [Span]) -> String {
        var result = ""
        for span in Emitter.folded(spans) {
            let text = span.text
            let head = text.prefix(while: Emitter.blank)
            let tail = text.reversed().prefix(while: Emitter.blank)
            let body = String(text.dropFirst(head.count)
                              .dropLast(tail.count))
            if body.isEmpty {
                result += text
            } else {
                result += String(head) + Emitter.marked(body, span)
                    + String(tail.reversed())
            }
        }
        return result
    }

    private static func blank(_ character: Character) -> Bool {
        character == " " || character == "\t"
    }

    private static func marked(_ body: String,
                               _ span: Span) -> String {
        var result = body
        if span.code {
            result = "`" + result.replacingOccurrences(
                of: "`", with: "") + "`"
        }
        if span.bold { result = "**" + result + "**" }
        if span.italic { result = "*" + result + "*" }
        if let link = span.link, !link.isEmpty {
            result = "[" + result + "](" + link + ")"
        }
        return result
    }

    private static func folded(_ spans: [Span]) -> [Span] {
        var result: [Span] = []
        for span in spans where !span.text.isEmpty {
            let last = result.last
            let same = last != nil && last!.bold == span.bold
                && last!.italic == span.italic
                && last!.code == span.code && last!.link == span.link
            if same {
                result[result.count - 1].text += span.text
            } else {
                result.append(span)
            }
        }
        return result
    }
}

struct DocsConverter {
    var format: Format? = nil

    func markdown(of url: URL) throws -> String {
        let chosen = format ?? DocsConverter.detected(url)
        let raw = try DocsConverter.contents(url)
        var elements: [DocsElement] = []
        if let chosen {
            elements = try DocsConverter.read(chosen, raw, url)
        } else {
            throw DocsError.unknownFormat(
                url.pathExtension.isEmpty
                    ? url.lastPathComponent : url.pathExtension)
        }
        return Emitter.markdown(elements)
    }

    private static func read(_ format: Format, _ raw: Data,
                             _ url: URL) throws -> [DocsElement] {
        let label = url.lastPathComponent
        var result: [DocsElement] = []
        switch format {
        case .docx:
            result = try WordReader(Archive(raw, label)).elements()
        case .xlsx:
            result = try SheetReader(Archive(raw, label)).elements()
        case .pptx:
            result = try SlideReader(Archive(raw, label)).elements()
        case .epub:
            result = try BookReader(Archive(raw, label)).elements()
        case .html:
            result = PageReader(DocsConverter.string(raw)).elements()
        }
        return result
    }

    private static func detected(_ url: URL) -> Format? {
        var result: Format? = nil
        let extended = url.pathExtension.lowercased()
        if extended == "htm" || extended == "xhtml" {
            result = .html
        } else {
            result = Format(rawValue: extended)
        }
        return result
    }

    private static func contents(_ url: URL) throws -> Data {
        let raw = try? Data(contentsOf: url)
        var result = Data()
        if let raw {
            result = raw
        } else {
            throw DocsError.unreadable(url)
        }
        return result
    }

    static func string(_ raw: Data) -> String {
        String(data: raw, encoding: .utf8)
            ?? String(data: raw, encoding: .isoLatin1) ?? ""
    }
}

struct Relations {
    private let targets: [String: (String, Bool)]
    private let base: String

    init(_ archive: Archive, _ part: String) {
        var found: [String: (String, Bool)] = [:]
        if let tree = archive.tree(part) {
            for relation in tree.all("Relationship") {
                let id = relation.attribute("Id") ?? ""
                let target = relation.attribute("Target") ?? ""
                let external =
                    relation.attribute("TargetMode") == "External"
                if !id.isEmpty { found[id] = (target, external) }
            }
        }
        targets = found
        base = Relations.folder(part)
    }

    // The rels part sits one folder deeper than the part it names.

    private static func folder(_ part: String) -> String {
        var pieces = part.split(separator: "/").map(String.init)
        if pieces.count >= 2 && pieces[pieces.count - 2] == "_rels" {
            pieces.removeLast(2)
        } else {
            pieces.removeLast()
        }
        return pieces.isEmpty ? "" : pieces.joined(separator: "/") + "/"
    }

    func target(_ id: String) -> String? {
        targets[id].map { pair in pair.0 }
    }

    func external(_ id: String) -> Bool {
        targets[id]?.1 ?? false
    }

    func part(_ id: String) -> String? {
        var result: String? = nil
        if let target = targets[id], !target.1 {
            result = Relations.resolved(base, target.0)
        }
        return result
    }

    func parts() -> [String] {
        targets.keys.sorted().compactMap { id in part(id) }
    }

    static func resolved(_ base: String, _ target: String) -> String {
        var pieces: [String] = []
        let combined = target.hasPrefix("/")
            ? String(target.dropFirst()) : base + target
        for piece in combined.split(separator: "/") {
            if piece == ".." {
                if !pieces.isEmpty { pieces.removeLast() }
            } else if piece != "." {
                pieces.append(String(piece))
            }
        }
        return pieces.joined(separator: "/")
    }
}

struct WordReader {
    private let body: Markup
    private let styles: [String: String]
    private let relations: Relations
    private let counted: [String: Bool]
    private let comments: [String: [String]]

    init(_ archive: Archive) throws {
        let tree = archive.tree("word/document.xml")
        if let tree {
            body = tree.first("w:body") ?? tree
            styles = WordReader.styles(archive)
            relations = Relations(
                archive, "word/_rels/document.xml.rels")
            counted = WordReader.numbering(archive)
            comments = WordReader.comments(archive)
        } else {
            throw DocsError.missingPart("word/document.xml")
        }
    }

    func elements() -> [DocsElement] {
        blocks(body)
    }

    private func blocks(_ parent: Markup) -> [DocsElement] {
        var result: [DocsElement] = []
        for node in parent.children {
            if node.name == "w:p" {
                result += paragraph(node)
            } else if node.name == "w:tbl" {
                result.append(table(node))
            } else if node.name == "w:sdt" {
                if let content = node.first("w:sdtContent") {
                    result += blocks(content)
                }
            }
        }
        return result
    }

    private func paragraph(_ node: Markup) -> [DocsElement] {
        let properties = node.first("w:pPr")
        let identifier = properties?.first("w:pStyle")?
            .attribute("w:val") ?? ""
        let style = styles[identifier] ?? identifier.lowercased()
        let text = Emitter.inline(spans(node, false, false))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var result: [DocsElement] = []
        if text.isEmpty {
            result = []
        } else if let level = WordReader.level(style) {
            result = [.heading(level, text)]
        } else if let listing = properties?.first("w:numPr") {
            let depth = Int(listing.first("w:ilvl")?
                .attribute("w:val") ?? "0") ?? 0
            let number = listing.first("w:numId")?
                .attribute("w:val") ?? ""
            let ordered = counted["\(number):\(depth)"] ?? false
            result = [.item(depth, text, ordered ? 1 : nil)]
        } else if style.contains("quote") {
            result = [.quote(text)]
        } else {
            result = [.paragraph(text)]
        }
        for reference in node.find("w:commentReference") {
            let id = reference.attribute("w:id") ?? ""
            for remark in comments[id] ?? [] {
                result.append(.quote(remark))
            }
        }
        return result
    }

    private static func level(_ style: String) -> Int? {
        var result: Int? = nil
        if style == "title" {
            result = 1
        } else if style.hasPrefix("heading") {
            let tail = style.dropFirst(7)
                .trimmingCharacters(in: .whitespaces)
            result = Int(tail).map { found in min(max(found, 1), 6) }
        }
        return result
    }

    private func spans(_ node: Markup, _ bold: Bool,
                       _ italic: Bool) -> [Span] {
        var result: [Span] = []
        for child in node.children {
            result += span(child, bold, italic)
        }
        return result
    }

    private func span(_ node: Markup, _ bold: Bool,
                      _ italic: Bool) -> [Span] {
        var result: [Span] = []
        switch node.name {
        case "w:r":
            result = run(node, bold, italic)
        case "w:hyperlink":
            result = linked(node, bold, italic)
        case "w:ins", "w:smartTag", "w:fldSimple", "w:sdtContent":
            result = spans(node, bold, italic)
        case "w:sdt":
            result = node.first("w:sdtContent")
                .map { content in spans(content, bold, italic) } ?? []
        case "m:oMathPara":
            result = [Span(text: "$$" + WordReader.latex(node) + "$$")]
        case "m:oMath":
            result = [Span(text: "$" + WordReader.latex(node) + "$")]
        default:
            result = []
        }
        return result
    }

    private func run(_ node: Markup, _ bold: Bool,
                     _ italic: Bool) -> [Span] {
        let properties = node.first("w:rPr")
        let strong = bold || WordReader.on(properties, "w:b")
        let slanted = italic || WordReader.on(properties, "w:i")
        var result: [Span] = []
        for child in node.children {
            var text: String? = nil
            switch child.name {
            case "w:t":
                text = child.flattened
            case "w:tab":
                text = "\t"
            case "w:br", "w:cr":
                text = "\n"
            case "w:noBreakHyphen":
                text = "-"
            case "w:drawing", "w:pict", "w:object":
                result.append(Span(text: picture(child)))
            default:
                text = nil
            }
            if let text {
                result.append(Span(text: text, bold: strong,
                                   italic: slanted))
            }
        }
        return result
    }

    private func linked(_ node: Markup, _ bold: Bool,
                        _ italic: Bool) -> [Span] {
        let id = node.attribute("r:id") ?? ""
        let anchor = node.attribute("w:anchor")
        var destination = relations.target(id) ?? ""
        if destination.isEmpty, let anchor {
            destination = "#" + anchor
        }
        let inner = Emitter.inline(spans(node, bold, italic))
        var result: [Span] = []
        if !inner.isEmpty {
            result = [Span(text: inner, link: destination)]
        }
        return result
    }

    // The payload is elided: base64 would dwarf the prose.

    private func picture(_ node: Markup) -> String {
        let described = node.find("wp:docPr").first
        let alt = PageReader.condensed(
            described?.attribute("descr")
                ?? described?.attribute("name") ?? "")
        let id = node.find("a:blip").first?
            .attribute("r:embed") ?? ""
        let part = relations.part(id) ?? ""
        let kind = part.hasSuffix(".jpg") || part.hasSuffix(".jpeg")
            ? "jpeg" : (part as NSString).pathExtension
        var result = ""
        if !part.isEmpty {
            result = "![\(alt)](data:image/\(kind);base64...)"
        }
        return result
    }

    private func table(_ node: Markup) -> DocsElement {
        var rows: [[Cell]] = []
        var origin: [Int: (Int, Int)] = [:]
        for row in node.all("w:tr") {
            var cells: [Cell] = []
            var column = 0
            for cell in row.all("w:tc") {
                let properties = cell.first("w:tcPr")
                let width = Int(properties?.first("w:gridSpan")?
                    .attribute("w:val") ?? "") ?? 1
                let merge = properties?.first("w:vMerge")
                let carried = merge != nil
                    && merge?.attribute("w:val") != "restart"
                let held = carried ? origin[column] : nil
                if let held {
                    let above = rows[held.0][held.1]
                    rows[held.0][held.1] = Cell(
                        text: above.text, column: above.column,
                        columns: above.columns, rows: above.rows + 1)
                } else {
                    origin[column] = (rows.count, cells.count)
                    cells.append(Cell(text: content(cell),
                                      column: column,
                                      columns: max(width, 1), rows: 1))
                }
                column += max(width, 1)
            }
            rows.append(cells)
        }
        return .table(rows)
    }

    private func content(_ cell: Markup) -> String {
        var lines: [String] = []
        for element in blocks(cell) {
            switch element {
            case .heading(_, let text):   lines.append(text)
            case .paragraph(let text):    lines.append(text)
            case .item(_, let text, _):   lines.append(text)
            case .quote(let text):        lines.append(text)
            case .code(_, let body):      lines.append(body)
            case .table(let rows):
                lines.append(rows.map { row in
                    row.map { cell in cell.text }
                        .joined(separator: " ")
                }.joined(separator: " "))
            case .rule:                   lines.append("")
            }
        }
        return lines.filter { line in !line.isEmpty }
            .joined(separator: "\n")
    }

    private static func on(_ properties: Markup?,
                           _ name: String) -> Bool {
        var result = false
        if let mark = properties?.first(name) {
            let value = mark.attribute("w:val") ?? "1"
            result = value != "0" && value != "false"
        }
        return result
    }

    private static func styles(_ archive: Archive) -> [String: String] {
        var result: [String: String] = [:]
        if let tree = archive.tree("word/styles.xml") {
            for style in tree.all("w:style") {
                let id = style.attribute("w:styleId") ?? ""
                let name = style.first("w:name")?
                    .attribute("w:val") ?? ""
                if !id.isEmpty { result[id] = name.lowercased() }
            }
        }
        return result
    }

    private static func numbering(_ archive: Archive)
            -> [String: Bool] {
        var abstracts: [String: [String: Bool]] = [:]
        var result: [String: Bool] = [:]
        if let tree = archive.tree("word/numbering.xml") {
            for abstract in tree.all("w:abstractNum") {
                let id = abstract.attribute("w:abstractNumId") ?? ""
                var levels: [String: Bool] = [:]
                for level in abstract.all("w:lvl") {
                    let depth = level.attribute("w:ilvl") ?? "0"
                    let format = level.first("w:numFmt")?
                        .attribute("w:val") ?? "bullet"
                    levels[depth] = format != "bullet"
                        && format != "none"
                }
                abstracts[id] = levels
            }
            for number in tree.all("w:num") {
                let id = number.attribute("w:numId") ?? ""
                let abstract = number.first("w:abstractNumId")?
                    .attribute("w:val") ?? ""
                for (depth, ordered) in abstracts[abstract] ?? [:] {
                    result["\(id):\(depth)"] = ordered
                }
            }
        }
        return result
    }

    private static func comments(_ archive: Archive)
            -> [String: [String]] {
        var result: [String: [String]] = [:]
        if let tree = archive.tree("word/comments.xml") {
            for comment in tree.all("w:comment") {
                let id = comment.attribute("w:id") ?? ""
                let said = comment.all("w:p").map { paragraph in
                    paragraph.find("w:t").map { text in
                        text.flattened
                    }.joined()
                }.filter { line in !line.isEmpty }
                if !id.isEmpty && !said.isEmpty { result[id] = said }
            }
        }
        return result
    }

    // Unknown shapes fall through: symbols kept, framing lost.

    private static func latex(_ node: Markup) -> String {
        var result = ""
        for child in node.children {
            switch child.name {
            case "m:t":
                result += child.flattened
            case "m:f":
                result += "\\frac{" + part(child, "m:num") + "}{"
                    + part(child, "m:den") + "}"
            case "m:sSup":
                result += part(child, "m:e") + "^{"
                    + part(child, "m:sup") + "}"
            case "m:sSub":
                result += part(child, "m:e") + "_{"
                    + part(child, "m:sub") + "}"
            case "m:sSubSup":
                result += part(child, "m:e") + "_{"
                    + part(child, "m:sub") + "}^{"
                    + part(child, "m:sup") + "}"
            case "m:rad":
                result += "\\sqrt{" + part(child, "m:e") + "}"
            case "m:d":
                result += "\\left(" + part(child, "m:e")
                    + "\\right)"
            case "m:nary":
                result += WordReader.operation(child) + "_{"
                    + part(child, "m:sub") + "}^{"
                    + part(child, "m:sup") + "}{"
                    + part(child, "m:e") + "}"
            default:
                result += latex(child)
            }
        }
        return result
    }

    private static func part(_ node: Markup,
                             _ name: String) -> String {
        node.first(name).map(WordReader.latex) ?? ""
    }

    private static func operation(_ node: Markup) -> String {
        let mark = node.first("m:naryPr")?.first("m:chr")?
            .attribute("m:val") ?? "\u{222B}"
        var result = "\\int"
        if mark == "\u{2211}" { result = "\\sum" }
        if mark == "\u{220F}" { result = "\\prod" }
        return result
    }
}

// Never fails; see okf/decisions/html-is-parsed-by-hand.md

final class HTMLParser {
    private let source: [Character]
    private var at = 0
    private var stack: [Markup] = []

    private static let voids: Set<String> = [
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"]

    private static let opaque: Set<String> = [
        "script", "style", "textarea"]

    static let blocks: Set<String> = [
        "address", "article", "aside", "blockquote", "details", "div",
        "dl", "fieldset", "figcaption", "figure", "footer", "form",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "main",
        "nav", "ol", "p", "pre", "section", "table", "ul"]

    private init(_ source: String) {
        self.source = Array(source)
    }

    static func tree(_ source: String) -> Markup {
        let reader = HTMLParser(source)
        let root = Markup("#document", [:])
        reader.stack = [root]
        reader.run()
        return root
    }

    private func run() {
        while at < source.count {
            if source[at] == "<" && at + 1 < source.count
                    && HTMLParser.markup(source[at + 1]) {
                tag()
            } else {
                text()
            }
        }
    }

    private static func markup(_ character: Character) -> Bool {
        character.isLetter || character == "/" || character == "!"
            || character == "?"
    }

    private func text() {
        var body = ""
        while at < source.count && !(source[at] == "<"
                && at + 1 < source.count
                && HTMLParser.markup(source[at + 1])) {
            body.append(source[at])
            at += 1
        }
        if !body.isEmpty {
            stack.last?.append(Markup(text: HTMLParser.decoded(body)))
        }
    }

    private func tag() {
        at += 1
        if source[at] == "!" {
            skipDeclaration()
        } else if source[at] == "?" {
            skipToClose()
        } else if source[at] == "/" {
            at += 1
            close(name())
            skipToClose()
        } else {
            open()
        }
    }

    private func skipDeclaration() {
        let comment = at + 2 < source.count && source[at + 1] == "-"
            && source[at + 2] == "-"
        if comment {
            at += 3
            while at + 2 < source.count && !(source[at] == "-"
                    && source[at + 1] == "-"
                    && source[at + 2] == ">") {
                at += 1
            }
            at = min(at + 3, source.count)
        } else {
            skipToClose()
        }
    }

    private func skipToClose() {
        while at < source.count && source[at] != ">" { at += 1 }
        at = min(at + 1, source.count)
    }

    private func name() -> String {
        var result = ""
        while at < source.count && (source[at].isLetter
                || source[at].isNumber || source[at] == "-"
                || source[at] == ":" || source[at] == "_") {
            result.append(source[at])
            at += 1
        }
        return result.lowercased()
    }

    private func open() {
        let tag = name()
        let (attributes, empty) = self.attributes()
        let node = Markup(tag, attributes)
        while let top = stack.last, HTMLParser.implies(tag, top.name) {
            stack.removeLast()
        }
        stack.last?.append(node)
        if !HTMLParser.voids.contains(tag) && !empty {
            stack.append(node)
        }
        if HTMLParser.opaque.contains(tag) {
            skipOpaque(tag)
        }
    }

    private func attributes() -> ([String: String], Bool) {
        var result: [String: String] = [:]
        while at < source.count && source[at] != ">" {
            while at < source.count && (source[at].isWhitespace
                    || source[at] == "/") {
                at += 1
            }
            let key = name()
            var value = ""
            skipSpace()
            if at < source.count && source[at] == "=" {
                at += 1
                skipSpace()
                value = attributeValue()
            }
            if key.isEmpty {
                while at < source.count && source[at] != ">"
                        && !source[at].isWhitespace {
                    at += 1
                }
            } else {
                result[key] = HTMLParser.decoded(value)
            }
        }
        let empty = at > 0 && at < source.count
            && source[at - 1] == "/"
        at = min(at + 1, source.count)
        return (result, empty)
    }

    private func attributeValue() -> String {
        var result = ""
        if at < source.count && (source[at] == "\""
                || source[at] == "'") {
            let quote = source[at]
            at += 1
            while at < source.count && source[at] != quote {
                result.append(source[at])
                at += 1
            }
            at = min(at + 1, source.count)
        } else {
            while at < source.count && !source[at].isWhitespace
                    && source[at] != ">" {
                result.append(source[at])
                at += 1
            }
        }
        return result
    }

    private func skipSpace() {
        while at < source.count && source[at].isWhitespace { at += 1 }
    }

    // A `<` in a regex or a `>` in a selector is text, not markup.

    private func skipOpaque(_ tag: String) {
        let sentinel = Array("</" + tag)
        var found = false
        var body = ""
        while at < source.count && !found {
            found = HTMLParser.matches(source, at, sentinel)
            if !found {
                body.append(source[at])
                at += 1
            }
        }
        if tag == "textarea" {
            stack.last?.append(Markup(text: HTMLParser.decoded(body)))
        }
        if found {
            at += sentinel.count
            close(tag)
            skipToClose()
        }
    }

    private static func matches(_ source: [Character], _ at: Int,
                                _ sentinel: [Character]) -> Bool {
        var step = 0
        var same = at + sentinel.count <= source.count
        while same && step < sentinel.count {
            same = Character(source[at + step].lowercased())
                == sentinel[step]
            step += 1
        }
        return same
    }

    // An end tag matching nothing is noise; the root never closes.

    private func close(_ tag: String) {
        var depth = stack.count - 1
        var found = -1
        while depth > 0 && found < 0 {
            if stack[depth].name == tag { found = depth }
            depth -= 1
        }
        if found > 0 { stack.removeSubrange(found..<stack.count) }
    }

    private static func implies(_ opening: String,
                                _ open: String) -> Bool {
        var result = false
        if open == "p" {
            result = HTMLParser.blocks.contains(opening)
        } else if open == "li" {
            result = opening == "li"
        } else if open == "dt" || open == "dd" {
            result = opening == "dt" || opening == "dd"
        } else if open == "option" {
            result = opening == "option" || opening == "optgroup"
        } else if open == "td" || open == "th" {
            result = opening == "td" || opening == "th"
                || opening == "tr"
        } else if open == "tr" {
            result = opening == "tr"
        } else if open == "thead" || open == "tbody" {
            result = opening == "tbody" || opening == "tfoot"
        }
        return result
    }

    private static let entities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"",
        "apos": "'", "nbsp": "\u{00A0}", "copy": "\u{00A9}",
        "reg": "\u{00AE}", "trade": "\u{2122}", "deg": "\u{00B0}",
        "hellip": "\u{2026}", "mdash": "\u{2014}",
        "ndash": "\u{2013}", "lsquo": "\u{2018}",
        "rsquo": "\u{2019}", "ldquo": "\u{201C}",
        "rdquo": "\u{201D}", "bull": "\u{2022}",
        "middot": "\u{00B7}", "times": "\u{00D7}",
        "divide": "\u{00F7}", "laquo": "\u{00AB}",
        "raquo": "\u{00BB}", "sect": "\u{00A7}",
        "para": "\u{00B6}", "dagger": "\u{2020}",
        "prime": "\u{2032}", "minus": "\u{2212}",
        "thinsp": "\u{2009}", "ensp": "\u{2002}",
        "emsp": "\u{2003}", "shy": "\u{00AD}", "euro": "\u{20AC}",
        "pound": "\u{00A3}", "yen": "\u{00A5}", "cent": "\u{00A2}"]

    static func decoded(_ text: String) -> String {
        var result = ""
        var reference = ""
        var inside = false
        for character in text {
            if inside && character == ";" {
                result += HTMLParser.resolved(reference)
                reference = ""
                inside = false
            } else if inside && reference.count < 32
                    && (character.isLetter || character.isNumber
                        || character == "#") {
                reference.append(character)
            } else if inside {
                result += "&" + reference
                reference = ""
                inside = character == "&"
                if !inside { result.append(character) }
            } else if character == "&" {
                inside = true
            } else {
                result.append(character)
            }
        }
        if inside { result += "&" + reference }
        return result
    }

    private static func resolved(_ reference: String) -> String {
        var result = "&" + reference + ";"
        if let named = HTMLParser.entities[reference] {
            result = named
        } else if reference.hasPrefix("#x")
                    || reference.hasPrefix("#X") {
            let digits = reference.dropFirst(2)
            if let code = UInt32(digits, radix: 16),
               let scalar = Unicode.Scalar(code) {
                result = String(Character(scalar))
            }
        } else if reference.hasPrefix("#") {
            let digits = reference.dropFirst()
            if let code = UInt32(digits), let scalar =
                Unicode.Scalar(code) {
                result = String(Character(scalar))
            }
        }
        return result
    }
}

final class PageReader {
    private struct Style {
        var bold = false
        var italic = false
        var code = false
        var link: String? = nil
    }

    private let root: Markup
    private let titled: Bool
    private var output: [DocsElement] = []
    private var pending: [Span] = []

    private static let ignored: Set<String> = [
        "head", "script", "style", "noscript", "nav", "header",
        "footer",
        "aside", "form", "svg", "iframe", "template", "button",
        "select", "canvas", "audio", "video", "map", "object"]

    init(_ source: String) {
        root = HTMLParser.tree(source)
        titled = true
    }

    init(tree: Markup) {
        root = tree
        titled = false
    }

    func elements() -> [DocsElement] {
        output = []
        pending = []
        let head = root.find("head").first ?? root
        if titled, let title = head.find("title").first {
            let text = PageReader.condensed(title.flattened)
                .trimmingCharacters(in: .whitespaces)
            if !text.isEmpty { output.append(.heading(1, text)) }
        }
        walk(root, Style(), 0)
        flush()
        return output
    }

    private func flush() {
        let text = Emitter.inline(pending)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        pending = []
        if !text.isEmpty { output.append(.paragraph(text)) }
    }

    private func walk(_ parent: Markup, _ style: Style,
                      _ depth: Int) {
        for node in parent.children where !skipped(node) {
            if node.name.isEmpty {
                pending.append(span(
                    PageReader.condensed(node.text), style))
            } else {
                visit(node, style, depth)
            }
        }
    }

    private func visit(_ node: Markup, _ style: Style, _ depth: Int) {
        let tag = node.name
        if let level = PageReader.level(tag) {
            flush()
            output.append(.heading(level, inline(node, style)))
        } else if tag == "br" {
            pending.append(Span(text: "\n"))
        } else if tag == "hr" {
            flush()
            output.append(.rule)
        } else if tag == "img" {
            pending.append(Span(text: PageReader.image(node)))
        } else if tag == "a" {
            walk(node, linking(node, style), depth)
        } else if tag == "strong" || tag == "b" {
            var marked = style
            marked.bold = true
            walk(node, marked, depth)
        } else if tag == "em" || tag == "i" || tag == "cite" {
            var marked = style
            marked.italic = true
            walk(node, marked, depth)
        } else if tag == "code" || tag == "kbd" || tag == "samp" {
            var marked = style
            marked.code = true
            walk(node, marked, depth)
        } else if tag == "pre" {
            flush()
            output.append(.code(PageReader.language(node),
                                  node.flattened
                                      .trimmingCharacters(
                                          in: .newlines)))
        } else if tag == "blockquote" {
            flush()
            output.append(.quote(PageReader.quoted(node)))
        } else if tag == "ul" || tag == "ol" || tag == "dl" {
            flush()
            list(node, tag == "ol", depth)
        } else if tag == "table" {
            flush()
            output.append(.table(PageReader.table(node)))
        } else if HTMLParser.blocks.contains(tag) || tag == "li"
                    || tag == "tr" || tag == "body" {
            flush()
            walk(node, style, depth)
            flush()
        } else {
            walk(node, style, depth)
        }
    }

    private func list(_ node: Markup, _ ordered: Bool, _ depth: Int) {
        for child in node.children where !skipped(child) {
            let listed = child.name == "li" || child.name == "dd"
                || child.name == "dt"
            if listed {
                let text = inline(child, Style())
                if !text.isEmpty {
                    output.append(.item(depth, text,
                                        ordered ? 1 : nil))
                }
                for nested in child.children
                        where nested.name == "ul"
                            || nested.name == "ol" {
                    list(nested, nested.name == "ol", depth + 1)
                }
            } else if child.name == "ul" || child.name == "ol" {
                list(child, child.name == "ol", depth + 1)
            }
        }
    }

    // A nested list is left to the caller so it keeps its depth.

    private func inline(_ node: Markup, _ style: Style) -> String {
        let held = pending
        pending = []
        for child in node.children where !skipped(child) {
            let nested = child.name == "ul" || child.name == "ol"
            if child.name.isEmpty {
                pending.append(span(
                    PageReader.condensed(child.text), style))
            } else if !nested {
                visit(child, style, 0)
            }
        }
        let text = Emitter.inline(pending)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        pending = held
        return text
    }

    private func span(_ text: String, _ style: Style) -> Span {
        Span(text: text, bold: style.bold, italic: style.italic,
             code: style.code, link: style.link)
    }

    private func linking(_ node: Markup, _ style: Style) -> Style {
        var result = style
        let target = node.attribute("href") ?? ""
        let title = node.attribute("title") ?? ""
        if !target.isEmpty && !target.hasPrefix("javascript:") {
            result.link = title.isEmpty
                ? target : target + " \"" + title + "\""
        }
        return result
    }

    private func skipped(_ node: Markup) -> Bool {
        PageReader.ignored.contains(node.name)
            || node.attributes["hidden"] != nil
            || node.attribute("aria-hidden") == "true"
            || node.attribute("role") == "navigation"
            || node.attribute("role") == "banner"
            || node.attribute("role") == "note"
    }

    private static func level(_ tag: String) -> Int? {
        var result: Int? = nil
        if tag.count == 2 && tag.hasPrefix("h") {
            result = Int(tag.dropFirst())
        }
        return result
    }

    private static func image(_ node: Markup) -> String {
        let source = node.attribute("src") ?? ""
        let alt = node.attribute("alt") ?? ""
        return source.isEmpty ? "" : "![\(alt)](\(source))"
    }

    private static func language(_ node: Markup) -> String {
        var result = ""
        for child in node.find("code") {
            for name in (child.attribute("class") ?? "")
                    .split(separator: " ") {
                if name.hasPrefix("language-") {
                    result = String(name.dropFirst(9))
                }
            }
        }
        return result
    }

    private static func quoted(_ node: Markup) -> String {
        let inner = PageReader(tree: node).elements()
        return Emitter.markdown(inner)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func table(_ node: Markup) -> [[Cell]] {
        var rows: [[Cell]] = []
        var carried: [Int: Int] = [:]
        for row in node.find("tr") {
            var cells: [Cell] = []
            var column = 0
            for cell in row.children where cell.name == "td"
                    || cell.name == "th" {
                while (carried[column] ?? 0) > 0 {
                    carried[column] = (carried[column] ?? 0) - 1
                    column += 1
                }
                let across = Int(cell.attribute("colspan") ?? "") ?? 1
                let down = Int(cell.attribute("rowspan") ?? "") ?? 1
                let text = PageReader(tree: cell).elements()
                cells.append(Cell(
                    text: Emitter.markdown(text).trimmingCharacters(
                        in: .whitespacesAndNewlines),
                    column: column, columns: max(across, 1),
                    rows: max(down, 1)))
                var step = 0
                while step < max(across, 1) {
                    carried[column + step] = max(down, 1) - 1
                    step += 1
                }
                column += max(across, 1)
            }
            rows.append(cells)
        }
        return rows
    }

    // A surviving newline would split the Markdown paragraph in two.

    static func condensed(_ text: String) -> String {
        var result = ""
        var spacing = false
        for character in text {
            let blank = character.isWhitespace
                || character == "\u{00A0}"
            if blank && !spacing {
                result.append(" ")
            } else if !blank {
                result.append(character)
            }
            spacing = blank
        }
        return result
    }
}

struct SheetReader {
    private let archive: Archive
    private let shared: [String]
    private let dated: [Bool]
    private let sheets: [(String, String)]

    init(_ archive: Archive) throws {
        let book = archive.tree("xl/workbook.xml")
        if let book {
            let relations = Relations(
                archive, "xl/_rels/workbook.xml.rels")
            self.archive = archive
            shared = SheetReader.strings(archive)
            dated = SheetReader.formats(archive)
            sheets = (book.first("sheets")?.all("sheet") ?? [])
                .map { sheet in
                    (sheet.attribute("name") ?? "Sheet",
                     relations.part(sheet.attribute("r:id") ?? "")
                        ?? "")
                }
        } else {
            throw DocsError.missingPart("xl/workbook.xml")
        }
    }

    func elements() -> [DocsElement] {
        var result: [DocsElement] = []
        for (name, part) in sheets {
            let tree = part.isEmpty ? nil : archive.tree(part)
            let rows = tree.map(grid) ?? []
            result.append(.heading(2, name))
            if !rows.isEmpty { result.append(.table(rows)) }
        }
        return result
    }

    private func grid(_ tree: Markup) -> [[Cell]] {
        let (spans, covered) = SheetReader.merges(tree)
        var rows: [[Cell]] = []
        var line = 0
        for row in tree.first("sheetData")?.all("row") ?? [] {
            line = Int(row.attribute("r") ?? "") ?? line + 1
            var cells: [Cell] = []
            var column = 0
            for cell in row.all("c") {
                let reference = cell.attribute("r") ?? ""
                let at = SheetReader.column(reference, column)
                let text = value(cell)
                let key = "\(at):\(line)"
                let span = spans[key] ?? (1, 1)
                if !covered.contains(key)
                        && (!text.isEmpty || span != (1, 1)) {
                    cells.append(Cell(text: text, column: at,
                                      columns: span.0,
                                      rows: span.1))
                }
                column = at + 1
            }
            rows.append(cells)
        }
        return rows
    }

    // Swallowed cells are still in sheetData, hence the covered set.

    private static func merges(_ tree: Markup)
            -> ([String: (Int, Int)], Set<String>) {
        var spans: [String: (Int, Int)] = [:]
        var covered: Set<String> = []
        let listed = tree.first("mergeCells")?.all("mergeCell") ?? []
        for merge in listed {
            let corners = (merge.attribute("ref") ?? "")
                .split(separator: ":").map(String.init)
            let left = SheetReader.column(corners.first ?? "", 0)
            let right = SheetReader.column(corners.last ?? "", 0)
            let top = SheetReader.line(corners.first ?? "")
            let bottom = SheetReader.line(corners.last ?? "")
            if corners.count == 2 && right >= left && bottom >= top {
                spans["\(left):\(top)"] = (right - left + 1,
                                            bottom - top + 1)
                for column in left...right {
                    for row in top...bottom where
                            column != left || row != top {
                        covered.insert("\(column):\(row)")
                    }
                }
            }
        }
        return (spans, covered)
    }

    private static func line(_ reference: String) -> Int {
        Int(reference.filter { character in character.isNumber })
            ?? 1
    }

    private func value(_ cell: Markup) -> String {
        let kind = cell.attribute("t") ?? "n"
        let stored = cell.first("v")?.flattened ?? ""
        var result = ""
        switch kind {
        case "s":
            let index = Int(stored) ?? -1
            result = index >= 0 && index < shared.count
                ? shared[index] : ""
        case "inlineStr":
            result = cell.first("is")?.flattened ?? ""
        case "b":
            result = stored == "1" ? "TRUE" : "FALSE"
        case "str", "e":
            result = stored
        default:
            result = number(cell, stored)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Only the cell's number format says the serial is a date.

    private func number(_ cell: Markup, _ stored: String) -> String {
        let style = Int(cell.attribute("s") ?? "") ?? -1
        let isDate = style >= 0 && style < dated.count && dated[style]
        var result = stored
        if isDate, let serial = Double(stored), serial > 0 {
            result = SheetReader.moment(serial)
        }
        return result
    }

    // 1899-12-30, not -31: it counts a 1900-02-29 that never was.

    private static func moment(_ serial: Double) -> String {
        let seconds = (serial - 25_569) * 86_400
        let when = Date(timeIntervalSince1970: seconds)
        let shape = DateFormatter()
        shape.timeZone = TimeZone(identifier: "UTC")
        shape.locale = Locale(identifier: "en_US_POSIX")
        let whole = serial == serial.rounded(.down)
        shape.dateFormat = whole ? "yyyy-MM-dd"
                                 : "yyyy-MM-dd HH:mm:ss"
        return shape.string(from: when)
    }

    private static func column(_ reference: String,
                               _ fallback: Int) -> Int {
        var result = 0
        var letters = 0
        for character in reference.uppercased() {
            if character.isLetter, let value = character.asciiValue {
                result = result * 26 + Int(value - 64)
                letters += 1
            }
        }
        return letters > 0 ? result - 1 : fallback
    }

    private static func strings(_ archive: Archive) -> [String] {
        var result: [String] = []
        if let tree = archive.tree("xl/sharedStrings.xml") {
            for item in tree.all("si") {
                var text = item.all("t").map { node in
                    node.flattened
                }.joined()
                for run in item.all("r") {
                    text += run.all("t").map { node in
                        node.flattened
                    }.joined()
                }
                result.append(text)
            }
        }
        return result
    }

    private static let builtinDates: Set<Int> = [
        14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47]

    private static func formats(_ archive: Archive) -> [Bool] {
        var custom: [Int: String] = [:]
        var result: [Bool] = []
        if let tree = archive.tree("xl/styles.xml") {
            for format in tree.first("numFmts")?.all("numFmt") ?? [] {
                let id = Int(format.attribute("numFmtId") ?? "") ?? -1
                custom[id] = format.attribute("formatCode") ?? ""
            }
            for entry in tree.first("cellXfs")?.all("xf") ?? [] {
                let id = Int(entry.attribute("numFmtId") ?? "") ?? 0
                let code = custom[id] ?? ""
                result.append(SheetReader.builtinDates.contains(id)
                              || SheetReader.temporal(code))
            }
        }
        return result
    }

    // [Red] and [$-409] would otherwise read as placeholders.

    private static func temporal(_ code: String) -> Bool {
        var literal = false
        var bracketed = false
        var found = false
        for character in code {
            if character == "\"" {
                literal = !literal
            } else if character == "[" {
                bracketed = true
            } else if character == "]" {
                bracketed = false
            } else if !literal && !bracketed {
                found = found || "ymdhs".contains(
                    Character(character.lowercased()))
            }
        }
        return found
    }
}

// A slide is a canvas: slide order is the only order pptx states.

struct SlideReader {
    private let archive: Archive
    private let parts: [String]

    init(_ archive: Archive) throws {
        let show = archive.tree("ppt/presentation.xml")
        if let show {
            let relations = Relations(
                archive, "ppt/_rels/presentation.xml.rels")
            self.archive = archive
            parts = (show.first("p:sldIdLst")?.all("p:sldId") ?? [])
                .compactMap { slide in
                    relations.part(slide.attribute("r:id") ?? "")
                }
        } else {
            throw DocsError.missingPart("ppt/presentation.xml")
        }
    }

    func elements() -> [DocsElement] {
        var result: [DocsElement] = []
        for (index, part) in parts.enumerated() {
            let relations = Relations(
                archive, SlideReader.rels(part))
            result.append(.heading(2, "Slide \(index + 1)"))
            if let tree = archive.tree(part),
               let canvas = tree.first("p:cSld")?
                .first("p:spTree") {
                result += shapes(canvas, relations)
            }
            result += notes(relations)
        }
        return result
    }

    private static func rels(_ part: String) -> String {
        var pieces = part.split(separator: "/").map(String.init)
        let name = pieces.popLast() ?? ""
        return (pieces + ["_rels", name + ".rels"])
            .joined(separator: "/")
    }

    private func shapes(_ parent: Markup,
                        _ relations: Relations) -> [DocsElement] {
        var result: [DocsElement] = []
        for node in parent.children {
            switch node.name {
            case "p:sp":
                result += frame(node)
            case "p:grpSp":
                result += shapes(node, relations)
            case "p:graphicFrame":
                result += framed(node, relations)
            case "p:pic":
                result += [picture(node, relations)]
            default:
                result += []
            }
        }
        return result
    }

    private func frame(_ shape: Markup) -> [DocsElement] {
        let role = shape.first("p:nvSpPr")?.first("p:nvPr")?
            .first("p:ph")?.attribute("type") ?? ""
        let titled = role == "title" || role == "ctrTitle"
        let paragraphs = shape.first("p:txBody")?.all("a:p") ?? []
        let lines = paragraphs.map { paragraph in
            (SlideReader.depth(paragraph),
             SlideReader.text(paragraph))
        }.filter { line in !line.1.isEmpty }
        var result: [DocsElement] = []
        if titled {
            result = lines.map { line in .heading(3, line.1) }
        } else if lines.count > 1 {
            result = lines.map { line in
                .item(line.0, line.1, nil)
            }
        } else {
            result = lines.map { line in .paragraph(line.1) }
        }
        return result
    }

    private func framed(_ shape: Markup,
                        _ relations: Relations) -> [DocsElement] {
        let data = shape.first("a:graphic")?.first("a:graphicData")
        var result: [DocsElement] = []
        if let table = data?.first("a:tbl") {
            result = [.table(SlideReader.grid(table))]
        } else if let reference = data?.first("c:chart") {
            let id = reference.attribute("r:id") ?? ""
            let part = relations.part(id) ?? ""
            result = chart(part)
        }
        return result
    }

    // The cached numbers are all there is; the drawing is not here.

    private func chart(_ part: String) -> [DocsElement] {
        var result: [DocsElement] = []
        if let tree = archive.tree(part) {
            let title = tree.find("c:title").first
                .map { node in
                    node.find("a:t").map { text in
                        text.flattened
                    }.joined()
                } ?? ""
            var names: [String] = []
            var columns: [[String]] = []
            var categories: [String] = []
            for series in tree.find("c:ser") {
                names.append(series.first("c:tx")
                    .map(SlideReader.cached)?.first ?? "Series")
                let found = series.first("c:cat")
                    .map(SlideReader.cached) ?? []
                if found.count > categories.count {
                    categories = found
                }
                columns.append(series.first("c:val")
                    .map(SlideReader.cached) ?? [])
            }
            if !title.isEmpty { result.append(.heading(3, title)) }
            if !columns.isEmpty {
                result.append(.table(SlideReader.tabulated(
                    names, categories, columns)))
            }
        }
        return result
    }

    private static func tabulated(_ names: [String],
                                  _ categories: [String],
                                  _ columns: [[String]]) -> [[Cell]] {
        var rows: [[Cell]] = []
        var head = [Cell(text: "", column: 0, columns: 1, rows: 1)]
        for (index, name) in names.enumerated() {
            head.append(Cell(text: name, column: index + 1,
                             columns: 1, rows: 1))
        }
        rows.append(head)
        let deep = columns.reduce(categories.count) { most, column in
            max(most, column.count)
        }
        for line in 0..<deep {
            let label = line < categories.count
                ? categories[line] : "\(line + 1)"
            var cells = [Cell(text: label, column: 0, columns: 1,
                              rows: 1)]
            for (index, column) in columns.enumerated() {
                let text = line < column.count ? column[line] : ""
                cells.append(Cell(text: text, column: index + 1,
                                  columns: 1, rows: 1))
            }
            rows.append(cells)
        }
        return rows
    }

    private static func cached(_ node: Markup) -> [String] {
        node.find("c:pt").map { point in
            point.find("c:v").map { value in
                value.flattened
            }.joined()
        }
    }

    private func picture(_ shape: Markup,
                         _ relations: Relations) -> DocsElement {
        let described = shape.first("p:nvPicPr")?.first("p:cNvPr")
        let alt = PageReader.condensed(
            described?.attribute("descr") ?? "")
        let name = described?.attribute("name") ?? "image"
        let id = shape.first("p:blipFill")?.first("a:blip")?
            .attribute("r:embed") ?? ""
        let part = relations.part(id) ?? ""
        let kind = (part as NSString).pathExtension
        let file = SlideReader.plain(name)
            + (kind.isEmpty ? "" : "." + kind)
        return .paragraph("![\(alt)](\(file))")
    }

    // The author named the shape; the media part is `image7.png`.

    private static func plain(_ name: String) -> String {
        String(name.unicodeScalars.filter { scalar in
            CharacterSet.alphanumerics.contains(scalar)
                || scalar == "_"
        }.map(Character.init))
    }

    private func notes(_ relations: Relations) -> [DocsElement] {
        var result: [DocsElement] = []
        for part in relations.parts() where part.contains("notesSlide") {
            let said = (archive.tree(part)?.find("a:p") ?? [])
                .map(SlideReader.text)
                .filter { line in !line.isEmpty }
            if !said.isEmpty {
                result.append(.heading(3, "Notes"))
                result += said.map { line in .paragraph(line) }
            }
        }
        return result
    }

    private static func depth(_ paragraph: Markup) -> Int {
        Int(paragraph.first("a:pPr")?.attribute("lvl") ?? "") ?? 0
    }

    static func text(_ paragraph: Markup) -> String {
        var spans: [Span] = []
        for node in paragraph.children {
            if node.name == "a:r" {
                let marks = node.first("a:rPr")
                spans.append(Span(
                    text: node.first("a:t")?.flattened ?? "",
                    bold: marks?.attribute("b") == "1",
                    italic: marks?.attribute("i") == "1"))
            } else if node.name == "a:br" {
                spans.append(Span(text: "\n"))
            } else if node.name == "a:fld" {
                spans.append(Span(
                    text: node.first("a:t")?.flattened ?? ""))
            }
        }
        return Emitter.inline(spans)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func grid(_ table: Markup) -> [[Cell]] {
        var rows: [[Cell]] = []
        for row in table.all("a:tr") {
            var cells: [Cell] = []
            var column = 0
            for cell in row.all("a:tc") {
                let across = Int(cell.attribute("gridSpan") ?? "") ?? 1
                let down = Int(cell.attribute("rowSpan") ?? "") ?? 1
                let merged = cell.attribute("hMerge") == "1"
                    || cell.attribute("vMerge") == "1"
                if !merged {
                    let said = (cell.first("a:txBody")?.all("a:p")
                        ?? []).map(SlideReader.text)
                        .filter { line in !line.isEmpty }
                    cells.append(Cell(
                        text: said.joined(separator: "\n"),
                        column: column, columns: max(across, 1),
                        rows: max(down, 1)))
                }
                column += max(across, 1)
            }
            rows.append(cells)
        }
        return rows
    }
}

struct BookReader {
    private let archive: Archive
    private let package: Markup
    private let base: String

    init(_ archive: Archive) throws {
        let container = archive.tree("META-INF/container.xml")
        let path = container?.find("rootfile").first?
            .attribute("full-path")
        let package = path.flatMap { name in archive.tree(name) }
        if let package, let path {
            self.archive = archive
            self.package = package
            base = BookReader.folder(path)
        } else {
            throw DocsError.missingPart("META-INF/container.xml")
        }
    }

    private static func folder(_ path: String) -> String {
        var pieces = path.split(separator: "/").map(String.init)
        pieces.removeLast()
        return pieces.isEmpty
            ? "" : pieces.joined(separator: "/") + "/"
    }

    func elements() -> [DocsElement] {
        var result = front()
        for part in spine() {
            if let source = archive.text(part) {
                result += PageReader(tree: HTMLParser.tree(source))
                    .elements()
            }
        }
        return result
    }

    private func front() -> [DocsElement] {
        let metadata = package.first("metadata")
        var result: [DocsElement] = []
        let title = metadata?.first("dc:title")?.flattened ?? ""
        let authors = (metadata?.all("dc:creator") ?? [])
            .map { node in node.flattened }
            .filter { name in !name.isEmpty }
        let about = metadata?.first("dc:description")?.flattened ?? ""
        if !title.isEmpty { result.append(.heading(1, title)) }
        if !authors.isEmpty {
            result.append(.paragraph("**Authors:** "
                + authors.joined(separator: ", ")))
        }
        if !about.isEmpty { result.append(.paragraph(about)) }
        return result
    }

    // The nav document is the book's own contents, not content.

    private func spine() -> [String] {
        var sources: [String: String] = [:]
        for item in package.first("manifest")?.all("item") ?? [] {
            let id = item.attribute("id") ?? ""
            let href = item.attribute("href") ?? ""
            let navigational =
                (item.attribute("properties") ?? "").contains("nav")
            if !id.isEmpty && !navigational {
                sources[id] = Relations.resolved(base, href)
            }
        }
        return (package.first("spine")?.all("itemref") ?? [])
            .compactMap { entry in
                sources[entry.attribute("idref") ?? ""]
            }
    }
}
