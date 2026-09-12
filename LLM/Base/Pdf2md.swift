import CoreGraphics
import Foundation
import PDFKit
import Vision

enum Mode: String, CaseIterable {
    case text
    case vision
    case geometry
    case docling
    case hybrid
}

enum ConversionError: Error, CustomStringConvertible {
    case unreadable(URL)
    case rasterFailed(Int)
    case notImplemented(Mode)

    var description: String {
        var result = ""
        switch self {
        case .unreadable(let url):
            result = "not a readable PDF with pages: \(url.path)"
        case .rasterFailed(let page):
            result = "cannot rasterize page \(page)"
        case .notImplemented(let mode):
            result = "mode '\(mode.rawValue)' is not implemented yet"
        }
        return result
    }
}

// Frames are Vision-normalized: 0...1, origin bottom-left, y upward.
struct Fragment {
    let index: Int
    let text: String
    let frame: CGRect
}

// see okf/decisions/cells-carry-their-spans.md
struct Field {
    let text: String
    let spans: [Int]
}

struct Table {
    let rows: [[Field]]
    let frame: CGRect?
}

enum Element {
    case paragraph(String)
    case table(Table)
}

struct Page {
    let number: Int
    let elements: [Element]
}

// see okf/constraints/every-span-reaches-the-output-once.md
struct Audit {
    let spans: Int
    let lost: Int
    let unaccounted: Int
}

// `position` is y for a level rule, x for an upright; `span` the rest.
struct Ruling {
    let position: CGFloat
    let span: ClosedRange<CGFloat>
}

struct Grid {
    let rows: [ClosedRange<CGFloat>]
    let columns: [ClosedRange<CGFloat>]
    let header: Int
}

// Every threshold below is a fraction of the page unless it is a count.
// The frame each is measured in, and the sweep behind its value, are in
// okf/glossary/the-converter-knobs.md.
struct Converter {
    var mode: Mode = .vision
    var scale: CGFloat = 3
    var bandTolerance: CGFloat = 0.6
    var columnGap: CGFloat = 0.004
    var writtenGap: CGFloat = 0.012
    var pageGap: CGFloat = 0.004
    var pageMiddle: ClosedRange<CGFloat> = 0.3...0.7
    var pageShare: CGFloat = 0.15
    var cellGap: CGFloat = 0.006
    var buckets = 1024
    var inkLevel: UInt8 = 250
    var glyphLevel: UInt8 = 128
    var inkShare: CGFloat = 0.35
    var rulingWidth: CGFloat = 0.1
    var rulingHeight: CGFloat = 0.03
    var rulingMerge: CGFloat = 0.005
    var spanAgreement: CGFloat = 0.5
    var fullWidth: CGFloat = 0.8
    var alignedRows = 3
    var proseCellShare: CGFloat = 0.3
    var proseCellLength = 30
    var cellVote: CGFloat = 0.10
    var spanWindow: CGFloat = 2.2
    var rowGapRatio: CGFloat = 1.35
    var trace = false
    var rowShare: CGFloat = 0.80
    var numericShare: CGFloat = 0.95
    var packing: CGFloat = 2.5
    var recoverRows = 3
    var onDocling: ((CGImage, Int) throws -> [Element])?
    var onReadings: ((Int, [Fragment], [Element], [Element]) -> Void)?
    var doclingScale: CGFloat = 0
    var onAudit: ((Int, Audit) -> Void)?

    private var wordGap: CGFloat {
        mode == .geometry ? writtenGap : columnGap
    }

    // TODO: see okf/decisions/a-second-output-shape-for-vit-clients.md

    func markdown(of url: URL) async throws -> String {
        let document = try Converter.require(PDFDocument(url: url),
                                             .unreadable(url))
        let pages = try await self.pages(of: document, url)
        return Converter.render(pages)
    }

    private func pages(of document: PDFDocument,
                       _ url: URL) async throws -> [Page] {
        var result: [Page] = []
        if document.pageCount == 0 {
            throw ConversionError.unreadable(url)
        }
        for index in 0..<document.pageCount {
            if let page = document.page(at: index) {
                let elements = try await self.elements(of: page, index + 1)
                result.append(Page(number: index + 1, elements: elements))
            }
        }
        return result
    }

    private func elements(of page: PDFPage,
                          _ number: Int) async throws -> [Element] {
        var result: [Element] = []
        switch mode {
        case .text:
            result = Converter.paragraphs(of: page.string ?? "")
        case .vision, .geometry:
            result = try await rebuilt(page, number).0
        case .docling:
            result = try told(page, number)
        case .hybrid:
            let (drawn, spans) = try await rebuilt(page, number)
            result = merged(try told(page, number), drawn, spans,
                            number)
        }
        return result
    }

    private func told(_ page: PDFPage, _ number: Int) throws -> [Element] {
        let read = try Converter.require(onDocling, .notImplemented(mode))
        let wanted = doclingScale > 0 ? doclingScale : scale
        return try read(try Converter.raster(page, wanted, number,
                                             false), number)
    }

    // see okf/findings/pairing-and-arbitrating-two-tables.md

    private func merged(_ told: [Element], _ drawn: [Element],
                        _ spans: [Fragment], _ number: Int) -> [Element] {
        let words = Converter.spoken(spans)
        let others = drawn.compactMap { element -> Table? in
            var result: Table? = nil
            if case .table(let table) = element { result = table }
            return result
        }
        var taken = [Bool](repeating: false, count: others.count)
        var placed: [Int: Int] = [:]
        var result: [Element] = []
        var swapped = 0
        var grounded: CGFloat = 0
        var counted = 0
        for element in told {
            switch element {
            case .paragraph:
                result.append(element)
            case .table(let raw):
                let table = Converter.linked(raw, spans)
                grounded += Converter.rate(table)
                counted += 1
                var keep = table
                if let at = Converter.nearest(table.rows, others, taken) {
                    taken[at] = true
                    placed[at] = result.count
                    if !trusted(table.rows, others[at].rows, words) {
                        keep = others[at]
                        swapped += 1
                    }
                }
                result.append(.table(keep))
            }
        }
        let found = recovered(others, taken, &result, placed)
        if let watch = onReadings {
            watch(number, spans, told.map { one in
                Converter.grounded(one, spans)
            }, drawn)
        }
        let share = counted > 0 ? grounded / CGFloat(counted) : 1
        report(String(format: "    hybrid: %d paired, %d taken from the "
                      + "drawn reading, %d recovered, %.3f of told cell "
                      + "tokens linked to spans", placed.count, swapped,
                      found, Double(share)))
        return result
    }

    // Descending, so an earlier insertion cannot shift a later one.

    private func recovered(_ others: [Table], _ taken: [Bool],
                           _ result: inout [Element],
                           _ placed: [Int: Int]) -> Int {
        var pending: [(Int, Table)] = []
        for (at, other) in others.enumerated()
        where !taken[at] && other.rows.count >= recoverRows
            && (other.rows.first?.count ?? 0) >= 2 {
            pending.append((at, other))
        }
        for (at, other) in pending.sorted(by: { left, right in
            left.0 > right.0
        }) {
            var after = 0
            for (index, position) in placed where index < at {
                after = max(after, position + 1)
            }
            result.insert(.table(other), at: min(after, result.count))
        }
        return pending.count
    }

    // see okf/decisions/cells-carry-their-spans.md

    private static func linked(_ table: Table,
                               _ spans: [Fragment]) -> Table {
        var pool: [String: [Int]] = [:]
        for (at, span) in spans.enumerated() where
            table.frame.map({ box in box.intersects(span.frame) }) ?? true {
            for word in Converter.tokens(span.text) {
                pool[word, default: []].append(at)
            }
        }
        var used = Set<Int>()
        var rows: [[Field]] = []
        for (down, row) in table.rows.enumerated() {
            var built: [Field] = []
            for (across, cell) in row.enumerated() {
                let want = Converter.expected(table, down, across,
                                              table.rows.count, row.count)
                var found: [Int] = []
                for word in Converter.tokens(cell.text) {
                    let one = Converter.closest(pool[word] ?? [], used,
                                                want, spans)
                    if let one {
                        used.insert(one)
                        found.append(one)
                    }
                }
                built.append(Field(text: cell.text, spans: found))
            }
            rows.append(built)
        }
        return Table(rows: rows, frame: table.frame)
    }

    private static func grounded(_ element: Element,
                                 _ spans: [Fragment]) -> Element {
        var result = element
        if case .table(let table) = element {
            result = .table(Converter.linked(table, spans))
        }
        return result
    }

    // Where the cell would sit if the table divided its box evenly.

    private static func expected(_ table: Table, _ down: Int,
                                 _ across: Int, _ rows: Int,
                                 _ columns: Int) -> CGPoint {
        var result = CGPoint(x: 0.5, y: 0.5)
        if let box = table.frame, rows > 0, columns > 0 {
            result = CGPoint(
                x: box.minX + (CGFloat(across) + 0.5) * box.width
                    / CGFloat(columns),
                y: box.maxY - (CGFloat(down) + 0.5) * box.height
                    / CGFloat(rows))
        }
        return result
    }

    private static func closest(_ pool: [Int], _ used: Set<Int>,
                                _ want: CGPoint,
                                _ spans: [Fragment]) -> Int? {
        var best: Int? = nil
        var nearest = CGFloat.infinity
        for at in pool where !used.contains(at) {
            let middle = CGPoint(x: spans[at].frame.midX,
                                 y: spans[at].frame.midY)
            let away = (middle.x - want.x) * (middle.x - want.x)
                + (middle.y - want.y) * (middle.y - want.y)
            if away < nearest {
                nearest = away
                best = at
            }
        }
        return best
    }

    // How much of a told table the page's own words account for.

    private static func rate(_ table: Table) -> CGFloat {
        var linked = 0
        var total = 0
        for row in table.rows {
            for cell in row {
                total += Converter.tokens(cell.text).count
                linked += cell.spans.count
            }
        }
        return total > 0 ? CGFloat(linked) / CGFloat(total) : 1
    }

    // A ONE-SIDED test: the drawn reading scores 1 by construction.

    private static func spoken(_ spans: [Fragment]) -> [String: Int] {
        var result: [String: Int] = [:]
        for span in spans {
            for word in Converter.tokens(span.text) {
                result[word, default: 0] += 1
            }
        }
        return result
    }

    // see okf/findings/pairing-and-arbitrating-two-tables.md

    private static func tokens(_ text: String) -> [String] {
        var result: [String] = []
        let plain = text.decomposedStringWithCompatibilityMapping
        for piece in plain.lowercased().split(whereSeparator: { character in
            character.isWhitespace
        }) {
            let trimmed = piece.trimmingCharacters(
                in: CharacterSet.alphanumerics.inverted)
            if !trimmed.isEmpty { result.append(trimmed) }
        }
        return result
    }

    // Keyed without spaces: two readings of one cell disagree
    // constantly about where a space falls.

    private static func bagged(_ rows: [[Field]]) -> [String: Int] {
        var result: [String: Int] = [:]
        for row in rows {
            for cell in row {
                let key = Converter.tokens(cell.text).joined()
                if !key.isEmpty { result[key, default: 0] += 1 }
            }
        }
        return result
    }

    // Paired on the CELL bags, so a pair survives disagreement about
    // grouping, spans and row count.

    private static func nearest(_ rows: [[Field]],
                                _ others: [Table],
                                _ taken: [Bool]) -> Int? {
        let mine = Converter.bagged(rows)
        var best: Int? = nil
        var strongest = 0.1
        for (index, other) in others.enumerated() where !taken[index] {
            let claim = Converter.overlap(mine,
                                          Converter.bagged(other.rows))
            if claim > strongest {
                strongest = claim
                best = index
            }
        }
        return best
    }

    private static func overlap(_ mine: [String: Int],
                                _ theirs: [String: Int]) -> CGFloat {
        var shared = 0
        for (key, count) in mine {
            shared += min(count, theirs[key] ?? 0)
        }
        let here = mine.values.reduce(0, +)
        let there = theirs.values.reduce(0, +)
        var result: CGFloat = 0
        if shared > 0 {
            let precision = CGFloat(shared) / CGFloat(here)
            let recall = CGFloat(shared) / CGFloat(there)
            result = 2 * precision * recall / (precision + recall)
        }
        return result
    }

    // Truncation AND drift, or folding on its own.
    // see okf/findings/pairing-and-arbitrating-two-tables.md

    private func trusted(_ told: [[Field]], _ drawn: [[Field]],
                         _ words: [String: Int]) -> Bool {
        let shrunk = drawn.count > 0
            && CGFloat(told.count) / CGFloat(drawn.count) < rowShare
        // Folded rows leave coverage at 1.000; only density sees them.
        let packed = Converter.density(told)
            > packing * Converter.density(drawn)
        var mine: [String: Int] = [:]
        for word in Converter.tokens(of: told)
        where word.contains(where: { character in character.isNumber }) {
            mine[word, default: 0] += 1
        }
        var shared = 0
        for (word, count) in mine {
            shared += min(count, words[word] ?? 0)
        }
        let total = mine.values.reduce(0, +)
        let covered = total > 0 ? CGFloat(shared) / CGFloat(total) : 1
        return !(packed || (shrunk && covered < numericShare))
    }

    private static func density(_ rows: [[Field]]) -> CGFloat {
        var cells = 0
        var words = 0
        for row in rows {
            for cell in row {
                let count = Converter.tokens(cell.text).count
                if count > 0 {
                    cells += 1
                    words += count
                }
            }
        }
        return cells > 0 ? CGFloat(words) / CGFloat(cells) : 1
    }

    private static func tokens(of rows: [[Field]]) -> [String] {
        rows.flatMap { row in
            row.flatMap { cell in Converter.tokens(cell.text) }
        }
    }

    private static func paragraphs(of text: String) -> [Element] {
        var result: [Element] = []
        var lines: [String] = []
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                if !lines.isEmpty {
                    result.append(.paragraph(lines.joined(separator: " ")))
                    lines = []
                }
            } else {
                lines.append(trimmed)
            }
        }
        if !lines.isEmpty {
            result.append(.paragraph(lines.joined(separator: " ")))
        }
        return result
    }

    // `smooth` belongs to the READER, like the scale does. Ink geometry
    // measures corridors on this raster and its thresholds were set on a
    // smoothed one; the model wants the lighter page.
    // see okf/findings/raster-must-not-be-resampled.md

    private static func raster(_ page: PDFPage, _ scale: CGFloat,
                               _ number: Int,
                               _ smooth: Bool) throws -> CGImage {
        let bounds = page.bounds(for: .mediaBox)
        let width = Int((bounds.width * scale).rounded())
        let height = Int((bounds.height * scale).rounded())
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        if let context {
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: CGFloat(width),
                                height: CGFloat(height)))
            // see okf/findings/raster-must-not-be-resampled.md
            context.setShouldSmoothFonts(smooth)
            context.setAllowsFontSmoothing(smooth)
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            page.draw(with: .mediaBox, to: context)
        }
        return try Converter.require(context?.makeImage(),
                                     .rasterFailed(number))
    }

    private static func recognize(_ image: CGImage) async throws
        -> [Fragment] {
        var request = RecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        let observations = try await request.perform(on: image)
        let spans = observations.flatMap { observation -> [Fragment] in
            var result: [Fragment] = []
            if let candidate = observation.topCandidates(1).first {
                result = Converter.words(of: candidate)
            }
            return result
        }
        return spans.enumerated().map { position, span in
            Fragment(index: position, text: span.text, frame: span.frame)
        }
    }

    // The spans come back too: `hybrid` judges the other reader on them.

    private func rebuilt(_ page: PDFPage, _ number: Int) async throws
        -> ([Element], [Fragment]) {
        let image = try Converter.raster(page, scale, number, true)
        let written = mode != .vision ? Converter.written(of: page) : []
        let share = written.isEmpty ? 0
            : Converter.explained(written, image, glyphLevel)
        let enough = !written.isEmpty && share >= inkShare
        if mode != .vision {
            report(String(format: "page %d: text layer %d words, "
                          + "covers %.3f of the ink%@", number,
                          written.count, Double(share),
                          enough ? "" : ", recognizing instead"))
        }
        // The thresholds belong to the SOURCE of the boxes, not to the
        // mode asked for. see okf/findings/a-gap-is-not-a-column-edge.md
        var reader = self
        var fragments = written
        reader.mode = .geometry
        if !enough {
            reader.mode = .vision
            fragments = try await Converter.recognize(image)
        }
        return (reader.layout(fragments, image, number), fragments)
    }

    // `page.selection(for:)` and never `characterBounds(at:)`. see
    // okf/platform/pdfkit-synthesizes-characters-that-own-no-glyph.md

    private static func written(of page: PDFPage) -> [Fragment] {
        let bounds = page.bounds(for: .mediaBox)
        let units = Array((page.string ?? "").utf16)
        var result: [Fragment] = []
        var from = -1
        for index in 0...units.count {
            let blank = index == units.count
                || Converter.blank(units[index])
            if blank {
                let word = from < 0 ? nil
                    : Converter.word(page, units, from..<index, bounds,
                                     result.count)
                if let word { result.append(word) }
                from = -1
            } else if from < 0 {
                from = index
            }
        }
        return result
    }

    private static func blank(_ unit: UInt16) -> Bool {
        Unicode.Scalar(unit).map { scalar in
            CharacterSet.whitespacesAndNewlines.contains(scalar)
        } ?? false
    }

    private static func word(_ page: PDFPage, _ units: [UInt16],
                             _ range: Range<Int>, _ bounds: CGRect,
                             _ index: Int) -> Fragment? {
        let selection = page.selection(
            for: NSRange(location: range.lowerBound, length: range.count))
        let box = selection?.bounds(for: page) ?? .null
        var result: Fragment? = nil
        if !box.isNull && !box.isEmpty {
            result = Fragment(
                index: index,
                text: String(decoding: units[range], as: UTF16.self),
                frame: Converter.normalized(box, bounds))
        }
        return result
    }

    private static func normalized(_ box: CGRect,
                                   _ bounds: CGRect) -> CGRect {
        CGRect(x: (box.minX - bounds.minX) / bounds.width,
               y: (box.minY - bounds.minY) / bounds.height,
               width: box.width / bounds.width,
               height: box.height / bounds.height)
    }

    // Through a mask, so the cost is one fill and not pixels times
    // words. see okf/findings/when-to-believe-a-pdf-text-layer.md

    private static func explained(_ fragments: [Fragment],
                                  _ image: CGImage,
                                  _ ink: UInt8) -> CGFloat {
        var result: CGFloat = 0
        let cover = Converter.mask(fragments, image)
        let drawn = image.dataProvider?.data
        let inside = cover?.dataProvider?.data
        if let cover, let drawn, let inside, image.bitsPerPixel == 8,
           cover.width == image.width, cover.height == image.height {
            let bytes = CFDataGetBytePtr(drawn)
            let over = CFDataGetBytePtr(inside)
            if let bytes, let over {
                result = withExtendedLifetime((drawn, inside)) {
                    Converter.covered(bytes, image, over, cover, ink)
                }
            }
        }
        return result
    }

    // No ink at all gives 1: nothing is left for a word to explain.

    private static func covered(_ bytes: UnsafePointer<UInt8>,
                                _ image: CGImage,
                                _ over: UnsafePointer<UInt8>,
                                _ cover: CGImage,
                                _ ink: UInt8) -> CGFloat {
        var inside = 0
        var total = 0
        for row in 0..<image.height {
            for column in 0..<image.width
            where bytes[row * image.bytesPerRow + column] < ink {
                total += 1
                if over[row * cover.bytesPerRow + column] < 128 {
                    inside += 1
                }
            }
        }
        return total > 0 ? CGFloat(inside) / CGFloat(total) : 1
    }

    private static func mask(_ fragments: [Fragment],
                             _ image: CGImage) -> CGImage? {
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        let context = CGContext(
            data: nil, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        if let context {
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.setFillColor(gray: 0, alpha: 1)
            for fragment in fragments {
                context.fill(CGRect(x: fragment.frame.minX * width,
                                    y: fragment.frame.minY * height,
                                    width: fragment.frame.width * width,
                                    height: fragment.frame.height * height))
            }
        }
        return context?.makeImage()
    }

    // A line crosses every column it touches, so it is broken back into
    // word boxes. see okf/findings/a-gap-is-not-a-column-edge.md

    private static func words(of text: RecognizedText) -> [Fragment] {
        let string = text.string
        var result: [Fragment] = []
        for word in string.split(whereSeparator: { character in
            character.isWhitespace
        }) {
            let range = word.startIndex..<word.endIndex
            if let box = text.boundingBox(for: range) {
                result.append(Fragment(index: 0, text: String(word),
                                       frame: box.boundingBox.cgRect))
            }
        }
        return result
    }

    // One text column at a time, and no table straddles two.

    private func layout(_ fragments: [Fragment], _ image: CGImage,
                        _ number: Int) -> [Element] {
        let level = Converter.scan(image, inkLevel, rulingWidth,
                                   level: true)
        let uprights = Converter.scan(image, inkLevel, rulingHeight,
                                      level: false)
        let sides = self.sides(of: fragments)
        report("page \(number): \(fragments.count) words, "
               + "\(level.count) rules, \(uprights.count) uprights, "
               + "\(sides.count) page columns")
        var result: [Element] = []
        for side in sides {
            let inside = fragments.filter { fragment in
                side.contains(fragment.frame.midX)
            }
            result += self.elements(inside, level, uprights,
                                    side, image)
        }
        let audit = Converter.audit(fragments, result)
        onAudit?(number, audit)
        report("    audit: \(audit.spans) spans, \(audit.lost) lost, "
               + "\(audit.unaccounted) unaccounted")
        return result
    }

    // see okf/findings/a-gap-is-not-a-column-edge.md

    private func sides(of fragments: [Fragment])
        -> [ClosedRange<CGFloat>] {
        let spans = columns(of: fragments, pageGap)
        let least = Int(CGFloat(fragments.count) * pageShare)
        var cut = CGFloat.infinity
        for (left, right) in zip(spans, spans.dropFirst()) {
            let middle = (left.upperBound + right.lowerBound) / 2
            let beyond = fragments.filter { fragment in
                fragment.frame.midX > middle
            }.count
            if pageMiddle.contains(middle) && beyond >= least
                && fragments.count - beyond >= least {
                cut = middle
            }
        }
        return Converter.joined(spans, cut)
    }

    private static func joined(_ spans: [ClosedRange<CGFloat>],
                               _ cut: CGFloat)
        -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        for span in spans {
            let last = result.last
            let together = last.map { last in
                (last.upperBound < cut) == (span.lowerBound < cut)
            } ?? false
            if let last, together {
                result[result.count - 1] = last.lowerBound...span.upperBound
            } else {
                result.append(span)
            }
        }
        return result
    }

    // see okf/constraints/every-span-reaches-the-output-once.md

    private static func audit(_ fragments: [Fragment],
                              _ elements: [Element]) -> Audit {
        var want: [String: Int] = [:]
        for fragment in fragments {
            want[fragment.text, default: 0] += 1
        }
        var have: [String: Int] = [:]
        for element in elements {
            for word in Converter.spoken(element)
                .split(whereSeparator: { character in
                    character.isWhitespace
                }) {
                have[String(word), default: 0] += 1
            }
        }
        var lost = 0
        var extra = 0
        for (word, count) in want {
            lost += max(count - (have[word] ?? 0), 0)
        }
        for (word, count) in have {
            extra += max(count - (want[word] ?? 0), 0)
        }
        return Audit(spans: fragments.count, lost: lost,
                     unaccounted: extra)
    }

    private func elements(_ fragments: [Fragment], _ level: [Ruling],
                          _ uprights: [Ruling],
                          _ side: ClosedRange<CGFloat>,
                          _ image: CGImage) -> [Element] {
        var placed = Set<Int>()
        var found: [(CGFloat, Element)] = []
        for region in self.regions(level, fragments, side) {
            let inside = within(fragments, region, side)
            report(String(format: "    table %.3f...%.3f",
                          Double(region.lowerBound),
                          Double(region.upperBound)))
            found.append((region.upperBound,
                          .table(cells(of: region, fragments, level,
                                       uprights, side, image))))
            for fragment in inside { placed.insert(fragment.index) }
        }
        let rest = fragments.filter { fragment in
            !placed.contains(fragment.index)
        }
        return (found + prose(rest)).sorted { left, right in
            left.0 > right.0
        }.map { pair in pair.1 }
    }

    // The guard tests the STRIP just added, never the whole region, and
    // `lone` admits a spanning header at the FIRST window only.
    // see okf/findings/where-a-ruled-table-starts-and-stops.md

    private func regions(_ level: [Ruling], _ fragments: [Fragment],
                         _ side: ClosedRange<CGFloat>)
        -> [ClosedRange<CGFloat>] {
        let mine = level.filter { rule in
            Converter.agrees(rule.span, side, spanAgreement)
        }.sorted { left, right in left.position > right.position }
        let lines = Converter.merged(mine, rulingMerge)
        var result: [ClosedRange<CGFloat>] = []
        var top = CGFloat.infinity
        var bottom = CGFloat.infinity
        for line in lines {
            let window = within(fragments, line...bottom, side)
            let holds = top < .infinity
                && columns(of: window, wordGap).count >= 2
            let heights = window.map { fragment in fragment.frame.height }
            let lone = top < .infinity && bottom == top
                && !window.isEmpty && bands(window).count == 1
                && top - line <= Converter.median(heights) * spanWindow
            if holds || lone {
                bottom = line
            } else {
                if top < .infinity && bottom < top {
                    result.append(bottom...top)
                }
                top = line
                bottom = line
            }
        }
        if top < .infinity && bottom < top { result.append(bottom...top) }
        return result
    }

    private func within(_ fragments: [Fragment],
                        _ region: ClosedRange<CGFloat>,
                        _ side: ClosedRange<CGFloat>) -> [Fragment] {
        fragments.filter { fragment in
            region.contains(fragment.frame.midY)
                && side.contains(fragment.frame.midX)
        }
    }

    // `ruled`: one rule per row. Otherwise the text lines are the rows,
    // which is the three-rule book-style table.

    private func cells(of region: ClosedRange<CGFloat>,
                       _ fragments: [Fragment], _ level: [Ruling],
                       _ uprights: [Ruling],
                       _ side: ClosedRange<CGFloat>,
                       _ image: CGImage) -> Table {
        let inside = within(fragments, region, side)
        let lines = bands(inside).map { band in Converter.extent(band) }
        let width = Converter.spread(inside)
        let inner = Converter.merged(level.filter { rule in
            rule.position > region.lowerBound + rulingMerge
                && rule.position < region.upperBound - rulingMerge
                && Converter.covers(rule.span, width, fullWidth)
        }.sorted { left, right in left.position > right.position },
                                     rulingMerge)
        let ruled = inner.count + 1 >= lines.count && lines.count > 1
        let rows = ruled ? Converter.slice(region, inner)
            : Converter.separated(lines, region)
        let corridors = cellGap > 0
            ? self.corridors(region, width, image)
            : columns(of: inside, wordGap)
        let posts = Converter.merged(uprights.filter { post in
            Converter.agrees(post.span, region, spanAgreement)
                && side.contains(post.position)
        }.sorted { left, right in left.position > right.position },
                                     rulingMerge).reversed()
        let bars = Converter.slices(width,
                                    Converter.cuts(Array(posts),
                                                   corridors, wordGap))
        let header = inner.isEmpty ? 1
            : max(rows.filter { row in
                row.lowerBound >= inner[0] - rulingMerge
            }.count, 1)
        report(String(format: "      rows %d (ruled %@) inner %d "
                      + "columns %d header %d", rows.count,
                      ruled ? "yes" : "no", inner.count, bars.count,
                      header))
        return Table(rows: Converter.laid(
            Grid(rows: rows, columns: bars, header: header), inside),
                     frame: Converter.box(width, region))
    }

    // see okf/decisions/turning-corridors-into-cells.md

    private static func settled(_ columns: [ClosedRange<CGFloat>],
                                _ fragments: [Fragment])
        -> [ClosedRange<CGFloat>] {
        var lower = columns.map { column in column.upperBound }
        var upper = columns.map { column in column.lowerBound }
        var sure = [Bool](repeating: false, count: columns.count)
        for fragment in fragments {
            let held = columns.indices.filter { index in
                columns[index].lowerBound <= fragment.frame.midX
                    && fragment.frame.midX <= columns[index].upperBound
                    && columns[index].lowerBound <= fragment.frame.minX
                    && fragment.frame.maxX <= columns[index].upperBound
            }
            if held.count == 1 {
                let index = held[0]
                lower[index] = min(lower[index], fragment.frame.minX)
                upper[index] = max(upper[index], fragment.frame.maxX)
                sure[index] = true
            }
        }
        return columns.indices.map { index in
            sure[index] ? lower[index]...upper[index] : columns[index]
        }
    }

    // Ordered by x ALONE: a midY sort never reaches the x tiebreak.

    private static func laid(_ grid: Grid,
                             _ fragments: [Fragment]) -> [[Field]] {
        let bars = Converter.settled(grid.columns, fragments)
        var result: [[Field]] = []
        for row in grid.rows {
            var held = [[Fragment]](repeating: [], count: bars.count)
            let line = fragments.filter { fragment in
                row.contains(fragment.frame.midY)
            }.sorted { left, right in
                left.frame.minX < right.frame.minX
            }
            for fragment in line {
                held[Converter.column(of: fragment, bars)].append(fragment)
            }
            let built = held.map { inside in Converter.field(inside) }
            if built.contains(where: { cell in !cell.text.isEmpty }) {
                result.append(built)
            }
        }
        return Converter.folded(Converter.occupied(result), grid.header)
    }

    private static func field(_ held: [Fragment]) -> Field {
        Field(text: held.map { one in one.text }.joined(separator: " "),
              spans: held.map { one in one.index })
    }

    private static func empty() -> Field {
        Field(text: "", spans: [])
    }

    // Rules and corridors both propose more edges than a table has.

    private static func occupied(_ rows: [[Field]]) -> [[Field]] {
        let width = rows.reduce(0) { widest, row in
            max(widest, row.count)
        }
        var used: [Bool] = Array(repeating: false, count: width)
        for row in rows {
            for (index, cell) in row.enumerated() where !cell.text.isEmpty {
                used[index] = true
            }
        }
        var result: [[Field]] = []
        for row in rows {
            var kept: [Field] = []
            for index in 0..<width where used[index] {
                kept.append(index < row.count ? row[index]
                            : Converter.empty())
            }
            result.append(kept)
        }
        return result
    }

    // Markdown has one header row and no spanning cells.

    private static func folded(_ rows: [[Field]],
                               _ header: Int) -> [[Field]] {
        var result = rows
        if header > 1 && rows.count > header {
            var merged = [Field](repeating: Converter.empty(),
                                 count: rows[0].count)
            for level in rows.prefix(header) {
                for (index, cell) in level.enumerated()
                    where !cell.text.isEmpty && index < merged.count {
                    let before = merged[index]
                    merged[index] = Field(
                        text: before.text.isEmpty ? cell.text
                            : before.text + " " + cell.text,
                        spans: before.spans + cell.spans)
                }
            }
            result = [merged] + rows.dropFirst(header)
        }
        return result
    }

    // Text the rules did not claim; some of it is still a table.
    // see okf/findings/an-unruled-table-is-told-apart-by-its-cells.md

    private func prose(_ fragments: [Fragment])
        -> [(CGFloat, Element)] {
        let bands = self.bands(fragments)
        let gap = Converter.medianGap(bands) * rowGapRatio
        var result: [(CGFloat, Element)] = []
        var index = 0
        while index < bands.count {
            let top = Converter.extent(bands[index]).upperBound
            let shared = aligned(bands, index)
            let candidate = shared - index >= alignedRows - 1
                ? speculative(Array(bands[index...shared]))
                : Table(rows: [], frame: nil)
            if Converter.tabular(candidate.rows, proseCellShare,
                                 proseCellLength) {
                result.append((top, .table(candidate)))
                index = shared + 1
            } else {
                var stop = index
                var next = index + 1
                while next < bands.count
                      && Converter.distance(bands[next - 1],
                                            bands[next]) <= gap {
                    stop = next
                    next += 1
                }
                let text = bands[index...stop].map { band in
                    Converter.text(of: band)
                }
                result.append((top,
                               .paragraph(text.joined(separator: " "))))
                index = stop + 1
            }
        }
        return result
    }

    private func speculative(_ run: [[Fragment]]) -> Table {
        let words = run.flatMap { band in band }
        let rows = run.map { band in Converter.extent(band) }
        let floor = rows[rows.count - 1].lowerBound
        let region = floor...rows[0].upperBound
        let width = Converter.spread(words)
        return Table(rows: Converter.laid(
            Grid(rows: Converter.separated(rows, region),
                 columns: columns(of: words, wordGap),
                 header: 1), words),
                     frame: Converter.box(width, region))
    }

    private static func box(_ width: ClosedRange<CGFloat>,
                            _ region: ClosedRange<CGFloat>) -> CGRect {
        CGRect(x: width.lowerBound, y: region.lowerBound,
               width: width.upperBound - width.lowerBound,
               height: region.upperBound - region.lowerBound)
    }

    // see okf/findings/an-unruled-table-is-told-apart-by-its-cells.md

    private static func tabular(_ rows: [[Field]], _ share: CGFloat,
                                _ length: Int) -> Bool {
        var filled = 0
        var wordy = 0
        for row in rows {
            for cell in row where !cell.text.isEmpty {
                filled += 1
                if cell.text.count > length { wordy += 1 }
            }
        }
        return filled > 0 && CGFloat(wordy) <= CGFloat(filled) * share
    }

    private func aligned(_ bands: [[Fragment]], _ start: Int) -> Int {
        var end = start
        var words = bands[start]
        var index = start + 1
        while index < bands.count
              && columns(of: words + bands[index], wordGap).count >= 2 {
            words += bands[index]
            end = index
            index += 1
        }
        return end
    }

    private func bands(_ fragments: [Fragment]) -> [[Fragment]] {
        let sorted = fragments.sorted { left, right in
            left.frame.midY > right.frame.midY
        }
        let heights = sorted.map { fragment in fragment.frame.height }
        let tolerance = Converter.median(heights) * bandTolerance
        var result: [[Fragment]] = []
        for fragment in sorted {
            let anchor = result.last?.first
            let near = abs((anchor?.frame.midY ?? -1)
                           - fragment.frame.midY) <= tolerance
            if anchor != nil && near {
                result[result.count - 1].append(fragment)
            } else {
                result.append([fragment])
            }
        }
        return result.map { band in
            band.sorted { left, right in
                left.frame.minX < right.frame.minX
            }
        }
    }

    // see okf/findings/a-gap-is-not-a-column-edge.md

    private func columns(of fragments: [Fragment], _ gap: CGFloat,
                         _ vote: CGFloat = 0) -> [ClosedRange<CGFloat>] {
        let lines = vote > 0 ? bands(fragments).map { band in band }
            : [fragments]
        var result: [ClosedRange<CGFloat>] = []
        for span in Converter.profile(lines, buckets, vote) {
            let last = result.last
            let joined = span.lowerBound - (last?.upperBound ?? -1) < gap
            if let last, joined {
                let upper = max(last.upperBound, span.upperBound)
                result[result.count - 1] = last.lowerBound...upper
            } else {
                result.append(span)
            }
        }
        return result
    }

    // Crossing ROWS are counted, not unioned.
    // see okf/findings/one-spanning-header-erases-every-corridor.md

    private static func profile(_ lines: [[Fragment]], _ buckets: Int,
                                _ vote: CGFloat)
        -> [ClosedRange<CGFloat>] {
        var crossings = [Int](repeating: 0, count: buckets)
        let last = buckets - 1
        for line in lines {
            var touched = [Bool](repeating: false, count: buckets)
            for fragment in line {
                let from = Converter.bucket(fragment.frame.minX, last)
                let upto = Converter.bucket(fragment.frame.maxX, last)
                for index in min(from, upto)...max(from, upto) {
                    touched[index] = true
                }
            }
            for index in 0..<buckets where touched[index] {
                crossings[index] += 1
            }
        }
        let wall = Int((CGFloat(lines.count) * vote).rounded(.down))
        var result: [ClosedRange<CGFloat>] = []
        var start = -1
        for index in 0...buckets {
            let ink = index < buckets && crossings[index] > wall
            if ink && start < 0 { start = index }
            if !ink && start >= 0 {
                let lower = CGFloat(start) / CGFloat(buckets)
                let upper = CGFloat(index) / CGFloat(buckets)
                result.append(lower...upper)
                start = -1
            }
        }
        return result
    }

    private static func bucket(_ x: CGFloat, _ last: Int) -> Int {
        min(max(Int(x * CGFloat(last + 1)), 0), last)
    }

    // A cell spanning two columns has its midpoint over the gap.

    private static func column(of fragment: Fragment,
                               _ columns: [ClosedRange<CGFloat>]) -> Int {
        var best = 0
        var strongest = -CGFloat.infinity
        for (index, column) in columns.enumerated() {
            let shared = min(column.upperBound, fragment.frame.maxX)
                - max(column.lowerBound, fragment.frame.minX)
            let claim = shared > 0 ? shared
                : -min(abs(column.lowerBound - fragment.frame.midX),
                       abs(column.upperBound - fragment.frame.midX))
            if claim > strongest {
                strongest = claim
                best = index
            }
        }
        return best
    }

    private static func extent(_ band: [Fragment]) -> ClosedRange<CGFloat> {
        let lower = band.map { fragment in fragment.frame.minY }.min() ?? 0
        let upper = band.map { fragment in fragment.frame.maxY }.max() ?? 0
        return lower...max(upper, lower)
    }

    // Descending cuts, top piece first.

    private static func slice(_ whole: ClosedRange<CGFloat>,
                              _ cuts: [CGFloat]) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var upper = whole.upperBound
        for cut in cuts where cut > whole.lowerBound && cut < upper {
            result.append(cut...upper)
            upper = cut
        }
        result.append(whole.lowerBound...upper)
        return result
    }

    // Unioned, not chosen between: a table often supplies one where it
    // lacks the other.

    private static func cuts(_ posts: [CGFloat],
                             _ corridors: [ClosedRange<CGFloat>],
                             _ apart: CGFloat) -> [CGFloat] {
        var found = posts
        for (left, right) in zip(corridors, corridors.dropFirst()) {
            found.append((left.upperBound + right.lowerBound) / 2)
        }
        var result: [CGFloat] = []
        for cut in found.sorted() {
            if cut - (result.last ?? -1) > apart { result.append(cut) }
        }
        return result
    }

    // Descenders make band extents overlap; cut halfway between.

    private static func separated(_ lines: [ClosedRange<CGFloat>],
                                  _ region: ClosedRange<CGFloat>)
        -> [ClosedRange<CGFloat>] {
        var cuts: [CGFloat] = []
        for (upper, lower) in zip(lines, lines.dropFirst()) {
            cuts.append((upper.lowerBound + lower.upperBound) / 2)
        }
        return Converter.slice(region, cuts)
    }

    // Ascending cuts, left piece first.

    private static func slices(_ whole: ClosedRange<CGFloat>,
                               _ cuts: [CGFloat]) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var lower = whole.lowerBound
        for cut in cuts where cut > lower && cut < whole.upperBound {
            result.append(lower...cut)
            lower = cut
        }
        result.append(lower...whole.upperBound)
        return result
    }

    private static func spread(_ fragments: [Fragment])
        -> ClosedRange<CGFloat> {
        let lower = fragments.map { f in f.frame.minX }.min() ?? 0
        let upper = fragments.map { f in f.frame.maxX }.max() ?? 1
        return lower...max(upper, lower)
    }

    private static func distance(_ upper: [Fragment],
                                 _ lower: [Fragment]) -> CGFloat {
        let bottom = upper.map { f in f.frame.minY }.min() ?? 0
        let top = lower.map { f in f.frame.maxY }.max() ?? 0
        return bottom - top
    }

    private static func medianGap(_ bands: [[Fragment]]) -> CGFloat {
        var gaps: [CGFloat] = []
        for (upper, lower) in zip(bands, bands.dropFirst()) {
            gaps.append(Converter.distance(upper, lower))
        }
        return max(Converter.median(gaps), 0)
    }

    private static func median(_ values: [CGFloat]) -> CGFloat {
        let sorted = values.sorted()
        var result: CGFloat = 0
        if !sorted.isEmpty { result = sorted[sorted.count / 2] }
        return result
    }

    private static func text(of band: [Fragment]) -> String {
        band.map { fragment in fragment.text }.joined(separator: " ")
    }

    // Off the raster, not the boxes. Pixel rows a drawn rule crosses
    // are skipped, and the rule is measured against the TABLE's width.
    // see okf/findings/one-spanning-header-erases-every-corridor.md

    private func corridors(_ region: ClosedRange<CGFloat>,
                           _ width: ClosedRange<CGFloat>,
                           _ image: CGImage) -> [ClosedRange<CGFloat>] {
        var crossings = [Int](repeating: 0, count: buckets)
        var inked = 0
        let data = image.dataProvider?.data
        if let data, image.bitsPerPixel == 8 {
            let bytes = CFDataGetBytePtr(data)
            if let bytes {
                let top = Converter.row(region.upperBound, image.height)
                let floor = Converter.row(region.lowerBound, image.height)
                let least = Int(CGFloat(image.width) * fullWidth
                                * (width.upperBound - width.lowerBound))
                let from = Converter.bucket(width.lowerBound,
                                            image.width - 1)
                let upto = Converter.bucket(width.upperBound,
                                            image.width - 1)
                for row in top...floor where Converter.inked(
                    bytes, image, row, inkLevel, least, true) == nil {
                    var touched = [Bool](repeating: false, count: buckets)
                    for column in from...upto
                    where bytes[row * image.bytesPerRow + column]
                        < inkLevel {
                        touched[Converter.bucket(
                            CGFloat(column) / CGFloat(image.width),
                            buckets - 1)] = true
                    }
                    var any = false
                    for index in 0..<buckets where touched[index] {
                        crossings[index] += 1
                        any = true
                    }
                    if any { inked += 1 }
                }
            }
        }
        let wall = Int((CGFloat(inked) * cellVote).rounded(.down))
        var covered = [Bool](repeating: false, count: buckets)
        for index in 0..<buckets {
            covered[index] = crossings[index] > wall
        }
        return Converter.runs(covered, cellGap)
    }

    private static func row(_ y: CGFloat, _ height: Int) -> Int {
        min(max(Int((1 - y) * CGFloat(height)), 0), height - 1)
    }

    private static func runs(_ covered: [Bool],
                             _ gap: CGFloat) -> [ClosedRange<CGFloat>] {
        var result: [ClosedRange<CGFloat>] = []
        var start = -1
        for index in 0...covered.count {
            let ink = index < covered.count && covered[index]
            if ink && start < 0 { start = index }
            if !ink && start >= 0 {
                let lower = CGFloat(start) / CGFloat(covered.count)
                let upper = CGFloat(index) / CGFloat(covered.count)
                let last = result.last
                if let last, lower - last.upperBound < gap {
                    result[result.count - 1] = last.lowerBound...upper
                } else {
                    result.append(lower...upper)
                }
                start = -1
            }
        }
        return result
    }

    private static func scan(_ image: CGImage, _ ink: UInt8,
                             _ share: CGFloat, level: Bool) -> [Ruling] {
        var result: [Ruling] = []
        let data = image.dataProvider?.data
        if let data, image.bitsPerPixel == 8 {
            let bytes = CFDataGetBytePtr(data)
            if let bytes {
                let across = level ? image.width : image.height
                let along = level ? image.height : image.width
                let least = Int(CGFloat(across) * share)
                var start = -1
                var span = 0...0
                for index in 0...along {
                    let line = index < along
                        ? Converter.inked(bytes, image, index, ink, least,
                                          level)
                        : nil
                    let extends = line.map { found in
                        start >= 0 && Converter.overlaps(found, span)
                    } ?? false
                    if extends, let line {
                        let lower = min(span.lowerBound, line.lowerBound)
                        let upper = max(span.upperBound, line.upperBound)
                        span = lower...upper
                    } else {
                        if start >= 0 {
                            result.append(Converter.ruling(start, span,
                                                           image, level))
                            result.append(Converter.ruling(index - 1, span,
                                                           image, level))
                        }
                        start = line == nil ? -1 : index
                        span = line ?? 0...0
                    }
                }
            }
        }
        return result
    }

    // see okf/findings/a-rule-is-a-contiguous-run-of-ink.md

    private static func inked(_ bytes: UnsafePointer<UInt8>,
                              _ image: CGImage, _ index: Int,
                              _ ink: UInt8, _ least: Int,
                              _ level: Bool) -> ClosedRange<Int>? {
        let across = level ? image.width : image.height
        var start = -1
        var last = -1
        var gap = 0
        var longest = -1
        var from = 0
        var upto = -1
        for step in 0..<across {
            let row = level ? index : step
            let column = level ? step : index
            let dark = bytes[row * image.bytesPerRow + column] < ink
            if dark {
                if start < 0 { start = step }
                last = step
                gap = 0
            } else if start >= 0 {
                gap += 1
                if gap > 2 {
                    if last - start > longest {
                        longest = last - start
                        from = start
                        upto = last
                    }
                    start = -1
                }
            }
        }
        if start >= 0 && last - start > longest {
            longest = last - start
            from = start
            upto = last
        }
        var result: ClosedRange<Int>? = nil
        if longest + 1 >= least { result = from...upto }
        return result
    }

    private static func ruling(_ index: Int, _ span: ClosedRange<Int>,
                               _ image: CGImage, _ level: Bool) -> Ruling {
        let across = CGFloat(level ? image.width : image.height)
        let along = CGFloat(level ? image.height : image.width)
        let lower = CGFloat(span.lowerBound) / across
        let upper = CGFloat(span.upperBound) / across
        let offset = CGFloat(index) / along
        var position = offset
        var range = lower...upper
        if level {
            position = 1 - offset
        } else {
            range = (1 - upper)...(1 - lower)
        }
        return Ruling(position: position, span: range)
    }

    private static func merged(_ rulings: [Ruling],
                               _ tolerance: CGFloat) -> [CGFloat] {
        var result: [CGFloat] = []
        for ruling in rulings {
            if (result.last ?? .infinity) - ruling.position > tolerance {
                result.append(ruling.position)
            }
        }
        return result
    }

    // `covers` scales the shared extent by the whole, `agrees` by the
    // SHORTER of the two.

    private static func covers(_ span: ClosedRange<CGFloat>,
                               _ whole: ClosedRange<CGFloat>,
                               _ share: CGFloat) -> Bool {
        let lower = max(span.lowerBound, whole.lowerBound)
        let upper = min(span.upperBound, whole.upperBound)
        let shared = max(upper - lower, 0)
        let width = whole.upperBound - whole.lowerBound
        return width > 0 && shared >= width * share
    }

    private static func agrees(_ span: ClosedRange<CGFloat>,
                               _ other: ClosedRange<CGFloat>,
                               _ share: CGFloat) -> Bool {
        let lower = max(span.lowerBound, other.lowerBound)
        let upper = min(span.upperBound, other.upperBound)
        let shared = max(upper - lower, 0)
        let shorter = min(span.upperBound - span.lowerBound,
                          other.upperBound - other.lowerBound)
        return shorter > 0 && shared >= shorter * share
    }

    private static func overlaps<T: Comparable>(
        _ left: ClosedRange<T>, _ right: ClosedRange<T>) -> Bool {
        max(left.lowerBound, right.lowerBound)
            <= min(left.upperBound, right.upperBound)
    }

    private func report(_ message: String) {
        if trace {
            FileHandle.standardError.write(Data((message + "\n").utf8))
        }
    }

    private static func render(_ pages: [Page]) -> String {
        var blocks: [String] = []
        for page in pages {
            for element in page.elements {
                blocks.append(Converter.render(element))
            }
        }
        return blocks.joined(separator: "\n\n") + "\n"
    }

    // The words alone: Markdown framing would break the audit.

    private static func spoken(_ element: Element) -> String {
        var result = ""
        switch element {
        case .paragraph(let text):
            result = text
        case .table(let table):
            result = table.rows.map { row in
                row.map { cell in cell.text }.joined(separator: " ")
            }.joined(separator: " ")
        }
        return result
    }

    private static func render(_ element: Element) -> String {
        var result = ""
        switch element {
        case .paragraph(let text):
            result = text
        case .table(let table):
            result = Converter.table(table.rows)
        }
        return result
    }

    private static func table(_ rows: [[Field]]) -> String {
        let width = rows.reduce(0) { widest, row in
            max(widest, row.count)
        }
        var lines: [String] = []
        for (index, row) in rows.enumerated() {
            var cells = row.map { cell in
                cell.text.replacingOccurrences(of: "|", with: "\\|")
            }
            while cells.count < width { cells.append("") }
            lines.append("| " + cells.joined(separator: " | ") + " |")
            if index == 0 {
                let rule = Array(repeating: "---", count: width)
                lines.append("| " + rule.joined(separator: " | ") + " |")
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func require<T>(_ value: T?,
                                   _ failure: ConversionError) throws -> T {
        switch value {
        case .some(let unwrapped): return unwrapped
        case .none:                throw failure
        }
    }
}
