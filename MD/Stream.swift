import Foundation

// Incremental parser for live LLM output. One instance per channel (assistant
// content, reasoning, tool).
public final class MarkdownStream {

    private var sealed: [Markdown.Document.Item] = []
    private var lines: [String] = []        // reference-stripped, complete
    private var sealedLines = 0             // lines folded into `sealed`
    private var partial = ""                // trailing line, no newline yet
    private var refs: [String: URL] = [:]   // backward-resolving link defs
    private var fenceMarker: String? = nil  // non-nil while inside a fence
    private let mathEnabled: Bool

    // `math` mirrors MarkdownStyle.renderMath: pass the same value the view
    // will render with, since the stream is where blocks are parsed.
    public init(math: Bool = true) { mathEnabled = math }

    public func append(_ text: String) {
        let buf = partial + text
        partial = ""
        let segs = buf
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var added = false
        var i = 0
        while i < segs.count - 1 {
            ingest(segs[i])
            added = true
            i += 1
        }
        partial = segs.last ?? ""
        if added { advanceSeal() }
    }

    // Sealed blocks plus the open block(s), rendered in full including the
    // in-progress trailing line. Ids are stable across snapshots.
    public func snapshot() -> Markdown.Document {
        var items = sealed
        var open = Array(lines[sealedLines...])
        if !partial.isEmpty { open.append(partial) }
        if !open.isEmpty {
            let bs = parsed { Markdown.blocks(open) }
            var id = sealed.count
            for b in bs {
                items.append(Markdown.Document.Item(id: id, block: b))
                id += 1
            }
        }
        return Markdown.Document(items: items)
    }

    // Seal the last open block and return the final document. Equal to
    // Markdown.parse(raw) block-for-block for well-formed input.
    public func finish() -> Markdown.Document {
        if !partial.isEmpty {
            ingest(partial)
            partial = ""
        }
        let tail = Array(lines[sealedLines...])
        let spans = parsed { Markdown.blockSpans(tail) }
        for s in spans {
            // Item ids ARE the sealed positions (count pre-append), so no
            // separate counter can drift from them.
            sealed.append(Markdown.Document.Item(id: sealed.count,
                                                 block: s.block))
        }
        sealedLines = lines.count
        return Markdown.Document(items: sealed)
    }

    public func reset() {
        sealed = []
        lines = []
        sealedLines = 0
        partial = ""
        refs = [:]
        fenceMarker = nil
    }

    private func parsed<T>(_ body: () -> T) -> T {
        Markdown.$mathEnabled.withValue(mathEnabled) {
            Markdown.$currentRefs.withValue(refs) { body() }
        }
    }

    // Fold one complete line into the stripped stream, diverting reference
    // definitions to `refs` exactly as Markdown.parse's first pass does.
    private func ingest(_ line: String) {
        let t = line.trimmedLeading()
        if let marker = fenceMarker {
            lines.append(line)
            if t.hasPrefix(marker) { fenceMarker = nil }
        } else if t.hasPrefix("```") || t.hasPrefix("~~~") {
            fenceMarker = String(t.prefix(3))
            lines.append(line)
        } else if let def = Markdown.parseLinkDefinition(line) {
            refs[def.label] = def.url
        } else {
            lines.append(line)
        }
    }

    // A block is settled once a later block exists after it, so a lazy
    // continuation only ever touches the still-open block.
    private func advanceSeal() {
        let tail = Array(lines[sealedLines...])
        let spans = parsed { Markdown.blockSpans(tail) }
        if spans.count >= 2 {
            let lastStart = spans[spans.count - 1].start
            var k = 0
            while k < spans.count - 1 {
                sealed.append(Markdown.Document.Item(
                    id: sealed.count, block: spans[k].block))
                k += 1
            }
            sealedLines += lastStart
        }
    }
}
