import Foundation

// Incremental parser for live LLM output. One instance per channel (assistant
// content, reasoning, tool).
public final class MarkdownStream {

    private var sealed: [Markdown.Document.Item] = []
    private var lines: [String] = []        // reference-stripped, complete
    private var sealedLines = 0             // lines folded into sealed+open
    private var open: Markdown.OpenBlock? = nil
    private var openStart = 0               // first line of `open`'s block
    private var partial = ""                // trailing line, no newline yet
    private var refs: [String: URL] = [:]   // backward-resolving link defs
    private var fence: String? = nil        // non-nil while inside a fence
    private let mathEnabled: Bool

    // `math` mirrors MarkdownStyle.renderMath: pass the same value the view
    // will render with, since the stream is where blocks are parsed.
    public init(math: Bool = true) { mathEnabled = math }

    public func append(_ text: String) {
        if let newline = text.lastIndex(of: "\n") {
            let whole = partial + text[..<newline]
            partial = String(text[text.index(after: newline)...])
            for line in whole.split(separator: "\n",
                                    omittingEmptySubsequences: false) {
                ingest(String(line))
            }
            advanceSeal()
        } else {
            partial += text
        }
    }

    // Sealed blocks plus the open block(s), rendered in full including the
    // in-progress trailing line. Ids are stable across snapshots.
    public func snapshot() -> Markdown.Document {
        var items = sealed
        var tail = Array(lines[sealedLines...])
        if !partial.isEmpty { tail.append(partial) }
        let spans = parsed { Markdown.openSpans(tail, resuming: open) }.spans
        for s in spans {
            items.append(Markdown.Document.Item(id: items.count,
                                                block: s.block))
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
        let spans = parsed { Markdown.openSpans(tail, resuming: open) }.spans
        for s in spans {
            // Item ids ARE the sealed positions (count pre-append), so no
            // separate counter can drift from them.
            sealed.append(Markdown.Document.Item(id: sealed.count,
                                                 block: s.block))
        }
        sealedLines = lines.count
        openStart = lines.count
        open = nil
        return Markdown.Document(items: sealed)
    }

    public func reset() {
        sealed = []
        lines = []
        sealedLines = 0
        open = nil
        openStart = 0
        partial = ""
        refs = [:]
        fence = nil
    }

    private func parsed<T>(_ body: () -> T) -> T {
        Markdown.$mathEnabled.withValue(mathEnabled) {
            Markdown.$currentRefs.withValue(refs) { body() }
        }
    }

    // Fold one complete line into the stripped stream, diverting reference
    // definitions to `refs` exactly as Markdown.parse's first pass does.
    private func ingest(_ line: String) {
        if let open = fence {
            lines.append(line)
            if Markdown.closesFence(line, open) { fence = nil }
        } else if let run = Markdown.openingFence(line) {
            fence = run
            lines.append(line)
        } else if let def = Markdown.parseLinkDefinition(line) {
            refs[def.label] = def.url
            if open != nil { sealedLines = openStart }
            open = nil
        } else {
            lines.append(line)
        }
    }

    // A block is settled once a later block exists after it, so a lazy
    // continuation only ever touches the still-open block.
    private func advanceSeal() {
        let tail = Array(lines[sealedLines...])
        let grown = parsed { Markdown.openSpans(tail, resuming: open) }
        let spans = grown.spans
        let lastStart = spans.last?.start ?? 0
        let resumed = open == nil ? 0 : 1
        var k = 0
        while k < spans.count - 1,
              k < resumed || spans[k].start < lastStart {
            sealed.append(Markdown.Document.Item(
                id: sealed.count, block: spans[k].block))
            k += 1
        }
        for rewrite in grown.rewrites where rewrite.by < lastStart {
            lines[sealedLines + rewrite.line] = rewrite.text
        }
        if let last = spans.last {
            if open == nil || spans.count >= 2 {
                openStart = sealedLines + last.start
            }
            open = grown.open
            sealedLines = open == nil ? openStart : sealedLines + grown.cut
        }
    }
}
