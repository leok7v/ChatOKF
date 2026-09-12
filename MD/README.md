# Markdown

A streaming-first, cross-platform Markdown package for Swift. It parses
Markdown into a block model, renders it in SwiftUI, and exports it to
HTML, PDF, and plain text.

It is built to render live LLM output. You append tokens as they arrive
(up to ~30 tokens/second) and only the last, still-growing block
re-lays-out on screen. Everything above the last block is already
parsed, measured, and left untouched.

The App (ChatOKF, `../App`) is the first consumer. The package is
feature-complete relative to the `md.too` viewer, but the
App is free to switch on only the subset it needs.


## Design goals

- Streaming-first. `append(token)` costs work proportional to the size of
  the currently-open block, never the whole document. A 5000-line chat
  transcript does not re-parse or re-measure when the next token lands.
- Feature-complete. Headings, paragraphs, fenced and indented code,
  blockquotes, ordered / unordered / task lists (nested), GFM tables,
  rules, images, reference-style links, inline emphasis, `<u>`, and
  TeX-style math.
- One block AST, many renderers. SwiftUI views, a single selectable
  NSAttributedString surface, PDF, HTML, and plain text all consume the
  same `[Block]`.
- Cross-platform with no `#if os(...)` in the logic. Platform code lives
  in SDK-split files (`*-iOS.swift` / `*-macOS.swift`), matching the App.
- No third-party dependencies. Foundation, CoreText, CoreGraphics,
  SwiftUI, and AppKit / UIKit only.


## Adding it to a target

`MD/` is a local SwiftPM package. A target depends on it the usual
way (SwiftPM `dependencies` / `.product(name: "Markdown", ...)`, or the
App's `project.yml` package list). Platforms: macOS 15, iOS 18, Swift 6,
matching the `LLM` package.

```swift
import MD
```


## Quick start: static document

Parse once, render the blocks.

```swift
let document = Markdown.parse(source)   // source: String
MarkdownView(document)        // SwiftUI
```

`MarkdownView` is the same view used for streaming. It is identity-driven:
each block carries a stable id, so re-rendering with a changed document
only touches the blocks that actually changed.


## Quick start: streaming from an LLM

Hold one `MarkdownStream` per assistant turn. Feed it token pieces as they
arrive. Read a snapshot at your own display cadence and hand it to the
same `MarkdownView`.

```swift
let stream = MarkdownStream()

// as each piece arrives from the model:
stream.append(piece)

// at your render cadence (for example a ~10 Hz tick while streaming):
let document = stream.snapshot()        // Markdown.Document

// when the turn ends:
let final = stream.finish()             // seals the last open block
```

`snapshot()` returns the sealed blocks plus the one open (still-growing)
block, rendered in full. Incomplete inline markup (`**bo` before `ld**`
arrives) shows as literal text and heals the moment it closes. An
unterminated code fence stays a code block until `finish()`.

The App drives cadence. There is no timer inside the package. Snapshotting
per token is fine, but coalescing to ~10 Hz while the reply streams keeps
SwiftUI relaxed (this is what `im.ai` did with its render tick).


## Consuming it in ChatOKF

Today `App/ContentView.swift` renders an assistant bubble with
`Text(answer)`. The streaming path lives in `App/ChatModel.swift`
(`sendText` / `sendVision` append to `messages[idx].text`). The change is:

- Give each streaming `Message` a `MarkdownStream`; `append` each `piece`
  in the `for await` loop instead of only string-concatenating.
- Render the bubble with `MarkdownView(stream.snapshot())`.
- Reasoning (`<think>`) can stay plain `Text`, or render through the same
  view with a different theme. The App decides per surface.

Nothing forces the App to adopt tables, math, or images on day one. Those
blocks render if present; the App can also strip or ignore block kinds it
does not want to lay out yet.


## Public API surface

Model (pure, no SwiftUI):

- `enum Block` - `heading`, `paragraph`, `code`, `quote`, `list`, `table`,
  `rule`, `image`. Same shape as `md.too`'s `Block`.
- `struct ListItem` - marker, optional `checked`, nested `[Block]`.
- `struct Markdown.Document` - an ordered list of identified blocks
  (`id` + `Block`). The `id` is stable across snapshots for a given
  streamed turn.
- `enum Markdown` - `static func parse(_ source: String) -> Document`. Full
  two-pass parse (reference definitions resolved in both directions).
- `final class MarkdownStream` - `append(_:)`, `snapshot() -> Document`,
  `finish() -> Document`, `reset()`.

Views (SwiftUI):

- `MarkdownView(_ document:)` - the block renderer, used for both static and
  streaming. Identity-driven so only changed blocks re-render.
- Rendering options via a `MarkdownStyle` / theme value: fonts, colors,
  spacing, code highlighting on/off, math on/off, selectable on/off,
  single-surface vs per-block, light / dark / system.
- A single-surface mode that renders the whole document as one selectable
  text view (drag-select across the document, selection snaps around
  atomic code / table / image units).

Export (from a `Document` or `[Block]`):

- `Markdown.html(_:)` - self-contained HTML, images inlined as base64
  data URIs.
- `Markdown.pdf(_:)` - paginated PDF via CoreText (A4 / Letter by region,
  headers, footers, page numbers).
- `Markdown.plain(_:)` - round-trips back to Markdown-ish plain text.

Images are fetched asynchronously and concurrently; the view shows a
placeholder until a picture resolves, and export prefetches before
rendering.


## Feature support

| Area        | Supported                                                  |
|-------------|------------------------------------------------------------|
| Blocks      | headings, paragraphs, fenced + indented code, blockquotes  |
|             | (nested), lists (ordered / unordered / task, nested,       |
|             | tight + loose), GFM tables, rules, images                  |
| Inline      | bold, italic, strikethrough, inline code, links, `<u>`     |
| Links       | inline and reference-style (with link definitions)         |
| Math        | `$inline$` and `$$display$$`, Unicode substitution         |
| Code        | syntax highlighting driven by a bundled `highlights.ini`,  |
|             | adaptive light / dark                                      |
| Images      | remote fetch, explicit `{width= height=}`, aspect fit      |
| Render      | SwiftUI blocks, single selectable surface                  |
| Export      | HTML, PDF, plain text                                       |
| Platforms   | macOS 15+, iOS 18+                                          |


## Limitations

- Streaming resolves reference-style links backward only: a definition
  that arrives after a block has already sealed does not retro-update that
  block. Static `Markdown.parse` resolves both directions.
- Math is Unicode substitution (Greek letters, operators, super / sub
  scripts, simple fractions), not a TeX layout engine. Complex layouts
  degrade to readable inline text.
- Table column widths and list tight / loose state of the open block may
  shift while it grows; both freeze when the block seals.
