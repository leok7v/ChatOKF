# Pdf2md and Docs2md: where they come from, and the contract

`LLM/src/Base/Pdf2md.swift` and `LLM/src/Base/Docs2md.swift` are both
VENDORED. They are developed in a separate effort against its own benchmarks
and arrive here by copy. Re-import is `cp`, and it must stay that way. (This
directory is named for that effort, which ships both readers, not for the one
file.)

Pdf2md reads PDFs, which have to be reconstructed from geometry. Docs2md
reads the formats that STATE their structure -- docx, xlsx, pptx, epub, html
-- so it infers nothing and needs no OCR. Same contract for both.

## The contract

DO NOT EDIT A VENDORED FILE. Specifically, do not:

- rename their types (Pdf2md: `Converter`, `Page`, `Element`, `Mode`,
  `Fragment`, `Grid`, `Ruling`, `Audit`, `ConversionError`. Docs2md:
  `Format`, `DocsError`, `Archive`, `Markup`, `MarkupParser`, `Cell`,
  `DocsElement`, `Span`, `Emitter`, `DocsConverter`, `Relations`,
  `WordReader`, `HTMLParser`, `PageReader`, `SheetReader`, `SlideReader`,
  `BookReader`),
- add `public` to anything in them,
- nest their types to tidy the namespace,
- reflow, reorder, or "while I am here" them.

Every one of those is tempting and every one converts the next re-import
from a copy into a merge, against files whose constants are still being
swept upstream. None of them buys anything: the names collide with nothing
in this module (checked), and `internal` is sufficient because the public
surface lives in a SEPARATE file beside it. Put every adaptation there.

If a file needs a change to be usable, ASK UPSTREAM FOR IT rather than
making it locally. That worked once already: the span audit was `private`
and folded into a trace string, and upstream exposed it as data on request
(`Audit` plus the `onAudit` sink). A local patch would have been faster and
would have cost every re-import after it.

## The gate

Docs2md has no audit, and needs none: these formats state their structure, so
`Docs2mdTests` asserts the structure exactly -- a heading is a heading, a
table row is a table row. Most of its containers are BUILT in the test, which
the reader makes cheap by accepting STORED zip entries, so a fixture is a few
parts of XML and a forty-line archive writer with nothing to license.

`fixtures/docs/` is the exception, and covers what a hand-cut fixture cannot:
whole documents from a real writer, with styles.xml, numbering.xml, slide
masters and every other part an authoring tool emits. They assert a ROUND
TRIP -- Markdown out to a document and back -- so a table has to survive two
independent implementations of OOXML. They are still nobody else's:
`harvest-report.md` is our own invented prose and sits beside them, and
regenerating the pair is two `pandoc` commands (named in the test).

For Pdf2md, `onAudit` is the correctness assertion available here. It fires once per
page whatever `trace` says, and the counts need no ground truth:

    lost == 0 && unaccounted == 0

`lost` counts recognized text that never reached the output; `unaccounted`
counts output text that was never recognized. The failure it catches is the
quiet one -- a span inside a table's rows but outside its columns simply
vanishes, and nothing else in the pipeline notices.

The F1 bench (cell F1, token F1, against LaTeX-derived ground truth) is the
QUALITY question and stays upstream, where the corpus and the harness live.
What runs here is a REGRESSION gate: did this re-import break something the
app depends on. Two different jobs; do not try to host the first one.

`harvest-report.pdf` is the one fixture NOT generated, and it is here because
the generated ones were not enough. An upstream build turned its five columns
into seven, split a header cell in half, and put `Tonnes` and `47.2` into
different columns although both begin at x=241.1 -- and every generated case
passed it. A drawn table is tidier than a typeset one: three columns, ruled,
one word per header. This page is the app's own Document Understanding
sample, so what it asserts is what a user sees. It is a COPY of
App/Resources/harvest-report.pdf, since LLM is a framework the app consumes
and must not reach back into it; replace the app's sample and replace this
with it. VERIFIED TO BITE: it is the only test that fails on the build that
regressed, which is the whole reason it exists.

## Why the fixtures here are generated, not borrowed

Docs2md is scored upstream against Microsoft's markitdown (MIT,
https://github.com/microsoft/markitdown), currently 48/48 across docx, xlsx,
pptx, epub and html. Its test documents were NOT borrowed either, for the
first reason below: markitdown's own licence covers the tool, not
necessarily what its fixtures contain.

The upstream PDF benchmark is `piushorn/pdf-parse-bench`:

- https://huggingface.co/datasets/piushorn/pdf-parse-bench
- https://github.com/phorn1/pdf-parse-bench

It is MIT (Copyright 2025 Pius Horn) and its PDFs are produced by pdfTeX
from generated LaTeX, so there is no scraped-document problem. Borrowing a
few would have been defensible. It was declined anyway, for two reasons.

FIRST, the content is not as synthetic as the description suggests. The
DOCUMENTS are generated with randomized geometry, and the prose between the
tables is word salad -- but the tables themselves carry real harvested data.
Sampled from the corpus: one page scores AnyLoc, MegaLoc, EgoNN, BEVPlace++
and AutoPlace on nuScenes and Boreas; another scores Deepfakes, FaceShifter,
InfoSwap, SimSwap and BlendFace on ID similarity and FVD; a third is a
coherent algae-growth parameter set. That is lifted from published papers,
or sampled from something derived from them. Upstream's LICENSE does carry a
data note, but it covers the FORMULAS set (Wikipedia, CC BY-SA 4.0) and says
nothing about tables. Numerical results are thin copyright and the risk is
small, but this repo publishes to a public mirror, and a small unresolved
question is still an unresolved question.

SECOND, and this is the better reason: fixtures we generate come with
GROUND TRUTH. We place every cell, so a test can assert cell-level
correctness rather than smoke and anchor strings. Borrowed PDFs would have
given a weaker gate AND a licence footnote.

The cost is honest and worth stating: generated fixtures exercise what we
THINK is hard. The upstream corpus is adversarial in ways we would not
invent. That difficulty is not lost, it is simply not hosted here -- it is
what the upstream bench is for.
