# VL OCR test images

Real-image OCR cases for the on-device vision path, to compare across model
sizes. Run via the app's [+] or a `chatokf <file.ggxf> "img:<img> <prompt>"`
turn, which takes the model's own media encoder at `--image-budget`.

The readings below are the baselines they were when taken: the Qwen3.5 and
QwenPaw CoreML sets named in them are archived (okf/references/aneq-archive.md)
and the `--vl-*` commands went with them. Every image is now scaled to the
token budget at its own aspect ratio on every model, so a re-run reads
handwriting the forced 256x256 resize of that era destroyed.

## handwriting-audit.jpeg (1103x1638)

Prompt: `OCR the text on this picture`

Ground truth (blue ink on grid paper, three sections):

    Tools for running the audit
      -> added info.
      -> Treasury status.
      -> Weekly for info., present, HR
    ---
    Emails & results
    checklist.
    Graph report + analysis.
    ---
    Create custom control of audit
      -> GRC control.
    Update report -> access to board
    Deadline & results.
    Reporting.

Reference readings (2026-07-14):
- Gemini: near-exact (only "outcome" for "custom" control).
- Qwen3.5-0.8B @256: largely hallucinated ("Take for using the cell...",
  "Click cells", "Deliveries & audit") -- the forced-256 resize wipes the pen
  strokes. This is the 0.8B baseline the bigger sets must beat.

## newspaper-1946.jpeg (1473x2000, The Oakland Observer, page 4)

Prompt: `OCR all text in the picture respond with text only`

A dense full newspaper page: a multi-column article ("French Institute Creates
A French Community At OU", ~8pt body) plus ads (MITZELFELD'S, HILLS THEATRE "My
Fair Lady", TUKO Suzuki, Dairy Queen "Fiestas/Splits", R.B. Dunlop Tire Sales
651-3422 / 673-9227 / after 5 p.m. 334-6452, M.G.M. Cleaners "Do You Have A Full
House?", Arnold Rexall Pharmacy). Ground-truth anchors: NDEA, 48 participants,
Don Iodice, Mme Genevieve Prevost, M. Charles Forton, Francis Tafoya, Pierre
Simonian. This is the stress case for tiling resolution + decode robustness.

Reference readings (2026-07-15, on-device, tiling):
- Qwen3.5-0.8B: mostly hallucinated body (small model + resolution).
- QwenPaw-Flash-2B @12 tiles: real ads (Dunlop phone exact) but confabulated the
  dense body. @24 tiles (maxTiles=20): body flips to real OCR (Prevost, "National
  Defense Education Act ... summer institutes", "language analysis and
  methodology").
- QwenPaw-Flash-4B: best OCR (exact headline, 48 participants, Prevost, Forton,
  Tafoya, Simonian) -- BUT first exposed a decode bug: the vision path used raw
  greedy with no token cap, so a long transcription degenerated into unbounded
  word-salad drifting to CJK. Fixed by sampling the vision reply with the model's
  penalties + a token cap; the collapse is gone and the OCR remains.

## Ad / column crops (1946, same Oakland Observer page)

Single-item crops from newspaper-1946.jpeg -- smaller and more legible than the
full page, so they isolate resolution (a crop at 512 should read cleanly where
the whole page at 512 still packs ~1024px of detail into one tile). Prompt:
`OCR all text in the picture respond with text only`.

- ad-rexall.jpg (231x375): "Arnold Rexall Pharmacy / Prescriptions / Cosmetics /
  Sundry Items / Liquor, Beer, Wine / 2026 Opdyke Rd. / Corner of Pontiac Road /
  333-7033".
- ad-mgm-cleaners.jpg (467x692): "DO YOU HAVE A FULL HOUSE?" + storage-service
  copy; "P.S. FREE ... Cuddly Teddy Bear, Pussy Cat, or Puppy Dog (Life Size)
  with $50 in M.G.M. cleaning receipts"; "M.G.M. Cleaners, Inc. / In Business
  for 21 Years / Auburn Rd., at Adams / Crooks Rd., at Auburn / Mound Rd., at 23
  Mile Rd."; "Open 7 A.M. to 8 P.M., Mon. thru Sat. In by 10 A.M.-Out by 5 P.M.".
- ad-tuko-suzuki.jpg (476x607): "TUKO / 872 E. Auburn, Near John R. Rochester
  UL 2-5363 / SUZUKI" (photo of a rider on a motorbike).
- article-french-institute.jpg (1034x304): headline "French Institute Creates A
  French Community At OU"; body anchors NDEA, 48 participants, "14 men and 34
  women, 10 of them nuns", Michigan + 14 other states, Pierre Simonian.
- ad-hills-theatre.jpg (491x698): "HILLS THEATRE / Rochester / Friday - Tuesday /
  ONE SHOWING NIGHTLY 7:30 / SUNDAY 2:30 & 7:30 / Program Information 651-8311 /
  FIRST TIME AT POPULAR PRICES! / MY FAIR LADY / ... / Winner of 8 Academy
  Awards!".

## heart.png + cat.png -- the cross-image reference case

Two images in ONE turn, where the answer requires relating them: heart.png is
a heart, and cat.png is a cat whose EYES are hearts.

Prompt: `Picture 1 shows a shape. Does that same shape appear anywhere inside
Picture 2? Answer yes or no, and say where.`

Run:

    chatokf-cli <qwen-set> --vl-images LLM/fixtures/vl/heart.png,LLM/fixtures/vl/cat.png "<prompt>"
    chatokf-cli <gemma.gguf> --gemma-chat --metal \
        --image LLM/fixtures/vl/heart.png,LLM/fixtures/vl/cat.png "<prompt>"

"Picture 1" / "Picture 2" are the names the model is actually given: Qwen's
template numbers attachments itself (add_vision_id), and ChatSession numbers
them for a template that does not (gemma).

Ground truth: YES -- the cat's eyes are hearts.

Measured 2026-07-31:

- Qwen3.5-0.8B (CoreML): CORRECT. Names both pictures and locates the shape
  ("two large, black hearts forming its eyes").
- gemma-4-E2B (Metal): says "No". It is NOT failing to see -- shown the cat
  ALONE it says the eyes are "two black hearts", and even with both images
  present, asked directly, it still says "heart-shaped". It describes both
  images correctly in one turn. Only the cross-image JUDGEMENT fails.

Do not re-run the sliding-window theory: at 64 soft tokens per image the whole
turn is ctx 180, far inside gemma's 512 window, and it still fails. Attention
geometry is refuted. The likelier cause is training exposure -- Qwen's
template ships multi-image numbering, gemma's has no such mechanism at all.

## elephant.jpeg -- a natural photo

Leo's own photograph (192x256): a man riding an elephant on a jungle track,
the animal filling most of the frame. It is here because the ad fixtures are
all dense TEXT and cannot separate reading from seeing; this one is ordinary
photography, where the only question is whether the model sees an animal.

    chatokf-cli <gemma.gguf> --metal --gemma-image-say \
        LLM/fixtures/vl/elephant.jpeg "What animal is the man riding?"

NOT YET MEASURED. It is also an EASIER case than the frame it succeeded --
the elephant is the subject here rather than something secondary behind the
man -- so a pass proves less. Read `quoted/elephant-zoo.jpeg` below for the
probe that actually discriminates.

## quoted/elephant-zoo.jpeg -- the one where the 12B breaks

A single frame from `me-at-the-zoo.mp4` (320x240), QUOTED for test use, which
is why it sits in `quoted/`: that directory is excluded from the public
mirror. A man faces the camera with an elephant plainly visible behind his
right shoulder, so the ground truth needs no argument, and the elephant being
SECONDARY is the whole value of the case.

It is the regression fixture for the 12B bidirectional-vision-attention fix
(bca2d31 + 2fe263b), which is why the pixels are kept rather than replaced.

    chatokf-cli <gemma.gguf> --metal --gemma-image-say \
        LLM/fixtures/vl/quoted/elephant-zoo.jpeg "What animal is behind the man?"

Measured 2026-08-04, same pixels into both:

- gemma-4-E2B: CORRECT ("an elephant"), at the full still budget and at the
  70-token video budget alike.
- gemma-4-12B (unified, encoder-free): WRONG. "I cannot see an animal ...
  the background appears to be a building with windows and some foliage."
  Asked to think first, it describes the frame as "a very wide, low
  resolution panoramic" with the man "bottom right", which this 4:3 frame is
  not.
- llama.cpp `llama-mtmd-cli --jinja` on unsloth's gemma-4-12b-it GGUF plus
  its mmproj: CORRECT, in detail ("large, grey, wrinkled ... large ears
  ... an elephant"). So the 12B CAN see it and our path is what loses it.
