# ChatOKF

A pure-Swift chat app that runs GGUF models on Metal and SIMD: the Qwen3.5
and Qwen3.8 hybrids, the ternary Bonsai builds of them, and gemma 4. macOS
and iOS from one codebase, with images, audio and video on the models that
have towers for them.

Pure Swift only: no C, no Python, no FFI. Metal and the system frameworks
are the whole dependency list; nothing runs on the Neural Engine.

## Build

Requirements: an Apple Silicon Mac and Xcode 26 (Swift 6, macOS 15 SDK). The
project also needs two Homebrew tools:

    brew install xcodegen xcode-build-server

`ChatOKF.xcodeproj` is NOT committed: it is generated from `project.yml` (the
source of truth) by xcodegen. So build settings live in `project.yml`, not the
Xcode UI, and there are no `project.pbxproj` merge conflicts. Regenerate it
after cloning and after any `project.yml` change:

    xcodegen generate

Then build from the command line, or open `ChatOKF.xcodeproj` in Xcode (Run, or
Product > Archive):

    xcodebuild -scheme ChatOKF -destination 'platform=macOS' build
    xcodebuild -scheme ChatOKF -destination 'generic/platform=iOS' build
    xcodebuild -scheme chatokf -configuration Release build

The last line builds the command-line tool at `Build/Products/Release/chatokf`.
It takes a path to any `.ggxf` file: a chat, `--bench` for the llama-bench
protocol, and the probe and gate modes the engines are gated against.

### Tests

    xcodebuild -scheme ChatOKF -destination 'platform=macOS' test

Green with no model on disk: the gates that need weights find a catalog model
in the app's own store, in a Hugging Face clone or under `~/Models`, and skip
by name when none is there.

The app is Apple-Silicon-only (arm64): `Float16` does not exist on Intel, so
`project.yml` excludes `x86_64` (the Mac App Store accepts an
Apple-Silicon-only app).

### Editor tooling (SourceKit-LSP)

`xcode-build-server` writes `buildServer.json` so SourceKit-LSP resolves against
the real Xcode build. Rerun it after generating, then restart the language
server:

    xcode-build-server config -scheme ChatOKF -project ChatOKF.xcodeproj

## Performance

Throughput on a MacBook Air (M3, 24 GB), Release build, same protocol as
`llama-bench -p 512 -n 128`: a 512-token prefill (`pp512`) then 128 decoded
tokens (`tg128`), reported tokens/sec. Every catalog model, one run each,
2026-09-05, on the Metal backend (`--cpu` opts out to the SIMD engine).

| Model               | File GB | Prefill pp512 | Decode tg128 |
|---------------------|--------:|--------------:|-------------:|
| Ternary-Bonsai-1.7B |    0.43 |   680.4 tok/s |   74.2 tok/s |
| gemma-4-E2B         |    2.48 |   452.7 tok/s |   41.5 tok/s |
| gemma-4-E2B-MTP     |    2.89 |   544.4 tok/s |   62.6 tok/s |
| Qwen3.5-4B          |    3.34 |    90.4 tok/s |    8.2 tok/s |
| gemma-4-E4B         |    3.53 |   203.4 tok/s |   19.6 tok/s |
| gemma-4-E4B-MTP     |    4.38 |   269.7 tok/s |   41.8 tok/s |
| gemma-4-12B         |    6.35 |   104.9 tok/s |   10.7 tok/s |
| Qwen3.5-9B          |    6.41 |    43.8 tok/s |    5.3 tok/s |
| gemma-4-12B-MTP     |    6.58 |   114.5 tok/s |   19.9 tok/s |
| Qwen3.8-27B-IQ1_S   |    6.63 |    28.7 tok/s |    3.8 tok/s |
| Ternary-Bonsai-27B  |    7.26 |    35.2 tok/s |    4.5 tok/s |
| Qwen3.8-27B-IQ2_XXS |    7.88 |    26.3 tok/s |    3.3 tok/s |
| Qwen3.8-27B-IQ3_XXS |   11.05 |    26.4 tok/s |    2.9 tok/s |
| Qwen3.8-27B-IQ4_XS  |   14.14 |    24.9 tok/s |    2.5 tok/s |
| Qwen3.8-27B-Q4_K_S  |   15.17 |    23.8 tok/s |    2.2 tok/s |

The `-MTP` files carry a drafter and decode with self-speculation, which is
what doubles their `tg128`. Debug builds of the app keep the engine at `-O`,
so the numbers hold there too.

<sub>MacBook Air (M3): 10-core GPU; 8-core CPU (4 performance + 4
efficiency); 100 GB/s unified-memory bandwidth; 24 GB RAM.</sub>

Prefill speed is the number that matters most in agentic use: when the model
searches the web, reads a Wikipedia article, or pulls in a news story, every
fetched byte is prompt to ingest, not text to generate. A tool round routinely
prefills 4-16 KB of page text to decode a two-sentence conclusion, so the
reading rate, not the talking rate, bounds how many sources a turn can afford.

## Append-only context, rollback, recurrent state

The hybrid trunk forces a session design that stateless servers never need,
and it is worth being explicit about why.

Three quarters of the layers are Gated DeltaNet: their memory is a fixed-size
recurrent state, a lossy fold of everything ingested so far. Unlike a KV
cache, that state is not addressable by prefix (there is no "reuse the first
N tokens" shortcut), and the only way to recompute it is to replay the entire
conversation through the model. So the engine owns its session state rather
than reconstructing it per request:

- **Append-only continuation.** Each turn renders only the delta (the
  previous stripped answer plus the new user turn) through the model's own chat
  template and appends it to the live state. Nothing is re-prefilled; total
  work over a conversation is O(conversation), and per-turn latency does not
  grow with history. The whole-history re-render plus common-prefix diff (the
  stateless-server pattern) is deliberately absent.
- **Marks and rollback.** Before the generation prompt of every turn the
  engine drops a mark: a deep snapshot of the recurrent state plus the paged
  KV (cheap, since completed KV pages are shared copy-on-write). The next turn
  rewinds to it, which is how transient bytes leave the context: raw
  `<think>` reasoning is dropped and the turn re-appends the stripped answer;
  a tool exchange is re-laid in the template's canonical form instead of the
  model's raw emission. Stop during prefill restores the pre-turn snapshot
  entirely, so the turn never happened.
- **Park / resume / persist.** The same snapshot primitive serializes: whole
  conversations park and resume over one loaded model, and the rendered
  system plus tools prefix is precooked to disk once and restored at launch,
  skipping most of the time-to-first-token.

## Memories, shared across models

What you ask the app to remember is kept as plain notes on the device, in
Google's [Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md):
one Markdown file per note with a short frontmatter (id, title, a
one-sentence description, tags, links), in a `memories.noindex` folder
under the app's Application Support. Nothing is hidden: a note is kept as
it is written, and the transcript shows one line naming it, `Remembered:
<title>`. The Memories tab in the sidebar lists every note to read or to
delete, grouped by area, and a delete erases the note and collapses every
link that named it down to the words it carried, so the store is never
left with a dangling reference; a deleted note waits thirty days in its
own trash. The model sees five tools, `memory_search`, `memory_read`,
`memory_create`, `memory_update` and `memory_forget`, and each call shows
in the transcript. At the start of every turn the question is embedded
with multilingual-e5-small, the dense pass is fused with a verbatim pass
by reciprocal rank, and the notes that fit are read into the context
silently. A note that restates one already on file is refused, so the
store does not fill with duplicates. One switch governs all of it:
Settings, Intelligence, Total Recall.

The notes are text and the index is built by an encoder that is not the
chat model, so a memory belongs to the store, not to the model that wrote
it. Measured on 2026-09-13: gemma-4-12B-MTP read a 53-page paper and, on
request, split it into six notes. A new conversation on
Qwen3.5-9B was asked what we remember about the Levin research program;
it recalled three of those notes silently, then called `memory_search`
and `memory_read` on `science/levin_research_program` and answered from
it. Switching models keeps everything the app has learned: a model is a
reader, the notes are the knowledge.

## Layout

- `LLM/` - the engine: `Base`, `Qwen`, `Gemma`, `Quantize`, `Slugs`, `TTS`
  (the seams, tokenizer, chat template, sampler and ChatSession in Base; a
  lineage per directory; the weight formats; wikipedia search; speech).
- `LLM/metal/` - the Metal kernels both lineages run on.
- `LLM/cli/` - chatokf, the macOS command-line harness for the engine.
- `Chat/` - the session driver and conversation store, no SwiftUI.
- `App/` - SwiftUI app (macOS + iOS), no `#if os` (SDK file split).
- `MD/` - Markdown transcript rendering package.
- `config/platform.xcconfig` - the SDK-scoped source split for the app target.

## Models

The app ships no weights. Model files download on demand from their pinned
Hugging Face commits (`ModelCatalog` + `HubFetch`: sha-pinned, digest-verified,
resumable) into the app's container, once; later launches are offline. A
downloaded file is ready the moment it lands: there is no compile step.

The one cache beside the models is the precooked system prefix under
`~/Library/Caches/<bundle-id>/precook/`, a parked engine state per model and
prompt that a launch restores instead of re-prefilling. Deleting it costs one
prefill per model.

## License

ChatOKF is **GPLv3 or later**. That is arithmetic, not preference -- it is the
strongest obligation among the parts it is built from:

| part | source | licence |
|---|---|---|
| speech model + voices | [KittenTTS](https://github.com/KittenML/KittenTTS) by KittenML | Apache-2.0 |
| English pronunciation data (`en_rules`, `en_list`) | [eSpeak NG](https://github.com/espeak-ng/espeak-ng) | **GPLv3 or later** |
| everything else here | this repo | Copyright (C) 2026 Leo Kuznetsov |

The pronunciation data is copyleft, and eSpeak NG grants **no exception for a
program's output** -- so shipping those files, or a lexicon derived by running
them, carries the same terms rather than escaping them. Apache-2.0 is
one-way compatible with GPLv3 (it may be combined into a GPLv3 work, though
not GPLv2), which is why the result is GPL **v3** specifically and cannot be
anything more permissive.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE).

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

### Credits

- **[KittenML / KittenTTS](https://github.com/KittenML/KittenTTS)** -- the
  speech model and its eight voices, Apache-2.0.
- **[eSpeak NG](https://github.com/espeak-ng/espeak-ng)** -- the English
  letter-to-sound rules and exception dictionary, Copyright (C) 2005-2014
  Jonathan Duddington and Copyright (C) 2016-2017 Reece H. Dunn, GPLv3+.

Model weights are covered by their own upstream licences, not by this repo's.

---

### Footnote: how the gemma-4-12B repack scores

The 12B GGUF this app downloads is a *recovery* of the quantization-aware
training's own 4-bit codes, not a fresh quantization of the released weights.
Google publishes a second 4-bit build of the same checkpoint, so the two can be
scored against the bf16 they both come from.

Relative weight error against that checkpoint:

| tensor | [ours](https://huggingface.co/leok7v/gemma-4-12b-it-qat) | [w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) |
|---|---|---|
| `gate_proj` layer 0 | **1.03e-03** | 6.66e-02 |
| `q_proj` layer 0 | **1.07e-03** | 6.67e-02 |
| `down_proj` layer 30 | **1.10e-03** | 6.66e-02 |

About 65x closer to the trained weights, in a file 6.35 GiB against 9.56 GiB.

Not a better search: the int4 **codes agree 99.40%** between the two files, so
both recover the same trained grid, and Google's build is an independent
witness that the recovery is right. The **scales** differ, theirs a median
1.0645x larger, because a min/max observer takes the block scale from the
block's extreme where this repack refits it by least squares over the settled
codes.

Their build is better on one tensor: it leaves the embedding table in bf16 and
therefore exact, at 1.88 GiB against 0.53 GiB here. Both leave the same modules
unquantized -- the vision patch dense, both multimodal projections and the
position table.

The measurement is reproduced by the converter tooling in the development
repo. Sources:
[base checkpoint](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized),
[Google's 4-bit build](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct),
[this repack](https://huggingface.co/leok7v/gemma-4-12b-it-qat).
