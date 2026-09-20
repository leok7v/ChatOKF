import Foundation
import LLM

let args = CommandArgs(CommandLine.arguments)
let rawArgs = args.all
let longDoc = args.flag("--longdoc")
let forceIngest = args.flag("--ingest")
let noCarry = args.flag("--no-carry")
let useGPU = !args.flag("--cpu")
// No memory reason to cap: the paged KV grows lazily.
let capVal = args.int("-n")
let maxTokens = capVal ?? Int.max
args.consume("--spec-n", hasValue: true)
let specNVal = Flags.int("spec-n")
// The plain bench decodes at ~640, where the KV cache is ~1% of a token's
// bytes; at 8K it is the dominant term on a dense model.
let benchCtxVal = args.int("--ctx")
// The acceptance test for a kernel REFACTOR, where cosine cannot see one ulp.
let metalGoldenDir = args.value("--metal-golden")
// Anything but `none` enables thinking; absent means empty-think.
let reVal = args.value("--reasoning-effort")
let enableThinking = (reVal.map { $0 != "none" } ?? false)
    || args.flag("--think")
let reasoningEffort = reVal.flatMap { v in
    ["none", "on"].contains(v) ? nil : v
}
args.consume("--seed", hasValue: true)
args.consume("--verbosity", hasValue: true)
args.consume("--diagnostics", hasValue: true)
let seedVal = Flags.uint64("seed") ?? 0
// Biases the curated branch-opening tokens down while thinking (arxiv
// 2606.00206).
let overthink = args.float("--overthink") ?? 0
// A no-EOS thinking runaway cannot hang; the n-gram loop breaker is always on.
let maxReasoning = args.int("--max-reasoning") ?? 0
// SOFT cap: end <think> at the next paragraph break past N, a cleaner cut.
let softReasoning = args.int("--soft-reasoning") ?? 0
// Leave the reasoning block open but spend none of it, which is what a
// system-block template (gemma-4) takes mid-conversation.
let suppressReasoning = args.flag("--no-reason")
let systemPrompt = args.text("--system") ?? "You are a helpful assistant."
let wmVal = args.value("--wiki-model")
let toolRunner: (any ToolRunner)? =
    SafeToolRunner(slugsPath: wmVal, wikipedia: wmVal != nil,
                   network: wmVal != nil)
// One JSON line per turn; the captured stderr is the full log, this the
// companion.
let trVal = args.value("--trace")
let traceURL: URL? = trVal.map { p in
    let dir = URL(fileURLWithPath: p, isDirectory: true)
    try? FileManager.default.createDirectory(
        at: dir, withIntermediateDirectories: true)
    let file = dir.appendingPathComponent("session.jsonl")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    return file
}
// The TTFT cache, per backend.
let pkVal = args.value("--precook")
let imageBudget = args.int("--image-budget") ?? 280
let greedyDecode = args.flag("--greedy")
let stopAtVal = args.int("--stop-at")
var stopArmed = true

if args.flag("--emit-iq-tables") {
    print(IQTablesEmit.header(), terminator: "")
    exit(0)
}

try probeTTS()
try probeVit()

let arg1 = args.rest.first ?? ""
var turnArgs: [String] { args.turns }

await probeNet()

if args.flag("--meta") { runMeta(args) }
if args.flag("--graft") { runGraft(args) }
if args.flag("--drafter") { runDrafter(args) }
if args.flag("--assist") { runAssist(args) }
if isModelFile(arg1), args.flag("--assist-probe") {
    try runAssistProbe(arg1, args)
}
if isModelFile(arg1), args.flag("--assist-bench") {
    try runAssistBench(arg1, args)
}
if args.flag("--splice") { runSplice(args) }
if isModelFile(arg1), args.flag("--replay-make") {
    try runReplayMake(arg1, args)
}
if isModelFile(arg1), args.flag("--replay") {
    try runReplayScore(arg1, args)
}
if isModelFile(arg1), rawArgs.contains("--pplit") {
    try runPplit(arg1, args)
}
if isModelFile(arg1), rawArgs.contains("--kld")
    || rawArgs.contains("--kld-dump") { try runDivergence(arg1, args) }
if args.flag("--puzzle-rescore") { runPuzzleRescore(args) }
if args.flag("--puzzle-gate") { await runPuzzleGate(args) }
if isModelFile(arg1) { try await runGgufMain() }

err("\(arg1): not a .ggxf -- this build runs GGUF models only\n")
exit(2)
