import Foundation

public enum Flags {

    public enum Scope: Sendable { case engine, diagnostic, experiment }

    public struct Knob: Sendable {
        public let name: String
        public let takesValue: Bool
        public let help: String
        public let scope: Scope
        public let defaultValue: String
    }

    public static let registry: [Knob] = [
        Knob(name: "spec-n", takesValue: true, help:
            "speculative/MTP draft token count", scope: .engine,
            defaultValue: ""),
        Knob(name: "srq", takesValue: false, help:
            "gemma4 static-range activation clamps", scope: .experiment,
            defaultValue: "1"),
        Knob(name: "gemma-batch", takesValue: true, help:
            "gemma4 prefill chunk width", scope: .engine, defaultValue: ""),
        Knob(name: "tool-grammar", takesValue: true, help:
            "off | structural | full tool-call grammar mask", scope: .engine,
            defaultValue: "structural"),
        Knob(name: "prefill-chunk", takesValue: true, help:
            "ternary-model prefill chunk width", scope: .engine,
            defaultValue: "512"),
        Knob(name: "skip", takesValue: true, help:
            "comma-separated kernel groups to omit from the forward",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "seed", takesValue: true, help:
            "sampler seed", scope: .engine, defaultValue: "0"),
        Knob(name: "metal-timing", takesValue: false, help:
            "per-commit GPU-vs-wall milliseconds on stderr",
            scope: .diagnostic, defaultValue: "0"),
        Knob(name: "mtp-drafts", takesValue: true, help:
            "MTP draft count for the ternary self-speculative decode",
            scope: .engine, defaultValue: ""),
        Knob(name: "mtp-ring-share", takesValue: true, help:
            "max fraction of device RAM the MTP state ring may cost",
            scope: .engine, defaultValue: "0.05"),
        Knob(name: "hess-wide", takesValue: false, help:
            "collect the ffn_down Hessian site too", scope: .diagnostic,
            defaultValue: "0"),
        Knob(name: "gpu-labels", takesValue: false, help:
            "name every dispatch with a debug group for Instruments",
            scope: .diagnostic, defaultValue: "0"),
        Knob(name: "top-clusters", takesValue: true, help:
            "gemma4 assist-head clustered lm_head candidate count",
            scope: .engine, defaultValue: "32"),
        Knob(name: "prefill-layers", takesValue: true, help:
            "gemma4 decoder layers per prefill command buffer",
            scope: .experiment, defaultValue: "1"),
        Knob(name: "centroids", takesValue: false, help:
            "gemma4 clustered lm_head decode", scope: .engine,
            defaultValue: "0"),
        Knob(name: "digit-exempt", takesValue: false, help:
            "exempt digit tokens from repetition penalties", scope: .engine,
            defaultValue: "1"),
        Knob(name: "title-instruction", takesValue: true, help:
            "override the auto-title generation instruction", scope: .engine,
            defaultValue: ""),
        Knob(name: "followup-instruction", takesValue: true, help:
            "override the follow-up-question generation instruction",
            scope: .engine, defaultValue: ""),
        Knob(name: "stall-probe", takesValue: false, help:
            "arm the transcript stall-detection shimmer", scope: .diagnostic,
            defaultValue: "0"),
        Knob(name: "speech-floor", takesValue: true, help:
            "gigabytes of RAM the reading voice requires", scope: .experiment,
            defaultValue: "3"),
        Knob(name: "recall-budget", takesValue: true, help:
            "seconds of prefill a silent memory recall may add to a turn",
            scope: .engine, defaultValue: "5"),
        Knob(name: "read-floor-mb", takesValue: true, help:
            "megabytes of jetsam headroom under which a document read stops "
            + "and answers from what was read (iOS)", scope: .engine,
            defaultValue: "300"),
        Knob(name: "search-parallel", takesValue: false, help:
            "use the Parallel web search service", scope: .engine,
            defaultValue: "1"),
        Knob(name: "search-mwmbl", takesValue: false, help:
            "use the Mwmbl web search index", scope: .engine,
            defaultValue: "1"),
        Knob(name: "debug", takesValue: false, help:
            "the diagnostics master switch, equal to --verbosity 1",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "verbosity", takesValue: true, help:
            "0 faults, 1 +load/turn, 2 +net/tools/voice, 3 +everything",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "diagnostics", takesValue: true, help:
            "comma-separated categories, replacing the verbosity set",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "log-load", takesValue: false, help:
            "launch, model prepare, prime and release", scope: .diagnostic,
            defaultValue: ""),
        Knob(name: "log-net", takesValue: false, help:
            "download lanes", scope: .diagnostic, defaultValue: ""),
        Knob(name: "log-turn", takesValue: false, help:
            "one line per reply, and one per attachment",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "log-tools", takesValue: false, help:
            "tool rounds and the network traffic they cause",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "log-voice", takesValue: false, help:
            "microphone capture and speech playback", scope: .diagnostic,
            defaultValue: ""),
        Knob(name: "log-perf", takesValue: false, help:
            "main-thread stalls and screen rebuild counts",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "log-memory", takesValue: false, help:
            "a memory line per download and prefill chunk", scope: .diagnostic,
            defaultValue: ""),
        Knob(name: "log-pressure", takesValue: false, help:
            "system memory-pressure warnings", scope: .diagnostic,
            defaultValue: ""),
        Knob(name: "log-transcript", takesValue: false, help:
            "every render/prefill/decode with its full text",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "gpu-capture", takesValue: true, help:
            "write a .gputrace document to this path (kernel-bench)",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "gpu-capture-only", takesValue: true, help:
            "restrict kernel-bench to tensor names containing this",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "bench-prompt", takesValue: true, help:
            "the app runs this prompt drafted and plain at launch, 512 = "
            + "the built-in 512-token text", scope: .diagnostic,
            defaultValue: ""),
        Knob(name: "bench-tokens", takesValue: true, help:
            "tokens per bench arm", scope: .diagnostic, defaultValue: "128"),
        Knob(name: "bench-cool", takesValue: true, help:
            "seconds idle between the bench arms", scope: .diagnostic,
            defaultValue: "90"),
        Knob(name: "bench-thinking", takesValue: false, help:
            "reason during the bench arms; off so a run is not inherited "
            + "from whatever the app last had on screen",
            scope: .diagnostic, defaultValue: "0"),
        Knob(name: "memories", takesValue: false, help:
            "the memory store; --no-memories runs a turn on the context "
            + "alone, neither recalling nor writing a note",
            scope: .engine, defaultValue: "1"),
        Knob(name: "read-file", takesValue: true, help:
            "a file in the app's Caches folder the app attaches under UL "
            + "and sends at launch; repeatable, each in its own chat",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "read-prompt", takesValue: true, help:
            "the question sent with read-file, default a summary request",
            scope: .diagnostic, defaultValue: ""),
        Knob(name: "prompt", takesValue: true, help:
            "a turn the app sends at launch, repeatable and in order; "
            + "'new' leaves the conversation and 'reopen' brings it back",
            scope: .diagnostic, defaultValue: ""),
    ]

    private static let byName: [String: Knob] = {
        var out: [String: Knob] = [:]
        for knob in registry { out[knob.name] = knob }
        return out
    }()

    public static let arguments = ProcessInfo.processInfo.arguments

    public static func value(_ name: String) -> String? {
        parsed[name]
            ?? UserDefaults.standard.string(forKey: Flags.storeKey(name))
            ?? byName[name]?.defaultValue
    }

    public static func int(_ name: String) -> Int? {
        value(name).flatMap { text in Int(text) }
    }

    public static func values(_ name: String) -> [String] {
        var out: [String] = []
        var i = 1
        while i < arguments.count {
            let token = arguments[i]
            if token == "--" + name, i + 1 < arguments.count {
                out.append(arguments[i + 1])
                i += 1
            } else if token.hasPrefix("--" + name + "=") {
                out.append(String(token.dropFirst(name.count + 3)))
            }
            i += 1
        }
        return out
    }

    public static func double(_ name: String) -> Double? {
        value(name).flatMap { text in Double(text) }
    }

    public static func uint64(_ name: String) -> UInt64? {
        value(name).flatMap { text in UInt64(text) }
    }

    public static func truthy(_ text: String) -> Bool {
        !text.isEmpty && text != "0" && text.lowercased() != "false"
    }

    public static func on(_ name: String) -> Bool {
        Flags.truthy(value(name) ?? "")
    }

    public static func store(_ name: String, _ value: Bool) {
        UserDefaults.standard.set(value ? "1" : "0",
                                  forKey: Flags.storeKey(name))
    }

    private static func storeKey(_ name: String) -> String { "flag.\(name)" }

    public static var given: String {
        arguments.dropFirst().joined(separator: " ")
    }

    // argv cannot change while the process runs, so it is read once, lock-free.
    private static let parsed: [String: String] = {
        var out: [String: String] = [:]
        var i = 1
        while i < arguments.count {
            let token = arguments[i]
            var consumed = 1
            if token.hasPrefix("--") {
                let body = String(token.dropFirst(2))
                if let cut = body.firstIndex(of: "=") {
                    out[String(body[..<cut])] =
                        String(body[body.index(after: cut)...])
                } else if body.hasPrefix("no-") {
                    out[String(body.dropFirst(3))] = "0"
                } else if let knob = byName[body], knob.takesValue,
                          i + 1 < arguments.count,
                          !arguments[i + 1].hasPrefix("--") {
                    out[body] = arguments[i + 1]
                    consumed = 2
                } else {
                    out[body] = "1"
                }
            }
            i += consumed
        }
        return out
    }()

}
