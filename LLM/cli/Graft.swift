import Foundation
import LLM

func runMeta(_ args: CommandArgs) {
    let v = args.values("--meta", 3)
    let src = v.count > 0 ? v[0] : ""
    let out = v.count > 1 ? v[1] : ""
    let cfg = v.count > 2 ? v[2] : ""
    var status: Int32 = 0
    if src.isEmpty || out.isEmpty || cfg.isEmpty {
        err("usage: chatokf-cli x --meta <src.gguf> <out.gguf> "
            + "<generation_config.json>\n")
        status = 2
    } else {
        do {
            let report = try GGUFGraft.meta(src, to: out, config: cfg)
            err(String(format: "[meta] %d tensors, %d keys, %d eos, "
                       + "%.1f GB\n", report.text, report.keys,
                       report.tower, Double(report.bytes) / 1e9))
        } catch {
            err("[meta] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runDrafter(_ args: CommandArgs) {
    let v = args.values("--drafter", 3)
    let text = v.count > 0 ? v[0] : ""
    let donor = v.count > 1 ? v[1] : ""
    let out = v.count > 2 ? v[2] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: chatokf-cli x --drafter <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(
                text: text, donor: donor, to: out, tensors: ["blk."],
                keys: ["qwen35.block_count",
                       "qwen35.nextn_predict_layers"])
            err(String(format: "[drafter] %d text + %d drafter tensors, "
                       + "%d keys, %.1f GB in %.1fs\n", report.text,
                       report.tower, report.keys,
                       Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
            if report.tower == 0 {
                err("[drafter] FAIL -- the donor adds no block\n")
                status = 1
            }
        } catch {
            err("[drafter] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runAssist(_ args: CommandArgs) {
    let v = args.values("--assist", 3)
    let text = v.count > 0 ? v[0] : ""
    let donor = v.count > 1 ? v[1] : ""
    let out = v.count > 2 ? v[2] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: chatokf-cli x --assist <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(
                text: text, donor: donor, to: out,
                tensors: ["assist."], keys: ["gemma4.assist."])
            err(String(format: "[assist] %d text + %d head tensors, "
                       + "%d keys, %.1f GB in %.1fs\n", report.text,
                       report.tower, report.keys,
                       Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
            if report.tower == 0 {
                err("[assist] FAIL -- the donor adds no head\n")
                status = 1
            }
        } catch {
            err("[assist] \(error)\n")
            status = 1
        }
    }
    exit(status)
}

func runGraft(_ args: CommandArgs) {
    let v = args.values("--graft", 3)
    let text = v.count > 0 ? v[0] : ""
    let donor = v.count > 1 ? v[1] : ""
    let out = v.count > 2 ? v[2] : ""
    var status: Int32 = 0
    if text.isEmpty || donor.isEmpty || out.isEmpty {
        err("usage: chatokf-cli x --graft <text.gguf> <donor.gguf> "
            + "<out.gguf>\n")
        status = 2
    } else {
        let t0 = Date()
        do {
            let report = try GGUFGraft.graft(text: text, donor: donor,
                                             to: out)
            err(String(format: "[graft] %d text + %d tower tensors, %d keys, "
                       + "%.1f GB in %.1fs\n", report.text, report.tower,
                       report.keys, Double(report.bytes) / 1e9,
                       Date().timeIntervalSince(t0)))
        } catch {
            err("[graft] \(error)\n")
            status = 1
        }
    }
    exit(status)
}
