import Foundation
import LLM

func runSplice(_ args: CommandArgs) {
    let v = args.values("--splice", 4)
    let base = v.count > 0 ? v[0] : ""
    let donor = v.count > 1 ? v[1] : ""
    let leaves = v.count > 2 ? v[2] : ""
    let out = v.count > 3 ? v[3] : ""
    var status: Int32 = 0
    if base.isEmpty || donor.isEmpty || leaves.isEmpty || out.isEmpty {
        err("usage: chatokf-cli x --splice <base.gguf> <donor.gguf> "
            + "<leaf,leaf,...> <out.gguf>\n")
        status = 2
    } else {
        let want = Set(leaves.split(separator: ",").map(String.init))
        do {
            let report = try GGUFGraft.splice(base: base, donor: donor,
                                              leaves: want, to: out)
            err(String(format: "[splice] %d of %d tensors from the donor, "
                       + "%.2f GB\n", report.tower, report.text,
                       Double(report.bytes) / 1e9))
            if report.tower == 0 {
                err("[splice] FAIL -- no tensor matched \(leaves)\n")
                status = 1
            }
        } catch {
            err("[splice] \(error)\n")
            status = 1
        }
    }
    exit(status)
}
