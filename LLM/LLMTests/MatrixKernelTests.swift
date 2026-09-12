import Foundation
import Testing
@testable import LLM

struct MatrixKernelTests {

    private func source(_ name: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("metal/\(name)")
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func emitted(_ text: String) -> [String] {
        var out: [String] = []
        for line in text.split(separator: "\n") {
            let s = line.trimmingCharacters(in: .whitespaces)
            if s.hasPrefix("GEMM_MM_KERNEL("),
               let open = s.firstIndex(of: "("),
               let comma = s.firstIndex(of: ",") {
                out.append(String(s[s.index(after: open) ..< comma])
                    .trimmingCharacters(in: .whitespaces))
            }
        }
        return out
    }

    @Test func everyMatrixKernelIsSkippedWithoutMatrixUnits() throws {
        var names = try emitted(source("Kernels.metal"))
        names += ["attn_batch_mm", "f16w_gemm_mm", "iq_gemm_mm_h"]
        #expect(names.count > 20)
        for name in names {
            #expect(MetalContext.needsMatrixUnits(name),
                    "\(name) would be built on a GPU without matrix units")
        }
    }

    @Test func plainSimdKernelsAreNotSkipped() {
        for name in ["q2_0_gemv", "rmsnorm", "attn_paged", "gdn_scan_batch",
                     "silu_mul", "rope", "embed_batch"] {
            #expect(!MetalContext.needsMatrixUnits(name))
        }
    }
}
