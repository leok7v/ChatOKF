import Foundation
import Testing
@testable import LLM

struct IQTablesTests {

    private func committed() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("metal/IQTables.h")
        return try String(contentsOf: url, encoding: .utf8)
    }

    @Test func theHeaderIsWhatTheSwiftGridsEmit() throws {
        let want = IQTablesEmit.header()
        let have = try committed()
        #expect(have == want, Comment(rawValue:
                "LLM/metal/IQTables.h is stale or hand-edited; run "
                + "`chatokf --emit-iq-tables > LLM/metal/IQTables.h`"))
    }

    @Test func aSixtyFourBitGridShipsLowWordFirst() {
        let split = IQTablesEmit.halves([0x0011223344556677, 0x8899aabbccddeeff])
        #expect(split == [0x44556677, 0x00112233, 0xccddeeff, 0x8899aabb])
    }

    @Test func everyGridReachesTheHeader() throws {
        let have = try committed()
        for name in ["iq1s_grid_gpu[2048]", "iq2xxs_grid_u32[512]",
                     "iq2xs_grid_u32[1024]", "iq2s_grid_u32[2048]",
                     "iq3xxs_grid[256]", "iq3s_grid[512]",
                     "ksigns_iq2xs[128]", "kmask_iq2xs[8]",
                     "kvalues_iq4nl[16]"] {
            #expect(have.contains(name), "\(name) is missing")
        }
    }
}
