import Foundation

public enum ModelCatalog {
    public struct Source: Sendable {
        public let repo: String
        public let revision: String
        public let bytes: Int64
        public let files: [String]?

        public init(repo: String, revision: String, bytes: Int64,
                    files: [String]? = nil) {
            self.repo = repo
            self.revision = revision
            self.bytes = bytes
            self.files = files
        }
    }

    public static let sources: [String: Source] = [
        "Qwen3.5-4B": Source(
            repo: "leok7v/Qwen3.5-4B",
            revision: "0aa2a28b493ae1c4c480b930699661f65d9d5061",
            bytes: 3_663_088_256,
            files: ["Qwen3.5-4B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.5-9B": Source(
            repo: "leok7v/Qwen3.5-9B",
            revision: "efa929be9c80e2e5148b88d6df0c479b7fe121e8",
            bytes: 7_053_200_960,
            files: ["Qwen3.5-9B-UD-Q4_K_XL.ggxf"]),
        "Qwen3.8-27B-IQ1_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "455d15270199764b3e6bb27bde0a9306cfd76f5c",
            bytes: 7_385_847_936,
            files: ["Qwen3.8-27B-IQ1_S.ggxf"]),
        "Qwen3.8-27B-IQ2_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "455d15270199764b3e6bb27bde0a9306cfd76f5c",
            bytes: 8_459_696_256,
            files: ["Qwen3.8-27B-IQ2_XXS.ggxf"]),
        "Qwen3.8-27B-IQ3_XXS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "455d15270199764b3e6bb27bde0a9306cfd76f5c",
            bytes: 11_862_468_736,
            files: ["Qwen3.8-27B-IQ3_XXS.ggxf"]),
        "Qwen3.8-27B-IQ4_XS": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "455d15270199764b3e6bb27bde0a9306cfd76f5c",
            bytes: 15_180_454_016,
            files: ["Qwen3.8-27B-IQ4_XS.ggxf"]),
        "Qwen3.8-27B-Q4_K_S": Source(
            repo: "leok7v/Qwen3.8-27B",
            revision: "455d15270199764b3e6bb27bde0a9306cfd76f5c",
            bytes: 16_285_821_056,
            files: ["Qwen3.8-27B-Q4_K_S.ggxf"]),
        "Ternary-Bonsai-27B": Source(
            repo: "leok7v/Ternary-Bonsai-27B-gguf",
            revision: "369aa33df3d8a8255a35a160305b6632eeb3f1a1",
            bytes: 7_794_369_344,
            files: ["Ternary-Bonsai-27B-Q2_0.ggxf"]),
        "Ternary-Bonsai-1.7B": Source(
            repo: "leok7v/Ternary-Bonsai-1.7B-gguf",
            revision: "0fb751628b00550b9656984c5870e4cfdea392a7",
            bytes: 463_292_000,
            files: ["Ternary-Bonsai-1.7B-Q2_0.ggxf"]),
        "gemma-4-E2B": Source(
            repo: "leok7v/gemma-4-e2b-it-qat",
            revision: "a98d8daed9141ec98f68f54fafa7a08efa8ca580",
            bytes: 2_665_414_656,
            files: ["gemma-4-e2b-it-qat.ggxf"]),
        "gemma-4-E4B": Source(
            repo: "leok7v/gemma-4-e4b-it-qat",
            revision: "712cfb35aad315ed3f8fc3f92c0c04b83ee2023f",
            bytes: 3_794_550_784,
            files: ["gemma-4-e4b-it-qat.ggxf"]),
        "gemma-4-12B": Source(
            repo: "leok7v/gemma-4-12b-it-qat",
            revision: "5e2780b22b34c107f97f7437042b893f41549067",
            bytes: 6_822_641_664,
            files: ["gemma-4-12b-it-qat.ggxf"]),
        "gemma-4-E2B-MTP": Source(
            repo: "leok7v/gemma-4-e2b-it-qat",
            revision: "9572f8df2aa70e2ca5a10faa680bfe1e52563328",
            bytes: 3_097_968_640,
            files: ["gemma-4-e2b-it-qat-mtp.ggxf"]),
        "gemma-4-E4B-MTP": Source(
            repo: "leok7v/gemma-4-e4b-it-qat",
            revision: "8b44354d1de72eff09f443ae1255bf326d30bb1f",
            bytes: 4_698_226_688,
            files: ["gemma-4-e4b-it-qat-mtp.ggxf"]),
        "gemma-4-12B-MTP": Source(
            repo: "leok7v/gemma-4-12b-it-qat",
            revision: "5e2780b22b34c107f97f7437042b893f41549067",
            bytes: 7_060_897_792,
            files: ["gemma-4-12b-it-qat-mtp.ggxf"]),
    ]

    public static func source(_ name: String) -> Source? {
        sources[name]
    }

    public static let ggufFiles: [String: String] = [
        "Qwen3.5-4B": "Qwen3.5-4B-UD-Q4_K_XL.ggxf",
        "Qwen3.5-9B": "Qwen3.5-9B-UD-Q4_K_XL.ggxf",
        "Qwen3.8-27B-IQ1_S": "Qwen3.8-27B-IQ1_S.ggxf",
        "Qwen3.8-27B-IQ2_XXS": "Qwen3.8-27B-IQ2_XXS.ggxf",
        "Qwen3.8-27B-IQ3_XXS": "Qwen3.8-27B-IQ3_XXS.ggxf",
        "Qwen3.8-27B-IQ4_XS": "Qwen3.8-27B-IQ4_XS.ggxf",
        "Qwen3.8-27B-Q4_K_S": "Qwen3.8-27B-Q4_K_S.ggxf",
        "Ternary-Bonsai-27B": "Ternary-Bonsai-27B-Q2_0.ggxf",
        "Ternary-Bonsai-1.7B": "Ternary-Bonsai-1.7B-Q2_0.ggxf",
        "gemma-4-E2B": "gemma-4-e2b-it-qat.ggxf",
        "gemma-4-E4B": "gemma-4-e4b-it-qat.ggxf",
        "gemma-4-12B": "gemma-4-12b-it-qat.ggxf",
        "gemma-4-E2B-MTP": "gemma-4-e2b-it-qat-mtp.ggxf",
        "gemma-4-E4B-MTP": "gemma-4-e4b-it-qat-mtp.ggxf",
        "gemma-4-12B-MTP": "gemma-4-12b-it-qat-mtp.ggxf",
    ]

    public static func isGguf(_ name: String) -> Bool {
        ggufFiles[name] != nil
    }

    public static func ggufPath(_ name: String, in dest: URL) -> String? {
        var result: String? = nil
        if let file = ggufFiles[name], let set = localSet(name, in: dest) {
            dropStale(name, in: dest)
            result = set.appendingPathComponent(file).path
        }
        return result
    }

    public static func dropStale(_ name: String, in dest: URL) {
        let fm = FileManager.default
        if let set = localSet(name, in: dest) {
            let home = dest.appendingPathComponent(name, isDirectory: true)
            let kids = (try? fm.contentsOfDirectory(
                at: home, includingPropertiesForKeys: nil)) ?? []
            for kid in kids
            where kid.lastPathComponent != set.lastPathComponent {
                try? fm.removeItem(at: kid)
            }
        }
    }

    // The pin IS the sha, so the app names the set dir offline with no
    // round-trip.
    public static func localSet(_ name: String, in dest: URL) -> URL? {
        var result: URL? = nil
        if let src = source(name) {
            result = dest.appendingPathComponent(name, isDirectory: true)
                .appendingPathComponent(src.revision, isDirectory: true)
        }
        return result
    }

    // Present ONLY with HubFetch's .complete sentinel: a half-download would
    // send Engine.loadEssential down a wrong file-presence branch.
    public static func isComplete(_ setDir: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: setDir.appendingPathComponent(HubFetch.sentinel).path)
    }
}
