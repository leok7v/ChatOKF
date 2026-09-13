import Foundation
@testable import LLM

enum TestWeights {

    static let repoRoot: String = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
    }()

    static let bundleId = "io.github.leok7v.ChatOKF"

    private static let home = NSHomeDirectory()

    static let stores: [URL] = [
        URL(fileURLWithPath: home + "/Library/Containers/" + bundleId
            + "/Data/Library/Application Support/models"),
        URL(fileURLWithPath: home + "/Library/Application Support/models"),
    ]

    static let clones = [home + "/huggingface.co", home + "/Models"]

    static func find(_ name: String) -> String? {
        var out: String? = nil
        if let src = ModelCatalog.source(name),
           let file = ModelCatalog.ggufFiles[name] {
            for store in stores where out == nil {
                if let set = ModelCatalog.localSet(name, in: store),
                   ModelCatalog.isComplete(set) {
                    out = readable(set.appendingPathComponent(file).path)
                }
            }
            if out == nil { out = clone(src.repo, file) }
        }
        return out
    }

    static func clone(_ repo: String, _ file: String) -> String? {
        var out: String? = nil
        for root in clones where out == nil {
            out = readable(root + "/" + repo + "/" + file)
        }
        return out
    }

    static func missing(_ name: String) -> String {
        let src = ModelCatalog.source(name)
        let file = ModelCatalog.ggufFiles[name] ?? name
        return "no \(file): download \(name) in the app, or clone "
            + "\(src?.repo ?? name) under ~/huggingface.co or ~/Models"
    }

    private static func readable(_ path: String) -> String? {
        FileManager.default.isReadableFile(atPath: path) ? path : nil
    }
}
