import Foundation
import LLM

public enum EncoderPack {

    static let repo = "leok7v/bekko-embedding-v1-a25m"
    static let revision = "b33646eb1f70cbc4bd9f01482d03485b7d18f8b2"
    public static let file = "bekko-embedding-v1-a25m-Q8_0.gguf"
    public static let bytes: Int64 = 137_000_480
    public static let name = "bekko-embedding-v1-a25m"

    public static let readme = URL(string:
        "https://huggingface.co/" + repo + "/blob/main/README.md")!

    public static let root: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        return support.appendingPathComponent("encoder", isDirectory: true)
    }()

    private static var home: URL {
        root.appendingPathComponent("a25m", isDirectory: true)
    }

    private static var fileURL: URL {
        home.appendingPathComponent(revision, isDirectory: true)
            .appendingPathComponent(file)
    }

    public static var path: String? {
        let whole = ModelCatalog.isComplete(fileURL.deletingLastPathComponent())
            && FileManager.default.fileExists(atPath: fileURL.path)
        return whole ? fileURL.path : nil
    }

    public static var onDisk: Bool { path != nil }

    public static var sizeText: String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    public static func fetch(
        onProgress: @escaping @Sendable (HubFetch.Status) -> Void
    ) async -> String? {
        var failure: String? = "download failed, check your connection"
        do {
            _ = try await HubFetch.fetch(
                repo: repo, into: home, revision: revision,
                files: [file], excludeFromBackup: true,
                background: isOS, report: onProgress)
            failure = nil
        } catch HubError.digest {
            failure = "download failed verification, try again"
        } catch HubError.space(let needed, let free, let freeable) {
            failure = HubFetch.spaceMessage(needed: needed, free: free,
                                            freeable: freeable)
        } catch {
            Diag.shared.report(.net, "[encoder] fetch failed: \(error)")
        }
        return failure
    }

    public static func erase() {
        try? FileManager.default.removeItem(at: root)
    }

}
