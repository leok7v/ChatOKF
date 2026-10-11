import Foundation
import LLM

public enum VoicePack {

    public struct Pack: Sendable, Equatable {
        public let name: String
        public let file: String
        public let bytes: Int64
    }

    static let repo = "leok7v/supertonic"
    static let revision = "c8b0e4ee7181c403f4c01ed2f76e9c725128eaf6"

    public static let readme = URL(string:
        "https://huggingface.co/" + repo + "/blob/main/README.md")!

    public static let q4 = Pack(name: "q4", file: "supertonic-q4.safetensors",
                                bytes: 79_154_016)
    public static let q8 = Pack(name: "q8", file: "supertonic-q8.safetensors",
                                bytes: 111_959_776)

    public static let chosen: Pack = {
        let byMemory = isOS && installedGB <= 4 ? q4 : q8
        let asked = [q4, q8].first { pack in
            pack.name == Flags.value("voice-pack")
        }
        return asked ?? byMemory
    }()

    public static let root: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        return support.appendingPathComponent("voice", isDirectory: true)
    }()

    private static var home: URL {
        root.appendingPathComponent(chosen.name, isDirectory: true)
    }

    private static var file: URL {
        home.appendingPathComponent(revision, isDirectory: true)
            .appendingPathComponent(chosen.file)
    }

    public static var path: String? {
        let whole = ModelCatalog.isComplete(file.deletingLastPathComponent())
            && FileManager.default.fileExists(atPath: file.path)
        return whole ? file.path : nil
    }

    public static var onDisk: Bool { path != nil }

    static let declinedKey = "voicePackDeclined"

    public static var declined: Bool {
        get { UserDefaults.standard.bool(forKey: declinedKey) }
        set { UserDefaults.standard.set(newValue, forKey: declinedKey) }
    }

    public static var sizeText: String {
        ByteCountFormatter.string(fromByteCount: chosen.bytes,
                                  countStyle: .file)
    }

    public static func fetch(
        onProgress: @escaping @Sendable (HubFetch.Status) -> Void
    ) async -> String? {
        var failure: String? = "download failed, check your connection"
        do {
            _ = try await HubFetch.fetch(
                repo: repo, into: home, revision: revision,
                files: [chosen.file], excludeFromBackup: true,
                background: isOS, report: onProgress)
            failure = nil
        } catch HubError.digest {
            failure = "download failed verification, try again"
        } catch HubError.space(let needed, let free, let freeable) {
            failure = HubFetch.spaceMessage(needed: needed, free: free,
                                            freeable: freeable)
        } catch {
            Diag.shared.report(.net, "[voice] fetch failed: \(error)")
        }
        return failure
    }

    public static func erase() {
        try? FileManager.default.removeItem(at: root)
    }

}
