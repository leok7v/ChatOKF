import Foundation
import LLM

public var installedGB: Int {
    Int((ProcessInfo.processInfo.physicalMemory + (1 << 29)) >> 30)
}

#if DEBUG
public let debugBuild = true
#else
public let debugBuild = false
#endif

public extension Bundle {

    private static let store: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true))
            ?? fm.temporaryDirectory
        return support.appendingPathComponent("models", isDirectory: true)
    }()

    static func modelStore() -> URL { store }

    static var appName: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName")
            as? String)
            ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleName")
                as? String)
            ?? "ChatOKF"
    }

}

public enum Models {

    private static func qwen38(_ gb: Int) -> String? {
        let out: String?
        switch gb {
        case ..<9: out = nil
        case ..<16: out = "Qwen3.8-27B-IQ1_S"
        case ..<24: out = "Qwen3.8-27B-IQ2_XXS"
        case ..<32: out = "Qwen3.8-27B-IQ3_XXS"
        case 32: out = "Qwen3.8-27B-IQ4_XS"
        default: out = "Qwen3.8-27B-Q4_K_S"
        }
        return out
    }

    private static func gemmaRung(_ gb: Int) -> String? {
        let out: String?
        switch gb {
        case ..<4: out = nil
        case ..<6: out = "gemma-4-E2B"
        case ..<8: out = "gemma-4-E2B-MTP"
        default: out = "gemma-4-E4B-MTP"
        }
        return out
    }

    public static var every: [String] {
        isOS
            ? ["gemma-4-E2B", "gemma-4-E2B-MTP",
               "gemma-4-E4B", "gemma-4-E4B-MTP", "Ternary-Bonsai-1.7B"]
            : ["Qwen3.5-4B", "Qwen3.5-9B",
               "Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ2_XXS",
               "Qwen3.8-27B-IQ3_XXS", "Qwen3.8-27B-IQ4_XS",
               "Qwen3.8-27B-Q4_K_S",
               "Ternary-Bonsai-27B", "Ternary-Bonsai-1.7B",
               "gemma-4-E2B", "gemma-4-E2B-MTP",
               "gemma-4-E4B", "gemma-4-E4B-MTP",
               "gemma-4-12B", "gemma-4-12B-MTP"]
    }

    public static var downloaded: Set<String> {
        var out: Set<String> = []
        for name in every {
            if let dir = ModelCatalog.localSet(name, in: Bundle.modelStore()),
               ModelCatalog.isComplete(dir) {
                out.insert(name)
            }
        }
        return out
    }

    public static var all: [String] {
        let gb = installedGB
        var band: Set<String> = []
        if isOS {
            if let rung = gemmaRung(gb) { band.insert(rung) }
            if gb <= 3 { band.insert("Ternary-Bonsai-1.7B") }
        } else {
            if gb >= 8 { band.insert("Qwen3.5-4B") }
            if gb >= 16 { band.insert("Qwen3.5-9B") }
            if let one = qwen38(gb) { band.insert(one) }
            band.insert("Ternary-Bonsai-27B")
            band.insert("Ternary-Bonsai-1.7B")
            band.insert("gemma-4-E4B")
            if gb >= 16 { band.insert("gemma-4-12B-MTP") }
        }
        let keep = band.union(downloaded)
        return every.filter { name in keep.contains(name) }
    }

    public static func offered(unlocked: Bool) -> [String] {
        unlocked ? every : all
    }

    public static var supported: Bool { !all.isEmpty }

    public static let fallback = "Ternary-Bonsai-1.7B"

    public static var start: String {
        let list = all
        var out = list.contains(fallback) ? fallback : (list.first ?? fallback)
        let preferred = [e2b, e4b, isOS ? gemmaRung(installedGB) : nil]
        for name in preferred.compactMap({ n in n }) where list.contains(name) {
            out = name
        }
        return out
    }

    private static let e2b = "gemma-4-E2B"
    private static let e4b = "gemma-4-E4B"

    private static func family(_ name: String) -> String {
        var out = name
        if name.hasPrefix("Qwen3.8-27B") { out = "Qwen3.8-27B" }
        if name.hasSuffix(mtp) { out = String(name.dropLast(mtp.count)) }
        return out
    }

    private static let mtp = "-MTP"

    private static func variant(_ name: String) -> String? {
        let out: String?
        switch name {
            case "Qwen3.8-27B-IQ1_S": out = "1-bit"
            case "Qwen3.8-27B-IQ2_XXS": out = "2-bit"
            case "Qwen3.8-27B-IQ3_XXS": out = "3-bit"
            case "Qwen3.8-27B-IQ4_XS": out = "4-bit IQ"
            case "Qwen3.8-27B-Q4_K_S": out = "4-bit K"
            default: out = name.hasSuffix(mtp) ? "MTP" : nil
        }
        return out
    }

    public static func display(_ name: String) -> String {
        var out = name
        switch name {
            case "Qwen3.5-4B": out = "Qwen3.5 4B"
            case "Qwen3.5-9B": out = "Qwen3.5 9B"
            case "Qwen3.8-27B-IQ1_S", "Qwen3.8-27B-IQ2_XXS",
                 "Qwen3.8-27B-IQ3_XXS", "Qwen3.8-27B-IQ4_XS",
                 "Qwen3.8-27B-Q4_K_S": out = "Qwen3.8 27B"
            case "Ternary-Bonsai-27B": out = "Bonsai 27B"
            case "Ternary-Bonsai-1.7B": out = "Bonsai 1.7B"
            case "gemma-4-E2B", "gemma-4-E2B-MTP": out = "Gemma E2B"
            case "gemma-4-E4B", "gemma-4-E4B-MTP": out = "Gemma E4B"
            case "gemma-4-12B", "gemma-4-12B-MTP": out = "Gemma 12B"
            default: out = name
        }
        return out
    }

    public static func qualified(_ name: String) -> String {
        var out = display(name)
        if let tag = variant(name) { out += " " + tag }
        return out
    }

    public static func display(_ name: String,
                               among visible: [String]) -> String {
        let kin = visible.filter { other in
            family(other) == family(name)
        }
        return kin.count > 1 ? qualified(name) : display(name)
    }

}
