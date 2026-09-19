import Foundation
import LLM

public enum ThermalBucket: String, Codable, Sendable {
    case cool
    case hot

    public static var current: ThermalBucket {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return .cool
        default: return .hot
        }
    }
}

public struct MTPSample: Sendable {
    public let model: String
    public let revision: String
    public let drafts: Int
    public let bucket: ThermalBucket
    public let tokens: Int
    public let seconds: Double
    public let accepted: Double
    public let gpuSeconds: Double
    public let wallSeconds: Double
    public let gpuRate: Double

    public init(model: String, revision: String, drafts: Int,
                bucket: ThermalBucket, tokens: Int, seconds: Double,
                accepted: Double, gpuSeconds: Double = 0,
                wallSeconds: Double = 0, gpuRate: Double = 0) {
        self.model = model
        self.revision = revision
        self.drafts = drafts
        self.bucket = bucket
        self.tokens = tokens
        self.seconds = seconds
        self.accepted = accepted
        self.gpuSeconds = gpuSeconds
        self.wallSeconds = wallSeconds
        self.gpuRate = gpuRate
    }

    public var rate: Double {
        seconds > 0 ? Double(tokens) / seconds : 0
    }

    public var busy: Double {
        wallSeconds > 0 && gpuSeconds > 0
            ? min(1, gpuSeconds / wallSeconds) : 0
    }
}

@MainActor public final class MTPTuning {

    struct Entry: Codable {
        var samples: Int
        var best: Double
        var bestGPU: Double
        var last: Double
        var accepted: Double
        var busy: Double
        var at: Date
    }

    public static let shared = MTPTuning()

    static let minimumTokens = 64
    static let decay = 0.97
    static let schema = 2

    private var table: [String: Entry] = [:]
    private let url: URL

    init(root: URL? = nil) {
        let fm = FileManager.default
        let base = root ?? (try? fm.url(for: .applicationSupportDirectory,
                                        in: .userDomainMask,
                                        appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        url = base.appendingPathComponent("mtp-tuning.json")
        table = MTPTuning.read(url)
    }

    static func key(_ s: MTPSample) -> String {
        "\(schema)/\(s.model)@\(s.revision.prefix(8))/"
            + "\(s.bucket.rawValue)/\(s.drafts)"
    }

    private static func read(_ url: URL) -> [String: Entry] {
        let data = try? Data(contentsOf: url)
        return data.flatMap { bytes in
            try? JSONDecoder().decode([String: Entry].self, from: bytes)
        } ?? [:]
    }

    private func write() {
        if let data = try? JSONEncoder().encode(table) {
            try? data.write(to: url, options: .atomic)
        }
    }

    public var entries: Int { table.count }

    private func entry(_ model: String, _ revision: String,
                       _ bucket: ThermalBucket, _ drafts: Int) -> Entry? {
        let probe = MTPSample(model: model, revision: revision,
                              drafts: drafts, bucket: bucket, tokens: 0,
                              seconds: 0, accepted: 0)
        return table[MTPTuning.key(probe)]
    }

    public func rate(_ model: String, _ revision: String,
                     _ bucket: ThermalBucket, drafts: Int) -> Double {
        entry(model, revision, bucket, drafts)?.best ?? 0
    }

    public func gpuRate(_ model: String, _ revision: String,
                        _ bucket: ThermalBucket, drafts: Int) -> Double {
        entry(model, revision, bucket, drafts)?.bestGPU ?? 0
    }

    public func busy(_ model: String, _ revision: String,
                     _ bucket: ThermalBucket, drafts: Int) -> Double {
        entry(model, revision, bucket, drafts)?.busy ?? 0
    }

    @discardableResult
    public func fold(_ s: MTPSample) -> Bool {
        var kept = false
        if s.tokens >= MTPTuning.minimumTokens, s.rate > 0 {
            let k = MTPTuning.key(s)
            var e = table[k] ?? Entry(samples: 0, best: 0, bestGPU: 0,
                                      last: 0, accepted: 0, busy: 0,
                                      at: Date())
            e.samples += 1
            e.best = max(e.best * MTPTuning.decay, s.rate)
            e.bestGPU = max(e.bestGPU * MTPTuning.decay, s.gpuRate)
            e.last = s.rate
            e.accepted = e.accepted == 0
                ? s.accepted : e.accepted * 0.8 + s.accepted * 0.2
            e.busy = e.busy == 0 ? s.busy : e.busy * 0.8 + s.busy * 0.2
            e.at = Date()
            table[k] = e
            write()
            kept = true
        }
        Diag.shared.report(.turn, String(
            format: "[mtp] n=%d %@ %.1f t/s (%.1f gpu) over %d tok, "
                + "accept %.0f%%, gpu %.2fs of %.2fs (%.0f%%), %@",
            s.drafts, s.bucket.rawValue, s.rate, s.gpuRate, s.tokens,
            s.accepted * 100, s.gpuSeconds, s.wallSeconds, s.busy * 100,
            kept ? "kept" : "too short to keep"))
        return kept
    }

    public func forget() {
        table = [:]
        try? FileManager.default.removeItem(at: url)
    }

}
