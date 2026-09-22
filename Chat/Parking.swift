import Foundation
import LLM

public struct ParkBudget: Sendable {
    public let free: Int
    public let ram: Int

    public static let diskShare = 0.25
    public static let ramShare = 0.15

    public init(free: Int, ram: Int) {
        self.free = free
        self.ram = ram
    }

    public static func current(at dir: URL) -> ParkBudget {
        let values = try? dir.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let free = values?.volumeAvailableCapacityForImportantUsage ?? 0
        return ParkBudget(free: Int(free),
                          ram: Int(ProcessInfo.processInfo.physicalMemory))
    }

    public func refusal(for bytes: Int) -> String? {
        var out: String? = nil
        if Double(bytes) > Double(free) * ParkBudget.diskShare {
            out = "over a quarter of the \(free >> 20) MB free on disk"
        } else if Double(bytes) > Double(ram) * ParkBudget.ramShare {
            out = "over 15 percent of the \(ram >> 20) MB of memory"
        }
        return out
    }
}

extension Session {

    public static let parked: URL = {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask, appropriateFor: nil,
                                   create: true)) ?? fm.temporaryDirectory
        var dir = support.appendingPathComponent("parked", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? dir.setResourceValues(values)
        Platform.protectUntilFirstUnlock(dir)
        return dir
    }()

    static func liveDir(_ name: String) -> URL {
        Session.parked.appendingPathComponent(
            "live." + Session.parkStamp(name), isDirectory: true)
    }

    static func revision8(_ name: String) -> String {
        String((ModelCatalog.source(name)?.revision ?? "local").prefix(8))
    }

    static func parkStamp(_ name: String) -> String {
        name + "." + Session.revision8(name)
    }

    static func parkURL(_ id: UUID, _ name: String) -> URL {
        Session.parked.appendingPathComponent(
            id.uuidString + "." + Session.parkStamp(name), isDirectory: true)
    }

    private static func parkRateKey(_ name: String) -> String {
        "park.\(name)"
    }

    static func parkRate(_ name: String) -> Double {
        UserDefaults.standard.double(forKey: Session.parkRateKey(name))
    }

    static func recordParkRate(_ name: String,
                               _ saved: (tokens: Int, bytes: Int)?) {
        if let saved, saved.tokens > 0 {
            UserDefaults.standard.set(
                Double(saved.bytes) / Double(saved.tokens),
                forKey: Session.parkRateKey(name))
        }
    }

    private static func parkedFiles() -> [(url: URL, bytes: Int, at: Date)] {
        let keys: [URLResourceKey] = [.contentModificationDateKey,
                                      .isDirectoryKey]
        let found = (try? FileManager.default.contentsOfDirectory(
            at: Session.parked, includingPropertiesForKeys: keys)) ?? []
        return found.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]))?
                .isDirectory == true
            && UUID(uuidString: String(
                url.lastPathComponent.split(separator: ".").first ?? "")) != nil
        }.map { url in
            let v = try? url.resourceValues(forKeys: Set(keys))
            return (url, ChatSession.allocated(url),
                    v?.contentModificationDate ?? .distantPast)
        }
    }

    public func hasParked(_ id: UUID) -> Bool {
        FileManager.default.fileExists(
            atPath: Session.parkURL(id, modelName).path)
    }

    public func dropParked(_ id: UUID) {
        for file in Session.parkedFiles()
        where file.url.lastPathComponent.hasPrefix(id.uuidString + ".") {
            try? FileManager.default.removeItem(at: file.url)
        }
    }

    public static func eraseParked() {
        try? FileManager.default.removeItem(at: Session.parked)
    }

    public func pruneParked(keeping ids: Set<UUID>) {
        var dropped = 0
        let all = (try? FileManager.default.contentsOfDirectory(
            at: Session.parked, includingPropertiesForKeys: nil)) ?? []
        for url in all where !url.lastPathComponent.hasPrefix("live.") {
            let head = url.lastPathComponent.split(separator: ".").first
            let id = head.flatMap { text in UUID(uuidString: String(text)) }
            let dir = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?
                .isDirectory == true
            if id == nil || !ids.contains(id!) || !dir {
                try? FileManager.default.removeItem(at: url)
                dropped += 1
            }
        }
        if dropped > 0 {
            Diag.shared.report(.load, "[park] dropped \(dropped) state(s) "
                               + "of conversations that are gone")
        }
    }

    func park(_ chat: ChatSession, as id: UUID, model name: String) async {
        let tokens = await chat.committedCount
        let live = await chat.liveDir
        let bytes = live.map { dir in ChatSession.allocated(dir) } ?? 0
        let url = Session.parkURL(id, name)
        let budget = ParkBudget.current(at: Session.parked)
        var why: String? = nil
        if tokens == 0 || live == nil {
            why = "nothing committed"
        } else {
            why = budget.refusal(for: bytes)
        }
        let label = String(id.uuidString.prefix(8))
        if let why {
            Diag.shared.report(.load, String(
                format: "[park] skipped %@ at %d tokens, %d MB: %@", label,
                tokens, bytes >> 20, why))
            await chat.detach()
            if let live { try? FileManager.default.removeItem(at: live) }
        } else {
            for other in Session.parkedFiles()
            where other.url != url && other.url != live {
                try? FileManager.default.removeItem(at: other.url)
                Diag.shared.report(.load, "[park] dropped "
                                   + other.url.lastPathComponent
                                   + ": one conversation parks")
            }
            let t0 = Date()
            do {
                try await chat.park(to: url, stamp: Session.parkStamp(name))
                let saved = await chat.lastSaved
                Session.recordParkRate(name, saved)
                Diag.shared.report(.load, String(
                    format: "[park] %@ at %d tokens, %d MB in %.2fs", label,
                    tokens, (saved?.bytes ?? 0) >> 20,
                    Date().timeIntervalSince(t0)))
            } catch {
                try? FileManager.default.removeItem(at: url)
                Diag.shared.report("[park] FAILED \(label): \(error)")
            }
        }
    }

    public func parkCurrent(_ id: UUID) async {
        if let session {
            await session.quiesce()
            await park(session, as: id, model: modelName)
        }
    }

    public func resumeParked(_ id: UUID, _ config: SessionConfig,
                             onEvent: @escaping @MainActor (TraceEvent) -> Void
    ) async -> Bool {
        var out = false
        let url = Session.parkURL(id, modelName)
        if ggufBackend != nil,
           FileManager.default.fileExists(atPath: url.path) {
            await drainMeta()
            await session?.endPriming()
            Footprint.report(.load, "resume outgoing released")
            forgetRecalled()
            await memories.awaitOpen()
            makeSession(config, onEvent: onEvent)
            if let session {
                let t0 = Date()
                out = (try? await session.resume(
                    from: url, stamp: Session.parkStamp(modelName))) == true
                let label = String(id.uuidString.prefix(8))
                if out {
                    Diag.shared.report(.load, String(
                        format: "[park] resumed %@ at %d tokens in %.2fs",
                        label, await session.committedCount,
                        Date().timeIntervalSince(t0)))
                } else {
                    Diag.shared.report("[park] resume FAILED \(label), "
                                       + "priming a fresh session instead")
                    primeSession(resetFirst: true)
                }
            }
        }
        return out
    }
}
