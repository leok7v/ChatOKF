import Foundation
import os

public final class Diag: @unchecked Sendable {

    public static let shared = Diag()

    private static let retention: TimeInterval = 24 * 3600

    private let log = Logger(subsystem: "io.github.leok7v.ChatOKF",
                             category: "diag")
    private let lock = NSLock()
    private var handle: FileHandle?
    private var filePath = ""
    private let stamp: DateFormatter
    public var path: String {
        lock.lock()
        defer { lock.unlock() }
        return filePath
    }

    private init() {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss.SSS"
        stamp = f
    }

    static let echoes = Bundle.main.bundleIdentifier == nil || DiagGate.debug

    private func opened() -> FileHandle? {
        if handle == nil {
            let url = Diag.startRun(folder: "diag.log")
            FileManager.default.createFile(atPath: url.path, contents: Data())
            handle = try? FileHandle(forWritingTo: url)
            filePath = url.path
        }
        return handle
    }

    nonisolated(unsafe)
    public static var memory: (@Sendable (String) -> Void)?

    nonisolated(unsafe)
    public static var memoryDetail: (@Sendable (String) -> Void)?

    public func report(_ s: String, file: String = #fileID,
                       line: Int = #line) {
        report(.fault, s, file: file, line: line)
    }

    public func report(_ gate: DiagGate, _ s: String,
                       file: String = #fileID, line: Int = #line) {
        let show = gate.on
        let name = file.split(separator: "/").last.map(String.init) ?? file
        let out = "\(stamp.string(from: Date())) \(name):\(line) \(s)"
        if show { log.notice("\(s, privacy: .public)") }
        // write(contentsOf:), not write(Data:): the latter raises an
        // uncatchable ObjC exception on iOS.
        if show && Diag.echoes {
            try? FileHandle.standardError.write(
                contentsOf: Data("\(out)\n".utf8))
        }
        if show && DiagGate.debug {
            lock.lock()
            try? opened()?.write(contentsOf: Data("\(out)\n".utf8))
            lock.unlock()
        }
    }

    public static func eraseCaches() {
        let fm = FileManager.default
        let root = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
        try? fm.removeItem(at: root)
    }

    public static func startRun(folder: String) -> URL {
        let fm = FileManager.default
        let dir = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app")
            .appendingPathComponent(folder, isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let current = dir.appendingPathComponent("current.txt")
        if fm.fileExists(atPath: current.path) {
            let mtime = (try? current.resourceValues(
                forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? Date()
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            let archive = dir.appendingPathComponent(
                "\(f.string(from: mtime)).txt")
            try? fm.moveItem(at: current, to: archive)
        }
        prune(dir)
        return current
    }

    private static func prune(_ dir: URL) {
        let fm = FileManager.default
        let now = Date()
        let files = (try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for f in files where f.lastPathComponent != "current.txt" {
            let mtime = (try? f.resourceValues(
                forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? now
            if now.timeIntervalSince(mtime) > retention {
                try? fm.removeItem(at: f)
            }
        }
    }
}
