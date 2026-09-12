import Foundation

public final class Relay: NSObject, URLSessionDownloadDelegate,
                          @unchecked Sendable {
    public static let shared = Relay()
    public static let identifier = "io.github.leok7v.ChatOKF.download"
    nonisolated(unsafe) public static var idle: (@Sendable () -> Void)?

    private let lock = NSLock()
    private var live: [String: Int64] = [:]

    private lazy var session: URLSession = {
        let c = URLSessionConfiguration.background(
            withIdentifier: Relay.identifier)
        c.isDiscretionary = false
        c.sessionSendsLaunchEvents = true
        return URLSession(configuration: c, delegate: self,
                          delegateQueue: nil)
    }()

    public func start() {
        _ = session
        Task { await dropStale() }
    }

    private func clearLive() {
        lock.lock()
        live.removeAll()
        lock.unlock()
    }

    func cancelAll() async {
        for task in await session.allTasks {
            task.cancel()
        }
        clearLive()
        Diag.shared.report(.net, "background tasks cancelled for foreground")
    }

    func dropStale() async {
        let fm = FileManager.default
        for task in await session.allTasks {
            let dst = Relay.resolve(task.taskDescription)
            let dir = dst?.deletingLastPathComponent().path
            let live = dir.map { d in fm.fileExists(atPath: d) } ?? false
            if !live {
                task.cancel()
                Diag.shared.report(.net, "cancelled stale "
                    + "\(task.taskDescription ?? "?")")
            }
        }
    }

    static func base() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory,
                                 in: .userDomainMask)[0]
    }

    static func stamp(_ dst: URL) -> String {
        let root = base().path
        var out = dst.path
        if out.hasPrefix(root + "/") {
            out = String(out.dropFirst(root.count + 1))
        }
        return out
    }

    static func resolve(_ stamp: String?) -> URL? {
        var out: URL? = nil
        if let stamp, !stamp.isEmpty {
            out = stamp.hasPrefix("/")
                ? URL(fileURLWithPath: stamp)
                : base().appendingPathComponent(stamp)
        }
        return out
    }

    func enqueue(_ url: URL, _ dst: URL, _ from: Int64, _ to: Int64) {
        var req = URLRequest(url: url)
        req.setValue("bytes=\(from)-\(to)", forHTTPHeaderField: "Range")
        let task = session.downloadTask(with: req)
        task.taskDescription = Relay.stamp(dst)
        lock.lock()
        live[Relay.stamp(dst)] = 0
        lock.unlock()
        task.resume()
        Diag.shared.report(.net, "enqueue \(dst.lastPathComponent)")
    }

    func busy() async -> Set<Int64> {
        var out: Set<Int64> = []
        for task in await session.allTasks {
            if let at = Relay.offset(task.taskDescription) {
                out.insert(at)
            }
        }
        return out
    }

    func state() async -> String {
        var run = 0
        var idle = 0
        var done = 0
        for task in await session.allTasks {
            if task.state == .running {
                run += 1
            } else if task.state == .suspended {
                idle += 1
            } else {
                done += 1
            }
        }
        return "running \(run) suspended \(idle) other \(done) "
            + "inflight \(inflight()) bytes"
    }

    func inflight() -> Int64 {
        lock.lock()
        let sum = live.values.reduce(Int64(0)) { acc, v in acc + v }
        lock.unlock()
        return sum
    }

    static func offset(_ path: String?) -> Int64? {
        var out: Int64? = nil
        if let tail = path?.split(separator: ".").last {
            out = Int64(tail)
        }
        return out
    }

    private func retire(_ task: URLSessionTask) {
        if let path = task.taskDescription {
            lock.lock()
            live[path] = nil
            lock.unlock()
        }
    }

    public func urlSession(_ session: URLSession,
                           downloadTask: URLSessionDownloadTask,
                           didFinishDownloadingTo location: URL) {
        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        let dst = Relay.resolve(downloadTask.taskDescription)
        if let dst, code == 200 || code == 206 {
            let fm = FileManager.default
            try? fm.removeItem(at: dst)
            do {
                try fm.moveItem(at: location, to: dst)
                Diag.shared.report(.net, "landed \(dst.lastPathComponent)")
            } catch {
                Diag.shared.report("landed BUT MOVE FAILED "
                    + "\(dst.lastPathComponent): \(error)")
            }
        } else {
            Diag.shared.report(.net, "piece failed code \(code)")
        }
        retire(downloadTask)
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask,
                           didCompleteWithError error: Error?) {
        let code = (error as? URLError)?.code
        if let error, code != .cancelled {
            Diag.shared.report(.net, "piece error "
                + error.localizedDescription)
        }
        retire(task)
    }

    public func urlSession(_ session: URLSession,
                           downloadTask: URLSessionDownloadTask,
                           didWriteData bytesWritten: Int64,
                           totalBytesWritten: Int64,
                           totalBytesExpectedToWrite: Int64) {
        if let path = downloadTask.taskDescription {
            lock.lock()
            live[path] = totalBytesWritten
            lock.unlock()
        }
    }

    public func urlSessionDidFinishEvents(
        forBackgroundURLSession session: URLSession) {
        Diag.shared.report(.net, "background session idle")
        Relay.idle?()
    }
}

extension HubFetch {
    static let window = 16

    static func piece(_ part: URL, _ at: Int64) -> URL {
        part.deletingLastPathComponent()
            .appendingPathComponent(part.lastPathComponent + ".\(at)")
    }

    static func assemble(_ part: URL) throws -> Int64 {
        let fm = FileManager.default
        var have = size(part)
        var more = true
        while more {
            let next = piece(part, have)
            if fm.fileExists(atPath: next.path) {
                try append(next, to: part)
                try? fm.removeItem(at: next)
                have = size(part)
                Diag.shared.report(.net, "assembled to \(have)")
            } else {
                more = false
            }
        }
        return have
    }

    static func carry(_ url: URL, _ e: Entry, _ part: URL, _ pump: Pump,
                      _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws {
        var have = size(part)
        while have < e.size {
            if BackgroundGate.shared.parked {
                try await drain(url, e, part, onBytes)
            } else {
                Diag.shared.report(.net, "foreground lanes take over")
                try await fill(url, e, part, pump, onBytes)
            }
            have = try assemble(part)
        }
    }

    static func drain(_ url: URL, _ e: Entry, _ part: URL,
                      _ onBytes: @escaping @Sendable (Int64) -> Void,
                      parked: @escaping @Sendable () -> Bool
                          = { BackgroundGate.shared.parked })
        async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: part.path) {
            fm.createFile(atPath: part.path, contents: nil)
        }
        let relay = Relay.shared
        var have = try assemble(part)
        var tick = 0
        var tries: [Int64: Int] = [:]
        while have < e.size && parked() {
            let busy = await relay.busy()
            var at = have
            var added = 0
            while at < e.size && busy.count + added < window {
                let already = busy.contains(at)
                    || fm.fileExists(atPath: piece(part, at).path)
                if !already {
                    let n = (tries[at] ?? 0) + 1
                    tries[at] = n
                    if n > 3 {
                        throw HubError.http(0, url.absoluteString)
                    }
                    relay.enqueue(url, piece(part, at), at,
                                  min(at + span, e.size) - 1)
                    added += 1
                }
                at += span
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
            have = try assemble(part)
            onBytes(have + relay.inflight())
            tick += 1
            if !parked() {
                await relay.cancelAll()
            }
            if tick % 10 == 0 {
                Diag.shared.report(.net, "drain \(have)/\(e.size) "
                    + (await relay.state()))
            }
        }
    }
}
