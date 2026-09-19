import CryptoKit
import Foundation

public enum HubError: Error {
    case http(Int, String)
    case malformed(String)
    case notFound(String)
    case digest(String)
}

public struct HubFetch: Sendable {
    public struct Status: Sendable {
        public let file: String
        public let done: Int64
        public let total: Int64
    }

    struct Entry: Sendable {
        let path: String
        let size: Int64
        let oid: String
        let lfs: Bool
    }

    static let host = "https://huggingface.co"
    static let sentinel = ".complete"
    static let retries = 8
    static let backoff = 30
    static let chunk = 1 << 20
    static let span: Int64 = 32 << 20
    static let lanes = 4

    public static func fetch(
        repo: String,
        prefix: String = "",
        into dest: URL,
        revision: String = "main",
        files: [String]? = nil,
        excludeFromBackup: Bool = false,
        background: Bool = false,
        report: @escaping @Sendable (Status) -> Void = { _ in }
    ) async throws -> URL {
        let fm = FileManager.default
        let sha = try await commit(repo, revision)
        let root = dest.appendingPathComponent(sha)
        let mark = root.appendingPathComponent(sentinel)
        if !fm.fileExists(atPath: mark.path) {
            let all = try await tree(repo, sha)
            let want: [Entry]
            if let files {
                want = all.filter { e in files.contains(e.path) }
            } else if prefix.isEmpty {
                want = all
            } else {
                want = all.filter { e in e.path.hasPrefix(prefix + "/") }
            }
            if want.isEmpty {
                throw HubError.notFound(files?.joined(separator: ",") ?? prefix)
            }
            try await install(repo, sha, want, root, background, report)
            fm.createFile(atPath: mark.path, contents: nil)
            prune(dest, keep: sha)
        }
        if excludeFromBackup {
            exclude(fromBackup: root)
        }
        return prefix.isEmpty ? root : root.appendingPathComponent(prefix)
    }

    static func prune(_ dest: URL, keep sha: String) {
        let fm = FileManager.default
        let kids = (try? fm.contentsOfDirectory(at: dest,
            includingPropertiesForKeys: nil)) ?? []
        for k in kids where k.lastPathComponent != sha {
            try? fm.removeItem(at: k)
        }
    }

    static func exclude(fromBackup url: URL) {
        var u = url
        var v = URLResourceValues()
        v.isExcludedFromBackup = true
        try? u.setResourceValues(v)
    }

    static func commit(_ repo: String, _ rev: String) async throws -> String {
        let url = "\(host)/api/models/\(repo)/revision/\(esc(rev))"
        let (data, _) = try await page(url)
        let obj = try? JSONSerialization.jsonObject(with: data)
        let dict = obj as? [String: Any]
        return try need(dict?["sha"] as? String, url)
    }

    static func tree(_ repo: String, _ sha: String) async throws -> [Entry] {
        var next: String? = "\(host)/api/models/\(repo)/tree/\(sha)"
            + "?recursive=true"
        var out: [Entry] = []
        while let url = next {
            let (data, link) = try await page(url)
            let obj = try? JSONSerialization.jsonObject(with: data)
            let arr = try need(obj as? [Any], url)
            for any in arr {
                if let e = entry(any) {
                    out.append(e)
                }
            }
            next = nextLink(link)
        }
        return out
    }

    static func entry(_ any: Any) -> Entry? {
        var out: Entry? = nil
        if let d = any as? [String: Any],
           d["type"] as? String == "file",
           let path = d["path"] as? String {
            let size = (d["size"] as? NSNumber)?.int64Value ?? 0
            let lfs = d["lfs"] as? [String: Any]
            let oid = (lfs?["oid"] as? String) ?? (d["oid"] as? String)
            if let oid {
                out = Entry(path: path, size: size, oid: oid, lfs: lfs != nil)
            }
        }
        return out
    }

    static func install(
        _ repo: String,
        _ sha: String,
        _ want: [Entry],
        _ root: URL,
        _ background: Bool,
        _ report: @escaping @Sendable (Status) -> Void
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let bytesTotal = want.reduce(Int64(0)) { $0 + $1.size }
        let meter = ByteMeter(total: bytesTotal, report: report)
        let pump = Pump()
        defer { pump.done() }
        for e in want {
            let dst = root.appendingPathComponent(e.path)
            try fm.createDirectory(at: dst.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            meter.setFile(e.path)
            let kept = fm.fileExists(atPath: dst.path)
                && (try? verify(dst, e)) != nil
            if !kept {
                try await pull(repo, sha, e, dst, pump,
                               background) { written in
                    meter.live(written)
                }
            }
            meter.complete(e.size)
        }
    }

    static func pull(
        _ repo: String, _ sha: String, _ e: Entry, _ dst: URL, _ pump: Pump,
        _ background: Bool,
        _ onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let fm = FileManager.default
        let raw = "\(host)/\(repo)/resolve/\(sha)/\(esc(e.path))"
        let url = try need(URL(string: raw), raw)
        let part = dst.appendingPathExtension("part")
        var attempt = 0
        var verified: URL? = nil
        var last: Error? = nil
        while attempt < retries && verified == nil {
            let had = size(part)
            do {
                Diag.memoryDetail?("fetch start \(e.path)")
                if background {
                    try await carry(url, e, part, pump, onBytes)
                } else {
                    try await fill(url, e, part, pump, onBytes)
                }
                Diag.memoryDetail?("fetch body \(e.path)")
                do {
                    try verify(part, e)
                    Diag.memoryDetail?("fetch verified \(e.path)")
                    verified = part
                } catch {
                    try? fm.removeItem(at: part)
                    sweep(part)
                    last = error
                }
            } catch {
                last = error
            }
            if verified == nil {
                attempt = size(part) > had ? 1 : attempt + 1
                Diag.shared.report(.net, "fetch \(e.path) attempt \(attempt) "
                    + "at \(size(part)): \(last.map { err in "\(err)" } ?? "")")
                if attempt < retries {
                    try await Task.sleep(
                        for: .seconds(min(backoff, 1 << (attempt - 1))))
                }
            }
        }
        let file = try need(verified, e.path, last)
        try? fm.removeItem(at: dst)
        try fm.moveItem(at: file, to: dst)
        sweep(part)
        Diag.memoryDetail?("fetch moved \(e.path)")
    }

    static func fill(_ url: URL, _ e: Entry, _ part: URL, _ pump: Pump,
                     _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: part.path) {
            fm.createFile(atPath: part.path, contents: nil)
        }
        var have = try assemble(part)
        while have < e.size && !BackgroundGate.shared.parked {
            try Task.checkCancellation()
            let wave = Wave(base: have, onBytes: onBytes)
            var pieces: [(Int, URL)] = []
            try await withThrowingTaskGroup(of: (Int, URL).self) { group in
                var lane = 0
                var at = have
                while lane < lanes && at < e.size {
                    let from = at
                    let to = min(from + span, e.size) - 1
                    let idx = lane
                    group.addTask {
                        var req = URLRequest(url: url)
                        req.setValue("bytes=\(from)-\(to)",
                                     forHTTPHeaderField: "Range")
                        let u = try await pump.body(req, lane: idx) { n in
                            wave.live(idx, n)
                        }
                        return (idx, u)
                    }
                    at = to + 1
                    lane += 1
                }
                for try await got in group {
                    pieces.append(got)
                }
            }
            pieces.sort { a, b in a.0 < b.0 }
            for piece in pieces {
                try append(piece.1, to: part)
                try? fm.removeItem(at: piece.1)
            }
            let now = try assemble(part)
            if now <= have {
                throw HubError.http(0, url.absoluteString)
            }
            have = now
            onBytes(have)
        }
    }

    static func sweep(_ part: URL) {
        let fm = FileManager.default
        let dir = part.deletingLastPathComponent()
        let stem = part.lastPathComponent + "."
        let kids = (try? fm.contentsOfDirectory(at: dir,
            includingPropertiesForKeys: nil)) ?? []
        for k in kids where k.lastPathComponent.hasPrefix(stem) {
            try? fm.removeItem(at: k)
        }
    }

    static func size(_ url: URL) -> Int64 {
        let a = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (a?[.size] as? NSNumber)?.int64Value ?? 0
    }

    static func append(_ src: URL, to dst: URL) throws {
        let r = try FileHandle(forReadingFrom: src)
        let w = try FileHandle(forWritingTo: dst)
        try w.seekToEnd()
        var more = true
        while more {
            try autoreleasepool {
                let c = try r.read(upToCount: chunk)
                if let c, !c.isEmpty {
                    try w.write(contentsOf: c)
                } else {
                    more = false
                }
            }
        }
        try r.close()
        try w.close()
    }

    static func verify(_ file: URL, _ e: Entry) throws {
        let got = e.lfs ? try sha256(file) : try blobSHA1(file)
        if got != e.oid {
            throw HubError.digest(e.path)
        }
    }

    static func sha256(_ file: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: file)
        var d = SHA256()
        // FileHandle.read returns autoreleased NSData and a tight loop has no
        // pool, so without one per chunk a multi-GB file lives to the end.
        var more = true
        while more {
            try autoreleasepool {
                let c = try h.read(upToCount: chunk)
                if let c, !c.isEmpty {
                    d.update(data: c)
                } else {
                    more = false
                }
            }
        }
        try h.close()
        return hex(d.finalize())
    }

    static func blobSHA1(_ file: URL) throws -> String {
        let fm = FileManager.default
        let attrs = try fm.attributesOfItem(atPath: file.path)
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let h = try FileHandle(forReadingFrom: file)
        var d = Insecure.SHA1()
        d.update(data: Data("blob \(size)\0".utf8))
        // Same pool, same reason -- see sha256 above.
        var more = true
        while more {
            try autoreleasepool {
                let c = try h.read(upToCount: chunk)
                if let c, !c.isEmpty {
                    d.update(data: c)
                } else {
                    more = false
                }
            }
        }
        try h.close()
        return hex(d.finalize())
    }

    static func page(_ url: String) async throws -> (Data, String?) {
        let req = URLRequest(url: try need(URL(string: url), url))
        let (data, resp) = try await URLSession.shared.data(for: req)
        let http = resp as? HTTPURLResponse
        let code = http?.statusCode ?? 0
        if code != 200 {
            throw HubError.http(code, url)
        }
        return (data, http?.value(forHTTPHeaderField: "Link"))
    }

    // RFC 5988: Link: <https://...>; rel="next"
    static func nextLink(_ header: String?) -> String? {
        var out: String? = nil
        if let h = header, h.contains("rel=\"next\"") {
            let open = h.split(separator: "<", maxSplits: 1)
            if open.count == 2 {
                let shut = open[1].split(separator: ">", maxSplits: 1)
                if !shut.isEmpty {
                    out = String(shut[0])
                }
            }
        }
        return out
    }

    static func need<T>(_ v: T?, _ what: String,
                        _ cause: Error? = nil) throws -> T {
        if v == nil {
            throw cause ?? HubError.malformed(what)
        }
        return v!
    }

    static func hex<D: Sequence>(_ d: D) -> String
        where D.Element == UInt8 {
        d.map { b in String(format: "%02x", b) }.joined()
    }

    static func esc(_ s: String) -> String {
        s.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? s
    }
}

private final class ByteMeter: @unchecked Sendable {
    private let total: Int64
    private let report: @Sendable (HubFetch.Status) -> Void
    private let lock = NSLock()
    private var base: Int64 = 0
    private var last: Int64 = 0
    private var file = ""

    init(total: Int64,
         report: @escaping @Sendable (HubFetch.Status) -> Void) {
        self.total = total
        self.report = report
    }

    func setFile(_ f: String) { lock.lock(); file = f; lock.unlock() }

    func live(_ fileBytes: Int64) {
        lock.lock()
        let now = base + fileBytes
        let emit = now - last >= 1 << 20
        if emit { last = now }
        let f = file
        lock.unlock()
        if emit { report(HubFetch.Status(file: f, done: now, total: total)) }
    }

    func complete(_ size: Int64) {
        lock.lock()
        base += size
        last = base
        let n = base, f = file
        lock.unlock()
        report(HubFetch.Status(file: f, done: n, total: total))
    }
}

private final class Wave: @unchecked Sendable {
    private let lock = NSLock()
    private let base: Int64
    private let onBytes: @Sendable (Int64) -> Void
    private var lanes: [Int: Int64] = [:]

    init(base: Int64, onBytes: @escaping @Sendable (Int64) -> Void) {
        self.base = base
        self.onBytes = onBytes
    }

    func live(_ lane: Int, _ n: Int64) {
        lock.lock()
        lanes[lane] = n
        let sum = lanes.values.reduce(Int64(0)) { acc, v in acc + v }
        lock.unlock()
        onBytes(base + sum)
    }
}

final class Pump: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private struct Waiter {
        let cont: CheckedContinuation<URL, Error>
        let onBytes: @Sendable (Int64) -> Void
    }

    static let recycle = 8

    private let lock = NSLock()
    private var waiters: [Int: Waiter] = [:]
    private var live: [Int: URLSession] = [:]
    private var served: [Int: Int] = [:]
    private var seq = 0

    private func session(_ lane: Int) -> URLSession {
        lock.lock()
        let n = (served[lane] ?? 0) + 1
        var s = live[lane]
        if n > Pump.recycle {
            s?.finishTasksAndInvalidate()
            s = nil
        }
        let out = s ?? URLSession(configuration: .default, delegate: self,
                                  delegateQueue: nil)
        served[lane] = s == nil ? 1 : n
        live[lane] = out
        lock.unlock()
        return out
    }

    func done() {
        lock.lock()
        let all = Array(live.values)
        live.removeAll()
        served.removeAll()
        lock.unlock()
        for s in all {
            s.finishTasksAndInvalidate()
        }
    }

    private func take(_ task: URLSessionTask) -> Waiter? {
        lock.lock()
        let w = waiters.removeValue(forKey: Pump.token(task))
        lock.unlock()
        return w
    }

    static func token(_ task: URLSessionTask) -> Int {
        Int(task.taskDescription ?? "") ?? -1
    }

    func body(_ req: URLRequest, lane: Int,
              _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            let s = session(lane)
            let task = s.downloadTask(with: req)
            lock.lock()
            seq += 1
            let key = seq
            waiters[key] = Waiter(cont: cont, onBytes: onBytes)
            lock.unlock()
            task.taskDescription = String(key)
            task.resume()
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                ProcessInfo.processInfo.globallyUniqueString)
        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        let w = take(downloadTask)
        if code != 200 && code != 206 {
            w?.cont.resume(throwing: HubError.http(
                code, downloadTask.originalRequest?.url?.absoluteString ?? ""))
        } else {
            do {
                try FileManager.default.moveItem(at: location, to: dst)
                w?.cont.resume(returning: dst)
            } catch {
                w?.cont.resume(throwing: error)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error {
            take(task)?.cont.resume(throwing: error)
        }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        lock.lock()
        let w = waiters[Pump.token(downloadTask)]
        lock.unlock()
        w?.onBytes(totalBytesWritten)
    }
}
