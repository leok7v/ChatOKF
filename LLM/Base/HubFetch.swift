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
    static let lead = 2 * lanes

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
            let had = committed(part)
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
                attempt = committed(part) > had ? 1 : attempt + 1
                Diag.shared.report(.net, "fetch \(e.path) attempt \(attempt) "
                    + "at \(committed(part)): "
                    + "\(last.map { err in "\(err)" } ?? "")")
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
                     _ onBytes: @escaping @Sendable (Int64) -> Void,
                     span: Int64 = HubFetch.span)
        async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: part.path) {
            fm.createFile(atPath: part.path, contents: nil)
        }
        let have = try assemble(part)
        if have < e.size {
            try reserve(part, e.size)
            let ledger = Ledger(part, have, e.size, span, onBytes)
            try await withThrowingTaskGroup(of: Void.self) { group in
                for lane in 0..<lanes {
                    group.addTask {
                        try await run(lane, url, pump, ledger)
                    }
                }
                for try await _ in group {}
            }
        }
    }

    static func run(_ lane: Int, _ url: URL, _ pump: Pump,
                    _ ledger: Ledger) async throws {
        while let from = try await ledger.take() {
            let to = min(from + ledger.span, ledger.size) - 1
            var req = URLRequest(url: url)
            req.setValue("bytes=\(from)-\(to)", forHTTPHeaderField: "Range")
            let n = try await pump.body(req, lane: lane, into: ledger.part,
                                        at: from) { n in
                ledger.live(lane, n)
            }
            if n != to - from + 1 {
                throw HubError.http(0, url.absoluteString)
            }
            try ledger.land(lane, from, n)
        }
    }

    static func reserve(_ part: URL, _ size: Int64) throws {
        let h = try FileHandle(forWritingTo: part)
        let end = Int64(try h.seekToEnd())
        if end < size {
            var store = fstore_t(fst_flags: UInt32(F_ALLOCATEALL),
                                 fst_posmode: F_PEOFPOSMODE, fst_offset: 0,
                                 fst_length: off_t(size - end),
                                 fst_bytesalloc: 0)
            _ = fcntl(h.fileDescriptor, F_PREALLOCATE, &store)
            try h.truncate(atOffset: UInt64(size))
        }
        try h.close()
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

    static func mark(_ part: URL) -> URL {
        part.deletingLastPathComponent()
            .appendingPathComponent(part.lastPathComponent + ".have")
    }

    static func committed(_ part: URL) -> Int64 {
        let text = try? String(contentsOf: mark(part), encoding: .utf8)
        let have = text.flatMap { s in Int64(s) } ?? Int64.max
        return min(have, size(part))
    }

    static func commit(_ part: URL, _ have: Int64) throws {
        try Data(String(have).utf8).write(to: mark(part), options: .atomic)
    }

    static func splice(_ src: URL, into dst: URL,
                       at offset: Int64) throws -> Int64 {
        let r = try FileHandle(forReadingFrom: src)
        let w = try FileHandle(forWritingTo: dst)
        try w.seek(toOffset: UInt64(offset))
        var end = offset
        var more = true
        while more {
            try autoreleasepool {
                let c = try r.read(upToCount: chunk)
                if let c, !c.isEmpty {
                    try w.write(contentsOf: c)
                    end += Int64(c.count)
                } else {
                    more = false
                }
            }
        }
        try r.close()
        try w.close()
        return end
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

final class Ledger: @unchecked Sendable {
    let part: URL
    let size: Int64
    let span: Int64
    private let lock = NSLock()
    private let onBytes: @Sendable (Int64) -> Void
    private var have: Int64
    private var next: Int64
    private var landed: [Int64: Int64] = [:]
    private var live: [Int: Int64] = [:]

    init(_ part: URL, _ have: Int64, _ size: Int64, _ span: Int64,
         _ onBytes: @escaping @Sendable (Int64) -> Void) {
        self.part = part
        self.size = size
        self.span = span
        self.onBytes = onBytes
        self.have = have
        self.next = have
    }

    private enum Offer {
        case piece(Int64)
        case wait
        case done
    }

    private func offer() -> Offer {
        lock.lock()
        defer { lock.unlock() }
        var out = Offer.wait
        if next >= size || BackgroundGate.shared.parked {
            out = .done
        } else if next < have + span * Int64(HubFetch.lead) {
            out = .piece(next)
            next += span
        }
        return out
    }

    func take() async throws -> Int64? {
        var out: Int64? = nil
        var wait = true
        while wait {
            try Task.checkCancellation()
            switch offer() {
            case .piece(let at):
                out = at
                wait = false
            case .done:
                wait = false
            case .wait:
                try await Task.sleep(for: .milliseconds(100))
            }
        }
        return out
    }

    func live(_ lane: Int, _ n: Int64) {
        lock.lock()
        live[lane] = n
        let sum = total()
        lock.unlock()
        onBytes(sum)
    }

    func land(_ lane: Int, _ from: Int64, _ n: Int64) throws {
        let sum = try record(lane, from, n)
        onBytes(sum)
    }

    private func record(_ lane: Int, _ from: Int64,
                        _ n: Int64) throws -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        live[lane] = 0
        landed[from] = n
        let before = have
        while let len = landed[have] {
            landed[have] = nil
            have += len
        }
        if have > before { try HubFetch.commit(part, have) }
        return total()
    }

    private func total() -> Int64 {
        let ahead = landed.values.reduce(Int64(0)) { acc, v in acc + v }
        let flowing = live.values.reduce(Int64(0)) { acc, v in acc + v }
        return have + ahead + flowing
    }
}

final class Pump: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private final class Waiter {
        let cont: CheckedContinuation<Int64, Error>
        let sink: FileHandle
        let onBytes: @Sendable (Int64) -> Void
        var written: Int64 = 0

        init(_ cont: CheckedContinuation<Int64, Error>, _ sink: FileHandle,
             _ onBytes: @escaping @Sendable (Int64) -> Void) {
            self.cont = cont
            self.sink = sink
            self.onBytes = onBytes
        }
    }

    static let recycle = 8

    private let configuration: URLSessionConfiguration
    private let lock = NSLock()
    private var waiters: [Int: Waiter] = [:]
    private var live: [Int: URLSession] = [:]
    private var served: [Int: Int] = [:]
    private var seq = 0

    init(configuration: URLSessionConfiguration = .default) {
        self.configuration = configuration
        super.init()
    }

    private func session(_ lane: Int) -> URLSession {
        lock.lock()
        let n = (served[lane] ?? 0) + 1
        var s = live[lane]
        if n > Pump.recycle {
            s?.finishTasksAndInvalidate()
            s = nil
        }
        let out = s ?? URLSession(configuration: configuration,
                                  delegate: self, delegateQueue: nil)
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

    private func settle(_ task: URLSessionTask, _ error: Error?) {
        if let w = take(task) {
            try? w.sink.close()
            if let error {
                w.cont.resume(throwing: error)
            } else {
                w.cont.resume(returning: w.written)
            }
        }
    }

    static func token(_ task: URLSessionTask) -> Int {
        Int(task.taskDescription ?? "") ?? -1
    }

    func body(_ req: URLRequest, lane: Int, into part: URL, at offset: Int64,
              _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws -> Int64 {
        let sink = try FileHandle(forWritingTo: part)
        try sink.seek(toOffset: UInt64(offset))
        return try await withCheckedThrowingContinuation { cont in
            let s = session(lane)
            let task = s.dataTask(with: req)
            lock.lock()
            seq += 1
            let key = seq
            waiters[key] = Waiter(cont, sink, onBytes)
            lock.unlock()
            task.taskDescription = String(key)
            task.resume()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable
                        (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        let ok = code == 200 || code == 206
        if !ok {
            settle(dataTask, HubError.http(
                code, dataTask.originalRequest?.url?.absoluteString ?? ""))
        }
        completionHandler(ok ? .allow : .cancel)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        lock.lock()
        let w = waiters[Pump.token(dataTask)]
        lock.unlock()
        if let w {
            do {
                try w.sink.write(contentsOf: data)
                w.written += Int64(data.count)
                w.onBytes(w.written)
            } catch {
                settle(dataTask, error)
                dataTask.cancel()
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        settle(task, error)
    }
}
