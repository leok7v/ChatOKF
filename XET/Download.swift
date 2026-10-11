import Foundation

final class Jobs: @unchecked Sendable {
    private let lock = NSLock()
    private let all: [Job]
    private let lead: Int64?
    private let landed: @Sendable (Int64) throws -> Void
    private var next = 0
    private var have: Int64 = 0
    private var ahead: [Int64: Int64] = [:]

    init(_ jobs: [Job], lead: Int64? = nil,
         landed: @escaping @Sendable (Int64) throws -> Void = { _ in }) {
        all = jobs
        self.lead = lead
        self.landed = landed
    }

    private enum Offer {
        case job(Job)
        case wait
        case done
    }

    private func offer() -> Offer {
        lock.lock()
        defer { lock.unlock() }
        var out = Offer.done
        if next < all.count {
            let job = all[next]
            let start = job.slices.map { s in s.offset }.min() ?? 0
            if let lead, start > have + lead {
                out = .wait
            } else {
                out = .job(job)
                next += 1
            }
        }
        return out
    }

    func take() async throws -> Job? {
        var out: Job? = nil
        var wait = true
        while wait {
            try Task.checkCancellation()
            switch offer() {
            case .job(let job):
                out = job
                wait = false
            case .done:
                wait = false
            case .wait:
                try await Task.sleep(for: .milliseconds(100))
            }
        }
        return out
    }

    func land(_ job: Job) throws {
        lock.lock()
        for s in job.slices { ahead[s.offset] = s.length }
        let before = have
        while let n = ahead[have] {
            ahead[have] = nil
            have += n
        }
        let now = have
        lock.unlock()
        if now > before { try landed(now) }
    }

    var committed: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return have
    }
}

final class Meter: @unchecked Sendable {
    private let lock = NSLock()
    private let report: @Sendable (Int64) -> Void
    private var landed: Int64 = 0
    private var live: [Int: Int64] = [:]

    init(_ report: @escaping @Sendable (Int64) -> Void) {
        self.report = report
    }

    func live(_ lane: Int, _ n: Int64) {
        lock.lock()
        live[lane] = n
        let sum = total()
        lock.unlock()
        report(sum)
    }

    func land(_ lane: Int, _ n: Int64) {
        lock.lock()
        live[lane] = 0
        landed += n
        let sum = total()
        lock.unlock()
        report(sum)
    }

    private func total() -> Int64 {
        landed + live.values.reduce(Int64(0)) { acc, v in acc + v }
    }
}

public final class Downloader: NSObject, URLSessionDataDelegate,
                               @unchecked Sendable {
    public static let lanes = 8
    public static let retries = 5
    public static let backoff = 30
    public static let recycle = 8

    private final class Lane: @unchecked Sendable {
        let cont: CheckedContinuation<Int64, Error>
        let job: Job
        let fd: Int32
        let base: Int64
        let onBytes: @Sendable (Int64) -> Void
        let stream = ChunkStream()
        var decoded: [Int64]
        var written: Int64 = 0

        init(_ cont: CheckedContinuation<Int64, Error>, _ job: Job,
             _ fd: Int32, _ base: Int64,
             _ onBytes: @escaping @Sendable (Int64) -> Void) {
            self.cont = cont
            self.job = job
            self.fd = fd
            self.base = base
            self.onBytes = onBytes
            decoded = [Int64](repeating: 0, count: job.slices.count)
        }

        func route(_ ordinal: Int, _ chunk: UnsafeRawBufferPointer) throws {
            let index = job.fetch.chunks.lowerBound + ordinal
            let count = Int64(chunk.count)
            for (i, s) in job.slices.enumerated()
            where s.chunks.contains(index) {
                let from = decoded[i]
                decoded[i] = from + count
                let lower = max(from, s.skip)
                let upper = min(from + count, s.skip + s.length)
                if lower < upper {
                    try Downloader.write(
                        fd, chunk.baseAddress! + Int(lower - from),
                        Int(upper - lower),
                        at: base + s.offset + (lower - s.skip))
                    written += upper - lower
                    onBytes(written)
                }
            }
        }

        func whole() throws {
            if stream.pending > 0 {
                throw XETError.truncated("a chunk, \(stream.pending) bytes")
            }
            for (i, s) in job.slices.enumerated()
            where decoded[i] != s.unpacked {
                throw XETError.truncated("a term: \(decoded[i]) of "
                                         + "\(s.unpacked) bytes")
            }
        }
    }

    private let configuration: URLSessionConfiguration
    private let lock = NSLock()
    private var waiters: [Int: Lane] = [:]
    private var live: [Int: URLSession] = [:]
    private var served: [Int: Int] = [:]
    private var seq = 0

    public init(configuration: URLSessionConfiguration = .ephemeral) {
        self.configuration = configuration
        super.init()
    }

    public static func download(
        _ r: Reconstruction, limit: Int64? = nil, into file: URL,
        at base: Int64 = 0, lanes: Int = Downloader.lanes,
        lead: Int64? = nil,
        configuration: URLSessionConfiguration = .ephemeral,
        progress: @escaping @Sendable (Int64) -> Void = { _ in },
        landed: @escaping @Sendable (Int64) throws -> Void = { _ in }
    ) async throws -> Int64 {
        let jobs = try r.jobs(limit: limit)
        let fd = open(file.path, O_WRONLY | O_CREAT, 0o644)
        if fd < 0 {
            throw XETError.file("\(file.path): "
                                + String(cString: strerror(errno)))
        }
        defer { close(fd) }
        let pump = Downloader(configuration: configuration)
        defer { pump.done() }
        let queue = Jobs(jobs, lead: lead, landed: landed)
        let meter = Meter(progress)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for lane in 0..<max(1, min(lanes, jobs.count)) {
                group.addTask {
                    try await pump.run(lane, queue, fd, base, meter)
                }
            }
            for try await _ in group {}
        }
        return jobs.reduce(Int64(0)) { sum, j in sum + j.length }
    }

    public static func data(_ r: Reconstruction, limit: Int64? = nil,
                            configuration: URLSessionConfiguration
                                = .ephemeral) async throws -> Data {
        let fm = FileManager.default
        let file = fm.temporaryDirectory
            .appendingPathComponent("xet-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: file) }
        fm.createFile(atPath: file.path, contents: nil)
        _ = try await download(r, limit: limit, into: file,
                               configuration: configuration)
        return try Data(contentsOf: file)
    }

    func run(_ lane: Int, _ queue: Jobs, _ fd: Int32, _ base: Int64,
             _ meter: Meter) async throws {
        while let job = try await queue.take() {
            try await pull(lane, job, fd, base) { n in meter.live(lane, n) }
            try queue.land(job)
            meter.land(lane, job.length)
        }
    }

    func pull(_ lane: Int, _ job: Job, _ fd: Int32, _ base: Int64,
              _ onBytes: @escaping @Sendable (Int64) -> Void) async throws {
        var attempt = 0
        var failure: Error? = XETError.truncated("nothing fetched")
        while failure != nil && attempt < Downloader.retries {
            do {
                try await body(job, lane: lane, fd, base, onBytes)
                failure = nil
            } catch {
                failure = error
                try Task.checkCancellation()
                attempt = Downloader.retryable(error) ? attempt + 1
                                                     : Downloader.retries
                if attempt < Downloader.retries {
                    try await Task.sleep(for: .seconds(
                        min(Downloader.backoff, 1 << (attempt - 1))))
                }
            }
        }
        if let failure { throw failure }
    }

    static func retryable(_ error: Error) -> Bool {
        var out = true
        if let xet = error as? XETError {
            switch xet {
            case .http(let code, _):
                out = code == 408 || code == 429 || code >= 500
            case .truncated:
                out = true
            default:
                out = false
            }
        } else if error is CancellationError {
            out = false
        }
        return out
    }

    static func write(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int,
                      at offset: Int64) throws {
        var done = 0
        while done < n {
            let w = pwrite(fd, p + done, n - done, off_t(offset + Int64(done)))
            if w < 0 {
                throw XETError.file(String(cString: strerror(errno)))
            }
            done += w
        }
    }

    private func session(_ lane: Int) -> URLSession {
        lock.lock()
        let n = (served[lane] ?? 0) + 1
        var s = live[lane]
        if n > Downloader.recycle {
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

    public func done() {
        lock.lock()
        let all = Array(live.values)
        live.removeAll()
        served.removeAll()
        lock.unlock()
        for s in all {
            s.finishTasksAndInvalidate()
        }
    }

    private func take(_ task: URLSessionTask) -> Lane? {
        lock.lock()
        let w = waiters.removeValue(forKey: Downloader.token(task))
        lock.unlock()
        return w
    }

    private func settle(_ task: URLSessionTask, _ error: Error?) {
        if let w = take(task) {
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

    func body(_ job: Job, lane: Int, _ fd: Int32, _ base: Int64,
              _ onBytes: @escaping @Sendable (Int64) -> Void)
        async throws {
        var req = URLRequest(url: job.fetch.url)
        req.setValue(job.fetch.range, forHTTPHeaderField: "Range")
        let s = session(lane)
        let task = s.dataTask(with: req)
        _ = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<Int64, Error>) in
                lock.lock()
                seq += 1
                let key = seq
                waiters[key] = Lane(cont, job, fd, base, onBytes)
                lock.unlock()
                task.taskDescription = String(key)
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                           didReceive response: URLResponse,
                           completionHandler: @escaping @Sendable
                               (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        lock.lock()
        let w = waiters[Downloader.token(dataTask)]
        lock.unlock()
        let whole = code == 200 && w?.job.fetch.bytes.lowerBound == 0
        let ok = code == 206 || whole
        if !ok {
            settle(dataTask, XETError.http(
                code, dataTask.originalRequest?.url?.host ?? ""))
        }
        completionHandler(ok ? .allow : .cancel)
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                           didReceive data: Data) {
        lock.lock()
        let w = waiters[Downloader.token(dataTask)]
        lock.unlock()
        if let w {
            do {
                try w.stream.push(data) { ordinal, chunk in
                    try w.route(ordinal, chunk)
                }
            } catch {
                settle(dataTask, error)
                dataTask.cancel()
            }
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask,
                           didCompleteWithError error: Error?) {
        lock.lock()
        let w = waiters[Downloader.token(task)]
        lock.unlock()
        var outcome = error
        if outcome == nil, let w {
            do {
                try w.whole()
            } catch {
                outcome = error
            }
        }
        settle(task, outcome)
    }
}
