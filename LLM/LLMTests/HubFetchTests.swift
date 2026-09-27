import CryptoKit
import Darwin
import Foundation
import Testing
@testable import LLM

private final class Served: @unchecked Sendable {
    private let lock = NSLock()
    private var blob = Data()
    private var count = 0
    private var bytes: Int64 = 0
    private var failAt: Int64? = nil
    private var slowEvery = 0
    private var delay: TimeInterval = 0

    func reset(_ blob: Data, failAt: Int64? = nil, slowEvery: Int = 0,
               delay: TimeInterval = 0) {
        lock.lock()
        self.blob = blob
        self.count = 0
        self.bytes = 0
        self.failAt = failAt
        self.slowEvery = slowEvery
        self.delay = delay
        lock.unlock()
    }

    func plan(_ from: Int64) -> (fail: Bool, sleep: TimeInterval) {
        lock.lock()
        count += 1
        let fail = failAt == from
        if fail { failAt = nil }
        let slow = slowEvery > 0 && count % slowEvery == 0
        lock.unlock()
        return (fail, slow ? delay : 0)
    }

    func body(_ from: Int64, _ to: Int64) -> Data {
        lock.lock()
        let hi = min(Int(to) + 1, blob.count)
        let out = blob.subdata(in: Int(from)..<hi)
        bytes += Int64(out.count)
        lock.unlock()
        return out
    }

    var requests: Int {
        lock.lock(); defer { lock.unlock() }; return count
    }

    var sent: Int64 {
        lock.lock(); defer { lock.unlock() }; return bytes
    }
}

private struct Box: @unchecked Sendable {
    let server: RangeServer
}

private final class RangeServer: URLProtocol {
    static let store = Served()

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest)
        -> URLRequest { request }

    private static func range(_ header: String?) -> (Int64, Int64) {
        let spec = header?.dropFirst("bytes=".count).split(separator: "-")
        let from = spec.flatMap { s in Int64(s[0]) } ?? 0
        let to = spec.flatMap { s in s.count > 1 ? Int64(s[1]) : nil }
            ?? Int64.max
        return (from, to)
    }

    override func startLoading() {
        let box = Box(server: self)
        let (from, to) = RangeServer.range(
            request.value(forHTTPHeaderField: "Range"))
        let plan = RangeServer.store.plan(from)
        DispatchQueue.global().async {
            box.server.serve(from, to, plan)
        }
    }

    private func serve(_ from: Int64, _ to: Int64,
                       _ plan: (fail: Bool, sleep: TimeInterval)) {
        if plan.sleep > 0 { Thread.sleep(forTimeInterval: plan.sleep) }
        if plan.fail {
            client?.urlProtocol(self, didFailWithError:
                                URLError(.networkConnectionLost))
        } else if let url = request.url {
            let body = RangeServer.store.body(from, to)
            let last = from + Int64(body.count) - 1
            let head = [
                "Content-Length": "\(body.count)",
                "Content-Range": "bytes \(from)-\(last)/*",
            ]
            let resp = HTTPURLResponse(url: url, statusCode: 206,
                                       httpVersion: "HTTP/1.1",
                                       headerFields: head)
            if let resp {
                client?.urlProtocol(self, didReceive: resp,
                                    cacheStoragePolicy: .notAllowed)
            }
            var at = 0
            while at < body.count {
                let n = min(64 << 10, body.count - at)
                client?.urlProtocol(self,
                                    didLoad: body.subdata(in: at..<at + n))
                at += n
            }
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

@Suite(.serialized)
struct HubFetchTests {
    private func scratch() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("assemble-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("blob.gguf.part")
    }

    private func write(_ url: URL, _ text: String) {
        try? Data(text.utf8).write(to: url)
    }

    private func drop(_ part: URL) {
        try? FileManager.default.removeItem(
            at: part.deletingLastPathComponent())
    }

    private func fixture(_ pieces: Int, _ span: Int64)
        -> (Data, HubFetch.Entry) {
        let size = Int(span) * pieces - 777
        var blob = Data(count: size)
        blob.withUnsafeMutableBytes { raw in
            arc4random_buf(raw.baseAddress, raw.count)
        }
        let oid = SHA256.hash(data: blob)
            .map { b in String(format: "%02x", b) }.joined()
        let e = HubFetch.Entry(path: "blob.gguf", size: Int64(size),
                               oid: oid, lfs: true)
        return (blob, e)
    }

    private func pump() -> Pump {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [RangeServer.self]
        return Pump(configuration: c)
    }

    private func url() throws -> URL {
        try #require(URL(string: "https://fixture.invalid/blob.gguf"))
    }

    private func prefix(_ part: URL, _ n: Int64) throws -> Data {
        let h = try FileHandle(forReadingFrom: part)
        let out = try h.read(upToCount: Int(n)) ?? Data()
        try h.close()
        return out
    }

    private func logicalWrites() -> Int64 {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { q in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, q)
            }
        }
        return rc == 0 ? Int64(info.ri_logical_writes) : 0
    }

    @Test func assembleTakesTheContiguousPrefixOnly() throws {
        let part = scratch()
        write(part, "")
        write(HubFetch.piece(part, 0), "hello ")
        write(HubFetch.piece(part, 6), "world")
        write(HubFetch.piece(part, 99), "orphan")
        let have = try HubFetch.assemble(part)
        #expect(have == 11)
        #expect(try String(decoding: Data(contentsOf: part), as: UTF8.self)
                == "hello world")
        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 0).path))
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 6).path))
        #expect(fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        HubFetch.sweep(part)
        #expect(!fm.fileExists(atPath: HubFetch.piece(part, 99).path))
        drop(part)
    }

    @Test func assembleNeverWritesAHole() throws {
        let part = scratch()
        write(part, "abc")
        write(HubFetch.piece(part, 99), "way past the end")
        let have = try HubFetch.assemble(part)
        #expect(have == 3)
        #expect(try Data(contentsOf: part).count == 3)
        #expect(FileManager.default.fileExists(
            atPath: HubFetch.piece(part, 99).path))
        drop(part)
    }

    @Test func assembleSplicesIntoAPreallocatedPart() throws {
        let part = scratch()
        var bytes = Data("hello ".utf8)
        bytes.append(Data(count: 58))
        try bytes.write(to: part)
        try HubFetch.commit(part, 6)
        write(HubFetch.piece(part, 6), "world")
        #expect(HubFetch.committed(part) == 6)
        let have = try HubFetch.assemble(part)
        #expect(have == 11)
        #expect(HubFetch.committed(part) == 11)
        #expect(HubFetch.size(part) == 64)
        #expect(try prefix(part, 11) == Data("hello world".utf8))
        drop(part)
    }

    @Test func fillResumesAfterALaneError() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(24, span)
        let part = scratch()
        RangeServer.store.reset(blob, failAt: 9 * span)
        let p = pump()
        await #expect(throws: (any Error).self) {
            try await HubFetch.fill(try url(), e, part, p, { _ in },
                                    span: span)
        }
        let have = HubFetch.committed(part)
        #expect(have == 9 * span)
        #expect(try prefix(part, have) == blob.prefix(Int(have)))
        RangeServer.store.reset(blob)
        try await HubFetch.fill(try url(), e, part, p, { _ in }, span: span)
        #expect(RangeServer.store.sent == e.size - have)
        #expect(HubFetch.committed(part) == e.size)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        p.done()
        drop(part)
    }

    @Test func fillResumesAfterACancel() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(24, span)
        let part = scratch()
        RangeServer.store.reset(blob, slowEvery: 1, delay: 0.02)
        let p = pump()
        let u = try url()
        let job = Task {
            try await HubFetch.fill(u, e, part, p, { _ in }, span: span)
        }
        while RangeServer.store.requests < 6 {
            try await Task.sleep(for: .milliseconds(5))
        }
        job.cancel()
        let outcome = await job.result
        #expect(throws: (any Error).self) { try outcome.get() }
        let have = HubFetch.committed(part)
        #expect(have > 0 && have < e.size)
        #expect(try prefix(part, have) == blob.prefix(Int(have)))
        RangeServer.store.reset(blob)
        try await HubFetch.fill(u, e, part, p, { _ in }, span: span)
        #expect(RangeServer.store.sent == e.size - have)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        p.done()
        drop(part)
    }

    @Test func fillResumesAPartWithoutAMarker() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(8, span)
        let part = scratch()
        let legacy: Int64 = 100_000
        try blob.prefix(Int(legacy)).write(to: part)
        #expect(HubFetch.committed(part) == legacy)
        RangeServer.store.reset(blob)
        let p = pump()
        try await HubFetch.fill(try url(), e, part, p, { _ in }, span: span)
        #expect(RangeServer.store.sent == e.size - legacy)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        p.done()
        drop(part)
    }

    @Test func fillWritesEachByteOnceAndDoesNotWaitOnASlowLane()
        async throws {
        let span: Int64 = 2 << 20
        let (blob, e) = fixture(32, span)
        let part = scratch()
        RangeServer.store.reset(blob, slowEvery: 4, delay: 0.1)
        let p = pump()
        let wrote = logicalWrites()
        let t0 = Date()
        try await HubFetch.fill(try url(), e, part, p, { _ in }, span: span)
        let wall = Date().timeIntervalSince(t0)
        let written = logicalWrites() - wrote
        print("fill \(e.size) bytes: wall \(wall) s, "
              + "logical writes \(written)")
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        #expect(written < e.size * 3 / 2)
        p.done()
        drop(part)
    }
}
