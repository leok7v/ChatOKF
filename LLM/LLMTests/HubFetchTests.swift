import CryptoKit
import Darwin
import Foundation
import Testing
import XET
@testable import LLM

private final class XetStore: @unchecked Sendable {
    static let chunk = 64 << 10
    static let perXorb = 4
    private let lock = NSLock()
    private var blob = Data()
    private var xet = true
    private var delay: TimeInterval = 0
    private var headCount = 0
    private var xorbCount = 0
    private var bridgeCount = 0

    struct Reply {
        let code: Int
        let headers: [String: String]
        let body: Data
        let sleep: TimeInterval
    }

    func reset(_ blob: Data, xet: Bool = true, delay: TimeInterval = 0) {
        lock.lock()
        self.blob = blob
        self.xet = xet
        self.delay = delay
        headCount = 0
        xorbCount = 0
        bridgeCount = 0
        lock.unlock()
    }

    var heads: Int { lock.lock(); defer { lock.unlock() }; return headCount }

    var xorbRequests: Int {
        lock.lock(); defer { lock.unlock() }; return xorbCount
    }

    var bridgeRequests: Int {
        lock.lock(); defer { lock.unlock() }; return bridgeCount
    }

    private var xorbBytes: Int { XetStore.chunk * XetStore.perXorb }

    private func plain(_ k: Int) -> Data {
        let lo = k * xorbBytes
        return blob.subdata(in: lo..<min(lo + xorbBytes, blob.count))
    }

    private func serialized(_ k: Int) -> (bytes: Data, starts: [Int]) {
        let p = plain(k)
        var out = Data()
        var starts: [Int] = []
        var at = 0
        while at < p.count {
            let n = min(XetStore.chunk, p.count - at)
            starts.append(out.count)
            out.append(contentsOf: ChunkHeader(compressed: n, scheme: 0,
                                               uncompressed: n).encoded)
            out.append(p.subdata(in: at..<at + n))
            at += n
        }
        starts.append(out.count)
        return (out, starts)
    }

    private func reconstruction(_ from: Int64, _ to: Int64) -> Data {
        let first = Int(from) / xorbBytes
        let last = Int(to) / xorbBytes
        var terms: [[String: Any]] = []
        var xorbs: [String: Any] = [:]
        for k in first...last {
            let s = serialized(k)
            let n = s.starts.count - 1
            terms.append(["hash": "x\(k)", "unpacked_length": plain(k).count,
                          "range": ["start": 0, "end": n]])
            xorbs["x\(k)"] = [[
                "url": "https://xorb.invalid/xorb/\(k)",
                "ranges": [["chunks": ["start": 0, "end": n],
                            "bytes": ["start": 0,
                                      "end": s.bytes.count - 1]]]]]
        }
        let top: [String: Any] = [
            "offset_into_first_range": Int(from) - first * xorbBytes,
            "terms": terms, "xorbs": xorbs]
        return (try? JSONSerialization.data(withJSONObject: top)) ?? Data()
    }

    private static func range(_ header: String?) -> (Int64, Int64) {
        let spec = header?.dropFirst("bytes=".count).split(separator: "-")
        let from = spec.flatMap { s in Int64(s[0]) } ?? 0
        let to = spec.flatMap { s in s.count > 1 ? Int64(s[1]) : nil }
            ?? Int64.max
        return (from, to)
    }

    private func ranged(_ whole: Data, _ from: Int64, _ to: Int64,
                        _ sleep: TimeInterval) -> Reply {
        let hi = min(Int(to) + 1, whole.count)
        let body = whole.subdata(in: min(Int(from), hi)..<hi)
        return Reply(code: 206, headers: [
            "Content-Range": "bytes \(from)-\(from + Int64(body.count) - 1)/*",
        ], body: body, sleep: sleep)
    }

    func answer(_ req: URLRequest) -> Reply {
        lock.lock()
        defer { lock.unlock() }
        let path = req.url?.path ?? ""
        let (from, to) = XetStore.range(req.value(forHTTPHeaderField: "Range"))
        var out = Reply(code: 404, headers: [:], body: Data(), sleep: 0)
        if path.contains("/resolve/") && req.httpMethod == "HEAD" {
            headCount += 1
            let linked = ["X-Xet-Hash": String(repeating: "f", count: 64),
                          "X-Linked-Size": "\(blob.count)"]
            out = Reply(code: 200, headers: xet ? linked : [:], body: Data(),
                        sleep: 0)
        } else if path.contains("/resolve/") {
            bridgeCount += 1
            out = ranged(blob, from, to, delay)
        } else if path.contains("xet-read-token") {
            let json = "{\"casUrl\":\"https://cas.invalid\","
                + "\"accessToken\":\"t\",\"exp\":4102444800}"
            out = Reply(code: 200, headers: [:], body: Data(json.utf8),
                        sleep: 0)
        } else if path.contains("/reconstructions/") {
            let last = Int64(blob.count - 1)
            out = Reply(code: 200, headers: [:],
                        body: reconstruction(from, min(to, last)), sleep: 0)
        } else if path.hasPrefix("/xorb/"),
                  let k = Int(path.dropFirst("/xorb/".count)) {
            xorbCount += 1
            out = ranged(serialized(k).bytes, from, to, delay)
        }
        return out
    }
}

private struct XetBox: @unchecked Sendable {
    let server: XetServer
}

private final class XetServer: URLProtocol {
    static let store = XetStore()

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest)
        -> URLRequest { request }

    override func startLoading() {
        let box = XetBox(server: self)
        DispatchQueue.global().async { box.server.serve() }
    }

    private func serve() {
        let reply = XetServer.store.answer(request)
        if reply.sleep > 0 { Thread.sleep(forTimeInterval: reply.sleep) }
        if let url = request.url {
            var head = reply.headers
            head["Content-Length"] = "\(reply.body.count)"
            if let resp = HTTPURLResponse(url: url, statusCode: reply.code,
                                          httpVersion: "HTTP/1.1",
                                          headerFields: head) {
                client?.urlProtocol(self, didReceive: resp,
                                    cacheStoragePolicy: .notAllowed)
            }
            var at = 0
            while at < reply.body.count {
                let n = min(64 << 10, reply.body.count - at)
                client?.urlProtocol(
                    self, didLoad: reply.body.subdata(in: at..<at + n))
                at += n
            }
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

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

@Suite(.enabled(if: ProcessInfo.processInfo.environment["CHATOKF_LIVE_NETWORK"]
                   != nil))
struct HubLiveTests {
    @Test(arguments: ["Ternary-Bonsai-1.7B", "gemma-4-E2B"])
    func aCatalogFileArrivesBothWaysAtAMeasuredRate(_ name: String)
        async throws {
        let src = try #require(ModelCatalog.source(name))
        let file = try #require(src.files?.first)
        for transport in [HubFetch.Transport.bridge, .chunks] {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("hub-live-\(UUID().uuidString)")
            let t0 = Date()
            let root = try await HubFetch.fetch(
                repo: src.repo, into: dir, revision: src.revision,
                files: [file], transport: transport)
            let secs = Date().timeIntervalSince(t0)
            let size = HubFetch.size(root.appendingPathComponent(file))
            print(String(format: "hub live %@ %@: %lld bytes in %.1f s, "
                         + "%.1f MiB/s", name, "\(transport)", size, secs,
                         Double(size) / secs / 1048576))
            #expect(size == src.bytes)
            try? FileManager.default.removeItem(at: dir)
        }
    }
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

    private func xetPump() -> Pump {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [XetServer.self]
        return Pump(configuration: c)
    }

    private func resolve() throws -> URL {
        try #require(URL(string: "https://fixture.invalid/leok7v/fixture/"
                         + "resolve/abc123/blob.gguf"))
    }

    private func committedReaches(_ part: URL, _ n: Int64) async throws {
        let began = Date()
        while HubFetch.committed(part) < n
                && Date().timeIntervalSince(began) < 60 {
            try await Task.sleep(for: .milliseconds(5))
        }
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

    @Test func chunksRebuildTheFileAndTheMarkerReachesTheEnd() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(24, span)
        let part = scratch()
        XetServer.store.reset(blob)
        let p = xetPump()
        try await HubFetch.foreground(try resolve(), e, part, p, .chunks,
                                      { _ in })
        #expect(HubFetch.committed(part) == e.size)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        #expect(XetServer.store.heads == 1)
        #expect(XetServer.store.xorbRequests == 24)
        #expect(XetServer.store.bridgeRequests == 0)
        p.done()
        drop(part)
    }

    @Test func chunksResumeFromTheMarkerAfterACancel() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(24, span)
        let part = scratch()
        XetServer.store.reset(blob, delay: 0.02)
        let p = xetPump()
        let u = try resolve()
        let job = Task {
            try await HubFetch.foreground(u, e, part, p, .chunks, { _ in })
        }
        try await committedReaches(part, 2 * span)
        job.cancel()
        let outcome = await job.result
        #expect(throws: (any Error).self) { try outcome.get() }
        let have = HubFetch.committed(part)
        #expect(have >= 2 * span && have < e.size)
        #expect(have % span == 0)
        #expect(try prefix(part, have) == blob.prefix(Int(have)))
        XetServer.store.reset(blob)
        try await HubFetch.foreground(u, e, part, p, .chunks, { _ in })
        #expect(XetServer.store.xorbRequests
                == Int((e.size - have + span - 1) / span))
        #expect(HubFetch.committed(part) == e.size)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        p.done()
        drop(part)
    }

    @Test func aFileWithoutAXetHashFallsToTheBridge() async throws {
        let span: Int64 = 256 << 10
        let (blob, e) = fixture(8, span)
        let part = scratch()
        XetServer.store.reset(blob, xet: false)
        let p = xetPump()
        try await HubFetch.foreground(try resolve(), e, part, p, .chunks,
                                      { _ in })
        #expect(XetServer.store.heads == 1)
        #expect(XetServer.store.xorbRequests == 0)
        #expect(XetServer.store.bridgeRequests > 0)
        #expect(throws: Never.self) { try HubFetch.verify(part, e) }
        p.done()
        drop(part)
    }

    @Test func aDownloadLargerThanTheVolumeIsRefusedBeforeItStarts() throws {
        let part = scratch()
        let root = part.deletingLastPathComponent()
        let r = HubFetch.room(near: root.appendingPathComponent("not/yet"))
        #expect(r.free > 0 && r.important > 0 && r.opportunistic > 0)
        let huge = HubFetch.Entry(path: "huge.gguf", size: Int64.max / 4,
                                  oid: "", lfs: true)
        let small = HubFetch.Entry(path: "small.gguf", size: 1, oid: "",
                                   lfs: true)
        #expect(throws: HubError.self) {
            try HubFetch.claim([huge], root, .important)
        }
        #expect(throws: Never.self) {
            try HubFetch.claim([small], root, .opportunistic)
        }
        let text = HubFetch.spaceMessage(needed: 4 << 30, free: 1 << 30,
                                         freeable: 3 << 30)
        #expect(text.contains("GB") && text.contains("remove"))
        let soon = HubFetch.spaceMessage(needed: 2 << 30, free: 1 << 30,
                                         freeable: 3 << 30)
        #expect(soon.contains("try again"))
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
