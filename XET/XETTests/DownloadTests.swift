import Foundation
import Testing
@testable import XET

private final class Xorbs: @unchecked Sendable {
    private let lock = NSLock()
    private var blobs: [String: [UInt8]] = [:]
    private var failures: [String: Int] = [:]
    private var count = 0

    func reset(_ blobs: [String: [UInt8]], failing: [String: Int] = [:]) {
        lock.lock()
        self.blobs = blobs
        self.failures = failing
        self.count = 0
        lock.unlock()
    }

    func serve(_ path: String, _ from: Int, _ to: Int)
        -> (fail: Bool, body: [UInt8]) {
        lock.lock()
        defer { lock.unlock() }
        count += 1
        let left = failures[path] ?? 0
        if left > 0 { failures[path] = left - 1 }
        let blob = blobs[path] ?? []
        let hi = min(to + 1, blob.count)
        return (left > 0, from < hi ? Array(blob[from..<hi]) : [])
    }

    var requests: Int {
        lock.lock(); defer { lock.unlock() }; return count
    }
}

private struct Box: @unchecked Sendable {
    let server: XorbServer
}

private final class XorbServer: URLProtocol {
    static let store = Xorbs()

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest)
        -> URLRequest { request }

    override func startLoading() {
        let box = Box(server: self)
        let spec = request.value(forHTTPHeaderField: "Range")?
            .dropFirst("bytes=".count).split(separator: "-") ?? []
        let from = spec.count > 0 ? Int(spec[0]) ?? 0 : 0
        let to = spec.count > 1 ? Int(spec[1]) ?? Int.max : Int.max
        let path = request.url?.path ?? ""
        DispatchQueue.global().async { box.server.serve(path, from, to) }
    }

    private func serve(_ path: String, _ from: Int, _ to: Int) {
        let (fail, body) = XorbServer.store.serve(path, from, to)
        if fail {
            client?.urlProtocol(self, didFailWithError:
                                URLError(.networkConnectionLost))
        } else if let url = request.url {
            let head = ["Content-Length": "\(body.count)",
                        "Content-Range": "bytes \(from)-\(to)/*"]
            if let resp = HTTPURLResponse(url: url, statusCode: 206,
                                          httpVersion: "HTTP/1.1",
                                          headerFields: head) {
                client?.urlProtocol(self, didReceive: resp,
                                    cacheStoragePolicy: .notAllowed)
            }
            var at = 0
            while at < body.count {
                let n = min(5000, body.count - at)
                client?.urlProtocol(self, didLoad: Data(body[at..<at + n]))
                at += n
            }
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}
}

private struct Built {
    let plain: [[UInt8]]
    let bytes: [UInt8]
    let starts: [Int]
}

private func xorb(_ sizes: [Int], seed: UInt64) -> Built {
    var plain: [[UInt8]] = []
    var bytes: [UInt8] = []
    var starts: [Int] = []
    for (i, n) in sizes.enumerated() {
        let p = noise(n, seed: seed + UInt64(i))
        plain.append(p)
        starts.append(bytes.count)
        bytes += chunk(UInt8(i % 3), p)
    }
    starts.append(bytes.count)
    return Built(plain: plain, bytes: bytes, starts: starts)
}

@Suite(.serialized) struct DownloadTests {
    private let a = xorb([70000, 131072, 131072, 3000, 131072, 50000], seed: 11)
    private let b = xorb([40000, 131072, 9000], seed: 99)

    private func configuration() -> URLSessionConfiguration {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [XorbServer.self]
        return c
    }

    private func fetch(_ name: String, _ built: Built,
                       _ chunks: Range<Int>) -> Fetch {
        let lo = Int64(built.starts[chunks.lowerBound])
        let hi = Int64(built.starts[chunks.upperBound] - 1)
        return Fetch(url: URL(string: "https://xorbs.invalid/\(name)")!,
                     chunks: chunks, bytes: lo...hi)
    }

    private func total(_ pieces: ArraySlice<[UInt8]>) -> Int64 {
        Int64(pieces.reduce(0) { sum, p in sum + p.count })
    }

    private func reconstruction() -> Reconstruction {
        let termA1 = Term(hash: "a", unpacked: total(a.plain[1..<4]),
                          chunks: 1..<4)
        let termB = Term(hash: "b", unpacked: total(b.plain[0..<3]),
                         chunks: 0..<3)
        let termA2 = Term(hash: "a", unpacked: total(a.plain[3..<6]),
                          chunks: 3..<6)
        return Reconstruction(
            skip: 0, terms: [termA1, termB, termA2],
            fetches: ["a": [fetch("a", a, 1..<6)],
                      "b": [fetch("b", b, 0..<3)]])
    }

    private func expected() -> [UInt8] {
        a.plain[1..<4].flatMap { p in p } + b.plain.flatMap { p in p }
            + a.plain[3..<6].flatMap { p in p }
    }

    @Test func aFileIsRebuiltFromTwoXorbsWithAReusedChunk() async throws {
        XorbServer.store.reset(["/a": a.bytes, "/b": b.bytes])
        let got = try await Downloader.data(reconstruction(),
                                            configuration: configuration())
        #expect(Array(got) == expected())
        #expect(XorbServer.store.requests == 2)
    }

    @Test func aRangeLandsAtItsOffsetInsideALargerFile() async throws {
        XorbServer.store.reset(["/a": a.bytes, "/b": b.bytes])
        let whole = expected()
        let r = reconstruction()
        let from: Int64 = 100000
        let length: Int64 = 300000
        let ranged = Reconstruction(skip: from, terms: r.terms,
                                    fetches: r.fetches)
        let fm = FileManager.default
        let file = fm.temporaryDirectory
            .appendingPathComponent("xet-range-\(UUID().uuidString)")
        fm.createFile(atPath: file.path, contents: Data(count: whole.count))
        let wrote = try await Downloader.download(
            ranged, limit: length, into: file, at: from,
            configuration: configuration())
        #expect(wrote == length)
        let got = try Data(contentsOf: file)
        #expect(got.count == whole.count)
        #expect(Array(got[Int(from)..<Int(from + length)])
                == Array(whole[Int(from)..<Int(from + length)]))
        #expect(got[0..<Int(from)].allSatisfy { b in b == 0 })
        try? fm.removeItem(at: file)
    }

    @Test func aFetchThatFailsOnceIsRetriedAndTheBytesAreRight()
        async throws {
        XorbServer.store.reset(["/a": a.bytes, "/b": b.bytes],
                               failing: ["/b": 1])
        let t0 = Date()
        let got = try await Downloader.data(reconstruction(),
                                            configuration: configuration())
        #expect(Array(got) == expected())
        #expect(XorbServer.store.requests == 3)
        #expect(Date().timeIntervalSince(t0) >= 1)
    }

    @Test func aMissingXorbIsAnError() async {
        XorbServer.store.reset(["/a": a.bytes])
        await #expect(throws: XETError.self) {
            _ = try await Downloader.data(reconstruction(),
                                          configuration: configuration())
        }
    }

    @Test func jobsCommitTheContiguousPrefixAndHoldTheLead() async throws {
        let f = fetch("a", a, 0..<1)
        let job = { (offset: Int64) in
            Job(fetch: f, slices: [Slice(chunks: 0..<1, unpacked: 10,
                                         skip: 0, length: 10,
                                         offset: offset)])
        }
        let seen = Progress()
        let jobs = Jobs([job(0), job(10), job(20), job(30)], lead: 15) { n in
            seen.note(n)
        }
        let first = try #require(try await jobs.take())
        let second = try #require(try await jobs.take())
        let third = Task { try await jobs.take() }
        try await Task.sleep(for: .milliseconds(250))
        #expect(seen.last == -1)
        try jobs.land(second)
        #expect(jobs.committed == 0)
        try jobs.land(first)
        #expect(jobs.committed == 20)
        #expect(seen.last == 20)
        let got = try #require(try await third.value)
        #expect(got.slices[0].offset == 20)
        let fourth = try #require(try await jobs.take())
        try jobs.land(fourth)
        #expect(jobs.committed == 20)
        try jobs.land(got)
        #expect(jobs.committed == 40)
        #expect(try await jobs.take() == nil)
    }

    @Test func progressReachesTheTotalExactlyOnce() async throws {
        XorbServer.store.reset(["/a": a.bytes, "/b": b.bytes])
        let seen = Progress()
        let fm = FileManager.default
        let file = fm.temporaryDirectory
            .appendingPathComponent("xet-progress-\(UUID().uuidString)")
        fm.createFile(atPath: file.path, contents: nil)
        let total = try await Downloader.download(
            reconstruction(), into: file, configuration: configuration(),
            progress: { n in seen.note(n) })
        #expect(seen.last == total)
        #expect(seen.monotonic)
        try? fm.removeItem(at: file)
    }
}

private final class Progress: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int64] = []

    func note(_ n: Int64) {
        lock.lock(); values.append(n); lock.unlock()
    }

    var last: Int64 {
        lock.lock(); defer { lock.unlock() }; return values.last ?? -1
    }

    var monotonic: Bool {
        lock.lock(); defer { lock.unlock() }
        return zip(values, values.dropFirst()).allSatisfy { a, b in a <= b }
    }
}
