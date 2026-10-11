import Foundation

public struct XETToken: Sendable, Equatable {
    public let cas: URL
    public let bearer: String
    public let expires: Date

    public init(cas: URL, bearer: String, expires: Date) {
        self.cas = cas
        self.bearer = bearer
        self.expires = expires
    }
}

public struct XETFile: Sendable, Equatable {
    public let id: String
    public let size: Int64
    public let location: URL?

    public init(id: String, size: Int64, location: URL?) {
        self.id = id
        self.size = size
        self.location = location
    }
}

public struct Term: Sendable, Equatable {
    public let hash: String
    public let unpacked: Int64
    public let chunks: Range<Int>

    public init(hash: String, unpacked: Int64, chunks: Range<Int>) {
        self.hash = hash
        self.unpacked = unpacked
        self.chunks = chunks
    }
}

public struct Fetch: Sendable, Hashable {
    public let url: URL
    public let chunks: Range<Int>
    public let bytes: ClosedRange<Int64>

    public init(url: URL, chunks: Range<Int>, bytes: ClosedRange<Int64>) {
        self.url = url
        self.chunks = chunks
        self.bytes = bytes
    }

    public var range: String { "bytes=\(bytes.lowerBound)-\(bytes.upperBound)" }
}

public struct Slice: Sendable, Equatable {
    public let chunks: Range<Int>
    public let unpacked: Int64
    public let skip: Int64
    public let length: Int64
    public let offset: Int64

    public init(chunks: Range<Int>, unpacked: Int64, skip: Int64,
                length: Int64, offset: Int64) {
        self.chunks = chunks
        self.unpacked = unpacked
        self.skip = skip
        self.length = length
        self.offset = offset
    }
}

public struct Job: Sendable, Equatable {
    public let fetch: Fetch
    public let slices: [Slice]

    public init(fetch: Fetch, slices: [Slice]) {
        self.fetch = fetch
        self.slices = slices
    }

    public var length: Int64 {
        slices.reduce(Int64(0)) { sum, s in sum + s.length }
    }
}

public struct Reconstruction: Sendable, Equatable {
    public let skip: Int64
    public let terms: [Term]
    public let fetches: [String: [Fetch]]

    public init(skip: Int64, terms: [Term], fetches: [String: [Fetch]]) {
        self.skip = skip
        self.terms = terms
        self.fetches = fetches
    }

    public init(json data: Data) throws {
        let obj = try JSONSerialization.jsonObject(with: data)
        let top = try Reconstruction.need(obj as? [String: Any], "object")
        let v2 = top["xorbs"] as? [String: Any]
        let v1 = top["fetch_info"] as? [String: Any]
        if v2 == nil && v1 == nil {
            throw XETError.malformed("neither xorbs nor fetch_info")
        }
        skip = Reconstruction.int64(top["offset_into_first_range"]) ?? 0
        terms = try Reconstruction.need(top["terms"] as? [Any], "terms")
            .map { any in try Reconstruction.term(any) }
        var out: [String: [Fetch]] = [:]
        for (hash, list) in v2 ?? [:] {
            out[hash] = try (list as? [Any] ?? []).flatMap { any in
                try Reconstruction.fetchesV2(any)
            }
        }
        for (hash, list) in v1 ?? [:] {
            out[hash] = try (list as? [Any] ?? []).map { any in
                try Reconstruction.fetchV1(any)
            }
        }
        fetches = out
    }

    public var unpacked: Int64 {
        terms.reduce(Int64(0)) { sum, t in sum + t.unpacked }
    }

    public func jobs(limit: Int64? = nil) throws -> [Job] {
        let available = unpacked - skip
        if available < 0 {
            throw XETError.malformed("offset \(skip) past \(unpacked) bytes")
        }
        let total = min(available, limit ?? available)
        let outputEnd = skip + total
        var termStart: Int64 = 0
        var order: [Fetch] = []
        var slices: [Fetch: [Slice]] = [:]
        for t in terms {
            let termEnd = termStart + t.unpacked
            let writeStart = max(termStart, skip)
            let writeEnd = min(termEnd, outputEnd)
            if writeStart < writeEnd {
                let f = try fetch(covering: t)
                if slices[f] == nil { order.append(f) }
                slices[f, default: []].append(Slice(
                    chunks: t.chunks, unpacked: t.unpacked,
                    skip: writeStart - termStart,
                    length: writeEnd - writeStart,
                    offset: writeStart - skip))
            }
            termStart = termEnd
        }
        return order.map { f in Job(fetch: f, slices: slices[f] ?? []) }
    }

    private func fetch(covering t: Term) throws -> Fetch {
        let found = (fetches[t.hash] ?? []).first { f in
            f.chunks.lowerBound <= t.chunks.lowerBound
                && f.chunks.upperBound >= t.chunks.upperBound
        }
        return try Reconstruction.need(found, "no fetch covers \(t.hash) "
            + "chunks \(t.chunks.lowerBound)..<\(t.chunks.upperBound)")
    }

    static func need<T>(_ v: T?, _ what: String) throws -> T {
        if v == nil { throw XETError.malformed(what) }
        return v!
    }

    static func int64(_ any: Any?) -> Int64? {
        (any as? NSNumber)?.int64Value
    }

    static func intRange(_ any: Any?, _ what: String) throws -> Range<Int> {
        let d = try need(any as? [String: Any], what)
        let lo = try need(int64(d["start"]), what + ".start")
        let hi = try need(int64(d["end"]), what + ".end")
        if hi < lo { throw XETError.malformed("\(what) \(lo)..<\(hi)") }
        return Int(lo)..<Int(hi)
    }

    static func byteRange(_ any: Any?, _ what: String) throws
        -> ClosedRange<Int64> {
        let d = try need(any as? [String: Any], what)
        let lo = try need(int64(d["start"]), what + ".start")
        let hi = try need(int64(d["end"]), what + ".end")
        if hi < lo { throw XETError.malformed("\(what) \(lo)...\(hi)") }
        return lo...hi
    }

    static func term(_ any: Any) throws -> Term {
        let d = try need(any as? [String: Any], "term")
        return Term(hash: try need(d["hash"] as? String, "term.hash"),
                    unpacked: try need(int64(d["unpacked_length"]),
                                       "term.unpacked_length"),
                    chunks: try intRange(d["range"], "term.range"))
    }

    static func fetchesV2(_ any: Any) throws -> [Fetch] {
        let d = try need(any as? [String: Any], "xorb fetch")
        let url = try need(URL(string: try need(d["url"] as? String,
                                                "xorb url")), "xorb url")
        return try need(d["ranges"] as? [Any], "xorb ranges").map { r in
            let e = try need(r as? [String: Any], "xorb range")
            return Fetch(url: url,
                         chunks: try intRange(e["chunks"], "range.chunks"),
                         bytes: try byteRange(e["bytes"], "range.bytes"))
        }
    }

    static func fetchV1(_ any: Any) throws -> Fetch {
        let d = try need(any as? [String: Any], "fetch_info")
        return Fetch(
            url: try need(URL(string: try need(d["url"] as? String,
                                               "fetch_info url")),
                          "fetch_info url"),
            chunks: try intRange(d["range"], "fetch_info.range"),
            bytes: try byteRange(d["url_range"], "fetch_info.url_range"))
    }
}

private final class NoRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping @Sendable
                        (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

public enum Hub {
    public static let host = "https://huggingface.co"

    public static func file(_ resolve: URL, hubToken: String? = nil,
                            configuration: URLSessionConfiguration
                                = .ephemeral) async throws -> XETFile? {
        let session = URLSession(configuration: configuration,
                                 delegate: NoRedirect(), delegateQueue: nil)
        var req = URLRequest(url: resolve)
        req.httpMethod = "HEAD"
        req.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if let hubToken {
            req.setValue("Bearer \(hubToken)",
                         forHTTPHeaderField: "Authorization")
        }
        let (_, resp) = try await session.data(for: req)
        session.finishTasksAndInvalidate()
        let http = resp as? HTTPURLResponse
        let code = http?.statusCode ?? 0
        if code < 200 || code >= 400 {
            throw XETError.http(code, resolve.absoluteString)
        }
        var out: XETFile? = nil
        if let id = http?.value(forHTTPHeaderField: "X-Xet-Hash") {
            let size = http?.value(forHTTPHeaderField: "X-Linked-Size")
                .flatMap { s in Int64(s) } ?? 0
            let location = http?.value(forHTTPHeaderField: "Location")
                .flatMap { s in URL(string: s) }
            out = XETFile(id: id, size: size, location: location)
        }
        return out
    }

    public static func refresh(for resolve: URL) -> URL? {
        let parts = resolve.pathComponents.filter { p in p != "/" }
        let typed = ["datasets", "spaces"].contains(parts.first ?? "")
        let kind = typed ? parts[0] : "models"
        let base = typed ? 1 : 0
        var out: URL? = nil
        if parts.count >= base + 4, parts[base + 2] == "resolve" {
            out = URL(string: "\(host)/api/\(kind)/\(parts[base])/"
                + "\(parts[base + 1])/xet-read-token/\(parts[base + 3])")
        }
        return out
    }

    public static func token(_ refresh: URL, hubToken: String? = nil,
                             configuration: URLSessionConfiguration
                                 = .ephemeral) async throws -> XETToken {
        var req = URLRequest(url: refresh)
        req.cachePolicy = .reloadIgnoringLocalCacheData
        if let hubToken {
            req.setValue("Bearer \(hubToken)",
                         forHTTPHeaderField: "Authorization")
        }
        let (data, resp) = try await fetch(req, configuration)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code != 200 { throw XETError.http(code, refresh.absoluteString) }
        let d = try Reconstruction.need(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "token")
        let cas = try Reconstruction.need(
            (d["casUrl"] as? String).flatMap { s in URL(string: s) },
            "token.casUrl")
        let bearer = try Reconstruction.need(d["accessToken"] as? String,
                                             "token.accessToken")
        let exp = Reconstruction.int64(d["exp"]) ?? 0
        return XETToken(cas: cas, bearer: bearer,
                        expires: Date(timeIntervalSince1970: Double(exp)))
    }

    public static func reconstruction(of id: String, token: XETToken,
                                      range: Range<Int64>? = nil,
                                      configuration: URLSessionConfiguration
                                          = .ephemeral)
        async throws -> Reconstruction {
        let (v2, code2) = try await query("v2", id, token, range,
                                          configuration)
        var data = v2
        if data == nil, code2 == 404 || code2 == 501 {
            let (v1, code1) = try await query("v1", id, token, range,
                                              configuration)
            data = v1
            if data == nil { throw XETError.http(code1, id) }
        } else if data == nil {
            throw XETError.http(code2, id)
        }
        return try Reconstruction(json: data!)
    }

    static func query(_ version: String, _ id: String, _ token: XETToken,
                      _ range: Range<Int64>?,
                      _ configuration: URLSessionConfiguration)
        async throws -> (Data?, Int) {
        let url = token.cas.appendingPathComponent(version)
            .appendingPathComponent("reconstructions")
            .appendingPathComponent(id)
        var req = URLRequest(url: url)
        req.setValue("Bearer \(token.bearer)",
                     forHTTPHeaderField: "Authorization")
        if let range, !range.isEmpty {
            req.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)",
                         forHTTPHeaderField: "Range")
        }
        let (data, resp) = try await fetch(req, configuration)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        return (code == 200 ? data : nil, code)
    }

    static func fetch(_ req: URLRequest,
                      _ configuration: URLSessionConfiguration)
        async throws -> (Data, URLResponse) {
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        return try await session.data(for: req)
    }
}
