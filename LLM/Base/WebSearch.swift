import Foundation

public enum SearchProvider: String, CaseIterable, Identifiable, Sendable {

    case parallel
    case mwmbl

    public var id: String { rawValue }

    public var label: String {
        let out: String
        switch self {
        case .parallel: out = "Parallel"
        case .mwmbl: out = "Mwmbl"
        }
        return out
    }

    public var detail: String {
        let out: String
        switch self {
        case .parallel:
            out = "A commercial search service, tried first because it "
                + "finds a great deal more. What you searched for and your "
                + "network address reach it, and its policy does not say "
                + "how long a search is kept."
        case .mwmbl:
            out = "A non-profit open search index, used when Parallel is "
                + "off or comes back empty. It keeps no record of a search "
                + "that points back to you and stores no network address."
        }
        return out
    }

    public var home: String {
        let out: String
        switch self {
        case .parallel: out = "https://parallel.ai"
        case .mwmbl: out = "https://mwmbl.org"
        }
        return out
    }

    public var privacy: String {
        let out: String
        switch self {
        case .parallel: out = "https://parallel.ai/privacy-policy"
        case .mwmbl: out = "https://mwmbl.org/privacy"
        }
        return out
    }

    public var flag: String { "search-\(rawValue)" }

    public var on: Bool { Flags.on(flag) }

    public func set(_ value: Bool) { Flags.store(flag, value) }

    public static var any: Bool {
        allCases.contains { provider in provider.on }
    }
}

struct SearchHit: Sendable {
    let title: String
    let url: String
    let extract: String
}

struct SearchError: Error, LocalizedError {
    let text: String
    var errorDescription: String? { text }
}

enum WebSearch {

    static let excerptLimit = 400
    static let proseLine = 40

    static func run(_ query: String, count: Int) async -> String {
        var hits: [SearchHit] = []
        if SearchProvider.parallel.on {
            hits = await ParallelSearch.shared.search(query, count: count)
        }
        if hits.isEmpty && SearchProvider.mwmbl.on {
            hits = await mwmbl(query, count: count)
        }
        return format(hits, query)
    }

    static func format(_ hits: [SearchHit], _ query: String) -> String {
        var result = "No web results for \"\(query)\". Do not search again; "
            + "answer the user from your own knowledge."
        if !hits.isEmpty {
            var lines: [String] = []
            var rank = 0
            for hit in hits {
                rank += 1
                var block = "\(rank). "
                    + (hit.title.isEmpty ? "(no title)" : hit.title) + "\n"
                if !hit.url.isEmpty { block += "   " + hit.url + "\n" }
                if !hit.extract.isEmpty {
                    block += "   " + hit.extract + "\n"
                }
                lines.append(block)
            }
            result = lines.joined()
        }
        return result
    }

    static func excerpt(_ raw: [String]) -> String {
        var kept: [String] = []
        for block in raw {
            for line in block.split(separator: "\n") {
                let text = line.trimmingCharacters(in: .whitespaces)
                if text.count >= proseLine && text.contains(" ") {
                    kept.append(text)
                }
            }
        }
        if kept.isEmpty, let first = raw.first { kept = [first] }
        return Tools.clampText(squeezed(kept.joined(separator: " ")),
                               excerptLimit)
    }

    static func squeezed(_ s: String) -> String {
        s.split(whereSeparator: { c in c.isWhitespace })
            .joined(separator: " ")
    }

    static func parallelHits(_ text: String, _ topk: Int) -> [SearchHit] {
        var out: [SearchHit] = []
        let obj = try? JSONSerialization.jsonObject(with: Data(text.utf8))
        if let root = obj as? [String: Any],
           let items = root["results"] as? [[String: Any]] {
            for item in items where out.count < topk {
                out.append(SearchHit(
                    title: item["title"] as? String ?? "",
                    url: item["url"] as? String ?? "",
                    extract: excerpt(item["excerpts"] as? [String] ?? [])))
            }
        }
        return out
    }

    static func mwmblHits(_ data: Data, _ topk: Int) -> [SearchHit] {
        var out: [SearchHit] = []
        let obj = try? JSONSerialization.jsonObject(with: data)
        if let arr = obj as? [[String: Any]] {
            for item in arr where out.count < topk {
                out.append(SearchHit(title: spanText(item["title"]),
                                     url: item["url"] as? String ?? "",
                                     extract: spanText(item["extract"])))
            }
        }
        return out
    }

    static func spanText(_ v: Any?) -> String {
        var text = ""
        if let spans = v as? [[String: Any]] {
            for span in spans {
                if let value = span["value"] as? String { text += value }
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func mwmbl(_ query: String,
                              count: Int) async -> [SearchHit] {
        var out: [SearchHit] = []
        var comps = URLComponents(
            string: "https://api.mwmbl.org/api/v1/search/")
        comps?.queryItems = [URLQueryItem(name: "s", value: query)]
        if let url = comps?.url {
            do {
                let (data, resp) = try await URLSession.shared.data(from: url)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                if code == 200 { out = mwmblHits(data, count) }
                if out.isEmpty {
                    Tools.diag("web_search mwmbl \"\(query)\" -> "
                        + "HTTP \(code), \(data.count) bytes")
                }
            } catch {
                Tools.diag("web_search mwmbl \"\(query)\" transport error: "
                    + error.localizedDescription)
            }
        }
        return out
    }
}

struct MCPTool: Sendable {

    let name: String
    private let arrays: [String]
    private let scalars: [String]

    init(_ tools: [[String: Any]], preferring wanted: String) {
        let chosen = tools.first(where: { tool in
            tool["name"] as? String == wanted
        }) ?? tools.first ?? [:]
        let schema = chosen["inputSchema"] as? [String: Any] ?? [:]
        let required = schema["required"] as? [String] ?? []
        let properties = schema["properties"] as? [String: Any] ?? [:]
        var wantsArray: [String] = []
        var wantsScalar: [String] = []
        for key in required {
            let spec = properties[key] as? [String: Any] ?? [:]
            if spec["type"] as? String == "array" {
                wantsArray.append(key)
            } else {
                wantsScalar.append(key)
            }
        }
        name = chosen["name"] as? String ?? wanted
        arrays = wantsArray
        scalars = wantsScalar
    }

    func arguments(_ query: String) -> [String: Any] {
        var out: [String: Any] = [:]
        for key in arrays { out[key] = [query] }
        for key in scalars { out[key] = query }
        return out
    }
}

actor ParallelSearch {

    static let shared = ParallelSearch()

    private static let endpoint =
        URL(string: "https://search.parallel.ai/mcp")!
    private static let version = "2025-06-18"
    private static let timeout: TimeInterval = 10

    private var session = ""
    private var spec: MCPTool?
    private var serial = 0

    func search(_ query: String, count: Int) async -> [SearchHit] {
        var out: [SearchHit] = []
        do {
            let tool = try await ready()
            let answer = try await rpc("tools/call",
                                       ["name": tool.name,
                                        "arguments": tool.arguments(query)])
            out = WebSearch.parallelHits(ParallelSearch.text(answer), count)
            if out.isEmpty {
                Tools.diag("web_search parallel \"\(query)\" -> no results")
            }
        } catch {
            session = ""
            spec = nil
            Tools.diag("web_search parallel \"\(query)\" failed: "
                + error.localizedDescription)
        }
        return out
    }

    private func ready() async throws -> MCPTool {
        var tool = spec
        if tool == nil {
            _ = try await rpc("initialize", handshake())
            _ = try await exchange(["jsonrpc": "2.0",
                                    "method": "notifications/initialized"])
            let listed = try await rpc("tools/list", [:])
            let made = MCPTool(listed["tools"] as? [[String: Any]] ?? [],
                               preferring: "web_search")
            spec = made
            tool = made
        }
        return tool ?? MCPTool([], preferring: "web_search")
    }

    private func rpc(_ method: String,
                     _ params: [String: Any]) async throws -> [String: Any] {
        serial += 1
        let message = try await exchange(["jsonrpc": "2.0", "id": serial,
                                          "method": method,
                                          "params": params])
        if message["error"] is [String: Any] {
            throw SearchError(text: "\(method) refused")
        }
        return message["result"] as? [String: Any] ?? [:]
    }

    private func exchange(_ body: [String: Any]) async throws
        -> [String: Any] {
        let data = try JSONSerialization.data(withJSONObject: body)
        let (answer, response) = try await URLSession.shared.data(
            for: request(data))
        let http = response as? HTTPURLResponse
        let code = http?.statusCode ?? 0
        if !(200 ... 299).contains(code) {
            throw SearchError(text: "HTTP \(code)")
        }
        if session.isEmpty {
            session = http?.value(forHTTPHeaderField: "Mcp-Session-Id") ?? ""
        }
        return ParallelSearch.decode(answer)
    }

    private func request(_ body: Data) -> URLRequest {
        var out = URLRequest(url: ParallelSearch.endpoint)
        out.httpMethod = "POST"
        out.httpBody = body
        out.timeoutInterval = ParallelSearch.timeout
        out.setValue("application/json", forHTTPHeaderField: "Content-Type")
        out.setValue("application/json, text/event-stream",
                     forHTTPHeaderField: "Accept")
        out.setValue(ParallelSearch.version,
                     forHTTPHeaderField: "MCP-Protocol-Version")
        if !session.isEmpty {
            out.setValue(session, forHTTPHeaderField: "Mcp-Session-Id")
        }
        return out
    }

    private func handshake() -> [String: Any] {
        ["protocolVersion": ParallelSearch.version,
         "capabilities": [:],
         "clientInfo": ["name": "ChatOKF", "version": "1.0"]]
    }

    static func decode(_ answer: Data) -> [String: Any] {
        let text = String(decoding: answer, as: UTF8.self)
        var body = text
        if text.hasPrefix("event:") || text.hasPrefix("data:") {
            body = text.split(separator: "\n").filter { line in
                line.hasPrefix("data: ")
            }.map { line in
                String(line.dropFirst(6))
            }.joined()
        }
        let object = try? JSONSerialization.jsonObject(
            with: Data(body.utf8))
        return object as? [String: Any] ?? [:]
    }

    static func text(_ result: [String: Any]) -> String {
        var out = ""
        for item in result["content"] as? [[String: Any]] ?? [] {
            out += item["text"] as? String ?? ""
        }
        return out
    }
}
