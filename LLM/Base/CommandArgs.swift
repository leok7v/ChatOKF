import Foundation

public final class CommandArgs {

    private let words: [String]
    private var taken: Set<Int> = []

    public init(_ words: [String]) { self.words = words }

    public var all: [String] { words }

    private func at(_ name: String) -> Int? { words.firstIndex(of: name) }

    public func flag(_ name: String) -> Bool {
        let i = at(name)
        if let i { taken.insert(i) }
        return i != nil
    }

    public func consume(_ name: String, hasValue: Bool = false) {
        if let i = at(name) {
            taken.insert(i)
            if hasValue && i + 1 < words.count { taken.insert(i + 1) }
        }
    }

    public func values(_ name: String, _ n: Int) -> [String] {
        var out: [String] = []
        if n > 0, let i = at(name), i + n < words.count,
           !words[(i + 1)...(i + n)].contains(where: { w in
               w.hasPrefix("--")
           }) {
            taken.insert(i)
            for k in 1...n {
                taken.insert(i + k)
                out.append(words[i + k])
            }
        }
        return out
    }

    public func value(_ name: String) -> String? {
        values(name, 1).first
    }

    public func parsed<T>(_ name: String, _ parse: (String) -> T?) -> T? {
        let i = at(name)
        let raw = i.flatMap { i in i + 1 < words.count ? words[i + 1] : nil }
        let out = raw.flatMap(parse)
        if let i, out != nil {
            taken.insert(i)
            taken.insert(i + 1)
        }
        return out
    }

    public func int(_ name: String) -> Int? { parsed(name, Int.init) }

    public func float(_ name: String) -> Float? { parsed(name, Float.init) }

    public func double(_ name: String) -> Double? {
        parsed(name, Double.init)
    }

    public func uint64(_ name: String) -> UInt64? {
        parsed(name, UInt64.init)
    }

    public func text(_ name: String) -> String? {
        value(name).map(CommandArgs.resolve)
    }

    public static func resolve(_ v: String) -> String {
        var out = v
        if v.hasPrefix("@") {
            let path = String(v.dropFirst())
            out = (try? String(contentsOfFile: path, encoding: .utf8)) ?? v
        }
        return out
    }

    public func rest(after name: String) -> [String] {
        var out: [String] = []
        if let i = at(name) {
            taken.insert(i)
            for k in (i + 1)..<words.count {
                taken.insert(k)
                out.append(words[k])
            }
        }
        return out
    }

    public var rest: [String] {
        var out: [String] = []
        for (i, w) in words.enumerated()
        where i > 0 && !taken.contains(i) && !w.hasPrefix("--") {
            out.append(w)
        }
        return out
    }

    public var turns: [String] { rest.dropFirst().map(CommandArgs.resolve) }
}
