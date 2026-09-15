import Foundation

public enum Texts {

    public static func text(_ name: String, trimmed: Bool = true) -> String {
        let url = Bundle.main.url(forResource: name, withExtension: "txt")
        let raw = url.flatMap { u in
            try? String(contentsOf: u, encoding: .utf8)
        } ?? ""
        return trimmed ? raw.trimmingCharacters(in: .whitespacesAndNewlines)
                       : raw
    }

}
