import Foundation

public enum Sentences {

    public static let maxLength = 240
    private static let abbreviations: Set<String> = [
        "e.g", "i.e", "etc", "vs", "cf", "ca", "approx", "fig", "eq", "no",
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "inc", "ltd"]

    public static func split(_ text: String) -> [String] {
        var parts: [String] = []
        let bytes = text.utf8
        var start = bytes.startIndex
        var word = bytes.startIndex
        var length = 0
        var i = bytes.startIndex
        while i < bytes.endIndex {
            let b = bytes[i]
            let blank = b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D
            let cut = b == 0x0A || (blank && (length >= maxLength ||
                                              terminal(text[word..<i])))
            if cut {
                parts.append(String(text[start..<i]))
                start = bytes.index(after: i)
                length = 0
            } else {
                length += 1
            }
            if blank { word = bytes.index(after: i) }
            i = bytes.index(after: i)
        }
        parts.append(String(text[start...]))
        let trimmed = parts.map { part in
            part.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed.filter { part in !part.isEmpty }
    }

    private static func terminal(_ word: Substring) -> Bool {
        let bare = word.reversed().drop(while: { ch in "\"')]".contains(ch) })
        var result = false
        if let mark = bare.first, ".!?".contains(mark) {
            let stem = String(bare.dropFirst().reversed()).lowercased()
            let named = stem.count > 1 && !abbreviations.contains(stem) &&
                        !stem.allSatisfy { ch in ch.isNumber }
            result = mark != "." || named
        }
        return result
    }

}
