import Foundation

public enum AttachmentRefs {

    private static let open: Character = "\u{2063}"
    private static let close: Character = "\u{2064}"

    public static func token(_ name: String) -> String {
        "\(open)@\(name)\(close)"
    }

    public static func insert(_ name: String, into text: String, at offset: Int)
        -> (text: String, caret: Int) {
        let clamped = max(0, min(offset, text.utf16.count))
        let at = String.Index(utf16Offset: clamped, in: text)
        let before = at > text.startIndex ? text[text.index(before: at)] : " "
        let after = at < text.endIndex ? text[at] : "x"
        let lead = " \n".contains(before) ? "" : " "
        let trail = " \n".contains(after) ? "" : " "
        let piece = lead + token(name) + trail
        var out = text
        out.insert(contentsOf: piece, at: at)
        return (out, clamped + piece.utf16.count)
    }

    public static func names(in text: String) -> [String] {
        var result: [String] = []
        var pending: String? = nil
        for ch in text {
            if ch == open {
                pending = ""
            } else if ch == close {
                if let inner = pending {
                    result.append(String(inner.drop(while: { c in c == "@" })))
                }
                pending = nil
            } else if pending != nil {
                pending?.append(ch)
            }
        }
        return result
    }

    public static func stripped(_ text: String) -> String {
        String(text.filter { c in c != open && c != close })
    }

    public static func scrub(_ name: String, from text: String) -> String {
        var out = text
        for variant in [token(name), "\(open)@\(name)", "@\(name)\(close)"] {
            out = out.replacingOccurrences(of: variant, with: "")
        }
        return out
    }

    public static func substitute(_ text: String,
                           _ replace: (String) -> String) -> String {
        var out = ""
        var pending: String? = nil
        for ch in text {
            if ch == open {
                pending = ""
            } else if ch == close {
                if let inner = pending {
                    out += replace(String(inner.drop(while: { c in c == "@" })))
                }
                pending = nil
            } else if pending != nil {
                pending?.append(ch)
            } else {
                out.append(ch)
            }
        }
        if let inner = pending { out += inner }
        return out
    }

}
