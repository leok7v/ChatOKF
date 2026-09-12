import Foundation

struct SpecialIndex: Sendable {
    private let specials: [(text: String, id: Int32)]
    private let heads: Set<Character>

    init(_ specials: [(text: String, id: Int32)]) {
        self.specials = specials
        var found = Set<Character>()
        for (text, _) in specials {
            if let head = text.first { found.insert(head) }
        }
        heads = found
    }

    func first(
        in s: Substring
    ) -> (range: Range<Substring.Index>, id: Int32)? {
        var best: (Range<Substring.Index>, Int32)? = nil
        var i = s.startIndex
        while i < s.endIndex && best == nil {
            if heads.contains(s[i]) {
                for (text, id) in specials where best == nil {
                    if s[i...].hasPrefix(text) {
                        best = (i ..< s.index(i, offsetBy: text.count), id)
                    }
                }
            }
            i = s.index(after: i)
        }
        return best.map { hit in (range: hit.0, id: hit.1) }
    }
}
