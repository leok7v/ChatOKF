import Foundation

// Runs BEFORE the phonemizer, whose tokenizer splits "2,925.26" into three
// cardinals; the decimal separator comes from the string, never the Locale.
public enum SpokenNumbers {

    public static func expand(_ line: String) -> String {
        let c = Array(line)
        var out = ""
        var i = 0
        while i < c.count {
            let amount = amount(c, i)
            let run = amount == nil ? numeral(c, i) : nil
            if let amount {
                out += amount.text
                i = amount.next
            } else if let run {
                out += run.text
                i = run.next
            } else {
                out.append(c[i])
                i += 1
            }
        }
        return out
    }

    private static func amount(_ c: [Character], _ i: Int)
        -> (text: String, next: Int)? {
        var result: (text: String, next: Int)? = nil
        var at = i
        var negative = false
        if c[at] == "-" && at + 1 < c.count {
            negative = true
            at += 1
        }
        if let unit = currencies[c[at]] {
            at += 1
            if at < c.count && c[at] == " " { at += 1 }
            if at < c.count && c[at] == "-" {
                negative = true
                at += 1
            }
            if at < c.count && c[at].isNumber {
                var end = at
                while end < c.count
                    && (c[end].isNumber || c[end] == "." || c[end] == ",") {
                    end += 1
                }
                while end > at && !c[end - 1].isNumber { end -= 1 }
                if !(end < c.count && c[end].isLetter) {
                    let token = String(c[at..<end])
                    // "$1.5 million": the unit follows the scale, and the
                    // fraction is a quantity, not a count of cents.
                    let scale = scaleWord(c, end)
                    var text = ""
                    if let scale {
                        text = say(token) + " " + scale.word + " "
                            + unit.majorPlural
                    } else {
                        text = money(token, unit)
                    }
                    if negative { text = "minus " + text }
                    result = (text, scale?.next ?? end)
                }
            }
        }
        return result
    }

    // A digit touching letters is a word to the rules engine:
    // "H2O", "3D" and "26th" are not quantities.
    private static func numeral(_ c: [Character], _ i: Int)
        -> (text: String, next: Int)? {
        var result: (text: String, next: Int)? = nil
        let afterLetter = i > 0 && (c[i - 1].isLetter || c[i - 1].isNumber)
        var start = i
        var negative = false
        if c[i] == "-" && i + 1 < c.count && c[i + 1].isNumber
            && !(i > 0 && c[i - 1].isLetter) {
            negative = true
            start = i + 1
        }
        if !afterLetter && start < c.count && c[start].isNumber {
            var end = start
            while end < c.count
                && (c[end].isNumber || c[end] == "." || c[end] == ",") {
                end += 1
            }
            // A trailing separator belongs to the sentence, not the number.
            while end > start && !c[end - 1].isNumber { end -= 1 }
            let followsLetter = end < c.count && c[end].isLetter
            if !followsLetter {
                var text = say(String(c[start..<end]))
                if negative { text = "minus " + text }
                var next = end
                // A trailing sign is silent otherwise, and a forecast is
                // mostly signs: the degree sign and the F would be lost.
                let whole = split(String(c[start..<end])).digits
                let one = (Int(whole) ?? 0) == 1
                if next < c.count && c[next] == "%" {
                    text += " percent"
                    next += 1
                } else if next < c.count && c[next] == "\u{00A2}" {
                    text += one ? " cent" : " cents"
                    next += 1
                } else if next < c.count && c[next] == "\u{00B0}" {
                    text += one ? " degree" : " degrees"
                    next += 1
                    if next < c.count && (c[next] == "F" || c[next] == "C") {
                        text += c[next] == "F" ? " Fahrenheit" : " Celsius"
                        next += 1
                    }
                }
                result = (text, next)
            }
        }
        return result
    }

    // `bare` is false once any separator appeared, which rules a year out.
    static func split(_ token: String)
        -> (digits: String, frac: String, bare: Bool) {
        let dot = token.contains(".")
        let comma = token.contains(",")
        var decimal: Character? = nil
        if dot && comma {
            // Both present: the LAST one separates the fraction. Locale-free,
            // and right for "2,925.26" and "1.234.567,89" alike.
            decimal = token.lastIndex(of: ".")! > token.lastIndex(of: ",")!
                ? "." : ","
        } else if dot || comma {
            decimal = soleSeparatorIsDecimal(token, dot ? "." : ",")
                ? (dot ? "." : ",") : nil
        }
        var whole = token
        var fraction = ""
        if let decimal, let at = token.lastIndex(of: decimal) {
            whole = String(token[token.startIndex..<at])
            fraction = String(token[token.index(after: at)...])
        }
        return (whole.filter { ch in ch.isNumber },
                fraction.filter { ch in ch.isNumber },
                !whole.contains(",") && !whole.contains("."))
    }

    static func say(_ token: String) -> String {
        let parts = split(token)
        let digits = parts.digits
        let frac = parts.frac
        let bare = parts.bare
        var out = ""
        // A bare four-digit integer in 1100...2099 reads as a year; grouping
        // separators rule it out, because a year is never written with them.
        let value = Int(digits) ?? 0
        if frac.isEmpty && bare && digits.count == 4
            && value >= 1100 && value <= 2099 {
            out = year(value)
        } else {
            out = cardinal(digits)
            if !frac.isEmpty { out += " point " + spelled(frac) }
        }
        return out
    }

    // One separator is ambiguous: exactly three digits after it is grouping
    // ("1,234", "1.234.567"); anything else is a fraction ("1,5", "0.75").
    private static func soleSeparatorIsDecimal(_ token: String,
                                               _ sep: Character) -> Bool {
        let parts = token.split(separator: sep, omittingEmptySubsequences: false)
        var grouping = parts.count >= 2 && !parts[0].isEmpty
            && parts[0].count <= 3
        for k in 1..<max(parts.count, 1) where grouping {
            if parts[k].count != 3 { grouping = false }
        }
        return !grouping
    }

    // The symbol is silent otherwise, so "$5" would read as a bare five. One
    // shape serves every currency: the reading is English whatever the region.
    struct Currency {
        let major: String
        let majorPlural: String
        let minor: String        // empty where the currency has no subunit
        let minorPlural: String
    }

    static let currencies: [Character: Currency] = [
        "$": Currency(major: "dollar", majorPlural: "dollars",
                      minor: "cent", minorPlural: "cents"),
        "£": Currency(major: "pound", majorPlural: "pounds",
                      minor: "penny", minorPlural: "pence"),
        "€": Currency(major: "euro", majorPlural: "euros",
                      minor: "cent", minorPlural: "cents"),
        "₹": Currency(major: "rupee", majorPlural: "rupees",
                      minor: "paisa", minorPlural: "paise"),
        "₽": Currency(major: "ruble", majorPlural: "rubles",
                      minor: "kopeck", minorPlural: "kopecks"),
        "¥": Currency(major: "yen", majorPlural: "yen",
                      minor: "", minorPlural: ""),
        "₩": Currency(major: "won", majorPlural: "won",
                      minor: "", minorPlural: ""),
    ]

    private static func unitName(_ n: Int, _ one: String,
                                 _ many: String) -> String {
        n == 1 ? one : many
    }

    private static let scaleWords = [
        "thousand", "million", "billion", "trillion",
    ]

    // A magnitude word after the figure, as amounts past a million are written.
    private static func scaleWord(_ c: [Character], _ from: Int)
        -> (word: String, next: Int)? {
        var at = from
        while at < c.count && c[at] == " " { at += 1 }
        var end = at
        while end < c.count && c[end].isLetter { end += 1 }
        let word = String(c[at..<end]).lowercased()
        var result: (word: String, next: Int)? = nil
        if end > at && scaleWords.contains(word) {
            result = (word, end)
        }
        return result
    }

    // The numeric VALUE as a cardinal, so a zero-padded minor ("01") reads
    // "one cent" rather than the identifier spelling "zero one".
    private static func value(_ digits: String) -> String {
        cardinal(String(Int(digits) ?? 0))
    }

    static func money(_ token: String, _ unit: Currency) -> String {
        let parts = split(token)
        let major = Int(parts.digits) ?? 0
        var out = ""
        // Exactly two fraction digits are subunits; "$1.5" is a quantity of
        // dollars, and "fifty cents" from one digit would be a guess.
        if parts.frac.count == 2 && !unit.minor.isEmpty {
            let minor = Int(parts.frac) ?? 0
            let majorWords = value(parts.digits) + " "
                + unitName(major, unit.major, unit.majorPlural)
            let minorWords = value(parts.frac) + " "
                + unitName(minor, unit.minor, unit.minorPlural)
            if major == 0 && minor > 0 {
                out = minorWords
            } else if minor == 0 {
                out = majorWords
            } else {
                out = majorWords + " and " + minorWords
            }
        } else if parts.frac.isEmpty {
            out = value(parts.digits) + " "
                + unitName(major, unit.major, unit.majorPlural)
        } else {
            out = say(token) + " " + unit.majorPlural
        }
        return out
    }

    private static let ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    private static let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    ]

    // Ascending, so the index is the power of a thousand.
    private static let scales = [
        "", " thousand", " million", " billion", " trillion",
        " quadrillion",
    ]

    static func spelled(_ digits: String) -> String {
        var parts: [String] = []
        for ch in digits {
            parts.append(ones[Int(String(ch)) ?? 0])
        }
        return parts.joined(separator: " ")
    }

    private static func under1000(_ n: Int) -> String {
        var out = ""
        var rest = n
        if rest >= 100 {
            out = ones[rest / 100] + " hundred"
            rest %= 100
            if rest > 0 { out += " " }
        }
        if rest >= 20 {
            out += tens[rest / 10]
            if rest % 10 > 0 { out += "-" + ones[rest % 10] }
        } else if rest > 0 || n == 0 {
            out += ones[rest]
        }
        return out
    }

    static func cardinal(_ digits: String) -> String {
        // Leading zeros are an identifier, not a quantity ("007", "0123"),
        // and a run too long for Int is one too.
        let trimmed = String(digits.drop(while: { ch in ch == "0" }))
        var out = ""
        if trimmed.isEmpty {
            out = digits.isEmpty ? "" : "zero"
        } else if digits.count != trimmed.count || digits.count > 18 {
            out = spelled(digits)
        } else {
            var groups: [Int] = []
            var rest = Int(trimmed) ?? 0
            while rest > 0 {
                groups.append(rest % 1000)
                rest /= 1000
            }
            var pieces: [String] = []
            var k = groups.count - 1
            while k >= 0 {
                if groups[k] > 0 && k < scales.count {
                    pieces.append(under1000(groups[k]) + scales[k])
                }
                k -= 1
            }
            out = pieces.joined(separator: " ")
        }
        return out
    }

    static func year(_ n: Int) -> String {
        let high = n / 100
        let low = n % 100
        var out = ""
        if low == 0 {
            out = under1000(high) + " hundred"
        } else if high % 10 == 0 && low < 10 {
            // "twenty oh five" is a style this does not take.
            out = cardinal(String(n))
        } else if low < 10 {
            out = under1000(high) + " oh " + ones[low]
        } else {
            out = under1000(high) + " " + under1000(low)
        }
        return out
    }
}
