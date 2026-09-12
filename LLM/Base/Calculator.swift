import Foundation

public actor Calculator {
    private var context: [String: Cx] = [:]

    private static let constants: [String: Cx] = [
        "pi": Cx(.pi), "e": Cx(M_E), "tau": Cx(.pi * 2), "i": Cx(0, 1),
    ]

    public init() {}

    public func reset() { context.removeAll() }

    public func evaluate(_ input: String) -> String {
        // Models write math.pow, Math.PI, np.sqrt and end with '= ?'; both are
        // noise to this grammar, not an assignment.
        let text = input
            .replacingOccurrences(of: #"(?i)\b(math|np)\."#, with: "",
                                  options: .regularExpression)
            .replacingOccurrences(of: #"\s*(=\s*\?+|\?+|=)\s*$"#, with: "",
                                  options: .regularExpression)
            .replacingOccurrences(of: "$", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var result = "error: empty expression"
        if !text.isEmpty {
            if let eq = text.firstIndex(of: "=") {
                let name = text[..<eq].trimmingCharacters(in: .whitespaces)
                let rhs = String(text[text.index(after: eq)...])
                let tail = rhs.trimmingCharacters(in: .whitespaces)
                if name == "i" || tail == "i" {
                    result = "error: 'i' is the imaginary unit and cannot "
                        + "be assigned"
                } else if Calculator.isName(name) {
                    result = assign(name, rhs)
                } else if Calculator.isName(tail) {
                    // Models name the RESULT after the expression: 350 * 0.4 =
                    // X1.
                    result = assign(tail, name)
                } else {
                    result = "error: this evaluates expressions, it does not "
                        + "solve equations. '\(name)' is not a variable name. "
                        + "Rearrange it yourself and send only the right-hand "
                        + "side, e.g. '(1.10 - 1) / 2'. Write every "
                        + "multiplication explicitly: '2 * x', not '2x'."
                }
            } else {
                result = evaluated(text)
            }
        }
        return result
    }

    private func assign(_ name: String, _ expr: String) -> String {
        let value = Calculator.value(of: expr, vars: context)
        var result = "error: cannot evaluate the expression"
        if let value, value.isFinite {
            context[name.lowercased()] = value
            result = Calculator.format(value)
        } else if value != nil {
            result = "error: result is not a finite number"
        }
        return result
    }

    private func evaluated(_ expr: String) -> String {
        let value = Calculator.value(of: expr, vars: context)
        var result = "error: cannot evaluate the expression"
        if let value {
            result = Calculator.format(value)
        }
        return result
    }

    static func value(of expr: String,
                      vars: [String: Cx]) -> Cx? {
        Parser(tokens: lex(expr), vars: vars).parse()
    }

    static func isName(_ s: String) -> Bool {
        var result = !s.isEmpty
        for (i, c) in s.enumerated() {
            let head = c == "_" || c.isLetter
            result = result && (i == 0 ? head : head || c.isNumber)
        }
        return result
    }

    static func resolve(_ name: String, _ vars: [String: Cx]) -> Cx? {
        let key = name.lowercased()
        return vars[key] ?? constants[key]
    }

    // Case-folded. log is base 10, log(x, b) base b, ln natural on the
    // principal branch; nil for an unknown name, wrong arity or a complex arg.
    static func apply(_ fn: String, _ a: [Cx]) -> Cx? {
        let name = fn.lowercased()
        var r: Cx? = nil
        switch (name, a.count) {
        case ("sin", 1): r = Cx.sin(a[0])
        case ("cos", 1): r = Cx.cos(a[0])
        case ("tan", 1): r = Cx.tan(a[0])
        case ("sinh", 1): r = Cx.sinh(a[0])
        case ("cosh", 1): r = Cx.cosh(a[0])
        case ("tanh", 1): r = Cx.tanh(a[0])
        case ("exp", 1): r = Cx.exp(a[0])
        case ("ln", 1): r = Cx.ln(a[0])
        case ("log", 1): r = Cx.div(Cx.ln(a[0]), Cx(Foundation.log(10.0)))
        case ("log", 2): r = Cx.div(Cx.ln(a[0]), Cx.ln(a[1]))
        case ("sqrt", 1): r = Cx.sqrt(a[0])
        case ("pow", 2): r = Cx.pow(a[0], a[1])
        case ("abs", 1): r = Cx(Foundation.hypot(a[0].re, a[0].im))
        case ("re", 1): r = Cx(a[0].re)
        case ("im", 1): r = Cx(a[0].im)
        case ("conj", 1): r = Cx(a[0].re, -a[0].im)
        case ("arg", 1): r = Cx(Foundation.atan2(a[0].im + 0, a[0].re))
        default: r = applyReal(name, a)
        }
        return r
    }

    private static func applyReal(_ name: String, _ a: [Cx]) -> Cx? {
        var r: Double? = nil
        if a.allSatisfy({ v in v.im == 0 }) {
            let x = a.map { v in v.re }
            switch (name, x.count) {
            case ("asin", 1): r = asin(x[0])
            case ("acos", 1): r = acos(x[0])
            case ("atan", 1): r = atan(x[0])
            case ("log2", 1): r = log2(x[0])
            case ("log10", 1): r = log10(x[0])
            case ("cbrt", 1): r = cbrt(x[0])
            case ("floor", 1): r = floor(x[0])
            case ("ceil", 1): r = ceil(x[0])
            case ("round", 1): r = x[0].rounded()
            case ("round", 2):
                let scale = Foundation.pow(10.0, x[1].rounded())
                r = (x[0] * scale).rounded() / scale
            case ("trunc", 1): r = trunc(x[0])
            case ("sign", 1): r = x[0] > 0 ? 1 : (x[0] < 0 ? -1 : 0)
            case ("hypot", 2): r = hypot(x[0], x[1])
            case ("atan2", 2): r = atan2(x[0], x[1])
            case ("mod", 2):
                r = x[1] == 0 ? .nan
                    : x[0].truncatingRemainder(dividingBy: x[1])
            case ("min", 2): r = Swift.min(x[0], x[1])
            case ("max", 2): r = Swift.max(x[0], x[1])
            default: r = nil
            }
        }
        return r.map { v in Cx(v) }
    }

    static func format(_ v: Cx) -> String {
        var result: String
        let z = snapped(v)
        if z.re.isNaN || z.im.isNaN {
            result = "error: undefined result"
        } else if z.re.isInfinite || z.im.isInfinite {
            result = "error: result is infinite"
        } else if z.im == 0 {
            result = formatReal(z.re)
        } else if z.re == 0 {
            result = formatImaginary(z.im)
        } else {
            let sign = z.im < 0 ? " - " : " + "
            result = formatReal(z.re) + sign + formatImaginary(abs(z.im))
        }
        return result
    }

    private static func snapped(_ v: Cx) -> Cx {
        var z = v
        if z.im != 0 {
            let scale = Swift.max(abs(z.re), abs(z.im), 1)
            if abs(z.im) <= 1e-12 * scale { z.im = 0 }
            if abs(z.re) <= 1e-12 * scale { z.re = 0 }
        }
        return z
    }

    // 12 significant digits: binary dust like 192.50000000000003 rounds away.
    private static func formatReal(_ v: Double) -> String {
        v == v.rounded() && abs(v) < 1e15
            ? String(Int64(v))
            : String(format: "%.12g", v)
    }

    private static func formatImaginary(_ v: Double) -> String {
        var result = formatReal(v) + "i"
        if v == 1 {
            result = "i"
        } else if v == -1 {
            result = "-i"
        }
        return result
    }
}

struct Cx: Equatable {
    var re: Double
    var im: Double

    init(_ re: Double, _ im: Double = 0) {
        self.re = re
        self.im = im
    }

    var isFinite: Bool { re.isFinite && im.isFinite }

    static func add(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re + b.re, a.im + b.im)
    }

    static func sub(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re - b.re, a.im - b.im)
    }

    static func mul(_ a: Cx, _ b: Cx) -> Cx {
        Cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }

    static func div(_ a: Cx, _ b: Cx) -> Cx {
        let den = b.re * b.re + b.im * b.im
        return Cx((a.re * b.re + a.im * b.im) / den,
                  (a.im * b.re - a.re * b.im) / den)
    }

    static func neg(_ a: Cx) -> Cx { Cx(-a.re, -a.im) }

    static func exp(_ z: Cx) -> Cx {
        let r = Foundation.exp(z.re)
        return z.im == 0
            ? Cx(r)
            : Cx(r * Foundation.cos(z.im), r * Foundation.sin(z.im))
    }

    // Principal branch. The `+ 0` normalizes a negative zero, which would flip
    // atan2 onto the -pi branch after a unary minus on a real.
    static func ln(_ z: Cx) -> Cx {
        z.im == 0 && z.re > 0
            ? Cx(Foundation.log(z.re))
            : Cx(Foundation.log(Foundation.hypot(z.re, z.im)),
                 Foundation.atan2(z.im + 0, z.re))
    }

    static func sqrt(_ z: Cx) -> Cx {
        let result: Cx
        if z.im == 0 {
            result = z.re >= 0
                ? Cx(Foundation.sqrt(z.re))
                : Cx(0, Foundation.sqrt(-z.re))
        } else {
            let r = Foundation.hypot(z.re, z.im)
            let s = z.im < 0 ? -1.0 : 1.0
            result = Cx(Foundation.sqrt((r + z.re) / 2),
                        s * Foundation.sqrt((r - z.re) / 2))
        }
        return result
    }

    static func pow(_ z: Cx, _ w: Cx) -> Cx {
        let result: Cx
        let realPow = z.im == 0 && w.im == 0
            ? Foundation.pow(z.re, w.re) : Double.nan
        if z.im == 0 && w.im == 0 && (!realPow.isNaN || z.re >= 0) {
            result = Cx(realPow)
        } else if z.re == 0 && z.im == 0 {
            result = w.im == 0 && w.re > 0 ? Cx(0) : Cx(.nan)
        } else {
            result = exp(mul(w, ln(z)))
        }
        return result
    }

    static func sin(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.sin(z.re))
            : Cx(Foundation.sin(z.re) * Foundation.cosh(z.im),
                 Foundation.cos(z.re) * Foundation.sinh(z.im))
    }

    static func cos(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.cos(z.re))
            : Cx(Foundation.cos(z.re) * Foundation.cosh(z.im),
                 -Foundation.sin(z.re) * Foundation.sinh(z.im))
    }

    static func tan(_ z: Cx) -> Cx {
        z.im == 0 ? Cx(Foundation.tan(z.re)) : div(sin(z), cos(z))
    }

    static func sinh(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.sinh(z.re))
            : Cx(Foundation.sinh(z.re) * Foundation.cos(z.im),
                 Foundation.cosh(z.re) * Foundation.sin(z.im))
    }

    static func cosh(_ z: Cx) -> Cx {
        z.im == 0
            ? Cx(Foundation.cosh(z.re))
            : Cx(Foundation.cosh(z.re) * Foundation.cos(z.im),
                 Foundation.sinh(z.re) * Foundation.sin(z.im))
    }

    static func tanh(_ z: Cx) -> Cx {
        z.im == 0 ? Cx(Foundation.tanh(z.re)) : div(sinh(z), cosh(z))
    }
}

private enum Tok: Equatable {
    case num(Double)
    case name(String)
    case sym(Character)
    case bad
}

private func lex(_ s: String) -> [Tok] {
    var out: [Tok] = []
    let c = Array(s)
    let n = c.count
    var i = 0
    while i < n {
        let ch = c[i]
        if ch == " " || ch == "\t" || ch == "\n" || ch == "\r" {
            i += 1
        } else if ch.isNumber || ch == "." {
            var j = i
            while j < n && c[j].isNumber { j += 1 }
            if j < n && c[j] == "." {
                j += 1
                while j < n && c[j].isNumber { j += 1 }
            }
            if j < n && (c[j] == "e" || c[j] == "E") {
                var k = j + 1
                if k < n && (c[k] == "+" || c[k] == "-") { k += 1 }
                if k < n && c[k].isNumber {
                    j = k + 1
                    while j < n && c[j].isNumber { j += 1 }
                }
            }
            out.append(Double(String(c[i..<j])).map(Tok.num) ?? .bad)
            i = j
        } else if ch.isLetter || ch == "_" {
            var j = i
            while j < n && (c[j].isLetter || c[j].isNumber || c[j] == "_") {
                j += 1
            }
            out.append(.name(String(c[i..<j])))
            i = j
        } else if ch == "*" && i + 1 < n && c[i + 1] == "*" {
            // Python's power operator, which models emit constantly (i**i).
            out.append(.sym("^"))
            i += 2
        } else if ch == ";" {
            out.append(.sym(","))
            i += 1
        } else if "+-*/^(),%".contains(ch) {
            out.append(.sym(ch))
            i += 1
        } else {
            out.append(.bad)
            i += 1
        }
    }
    return out
}

private final class Parser {
    private let toks: [Tok]
    private let vars: [String: Cx]
    private var pos = 0

    init(tokens: [Tok], vars: [String: Cx]) {
        self.toks = tokens
        self.vars = vars
    }

    func parse() -> Cx? {
        let value = expr()
        return (value != nil && pos == toks.count) ? value : nil
    }

    private func peek() -> Tok? { pos < toks.count ? toks[pos] : nil }

    private func expr() -> Cx? {
        var result = term()
        while result != nil, let t = peek(),
              t == .sym("+") || t == .sym("-") {
            pos += 1
            let rhs = term()
            if let l = result, let r = rhs {
                result = t == .sym("+") ? Cx.add(l, r) : Cx.sub(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    private func termOp() -> Character? {
        var result: Character? = nil
        switch peek() {
        case .sym("*"), .sym("/"):
            pos += 1
            if case .sym(let ch) = toks[pos - 1] { result = ch }
        case .name, .sym("("):
            result = "*"
        default:
            result = nil
        }
        return result
    }

    private func term() -> Cx? {
        var result = factor()
        while result != nil, let op = termOp() {
            let rhs = factor()
            if let l = result, let r = rhs {
                result = op == "/" ? Cx.div(l, r) : Cx.mul(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    private func factor() -> Cx? {
        var result: Cx? = nil
        if let t = peek(), t == .sym("-") || t == .sym("+") {
            pos += 1
            let v = factor()
            if let v { result = t == .sym("-") ? Cx.neg(v) : v }
        } else {
            result = power()
        }
        return result
    }

    // 5% is 0.05 only when no operand follows; 22 % 7 stays rejected rather
    // than being misread as 0.22 * 7.
    private func percentAhead() -> Bool {
        var result = peek() == .sym("%")
        if result, pos + 1 < toks.count {
            switch toks[pos + 1] {
            case .num, .name, .sym("("): result = false
            default: break
            }
        }
        return result
    }

    private func power() -> Cx? {
        var result = primary()
        while result != nil, percentAhead() {
            pos += 1
            result = result.map { v in Cx.div(v, Cx(100)) }
        }
        if result != nil, peek() == .sym("^") {
            pos += 1
            let rhs = factor()
            if let l = result, let r = rhs {
                result = Cx.pow(l, r)
            } else {
                result = nil
            }
        }
        return result
    }

    private func primary() -> Cx? {
        var result: Cx? = nil
        switch peek() {
        case .num(let v):
            pos += 1
            result = Cx(v)
        case .sym("("):
            pos += 1
            let inner = expr()
            if inner != nil, peek() == .sym(")") {
                pos += 1
                result = inner
            }
        case .name(let id):
            pos += 1
            if peek() == .sym("(") {
                pos += 1
                if let args = argList() {
                    result = Calculator.apply(id, args)
                }
            } else {
                result = Calculator.resolve(id, vars)
            }
        default:
            result = nil
        }
        return result
    }

    private func argList() -> [Cx]? {
        var result: [Cx]? = nil
        if peek() == .sym(")") {
            pos += 1
            result = []
        } else {
            var list: [Cx] = []
            var value = expr()
            while let v = value {
                list.append(v)
                if peek() == .sym(",") {
                    pos += 1
                    value = expr()
                } else {
                    value = nil
                }
            }
            if !list.isEmpty, peek() == .sym(")") {
                pos += 1
                result = list
            }
        }
        return result
    }
}
