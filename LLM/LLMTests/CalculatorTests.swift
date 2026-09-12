import XCTest
@testable import LLM

// The complex calculator behind the always-available tool: real behavior stays
final class CalculatorTests: XCTestCase {

    private func eval(_ s: String) async -> String {
        await Calculator().evaluate(s)
    }

    func testRealArithmeticUnchanged() async {
        let a = await eval("2 + 3 * 4")
        XCTAssertEqual(a, "14")
        let b = await eval("-2^2")
        XCTAssertEqual(b, "-4")
        let c = await eval("cos(1)")
        XCTAssertEqual(c, "0.540302305868")
        let d = await eval("sin(1)")
        XCTAssertEqual(d, "0.841470984808")
        let e = await eval("2^10")
        XCTAssertEqual(e, "1024")
    }

    func testCurrencySignsAreIgnored() async {
        let a = await eval("$1.10 - $1.00")
        XCTAssertEqual(a, "0.1")
        let b = await eval("$100 * 2")
        XCTAssertEqual(b, "200")
        let c = await eval("x = $5")
        XCTAssertEqual(c, "5")
    }

    func testRealErrorsUnchanged() async {
        let bad = await eval("1 +")
        XCTAssertTrue(bad.hasPrefix("error"))
        let div = await eval("mod(1, 0)")
        XCTAssertTrue(div.hasPrefix("error"))
        let empty = await eval("")
        XCTAssertTrue(empty.hasPrefix("error"))
    }

    func testEulerIdentity() async {
        let a = await eval("exp(i*pi)")
        XCTAssertEqual(a, "-1")
        let b = await eval("exp(i*pi) + 1")
        XCTAssertEqual(b, "0")
    }

    func testEulerOneRadian() async {
        let a = await eval("exp(i*1)")
        XCTAssertEqual(a, "0.540302305868 + 0.841470984808i")
        let b = await eval("cos(1) + i*sin(1)")
        XCTAssertEqual(b, a)
    }

    func testIToTheI() async {
        let a = await eval("i^i")
        XCTAssertEqual(a, "0.207879576351")
        let b = await eval("exp(-pi/2)")
        XCTAssertEqual(b, a)
    }

    func testPythonPowerOperator() async {
        let a = await eval("i**i")
        XCTAssertEqual(a, "0.207879576351")
        let b = await eval("2**10")
        XCTAssertEqual(b, "1024")
        let c = await eval("1000 * ((1 + 0.05) ** 22)")
        XCTAssertEqual(c, "2925.26071992")
    }

    // The exact 0.8B shapes that burned a whole round budget in-app:
    // textbook "= ?" suffixes and a percent literal.
    func testTextbookEqualsQuestionSuffix() async {
        let a = await eval("1000 * ((1 + 0.05)^(22*1)) = ?")
        XCTAssertEqual(a, "2925.26071992")
        let b = await eval("1000 * pow((1 + 0.05), 22) =?")
        XCTAssertEqual(b, "2925.26071992")
        let c = await eval("2 + 2?")
        XCTAssertEqual(c, "4")
        let d = await eval("x = 5")
        XCTAssertEqual(d, "5", "real assignments keep working")
        let e = await eval("350 * 0.4 = X1")
        XCTAssertEqual(e, "140", "reversed assignment names the result")
        let f = await eval("2 + 2 = 4")
        XCTAssertTrue(f.hasPrefix("error"),
            "an equation is neither assignment direction")
    }

    func testPostfixPercent() async {
        let a = await eval("5%")
        XCTAssertEqual(a, "0.05")
        let b = await eval("1000 * 5% + 7")
        XCTAssertEqual(b, "57")
        let c = await eval("1000 * (1 + 5%)^22 = ?")
        XCTAssertEqual(c, "2925.26071992")
        let mod = await eval("22 % 7")
        XCTAssertTrue(mod.hasPrefix("error"),
            "binary modulo intent must not silently misread as percent")
    }

    func testPythonNamespacesAndCase() async {
        let a = await eval("1000 * math.pow(1 + 0.05, 22)")
        XCTAssertEqual(a, "2925.26071992")
        let b = await eval("2 * Math.PI")
        XCTAssertEqual(b, "6.28318530718")
        let c = await eval("np.sqrt(16)")
        XCTAssertEqual(c, "4")
    }

    func testSemicolonSeparatesArguments() async {
        let a = await eval("1000 * POW(1.05; 22)")
        XCTAssertEqual(a, "2925.26071992")
        let b = await eval("max(3; 4)")
        XCTAssertEqual(b, "4")
    }

    func testRoundToDecimals() async {
        let a = await eval("round(2925.26071992, 2)")
        XCTAssertEqual(a, "2925.26")
        let b = await eval("round(2.5)")
        XCTAssertEqual(b, "3")
        let c = await eval("round(1234.5678, -2)")
        XCTAssertEqual(c, "1200")
    }

    func testVariableNamesFoldCase() async {
        let calc = Calculator()
        let stored = await calc.evaluate("Total = 5")
        XCTAssertEqual(stored, "5")
        let a = await calc.evaluate("total * 2")
        XCTAssertEqual(a, "10")
        let b = await calc.evaluate("TOTAL + 1")
        XCTAssertEqual(b, "6")
        let c = await calc.evaluate("Pi = 3")
        XCTAssertEqual(c, "3")
        let d = await calc.evaluate("2 * pi")
        XCTAssertEqual(d, "6", "a user variable shadows the constant")
    }

    func testSqrtOfNegativeAndI() async {
        let a = await eval("sqrt(-1)")
        XCTAssertEqual(a, "i")
        let b = await eval("sqrt(i)")
        XCTAssertEqual(b, "0.707106781187 + 0.707106781187i")
        let sq = await eval("(0.707106781187 + 0.707106781187i)^2")
        XCTAssertTrue(sq.hasSuffix("i"), "squares back to ~i: \(sq)")
    }

    func testLnOfNegative() async {
        let a = await eval("ln(-1)")
        XCTAssertEqual(a, "3.14159265359i")
        let b = await eval("ln(-e)")
        XCTAssertEqual(b, "1 + 3.14159265359i")
    }

    func testComplexHelpers() async {
        let mag = await eval("abs(3 + 4i)")
        XCTAssertEqual(mag, "5")
        let re = await eval("re(3 + 4i)")
        XCTAssertEqual(re, "3")
        let im = await eval("im(3 + 4i)")
        XCTAssertEqual(im, "4")
        let conj = await eval("conj(3 + 4i)")
        XCTAssertEqual(conj, "3 - 4i")
        let arg = await eval("arg(i)")
        XCTAssertEqual(arg, "1.57079632679")
    }

    func testImplicitMultiplication() async {
        let a = await eval("2i")
        XCTAssertEqual(a, "2i")
        let b = await eval("2pi")
        XCTAssertEqual(b, "6.28318530718")
        let c = await eval("2sin(1)")
        XCTAssertEqual(c, "1.68294196962")
        let d = await eval("0.5403 + 0.8415i")
        XCTAssertEqual(d, "0.5403 + 0.8415i")
    }

    func testRealOnlyFunctionsRejectComplex() async {
        let a = await eval("floor(i)")
        XCTAssertTrue(a.hasPrefix("error"))
        let b = await eval("atan2(i, 1)")
        XCTAssertTrue(b.hasPrefix("error"))
    }

    func testVariablesHoldComplexValues() async {
        let calc = Calculator()
        let stored = await calc.evaluate("z = exp(i*pi/4)")
        XCTAssertEqual(stored,
                       "0.707106781187 + 0.707106781187i")
        let mag = await calc.evaluate("abs(z)")
        XCTAssertEqual(mag, "1")
        let reserved = await calc.evaluate("i = 3")
        XCTAssertTrue(reserved.hasPrefix("error"))
    }

    func testPureImaginaryFormatting() async {
        let one = await eval("i")
        XCTAssertEqual(one, "i")
        let minus = await eval("-i")
        XCTAssertEqual(minus, "-i")
        let scaled = await eval("i * 3")
        XCTAssertEqual(scaled, "3i")
    }
}
