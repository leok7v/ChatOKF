import Foundation
import XCTest
@testable import MD

// One MathFontFile serves every formula and fills its caches lazily; an
// export lays out off the main thread while the screen draws its own.
final class MathConcurrencyTests: XCTestCase {

    private static let formulas = [
        "\\frac{a}{b}", "x^2 + y^2 = z^2", "\\sum_{i=1}^{n} i",
        "\\sqrt{\\alpha + \\beta}", "\\int_0^1 f(x)\\,dx",
        "\\frac{\\partial u}{\\partial t}", "a \\\\ b", "x &= 1 \\\\ y &= 2",
    ]

    // Lay out AND rasterize on every thread, because the two touch different
    // caches and only doing both reaches the one that raced.
    private static func work(_ tex: String, _ size: CGFloat) -> Bool {
        var ok = false
        if let layout = TeX.layout(tex, size: size) {
            ok = layout.width > 0
                && layout.cgImage(scale: 1, padding: 2, background: nil,
                                  color: CGColor(gray: 0, alpha: 1)) != nil
        }
        return ok
    }

    func testConcurrentLayoutAndDraw() {
        let formulas = Self.formulas
        let done = expectation(description: "every thread finished")
        done.expectedFulfillmentCount = formulas.count
        nonisolated(unsafe) let drawn = NSCountedSet()
        let tally = NSLock()
        // Sizes differ per thread on purpose: identical sizes would let the
        // first thread warm the CTFont cache and hide the race.
        for (i, tex) in formulas.enumerated() {
            DispatchQueue.global().async {
                var made = 0
                for round in 0..<8 {
                    let size = CGFloat(14 + i) + CGFloat(round) * 0.5
                    if Self.work(tex, size) { made += 1 }
                }
                tally.lock()
                drawn.add(made)
                tally.unlock()
                done.fulfill()
            }
        }
        wait(for: [done], timeout: 60)
        XCTAssertEqual(drawn.count(for: 8), formulas.count,
                       "a thread failed to lay out or rasterize")
    }

    // The same font instance is handed out every time, which is the whole
    // reason the lock has to exist rather than each caller parsing its own.
    func testEveryThreadSharesOneFont() throws {
        let first = try MathFontFile.shared()
        let second = try MathFontFile.shared()
        XCTAssertTrue(first === second)
    }
}
