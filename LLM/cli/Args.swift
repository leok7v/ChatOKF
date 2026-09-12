import Foundation

func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

func isModelFile(_ path: String) -> Bool {
    path.hasSuffix(".ggxf") || path.hasSuffix(".gguf")
}

// The leading '\r' overwrites the last report; the path is clipped from the
// LEFT, the end that repeats, in a fixed width so no tail lingers.
func progressLine(_ done: Int64, _ total: Int64, _ file: String) -> String {
    let gb = 1_000_000_000.0
    let width = 50
    let pct = total > 0 ? Double(done) * 100 / Double(total) : 0
    var name = file
    if name.count > width { name = "..." + String(name.suffix(width - 3)) }
    return String(format: "\r  %5.1f%%  %.2f / %.2f GB  %@", pct,
                  Double(done) / gb, Double(total) / gb,
                  name.padding(toLength: width, withPad: " ", startingAt: 0))
}
