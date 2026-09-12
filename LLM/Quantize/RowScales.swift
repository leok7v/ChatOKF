import Foundation

public struct RowScales: Sendable {
    public private(set) var byLeaf: [String: [Float]] = [:]

    public var isEmpty: Bool { byLeaf.isEmpty }

    public var count: Int { byLeaf.count }

    public init() {}

    public init(path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        var at = 0
        let magic = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
        }
        at += 4
        if magic != 0x3143_5352 {
            throw GGUFErr.parse("\(path) is not a row-scale file")
        } else {
            let count = Int(RowScales.u32(data, &at))
            for _ in 0..<count {
                let bytes = Int(RowScales.u32(data, &at))
                let name = String(decoding: data.subdata(in: at..<(at + bytes)),
                                  as: UTF8.self)
                at += bytes
                let n = Int(RowScales.u32(data, &at))
                var row = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    row[i] = Float(bitPattern: RowScales.u32(data, &at))
                }
                byLeaf[name] = row
            }
        }
    }

    public func rows(_ gguf: String) -> [Float]? {
        byLeaf[gguf] ?? byLeaf[gguf.replacingOccurrences(of: ".weight",
                                                        with: "")]
    }

    public func scale(_ values: inout [Float], _ gguf: String,
                      _ width: Int) -> Bool {
        var did = false
        if let row = rows(gguf), width > 0, values.count % width == 0,
           values.count / width == row.count {
            for r in 0..<row.count where row[r] != 1 {
                let c = row[r]
                for i in (r * width)..<((r + 1) * width) { values[i] *= c }
            }
            did = true
        }
        return did
    }

    static func u32(_ d: Data, _ at: inout Int) -> UInt32 {
        let v = d.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: at, as: UInt32.self)
        }
        at += 4
        return v
    }
}
