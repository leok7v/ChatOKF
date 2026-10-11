import Foundation

struct PieceIndex {

    static let space = "\u{2581}"

    private let gguf: GGUF
    private let base: UnsafeRawPointer
    private let starts: [Int32]
    let scores: [Float]
    private let slots: [Int32]
    private let mask: Int
    let maxPieceBytes: Int

    var count: Int { starts.count }

    static func from(gguf g: GGUF) -> PieceIndex? {
        var result: PieceIndex? = nil
        if g.string("tokenizer.ggml.model") == "llama",
           let table = g.stringTable("tokenizer.ggml.tokens"),
           let scores = g.doubles("tokenizer.ggml.scores"),
           table.count == scores.count, table.count > 0 {
            result = PieceIndex(gguf: g, table: table, scores: scores)
        }
        return result
    }

    private init(gguf g: GGUF, table: GGUFStringTable, scores: [Double]) {
        var capacity = 1
        while capacity < table.count * 2 { capacity *= 2 }
        let wrap = capacity - 1
        var index = [Int32](repeating: -1, count: capacity)
        var offsets: [Int32] = []
        offsets.reserveCapacity(table.count)
        var longest = 0
        table.forEach { offset, bytes in
            let id = Int32(offsets.count)
            offsets.append(Int32(offset))
            longest = max(longest, bytes.count)
            var slot = Int(PieceIndex.fnv(bytes.baseAddress!, bytes.count)
                           & UInt64(wrap))
            while index[slot] >= 0 { slot = (slot + 1) & wrap }
            index[slot] = id
        }
        gguf = g
        base = table.base
        starts = offsets
        self.scores = scores.map { score in Float(score) }
        slots = index
        mask = wrap
        maxPieceBytes = longest
    }

    private static func fnv(_ p: UnsafeRawPointer, _ n: Int) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for i in 0..<n {
            h ^= UInt64(p.load(fromByteOffset: i, as: UInt8.self))
            h = h &* 0x0000_0100_0000_01b3
        }
        return h
    }

    func lookup(_ p: UnsafeRawPointer, _ n: Int)
        -> (id: Int32, score: Float)? {
        var slot = Int(PieceIndex.fnv(p, n) & UInt64(mask))
        var out: (id: Int32, score: Float)? = nil
        while out == nil && slots[slot] >= 0 {
            let id = slots[slot]
            let entry = base + Int(starts[Int(id)])
            let len = Int(entry.loadUnaligned(as: UInt64.self))
            if len == n && memcmp(entry + 8, p, n) == 0 {
                out = (id, scores[Int(id)])
            } else {
                slot = (slot + 1) & mask
            }
        }
        return out
    }

    func bytes(of id: Int32) -> [UInt8] {
        let entry = base + Int(starts[Int(id)])
        let len = Int(entry.loadUnaligned(as: UInt64.self))
        return Array(UnsafeRawBufferPointer(start: entry + 8, count: len))
    }

    func id(of piece: String) -> Int32? {
        var out: Int32? = nil
        let bytes = Array(piece.utf8)
        bytes.withUnsafeBytes { raw in
            if let p = raw.baseAddress { out = lookup(p, raw.count)?.id }
        }
        return out
    }
}
