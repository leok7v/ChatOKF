// Port of the reference slugs.c query path. The SLGX trailer follows the
// GGUF payload, which GGUF readers ignore, so the file stays a valid GGUF.
import Foundation

public struct SlugHit: Sendable {
    public let id: String
    public let title: String
    public let distance: Int

    // Past this Hamming distance the corpus has no real neighbour, so the
    // caller answers from its own knowledge (slugs.c's [low] flag).
    public static let lowConfidenceDistance = 82
    public var isConfident: Bool { distance <= SlugHit.lowConfidenceDistance }
}

// Holds raw pointers into the GGUF mmap, so it pins the GGUF.
final class SignIndex: VectorIndex {
    static let embedDim = 384
    static let pcaDim = 256
    static let keyBytes = 32
    static let lowConfidence = 82
    private static let slgxMagic: UInt32 = 0x5847_4C53   // "SLGX"

    private let gguf: GGUF
    private let mean: [Float]
    private let components: [Float]
    private let keys: UnsafeRawPointer
    let count: Int
    // Slug table (in the mmap): offsets[count+1] then packed "pageid\ttitle".
    private let offsets: UnsafePointer<UInt32>
    private let strings: UnsafeRawPointer

    init?(gguf g: GGUF) {
        let base = g.map
        let size = g.mapSize
        var found: (pca: UnsafeRawPointer, keys: UnsafeRawPointer,
                    keysLen: Int, tbl: UnsafeRawPointer, tblLen: Int)? = nil
        let tailLen = 16    // u64 base + u32 count + u32 magic
        if size >= tailLen {
            let tail = base + size - tailLen
            let magic = tail.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
            let n = Int(tail.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let entLen = 32                    // name[16] + u64 off + u64 size
            let ents = n * entLen + tailLen
            if magic == SignIndex.slgxMagic && ents <= size {
                var pca: UnsafeRawPointer? = nil
                var keysP: UnsafeRawPointer? = nil, keysN = 0
                var tblP: UnsafeRawPointer? = nil, tblN = 0
                let ep = base + size - ents
                for i in 0..<n {
                    let e = ep + i * entLen
                    let name = SignIndex.entryName(e)
                    let off = Int(e.loadUnaligned(fromByteOffset: 16,
                                                  as: UInt64.self))
                    let blen = Int(e.loadUnaligned(fromByteOffset: 24,
                                                   as: UInt64.self))
                    if off + blen <= size {
                        let p = base + off
                        if name == "pca" { pca = p }
                        if name == "keys" { keysP = p; keysN = blen }
                        if name == "slugs" { tblP = p; tblN = blen }
                    }
                }
                if let pca, let keysP, let tblP {
                    found = (pca, keysP, keysN, tblP, tblN)
                }
            }
        }
        if let f = found {
            gguf = g
            let mdim = SignIndex.embedDim
            let cdim = SignIndex.pcaDim * SignIndex.embedDim
            mean = SignIndex.floats(f.pca, 0, mdim)
            components = SignIndex.floats(f.pca, mdim * 4, cdim)
            keys = f.keys
            count = f.keysLen / SignIndex.keyBytes
            // slug table: [u64 count][u32 offsets[count+1]][u8 strings]
            let tblCount = Int(f.tbl.loadUnaligned(as: UInt64.self))
            offsets = (f.tbl + 8).assumingMemoryBound(to: UInt32.self)
            strings = f.tbl + 8 + (tblCount + 1) * 4
        } else {
            return nil
        }
    }

    private static func entryName(_ e: UnsafeRawPointer) -> String {
        var bytes: [UInt8] = []
        var i = 0
        while i < 16 {
            let c = e.load(fromByteOffset: i, as: UInt8.self)
            if c != 0 { bytes.append(c) }
            i += 1
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func floats(_ p: UnsafeRawPointer, _ byteOff: Int,
                               _ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBytes {
            _ = memcpy($0.baseAddress!, p + byteOff, n * 4)
        }
        return out
    }

    // MSB-first within a byte, matching slugs.c.
    private func signKey(_ emb: [Float]) -> [UInt8] {
        var centered = [Float](repeating: 0, count: SignIndex.embedDim)
        for j in 0..<SignIndex.embedDim { centered[j] = emb[j] - mean[j] }
        var key = [UInt8](repeating: 0, count: SignIndex.keyBytes)
        components.withUnsafeBufferPointer { cp in
            centered.withUnsafeBufferPointer { xp in
                for b in 0..<SignIndex.pcaDim {
                    let row = cp.baseAddress! + b * SignIndex.embedDim
                    var acc: Float = 0
                    for j in 0..<SignIndex.embedDim { acc += row[j] * xp[j] }
                    if acc > 0 { key[b / 8] |= UInt8(1 << (7 - (b % 8))) }
                }
            }
        }
        return key
    }

    private static func hamming(_ a: UnsafeRawPointer,
                                _ b: UnsafeRawPointer) -> Int {
        var d = 0
        for w in 0..<4 {
            let x = a.loadUnaligned(fromByteOffset: w * 8, as: UInt64.self)
            let y = b.loadUnaligned(fromByteOffset: w * 8, as: UInt64.self)
            d += (x ^ y).nonzeroBitCount
        }
        return d
    }

    func search(_ emb: [Float], topK: Int) -> [(index: Int, distance: Int)] {
        let key = signKey(emb)
        let k = max(1, topK)
        var top = [(index: Int, distance: Int)](
            repeating: (0, SignIndex.pcaDim + 1), count: k)
        key.withUnsafeBytes { qb in
            let q = qb.baseAddress!
            for i in 0..<count {
                let d = SignIndex.hamming(q, keys + i * SignIndex.keyBytes)
                if d < top[k - 1].distance {
                    var p = k - 1
                    while p > 0 && top[p - 1].distance > d {
                        top[p] = top[p - 1]
                        p -= 1
                    }
                    top[p] = (i, d)
                }
            }
        }
        return top
    }

    func record(_ i: Int) -> (id: String, title: String) {
        let lo = Int(offsets[i])
        let hi = Int(offsets[i + 1])
        var bytes = [UInt8](repeating: 0, count: hi - lo)
        bytes.withUnsafeMutableBytes {
            _ = memcpy($0.baseAddress!, strings + lo, hi - lo)
        }
        var tab = 0
        while tab < bytes.count && bytes[tab] != 0x09 { tab += 1 }
        let id = String(decoding: bytes[0..<tab], as: UTF8.self)
        let titleStart = tab < bytes.count ? tab + 1 : bytes.count
        let title = String(decoding: bytes[titleStart...], as: UTF8.self)
        return (id, title)
    }
}
