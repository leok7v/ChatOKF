import Compression
import Foundation
import Testing
@testable import XET

func literals(_ plain: [UInt8]) -> [UInt8] {
    var out: [UInt8] = [UInt8(min(plain.count, 15) << 4)]
    var rest = plain.count - 15
    while rest >= 255 {
        out.append(255)
        rest -= 255
    }
    if plain.count >= 15 { out.append(UInt8(rest)) }
    return out + plain
}

func lz4(_ plain: [UInt8]) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: plain.count + plain.count / 200 + 64)
    let n = plain.withUnsafeBufferPointer { src in
        out.withUnsafeMutableBufferPointer { dst in
            compression_encode_buffer(dst.baseAddress!, dst.count,
                                      src.baseAddress!, src.count, nil,
                                      COMPRESSION_LZ4_RAW)
        }
    }
    return n > 0 ? Array(out[0..<n]) : literals(plain)
}

func frame(_ plain: [UInt8], block: Int = 1 << 16,
           stored: Bool = false) -> [UInt8] {
    var out: [UInt8] = [0x04, 0x22, 0x4D, 0x18, 0x60, 0x40, 0x00]
    var at = 0
    while at < plain.count {
        let piece = Array(plain[at..<min(at + block, plain.count)])
        let body = stored ? piece : lz4(piece)
        let size = UInt32(body.count) | (stored ? 0x8000_0000 : 0)
        out += [UInt8(size & 0xFF), UInt8(size >> 8 & 0xFF),
                UInt8(size >> 16 & 0xFF), UInt8(size >> 24 & 0xFF)]
        out += body
        at += piece.count
    }
    return out + [0, 0, 0, 0]
}

func pack(_ plain: [UInt8], framed: Bool) -> [UInt8] {
    framed ? frame(plain) : lz4(plain)
}

func chunk(_ scheme: UInt8, _ plain: [UInt8], framed: Bool = false)
    -> [UInt8] {
    let payload: [UInt8]
    switch scheme {
    case 0: payload = plain
    case 1: payload = pack(plain, framed: framed)
    default: payload = pack(Chunk.group(plain), framed: framed)
    }
    let h = ChunkHeader(compressed: payload.count, scheme: scheme,
                        uncompressed: plain.count)
    return h.encoded + payload
}

func noise(_ n: Int, seed: UInt64 = 7) -> [UInt8] {
    var x = seed
    return (0..<n).map { _ in
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        return UInt8(truncatingIfNeeded: x >> 24)
    }
}

func decoded(_ bytes: [UInt8], piece: Int) throws -> [[UInt8]] {
    let stream = ChunkStream()
    var out: [[UInt8]] = []
    var at = 0
    while at < bytes.count {
        let end = min(at + piece, bytes.count)
        try stream.push(Data(bytes[at..<end])) { ordinal, chunk in
            #expect(ordinal == out.count)
            out.append(Array(chunk))
        }
        at = end
    }
    #expect(stream.pending == 0)
    return out
}

@Suite struct ChunkTests {

    @Test func headerRoundTrips() throws {
        let h = ChunkHeader(compressed: 28951, scheme: 1, uncompressed: 131072)
        #expect(h.encoded == [0x00, 0x17, 0x71, 0x00, 0x01, 0x00, 0x00, 0x02])
        let parsed = try h.encoded.withUnsafeBytes { raw in
            try ChunkHeader.parse(raw)
        }
        #expect(parsed == h)
    }

    @Test func headerRefusesWhatTheSpecForbids() {
        let bad: [[UInt8]] = [
            [1, 5, 0, 0, 0, 5, 0, 0],
            [0, 5, 0, 0, 3, 5, 0, 0],
            [0, 5, 0, 0, 1, 1, 0, 2],
            [0, 5, 0, 0, 0, 6, 0, 0],
        ]
        for b in bad {
            #expect(throws: XETError.self) {
                _ = try b.withUnsafeBytes { raw in try ChunkHeader.parse(raw) }
            }
        }
    }

    @Test func regroupMatchesTheSpecVectors() {
        let seven: [UInt8] = [0, 4, 1, 5, 2, 6, 3]
        var out = [UInt8](repeating: 0, count: 7)
        out.withUnsafeMutableBytes { o in Chunk.regroup(seven, 7, into: o) }
        #expect(out == [0, 1, 2, 3, 4, 5, 6])
        let nine: [UInt8] = [0, 4, 8, 1, 5, 2, 6, 3, 7]
        var nineOut = [UInt8](repeating: 0, count: 9)
        nineOut.withUnsafeMutableBytes { o in Chunk.regroup(nine, 9, into: o) }
        #expect(nineOut == [0, 1, 2, 3, 4, 5, 6, 7, 8])
    }

    @Test func groupAndRegroupAreInverse() {
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 255, 256, 257, 1000, 131072] {
            let plain = noise(n, seed: UInt64(n) + 1)
            let grouped = Chunk.group(plain)
            var back = [UInt8](repeating: 0, count: n)
            back.withUnsafeMutableBytes { o in
                Chunk.regroup(grouped, n, into: o)
            }
            #expect(back == plain, "length \(n)")
        }
    }

    @Test func everySchemeDecodesToThePlainBytes() throws {
        var text = [UInt8]("the quick brown fox jumps over the lazy dog ".utf8)
        while text.count < 5000 { text += text }
        for plain in [noise(131072), text, [UInt8](repeating: 0x41, count: 40),
                      noise(1), noise(3)] {
            for scheme: UInt8 in [0, 1, 2] {
                for framed in [false, true] {
                    let got = try decoded(chunk(scheme, plain, framed: framed),
                                          piece: 1 << 20)
                    #expect(got == [plain],
                            "scheme \(scheme) \(plain.count) framed \(framed)")
                }
            }
        }
    }

    @Test func theFramesTheLZ4ToolWritesDecode() throws {
        let hello: [UInt8] = [
            0x04, 0x22, 0x4d, 0x18, 0x64, 0x40, 0xa7, 0x05, 0x00, 0x00, 0x80,
            0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x00, 0x00, 0x00, 0x00, 0xf9, 0x77,
            0x00, 0xfb]
        let forty: [UInt8] = [
            0x04, 0x22, 0x4d, 0x18, 0x64, 0x40, 0xa7, 0x0b, 0x00, 0x00, 0x00,
            0x1f, 0x41, 0x01, 0x00, 0x0f, 0x50, 0x41, 0x41, 0x41, 0x41, 0x41,
            0x00, 0x00, 0x00, 0x00, 0x83, 0xc5, 0x41, 0x68]
        let alphabet: [UInt8] = [
            0x04, 0x22, 0x4d, 0x18, 0x64, 0x40, 0xa7, 0x25, 0x00, 0x00, 0x00,
            0xff, 0x0b, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
            0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74,
            0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x1a, 0x00, 0x02, 0x50, 0x76,
            0x77, 0x78, 0x79, 0x7a, 0x00, 0x00, 0x00, 0x00, 0xc3, 0x96, 0xf2,
            0xcd]
        let cases: [([UInt8], [UInt8])] = [
            (hello, [UInt8]("hello".utf8)),
            (forty, [UInt8](repeating: 0x41, count: 40)),
            (alphabet, [UInt8]("abcdefghijklmnopqrstuvwxyz".utf8)
                + [UInt8]("abcdefghijklmnopqrstuvwxyz".utf8)),
        ]
        for (payload, plain) in cases {
            let h = ChunkHeader(compressed: payload.count, scheme: 1,
                                uncompressed: plain.count)
            #expect(try decoded(h.encoded + payload, piece: 1 << 20)
                    == [plain])
        }
    }

    @Test func storedBlocksAndDependentBlocksDecode() throws {
        let plain = noise(100_000, seed: 21)
        #expect(try decoded(chunk(1, plain, framed: true), piece: 1 << 20)
                == [plain])
        let storedChunk = ChunkHeader(
            compressed: frame(plain, stored: true).count, scheme: 1,
            uncompressed: plain.count).encoded + frame(plain, stored: true)
        #expect(try decoded(storedChunk, piece: 4096) == [plain])
        let dependent: [UInt8] = [0x04, 0x22, 0x4D, 0x18, 0x40, 0x40, 0x00,
                                  5, 0, 0, 0, 0x40, 0x61, 0x62, 0x63, 0x64,
                                  3, 0, 0, 0, 0x00, 0x04, 0x00,
                                  0, 0, 0, 0]
        let h = ChunkHeader(compressed: dependent.count, scheme: 1,
                            uncompressed: 8)
        #expect(try decoded(h.encoded + dependent, piece: 1 << 20)
                == [[UInt8]("abcdabcd".utf8)])
    }

    @Test func theStreamIsTheSameHoweverTheBytesArrive() throws {
        let plains = [noise(131072), noise(100, seed: 3),
                      noise(1 << 16, seed: 9)]
        var xorb: [UInt8] = []
        for (i, p) in plains.enumerated() {
            xorb += chunk(UInt8(i % 3), p)
        }
        for piece in [1, 7, 4096, 1 << 20] {
            let got = try decoded(xorb, piece: piece)
            #expect(got == plains, "piece \(piece)")
        }
    }

    @Test func aTruncatedStreamKeepsItsTail() throws {
        let xorb = chunk(1, noise(1000))
        let stream = ChunkStream()
        try stream.push(Data(xorb.dropLast(10))) { _, _ in }
        #expect(stream.ordinal == 0)
        #expect(stream.pending == xorb.count - 10)
    }

    @Test func aPayloadThatDoesNotFillItsHeaderIsRefused() {
        let short = ChunkHeader(compressed: 3, scheme: 1, uncompressed: 5)
            .encoded + [0x50, 0x68, 0x65]
        let long = ChunkHeader(compressed: 6, scheme: 1, uncompressed: 4)
            .encoded + [0x50, 0x68, 0x65, 0x6C, 0x6C, 0x6F]
        for xorb in [short, long] {
            let stream = ChunkStream()
            #expect(throws: XETError.self) {
                try stream.push(Data(xorb)) { _, _ in }
            }
        }
    }

    @Test func theLiteralEncoderAgreesWithTheFramework() throws {
        let plain = noise(5000, seed: 5)
        #expect(try decoded(ChunkHeader(
            compressed: literals(plain).count, scheme: 1,
            uncompressed: plain.count).encoded + literals(plain),
            piece: 1 << 20) == [plain])
        #expect(try decoded(chunk(1, [UInt8](repeating: 0x41, count: 40)),
                            piece: 1) == [[UInt8](repeating: 0x41, count: 40)])
    }
}
