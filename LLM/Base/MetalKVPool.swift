import Foundation
import Metal

final class MetalKVPool {
    private let device: MTLDevice
    let P: Int
    let kvDim: Int
    private(set) var kPages: [MTLBuffer] = []
    private(set) var vPages: [MTLBuffer] = []
    private(set) var len = 0
    static let maxPages = 2048   // must match KV_MAXP in the shader
    private(set) var kAddr: MTLBuffer
    private(set) var vAddr: MTLBuffer

    init(device: MTLDevice, P: Int, kvDim: Int) {
        self.device = device
        self.P = P
        self.kvDim = kvDim
        let bytes = MetalKVPool.maxPages * 8
        kAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
        vAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
    }

    private var pageBytes: Int { P * kvDim * MemoryLayout<Float16>.stride }

    private func newPage() -> MTLBuffer {
        device.makeBuffer(length: pageBytes, options: .storageModeShared)!
    }

    func tailForAppend() -> (k: MTLBuffer, v: MTLBuffer, slot: Int) {
        if len % P == 0 { kPages.append(newPage()); vPages.append(newPage()) }
        return (kPages[kPages.count - 1], vPages[vPages.count - 1], len % P)
    }
    func commitAppend() { len += 1 }

    func appendBatch(_ n: Int) {
        for _ in 0..<n { _ = tailForAppend(); commitAppend() }
        refreshTable()
    }

    func refreshTable() {
        let n = kPages.count
        precondition(n <= MetalKVPool.maxPages,
                     "KV context exceeds P * \(MetalKVPool.maxPages) positions")
        let ka = kAddr.contents().assumingMemoryBound(to: UInt64.self)
        let va = vAddr.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<n { ka[i] = kPages[i].gpuAddress; va[i] = vPages[i].gpuAddress }
    }

    var residentPages: [MTLResource] { kPages + vPages }

    var pageCount: Int { kPages.count }
    var addressableLength: Int { kPages.count * P }
    var pageBytesTotal: Int { (kPages.count + vPages.count) * pageBytes }

    func truncate(to newLength: Int) {
        precondition(newLength <= kPages.count * P,
                     "truncate to \(newLength) over \(kPages.count) pages: "
                     + "a bookmark cannot be restored across a reset")
        len = newLength
        let need = (newLength + P - 1) / P
        while kPages.count > need { kPages.removeLast(); vPages.removeLast() }
    }

    // A bookmark shares completed pages (immutable) and DUPLICATES the partial
    // tail, so the snapshot and the pool never write the same page.
    struct Snapshot: @unchecked Sendable {
        let kPages: [MTLBuffer]
        let vPages: [MTLBuffer]
        let len: Int
    }
    func snapshot() -> Snapshot {
        var kp = kPages, vp = vPages
        if len % P != 0, !kp.isEmpty {
            kp[kp.count - 1] = dup(kp[kp.count - 1])
            vp[vp.count - 1] = dup(vp[vp.count - 1])
        }
        return Snapshot(kPages: kp, vPages: vp, len: len)
    }
    func restore(_ s: Snapshot) {
        kPages = s.kPages
        vPages = s.vPages
        len = s.len
        // Give the pool its own tail so later appends do not mutate the
        // snapshot's tail (a snapshot may be restored more than once).
        if len % P != 0, !kPages.isEmpty {
            kPages[kPages.count - 1] = dup(kPages[kPages.count - 1])
            vPages[vPages.count - 1] = dup(vPages[vPages.count - 1])
        }
    }

    func flatten(_ pages: [MTLBuffer], len: Int) -> [Float] {
        var out = [Float](repeating: 0, count: len * kvDim)
        out.withUnsafeMutableBufferPointer { ob in
            for i in 0..<len {
                let src = pages[i / P].contents()
                    .assumingMemoryBound(to: Float16.self) + (i % P) * kvDim
                for c in 0..<kvDim { ob[i * kvDim + c] = Float(src[c]) }
            }
        }
        return out
    }

    func fill(k: StateBytes.FloatSpan, v: StateBytes.FloatSpan,
              count: Int) {
        for i in 0..<count {
            let tail = tailForAppend()
            let kd = tail.k.contents()
                .assumingMemoryBound(to: Float16.self) + tail.slot * kvDim
            let vd = tail.v.contents()
                .assumingMemoryBound(to: Float16.self) + tail.slot * kvDim
            for c in 0..<kvDim {
                kd[c] = Float16(k.f(i * kvDim + c))
                vd[c] = Float16(v.f(i * kvDim + c))
            }
            commitAppend()
        }
        refreshTable()
    }

    private func dup(_ b: MTLBuffer) -> MTLBuffer {
        let c = device.makeBuffer(length: b.length, options: .storageModeShared)!
        memcpy(c.contents(), b.contents(), b.length)
        return c
    }
}
