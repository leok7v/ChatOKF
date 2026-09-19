import Metal
import XCTest
@testable import LLM

final class MetalKVPoolTests: XCTestCase {

    private func device() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        return device
    }

    private func pool(window: Int?, kvDim: Int = 8) throws -> MetalKVPool {
        let p = MetalKVPool(device: try device(), P: 4, kvDim: kvDim,
                            window: window, capacity: 16)
        try p.attachTemp()
        return p
    }

    private func row(_ b: MTLBuffer, _ slot: Int, _ kvDim: Int)
        -> UnsafeMutablePointer<Float16> {
        b.contents().assumingMemoryBound(to: Float16.self) + slot * kvDim
    }

    private func allocated(_ url: URL) -> Int {
        var s = stat()
        return stat(url.path, &s) == 0 ? Int(s.st_blocks) * 512 : 0
    }

    func testEvictionKeepsTheWindowAndTheFloorsWindow() throws {
        let fresh = try pool(window: 4)
        fresh.appendBatch(20)
        fresh.evict(floor: 0)
        XCTAssertEqual(fresh.livePages, 1, "rows 16 to 19 are the window")
        XCTAssertEqual(fresh.pages(rows: 16, 19).count, 2)
        XCTAssertEqual(fresh.pages(rows: 0, 3).count, 0)
        XCTAssertEqual(fresh.pageBytesTotal, 2 * 4 * 8 * 2)
        let p = try pool(window: 4)
        p.appendBatch(24)
        p.evict(floor: 10)
        XCTAssertEqual(p.pages(rows: 7, 9).count, 4,
                       "rows 7 to 9 are what a rewind to 10 attends over")
        XCTAssertEqual(p.pages(rows: 12, 15).count, 0)
        XCTAssertEqual(p.livePages, 3)
        p.truncate(to: 10)
        XCTAssertEqual(p.len, 10)
        _ = p.tailForAppend()
        p.commitAppend()
        XCTAssertEqual(p.len, 11)
        XCTAssertEqual(p.firstLive(below: 11), 4)
        let full = try pool(window: nil)
        full.appendBatch(20)
        full.evict(floor: 0)
        XCTAssertEqual(full.livePages, 5, "a full pool never evicts")
        XCTAssertEqual(full.residentPages.count, 10)
    }

    func testAttachFromARowLeavesZeroPagesBelow() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kv_\(UUID().uuidString)")
        let p = MetalKVPool(device: try device(), P: 4, kvDim: 8,
                            window: 4, capacity: 16)
        try p.attach(url, first: 8, len: 14)
        XCTAssertEqual(p.len, 14)
        XCTAssertEqual(p.pageCount, 4)
        XCTAssertEqual(p.livePages, 2)
        XCTAssertEqual(p.firstLive(below: 14), 8)
        XCTAssertEqual(p.pages(rows: 0, 7).count, 0)
        p.detach()
        try? FileManager.default.removeItem(at: url)
    }

    func testARoundTripThroughTheFileFindsTheRows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kv_\(UUID().uuidString)")
        let p = MetalKVPool(device: try device(), P: 4, kvDim: 8,
                            window: nil, capacity: 16)
        try p.attach(url)
        p.appendBatch(6)
        for r in 0..<6 {
            row(p.kPages[r / 4], r % 4, 8)[3] = Float16(r) + 0.5
            row(p.vPages[r / 4], r % 4, 8)[5] = Float16(r) - 0.25
        }
        p.detach()
        let q = MetalKVPool(device: try device(), P: 4, kvDim: 8,
                            window: nil, capacity: 16)
        try q.attach(url, len: 6)
        XCTAssertEqual(q.len, 6)
        XCTAssertEqual(q.pageCount, 2)
        XCTAssertEqual(q.livePages, 2)
        XCTAssertEqual(Float(row(q.kPages[1], 1, 8)[3]), 5.5)
        XCTAssertEqual(Float(row(q.vPages[0], 2, 8)[5]), 1.75)
        q.detach()
        try? FileManager.default.removeItem(at: url)
    }

    func testAnEvictedPageIsPunchedOutOfTheFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kv_\(UUID().uuidString)")
        let p = MetalKVPool(device: try device(), P: 4, kvDim: 2048,
                            window: 4, capacity: 8)
        try p.attach(url)
        p.appendBatch(20)
        for i in 0..<5 {
            row(p.kPages[i], 0, 2048)[0] = 1
            row(p.vPages[i], 0, 2048)[0] = 1
        }
        p.sync()
        let before = allocated(url)
        p.evict(floor: 0)
        XCTAssertEqual(p.livePages, 1)
        let after = allocated(url)
        XCTAssertLessThan(after, before, "eviction returned no blocks")
        XCTAssertNotEqual(Float(row(p.kPages[4], 0, 2048)[0]), 0,
                          "the window page was lost")
        p.detach()
        try? FileManager.default.removeItem(at: url)
    }

    func testAGPUWriteReachesTheFileAfterAWriteBack() throws {
        let device = try device()
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no command queue")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kv_\(UUID().uuidString)")
        let p = MetalKVPool(device: device, P: 4, kvDim: 2048,
                            window: nil, capacity: 8)
        try p.attach(url)
        p.appendBatch(8)
        let cb = queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.fill(buffer: p.kPages[1], range: 0 ..< p.pageBytes, value: 0x5A)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        p.touch()
        p.writeBack()
        let h = try FileHandle(forReadingFrom: url)
        try h.seek(toOffset: UInt64(p.pageBytes + 1000))
        let byte = try h.read(upToCount: 1)
        try h.close()
        XCTAssertEqual(byte, Data([0x5A]), "the GPU write did not reach the file")
        p.detach()
        try? FileManager.default.removeItem(at: url)
    }

    func testASnapshotRestoresARewrittenTail() throws {
        let p = try pool(window: nil)
        p.appendBatch(6)
        row(p.kPages[1], 1, 8)[0] = 1
        row(p.vPages[1], 1, 8)[0] = 1
        let s = p.snapshot()
        p.truncate(to: 4)
        p.appendBatch(2)
        row(p.kPages[1], 1, 8)[0] = 2
        row(p.vPages[1], 1, 8)[0] = 2
        p.restore(s)
        XCTAssertEqual(p.len, 6)
        XCTAssertEqual(Float(row(p.kPages[1], 1, 8)[0]), 1)
        XCTAssertEqual(Float(row(p.vPages[1], 1, 8)[0]), 1)
        p.truncate(to: 0)
        p.restore(s)
        XCTAssertEqual(p.pageCount, 2, "a restore past a reset remaps")
        XCTAssertEqual(Float(row(p.kPages[1], 1, 8)[0]), 1)
    }
}
