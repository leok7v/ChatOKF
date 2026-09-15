import Metal
import XCTest


final class FileBackedKVTests: XCTestCase {

    private static let pageBytes = 16384

    private func allocated(_ path: String) -> Int {
        var st = stat()
        return stat(path, &st) == 0 ? Int(st.st_blocks) * 512 : -1
    }

    private func fileByte(_ path: String, at offset: Int) throws -> UInt8 {
        let h = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? h.close() }
        try h.seek(toOffset: UInt64(offset))
        return try h.read(upToCount: 1)?.first ?? 0
    }

    func testAGPUWritesThroughASharedFileMappingAndTheFileIsSparse()
        throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("no Metal device")
        }
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("kv_\(UUID().uuidString).pool").path
        let reserve = 64 << 20
        let fd = open(path, O_RDWR | O_CREAT, 0o600)
        XCTAssertGreaterThanOrEqual(fd, 0)
        XCTAssertEqual(ftruncate(fd, off_t(reserve)), 0)
        XCTAssertLessThan(allocated(path), 1 << 20,
                          "a reserved file must start sparse")
        let raw = mmap(nil, reserve, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FILE, fd, 0)
        XCTAssertNotEqual(raw, MAP_FAILED)
        let base = raw!
        let buffer = device.makeBuffer(
            bytesNoCopy: base, length: reserve,
            options: .storageModeShared, deallocator: nil)
        XCTAssertNotNil(buffer,
                        "Metal refused a no-copy buffer over a file mapping")
        if let buffer {
            let written = 16 << 20
            let cb = queue.makeCommandBuffer()!
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: buffer, range: 0 ..< written, value: 0xA5)
            blit.fill(buffer: buffer,
                      range: (written - FileBackedKVTests.pageBytes)
                          ..< written, value: 0x3C)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            XCTAssertEqual(bytes[0], 0xA5, "the CPU does not see the GPU write")
            XCTAssertEqual(bytes[written - 1], 0x3C)
            XCTAssertEqual(bytes[written], 0, "untouched pages read as zero")
            XCTAssertEqual(try fileByte(path, at: 0), 0xA5,
                           "the file does not carry the GPU write")
            XCTAssertEqual(try fileByte(path, at: written - 1), 0x3C)
            XCTAssertEqual(msync(base, reserve, MS_SYNC), 0)
            XCTAssertEqual(fcntl(fd, F_FULLFSYNC), 0)
            XCTAssertEqual(try fileByte(path, at: written / 2), 0xA5)
            let onDisk = allocated(path)
            print("file-backed kv: \(onDisk) bytes allocated of \(reserve) "
                  + "reserved, \(written) written")
            XCTAssertLessThan(onDisk, written + (4 << 20),
                              "the untouched reserve got allocated: \(onDisk)")
            let again = device.makeBuffer(
                bytesNoCopy: base, length: reserve,
                options: .storageModeShared, deallocator: nil)
            XCTAssertNotNil(again, "a second buffer over the same mapping")
        }
        XCTAssertEqual(munmap(base, reserve), 0)
        close(fd)
        let remapped = open(path, O_RDONLY)
        let read = mmap(nil, reserve, PROT_READ, MAP_SHARED | MAP_FILE,
                        remapped, 0)
        XCTAssertNotEqual(read, MAP_FAILED)
        if let read, read != MAP_FAILED {
            let bytes = read.assumingMemoryBound(to: UInt8.self)
            XCTAssertEqual(bytes[1 << 20], 0xA5,
                           "a fresh mapping does not see what was written")
            let reader = device.makeBuffer(
                bytesNoCopy: read, length: reserve,
                options: .storageModeShared, deallocator: nil)
            XCTAssertNotNil(reader, "no Metal buffer over a read-only mapping")
            munmap(read, reserve)
        }
        close(remapped)
        try? FileManager.default.removeItem(atPath: path)
    }

}
