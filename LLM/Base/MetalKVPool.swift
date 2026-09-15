import Foundation
import Metal

enum PoolError: Error {
    case open(String, Int32)
    case map(String, Int32)
}

final class MetalKVPool {
    private let device: MTLDevice
    let P: Int
    let kvDim: Int
    let window: Int?
    let capacity: Int
    private(set) var kPages: [MTLBuffer] = []
    private(set) var vPages: [MTLBuffer] = []
    private(set) var len = 0
    static let maxPages = 2048   // must match KV_MAXP in the shader
    private(set) var kAddr: MTLBuffer
    private(set) var vAddr: MTLBuffer
    private var tombstone: MTLBuffer?
    private var fd: Int32 = -1
    private var base: UnsafeMutableRawPointer?
    private var tempURL: URL?
    private var touched = 0
    private var scratch: UnsafeMutableRawPointer?
    var starved: (() -> Void)?

    init(device: MTLDevice, P: Int, kvDim: Int, window: Int? = nil,
         capacity: Int = MetalKVPool.maxPages) {
        self.device = device
        self.P = P
        self.kvDim = kvDim
        self.window = window
        self.capacity = min(max(capacity, 1), MetalKVPool.maxPages)
        let bytes = MetalKVPool.maxPages * 8
        kAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
        vAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
    }

    deinit {
        detach()
        if let scratch { free(scratch) }
        if let tempURL { try? FileManager.default.removeItem(at: tempURL) }
    }

    var pageBytes: Int { P * kvDim * MemoryLayout<Float16>.stride }
    var fileBytes: Int { 2 * capacity * pageBytes }
    var attached: Bool { base != nil }

    static func pagesFor(_ g: GGUF, P: Int) -> Int {
        let rows = ModelShape.trainedContext(g)
        return rows > 0 ? min(maxPages, (rows + P - 1) / P) : maxPages
    }

    func sync() {
        if let base { msync(base, fileBytes, MS_SYNC) }
    }

    func export(_ url: URL) {
        if let base {
            let out = open(url.path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
            if out >= 0 {
                ftruncate(out, off_t(fileBytes))
                if scratch == nil { scratch = malloc(pageBytes) }
                if let scratch {
                    for i in kPages.indices where isLive(i) {
                        for v in [false, true] {
                            let at = offset(i, v: v)
                            memcpy(scratch, base + at, pageBytes)
                            _ = pwrite(out, scratch, pageBytes, off_t(at))
                        }
                    }
                }
                fsync(out)
                close(out)
            }
        }
    }

    func writeBack() {
        if fd >= 0, let base {
            if scratch == nil { scratch = malloc(pageBytes) }
            if let scratch {
                var wrote = 0
                var short = 0
                var fault: Int32 = 0
                for i in kPages.indices where isLive(i) {
                    for v in [false, true] {
                        let at = offset(i, v: v)
                        memcpy(scratch, base + at, pageBytes)
                        let n = pwrite(fd, scratch, pageBytes, off_t(at))
                        if n == pageBytes {
                            wrote += 1
                        } else {
                            short += 1
                            if fault == 0 { fault = errno }
                        }
                    }
                }
                let synced = fsync(fd)
                Diag.memory?("writeBack \(wrote) page(s), \(short) short, "
                             + "errno \(fault), fsync \(synced)")
            }
        }
    }

    func attach(_ url: URL, first: Int = 0, len rows: Int = 0) throws {
        detach()
        let opened = open(url.path, O_RDWR | O_CREAT, 0o600)
        if opened < 0 { throw PoolError.open(url.path, errno) }
        ftruncate(opened, off_t(fileBytes))
        let raw = mmap(nil, fileBytes, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FILE, opened, 0)
        if raw == MAP_FAILED || raw == nil {
            let failed = errno
            close(opened)
            Diag.shared.report("[kv] mmap \(fileBytes >> 20) MB failed, "
                               + "errno \(failed): \(url.lastPathComponent)")
            throw PoolError.map(url.path, failed)
        }
        Diag.memory?("kv map \(fileBytes >> 20) MB, \(capacity) pages of "
                     + "\(pageBytes >> 10) KB")
        fd = opened
        base = raw
        let pages = (rows + P - 1) / P
        for i in 0..<pages {
            if i < first / P {
                kPages.append(zeroPage())
                vPages.append(zeroPage())
            } else {
                kPages.append(slice(i, v: false))
                vPages.append(slice(i, v: true))
            }
        }
        len = rows
        touched = rows
        refreshTable()
    }

    static let vmPage = 16384

    @_optimize(none)
    private static func mark(_ p: UnsafeMutableRawPointer) {
        p.storeBytes(of: p.load(as: UInt16.self), as: UInt16.self)
    }

    func touch() {
        let rowBytes = kvDim * MemoryLayout<Float16>.stride
        var row = min(touched, len)
        while row < len {
            let page = row / P
            let end = min(len, (page + 1) * P)
            if isLive(page) {
                var off = ((row % P) * rowBytes) / MetalKVPool.vmPage
                    * MetalKVPool.vmPage
                let stop = (end - page * P) * rowBytes
                while off < stop {
                    MetalKVPool.mark(kPages[page].contents() + off)
                    MetalKVPool.mark(vPages[page].contents() + off)
                    off += MetalKVPool.vmPage
                }
            }
            row = end
        }
        touched = max(0, len - 1)
    }

    func attachTemp() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kvpool-\(UUID().uuidString)")
        try attach(url)
        tempURL = url
    }

    func detach() {
        kPages = []
        vPages = []
        len = 0
        touched = 0
        if let base {
            munmap(base, fileBytes)
            self.base = nil
        }
        if fd >= 0 {
            close(fd)
            fd = -1
        }
    }

    private func offset(_ page: Int, v: Bool) -> Int {
        (v ? capacity + page : page) * pageBytes
    }

    private func slice(_ page: Int, v: Bool) -> MTLBuffer {
        if base == nil {
            do {
                try attachTemp()
            } catch {
                Diag.shared.report("[kv] cannot map \(fileBytes >> 20) MB: "
                                   + "\(error)")
            }
        }
        var out = zeroPage()
        if let base, page < capacity {
            let at = offset(page, v: v)
            var store = fstore_t(fst_flags: UInt32(F_ALLOCATEALL),
                                 fst_posmode: F_VOLPOSMODE,
                                 fst_offset: off_t(at),
                                 fst_length: off_t(pageBytes),
                                 fst_bytesalloc: 0)
            if fcntl(fd, F_PREALLOCATE, &store) < 0,
               errno == ENOSPC || errno == EDQUOT {
                starved?()
            }
            out = device.makeBuffer(
                bytesNoCopy: base + at, length: pageBytes,
                options: .storageModeShared, deallocator: nil)!
        } else {
            starved?()
        }
        return out
    }

    private func punch(_ page: Int) {
        for v in [false, true] {
            var hole = fpunchhole_t(fp_flags: 0, reserved: 0,
                                    fp_offset: off_t(offset(page, v: v)),
                                    fp_length: off_t(pageBytes))
            _ = fcntl(fd, F_PUNCHHOLE, &hole)
        }
    }

    private func zeroPage() -> MTLBuffer {
        if tombstone == nil {
            let page = device.makeBuffer(length: pageBytes,
                                         options: .storageModeShared)!
            memset(page.contents(), 0, page.length)
            tombstone = page
        }
        return tombstone!
    }

    func isLive(_ page: Int) -> Bool { kPages[page] !== tombstone }

    var livePages: Int { kPages.indices.filter(isLive).count }

    func evict(floor: Int) {
        if let window, len > 0 {
            let lo1 = max(0, floor - window + 1)
            let hi1 = floor - 1
            let lo2 = max(0, len - window)
            let hi2 = len - 1
            for i in kPages.indices where isLive(i) {
                let a = i * P
                let b = a + P - 1
                let kept = (hi1 >= lo1 && b >= lo1 && a <= hi1)
                    || (b >= lo2 && a <= hi2)
                if !kept {
                    kPages[i] = zeroPage()
                    vPages[i] = zeroPage()
                    punch(i)
                }
            }
            refreshTable()
        }
    }

    func firstLive(below rows: Int) -> Int {
        var first = 0
        let pages = (rows + P - 1) / P
        while first < pages && !isLive(first) { first += 1 }
        for i in first..<pages {
            precondition(isLive(i), "a hole below row \(rows): page \(i) of "
                         + "\(pages) was evicted inside the kept range")
        }
        return first * P
    }

    func tailForAppend() -> (k: MTLBuffer, v: MTLBuffer, slot: Int) {
        if len % P == 0 {
            let page = kPages.count
            kPages.append(slice(page, v: false))
            vPages.append(slice(page, v: true))
        }
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

    var residentPages: [MTLResource] { pages(rows: 0, len - 1) }

    var liveBytes: Int { 2 * livePages * pageBytes }

    var probe: String {
        var out = "empty"
        let page = (len - 1) / P
        if len > 0, isLive(page) {
            let rowBytes = kvDim * MemoryLayout<Float16>.stride
            let at = ((len - 1) % P) * rowBytes
            let k = kPages[page].contents()
                .assumingMemoryBound(to: UInt8.self) + at
            let v = vPages[page].contents()
                .assumingMemoryBound(to: UInt8.self) + at
            var sk = 0
            var sv = 0
            for i in 0..<rowBytes { sk += Int(k[i]); sv += Int(v[i]) }
            out = "row \(len - 1) k \(sk) v \(sv)"
        }
        return out
    }

    func pages(rows lo: Int, _ hi: Int) -> [MTLResource] {
        var out: [MTLResource] = []
        if !kPages.isEmpty, hi >= 0 {
            let first = max(0, lo) / P
            let last = min(hi / P, kPages.count - 1)
            if first <= last {
                for i in first...last where isLive(i) {
                    out.append(kPages[i])
                    out.append(vPages[i])
                }
            }
        }
        return out
    }

    var pageCount: Int { kPages.count }
    var addressableLength: Int { kPages.count * P }
    var pageBytesTotal: Int { 2 * livePages * pageBytes }

    func truncate(to newLength: Int) {
        precondition(newLength <= kPages.count * P,
                     "truncate to \(newLength) over \(kPages.count) pages: "
                     + "a bookmark cannot be restored across a reset")
        len = newLength
        touched = min(touched, newLength)
        let need = (newLength + P - 1) / P
        while kPages.count > need { kPages.removeLast(); vPages.removeLast() }
        if newLength % P != 0, !isLive(newLength / P) {
            kPages[newLength / P] = slice(newLength / P, v: false)
            vPages[newLength / P] = slice(newLength / P, v: true)
            refreshTable()
        }
    }

    struct Snapshot: @unchecked Sendable {
        let len: Int
        let kTail: Data
        let vTail: Data
    }

    private var tailBytes: Int { (len % P) * kvDim * MemoryLayout<Float16>.stride }

    func snapshot() -> Snapshot {
        var k = Data()
        var v = Data()
        if len % P != 0, !kPages.isEmpty {
            k = Data(bytes: kPages[len / P].contents(), count: tailBytes)
            v = Data(bytes: vPages[len / P].contents(), count: tailBytes)
        }
        return Snapshot(len: len, kTail: k, vTail: v)
    }

    func restore(_ s: Snapshot) {
        let need = (s.len + P - 1) / P
        while kPages.count > need { kPages.removeLast(); vPages.removeLast() }
        while kPages.count < need {
            let page = kPages.count
            kPages.append(slice(page, v: false))
            vPages.append(slice(page, v: true))
        }
        len = s.len
        touched = min(touched, s.len)
        if !s.kTail.isEmpty {
            s.kTail.withUnsafeBytes { raw in
                _ = memcpy(kPages[len / P].contents(), raw.baseAddress!,
                           raw.count)
            }
            s.vTail.withUnsafeBytes { raw in
                _ = memcpy(vPages[len / P].contents(), raw.baseAddress!,
                           raw.count)
            }
        }
        refreshTable()
    }
}
