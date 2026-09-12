
import Accelerate
import Foundation

let tensorMaxDims = 4

final class Tensor {
    var ne   = [Int64](repeating: 0, count: tensorMaxDims)
    var nb   = [Int64](repeating: 0, count: tensorMaxDims)
    var ndim: Int32 = 0
    var data: UnsafeMutablePointer<Float>!
    var arena: Arena?
    var name = ""
}

let tensorAlign = 64

private let arenaPageSizeCached: Int = {
    let ps = sysconf(_SC_PAGESIZE)
    return ps > 0 ? Int(ps) : 4096
}()

private func arenaPageSize() -> Int {
    return arenaPageSizeCached
}

private func arenaPagesAlloc(_ bytes: Int, _ outMapped: inout Int)
        -> UnsafeMutableRawPointer? {
    let pg = arenaPageSize()
    var rounded = (bytes + pg - 1) & ~(pg - 1)
    var result = mmap(nil, rounded, PROT_READ | PROT_WRITE,
                      MAP_ANON | MAP_PRIVATE, -1, 0)
    if result == MAP_FAILED { result = nil; rounded = 0 }
    outMapped = rounded
    return result
}

private func arenaPagesFree(_ p: UnsafeMutableRawPointer?,
                            _ mapped: Int) {
    if p != nil && mapped > 0 {
        munmap(p, mapped)
    }
}

final class ArenaSlab {
    var data: UnsafeMutableRawPointer?
    var capacity = 0
    var mapped   = 0
    var used     = 0
    var next: ArenaSlab?
}

private func arenaSlabNew(_ bytes: Int) -> ArenaSlab {
    let s = ArenaSlab()
    var mapped = 0
    s.data = arenaPagesAlloc(bytes, &mapped)
    assert(s.data != nil)
    s.capacity = bytes
    s.mapped = mapped
    s.used = 0
    s.next = nil
    return s
}

private func arenaSlabFree(_ s: ArenaSlab?) {
    if let s = s {
        arenaPagesFree(s.data, s.mapped)
        s.data = nil
        s.mapped = 0
    }
}

final class Arena {
    var head: ArenaSlab
    var first: ArenaSlab
    var initialBytes: Int

    init(_ initialBytes: Int) {
        let sz = initialBytes < 4096 ? 4096 : initialBytes
        self.initialBytes = sz
        self.first = arenaSlabNew(sz)
        self.head = self.first
    }
}

func arenaNew(_ initialBytes: Int) -> Arena {
    return Arena(initialBytes)
}

func arenaFree(_ a: Arena?) {
    if let a = a {
        var s: ArenaSlab? = a.first
        while s != nil {
            let next = s!.next
            arenaSlabFree(s)
            s = next
        }
    }
}

func arenaReset(_ a: Arena) {
    var s = a.first.next
    while s != nil {
        let next = s!.next
        arenaSlabFree(s)
        s = next
    }
    a.first.next = nil
    a.first.used = 0
    a.head = a.first
}

func arenaUsed(_ a: Arena) -> Int {
    var total = 0
    var s: ArenaSlab? = a.first
    while s != nil {
        total += s!.used
        s = s!.next
    }
    return total
}

func arenaCapacity(_ a: Arena) -> Int {
    var total = 0
    var s: ArenaSlab? = a.first
    while s != nil {
        total += s!.capacity
        s = s!.next
    }
    return total
}

// Every op allocates its OUTPUT into the ambient, THREAD-LOCAL arena, so a
// weight's products land in the scratch that resets between stages.

private final class ArenaSlot {
    var arena: Arena?
}

private let arenaSlotKey = "chatokf.tensor.arena"

private var arenaSlot: ArenaSlot {
    let store = Thread.current.threadDictionary
    var slot = store[arenaSlotKey] as? ArenaSlot
    if slot == nil {
        let fresh = ArenaSlot()
        store[arenaSlotKey] = fresh
        slot = fresh
    }
    return slot!
}

func arenaSetActive(_ a: Arena?) { arenaSlot.arena = a }
func arenaGetActive() -> Arena?  { return arenaSlot.arena }

private func arenaAout(_ fallback: Arena?) -> Arena {
    let active = arenaSlot.arena
    return active != nil ? active! : fallback!
}

private func arenaAlloc(_ a: Arena, _ bytes: Int)
        -> UnsafeMutableRawPointer {
    let rounded = (bytes + (tensorAlign - 1)) & ~(tensorAlign - 1)
    var s = a.head
    var alignedUsed = (s.used + (tensorAlign - 1))
                      & ~(tensorAlign - 1)
    if alignedUsed + rounded > s.capacity {
        let minNewSlab = 1 << 20
        var newCap = rounded + tensorAlign
        if newCap < minNewSlab { newCap = minNewSlab }
        let ns = arenaSlabNew(newCap)
        s.next = ns
        a.head = ns
        s = ns
        alignedUsed = 0
    }
    let out = s.data! + alignedUsed
    s.used = alignedUsed + rounded
    return out
}

private func tensorSetPackedStrides(_ t: Tensor) {
    t.nb[0] = Int64(MemoryLayout<Float>.size)
    for i in 1..<tensorMaxDims {
        t.nb[i] = t.nb[i - 1] * t.ne[i - 1]
    }
}

func tensorNelements(_ t: Tensor) -> Int64 {
    var n: Int64 = 1
    for i in 0..<tensorMaxDims {
        n *= t.ne[i]
    }
    return n
}

func tensorNbytes(_ t: Tensor) -> Int {
    return Int(tensorNelements(t)) * MemoryLayout<Float>.size
}

func tensorIsPacked(_ t: Tensor) -> Bool {
    var packed = true
    var expected = Int64(MemoryLayout<Float>.size)
    for i in 0..<tensorMaxDims {
        if t.ne[i] > 1 && t.nb[i] != expected { packed = false }
        expected *= t.ne[i]
    }
    return packed
}

func tensorSameShape(_ a: Tensor, _ b: Tensor) -> Bool {
    var same = (a.ndim == b.ndim)
    for i in 0..<tensorMaxDims {
        if a.ne[i] != b.ne[i] { same = false }
    }
    return same
}

private func tensorAllocHeader(_ a: Arena) -> Tensor {
    let t = Tensor()
    t.arena = a
    for i in 0..<tensorMaxDims { t.ne[i] = 1 }
    return t
}

private func tensorAllocWithData(_ a: Arena, _ ndim: Int32,
                                 _ n0: Int64, _ n1: Int64,
                                 _ n2: Int64, _ n3: Int64) -> Tensor {
    let t = tensorAllocHeader(a)
    t.ndim = ndim
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3
    tensorSetPackedStrides(t)
    let bytes = Int(tensorNelements(t)) * MemoryLayout<Float>.size
    t.data = arenaAlloc(a, bytes)
        .assumingMemoryBound(to: Float.self)
    return t
}

func tensorNew1d(_ a: Arena, _ n0: Int64) -> Tensor {
    return tensorAllocWithData(a, 1, n0, 1, 1, 1)
}

func tensorNew2d(_ a: Arena, _ n0: Int64, _ n1: Int64) -> Tensor {
    return tensorAllocWithData(a, 2, n0, n1, 1, 1)
}

func tensorNew3d(_ a: Arena, _ n0: Int64, _ n1: Int64,
                        _ n2: Int64) -> Tensor {
    return tensorAllocWithData(a, 3, n0, n1, n2, 1)
}

func tensorNew4d(_ a: Arena, _ n0: Int64, _ n1: Int64,
                        _ n2: Int64, _ n3: Int64) -> Tensor {
    return tensorAllocWithData(a, 4, n0, n1, n2, n3)
}

func tensorNewNd(_ a: Arena, _ ndim: Int32,
                        _ ne: [Int64]) -> Tensor {
    let n0 = ne[0], n1 = ne[1], n2 = ne[2], n3 = ne[3]
    var t: Tensor
    if ndim == 1      { t = tensorAllocWithData(a, 1, n0, 1,  1,  1)  }
    else if ndim == 2 { t = tensorAllocWithData(a, 2, n0, n1, 1,  1)  }
    else if ndim == 3 { t = tensorAllocWithData(a, 3, n0, n1, n2, 1)  }
    else              { t = tensorAllocWithData(a, 4, n0, n1, n2, n3) }
    return t
}

func tensorWrap1d(_ a: Arena, _ data: UnsafeMutablePointer<Float>,
                         _ n0: Int64) -> Tensor {
    let t = tensorAllocHeader(a)
    t.ndim = 1
    t.ne[0] = n0
    tensorSetPackedStrides(t)
    t.data = data
    return t
}

func tensorWrap2d(_ a: Arena, _ data: UnsafeMutablePointer<Float>,
                         _ n0: Int64, _ n1: Int64) -> Tensor {
    let t = tensorAllocHeader(a)
    t.ndim = 2
    t.ne[0] = n0; t.ne[1] = n1
    tensorSetPackedStrides(t)
    t.data = data
    return t
}

func tensorWrap3d(_ a: Arena, _ data: UnsafeMutablePointer<Float>,
                         _ n0: Int64, _ n1: Int64, _ n2: Int64) -> Tensor {
    let t = tensorAllocHeader(a)
    t.ndim = 3
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2
    tensorSetPackedStrides(t)
    t.data = data
    return t
}

func tensorWrap4d(_ a: Arena, _ data: UnsafeMutablePointer<Float>,
                         _ n0: Int64, _ n1: Int64, _ n2: Int64,
                         _ n3: Int64) -> Tensor {
    let t = tensorAllocHeader(a)
    t.ndim = 4
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3
    tensorSetPackedStrides(t)
    t.data = data
    return t
}

func tensorWrapNd(_ a: Arena, _ ndim: Int32,
                         _ data: UnsafeMutablePointer<Float>,
                         _ ne: [Int64]) -> Tensor {
    let n0 = ne[0], n1 = ne[1], n2 = ne[2], n3 = ne[3]
    var t: Tensor
    if ndim == 1      { t = tensorWrap1d(a, data, n0) }
    else if ndim == 2 { t = tensorWrap2d(a, data, n0, n1) }
    else if ndim == 3 { t = tensorWrap3d(a, data, n0, n1, n2) }
    else              { t = tensorWrap4d(a, data, n0, n1, n2, n3) }
    return t
}

func tensorSetName(_ t: Tensor, _ name: String) {
    var bytes = Array(name.utf8)
    if bytes.count > 31 { bytes = Array(bytes[0..<31]) }
    t.name = String(decoding: bytes, as: UTF8.self)
}

private func tensorMakeView(_ src: Tensor, _ ndim: Int32,
                            _ n0: Int64, _ n1: Int64,
                            _ n2: Int64, _ n3: Int64,
                            _ nb0: Int64, _ nb1: Int64,
                            _ nb2: Int64, _ nb3: Int64,
                            _ offset: Int) -> Tensor {
    let out = tensorAllocHeader(arenaAout(src.arena))
    out.ndim = ndim
    out.ne[0] = n0;  out.ne[1] = n1;  out.ne[2] = n2;  out.ne[3] = n3
    out.nb[0] = nb0; out.nb[1] = nb1; out.nb[2] = nb2; out.nb[3] = nb3
    out.data = (UnsafeMutableRawPointer(src.data) + offset)
        .assumingMemoryBound(to: Float.self)
    return out
}

func tensorView1d(_ t: Tensor, _ n0: Int64,
                         _ offset: Int) -> Tensor {
    return tensorMakeView(t, 1, n0, 1, 1, 1,
                          t.nb[0], t.nb[0] * n0,
                          t.nb[0] * n0, t.nb[0] * n0,
                          offset)
}

func tensorView2d(_ t: Tensor, _ n0: Int64, _ n1: Int64,
                         _ nb1: Int, _ offset: Int) -> Tensor {
    return tensorMakeView(t, 2, n0, n1, 1, 1,
                          t.nb[0], Int64(nb1),
                          Int64(nb1) * n1, Int64(nb1) * n1,
                          offset)
}

func tensorView3d(_ t: Tensor, _ n0: Int64, _ n1: Int64,
                         _ n2: Int64, _ nb1: Int, _ nb2: Int,
                         _ offset: Int) -> Tensor {
    return tensorMakeView(t, 3, n0, n1, n2, 1,
                          t.nb[0], Int64(nb1), Int64(nb2),
                          Int64(nb2) * n2,
                          offset)
}

func tensorReshape2d(_ src: Tensor, _ n0: Int64,
                            _ n1: Int64) -> Tensor {
    assert(tensorIsPacked(src))
    assert(n0 * n1 == tensorNelements(src))
    let t = tensorAllocHeader(arenaAout(src.arena))
    t.ndim = 2
    t.ne[0] = n0; t.ne[1] = n1
    tensorSetPackedStrides(t)
    t.data = src.data
    return t
}

func tensorReshape3d(_ src: Tensor, _ n0: Int64, _ n1: Int64,
                            _ n2: Int64) -> Tensor {
    assert(tensorIsPacked(src))
    assert(n0 * n1 * n2 == tensorNelements(src))
    let t = tensorAllocHeader(arenaAout(src.arena))
    t.ndim = 3
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2
    tensorSetPackedStrides(t)
    t.data = src.data
    return t
}

func tensorReshape4d(_ src: Tensor, _ n0: Int64, _ n1: Int64,
                            _ n2: Int64, _ n3: Int64) -> Tensor {
    assert(tensorIsPacked(src))
    assert(n0 * n1 * n2 * n3 == tensorNelements(src))
    let t = tensorAllocHeader(arenaAout(src.arena))
    t.ndim = 4
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3
    tensorSetPackedStrides(t)
    t.data = src.data
    return t
}

func tensorPermute(_ src: Tensor, _ p0: Int, _ p1: Int,
                          _ p2: Int, _ p3: Int) -> Tensor {
    assert(p0 >= 0 && p0 < 4)
    assert(p1 >= 0 && p1 < 4)
    assert(p2 >= 0 && p2 < 4)
    assert(p3 >= 0 && p3 < 4)
    let t = tensorAllocHeader(arenaAout(src.arena))
    t.ndim = src.ndim
    let perm = [p0, p1, p2, p3]
    for i in 0..<tensorMaxDims {
        t.ne[i] = src.ne[perm[i]]
        t.nb[i] = src.nb[perm[i]]
    }
    t.data = src.data
    return t
}

func tensorTranspose(_ src: Tensor) -> Tensor {
    return tensorPermute(src, 1, 0, 2, 3)
}

func tensorCont(_ src: Tensor) -> Tensor {
    let t = tensorAllocWithData(arenaAout(src.arena), src.ndim,
                                src.ne[0], src.ne[1],
                                src.ne[2], src.ne[3])
    let n0 = src.ne[0]
    let n1 = src.ne[1]
    let n2 = src.ne[2]
    let n3 = src.ne[3]
    let s0 = src.nb[0]
    let s1 = src.nb[1]
    let s2 = src.nb[2]
    let s3 = src.nb[3]
    let sb = UnsafeRawPointer(src.data!)
    var dst = t.data!
    for i3 in 0..<n3 {
        for i2 in 0..<n2 {
            for i1 in 0..<n1 {
                let ro: Int64 = i3 * s3 + i2 * s2 + i1 * s1
                let row = sb + Int(ro)
                if s0 == Int64(MemoryLayout<Float>.size) {
                    memcpy(dst, row,
                           Int(n0) * MemoryLayout<Float>.size)
                    dst += Int(n0)
                } else {
                    for i0 in 0..<n0 {
                        dst.pointee = (row + Int(i0 * s0))
                            .loadUnaligned(as: Float.self)
                        dst += 1
                    }
                }
            }
        }
    }
    return t
}

func tensorCont2d(_ src: Tensor, _ n0: Int64,
                         _ n1: Int64) -> Tensor {
    let packed = tensorCont(src)
    return tensorReshape2d(packed, n0, n1)
}

func tensorCpy(_ src: Tensor, _ dst: Tensor) {
    assert(tensorSameShape(src, dst))
    let n0 = src.ne[0]
    let n1 = src.ne[1]
    let n2 = src.ne[2]
    let n3 = src.ne[3]
    for i3 in 0..<n3 {
        for i2 in 0..<n2 {
            for i1 in 0..<n1 {
                let so: Int64 = i3 * src.nb[3] + i2 * src.nb[2]
                    + i1 * src.nb[1]
                let srow = UnsafeRawPointer(src.data!) + Int(so)
                let dof: Int64 = i3 * dst.nb[3] + i2 * dst.nb[2]
                    + i1 * dst.nb[1]
                let drow = UnsafeMutableRawPointer(dst.data!) + Int(dof)
                for i0 in 0..<n0 {
                    (drow + Int(i0 * dst.nb[0]))
                        .assumingMemoryBound(to: Float.self).pointee =
                        (srow + Int(i0 * src.nb[0]))
                            .loadUnaligned(as: Float.self)
                }
            }
        }
    }
}

func tensorConcat(_ a: Tensor, _ b: Tensor,
                         _ axis: Int) -> Tensor {
    assert(axis >= 0 && axis < tensorMaxDims)
    assert(a.ndim == b.ndim)
    for i in 0..<tensorMaxDims {
        if i != axis { assert(a.ne[i] == b.ne[i]) }
    }
    assert(tensorIsPacked(a) && tensorIsPacked(b))
    var outNe = [Int64](repeating: 0, count: tensorMaxDims)
    for i in 0..<tensorMaxDims {
        outNe[i] = (i == axis) ? (a.ne[i] + b.ne[i]) : a.ne[i]
    }
    let t = tensorAllocWithData(arenaAout(a.arena), a.ndim,
                                outNe[0], outNe[1],
                                outNe[2], outNe[3])
    var outer: Int64 = 1
    for i in (axis + 1)..<tensorMaxDims { outer *= a.ne[i] }
    var inner: Int64 = 1
    for i in 0..<axis { inner *= a.ne[i] }
    let rowA = Int(inner) * Int(a.ne[axis]) * MemoryLayout<Float>.size
    let rowB = Int(inner) * Int(b.ne[axis]) * MemoryLayout<Float>.size
    var dst = UnsafeMutableRawPointer(t.data!)
    let sa = UnsafeRawPointer(a.data!)
    let sb = UnsafeRawPointer(b.data!)
    for k in 0..<outer {
        memcpy(dst, sa + Int(k) * rowA, rowA)
        dst += rowA
        memcpy(dst, sb + Int(k) * rowB, rowB)
        dst += rowB
    }
    return t
}

func tensorRepeatTo(_ src: Tensor, _ ndim: Int32,
                           _ n0: Int64, _ n1: Int64,
                           _ n2: Int64, _ n3: Int64) -> Tensor {
    let template = Tensor()
    template.ndim = ndim
    template.ne[0] = n0; template.ne[1] = n1
    template.ne[2] = n2; template.ne[3] = n3
    return tensorRepeat(src, template)
}

func tensorRepeat(_ src: Tensor,
                         _ shapeLike: Tensor) -> Tensor {
    for i in 0..<tensorMaxDims {
        assert(src.ne[i] == 1 || src.ne[i] == shapeLike.ne[i])
    }
    let t = tensorAllocWithData(arenaAout(src.arena),
                                shapeLike.ndim,
                                shapeLike.ne[0],
                                shapeLike.ne[1],
                                shapeLike.ne[2],
                                shapeLike.ne[3])
    let n0 = t.ne[0], n1 = t.ne[1]
    let n2 = t.ne[2], n3 = t.ne[3]
    let r0: Int64 = src.ne[0] == 1 ? 0 : src.nb[0]
    let r1: Int64 = src.ne[1] == 1 ? 0 : src.nb[1]
    let r2: Int64 = src.ne[2] == 1 ? 0 : src.nb[2]
    let r3: Int64 = src.ne[3] == 1 ? 0 : src.nb[3]
    var dst = t.data!
    let sb = UnsafeRawPointer(src.data!)
    for i3 in 0..<n3 {
        for i2 in 0..<n2 {
            for i1 in 0..<n1 {
                let ro: Int64 = i3 * r3 + i2 * r2 + i1 * r1
                let row = sb + Int(ro)
                for i0 in 0..<n0 {
                    dst.pointee = (row + Int(i0 * r0))
                        .loadUnaligned(as: Float.self)
                    dst += 1
                }
            }
        }
    }
    return t
}

func tensorGetRows(_ data: Tensor, _ ids: [Int32],
                          _ nIds: Int) -> Tensor {
    assert(data.ndim == 2)
    assert(tensorIsPacked(data))
    let embed = data.ne[0]
    let vocab = data.ne[1]
    let t = tensorNew2d(arenaAout(data.arena), embed, Int64(nIds))
    for i in 0..<nIds {
        let row = ids[i]
        assert(row >= 0 && Int64(row) < vocab)
        memcpy(t.data + i * Int(embed),
               data.data + Int(row) * Int(embed),
               Int(embed) * MemoryLayout<Float>.size)
    }
    return t
}

private func tensorBroadcastable(_ x: Tensor, _ y: Tensor) -> Bool {
    var i = 0
    while i < tensorMaxDims
          && (y.ne[i] == 1 || y.ne[i] == x.ne[i]) {
        i += 1
    }
    return i == tensorMaxDims
}

enum TensorBin {
    case add, sub, mul, div
}

private func tensorVecVv(_ op: TensorBin, _ x: UnsafePointer<Float>,
                         _ y: UnsafePointer<Float>,
                         _ out: UnsafeMutablePointer<Float>,
                         _ n: Int64) {
    let N = vDSP_Length(n)
    switch op {
        case .add: vDSP_vadd(x, 1, y, 1, out, 1, N)
        case .sub: vDSP_vsub(y, 1, x, 1, out, 1, N)
        case .mul: vDSP_vmul(x, 1, y, 1, out, 1, N)
        case .div: vDSP_vdiv(y, 1, x, 1, out, 1, N)
    }
}

private func tensorVecVs(_ op: TensorBin, _ x: UnsafePointer<Float>,
                         _ s: Float,
                         _ out: UnsafeMutablePointer<Float>,
                         _ n: Int64) {
    let N = vDSP_Length(n)
    var s = s
    var arg: Float = 0.0
    switch op {
        case .add: vDSP_vsadd(x, 1, &s, out, 1, N)
        case .sub: arg = -s
                   vDSP_vsadd(x, 1, &arg, out, 1, N)
        case .mul: vDSP_vsmul(x, 1, &s, out, 1, N)
        case .div: arg = 1.0 / s
                   vDSP_vsmul(x, 1, &arg, out, 1, N)
    }
}

private func tensorScalarOp(_ op: TensorBin, _ a: Float,
                            _ b: Float) -> Float {
    switch op {
        case .add: return a + b
        case .sub: return a - b
        case .mul: return a * b
        case .div: return a / b
    }
}

private func tensorApplyBinop(_ x: Tensor, _ y: Tensor,
                              _ op: TensorBin) -> Tensor {
    assert(tensorBroadcastable(x, y))
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n0 = x.ne[0], n1 = x.ne[1]
    let n2 = x.ne[2], n3 = x.ne[3]
    let xb = UnsafePointer<Float>(x.data!)
    let yb = UnsafePointer<Float>(y.data!)
    let ob = out.data!
    let total = tensorNelements(x)
    if tensorSameShape(x, y) && tensorIsPacked(x)
       && tensorIsPacked(y) {
        tensorVecVv(op, xb, yb, ob, total)
    } else if tensorNelements(y) == 1 {
        if tensorIsPacked(x) {
            tensorVecVs(op, xb, yb[0], ob, total)
        } else {
            let s = yb[0]
            for i3 in 0..<n3 {
              for i2 in 0..<n2 {
                for i1 in 0..<n1 {
                  for i0 in 0..<n0 {
                    let xo: Int64 = i3 * x.nb[3] + i2 * x.nb[2]
                        + i1 * x.nb[1] + i0 * x.nb[0]
                    let xp = (UnsafeRawPointer(xb) + Int(xo))
                        .loadUnaligned(as: Float.self)
                    let oi = ((i3 * n2 + i2) * n1 + i1) * n0 + i0
                    ob[Int(oi)] = tensorScalarOp(op, xp, s)
                  }
                }
              }
            }
        }
    } else {
        let ys0: Int64 = (y.ne[0] == 1) ? 0 : y.nb[0]
        let ys1: Int64 = (y.ne[1] == 1) ? 0 : y.nb[1]
        let ys2: Int64 = (y.ne[2] == 1) ? 0 : y.nb[2]
        let ys3: Int64 = (y.ne[3] == 1) ? 0 : y.nb[3]
        let xInnerPacked =
            (x.nb[0] == Int64(MemoryLayout<Float>.size))
        let yInnerPackedOrScalar =
            (ys0 == 0) || (ys0 == Int64(MemoryLayout<Float>.size))
        for i3 in 0..<n3 {
          for i2 in 0..<n2 {
            for i1 in 0..<n1 {
              let xo: Int64 = i3 * x.nb[3] + i2 * x.nb[2]
                  + i1 * x.nb[1]
              let xrow = UnsafeRawPointer(xb) + Int(xo)
              let yo: Int64 = i3 * ys3 + i2 * ys2 + i1 * ys1
              let yrow = UnsafeRawPointer(yb) + Int(yo)
              let oo: Int64 = ((i3 * n2 + i2) * n1 + i1) * n0
              let orow = ob + Int(oo)
              if xInnerPacked && yInnerPackedOrScalar {
                  if ys0 == 0 {
                      tensorVecVs(op,
                          xrow.assumingMemoryBound(to: Float.self),
                          yrow.loadUnaligned(as: Float.self),
                          orow, n0)
                  } else {
                      tensorVecVv(op,
                          xrow.assumingMemoryBound(to: Float.self),
                          yrow.assumingMemoryBound(to: Float.self),
                          orow, n0)
                  }
              } else if ys0 == 0 {
                  let yv = yrow.loadUnaligned(as: Float.self)
                  for i0 in 0..<n0 {
                      let xv = (xrow + Int(i0 * x.nb[0]))
                          .loadUnaligned(as: Float.self)
                      orow[Int(i0)] = tensorScalarOp(op, xv, yv)
                  }
              } else {
                  for i0 in 0..<n0 {
                      let xv = (xrow + Int(i0 * x.nb[0]))
                          .loadUnaligned(as: Float.self)
                      let yv = (yrow + Int(i0 * ys0))
                          .loadUnaligned(as: Float.self)
                      orow[Int(i0)] = tensorScalarOp(op, xv, yv)
                  }
              }
            }
          }
        }
    }
    return out
}

func tensorAdd(_ x: Tensor, _ y: Tensor) -> Tensor {
    return tensorApplyBinop(x, y, .add)
}
func tensorSub(_ x: Tensor, _ y: Tensor) -> Tensor {
    return tensorApplyBinop(x, y, .sub)
}
func tensorMul(_ x: Tensor, _ y: Tensor) -> Tensor {
    return tensorApplyBinop(x, y, .mul)
}
func tensorDiv(_ x: Tensor, _ y: Tensor) -> Tensor {
    return tensorApplyBinop(x, y, .div)
}

enum TensorUnop {
    case scale, sigmoid, tanh, lrelu
    case gelu,  step,    sin,  cos
    case exp,   sqrt
}

private func tensorVecUnary(_ op: TensorUnop,
                            _ x: UnsafePointer<Float>,
                            _ out: UnsafeMutablePointer<Float>,
                            _ n: Int64, _ param: Float) {
    var len = Int32(n)
    var param = param
    let N = vDSP_Length(n)
    switch op {
    case .scale:
        var s = param
        vDSP_vsmul(x, 1, &s, out, 1, N)
    case .sigmoid:
        vDSP_vneg(x, 1, out, 1, N)
        vvexpf(out, out, &len)
        var one: Float = 1.0
        vDSP_vsadd(out, 1, &one, out, 1, N)
        vvrecf(out, out, &len)
    case .tanh: vvtanhf(out, x, &len)
    case .lrelu:
        vDSP_vsmul(x, 1, &param, out, 1, N)
        vDSP_vmax(x, 1, out, 1, out, 1, N)
    case .gelu:
        var invSqrt2 = Float(0.5.squareRoot())
        vDSP_vsmul(x, 1, &invSqrt2, out, 1, N)
        for i in 0..<n {
            let v = out[Int(i)]
            out[Int(i)] = 0.5 * x[Int(i)] * (1.0 + erff(v))
        }
    case .step:
        for i in 0..<n {
            out[Int(i)] = x[Int(i)] > 0.0 ? 1.0 : 0.0
        }
    case .sin:  vvsinf (out, x, &len)
    case .cos:  vvcosf (out, x, &len)
    case .exp:  vvexpf (out, x, &len)
    case .sqrt: vvsqrtf(out, x, &len)
    }
}

private func tensorScalarUnary(_ op: TensorUnop, _ v: Float,
                               _ p: Float) -> Float {
    switch op {
    case .scale:   return v * p
    case .sigmoid: return 1.0 / (1.0 + expf(-v))
    case .tanh:    return tanhf(v)
    case .lrelu:   return v > 0.0 ? v : v * p
    case .gelu:    return 0.5 * v *
                          (1.0 + erff(v * Float(0.5.squareRoot())))
    case .step:    return v > 0.0 ? 1.0 : 0.0
    case .sin:     return sinf(v)
    case .cos:     return cosf(v)
    case .exp:     return expf(v)
    case .sqrt:    return sqrtf(v)
    }
}

private func tensorApplyUnary(_ x: Tensor, _ op: TensorUnop,
                              _ param: Float) -> Tensor {
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n = tensorNelements(x)
    if tensorIsPacked(x) {
        tensorVecUnary(op, x.data, out.data, n, param)
    } else {
        let n0 = x.ne[0], n1 = x.ne[1]
        let n2 = x.ne[2], n3 = x.ne[3]
        for i3 in 0..<n3 {
            for i2 in 0..<n2 {
                for i1 in 0..<n1 {
                    for i0 in 0..<n0 {
                        let xo: Int64 = i3 * x.nb[3] + i2 * x.nb[2]
                            + i1 * x.nb[1] + i0 * x.nb[0]
                        let xp = (UnsafeRawPointer(x.data!) + Int(xo))
                            .loadUnaligned(as: Float.self)
                        let oi = ((i3 * n2 + i2) * n1 + i1)
                            * n0 + i0
                        out.data[Int(oi)] =
                            tensorScalarUnary(op, xp, param)
                    }
                }
            }
        }
    }
    return out
}

func tensorScale(_ x: Tensor, _ s: Float) -> Tensor {
    return tensorApplyUnary(x, .scale, s)
}

func tensorSigmoid(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .sigmoid, 0.0)
}

func tensorTanh(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .tanh, 0.0)
}

func tensorLeakyRelu(_ x: Tensor, _ slope: Float) -> Tensor {
    return tensorApplyUnary(x, .lrelu, slope)
}

func tensorGeluErf(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .gelu, 0.0)
}

func tensorStep(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .step, 0.0)
}

func tensorSin(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .sin, 0.0)
}

func tensorCos(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .cos, 0.0)
}

func tensorExp(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .exp, 0.0)
}

func tensorSqrt(_ x: Tensor) -> Tensor {
    return tensorApplyUnary(x, .sqrt, 0.0)
}

func tensorAtan2(_ y: Tensor, _ x: Tensor) -> Tensor {
    assert(tensorSameShape(x, y))
    assert(tensorIsPacked(x) && tensorIsPacked(y))
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n = tensorNelements(x)
    for i in 0..<n {
        out.data[Int(i)] = atan2f(y.data[Int(i)], x.data[Int(i)])
    }
    return out
}

func tensorNorm(_ x: Tensor, _ axis: Int,
                       _ eps: Float) -> Tensor {
    assert(axis == 0)
    assert(tensorIsPacked(x))
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n0 = x.ne[0]
    let outer = tensorNelements(x) / n0
    let xb = UnsafePointer<Float>(x.data!)
    let ob = out.data!
    let N = vDSP_Length(n0)
    for r in 0..<outer {
        let row = xb + Int(r * n0)
        let orow = ob + Int(r * n0)
        var mean: Float = 0.0
        var meanSq: Float = 0.0
        vDSP_meanv (row, 1, &mean,   N)
        vDSP_measqv(row, 1, &meanSq, N)
        let varv = meanSq - mean * mean
        var invstd = 1.0 / sqrtf(varv + eps)
        var bias = -mean * invstd
        vDSP_vsmsa(row, 1, &invstd, &bias, orow, 1, N)
    }
    return out
}

func tensorSoftmax(_ x: Tensor, _ axis: Int,
                          _ scale: Float) -> Tensor {
    assert(axis == 0)
    assert(tensorIsPacked(x))
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n0 = x.ne[0]
    let outer = tensorNelements(x) / n0
    let xb = UnsafePointer<Float>(x.data!)
    let ob = out.data!
    let N = vDSP_Length(n0)
    var Nint = Int32(n0)
    var scale = scale
    for r in 0..<outer {
        let row = xb + Int(r * n0)
        let orow = ob + Int(r * n0)
        vDSP_vsmul(row, 1, &scale, orow, 1, N)
        var mx: Float = 0.0
        vDSP_maxv(orow, 1, &mx, N)
        var negmx = -mx
        vDSP_vsadd(orow, 1, &negmx, orow, 1, N)
        vvexpf(orow, orow, &Nint)
        var sum: Float = 0.0
        vDSP_sve(orow, 1, &sum, N)
        var inv = 1.0 / sum
        vDSP_vsmul(orow, 1, &inv, orow, 1, N)
    }
    return out
}

func tensorCumsum(_ x: Tensor, _ axis: Int) -> Tensor {
    assert(axis == 0)
    assert(tensorIsPacked(x))
    let out = tensorAllocWithData(arenaAout(x.arena), x.ndim,
                                  x.ne[0], x.ne[1],
                                  x.ne[2], x.ne[3])
    let n0 = x.ne[0]
    let outer = tensorNelements(x) / n0
    let xb = UnsafePointer<Float>(x.data!)
    let ob = out.data!
    for r in 0..<outer {
        let row = xb + Int(r * n0)
        let orow = ob + Int(r * n0)
        var acc: Float = 0.0
        for i in 0..<n0 {
            acc += row[Int(i)]
            orow[Int(i)] = acc
        }
    }
    return out
}

private func tensorSgemmAccelerate(_ M: Int32, _ N: Int32, _ K: Int32,
                                   _ A: UnsafePointer<Float>,
                                   _ lda: Int32,
                                   _ Bt: UnsafePointer<Float>,
                                   _ ldb: Int32,
                                   _ C: UnsafeMutablePointer<Float>,
                                   _ ldc: Int32) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K,
                1.0, A, lda, Bt, ldb, 0.0, C, ldc)
}

private func tensorSgemmKernel4x8(_ K: Int32,
                                  _ A: UnsafePointer<Float>,
                                  _ lda: Int32,
                                  _ Bt: UnsafePointer<Float>,
                                  _ ldb: Int32,
                                  _ C: UnsafeMutablePointer<Float>,
                                  _ ldc: Int32) {
    var c00: Float = 0, c01: Float = 0, c02: Float = 0, c03: Float = 0
    var c04: Float = 0, c05: Float = 0, c06: Float = 0, c07: Float = 0
    var c10: Float = 0, c11: Float = 0, c12: Float = 0, c13: Float = 0
    var c14: Float = 0, c15: Float = 0, c16: Float = 0, c17: Float = 0
    var c20: Float = 0, c21: Float = 0, c22: Float = 0, c23: Float = 0
    var c24: Float = 0, c25: Float = 0, c26: Float = 0, c27: Float = 0
    var c30: Float = 0, c31: Float = 0, c32: Float = 0, c33: Float = 0
    var c34: Float = 0, c35: Float = 0, c36: Float = 0, c37: Float = 0
    let lda = Int(lda), ldb = Int(ldb), ldc = Int(ldc)
    for k in 0..<Int(K) {
        let a0 = A[0 * lda + k]
        let a1 = A[1 * lda + k]
        let a2 = A[2 * lda + k]
        let a3 = A[3 * lda + k]
        let b0 = Bt[0 * ldb + k]
        let b1 = Bt[1 * ldb + k]
        let b2 = Bt[2 * ldb + k]
        let b3 = Bt[3 * ldb + k]
        let b4 = Bt[4 * ldb + k]
        let b5 = Bt[5 * ldb + k]
        let b6 = Bt[6 * ldb + k]
        let b7 = Bt[7 * ldb + k]
        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3
        c04 += a0 * b4; c05 += a0 * b5; c06 += a0 * b6; c07 += a0 * b7
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3
        c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3
        c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3
        c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7
    }
    C[0 * ldc + 0] = c00; C[0 * ldc + 1] = c01
    C[0 * ldc + 2] = c02; C[0 * ldc + 3] = c03
    C[0 * ldc + 4] = c04; C[0 * ldc + 5] = c05
    C[0 * ldc + 6] = c06; C[0 * ldc + 7] = c07
    C[1 * ldc + 0] = c10; C[1 * ldc + 1] = c11
    C[1 * ldc + 2] = c12; C[1 * ldc + 3] = c13
    C[1 * ldc + 4] = c14; C[1 * ldc + 5] = c15
    C[1 * ldc + 6] = c16; C[1 * ldc + 7] = c17
    C[2 * ldc + 0] = c20; C[2 * ldc + 1] = c21
    C[2 * ldc + 2] = c22; C[2 * ldc + 3] = c23
    C[2 * ldc + 4] = c24; C[2 * ldc + 5] = c25
    C[2 * ldc + 6] = c26; C[2 * ldc + 7] = c27
    C[3 * ldc + 0] = c30; C[3 * ldc + 1] = c31
    C[3 * ldc + 2] = c32; C[3 * ldc + 3] = c33
    C[3 * ldc + 4] = c34; C[3 * ldc + 5] = c35
    C[3 * ldc + 6] = c36; C[3 * ldc + 7] = c37
}

private func tensorSgemmEdge(_ M: Int32, _ N: Int32, _ K: Int32,
                             _ A: UnsafePointer<Float>, _ lda: Int32,
                             _ Bt: UnsafePointer<Float>, _ ldb: Int32,
                             _ C: UnsafeMutablePointer<Float>,
                             _ ldc: Int32) {
    let lda = Int(lda), ldb = Int(ldb), ldc = Int(ldc)
    for m in 0..<Int(M) {
        for n in 0..<Int(N) {
            var acc: Float = 0.0
            for k in 0..<Int(K) {
                acc += A[m * lda + k] * Bt[n * ldb + k]
            }
            C[m * ldc + n] = acc
        }
    }
}

private func tensorSgemmTiled(_ M: Int32, _ N: Int32, _ K: Int32,
                              _ A: UnsafePointer<Float>, _ lda: Int32,
                              _ Bt: UnsafePointer<Float>, _ ldb: Int32,
                              _ C: UnsafeMutablePointer<Float>,
                              _ ldc: Int32) {
    let MR: Int32 = 4, NR: Int32 = 8
    let mMain = (M / MR) * MR
    let nMain = (N / NR) * NR
    for m in stride(from: 0, to: mMain, by: Int(MR)) {
        for n in stride(from: 0, to: nMain, by: Int(NR)) {
            tensorSgemmKernel4x8(K,
                                 A  + Int(m) * Int(lda), lda,
                                 Bt + Int(n) * Int(ldb), ldb,
                                 C  + Int(m) * Int(ldc) + Int(n), ldc)
        }
        if nMain < N {
            tensorSgemmEdge(MR, N - nMain, K,
                            A  + Int(m) * Int(lda), lda,
                            Bt + Int(nMain) * Int(ldb), ldb,
                            C  + Int(m) * Int(ldc) + Int(nMain), ldc)
        }
    }
    if mMain < M {
        tensorSgemmEdge(M - mMain, N, K,
                        A  + Int(mMain) * Int(lda), lda,
                        Bt, ldb,
                        C  + Int(mMain) * Int(ldc), ldc)
    }
}

// Accelerate is the only matmul; the hand-tiled sgemm beside it is the C's
// alternative, and the bit gate is defined over this one.

func tensorMulMat(_ w: Tensor, _ x: Tensor) -> Tensor {
    var w = w
    var x = x
    if !tensorIsPacked(w) { w = tensorCont(w) }
    if !tensorIsPacked(x) { x = tensorCont(x) }
    assert(w.ne[0] == x.ne[0])
    assert(w.ne[2] == x.ne[2])
    assert(w.ne[3] == x.ne[3])
    let K  = w.ne[0]
    let Nw = w.ne[1]
    let Nx = x.ne[1]
    let B2 = w.ne[2]
    let B3 = w.ne[3]
    let oa = arenaAout(w.arena)
    let ndim: Int32 = (B3 > 1) ? 4 : (B2 > 1) ? 3 : 2
    let ne: [Int64] = [Nw, Nx, B2, B3]
    let out = tensorNewNd(oa, ndim, ne)
    let wStride = Int(Nw) * Int(K)
    let xStride = Int(Nx) * Int(K)
    let oStride = Int(Nw) * Int(Nx)
    for b3 in 0..<B3 {
        for b2 in 0..<B2 {
            let b = Int(b3) * Int(B2) + Int(b2)
            let wB = UnsafePointer<Float>(w.data!) + b * wStride
            let xB = UnsafePointer<Float>(x.data!) + b * xStride
            let oB = out.data! + b * oStride
            tensorSgemmAccelerate(Int32(Nx), Int32(Nw), Int32(K),
                                  xB, Int32(K),
                                  wB, Int32(K),
                                  oB, Int32(Nw))
        }
    }
    return out
}

func tensorConvOutLen(_ LIn: Int64, _ K: Int, _ stride: Int,
                              _ pad: Int, _ dilation: Int) -> Int64 {
    return (LIn + Int64(2 * pad) - Int64(dilation * (K - 1)) - 1)
        / Int64(stride) + 1
}

func tensorIm2col(_ x: Tensor, _ kernel: Int, _ stride: Int,
                         _ pad: Int, _ dilation: Int) -> Tensor {
    assert(tensorIsPacked(x))
    let LIn = x.ne[0]
    let Cin = x.ne[1]
    let B   = x.ne[2]
    let Lout = tensorConvOutLen(LIn, kernel, stride, pad, dilation)
    let out = tensorNew3d(arenaAout(x.arena), Lout,
                          Cin * Int64(kernel), B)
    let xb = UnsafePointer<Float>(x.data!)
    let ob = out.data!
    for b in 0..<B {
        for c in 0..<Cin {
            for k in 0..<kernel {
                let off = b * Cin * Int64(kernel) * Lout
                    + (c * Int64(kernel) + Int64(k)) * Lout
                for lo in 0..<Lout {
                    let li = lo * Int64(stride) - Int64(pad)
                        + Int64(k * dilation)
                    var v: Float = 0.0
                    if li >= 0 && li < LIn {
                        v = xb[Int(b * Cin * LIn + c * LIn + li)]
                    }
                    ob[Int(off + lo)] = v
                }
            }
        }
    }
    return out
}

func tensorConv1d(_ w: Tensor, _ x: Tensor, _ stride: Int,
                         _ pad: Int, _ dilation: Int) -> Tensor {
    assert(tensorIsPacked(w))
    assert(tensorIsPacked(x))
    let K    = w.ne[0]
    let Cin  = w.ne[1]
    let Cout = w.ne[2]
    let LIn  = x.ne[0]
    let B    = x.ne[2]
    assert(x.ne[1] == Cin)
    let Lout = tensorConvOutLen(LIn, Int(K), stride, pad, dilation)
    let cols = tensorIm2col(x, Int(K), stride, pad, dilation)
    let out  = tensorNew3d(arenaAout(w.arena), Lout, Cout, B)
    for b in 0..<B {
        let cB = UnsafePointer<Float>(cols.data!)
            + Int(b) * Int(Cin * K) * Int(Lout)
        let oB = out.data! + Int(b) * Int(Cout) * Int(Lout)
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    Int32(Cout), Int32(Lout), Int32(Cin * K),
                    1.0,
                    w.data, Int32(Cin * K),
                    cB,     Int32(Lout),
                    0.0,
                    oB,     Int32(Lout))
    }
    return out
}

func tensorConv1dDw(_ w: Tensor, _ x: Tensor, _ stride: Int,
                           _ pad: Int, _ dilation: Int) -> Tensor {
    assert(tensorIsPacked(w))
    assert(tensorIsPacked(x))
    let K = w.ne[0]
    let C = w.ne[2]
    assert(w.ne[1] == 1)
    assert(x.ne[1] == C)
    let LIn = x.ne[0]
    let B   = x.ne[2]
    let Lout = tensorConvOutLen(LIn, Int(K), stride, pad, dilation)
    let out = tensorNew3d(arenaAout(w.arena), Lout, C, B)
    let xb = UnsafePointer<Float>(x.data!)
    let wb = UnsafePointer<Float>(w.data!)
    let ob = out.data!
    for b in 0..<B {
        for c in 0..<C {
            let xrow = xb + Int(b * C * LIn + c * LIn)
            let wrow = wb + Int(c * K)
            let orow = ob + Int(b * C * Lout + c * Lout)
            for lo in 0..<Lout {
                var acc: Float = 0.0
                for k in 0..<Int(K) {
                    let li = lo * Int64(stride) - Int64(pad)
                        + Int64(k * dilation)
                    if li >= 0 && li < LIn {
                        acc += wrow[k] * xrow[Int(li)]
                    }
                }
                orow[Int(lo)] = acc
            }
        }
    }
    return out
}

func tensorConvTranspose1d(_ w: Tensor, _ x: Tensor,
                                  _ stride: Int, _ pad: Int) -> Tensor {
    assert(tensorIsPacked(w))
    assert(tensorIsPacked(x))
    let K    = w.ne[0]
    let Cout = w.ne[1]
    let Cin  = w.ne[2]
    let Lin  = x.ne[0]
    let B    = x.ne[2]
    assert(x.ne[1] == Cin)
    let Lfull = (Lin - 1) * Int64(stride) + K
    var Lout = Lfull - Int64(2 * pad)
    if Lout < 0 { Lout = 0 }
    let out = tensorNew3d(arenaAout(w.arena), Lout, Cout, B)
    memset(out.data, 0, tensorNbytes(out))
    for b in 0..<B {
        for co in 0..<Cout {
            let orow = out.data! + Int(b * Cout * Lout + co * Lout)
            for ci in 0..<Cin {
                let xrow = UnsafePointer<Float>(x.data!)
                    + Int(b * Cin * Lin + ci * Lin)
                let wrow = UnsafePointer<Float>(w.data!)
                    + Int(ci * Cout * K + co * K)
                for li in 0..<Lin {
                    let xv = xrow[Int(li)]
                    let base = li * Int64(stride) - Int64(pad)
                    for k in 0..<K {
                        let lo = base + k
                        if lo >= 0 && lo < Lout {
                            orow[Int(lo)] += wrow[Int(k)] * xv
                        }
                    }
                }
            }
        }
    }
    return out
}
