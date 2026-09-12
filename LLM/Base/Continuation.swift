import Foundation

public enum Continuation {
    // A block's OWN inner placeholders pass through untouched: they are the
    // soft positions the features land on, not further attachments.
    public static func expandSpans(_ ids: [Int32], _ spans: [SoftSpan]) -> [Int32] {
        var queues: [Int32: [SoftSpan]] = [:]
        for span in spans {
            queues[span.placeholder, default: []].append(span)
        }
        var used: [Int32: Int] = [:]
        var out: [Int32] = []
        var i = 0
        while i < ids.count {
            let id = ids[i]
            let at = used[id] ?? 0
            if let queue = queues[id], at < queue.count {
                let span = queue[at]
                if let wrap = span.wrap, out.last == wrap.begin {
                    out.removeLast()
                }
                out.append(contentsOf: span.ids)
                used[id] = at + 1
                if let wrap = span.wrap, i + 1 < ids.count,
                   ids[i + 1] == wrap.end {
                    i += 1
                }
            } else {
                out.append(id)
            }
            i += 1
        }
        return out
    }

    // 5 identical short blocks (k <= 4) or 3 long ones (k <= 64) is a loop;
    // a k-gram of only structural bytes needs 24, a table separator repeats.
    static func isLooping(_ ids: [Int32], reps: Int = 5,
                          structuralReps: Int = 24,
                          longK: Int = 64, longReps: Int = 3,
                          tokenBytes: ((Int32) -> [UInt8])? = nil) -> Bool {
        var result = false
        var k = 1
        while k <= longK && !result {
            let structural = tokenBytes.map { bytes in
                structuralGram(ids, k, bytes)
            } ?? false
            let need = structural ? structuralReps
                                  : (k <= 4 ? reps : longReps)
            result = tailRepeats(ids, k, need)
            k += 1
        }
        return result
    }

    // Numerals with the group comma are structural: a large number is a
    // legitimate ",000" cycle, like a table's separator row.
    private static let structuralBytes: Set<UInt8> =
        Set("|-=+:_*#~. \t\n0123456789,".utf8)

    private static func structuralGram(_ ids: [Int32], _ k: Int,
                                       _ bytes: (Int32) -> [UInt8]) -> Bool {
        var structural = ids.count >= k
        var i = max(0, ids.count - k)
        while structural && i < ids.count {
            let b = bytes(ids[i])
            structural = !b.isEmpty && b.allSatisfy { byte in
                structuralBytes.contains(byte)
            }
            i += 1
        }
        return structural
    }

    private static func tailRepeats(_ ids: [Int32], _ k: Int,
                                    _ reps: Int) -> Bool {
        var result = ids.count >= k * reps
        var block = 1
        while block < reps && result {
            var i = 0
            while i < k && result {
                let a = ids[ids.count - 1 - i]
                let b = ids[ids.count - 1 - block * k - i]
                if a != b { result = false }
                i += 1
            }
            block += 1
        }
        return result
    }
}
