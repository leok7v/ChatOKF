import Foundation

public enum GGUFGraft {
    public struct Report: Sendable {
        public var text = 0
        public var tower = 0
        public var keys = 0
        public var bytes = 0
    }

    public static func graft(text: String, donor: String,
                             to out: String,
                             tensors: [String] = ["v.", "mm."],
                             keys: [String] = ["clip."]) throws -> Report {
        let lhs = try GGUF(path: text)
        let rhs = try GGUF(path: donor)
        let lhsHead = try Header(path: text)
        let rhsHead = try Header(path: donor)
        let present = Set(lhsHead.order)
        let tower = rhsHead.order.filter { name in
            tensors.contains { p in name.hasPrefix(p) }
                && !present.contains(name)
        }
        let known = Set(lhsHead.kv.map { pair in pair.0 })
        let wanted = rhsHead.kv.filter { pair in
            keys.contains { p in pair.0.hasPrefix(p) }
        }
        let supplied = Dictionary(wanted, uniquingKeysWith: { a, _ in a })
        let clip = wanted.filter { pair in !known.contains(pair.0) }
        var merged: [(String, Data)] = []
        var placed = false
        for pair in lhsHead.kv {
            if pair.0.hasPrefix("tokenizer.") && !placed {
                merged.append(contentsOf: clip)
                placed = true
            }
            merged.append((pair.0, supplied[pair.0] ?? pair.1))
        }
        if !placed { merged.append(contentsOf: clip) }

        let align = lhs.int("general.alignment") ?? 32
        let writer = try GGUFWriter(path: out, alignment: align)
        for pair in merged { writer.metaRaw(pair.0, pair.1) }
        for name in lhsHead.order {
            let t = lhs.tensor(name)
            writer.declare(name, dims: t.dims, type: t.type)
        }
        for name in tower {
            let t = rhs.tensor(name)
            writer.declare(name, dims: t.dims, type: t.type)
        }
        writer.finishHeader()
        for name in lhsHead.order {
            let t = lhs.tensor(name)
            try writer.append(UnsafeRawBufferPointer(start: t.base,
                                                     count: t.byteCount),
                              expecting: name)
        }
        for name in tower {
            let t = rhs.tensor(name)
            try writer.append(UnsafeRawBufferPointer(start: t.base,
                                                     count: t.byteCount),
                              expecting: name)
        }
        try writer.close()
        var report = Report()
        report.text = lhsHead.order.count
        report.tower = tower.count
        report.keys = merged.count
        report.bytes = writer.declaredBytes
        return report
    }

    public static func meta(_ src: String, to out: String,
                            config: String) throws -> Report {
        let g = try GGUF(path: src)
        let head = try Header(path: src)
        let json = try String(contentsOf: URL(fileURLWithPath: config),
                              encoding: .utf8)
        let obj = try JSONSerialization.jsonObject(with: Data(json.utf8))
        let dict = obj as? [String: Any]
        if dict?["_sampling_presets"] == nil {
            throw GGUFErr.parse("\(config) has no _sampling_presets")
        }
        var eos: [Int32] = []
        if let one = dict?["eos_token_id"] as? NSNumber {
            eos = [one.int32Value]
        } else if let many = dict?["eos_token_id"] as? [NSNumber] {
            eos = many.map { n in n.int32Value }
        }
        let arch = g.string("general.architecture") ?? ""
        let drop: Set<String> = ["general.sampling.presets_json",
                                 arch + ".source.generation_config_json"]
        let cfgKey = Tokenizer.generationConfigKey
        let eosKey = "tokenizer.ggml.eos_token_ids"
        let cfgVal = GGUFWriter.encoded(.string(json))
        let eosVal = GGUFWriter.encoded(.i32s(eos))
        var merged: [(String, Data)] = []
        var wroteCfg = false
        var wroteEos = false
        for pair in head.kv where !drop.contains(pair.0) {
            if pair.0 == cfgKey {
                merged.append((cfgKey, cfgVal))
                wroteCfg = true
            } else if pair.0 == eosKey && !eos.isEmpty {
                merged.append((eosKey, eosVal))
                wroteEos = true
            } else {
                merged.append(pair)
            }
        }
        if !wroteCfg { merged.append((cfgKey, cfgVal)) }
        if !wroteEos && !eos.isEmpty { merged.append((eosKey, eosVal)) }

        let align = g.int("general.alignment") ?? 32
        let writer = try GGUFWriter(path: out, alignment: align)
        for pair in merged { writer.metaRaw(pair.0, pair.1) }
        for name in head.order {
            let t = g.tensor(name)
            writer.declare(name, dims: t.dims, type: t.type)
        }
        writer.finishHeader()
        for name in head.order {
            let t = g.tensor(name)
            try writer.append(UnsafeRawBufferPointer(start: t.base,
                                                     count: t.byteCount),
                              expecting: name)
        }
        try writer.close()
        var report = Report()
        report.text = head.order.count
        report.keys = merged.count
        report.tower = eos.count
        report.bytes = writer.declaredBytes
        return report
    }

    public static func splice(base: String, donor: String, leaves: Set<String>,
                              to out: String) throws -> Report {
        let lhs = try GGUF(path: base)
        let rhs = try GGUF(path: donor)
        let head = try Header(path: base)
        let taken = head.order.filter { name in
            leaves.contains(leafOf(name)) && rhs.maybe(name) != nil
        }
        let swap = Set(taken)
        let align = lhs.int("general.alignment") ?? 32
        let writer = try GGUFWriter(path: out, alignment: align)
        for pair in head.kv { writer.metaRaw(pair.0, pair.1) }
        for name in head.order {
            let t = swap.contains(name) ? rhs.tensor(name) : lhs.tensor(name)
            writer.declare(name, dims: t.dims, type: t.type)
        }
        writer.finishHeader()
        for name in head.order {
            let t = swap.contains(name) ? rhs.tensor(name) : lhs.tensor(name)
            try writer.append(UnsafeRawBufferPointer(start: t.base,
                                                     count: t.byteCount),
                              expecting: name)
        }
        try writer.close()
        var report = Report()
        report.text = head.order.count
        report.tower = taken.count
        report.keys = head.kv.count
        report.bytes = writer.declaredBytes
        return report
    }

    static func leafOf(_ name: String) -> String {
        let part = name.split(separator: ".")
        return part.count > 2 && part[0] == "blk"
            ? part[2...].joined(separator: ".") : name
    }
}

private struct Header {
    var kv: [(String, Data)] = []
    var order: [String] = []

    init(path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path),
                            options: .mappedIfSafe)
        var at = 0
        let magic = Header.u32(data, at)
        at += 4
        if magic != 0x4655_4747 {
            throw GGUFErr.parse("bad magic in \(path)")
        } else {
            at += 4
            let tensors = Int(Header.u64(data, at))
            at += 8
            let count = Int(Header.u64(data, at))
            at += 8
            for _ in 0..<count {
                let key = Header.str(data, &at)
                let from = at
                Header.skipValue(data, &at)
                kv.append((key, data.subdata(in: from..<at)))
            }
            for _ in 0..<tensors {
                order.append(Header.str(data, &at))
                let nd = Int(Header.u32(data, at))
                at += 4 + nd * 8 + 4 + 8
            }
        }
    }

    static func u32(_ d: Data, _ at: Int) -> UInt32 {
        d.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: at, as: UInt32.self)
        }
    }

    static func u64(_ d: Data, _ at: Int) -> UInt64 {
        d.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: at, as: UInt64.self)
        }
    }

    static func str(_ d: Data, _ at: inout Int) -> String {
        let n = Int(u64(d, at))
        at += 8
        let s = String(decoding: d.subdata(in: at..<(at + n)), as: UTF8.self)
        at += n
        return s
    }

    static func width(_ tag: UInt32) -> Int {
        let out: Int
        switch tag {
        case 0, 1, 7: out = 1
        case 2, 3: out = 2
        case 4, 5, 6: out = 4
        default: out = 8
        }
        return out
    }

    static func skipValue(_ d: Data, _ at: inout Int) {
        let tag = u32(d, at)
        at += 4
        skipScalar(d, &at, tag)
    }

    static func skipScalar(_ d: Data, _ at: inout Int, _ tag: UInt32) {
        if tag == 8 {
            at += 8 + Int(u64(d, at))
        } else if tag == 9 {
            let element = u32(d, at)
            at += 4
            let n = Int(u64(d, at))
            at += 8
            if element == 8 {
                for _ in 0..<n { at += 8 + Int(u64(d, at)) }
            } else {
                at += n * width(element)
            }
        } else {
            at += width(tag)
        }
    }
}
