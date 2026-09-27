import XCTest
@testable import LLM

final class TokenizerTests: XCTestCase {
    // The set sits either directly under models/Qwen3.5-0.8B (a dev copy) or
    // in a {sha} subdir (the HubFetch staging layout).
    private func modelsDir() -> URL? {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let base = root.appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var found: URL? = nil
        if fm.fileExists(
            atPath: base.appendingPathComponent("tokenizer.json").path) {
            found = base
        } else if let subs = try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) {
            for sub in subs where fm.fileExists(
                atPath: sub.appendingPathComponent("tokenizer.json").path) {
                found = sub
            }
        }
        return found
    }

    func testParityWithReference() throws {
        guard let dir = modelsDir() else {
            throw XCTSkip("models/ not present")
        }
        let tok = try Tokenizer(modelsDir: dir)
        let text = "The ISDA Master Agreement is the most widely "
            + "used, don't you think? <|im_start|>user"
        let expected: [Int32] = [
            760, 3314, 6151, 10507, 21800, 369, 279, 1379, 13177, 1429,
            11, 1459, 914, 488, 1683, 30, 220, 248045, 846,
        ]
        XCTAssertEqual(tok.encode(text, addSpecial: true), expected)
        XCTAssertEqual(tok.decode(expected), text)
    }

    private struct ReferenceMerges {
        private struct Rule {
            let rank: Int32
            let merged: Int32
        }

        private let symbol: [String]
        private let charId: [Character: Int32]
        private let rule: [UInt64: Rule]

        private static func key(_ a: Int32, _ b: Int32) -> UInt64 {
            UInt64(UInt32(bitPattern: a)) << 32
                | UInt64(UInt32(bitPattern: b))
        }

        init(_ merges: [String]) {
            var text: [String] = []
            var ids = [String: Int32](minimumCapacity: merges.count * 2)
            var rules = [UInt64: Rule](minimumCapacity: merges.count)
            func intern(_ s: String) -> Int32 {
                let known = ids[s]
                let out: Int32
                if let known {
                    out = known
                } else {
                    out = Int32(text.count)
                    ids[s] = out
                    text.append(s)
                }
                return out
            }
            for (i, m) in merges.enumerated() {
                if let space = m.firstIndex(of: " ") {
                    let a = String(m[..<space])
                    let b = String(m[m.index(after: space)...])
                    if a.last != nil && b.first != nil {
                        let pair = ReferenceMerges.key(intern(a), intern(b))
                        rules[pair] = Rule(rank: Int32(i),
                                           merged: intern(a + b))
                    }
                }
            }
            var chars = [Character: Int32](minimumCapacity: 8192)
            for (i, s) in text.enumerated() where s.count == 1 {
                chars[s.first!] = Int32(i)
            }
            symbol = text
            charId = chars
            rule = rules
        }

        func merged(_ token: ArraySlice<Character>) -> [String] {
            var extra: [String] = []
            var word: [Int32] = []
            word.reserveCapacity(token.count)
            for ch in token {
                if let id = charId[ch] {
                    word.append(id)
                } else {
                    word.append(Int32(symbol.count + extra.count))
                    extra.append(String(ch))
                }
            }
            var merging = word.count >= 2
            while merging {
                var minRank = Int32.max
                var at = -1
                for i in 0 ..< (word.count - 1) {
                    let found = rule[ReferenceMerges.key(word[i], word[i + 1])]
                    if let found, found.rank < minRank {
                        minRank = found.rank
                        at = i
                    }
                }
                if at < 0 {
                    merging = false
                } else {
                    let a = word[at]
                    let b = word[at + 1]
                    let into = rule[ReferenceMerges.key(a, b)]!.merged
                    var next: [Int32] = []
                    next.reserveCapacity(word.count)
                    var i = 0
                    while i < word.count {
                        if i < word.count - 1 && word[i] == a
                            && word[i + 1] == b {
                            next.append(into)
                            i += 2
                        } else {
                            next.append(word[i])
                            i += 1
                        }
                    }
                    word = next
                    merging = word.count >= 2
                }
            }
            var out: [String] = []
            out.reserveCapacity(word.count)
            for id in word {
                let i = Int(id)
                out.append(i < symbol.count ? symbol[i]
                                            : extra[i - symbol.count])
            }
            return out
        }

        func symbols(_ text: String, segmenting table: MergeTable)
            -> [String] {
            var out: [String] = []
            for segment in table.segments(text) {
                out.append(contentsOf: merged(segment))
            }
            return out
        }
    }

    private struct Lcg {
        var state: UInt64
        mutating func next(_ bound: Int) -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(bound))
        }
    }

    private static let chinese: [String] = [
        "长江是亚洲第一长河，发源于青藏高原，流经十一个省份后注入东海。",
        "北京的秋天天气晴朗，人们喜欢到香山看红叶，到公园里散步聊天。",
        "这家餐厅的招牌菜是红烧肉和清蒸鱼，味道地道，价格也很公道。",
        "图书馆每天早上八点开门，学生们排队进去占座位复习功课。",
        "科学家发现了一种新的深海生物，它能够在没有阳光的环境中生存。",
        "春节期间，家家户户贴春联、包饺子，晚上还要一起看电视节目。",
        "高速铁路把两座城市之间的旅行时间从五个小时缩短到了一个半小时。",
        "他每天坚持跑步半小时，一年下来体重减轻了十公斤，精神也好多了。",
        "这部电影讲述了一位老师在山区支教二十年的故事，感动了无数观众。",
        "手机电池的续航能力取决于屏幕亮度、网络信号和后台运行的程序。",
        "博物馆新展出的青铜器距今已有三千多年历史，纹饰精美，保存完好。",
        "经济学家认为，利率的变化会影响企业投资和普通家庭的消费决策。",
    ]

    private static let english: [String] = [
        "The quarterly logistics review covered warehouse throughput. ",
        "A river slows as it meets standing water and drops its load. ",
        "She parked the bicycle under the awning before the rain came. ",
        "Interest rates shape both corporate investment and households. ",
        "The museum's bronze vessels are three thousand years old. ",
        "He ran for half an hour every morning and lost ten kilograms. ",
        "The library opens at eight and the queue forms well before. ",
        "Battery life depends on brightness, signal and background apps. ",
    ]

    private func paragraph(_ sentences: [String], chars: Int,
                           seed: UInt64) -> String {
        var rng = Lcg(state: seed)
        var out = ""
        while out.count < chars {
            out += sentences[rng.next(sentences.count)]
        }
        return String(out.prefix(chars))
    }

    private func randomRun(chars: Int, seed: UInt64,
                           first: UInt32, count: UInt32) -> String {
        var rng = Lcg(state: seed)
        var out = ""
        for _ in 0 ..< chars {
            out.unicodeScalars.append(
                UnicodeScalar(first + UInt32(rng.next(Int(count))))!)
        }
        return out
    }

    private func corpus() -> [(String, String)] {
        [("cjk paragraph", paragraph(TokenizerTests.chinese, chars: 3000,
                                     seed: 1)),
         ("cjk random", randomRun(chars: 3000, seed: 2, first: 0x4E00,
                                  count: 0x51A5)),
         ("ascii prose", paragraph(TokenizerTests.english, chars: 3000,
                                   seed: 3)),
         ("ascii word", randomRun(chars: 3000, seed: 4, first: 0x61,
                                  count: 26))]
    }

    private func gpt2Mapped(_ text: String) -> String {
        let b2u = Tokenizer.bytesToUnicode()
        var out = ""
        for byte in text.utf8 { out.append(b2u[byte]!) }
        return out
    }

    private func compare(_ label: String, _ text: String,
                         _ table: MergeTable, _ old: ReferenceMerges) {
        let began = Date()
        let want = old.symbols(text, segmenting: table)
        let oldSeconds = Date().timeIntervalSince(began)
        let again = Date()
        let got = table.symbols(text)
        let newSeconds = Date().timeIntervalSince(again)
        XCTAssertEqual(got, want, label)
        print(String(format: "[bpe] %@: %d chars, %d symbols, old %.4f s, "
                         + "new %.4f s", label, text.count, got.count,
                     oldSeconds, newSeconds))
    }

    func testMergesMatchTheReferenceOnLongWords() throws {
        let sets = [("Qwen3.5-4B", qwenGgufPath),
                    ("gemma-4-E2B", gemmaGgufPath)]
        var seen = 0
        for (name, path) in sets {
            if let path {
                seen += 1
                let g = try GGUF(path: path)
                let merges = g.strings("tokenizer.ggml.merges") ?? []
                let table = MergeTable(merges)
                let old = ReferenceMerges(merges)
                let qwen = name.hasPrefix("Qwen")
                for (label, text) in corpus() {
                    let shaped = qwen ? gpt2Mapped(text)
                        : text.replacingOccurrences(of: " ", with: "\u{2581}")
                    compare(name + " " + label, shaped, table, old)
                }
                if qwen {
                    let tok = try Tokenizer(gguf: g)
                    let text = corpus()[0].1
                    let began = Date()
                    let ids = tok.encode(text)
                    print(String(format: "[bpe] Tokenizer.encode of %d cjk "
                                     + "chars: %d ids in %.4f s", text.count,
                                 ids.count, Date().timeIntervalSince(began)))
                }
            }
        }
        if seen == 0 { throw XCTSkip("no Qwen3.5-4B or gemma-4-E2B gguf") }
    }
}
