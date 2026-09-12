import XCTest
@testable import LLM

// A JSON-like model backing the six host callbacks, mirroring the JN host in
// the C golden harness.
private indirect enum MV {
    case obj([(String, MV)])
    case arr([MV])
    case str(String)
    case int(Int)
    case bool(Bool)
}

extension MV: ExpressibleByIntegerLiteral {
    init(integerLiteral value: Int) { self = .int(value) }
}

extension MV: ExpressibleByStringLiteral {
    init(stringLiteral value: String) { self = .str(value) }
}

extension MV: ExpressibleByBooleanLiteral {
    init(booleanLiteral value: Bool) { self = .bool(value) }
}

private final class Model: JinjaHost {
    private let hostTag = 7
    private var isObj: [Bool] = []
    private var objEntries: [[(String, JinjaValue)]] = []
    private var arrItems: [[JinjaValue]] = []
    private var json: [String] = []

    func val(_ mv: MV) -> JinjaValue {
        intern(mv)
    }

    private func jsonOf(_ v: JinjaValue) -> String {
        switch v {
        case .str(let s): return "\"\(s)\""
        case .int(let n): return "\(n)"
        case .bool(let b): return b ? "true" : "false"
        case .node(let h, _): return json[h]
        default: return "null"
        }
    }

    private func intern(_ mv: MV) -> JinjaValue {
        var result = JinjaValue.undefined
        switch mv {
        case .str(let s): result = .str(s)
        case .int(let n): result = .int(n)
        case .bool(let b): result = .bool(b)
        case .obj(let entries): result = internObj(entries)
        case .arr(let elems): result = internArr(elems)
        }
        return result
    }

    private func internObj(_ entries: [(String, MV)]) -> JinjaValue {
        var resolved: [(String, JinjaValue)] = []
        var parts: [String] = []
        for entry in entries {
            let jv = intern(entry.1)
            resolved.append((entry.0, jv))
            parts.append("\"\(entry.0)\": \(jsonOf(jv))")
        }
        let h = isObj.count
        isObj.append(true)
        objEntries.append(resolved)
        arrItems.append([])
        json.append("{" + parts.joined(separator: ", ") + "}")
        return .node(handle: h, tag: hostTag)
    }

    private func internArr(_ elems: [MV]) -> JinjaValue {
        var resolved: [JinjaValue] = []
        var parts: [String] = []
        for elem in elems {
            let jv = intern(elem)
            resolved.append(jv)
            parts.append(jsonOf(jv))
        }
        let h = isObj.count
        isObj.append(false)
        objEntries.append([])
        arrItems.append(resolved)
        json.append("[" + parts.joined(separator: ", ") + "]")
        return .node(handle: h, tag: hostTag)
    }

    private func handle(_ v: JinjaValue) -> Int? {
        var result: Int? = nil
        if case .node(let h, let t) = v, t == hostTag { result = h }
        return result
    }

    private func count(_ h: Int) -> Int {
        isObj[h] ? objEntries[h].count : arrItems[h].count
    }

    func get(_ obj: JinjaValue, _ name: String) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj), isObj[h] {
            for entry in objEntries[h] where entry.0 == name {
                result = entry.1
            }
        }
        return result
    }

    func index(_ obj: JinjaValue, _ i: Int) -> JinjaValue {
        var result = JinjaValue.undefined
        if let h = handle(obj) {
            result = isObj[h] ? .str(objEntries[h][i].0) : arrItems[h][i]
        }
        return result
    }

    func len(_ obj: JinjaValue) -> Int {
        var result = 0
        if let h = handle(obj) { result = count(h) }
        return result
    }

    func truthy(_ v: JinjaValue) -> Bool {
        var result = false
        if let h = handle(v) { result = count(h) != 0 }
        return result
    }

    func test(_ v: JinjaValue, _ name: String) -> Bool {
        var result = false
        if let h = handle(v) {
            if name == "mapping" {
                result = isObj[h]
            } else if name == "sequence" || name == "iterable" {
                result = !isObj[h]
            }
        }
        return result
    }

    func method(_ obj: JinjaValue, _ name: String,
                _ arg: JinjaValue) -> JinjaValue {
        var result = JinjaValue.undefined
        if name == "tojson", let h = handle(obj) {
            result = .str(json[h])
        }
        return result
    }
}

final class JinjaGoldenTests: XCTestCase {
    private func expect(_ tmpl: String, _ want: String,
                        _ vars: [(String, JinjaValue)] = [],
                        _ host: JinjaHost = Model(),
                        line: UInt = #line) {
        do {
            let got = try jinjaRender(tmpl, host: host, vars: vars)
            XCTAssertEqual(got, want, tmpl, line: line)
        } catch {
            XCTFail("threw \(error): \(tmpl)", line: line)
        }
    }

    func testTupleMembership() {
        expect("{{ 'low' in ('xhigh', 'medium', 'low') }}", "True")
        expect("{{ 'medium' in ('xhigh', 'medium', 'low') }}", "True")
        expect("{{ 'xhigh' in ('xhigh', 'medium', 'low') }}", "True")
        expect("{{ 'other' in ('xhigh', 'medium', 'low') }}", "False")
        expect("{{ 'low' not in ('xhigh', 'medium', 'low') }}", "False")
        expect("{{ 2 in (1, 2, 3) }}", "True")
        expect("{{ (1 + 2) * 2 }}", "6")
        expect("{{ 'a' in ('a',) }}", "True")
        expect("{{ ('a', 'b')|length }}", "2")
        expect("{{ ('a', 'b')|join('-') }}", "a-b")
    }

    func testSequenceFilters() {
        let m = Model()
        expect("{{ items|count }}", "3",
               [("items", m.val(.arr([1, 2, 3])))], m)
        expect("{{ items|reverse|join(',') }}", "3,2,1",
               [("items", m.val(.arr([1, 2, 3])))], m)
        expect("{{ nums|sort|join(',') }}", "1,2,3",
               [("nums", m.val(.arr([3, 1, 2])))], m)
        expect("{{ nums|sort(reverse=true)|join(',') }}", "3,2,1",
               [("nums", m.val(.arr([3, 1, 2])))], m)
        expect("{{ xs|unique|join(',') }}", "1,2,3",
               [("xs", m.val(.arr([1, 1, 2, 3, 3])))], m)
        expect("{{ nums|min }}/{{ nums|max }}", "1/3",
               [("nums", m.val(.arr([3, 1, 2])))], m)
        expect("{{ nums|sum }}", "6",
               [("nums", m.val(.arr([1, 2, 3])))], m)
        expect("{{ neg|abs }}", "5", [("neg", m.val(-5))], m)
    }

    func testAttributeFilters() {
        let m = Model()
        let people = m.val(.arr([
            .obj([("name", "a"), ("age", 30)]),
            .obj([("name", "b"), ("age", 20)]),
        ]))
        expect("{{ people|sort(attribute='age')" +
               "|map(attribute='name')|join(',') }}", "b,a",
               [("people", people)], m)
        let active = m.val(.arr([
            .obj([("name", "a"), ("active", true)]),
            .obj([("name", "b"), ("active", false)]),
        ]))
        expect("{{ people|selectattr('active')" +
               "|map(attribute='name')|join(',') }}", "a",
               [("people", active)], m)
        expect("{{ people|rejectattr('active')" +
               "|map(attribute='name')|join(',') }}", "b",
               [("people", active)], m)
        let named = m.val(.arr([
            .obj([("name", "a")]), .obj([("name", "b")]),
        ]))
        expect("{{ words|map('upper')|join(',') }}", "A,B",
               [("words", m.val(.arr(["a", "b"])))], m)
        expect("{{ people|map(attribute='name')|join(',') }}", "a,b",
               [("people", named)], m)
    }

    func testGroupAndDictsort() {
        let m = Model()
        let items = m.val(.arr([
            .obj([("cat", "x"), ("name", "a")]),
            .obj([("cat", "y"), ("name", "b")]),
            .obj([("cat", "x"), ("name", "c")]),
        ]))
        expect("{% for k, g in items|groupby('cat') -%}" +
               "{{ k }}:{{ g|map(attribute='name')|join('+') }};" +
               "{%- endfor %}", "x:a+c;y:b;", [("items", items)], m)
        let d = m.val(.obj([("b", 2), ("a", 1)]))
        expect("{% for k, v in d|dictsort -%}{{ k }}={{ v }};" +
               "{%- endfor %}", "a=1;b=2;", [("d", d)], m)
    }

    func testStringFilters() {
        expect("{{ s|title }}", "Hello World",
               [("s", .str("hello world"))])
        expect("{{ ''|default('x', true) }}/{{ ''|default('x') }}", "x/")
        expect("{{ 'a.b'|replace('.', '_') }}", "a_b")
        expect("{{ 'hi there'|capitalize }}", "Hi there")
        expect("{{ '  hi  '|trim }}", "hi")
    }

    func testSelectFilters() {
        let m = Model()
        expect("{{ flags|select|join(',') }}", "1,2,3",
               [("flags", m.val(.arr([0, 1, 2, 0, 3])))], m)
        expect("{{ flags|reject|join(',') }}", "0,0",
               [("flags", m.val(.arr([0, 1, 2, 0, 3])))], m)
    }

    func testTypeTests() {
        let m = Model()
        expect("{% for n in nums -%}" +
               "{{ n }}:{{ 'E' if n is even else 'O' }};{%- endfor %}",
               "1:O;2:E;3:O;", [("nums", m.val(.arr([1, 2, 3])))], m)
        expect("{{ x is number }}/{{ s is number }}", "True/False",
               [("x", .int(3)), ("s", .str("a"))])
        expect("{{ s is string }}/{{ x is string }}", "True/False",
               [("x", .int(3)), ("s", .str("a"))])
        expect("{{ d is mapping }}/{{ xs is mapping }}", "True/False",
               [("d", m.val(.obj([("a", 1)]))),
                ("xs", m.val(.arr([1])))], m)
        expect("{{ xs is sequence }}/{{ x is sequence }}", "True/False",
               [("xs", m.val(.arr([1]))), ("x", .int(3))], m)
    }

    func testSlicing() {
        let m = Model()
        let xs = m.val(.arr([1, 2, 3, 4]))
        expect("{{ xs[1:]|join(',') }}", "2,3,4", [("xs", xs)], m)
        expect("{{ xs[:2]|join(',') }}", "1,2", [("xs", xs)], m)
        expect("{{ xs[1:3]|join(',') }}", "2,3", [("xs", xs)], m)
        expect("{{ xs[::-1]|join(',') }}", "4,3,2,1", [("xs", xs)], m)
        expect("{{ xs[::2]|join(',') }}", "1,3", [("xs", xs)], m)
        expect("{{ xs[-2:]|join(',') }}", "3,4", [("xs", xs)], m)
        expect("{{ xs[:-1]|join(',') }}", "1,2,3", [("xs", xs)], m)
        expect("{{ xs[1:4:2]|join(',') }}", "2,4", [("xs", xs)], m)
        expect("{{ xs[3:0:-1]|join(',') }}", "4,3,2", [("xs", xs)], m)
    }

    func testLiteralsAndLogic() {
        let m = Model()
        expect("{{ {'a': 1, 'b': 2}|length }}", "2")
        expect("{% for k, v in {'b': 2, 'a': 1}|dictsort -%}" +
               "{{ k }}{{ v }}{%- endfor %}", "a1b2")
        expect("{{ 'yes' if cond }}", "yes", [("cond", .bool(true))])
        expect("[{{ 'yes' if cond }}]", "[]", [("cond", .bool(false))])
        expect("{{ [1, 2, 3]|join('-') }}", "1-2-3")
        expect("{{ a or 'def' }}", "def", [("a", .str(""))])
        expect("{{ a and 'x' }}", "x", [("a", .str("y"))])
        _ = m
    }

    func testRangeAndBlocks() {
        expect("{{ range(3)|join(',') }}", "0,1,2")
        expect("{{ range(2, 5)|join(',') }}", "2,3,4")
        expect("{{ range(0, 10, 3)|join(',') }}", "0,3,6,9")
        expect("{%- set x %}hi{% endset -%}[{{ x }}]", "[hi]")
        expect("{%- filter upper -%}hi there{%- endfilter -%}", "HI THERE")
    }

    func testLoopsAndMacros() {
        let m = Model()
        expect("{%- for i in xs -%}{{ i }}{%- else -%}none" +
               "{%- endfor -%}", "none", [("xs", m.val(.arr([])))], m)
        expect("{%- for i in xs -%}{{ i }}{%- else -%}none" +
               "{%- endfor -%}", "12", [("xs", m.val(.arr([1, 2])))], m)
        expect("{%- for a in outer -%}{%- for b in inner -%}{%- endfor -%}" +
               "{{ loop.index }};{%- endfor -%}", "1;2;",
               [("outer", m.val(.arr([1, 2]))),
                ("inner", m.val(.arr([9])))], m)
        expect("{%- macro f(a, b=2) -%}{{ a }}-{{ b }}{%- endmacro -%}" +
               "{{ f(1) }}|{{ f(1, b=9) }}", "1-2|1-9")
    }

    func testArithmetic() {
        expect("{{ 6 * 7 }}", "42")
        expect("{{ 7 % 3 }}", "1")
        expect("{{ -7 % 3 }}", "2")
        expect("{{ 7 // 2 }}", "3")
        expect("{{ -7 // 2 }}", "-4")
        expect("{{ 2 ** 10 }}", "1024")
        expect("{{ -2 ** 2 }}", "4")
        expect("{{ 2 + 3 * 4 }}", "14")
        expect("{{ (2 + 3) * 4 }}", "20")
    }

    func testModInLoops() {
        let m = Model()
        expect("{%- for n in nums -%}{{ n % 2 }}{%- endfor -%}", "0101",
               [("nums", m.val(.arr([0, 1, 2, 3])))], m)
        expect("{%- for i in nums -%}" +
               "{{ 'A' if i % 2 == 0 else 'B' }}{%- endfor -%}", "ABAB",
               [("nums", m.val(.arr([0, 1, 2, 3])))], m)
    }

    func testDeadPaths() {
        expect("{% if False %}{{ x|nofilter }}{% endif %}ok", "ok")
        expect("{% if False %}{{ x is notest }}{% endif %}ok", "ok")
        expect("{% if False %}{{ nofunc() }}{% endif %}ok", "ok")
        expect("{% if False %}{{ strftime_now('%Y') }}{% endif %}ok", "ok")
    }

    func testEquality() {
        expect("{{ 'hi' == false }}", "False")
        expect("{{ '' == false }}", "False")
        expect("{{ 0 == false }}/{{ 1 == true }}/{{ 2 == true }}",
               "True/True/False")
        expect("{{ '1' == 1 }}", "False")
        expect("{{ none == false }}/{{ none == none }}", "False/True")
        expect("{{ 'hi' != false }}/{{ '' != false }}", "True/True")
    }

    func testComparisonTests() {
        let m = Model()
        let roles = m.val(.arr([
            .obj([("role", "sys")]), .obj([("role", "usr")]),
            .obj([("role", "sys")]),
        ]))
        expect("{{ people|selectattr('role', 'equalto', 'sys')" +
               "|list|length }}", "2", [("people", roles)], m)
        let two = m.val(.arr([
            .obj([("role", "sys")]), .obj([("role", "usr")]),
        ]))
        expect("{{ people|rejectattr('role', 'equalto', 'sys')" +
               "|map(attribute='role')|join(',') }}", "usr",
               [("people", two)], m)
        let ages = m.val(.arr([
            .obj([("name", "a"), ("age", 30)]),
            .obj([("name", "b"), ("age", 20)]),
            .obj([("name", "c"), ("age", 40)]),
        ]))
        expect("{{ people|selectattr('age', 'greaterthan', 25)" +
               "|map(attribute='name')|join(',') }}", "a,c",
               [("people", ages)], m)
    }

    func testHasRoleIdiom() {
        let m = Model()
        let present = m.val(.arr([
            .obj([("role", "system")]), .obj([("role", "user")]),
        ]))
        expect("{{ messages|selectattr('role', 'equalto', 'system')" +
               "|list|length > 0 }}", "True",
               [("messages", present)], m)
        let absent = m.val(.arr([.obj([("role", "user")])]))
        expect("{{ messages|selectattr('role', 'equalto', 'system')" +
               "|list|length > 0 }}", "False",
               [("messages", absent)], m)
    }

    func testDefinedness() {
        expect("{{ range is defined }}", "True")
        expect("{{ namespace is defined }}", "True")
        expect("{{ nope_xyz is defined }}", "False")
        expect("{{ nope_xyz is undefined }}", "True")
        expect("{% set x = 1 %}{{ x is defined }}", "True")
    }

    // Exercises the real Qwen3.5 template with messages + a tool + the
    // empty-think prefix: assert the role markup renders, not a byte match.
    func testQwenTemplateSmoke() throws {
        let base = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/Qwen3.5-0.8B")
        let fm = FileManager.default
        var template: String? = try? String(
            contentsOf: base.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8)
        if template == nil {
            for sub in (try? fm.contentsOfDirectory(
                at: base, includingPropertiesForKeys: nil)) ?? [] {
                if let s = try? String(
                    contentsOf: sub.appendingPathComponent(
                        "chat_template.jinja"), encoding: .utf8) {
                    template = s
                }
            }
        }
        guard let tmpl = template else {
            throw XCTSkip("chat_template.jinja not present")
        }
        let m = Model()
        let messages = m.val(.arr([
            .obj([("role", "user"), ("content", "Hi")]),
        ]))
        let tools = m.val(.arr([
            .obj([("name", "get_time"),
                  ("description", "Get the current time"),
                  ("parameters", .obj([("type", "object")]))]),
        ]))
        let got = try jinjaRender(tmpl, host: m, vars: [
            ("messages", messages),
            ("tools", tools),
            ("add_generation_prompt", .bool(true)),
        ])
        XCTAssertFalse(got.isEmpty)
        XCTAssertTrue(got.contains("<|im_start|>system"), got)
        XCTAssertTrue(got.contains("<tools>"), got)
        XCTAssertTrue(got.contains("get_time"), got)
        XCTAssertTrue(got.contains("<|im_start|>user\nHi<|im_end|>"), got)
        XCTAssertTrue(got.contains("<|im_start|>assistant"), got)
        XCTAssertTrue(got.contains("<think>\n\n</think>"), got)
    }
}
