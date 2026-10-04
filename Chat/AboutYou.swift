import Foundation
import LLM
import Observation

@MainActor @Observable public final class AboutYou {

    public static let shared = AboutYou()

    static let nameKey = "you.name"
    static let genderKey = "you.gender"
    static let ageKey = "you.age"
    nonisolated static let fieldLimit = 40
    nonisolated static let kept =
        " Remember this for the whole conversation."
    nonisolated static let byName = " Use the name where a person "
        + "naturally would: in a greeting, in a goodbye, or to make a point "
        + "personal. Most replies need no name."

    public var name: String {
        didSet { UserDefaults.standard.set(name, forKey: AboutYou.nameKey) }
    }

    public var gender: String {
        didSet {
            UserDefaults.standard.set(gender, forKey: AboutYou.genderKey)
        }
    }

    public var age: String {
        didSet { UserDefaults.standard.set(age, forKey: AboutYou.ageKey) }
    }

    public init() {
        let d = UserDefaults.standard
        name = d.string(forKey: AboutYou.nameKey) ?? ""
        gender = d.string(forKey: AboutYou.genderKey) ?? ""
        age = d.string(forKey: AboutYou.ageKey) ?? ""
    }

    public var line: String {
        let asked = Flags.value("about-you") ?? ""
        return asked.isEmpty
            ? AboutYou.line(name: name, gender: gender, age: age) : asked
    }

    nonisolated public static func line(name: String, gender: String,
                                        age: String) -> String {
        let said = [name, gender, age].map { text in
            String(text.split(whereSeparator: { c in c.isWhitespace })
                .joined(separator: " ").prefix(AboutYou.fieldLimit))
        }
        var facts: [String] = []
        if !said[0].isEmpty { facts.append("name " + said[0]) }
        if !said[1].isEmpty { facts.append("gender " + said[1]) }
        if !said[2].isEmpty { facts.append("age " + said[2]) }
        let asks = said[0].isEmpty
            ? "" : AboutYou.byName
        return facts.isEmpty ? "" : "[About the user you are talking to: "
            + facts.joined(separator: ", ") + "." + AboutYou.kept + asks
            + "]"
    }
}
