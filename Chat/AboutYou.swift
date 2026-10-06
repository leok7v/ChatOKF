import Foundation
import LLM
import Observation

@MainActor @Observable public final class AboutYou {

    public static let shared = AboutYou()

    static let nameKey = "you.name"
    nonisolated static let nameLimit = 40
    nonisolated static let byName = " Remember this for the whole "
        + "conversation. Use the name where a person naturally would: in a "
        + "greeting, in a goodbye, or to make a point personal. Most "
        + "replies need no name.]"

    public var name: String {
        didSet { UserDefaults.standard.set(name, forKey: AboutYou.nameKey) }
    }

    public init() {
        name = UserDefaults.standard.string(forKey: AboutYou.nameKey) ?? ""
    }

    public var line: String {
        let asked = Flags.value("about-you") ?? ""
        return asked.isEmpty ? AboutYou.line(name: name) : asked
    }

    nonisolated public static func called(_ name: String) -> String {
        String(name.split(whereSeparator: { c in c.isWhitespace })
            .joined(separator: " ").prefix(AboutYou.nameLimit))
    }

    nonisolated public static func line(name: String) -> String {
        let said = AboutYou.called(name)
        return said.isEmpty ? "" : "[You are talking to " + said
            + ". In your private reasoning, refer to this person as " + said
            + ". In your replies, speak to " + said + " directly as you."
            + AboutYou.byName
    }
}
