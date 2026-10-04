import Foundation
import LLM

public struct ConversationNote: Sendable {

    public let conversation: UUID
    public let title: String
    public let asked: [String]
    public let exchange: String
    public let concludes: Bool

    public init(conversation: UUID, title: String, asked: [String],
                exchange: String, concludes: Bool) {
        self.conversation = conversation
        self.title = title
        self.asked = asked
        self.exchange = exchange
        self.concludes = concludes
    }

    public static let area = "chat"
    public static let type = "Conversation"
    static let tag = "conversation"
    static let promptLimit = 200
    static let promptWords = 3
    static let promptCount = 12
    static let lineLimit = 240

    public static func id(_ conversation: UUID) -> String {
        area + "/" + String(conversation.uuidString.lowercased().prefix(8))
    }

    public static func asked(_ prompts: [String],
                             samples: Set<String>) -> [String] {
        let kept = prompts.map { prompt in
            prompt.split(whereSeparator: { c in c.isWhitespace })
                .joined(separator: " ")
        }.filter { prompt in
            prompt.count <= promptLimit && !samples.contains(prompt)
                && prompt.split(separator: " ").count >= promptWords
        }
        return Array(kept.suffix(promptCount))
    }

    var instruction: String {
        "In one plain sentence, state the answer your last reply gave to "
            + "the user's last message. Start with the fact itself, no "
            + "preamble, no question. Reply with the sentence only."
    }

    static let aboutItself = ["i ", "i'", "we ", "as an ai"]
    static let lineWords = 3

    static func numbers(_ text: String) -> [String] {
        text.split(whereSeparator: { c in !c.isNumber })
            .map(String.init)
    }

    static func names(_ line: String) -> [String] {
        let words = line.split(whereSeparator: { c in
            !c.isLetter && c != "-"
        }).map(String.init)
        return words.dropFirst().filter { word in
            word.first?.isUppercase == true && word.count > 1
        }
    }

    static func echoes(_ line: String, _ asked: [String]) -> Bool {
        let said = line.lowercased()
        return asked.contains { prompt in
            let lower = prompt.lowercased()
            return lower.contains(said) || said.contains(lower)
        }
    }

    public static func concluded(_ raw: String, in exchange: String,
                                 asked: [String] = []) -> String {
        let line = raw.split(whereSeparator: \.isNewline)
            .map { part in
                part.trimmingCharacters(
                    in: CharacterSet(charactersIn: " \t`\"*#"))
            }
            .first { part in !part.isEmpty } ?? ""
        let known = Set(numbers(exchange))
        let invented = numbers(line).contains { n in !known.contains(n) }
            || names(line).contains { name in !exchange.contains(name) }
        let lower = line.lowercased()
        let narrated = aboutItself.contains { opening in
            lower.hasPrefix(opening)
        }
        let usable = line.split(separator: " ").count >= lineWords
            && line.count <= lineLimit
            && !line.hasSuffix("?") && !invented && !narrated
            && !echoes(line, asked)
        return usable ? line : ""
    }

    static let describedLimit = 220
    static let concludedCount = 6
    static let concludedHead = "Concluded:"

    static func latest(_ lines: [String]) -> String {
        var kept: [String] = []
        var size = 0
        for line in lines.reversed()
        where size + line.count <= ConversationNote.describedLimit {
            kept.insert(line, at: 0)
            size += line.count
        }
        return kept.joined(separator: " ")
    }

    func description(_ reached: [String]) -> String {
        "Asked: " + ConversationNote.latest(asked) + (reached.isEmpty
            ? "" : " Concluded: " + ConversationNote.latest(reached))
    }

    func reached(_ concluded: String, after earlier: String) -> [String] {
        var out = ConversationNote.conclusions(in: earlier)
            .filter { line in line != concluded }
        if !concluded.isEmpty { out.append(concluded) }
        return Array(out.suffix(ConversationNote.concludedCount))
    }

    static func conclusions(in body: String) -> [String] {
        let lines = body.components(separatedBy: "\n")
        let from = lines.firstIndex(of: concludedHead)
        return from.map { at in
            lines[(at + 1)...].filter { line in line.hasPrefix("- ") }
                .map { line in String(line.dropFirst(2)) }
        } ?? []
    }

    func body(_ reached: [String]) -> String {
        var out = "User asked:\n"
            + asked.map { prompt in "- " + prompt }.joined(separator: "\n")
        if !reached.isEmpty {
            out += "\n\n" + ConversationNote.concludedHead + "\n"
                + reached.map { line in "- " + line }.joined(separator: "\n")
        }
        return out
    }
}
