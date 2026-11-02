import Foundation
import LLM

@MainActor final class RecallDrill {
    let memories: Memories
    var recalled: Set<String> = []
    static let pp = 100.0

    init(root: String) {
        memories = Memories(root: URL(fileURLWithPath: root,
                                      isDirectory: true))
    }

    func open() async {
        await memories.awaitOpen()
        err("[memories] active \(memories.active) open \(memories.isOpen), "
            + "\(memories.count) note(s) at \(memories.root.path)\n")
    }

    func before(_ user: String) async -> String {
        var out = user
        let found = await memories.recall(user, also: [], pp: RecallDrill.pp,
                                          excluding: recalled)
        if let found {
            recalled.formUnion(found.ids)
            err("[recall] \(found.ids.joined(separator: " ")) ~"
                + "\(found.tokens) tok, standout "
                + String(format: "%.1f", found.standout) + "\n")
            out = found.block + user
        } else {
            err("[recall] nothing\n")
        }
        return out
    }

    func after(_ session: ChatSession, said: String, answer: String) async {
        let exchange = "user: " + said + "\nassistant: " + answer
        let coverage = await memories.coverage(exchange)
        if coverage.covered {
            err("[extract] covered by " + coverage.known.map { note in
                note.id }.joined(separator: " ") + ", skipped\n")
        } else {
            let raw = await session.extractNotes(
                Memories.extractionInstruction(known: coverage.known,
                                               said: said))
            let drafts = Memories.parseDrafts(raw)
            let kept = await memories.remember(drafts, said: said,
                                               source: nil,
                                               excluding: recalled)
            let ids = kept.map { note in note.id }.joined(separator: " ")
            let head = String(raw.prefix(240))
                .replacingOccurrences(of: "\n", with: " | ")
            err("[extract] \(drafts.count) draft(s), kept \(ids); raw: "
                + head + "\n")
        }
    }
}
