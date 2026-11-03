import Foundation
import LLM

@MainActor final class MemoryDrill {
    let memories: Memories
    let runner: MemoryToolRunner
    private var recalled: Set<String> = []
    static let pp = 100.0

    init(root: String, network: Bool) {
        memories = Memories(root: URL(fileURLWithPath: root,
                                      isDirectory: true))
        memories.enabled = true
        runner = MemoryToolRunner(inner: SafeToolRunner(network: network),
                                  memories: memories,
                                  writes: Flags.on("note-tools"))
    }

    func open(importing dir: String?) async {
        await memories.awaitOpen()
        if !memories.isOpen {
            err("[memories] FAILED to open \(memories.root.path): no "
                + "encoder; pass --encoder-folder\n")
            exit(2)
        }
        if let dir {
            let count = await memories.importMarkdown(from: dir)
            err("[memories] imported \(count) file(s) from \(dir)\n")
        }
        err("[memories] active \(memories.active) open \(memories.isOpen), "
            + "\(memories.count) note(s) at \(memories.root.path)\n")
    }

    func before(_ user: String, pp: Double) async -> String {
        var out = user
        let rate = pp > 0 ? pp : MemoryDrill.pp
        let found = await memories.recall(user, also: [], pp: rate,
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
        if !Memories.speaksOfSelf(said) {
            err("[extract] nothing about the user, skipped\n")
        } else if coverage.covered {
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

    func caption(_ session: ChatSession, picture: String,
                 answer: String) async {
        let began = Date()
        let caption = NoteFind.cut(
            await session.conclude(Memories.captionInstruction), limit: 400)
        let data = (try? Data(contentsOf: URL(fileURLWithPath: picture)))
            ?? Data()
        await memories.keepPicture(names: [picture],
                                   stamp: NoteFind.stamp(data),
                                   caption: caption, answer: answer)
        err(String(format: "[caption] %.1fs: %@\n",
                   Date().timeIntervalSince(began), caption))
    }
}
