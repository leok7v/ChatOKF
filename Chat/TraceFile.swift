import Foundation
import LLM
import os

public final class TraceFile {

    private let log = Logger(subsystem: "io.github.leok7v.ChatOKF",
                             category: "trace")
    private let url: URL
    private let fmt: DateFormatter
    private var handle: FileHandle?

    public init() {
        url = Diag.startRun(folder: "transcript.log")
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss.SSS"
        fmt = f
        log.info("transcript \(self.url.path, privacy: .public)")
    }

    public var path: String { url.path }

    public func note(_ text: String) {
        write(text + "\n")
    }

    public func append(_ e: TraceEvent) {
        write(line(e) + "\n")
    }

    private func line(_ e: TraceEvent) -> String {
        let dur = e.t1.timeIntervalSince(e.t0)
        var head = fmt.string(from: e.t1) + " [" + e.kind.rawValue + "]"
        if e.tokens > 0 { head += " \(e.tokens) tok" }
        if e.ctx >= 0 { head += " ctx \(e.ctx)" }
        if dur >= 0.001 { head += String(format: " %.2fs", dur) }
        head += " | " + e.summary
        if !e.text.isEmpty {
            let body = e.text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { part in "  " + part }
                .joined(separator: "\n")
            head += "\n" + body
        }
        return head
    }

    private func write(_ s: String) {
        let fm = FileManager.default
        if handle == nil {
            try? fm.createDirectory(at: url.deletingLastPathComponent(),
                                    withIntermediateDirectories: true)
            fm.createFile(atPath: url.path, contents: Data())
            handle = try? FileHandle(forWritingTo: url)
        }
        if let handle {
            try? handle.write(contentsOf: Data(s.utf8))
        }
    }

}
