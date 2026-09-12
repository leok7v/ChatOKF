import Foundation

// The ONLY file that may adapt the vendored Docs2md.swift, which must stay
// byte-identical so a re-import is a copy. See LLM/fixtures/pdf2md/ORIGIN.md.

public enum Docs2md {
    // SYNCHRONOUS and real work: a main-actor caller hands it to a detached
    // task. No progress: these formats state their structure, nothing to count.

    public static func markdown(of url: URL) throws -> String {
        try DocsConverter().markdown(of: url)
    }

    public static let readable: [String] =
        Format.allCases.map { format in format.rawValue } + ["htm", "xhtml"]
}
