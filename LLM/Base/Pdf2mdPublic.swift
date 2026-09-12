import Foundation

// The ONLY file that may adapt the vendored Pdf2md.swift, which must stay
// byte-identical so a re-import is a copy. See LLM/fixtures/pdf2md/ORIGIN.md.

public enum Pdf2md {
    // `.geometry` reads the page's OWN text layer and falls back to recognizing
    // the page by itself when the layer explains too little of the ink.

    public static func markdown(of url: URL,
                                onPage: (@Sendable (Int) -> Void)? = nil)
        async throws -> String {
        var converter = Converter()
        converter.mode = .geometry
        if let onPage {
            converter.onAudit = { page, _ in onPage(page) }
        }
        return try await converter.markdown(of: url)
    }
}
