import Foundation
import LLM

@MainActor func runJudge(_ loaded: LoadedChat, _ path: String) async throws {
    let judge = Judge(backend: loaded.backend, template: loaded.template,
                      vocabSize: loaded.vocabCount)
    try await judge.run(file: URL(fileURLWithPath: path)) { line in
        print(line)
    }
    exit(0)
}
