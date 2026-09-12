import Foundation

public final class Gemma4Backend: EngineBackend<Gemma4Engine, GemmaTokenizer>,
    @unchecked Sendable {

    public override func supportsSoftTokens() async -> Bool { true }

    public override func extendSoft(_ ids: [Int32],
                                    spans: [SoftSpan]) async throws -> Int32 {
        let feed = SoftFeed(spans)
        let out = engine.extend(ids, softAt: { id in feed.row(id) })
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }
}
