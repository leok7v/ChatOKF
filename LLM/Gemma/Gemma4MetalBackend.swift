import Foundation

public final class Gemma4MetalBackend:
    EngineBackend<Gemma4MetalEngine, GemmaTokenizer>, @unchecked Sendable {

    public var ctx: MetalContext { engine.ctx }

    public var draftWidth: Int { engine.specN }

    public override func supportsSoftTokens() async -> Bool { ctx.matrixUnits }

    public override func extendSoft(_ ids: [Int32],
                                    spans: [SoftSpan]) async throws -> Int32 {
        let feed = SoftFeed(spans)
        let out = engine.extend(ids, softAt: { id in feed.row(id) })
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }
}
