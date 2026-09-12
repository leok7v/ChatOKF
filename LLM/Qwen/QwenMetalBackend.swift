import Foundation

public final class QwenMetalBackend: EngineBackend<QwenMetalEngine, Tokenizer>,
    @unchecked Sendable {
    private let mmprojPath: String?
    // Built on the first image turn and kept RESIDENT: a per-turn rebuild costs
    // ~2s and the f16 weights fit beside the LM on the 16GB hosts.
    private var vit: QwenMetalViT?

    public init(engine: QwenMetalEngine, tokenizer: Tokenizer,
                mmprojPath: String? = nil) {
        self.mmprojPath = mmprojPath
        super.init(engine: engine, tokenizer: tokenizer)
    }

    public override func supportsSoftTokens() async -> Bool {
        await supportsVision()
    }

    public override func extendSoft(_ ids: [Int32],
                                    spans: [SoftSpan]) async throws -> Int32 {
        let merge = try tower().cfg.merge
        var placed: [(start: Int, gh: Int, gw: Int)] = []
        var feats: [[Float]] = []
        var i = 0
        for span in spans {
            let side = Int(Double(span.rows).squareRoot().rounded()) * merge
            let grid = span.grid ?? (h: side, w: side)
            let per = (grid.h / merge) * (grid.w / merge)
            let width = span.features.count / span.rows
            var done = 0
            while done < span.rows {
                while i < ids.count && ids[i] != span.placeholder { i += 1 }
                placed.append((start: i, gh: grid.h, gw: grid.w))
                feats.append(Array(span.features[
                    (done * width) ..< ((done + per) * width)]))
                i += per
                done += per
            }
        }
        let out = engine.extendVision(ids, feats: feats, spans: placed)
        if engine.shouldStop() { throw EngineError.stopped }
        return out
    }

    public func media() -> QwenMedia? {
        let ids = tokenizer.encode("<|image_pad|>", addSpecial: true)
        var out: QwenMedia? = nil
        if mmprojPath != nil, engine.ctx.matrixUnits, ids.count == 1 {
            out = QwenMedia(backend: self, tokenizer: tokenizer, pad: ids[0])
        }
        return out
    }

    func tower() throws -> QwenMetalViT {
        if vit == nil, let mmprojPath, engine.ctx.matrixUnits {
            vit = try QwenMetalViT(path: mmprojPath)
        }
        if vit == nil { throw EngineError.missingModel("mmproj") }
        return vit!
    }

    func encode(pixels: [Float], gridH: Int, gridW: Int) throws -> [Float] {
        try tower().forward(pixels: pixels, gridH: gridH, gridW: gridW)
    }

    func encode(pair a: [Float], _ b: [Float], gridH: Int,
                gridW: Int) throws -> [Float] {
        try tower().forward(pair: a, b, gridH: gridH, gridW: gridW)
    }

    // The tower's GEMM and the vision prefill are simdgroup-matrix kernels,
    // so without matrix units there is no vision path and no attach UI.
    public override func supportsVision() async -> Bool {
        mmprojPath != nil && engine.ctx.matrixUnits
    }
}

public struct QwenMetalChat {
    public let engine: QwenMetalEngine
    public let tokenizer: Tokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets
    public let mmprojPath: String?
    public let shape: ModelShape

    public init(ggufPath: String, pageP: Int = 512) throws {
        let m = try QwenModel(path: ggufPath)
        engine = try QwenMetalEngine(m, pageP: pageP)
        var tok = try Tokenizer(gguf: m.gguf)
        tok.addStops(Tokenizer.stopIds(besideSet: URL(
            fileURLWithPath: ggufPath).deletingLastPathComponent()))
        tokenizer = tok
        chatTemplate = m.gguf.string("tokenizer.chat_template")
            ?? QwenChat.fallbackTemplate
        samplingPresets = SamplingPresets.require(gguf: m.gguf,
                                                  path: ggufPath)
        let inside = m.gguf.int("clip.vision.block_count") != nil
        let mmproj = inside ? ggufPath : QwenMetalChat.mmprojBeside(ggufPath)
        mmprojPath = mmproj
        shape = ModelShape(gguf: m.gguf,
                           sidecars: inside ? [] : QwenMetalChat.eye(mmproj))
    }

    // Everything in an mmproj serves the eye, so the file size IS the tower's.

    static func eye(_ mmproj: String?) -> [ModelShape.Tower] {
        var out: [ModelShape.Tower] = []
        if let mmproj,
           let size = (try? FileManager.default
               .attributesOfItem(atPath: mmproj))?[.size] as? Int {
            out.append(ModelShape.Tower(name: "vision", bytes: size))
        }
        return out
    }

    static func mmprojBeside(_ path: String) -> String? {
        let dir = (path as NSString).deletingLastPathComponent
        let names = (try? FileManager.default
            .contentsOfDirectory(atPath: dir)) ?? []
        return names
            .first { n in n.contains("mmproj")
                && (n.hasSuffix(".ggxf") || n.hasSuffix(".gguf")) }
            .map { n in dir + "/" + n }
    }

    public var mtpDrafts: Int {
        var out = 0
        if let n = Flags.int("mtp-drafts") {
            out = n
        } else {
            let gb = Double(engine.model.gguf.mapSize) / 1_073_741_824
            let trunk = QwenMetalChat.dominantType(engine.model.gguf)
            let narrow = MetalEnc.narrowIQName(trunk) != nil || trunk == .q4_0
            let wide = MetalEnc.fusedTypes.contains(trunk) && gb >= 5
            out = gb < 3 || !narrow ? 0 : (wide ? 2 : 1)
        }
        return out
    }

    static func dominantType(_ g: GGUF) -> GGUFType {
        var bytes: [GGUFType: Int] = [:]
        for (name, t) in g.tensors
        where !name.hasPrefix("v.") && !name.hasPrefix("mm.") {
            bytes[t.type, default: 0] += t.byteCount
        }
        let top = bytes.max { a, b in a.value < b.value }
        return top?.key ?? .f32
    }

    public func backend() -> QwenMetalBackend {
        QwenMetalBackend(engine: engine, tokenizer: tokenizer,
                     mmprojPath: mmprojPath)
    }
}
