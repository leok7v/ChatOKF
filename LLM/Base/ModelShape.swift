import Foundation

public struct ModelShape: Sendable {

    public struct Tower: Sendable {
        public let name: String
        public let bytes: Int

        public init(name: String, bytes: Int) {
            self.name = name
            self.bytes = bytes
        }
    }

    public struct Eye: Sendable {
        public let weights: Int
        public let layers: Int
        public let width: Int
        public let projector: Int
        public let merge: Int
        public let k: Double

        public init(weights: Int, layers: Int, width: Int, projector: Int,
                    merge: Int, k: Double) {
            self.weights = weights
            self.layers = layers
            self.width = width
            self.projector = projector
            self.merge = merge
            self.k = k
        }

        public func flops(_ softTokens: Int) -> Double {
            let patches = Double(layers > 0 ? softTokens * merge * merge
                                            : softTokens)
            return 2 * Double(weights) * patches
                + 4 * Double(layers) * patches * patches * Double(width)
                + 2 * Double(projector) * Double(softTokens)
        }
    }

    public static let clipK = 2.4
    public static let gemmaK = 6.6
    public static let unifiedK = 150.0

    public let towers: [Tower]
    // Positions the checkpoint was TRAINED to address, not what a device can
    // allocate.
    public let trainedContext: Int
    public let embedding: Int
    public let eye: Eye?
    public let textWeights: Int

    public init(towers: [Tower], trainedContext: Int, embedding: Int,
                eye: Eye?, textWeights: Int) {
        self.towers = towers
        self.trainedContext = trainedContext
        self.embedding = embedding
        self.eye = eye
        self.textWeights = textWeights
    }

    public static func lookSeconds(_ eye: Eye, textWeights: Int,
                                   softTokens: Int, pp: Double,
                                   k: Double) -> Double {
        let rate = max(pp, 1) * 2 * Double(max(textWeights, 1))
        return eye.flops(softTokens) / rate * k
    }

    public static func pictureSeconds(_ eye: Eye, textWeights: Int,
                                      softTokens: Int, pp: Double,
                                      k: Double) -> Double {
        lookSeconds(eye, textWeights: textWeights, softTokens: softTokens,
                    pp: pp, k: k)
            + Double(softTokens) / max(pp, 1)
    }

    static func grouped(_ text: Int, _ vision: Int,
                        _ audio: Int) -> [Tower] {
        var out = [Tower(name: "text", bytes: text)]
        if vision > 0 { out.append(Tower(name: "vision", bytes: vision)) }
        if audio > 0 { out.append(Tower(name: "audio", bytes: audio)) }
        return out
    }
}

extension ModelShape {

    init(gguf g: GGUF, sidecars: [Tower] = [], eyeFile: GGUF? = nil) {
        let arch = g.string("general.architecture") ?? ""
        var text = 0
        var vision = 0
        var audio = 0
        for (name, t) in g.tensors {
            switch ModelShape.tower(of: name) {
            case "audio": audio += t.byteCount
            case "vision": vision += t.byteCount
            default: text += t.byteCount
            }
        }
        self.init(
            towers: ModelShape.grouped(text, vision, audio) + sidecars,
            trainedContext: ModelShape.trainedContext(g),
            embedding: g.int(arch + ".embedding_length") ?? 0,
            eye: ModelShape.eye(eyeFile ?? g),
            textWeights: ModelShape.textWeights(g))
    }

    static func tower(of tensor: String) -> String {
        var out = "text"
        if tensor.hasPrefix("a.") || tensor.hasPrefix("mm.audio") {
            out = "audio"
        } else if tensor.hasPrefix("v.") || tensor.hasPrefix("mm.") {
            out = "vision"
        }
        return out
    }

    static func textWeights(_ g: GGUF) -> Int {
        g.tensors.values.filter { t in
            ModelShape.tower(of: t.name) == "text" && t.dims.count >= 2
                && !t.name.contains("token_embd") && t.name != "output.weight"
                && !t.name.hasPrefix("assist.") && !t.name.contains(".nextn.")
        }.reduce(0) { sum, t in sum + t.count }
    }

    static func eye(_ g: GGUF) -> Eye? {
        var weights = 0
        var projector = 0
        var layers = Set<String>()
        for t in g.tensors.values where t.dims.count >= 2 {
            if t.name.hasPrefix("v.blk.") {
                weights += t.count
                layers.insert(t.name.split(separator: ".")[2].description)
            } else if t.name == "v.patch_embd.weight" {
                weights += t.count
            } else if t.name.hasPrefix("mm."), !t.name.hasPrefix("mm.audio") {
                projector += t.count
            }
        }
        let width = g.maybe("v.blk.0.attn_out.weight")?.dims[0] ?? 0
        return weights + projector > 0
            ? Eye(weights: weights, layers: layers.count, width: width,
                  projector: projector, merge: ModelShape.merge(g, width),
                  k: ModelShape.k(g, layers: layers.count))
            : nil
    }

    static func k(_ g: GGUF, layers: Int) -> Double {
        var out = ModelShape.gemmaK
        if layers == 0 {
            out = ModelShape.unifiedK
        } else if g.int("clip.vision.block_count") != nil {
            out = ModelShape.clipK
        }
        return out
    }

    static func merge(_ g: GGUF, _ width: Int) -> Int {
        let into = g.maybe("mm.0.weight")?.dims[0] ?? 0
        let side = width > 0 && into > width
            ? Int(Double(into / width).squareRoot().rounded()) : 1
        return g.int("clip.vision.spatial_merge_size")
            ?? g.int("gemma4.vision.pool_kernel") ?? side
    }

    static func trainedContext(_ g: GGUF) -> Int {
        let arch = g.string("general.architecture") ?? ""
        return g.int(arch + ".context_length") ?? ModelShape.sourceContext(g)
    }

    private static func sourceContext(_ g: GGUF) -> Int {
        var out = 0
        if let raw = g.string("gemma4.source.config_json"),
           let data = raw.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data))
               as? [String: Any],
           let text = root["text_config"] as? [String: Any],
           let n = (text["max_position_embeddings"] as? NSNumber)?.intValue {
            out = n
        }
        return out
    }
}
