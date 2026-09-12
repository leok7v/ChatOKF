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

    public let towers: [Tower]
    // Positions the checkpoint was TRAINED to address, not what a device can
    // allocate.
    public let trainedContext: Int
    public let embedding: Int

    public init(towers: [Tower], trainedContext: Int, embedding: Int) {
        self.towers = towers
        self.trainedContext = trainedContext
        self.embedding = embedding
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

    init(gguf g: GGUF, sidecars: [Tower] = []) {
        let arch = g.string("general.architecture") ?? ""
        var text = 0
        var vision = 0
        var audio = 0
        for (name, t) in g.tensors {
            if name.hasPrefix("a.") || name.hasPrefix("mm.audio") {
                audio += t.byteCount
            } else if name.hasPrefix("v.") || name.hasPrefix("mm.") {
                vision += t.byteCount
            } else {
                text += t.byteCount
            }
        }
        self.init(
            towers: ModelShape.grouped(text, vision, audio) + sidecars,
            trainedContext: g.int(arch + ".context_length")
                ?? ModelShape.sourceContext(g),
            embedding: g.int(arch + ".embedding_length") ?? 0)
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
