import Foundation

public typealias QwenBackend = EngineBackend<QwenEngine, Tokenizer>

public struct QwenChat {
    public let engine: QwenEngine
    public let tokenizer: Tokenizer
    public let chatTemplate: String
    public let samplingPresets: SamplingPresets

    // Minimal ChatML fallback when a GGUF ships no chat_template (rare).
    static let fallbackTemplate = """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """

    public init(ggufPath: String) throws {
        let m = try QwenModel(path: ggufPath)
        engine = QwenEngine(m)
        var tok = try Tokenizer(gguf: m.gguf)
        tok.addStops(Tokenizer.stopIds(besideSet: URL(
            fileURLWithPath: ggufPath).deletingLastPathComponent()))
        tokenizer = tok
        chatTemplate = m.gguf.string("tokenizer.chat_template")
            ?? QwenChat.fallbackTemplate
        samplingPresets = SamplingPresets.require(gguf: m.gguf,
                                                  path: ggufPath)
    }

    public func backend() -> QwenBackend {
        QwenBackend(engine: engine, tokenizer: tokenizer)
    }
}
