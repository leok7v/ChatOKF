import Foundation

struct Credit: Identifiable {
    let name: String
    let author: String
    let terms: String
    let source: String
    let changed: String
    var id: String { name }
    var link: URL? { URL(string: source) }
}

enum Credits {

    // Apache-2.0 5(b) requires `changed` to state the modification on every
    // model below.

    static let app = Credit(
        name: "ChatOKF",
        author: "leok7v",
        terms: "GPL-3.0",
        source: "https://leok7v.github.io/ChatOKF/",
        changed: "Runs every model below on this device. Nothing you type "
            + "leaves it except where a tool you switched on says otherwise.")

    static let all: [Credit] = [
        Credit(name: "Gemma 4",
               author: "Google DeepMind",
               terms: "Apache-2.0",
               source: "https://huggingface.co/google/gemma-4-12B-it",
               changed: "Repacked into a single GGUF. The quantization-aware "
                   + "training left 4-bit codes behind in half precision, and "
                   + "those codes are recovered as they were trained rather "
                   + "than quantized again."),
        Credit(name: "Qwen3.5",
               author: "Qwen, Alibaba",
               terms: "Apache-2.0",
               source: "https://huggingface.co/Qwen/Qwen3.5-4B",
               changed: "Runs from a community GGUF quantization, with our "
                   + "vision tower and sampling card grafted into the one "
                   + "file."),
        Credit(name: "Qwen3.8",
               author: "Qwen, Alibaba",
               terms: "Apache-2.0",
               source: "https://huggingface.co/Qwen/Qwen3.8-27B",
               changed: "Runs from a community GGUF quantization, with our "
                   + "vision tower and sampling card grafted into the one "
                   + "file, and the community's own speculative-decode "
                   + "layer folded in where its build shipped without one."),
        Credit(name: "Ternary Bonsai",
               author: "PrismML",
               terms: "Apache-2.0",
               source: "https://huggingface.co/prism-ml/"
                   + "Ternary-Bonsai-27B-gguf",
               changed: "Re-hosted as one file: the vision projector and "
                   + "the sampling card ride in the weight file's header, "
                   + "and the tensor data is PrismML's, unchanged."),
        Credit(name: "KittenTTS",
               author: "KittenML",
               terms: "Apache-2.0",
               source: "https://huggingface.co/KittenML/kitten-tts-nano-0.2",
               changed: "The voice you hear. Reimplemented in Swift by way of "
                   + "leok7v/tts.cli, so it synthesises on device with no "
                   + "network and no system speech service."),
        Credit(name: "all-MiniLM-L6-v2",
               author: "Sentence-Transformers",
               terms: "Apache-2.0",
               source: "https://huggingface.co/sentence-transformers/"
                   + "all-MiniLM-L6-v2",
               changed: "Quantized to GGUF. It matches your question to a "
                   + "Wikipedia article on this device, which is what keeps "
                   + "the question itself off the network."),
        Credit(name: "multilingual-e5-small",
               author: "intfloat, Microsoft",
               terms: "MIT",
               source: "https://huggingface.co/intfloat/multilingual-e5-small",
               changed: "Quantized to 8-bit GGUF by cstr. It turns your "
                   + "questions and the assistant's memories into vectors on "
                   + "this device, so the memory that fits a question can be "
                   + "found without the question leaving."),
        Credit(name: "Open Knowledge Format",
               author: "Google Cloud",
               terms: "Apache-2.0",
               source: "https://github.com/GoogleCloudPlatform/"
                   + "open-knowledge-format/blob/main/SPEC.md",
               changed: "The shape of the assistant's memories: one note "
                   + "per file, a short header and plain links, as the "
                   + "specification reads them. The search over them, the "
                   + "drafts and the Keep step are this app's own."),
        Credit(name: "Parallel",
               author: "Parallel Web Systems",
               terms: "Commercial service",
               source: "https://parallel.ai/privacy-policy",
               changed: "Answers web searches while it is switched on in "
                   + "Settings. What you searched for and your network "
                   + "address reach it; nothing else from the conversation "
                   + "does."),
        Credit(name: "Mwmbl",
               author: "Mwmbl, a non-profit",
               terms: "Open source, independent index",
               source: "https://mwmbl.org/privacy",
               changed: "Answers web searches when Parallel is switched off "
                   + "or comes back empty. It stores no network address and "
                   + "keeps no search that points back to you."),
        Credit(name: "Wikipedia",
               author: "Wikimedia Foundation and contributors",
               terms: "CC BY-SA 4.0",
               source: "https://simple.wikipedia.org",
               changed: "Article titles are indexed on device; article text "
                   + "is fetched only for the article the model chose."),
    ]

}
