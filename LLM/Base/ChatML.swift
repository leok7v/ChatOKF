// Not on the app path: the tokenizer gate encodes it and diffs the ids
// against a Python-captured reference, so it must stay byte-stable.
enum ChatML {
    static func prompt(system: String?, user: String) -> String {
        var s = ""
        if let system {
            s += "<|im_start|>system\n\(system)<|im_end|>\n"
        }
        s += "<|im_start|>user\n\(user)<|im_end|>\n"
        s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return s
    }
}
