import Foundation
import Testing

let qwenGgufPath = TestWeights.find("Qwen3.5-4B")

let needsQwenWeights = ConditionTrait.enabled(
    if: qwenGgufPath != nil,
    Comment(rawValue: TestWeights.missing("Qwen3.5-4B")))
