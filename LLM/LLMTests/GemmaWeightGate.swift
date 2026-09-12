import Foundation
import Testing

let gemmaGgufPath = TestWeights.find("gemma-4-E2B")

let needsGemmaWeights = ConditionTrait.enabled(
    if: gemmaGgufPath != nil,
    Comment(rawValue: TestWeights.missing("gemma-4-E2B")))

let gemmaMTPGgufPath = TestWeights.find("gemma-4-E2B-MTP")

let needsGemmaMTPWeights = ConditionTrait.enabled(
    if: gemmaMTPGgufPath != nil,
    Comment(rawValue: TestWeights.missing("gemma-4-E2B-MTP")))
