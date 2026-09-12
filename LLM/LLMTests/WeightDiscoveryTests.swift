import XCTest
@testable import LLM

final class WeightDiscoveryTests: XCTestCase {

    func testEveryResolvedPathIsTheCatalogFile() {
        for name in ModelCatalog.ggufFiles.keys {
            if let path = TestWeights.find(name) {
                XCTAssertEqual(URL(fileURLWithPath: path).lastPathComponent,
                               ModelCatalog.ggufFiles[name])
                XCTAssertTrue(FileManager.default.isReadableFile(
                    atPath: path), path)
            }
        }
    }

    func testGemmaGateResolvesWhenItsCloneIsPresent() {
        let src = ModelCatalog.source("gemma-4-E2B")!
        let file = ModelCatalog.ggufFiles["gemma-4-E2B"]!
        let cloned = TestWeights.clones.contains { root in
            FileManager.default.isReadableFile(
                atPath: root + "/" + src.repo + "/" + file)
        }
        if cloned {
            XCTAssertNotNil(gemmaGgufPath,
                            "the clone is present but discovery missed it")
        }
    }
}
