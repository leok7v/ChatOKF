import Foundation
import Testing
@testable import XET

let xorbA = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
let xorbB = "fedcba0987654321098765432109876543210fedcba098765432109876543"

let specV2 = """
{
  "offset_into_first_range": 0,
  "terms": [
    { "hash": "\(xorbA)", "unpacked_length": 263873,
      "range": { "start": 1, "end": 4 } },
    { "hash": "\(xorbB)", "unpacked_length": 143890,
      "range": { "start": 0, "end": 3 } },
    { "hash": "\(xorbA)", "unpacked_length": 3063572,
      "range": { "start": 3, "end": 43 } }
  ],
  "xorbs": {
    "\(xorbA)": [ { "url": "https://transfer.invalid/xorb/default/a",
      "ranges": [ { "chunks": { "start": 1, "end": 43 },
                    "bytes": { "start": 57980, "end": 1433008 } } ] } ],
    "\(xorbB)": [ { "url": "https://transfer.invalid/xorb/default/b",
      "ranges": [ { "chunks": { "start": 0, "end": 3 },
                    "bytes": { "start": 0, "end": 65670 } } ] } ]
  }
}
"""

let specV1 = """
{
  "offset_into_first_range": 100000,
  "terms": [ { "hash": "\(xorbA)", "unpacked_length": 237171,
               "range": { "start": 0, "end": 2 } } ],
  "fetch_info": { "\(xorbA)": [ { "url": "https://transfer.invalid/a",
    "range": { "start": 0, "end": 2 },
    "url_range": { "start": 0, "end": 28958 } } ] }
}
"""

@Suite struct ReconstructionTests {

    @Test func theV2ShapeParsesIntoTermsAndFetches() throws {
        let r = try Reconstruction(json: Data(specV2.utf8))
        #expect(r.skip == 0)
        #expect(r.terms.count == 3)
        #expect(r.terms[2] == Term(hash: xorbA, unpacked: 3063572,
                                   chunks: 3..<43))
        #expect(r.fetches[xorbA]?.count == 1)
        #expect(r.fetches[xorbA]?[0].bytes == 57980...1433008)
        #expect(r.fetches[xorbA]?[0].range == "bytes=57980-1433008")
        #expect(r.unpacked == 263873 + 143890 + 3063572)
    }

    @Test func theV1ShapeParsesTheSameWay() throws {
        let r = try Reconstruction(json: Data(specV1.utf8))
        #expect(r.skip == 100000)
        #expect(r.fetches[xorbA]?[0].chunks == 0..<2)
        #expect(r.fetches[xorbA]?[0].bytes == 0...28958)
    }

    @Test func jobsGroupTermsByTheFetchThatServesThem() throws {
        let r = try Reconstruction(json: Data(specV2.utf8))
        let jobs = try r.jobs()
        #expect(jobs.count == 2)
        #expect(jobs[0].fetch.url.absoluteString.hasSuffix("/a"))
        #expect(jobs[0].slices.count == 2)
        #expect(jobs[0].slices[0] == Slice(chunks: 1..<4, unpacked: 263873,
                                           skip: 0, length: 263873,
                                           offset: 0))
        #expect(jobs[0].slices[1] == Slice(chunks: 3..<43,
                                           unpacked: 3063572, skip: 0,
                                           length: 3063572,
                                           offset: 263873 + 143890))
        #expect(jobs[1].slices == [Slice(chunks: 0..<3, unpacked: 143890,
                                         skip: 0, length: 143890,
                                         offset: 263873)])
        #expect(jobs.reduce(Int64(0)) { sum, j in sum + j.length }
                == r.unpacked)
    }

    @Test func aRangeSkipsIntoTheFirstTermAndStopsAtTheLimit() throws {
        let r = try Reconstruction(json: Data(specV1.utf8))
        let jobs = try r.jobs(limit: 65536)
        #expect(jobs.count == 1)
        #expect(jobs[0].slices == [Slice(chunks: 0..<2, unpacked: 237171,
                                         skip: 100000, length: 65536,
                                         offset: 0)])
        let whole = try r.jobs()
        #expect(whole[0].slices[0].length == 237171 - 100000)
    }

    @Test func aLimitAcrossTermsCutsTheLastOne() throws {
        let r = try Reconstruction(json: Data(specV2.utf8))
        let jobs = try r.jobs(limit: 263873 + 1000)
        #expect(jobs.count == 2)
        #expect(jobs[1].slices[0].length == 1000)
        #expect(jobs[0].slices.count == 1)
    }

    @Test func aTermNoFetchCoversIsRefused() throws {
        let r = Reconstruction(
            skip: 0, terms: [Term(hash: xorbA, unpacked: 10, chunks: 5..<9)],
            fetches: [xorbA: [Fetch(url: URL(string: "https://x.invalid")!,
                                    chunks: 0..<6, bytes: 0...1)]])
        #expect(throws: XETError.self) { _ = try r.jobs() }
    }

    @Test func theRefreshURLFollowsTheRepoType() {
        let model = URL(string: "https://huggingface.co/leok7v/supertonic/"
            + "resolve/main/supertonic-q8.safetensors")!
        #expect(Hub.refresh(for: model)?.absoluteString
                == "https://huggingface.co/api/models/leok7v/supertonic/"
                + "xet-read-token/main")
        let dataset = URL(string: "https://huggingface.co/datasets/xet-team/"
            + "xet-spec-reference-files/resolve/abc123/some/file.csv")!
        #expect(Hub.refresh(for: dataset)?.absoluteString
                == "https://huggingface.co/api/datasets/xet-team/"
                + "xet-spec-reference-files/xet-read-token/abc123")
        #expect(Hub.refresh(for: URL(string: "https://huggingface.co/x")!)
                == nil)
    }
}
