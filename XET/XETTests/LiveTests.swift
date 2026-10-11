import Foundation
import Testing
@testable import XET

@Suite(.enabled(if: ProcessInfo.processInfo.environment["CHATOKF_LIVE_NETWORK"]
                   != nil))
struct LiveTests {
    private let resolve = URL(string: "https://huggingface.co/datasets/"
        + "xet-team/xet-spec-reference-files/resolve/main/"
        + "Electric_Vehicle_Population_Data_20250917.csv")!

    private func plain(_ range: Range<Int64>) async throws -> Data {
        var req = URLRequest(url: resolve)
        req.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)",
                     forHTTPHeaderField: "Range")
        let (data, resp) = try await URLSession.shared.data(for: req)
        #expect((resp as? HTTPURLResponse)?.statusCode == 206)
        return data
    }

    @Test func aRangeOfAPublicFileMatchesTheLFSBridgeByteForByte()
        async throws {
        let file = try #require(try await Hub.file(resolve))
        #expect(file.id.count == 64)
        #expect(file.size == 63527244)
        let token = try await Hub.token(try #require(Hub.refresh(for: resolve)))
        #expect(token.expires > Date())
        let range: Range<Int64> = 100_000..<(100_000 + 300_000)
        let r = try await Hub.reconstruction(of: file.id, token: token,
                                             range: range)
        #expect(r.skip == 100_000 - 0 || r.skip < 100_000)
        let ours = try await Downloader.data(r, limit: Int64(range.count))
        let theirs = try await plain(range)
        #expect(ours.count == range.count)
        #expect(ours == theirs)
    }

    @Test func theWholeFileReconstructsToItsLinkedSize() async throws {
        let file = try #require(try await Hub.file(resolve))
        let token = try await Hub.token(try #require(Hub.refresh(for: resolve)))
        let r = try await Hub.reconstruction(of: file.id, token: token)
        #expect(r.unpacked == file.size)
        let jobs = try r.jobs()
        #expect(jobs.reduce(Int64(0)) { sum, j in sum + j.length }
                == file.size)
        let t0 = Date()
        let got = try await Downloader.data(r)
        let secs = Date().timeIntervalSince(t0)
        print(String(format: "xet live whole file: %d bytes, %d job(s) in "
                     + "%.1f s, %.1f MiB/s", got.count, jobs.count, secs,
                     Double(got.count) / secs / 1048576))
        #expect(got.count == Int(file.size))
    }
}
