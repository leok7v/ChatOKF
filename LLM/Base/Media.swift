import CoreGraphics
import Foundation

public struct Modalities: Sendable {
    public let images: Bool
    public let audio: Bool
    public let video: Bool

    public init(images: Bool, audio: Bool, video: Bool) {
        self.images = images
        self.audio = audio
        self.video = video
    }

    public static let none = Modalities(images: false, audio: false,
                                        video: false)
    public var any: Bool { images || audio || video }
}

public struct Attached: Sendable {
    public var parts: [ContentPart]
    public var spans: [SoftSpan]

    public init(parts: [ContentPart] = [], spans: [SoftSpan] = []) {
        self.parts = parts
        self.spans = spans
    }

    public var rows: Int {
        spans.reduce(0) { sum, span in sum + span.rows }
    }

    public mutating func append(_ other: Attached) {
        parts.append(contentsOf: other.parts)
        spans.append(contentsOf: other.spans)
    }
}

public struct MediaError: Error, CustomStringConvertible {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

public struct VideoStrip {
    public let placeholder: Int32
    public let begin: Int32
    public let end: Int32
    public private(set) var ids: [Int32] = []
    public private(set) var features: [Float] = []
    public private(set) var grid: (h: Int, w: Int)? = nil

    public init(placeholder: Int32, begin: Int32, end: Int32) {
        self.placeholder = placeholder
        self.begin = begin
        self.end = end
    }

    public mutating func add(stamp: [Int32], rows: [Float], count: Int,
                             grid g: (h: Int, w: Int)? = nil) {
        ids.append(contentsOf: stamp)
        ids.append(contentsOf: SoftSpan.bracket(
            begin: begin, placeholder: placeholder, end: end, count: count))
        features.append(contentsOf: rows)
        grid = g
    }

    public enum Bracket {
        case template
        case block
    }

    public func span(bracket: Bracket) throws -> SoftSpan {
        if ids.isEmpty {
            throw MediaError("That video could not be read. This device may "
                + "not support the format it was compressed with.")
        }
        return SoftSpan(placeholder: placeholder, ids: ids,
                        features: features, grid: grid,
                        wrap: bracket == .block ? (begin: begin, end: end)
                                                : nil)
    }
}

public protocol MediaEncoder: AnyObject, Sendable {
    var modalities: Modalities { get }
    var audioSampleRate: Double { get }
    var maxAudioSeconds: Double { get }
    func image(_ data: Data, budget: Int) throws -> Attached
    func audio(_ pcm: [Float]) throws -> [SoftSpan]
    func audio(url: URL) async throws -> [SoftSpan]
    func video(url: URL, budget: Int,
               onFrame: ((CGImage, Double) -> Void)?) async throws -> Attached
}
