import CoreGraphics
import Foundation
import LLM

public struct PlannedTurn: Sendable {
    public let prompt: String
    public let answer: String
    public let soft: URL?
    public let images: [CGImage]

    public init(prompt: String, answer: String, soft: URL?,
                images: [CGImage]) {
        self.prompt = prompt
        self.answer = answer
        self.soft = soft
        self.images = images
    }
}

public enum ResumePlan: Sendable {
    case parked
    case replay([PlannedTurn])
    case needsModel(String)
    case readOnly(String)
}

extension Session {

    public func resumePlan(_ id: UUID, _ messages: [Message]) async
        -> ResumePlan {
        var out = ResumePlan.readOnly("no model is loaded")
        if hasParked(id) {
            out = .parked
        } else if ggufBackend != nil {
            let urls = messages.compactMap { m in m.soft }
            let stamps = await Task.detached { Session.stamps(urls) }.value
            out = Session.plan(messages, loaded: modelName,
                               sees: modalities.images, stamps: stamps)
        }
        let label = String(id.uuidString.prefix(8))
        let verdict: String
        switch out {
        case .parked: verdict = "parked"
        case .replay(let turns): verdict = "replay \(turns.count) turn(s)"
        case .needsModel(let name): verdict = "needs \(name)"
        case .readOnly(let why): verdict = "read only, \(why)"
        }
        Diag.shared.report(.load, "[replay] \(label): \(verdict)")
        return out
    }

    nonisolated static func stamps(_ urls: [URL]) -> [URL: String] {
        var out: [URL: String] = [:]
        for url in urls {
            if let stamp = SoftFile.stamp(of: url) { out[url] = stamp }
        }
        return out
    }

    nonisolated static func stampModel(_ stamp: String) -> String {
        var out = stamp
        if let dot = stamp.range(of: ".", options: .backwards),
           stamp.distance(from: dot.upperBound, to: stamp.endIndex) == 8 {
            out = String(stamp[..<dot.lowerBound])
        }
        return out
    }

    nonisolated public static func towerFamily(_ name: String) -> String {
        var out = name
        if name.hasPrefix("Qwen3.8-27B") { out = "Qwen3.8-27B" }
        if out.hasSuffix("-MTP") { out = String(out.dropLast(4)) }
        return out
    }

    nonisolated static func plan(_ messages: [Message], loaded: String,
                                 sees: Bool,
                                 stamps: [URL: String]) -> ResumePlan {
        var turns: [PlannedTurn] = []
        var why: String? = nil
        var needs: String? = nil
        var i = 0
        while i < messages.count && why == nil && needs == nil {
            let m = messages[i]
            let next = i + 1 < messages.count ? messages[i + 1] : nil
            let answer = next.map { n in n.fromUser ? "" : n.text } ?? ""
            if m.fromUser && !answer.isEmpty {
                if let soft = m.soft {
                    if let stamp = stamps[soft] {
                        let model = Session.stampModel(stamp)
                        if Session.towerFamily(model)
                            == Session.towerFamily(loaded) {
                            turns.append(PlannedTurn(
                                prompt: m.laid, answer: answer, soft: soft,
                                images: []))
                        } else {
                            needs = model
                        }
                    } else {
                        why = "its attachments were not kept"
                    }
                } else if m.text.hasPrefix("Spoken,") {
                    why = "what was spoken was not kept"
                } else if !m.clips.isEmpty || !m.posters.isEmpty {
                    why = "its video was not kept"
                } else if !m.images.isEmpty && !sees {
                    why = "needs a model that sees pictures"
                } else {
                    turns.append(PlannedTurn(prompt: m.laid, answer: answer,
                                             soft: nil, images: m.images))
                }
            }
            i += 1
        }
        let out: ResumePlan
        if let needs {
            out = .needsModel(needs)
        } else if let why {
            out = .readOnly(why)
        } else if turns.isEmpty {
            out = .readOnly("nothing to resume")
        } else {
            out = .replay(turns)
        }
        return out
    }

    public func replay(_ turns: [PlannedTurn], budget: Int,
                       _ config: SessionConfig,
                       onEvent: @escaping @MainActor (TraceEvent) -> Void)
        async -> Bool {
        var out = false
        if ggufBackend != nil {
            await newChatEngine(config, onEvent: onEvent)
            await awaitPrimed()
            if let session {
                let t0 = Date()
                let encoder = media
                do {
                    let laid = try await Task.detached {
                        try Session.materialized(turns, encoder, budget)
                    }.value
                    let tokens = try await session.replay(laid)
                    Diag.shared.report(.load, String(
                        format: "[replay] %d turn(s), %d tokens in %.1fs",
                        turns.count, tokens, Date().timeIntervalSince(t0)))
                    out = true
                } catch {
                    Diag.shared.report("[replay] not resumed after "
                                       + "\(turns.count) turn(s): \(error)")
                }
            }
        }
        return out
    }

    nonisolated static func materialized(
        _ turns: [PlannedTurn], _ media: (any MediaEncoder)?,
        _ budget: Int) throws -> [ChatSession.ReplayTurn] {
        var out: [ChatSession.ReplayTurn] = []
        for turn in turns {
            var parts: [ContentPart] = []
            var spans: [SoftSpan] = []
            var labelled = true
            var prompt = turn.prompt
            if let url = turn.soft {
                let file = try SoftFile.read(url)
                parts = file.parts
                spans = file.spans
                labelled = file.labelled
            } else if let media {
                for cg in turn.images {
                    if let data = VisionPreprocess.jpeg(cg) {
                        let got = try media.image(data, budget: budget)
                        parts += got.parts
                        spans += got.spans
                    }
                }
            }
            if !spans.isEmpty {
                if prompt.isEmpty { prompt = Session.softDefaultPrompt(parts) }
                parts.append(.text(prompt))
            }
            out.append(ChatSession.ReplayTurn(
                prompt: prompt, answer: turn.answer, parts: parts,
                spans: spans, labelled: labelled))
        }
        return out
    }

}
