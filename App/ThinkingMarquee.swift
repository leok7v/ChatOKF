import Chat
import SwiftUI

struct ThinkingMarquee: View {

    let text: String
    @State private var stream: [String] = []
    @State private var taken = 0
    @State private var travel: CGFloat = 0
    @State private var width: CGFloat = 0
    @State private var viewport: CGFloat = 0
    @State private var measured = 0
    @State private var scrolling = false
    @State private var origin: CGFloat?
    private static let charsPerSecond: CGFloat = 14
    private static let gap = "    "

    private var line: String { stream.joined(separator: Self.gap) }

    private var pxPerChar: CGFloat {
        max(width / CGFloat(max(line.count, 1)), 4)
    }

    var body: some View {
        Text(" ")
            .appFont(.caption)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .leading) {
                HStack(spacing: 0) {
                    Text(line)
                        .appFont(.caption)
                        .foregroundStyle(Color(white: 0.53))
                        .lineLimit(1)
                        .fixedSize()
                        .id(line)
                        .onGeometryChange(for: CGFloat.self, of: { geo in
                            geo.size.width
                        }, action: { w in
                            width = w
                            measured += 1
                        })
                }
                .modifier(Shift(x: -travel))
            }
            .clipped()
            .onGeometryChange(for: CGFloat.self, of: { geo in
                geo.size.width
            }, action: { w in viewport = w })
            .onGeometryChange(for: CGFloat.self, of: { geo in
                geo.frame(in: .global).minX
            }, action: { x in
                if let was = origin { travel += x - was }
                origin = x
            })
            .mask(LinearGradient(
                stops: [.init(color: .clear, location: 0),
                        .init(color: .black, location: 0.12)],
                startPoint: .leading, endPoint: .trailing))
            .padding(.trailing, 8)
            .allowsHitTesting(false)
            .onAppear { take(Sentences.split(text)) }
            .onChange(of: text) { _, now in take(Sentences.split(now)) }
            .task(id: measured) { if !scrolling { run() } }
            .task(id: viewport) { if !scrolling { run() } }
    }

    private func take(_ parts: [String]) {
        let complete = parts.count - 1
        if complete > taken {
            var next = stream + parts[taken..<complete]
            taken = complete
            let shown = Int((travel + viewport) / pxPerChar)
            let budget = Int(2 * viewport / pxPerChar)
            var offset = 0
            var pending = next.count
            for (i, piece) in next.enumerated() {
                if offset > shown { pending = min(pending, i) }
                offset += piece.count + Self.gap.count
            }
            var backlog = next[pending...].reduce(0) { sum, piece in
                sum + piece.count + Self.gap.count
            }
            while backlog > budget && next.count - 1 > pending {
                backlog -= next[pending].count + Self.gap.count
                next.remove(at: pending)
            }
            stream = next
        }
    }

    private func run() {
        let end = max(0, width - viewport)
        let speed = Self.charsPerSecond * pxPerChar
        if viewport > 0 && end - travel >= viewport {
            scrolling = true
            withAnimation(.linear(duration: (end - travel) / speed),
                          completionCriteria: .logicallyComplete) {
                travel = end
            } completion: {
                Task { @MainActor in
                    scrolling = false
                    run()
                }
            }
        }
    }

}

struct Shift: AnimatableModifier {

    var x: CGFloat

    nonisolated var animatableData: CGFloat {
        get { x }
        set { x = newValue }
    }

    func body(content: Content) -> some View {
        content.offset(x: x)
    }

}
