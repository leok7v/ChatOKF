import Chat
import ImageIO
import LLM
import SwiftUI

private func elapsed(_ t: Date, since zero: Date) -> String {
    let ms = Int((max(t.timeIntervalSince(zero), 0) * 1000).rounded())
    let out: String
    if ms >= 3_600_000 {
        out = String(format: "%02d:%02d:%02d.%03d", ms / 3_600_000,
                     ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
    } else {
        out = String(format: "%02d:%02d.%03d",
                     ms / 60_000, ms / 1000 % 60, ms % 1000)
    }
    return out
}

struct DebugView: View {

    let model: ChatModel
    let onClose: () -> Void
    @State private var selected: UUID?

    private var zero: Date {
        min(Instrument.launched, model.traceEvents.first?.t1
            ?? Instrument.launched)
    }

    var body: some View {
        Group {
            if isOS {
                NavigationStack {
                    content.navigationTitle("Session Debug")
                        .toolbar { DoneToolbar(action: onClose) }
                }
            } else {
                VStack(spacing: 0) {
                    header
                    content
                }
            }
        }
    }

    private var content: some View {
        VStack(spacing: 0) {
            if isOS { eventCount.padding(.horizontal, 16) }
            TraceGraph(events: model.traceEvents, zero: zero,
                       selected: $selected)
                .frame(height: 190)
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            Divider()
            transcript
            Divider()
            footer
        }
    }

    private var eventCount: some View {
        Text("\(model.traceEvents.count) events")
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var header: some View {
        HStack {
            Label("Session Debug", systemImage: "ladybug")
                .appFont(.headline)
            eventCount.fixedSize()
            Spacer()
            DoneButton(action: onClose)
            EscapeToClose(action: onClose)
        }
        .padding(12)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.traceEvents) { e in
                        TraceRow(e: e, zero: zero, selected: e.id == selected)
                            .id(e.id)
                            .onTapGesture {
                                selected = selected == e.id ? nil : e.id
                            }
                    }
                }
                .padding(12)
            }
            .onChange(of: selected) { _, sel in
                if let sel {
                    withAnimation { proxy.scrollTo(sel, anchor: .center) }
                }
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 8) {
            if !model.tracePath.isEmpty {
                logButton("Transcript", model.tracePath)
                if !model.diagPath.isEmpty {
                    Text("\u{00B7}").foregroundStyle(.tertiary)
                }
            }
            if !model.diagPath.isEmpty {
                logButton("Diagnostics", model.diagPath)
            }
            Spacer()
        }
        .appFont(.caption2)
        .padding(8)
    }

    @State private var copied: String?

    private func logButton(_ label: String, _ path: String) -> some View {
        Button { copyLog(path) } label: {
            HStack(spacing: 4) {
                Image(systemName: copied == path
                      ? "checkmark" : "doc.on.doc")
                Text("\(label): \(DebugView.folder(path)) "
                     + "(\(DebugView.size(path)))")
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help("Copy the \(label.lowercased()) log to the clipboard")
    }

    private func copyLog(_ path: String) {
        let text = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        setClipboard(text)
        copied = path
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(1200))
            if copied == path { copied = nil }
        }
    }

    private static func folder(_ path: String) -> String {
        URL(fileURLWithPath: path).deletingLastPathComponent()
            .lastPathComponent
    }

    private static func size(_ path: String) -> String {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let bytes = (attrs?[.size] as? Int64) ?? 0
        return ByteCountFormatter.string(fromByteCount: bytes,
                                         countStyle: .file)
    }
}

private struct TraceGraph: View {
    let events: [TraceEvent]
    let zero: Date
    @Binding var selected: UUID?
    @State private var hovered: TraceEvent?
    @State private var hoverAt: CGPoint = .zero
    @Environment(\.appTextScale) private var textScale

    private static let pad: CGFloat = 8

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                draw(&ctx, size)
            }
            .background(Color.gray.opacity(0.06),
                        in: RoundedRectangle(cornerRadius: 8))
            .onTapGesture(coordinateSpace: .local) { p in
                selected = nearest(p.x, geo.size)?.id
            }
            .onContinuousHover(coordinateSpace: .local) { phase in
                switch phase {
                case .active(let p):
                    hovered = nearest(p.x, geo.size)
                    hoverAt = p
                case .ended:
                    hovered = nil
                }
            }
            .overlay(alignment: .topLeading) {
                if let hovered {
                    HoverBubble(e: hovered, zero: zero)
                        .offset(
                            x: min(max(hoverAt.x + 12, 0),
                                   max(geo.size.width - 240, 0)),
                            y: min(max(hoverAt.y - 56, 0),
                                   max(geo.size.height - 64, 0)))
                        .allowsHitTesting(false)
                }
            }
        }
    }

    private var span: (start: Date, seconds: Double) {
        let start = events.first?.t1 ?? Date()
        let end = events.last?.t1 ?? start
        return (start, max(end.timeIntervalSince(start), 1))
    }

    private var maxCtx: Int {
        max(events.map { e in e.ctx }.max() ?? 1, 1)
    }

    private func x(_ t: Date, _ w: CGFloat) -> CGFloat {
        let s = span
        let f = t.timeIntervalSince(s.start) / s.seconds
        return TraceGraph.pad + CGFloat(f) * (w - 2 * TraceGraph.pad)
    }

    private func y(_ ctx: Int, _ h: CGFloat) -> CGFloat {
        let markerLane = 24.0
        let f = CGFloat(ctx) / CGFloat(maxCtx)
        return h - TraceGraph.pad
            - f * (h - markerLane - 2 * TraceGraph.pad)
    }

    private static func markerColor(_ kind: TraceEvent.Kind) -> Color? {
        switch kind {
        case .toolCall: return .orange
        case .toolResult: return .teal
        case .inject: return .yellow
        case .rewind, .reset: return .red
        default: return nil
        }
    }

    private static func rate(_ e: TraceEvent) -> Double {
        let dur = e.t1.timeIntervalSince(e.t0)
        let rated = e.kind == .prefill || e.kind == .decode
        return rated && e.tokens > 0 && dur > 0
            ? Double(e.tokens) / dur : 0
    }

    private var maxRate: Double {
        max(events.map { e in TraceGraph.rate(e) }.max() ?? 1, 1)
    }

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize) {
        var line = Path()
        var lastY: CGFloat? = nil
        let rateTop = maxRate
        for e in events {
            let ex = x(e.t1, size.width)
            if e.ctx >= 0 {
                let ey = y(e.ctx, size.height)
                if let ly = lastY {
                    line.addLine(to: CGPoint(x: ex, y: ly))
                    line.addLine(to: CGPoint(x: ex, y: ey))
                } else {
                    line.move(to: CGPoint(x: ex, y: ey))
                }
                lastY = ey
            }
            if e.kind == .user {
                var rule = Path()
                rule.move(to: CGPoint(x: ex, y: 20))
                rule.addLine(to: CGPoint(x: ex, y: size.height - 4))
                ctx.stroke(rule, with: .color(.accentColor.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
            let r = TraceGraph.rate(e)
            if r > 0 {
                let ry = y(Int(r / rateTop * Double(maxCtx)), size.height)
                let box = CGRect(x: ex - 2.5, y: ry - 2.5,
                                 width: 5, height: 5)
                if e.kind == .decode {
                    ctx.fill(Path(ellipseIn: box), with: .color(.green))
                } else {
                    ctx.stroke(Path(ellipseIn: box),
                               with: .color(.green), lineWidth: 1)
                }
            }
            if let color = TraceGraph.markerColor(e.kind) {
                let radius: CGFloat = e.id == selected ? 5 : 3.5
                let dot = Path(ellipseIn: CGRect(
                    x: ex - radius, y: 10 - radius,
                    width: 2 * radius, height: 2 * radius))
                ctx.fill(dot, with: .color(color))
            }
        }
        ctx.stroke(line, with: .color(.blue), lineWidth: 1.5)
        let label = appTextFont(.caption2, textScale)
        ctx.draw(Text("ctx \(maxCtx)")
                     .font(label).foregroundStyle(.secondary),
                 at: CGPoint(x: size.width - 34, y: 8))
        ctx.draw(Text(String(format: "%.0f t/s", maxRate))
                     .font(label).foregroundStyle(.green.opacity(0.7)),
                 at: CGPoint(x: size.width - 34, y: 22))
    }

    private func nearest(_ tapX: CGFloat, _ size: CGSize) -> TraceEvent? {
        var best: (e: TraceEvent, d: CGFloat)? = nil
        for e in events {
            var d = abs(x(e.t1, size.width) - tapX)
            if TraceGraph.markerColor(e.kind) != nil || e.kind == .user {
                d -= 6
            }
            if best == nil || d < best!.d { best = (e, d) }
        }
        return best?.e
    }
}

private struct HoverBubble: View {
    let e: TraceEvent
    let zero: Date

    private var stats: String {
        var parts: [String] = []
        if e.tokens > 0 { parts.append("\(e.tokens) tok") }
        if e.ctx >= 0 { parts.append("ctx \(e.ctx)") }
        let dur = e.t1.timeIntervalSince(e.t0)
        if dur >= 0.001 { parts.append(String(format: "%.2fs", dur)) }
        if e.tokens > 0 && dur > 0 {
            parts.append(String(format: "%.0f t/s",
                                Double(e.tokens) / dur))
        }
        return parts.joined(separator: "  ")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(elapsed(e.t1, since: zero))  \(e.kind.rawValue)")
                .appFont(.caption2).monospaced()
                .foregroundStyle(.secondary)
            Text(e.summary)
                .appFont(.caption)
                .lineLimit(2)
            if !stats.isEmpty {
                Text(stats)
                    .appFont(.caption2).monospaced()
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .frame(maxWidth: 240, alignment: .leading)
        .background(.thinMaterial,
                    in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8)
            .stroke(.separator, lineWidth: 0.5))
    }
}

private struct TraceRow: View {
    let e: TraceEvent
    let zero: Date
    let selected: Bool
    @State private var payloadHeight: CGFloat = 20

    private var color: Color {
        switch e.kind {
        case .user: return .accentColor
        case .render: return .secondary
        case .prefill: return .blue
        case .prime: return .mint
        case .vision: return .purple
        case .decode: return .primary
        case .toolCall: return .orange
        case .toolResult: return .teal
        case .inject: return .yellow
        case .rewind, .reset: return .red
        case .answer: return .green
        case .diag: return .indigo
        }
    }

    private var meta: String {
        var parts: [String] = []
        if e.tokens > 0 { parts.append("\(e.tokens) tok") }
        if e.ctx >= 0 { parts.append("ctx \(e.ctx)") }
        let dur = e.t1.timeIntervalSince(e.t0)
        if dur >= 0.001 { parts.append(String(format: "%.2fs", dur)) }
        return parts.joined(separator: "  ")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(elapsed(e.t1, since: zero))
                    .appFont(.caption).monospaced()
                    .foregroundStyle(.secondary)
                Text(e.kind.rawValue)
                    .appFont(.caption).bold()
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(color.opacity(0.18), in: Capsule())
                    .foregroundStyle(color)
                if !meta.isEmpty {
                    Text(meta)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(e.summary)
                    .appFont(.caption)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            if let cg = TraceRow.thumb(e.image) {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 128, maxHeight: 128)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .padding(.leading, 4)
            }
            if !e.text.isEmpty {
                payload
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(selected ? Color.accentColor.opacity(0.12) : .clear,
                    in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
    }

    @ViewBuilder private var payload: some View {
        let text = Text(e.text)
            .appFont(.caption2).monospaced()
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .padding(.leading, 4)
        if selected {
            ScrollView {
                text
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self, of: { geo in
                        geo.size.height
                    }, action: { h in payloadHeight = h })
            }
            .frame(height: min(max(payloadHeight, 20), 300))
        } else {
            text.lineLimit(2)
        }
    }

    private static func thumb(_ data: Data?) -> CGImage? {
        data.flatMap { d in VisionPreprocess.image(d) }
    }

}
