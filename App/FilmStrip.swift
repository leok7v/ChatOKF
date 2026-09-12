import Chat
import CoreGraphics
import SwiftUI

struct FilmStrip: View {

    var model: ChatModel
    let quiet: Bool

    @State private var shown: CGImage?
    @State private var arriving: CGImage?
    @State private var fade: Double = 0
    @State private var previous: Date?
    @State private var alive: Double = 1

    var body: some View {
        ZStack {
            if let shown { FilmFrame(image: shown, fade: 1) }
            if let arriving { FilmFrame(image: arriving, fade: fade) }
        }
        .compositingGroup()
        .opacity(FilmFrame.dim * alive)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)
        .opacity(quiet ? 1 : 0)
        .animation(.easeInOut(duration: 0.2), value: quiet)
        .onChange(of: model.lookingAt) { _, next in
            accept(next?.image)
        }
        .onChange(of: model.watching) { _, live in
            if live {
                alive = 1
            } else {
                withAnimation(.linear(duration: 5)) { alive = 0 }
            }
        }
    }

    private func accept(_ next: CGImage?) {
        let now = Date()
        let gap = previous.map { at in now.timeIntervalSince(at) } ?? 0.35
        previous = now
        if let next {
            shown = arriving ?? shown
            arriving = next
            fade = 0
            let seconds = min(max(gap * 0.3, 0.08), 0.25)
            Task { @MainActor in
                withAnimation(.easeInOut(duration: seconds)) { fade = 1 }
            }
        } else {
            shown = nil
            arriving = nil
            previous = nil
        }
    }

}
