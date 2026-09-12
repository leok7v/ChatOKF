import SwiftUI

struct HeardRing: View {

    let size: CGFloat
    let level: Double
    let hearing: Bool

    @State private var spin = false

    var body: some View {
        ZStack {
            ring(scale: 1.18 + 0.30 * level, width: 3, opacity: 0.85)
            ring(scale: 1.34 + 0.46 * level, width: 2, opacity: 0.40)
        }
        .rotationEffect(.degrees(spin ? 360 : 0))
        .animation(.linear(duration: 6).repeatForever(autoreverses: false),
                   value: spin)
        .animation(.spring(response: 0.28, dampingFraction: 0.6),
                   value: level)
        .animation(.easeOut(duration: 0.25), value: hearing)
        .allowsHitTesting(false)
        .onAppear { spin = true }
    }

    private func ring(scale: CGFloat, width: CGFloat,
                      opacity: Double) -> some View {
        Circle()
            .strokeBorder(HeardRing.spectrum, lineWidth: width)
            .frame(width: size, height: size)
            .scaleEffect(hearing ? scale : 1.02)
            .opacity(hearing ? opacity : 0)
    }

    private static let spectrum = AngularGradient(
        colors: [.red, .orange, .yellow, .green, .cyan, .blue, .purple, .red],
        center: .center)

}
