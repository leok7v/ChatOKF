import CoreGraphics
import SwiftUI

struct FilmFrame: View {

    let image: CGImage
    let fade: Double
    static let dim: Double = 0.70
    var blur: CGFloat = 1

    var body: some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .blur(radius: blur)
            .mask(FilmFrame.edges)
            .opacity(fade)
    }

    static var edges: some View {
        LinearGradient(
            stops: [.init(color: .clear, location: 0),
                    .init(color: .black, location: 0.18),
                    .init(color: .black, location: 0.82),
                    .init(color: .clear, location: 1)],
            startPoint: .leading, endPoint: .trailing)
        .mask {
            LinearGradient(
                stops: [.init(color: .clear, location: 0),
                        .init(color: .black, location: 0.22),
                        .init(color: .black, location: 0.78),
                        .init(color: .clear, location: 1)],
                startPoint: .top, endPoint: .bottom)
        }
    }

}
