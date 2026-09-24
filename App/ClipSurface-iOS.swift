import AVFoundation
import SwiftUI
import UIKit

final class ClipView: UIView {

    override class var layerClass: AnyClass { AVPlayerLayer.self }

    private var playerLayer: AVPlayerLayer? { layer as? AVPlayerLayer }

    var player: AVPlayer? {
        get { playerLayer?.player }
        set { playerLayer?.player = newValue }
    }

    var gravity: AVLayerVideoGravity {
        get { playerLayer?.videoGravity ?? .resizeAspect }
        set { playerLayer?.videoGravity = newValue }
    }
}

struct ClipSurface: UIViewRepresentable {

    let player: AVPlayer?

    func makeUIView(context: Context) -> ClipView {
        let view = ClipView()
        view.gravity = .resizeAspect
        view.player = player
        return view
    }

    func updateUIView(_ view: ClipView, context: Context) {
        if view.player !== player { view.player = player }
    }

}
