import AVKit
import SwiftUI

struct ClipSurface: UIViewControllerRepresentable {

    let player: AVPlayer?

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let view = AVPlayerViewController()
        view.showsPlaybackControls = false
        view.canStartPictureInPictureAutomaticallyFromInline = false
        view.updatesNowPlayingInfoCenter = false
        view.videoGravity = .resizeAspect
        view.player = player
        return view
    }

    func updateUIViewController(_ view: AVPlayerViewController,
                                context: Context) {
        if view.player !== player { view.player = player }
    }

}
