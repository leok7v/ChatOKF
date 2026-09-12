import CoreGraphics

public struct VideoPeek: @unchecked Sendable, Equatable {

    public let index: Int
    public let image: CGImage

    public static func == (a: VideoPeek, b: VideoPeek) -> Bool {
        a.index == b.index
    }

    public init?(index: Int, full: CGImage, side: Int = 480) {
        let w = full.width, h = full.height
        let factor = min(1, Double(side) / Double(max(w, h)))
        let tw = max(1, Int(Double(w) * factor))
        let th = max(1, Int(Double(h) * factor))
        var made: CGImage? = nil
        if let space = CGColorSpace(name: CGColorSpace.sRGB),
           let ctx = CGContext(
               data: nil, width: tw, height: th, bitsPerComponent: 8,
               bytesPerRow: 0, space: space,
               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) {
            ctx.interpolationQuality = .medium
            ctx.draw(full, in: CGRect(x: 0, y: 0, width: tw, height: th))
            made = ctx.makeImage()
        }
        if let made {
            self.index = index
            self.image = made
        } else {
            return nil
        }
    }

    public init(index: Int, ready: CGImage) {
        self.index = index
        self.image = ready
    }
}
