import Foundation

final class DemoCapture: @unchecked Sendable {

    static let shared = DemoCapture()

    var waiting: Bool { false }

    func start(to url: URL, done: @escaping @Sendable (String?) -> Void) {
        done("capture: not on macOS")
    }

    func stop(done: @escaping @Sendable (String) -> Void) {
        done("capture: not on macOS")
    }

}
