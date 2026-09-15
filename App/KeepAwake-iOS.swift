import UIKit

@MainActor enum KeepAwake {

    static func hold(_ on: Bool) {
        UIApplication.shared.isIdleTimerDisabled = on
    }

}
