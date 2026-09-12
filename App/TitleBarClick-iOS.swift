import SwiftUI

struct TitleBarClickMonitor: ViewModifier {

    let onClick: () -> Void

    func body(content: Content) -> some View { content }

}
