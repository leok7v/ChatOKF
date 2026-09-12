import SwiftUI

struct OptionKeyMonitor: ViewModifier {

    @Binding var down: Bool

    func body(content: Content) -> some View { content }

}
