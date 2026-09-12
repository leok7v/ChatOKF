import SwiftUI

struct ChatChrome<Leading: View, Center: View, Trailing: View>: ViewModifier {

    let leading: Leading
    let center: Center
    let trailing: Trailing

    func body(content: Content) -> some View {
        content.toolbar {
            ToolbarItem(placement: .navigation) { leading }
            ToolbarItem(placement: .principal) { center }
            ToolbarItem(placement: .primaryAction) { trailing }
        }
    }

}
