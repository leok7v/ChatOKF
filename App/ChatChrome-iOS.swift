import SwiftUI

struct ChatChrome<Leading: View, Center: View, Trailing: View>: ViewModifier {

    let leading: Leading
    let center: Center
    let trailing: Trailing

    func body(content: Content) -> some View {
        content.safeAreaInset(edge: .top, spacing: 0) {
            HStack(spacing: 0) {
                leading
                Spacer(minLength: 8)
                center
                Spacer(minLength: 8)
                trailing
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }

}
