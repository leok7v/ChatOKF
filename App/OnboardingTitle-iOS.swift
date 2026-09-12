import Chat
import SwiftUI

struct OnboardingTitle: ViewModifier {

    func body(content: Content) -> some View {
        VStack(spacing: 0) {
            Text(Bundle.appName)
                .appFont(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

}
