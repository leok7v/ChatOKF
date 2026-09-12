import SwiftUI

struct TightSections: ViewModifier {

    func body(content: Content) -> some View {
        content.listSectionSpacing(.compact)
    }

}
