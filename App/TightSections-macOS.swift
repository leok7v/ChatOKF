import SwiftUI

// listSectionSpacing is unavailable on macOS.

struct TightSections: ViewModifier {
    func body(content: Content) -> some View { content }
}
