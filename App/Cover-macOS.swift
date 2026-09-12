import SwiftUI

extension View {
    func cover<Content: View>(isPresented: Binding<Bool>,
                              @ViewBuilder content: @escaping () -> Content)
        -> some View {
        sheet(isPresented: isPresented, content: content)
    }
}
