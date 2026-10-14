import SwiftUI
import UIKit

struct SharedFile: Identifiable {
    let id = UUID()
    let url: URL
}

extension View {
    func shareSheet(_ file: Binding<SharedFile?>) -> some View {
        sheet(item: file) { shared in
            ActivitySheet(url: shared.url)
                .presentationDetents([.medium, .large])
                .ignoresSafeArea()
        }
    }
}

private struct ActivitySheet: UIViewControllerRepresentable {

    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url],
                                 applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController,
                                context: Context) {}

}
