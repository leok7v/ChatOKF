import SwiftUI
import UniformTypeIdentifiers

struct ComposerMenuStyle: ViewModifier {

    func body(content: Content) -> some View {
        content.menuStyle(.borderlessButton)
    }

}

struct AttachButton: View {

    let model: ChatModel
    @State private var showImporter = false

    var body: some View {
        Button { showImporter = true } label: {
            Image(systemName: model.attachGlyph)
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.hasAttachments ? Color.accentColor
                                              : Color.secondary)
        .disabled(model.busy)
        .help(model.attachHelp)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: model.attachableTypes,
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                model.handleDrop(urls, at: model.caret)
            }
        }
    }

}
