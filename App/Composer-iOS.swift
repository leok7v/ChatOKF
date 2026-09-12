import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct ComposerMenuStyle: ViewModifier {

    func body(content: Content) -> some View {
        content.menuStyle(.automatic)
    }

}

struct AttachButton: View {

    let model: ChatModel
    @State private var showImporter = false
    @State private var showPhotos = false
    @State private var photos: [PhotosPickerItem] = []

    var body: some View {
        Group {
            if model.canAttachImages {
                Menu { pickers } label: { glyph }
            } else {
                Button { showImporter = true } label: { glyph }
            }
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.hasAttachments ? Color.accentColor
                                              : Color.secondary)
        .disabled(model.busy)
        .photosPicker(isPresented: $showPhotos, selection: $photos,
                      maxSelectionCount: ChatModel.maxImages,
                      matching: model.canAttachVideo
                          ? .any(of: [.images, .videos]) : .images)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: model.attachableTypes,
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                model.handleDrop(urls, at: model.caret)
            }
        }
        .onChange(of: photos) { _, items in load(items) }
    }

    @ViewBuilder
    private var pickers: some View {
        Button(model.canAttachVideo ? "Photos & Videos"
                                    : "Photo Library") {
            showPhotos = true
        }
        Button("Files") { showImporter = true }
    }

    private var glyph: some View {
        Image(systemName: model.attachGlyph)
            .frame(width: 22, height: 22)
    }

    private func load(_ items: [PhotosPickerItem]) {
        if !items.isEmpty {
            Task { @MainActor in
                for item in items {
                    let movie = item.supportedContentTypes.contains { t in
                        t.conforms(to: .movie)
                    }
                    if movie {
                        if let movie = try? await item.loadTransferable(
                            type: PickedMovie.self) {
                            model.attachClip(movie.url, isVideo: true,
                                             at: model.caret, file: movie.name)
                        }
                    } else if let data = try? await item.loadTransferable(
                        type: Data.self) {
                        model.attachImage(data, name: "image", at: model.caret)
                    }
                }
                photos = []
            }
        }
    }
}

struct PickedMovie: Transferable {

    let url: URL
    let name: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let name = received.file.lastPathComponent
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "-" + name)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedMovie(url: copy, name: name)
        }
    }

}
