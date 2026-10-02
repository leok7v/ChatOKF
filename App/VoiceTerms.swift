import Chat
import SwiftUI

enum VoiceTerms {

    static let licenceURL = URL(string:
        "https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE")!

    static let restrictions = Texts.text("voice-restrictions")

    private static let key = "voiceTermsAccepted.openrail-m"

    static var accepted: Bool {
        UserDefaults.standard.bool(forKey: key)
    }

    static func accept() {
        UserDefaults.standard.set(true, forKey: key)
    }

}

struct VoiceRestrictions: View {

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Use Restrictions", systemImage: "exclamationmark.shield")
                .appFont(.headline)
                .foregroundStyle(.orange)
            Text("Attachment A of the BigScience Open RAIL-M License")
                .appFont(.caption)
                .foregroundStyle(.secondary)
            Text(VoiceTerms.restrictions)
                .appFont(.callout)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

}

struct VoiceTermsView: View {

    let size: String
    let onAgree: () -> Void
    let onCancel: () -> Void

    var body: some View {
        ScrollView {
            terms
                .padding(24)
                .frame(maxWidth: 560)
                .frame(maxWidth: .infinity)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { answers }
    }

    private var terms: some View {
        VStack(spacing: 14) {
            Image(systemName: "waveform")
                .appFont(.largeTitle)
                .foregroundStyle(.secondary)
            Text("The Reading Voice").appFont(.title2).bold()
            Text("Replies are read aloud by Supertonic 3, a speech model by "
               + "Supertone Inc. It downloads once, \(size), and then speaks "
               + "on this device with no network. Supertone provides it under "
               + "the BigScience Open RAIL-M License, which restricts what "
               + "the voice may be used for. By continuing you agree to that "
               + "license and to the restrictions below.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            VoiceRestrictions()
                .padding(14)
                .background(Color.primary.opacity(0.055),
                            in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(.separator.opacity(0.6), lineWidth: 0.5)
                }
            Link("BigScience Open RAIL-M License, the full text",
                 destination: VoiceTerms.licenceURL)
                .appFont(.callout)
            Text("Supertonic is a Supertone model. This app is not "
               + "affiliated with or endorsed by Supertone.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
    }

    private var answers: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 14) {
                Button("Cancel", action: onCancel)
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
                Button("Agree and Download", action: onAgree)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(16)
            .frame(maxWidth: .infinity)
        }
        .background(.bar)
    }

}
