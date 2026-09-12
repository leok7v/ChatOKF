import Chat
import LLM
import SwiftUI

enum GemmaTerms {

    static let modelNames: Set<String> = []

    static func applies(to name: String) -> Bool { modelNames.contains(name) }

    static let termsURL = URL(string: "https://ai.google.dev/gemma/terms")!
    static let policyURL =
        URL(string: "https://ai.google.dev/gemma/prohibited_use_policy")!

    private static let key = "gemmaTermsAccepted"

    static var accepted: Bool {
        UserDefaults.standard.bool(forKey: key)
    }

    static func accept() {
        UserDefaults.standard.set(true, forKey: key)
    }

}

struct GemmaTermsView: View {

    let model: String
    let onAgree: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.text")
                .appFont(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Gemma Terms of Use").appFont(.title2).bold()
            Text("\(Models.display(model)) is provided by "
               + "Google under the Gemma Terms of Use, not the open-source "
               + "licence the other models use. By continuing you agree to "
               + "those terms and to the Prohibited Use Policy, which "
               + "restrict what the model may be used for.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 6) {
                Link("Gemma Terms of Use", destination: GemmaTerms.termsURL)
                Link("Prohibited Use Policy", destination: GemmaTerms.policyURL)
            }
            .appFont(.callout)
            Text("Gemma is a Google model. This app is not affiliated with "
               + "or endorsed by Google.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            VStack(spacing: 14) {
                Button("Agree and Continue", action: onAgree)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                Button("Cancel", action: onCancel)
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: 460)
    }

}
