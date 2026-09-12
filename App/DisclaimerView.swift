import SwiftUI

struct DisclaimerView: View {

    let onAgree: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Before you begin")
                        .appFont(.title2).bold()
                    Text("ChatOKF runs a small language model entirely on this "
                        + "device, on its GPU. Your conversations "
                        + "stay local and are never sent to any server. "
                        + "Optional lookups (Wikipedia, web search; see "
                        + "Settings) download public data from the web.")
                    Text("Treat every response critically.").appFont(.headline)
                    Text("A language model predicts plausible text; it does not "
                        + "verify facts and has no understanding of truth. Output "
                        + "may be confident and still be wrong, incomplete, or "
                        + "biased. Check it against reliable sources, and do not "
                        + "use it as the basis for legal, medical, financial, or "
                        + "safety decisions.")
                    Text("No warranty.").appFont(.headline)
                    Text("This software is provided \"as is\", without warranty of "
                        + "any kind, and is used at your own risk.")
                    Text("By continuing you acknowledge these terms.")
                        .foregroundStyle(.secondary)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            Button(action: onAgree) {
                Text("I Understand and Agree")
                    .frame(maxWidth: .infinity)
            }
            .keyboardShortcut(.defaultAction)
            .controlSize(.large)
            .padding(16)
        }
    }

}
