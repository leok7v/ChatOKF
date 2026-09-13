import SwiftUI

struct EULAView: View {

    let onAgree: () -> Void
    @State private var atBottom = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(Array(EULAView.paragraphs.enumerated()),
                            id: \.offset) { _, paragraph in
                        Text(paragraph)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(20)
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
            }
            // visibleRect, not contentOffset + containerSize: that form
            // under-counts by the top inset and leaves Accept unreachable.
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.visibleRect.maxY >= geo.contentSize.height - 24
            } action: { _, bottom in
                if bottom { atBottom = true }
            }
            Divider()
            Button(action: onAgree) {
                Text(atBottom ? "Accept" : "Scroll to the end to continue")
                    .frame(maxWidth: .infinity)
            }
            .keyboardShortcut(.defaultAction)
            .controlSize(.large)
            .disabled(!atBottom)
            .padding(16)
        }
    }

    private static let paragraphs: [AttributedString] = agreement
        .components(separatedBy: "\n\n")
        .map { part in
            part.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { part in !part.isEmpty }
        .map { part in
            (try? AttributedString(markdown: part, options: .init(
                interpretedSyntax: .inlineOnlyPreservingWhitespace)))
                ?? AttributedString(part)
        }

}

private let agreement = """
**_Read this agreement in full before proceeding. \
Scrolling to the bottom and clicking Accept \
constitutes your legally binding acceptance of \
every term stated herein._**

**Definitions.** "The Application" means the \
ChatOKF software product, including all versions, \
updates, and associated components. "The Authors" \
means the individual developers and contributors \
of the Application. "You" means the natural \
person or legal entity installing or using the \
Application. "Model" means any third-party \
artificial intelligence model accessed through \
the Application.

**1. Disclaimer of Warranties.** The Application \
is provided **"as is"** and **"as available"** \
without warranty of any kind, whether express, \
implied, statutory, or otherwise. The Authors \
expressly disclaim all implied warranties, \
including merchantability, fitness for a \
particular purpose, title, accuracy, and \
non-infringement. No oral or written statement \
by the Authors creates any warranty not stated \
here. The entire risk as to the quality and \
performance of the Application rests solely \
with You.

**2. Disclaimer of Liability.** To the maximum \
extent permitted by applicable law, the Authors \
shall not be liable for any damages whatsoever \
arising out of or related to Your use of, or \
inability to use, the Application or any content \
generated through it. This covers all categories \
of damages, including direct, indirect, \
incidental, special, consequential, exemplary, \
and punitive damages, loss of profits, loss of \
data, business interruption, and personal injury, \
even if the Authors have been advised of the \
possibility of such damages and regardless of \
the legal theory under which liability is asserted.

**3. No Professional Advice.** Nothing produced \
by the Application constitutes legal, medical, \
financial, psychological, or any other form of \
regulated professional advice. You assume full \
responsibility for any decision made in reliance \
on Application output and are solely responsible \
for seeking qualified professional counsel for \
all critical matters.

**4. Accuracy and Safety Not Guaranteed.** The \
Application may produce responses that are \
inaccurate, incomplete, outdated, biased, \
offensive, or otherwise unsuitable for any \
given purpose. You bear sole responsibility for \
evaluating, verifying, and determining the \
fitness of any output before acting upon it.

**5. No Data Persistence Guarantee.** The Authors \
make no guarantee that any user-generated data \
will be preserved, recoverable, or protected \
against loss. You are solely responsible for \
maintaining independent backups of any data you \
wish to retain.

**6. Your Content and Stored Memories.** The \
Application keeps on Your device, and nowhere \
else, the conversations You conduct, the \
documents, images, audio and other material You \
provide, and the notes ("memories") it forms from \
those conversations, whether written by a Model or \
by You. You are solely responsible for all such \
content: for its legality, accuracy and \
suitability, for holding the rights to any \
material You provide, and for the safekeeping of \
Your device and of any backup or export that \
carries this content. The Authors never receive, \
review, transmit or curate this content and \
accept no responsibility for it, for anything a \
Model derives from it, or for any consequence of \
its loss, disclosure, inaccuracy or use. Memories \
are formed and recalled on a best-effort basis \
and may be incomplete, mistaken or absent.

**7. Third-Party Model Licenses.** The Application \
provides a technical interface to third-party AI \
Models. Each Model is governed exclusively by its \
own license agreement, terms of use, and applicable \
laws, entirely independent of this Agreement. It \
is Your sole responsibility to identify, obtain, \
read, and accept the applicable license terms for \
every Model You access before use. The Authors are \
not a party to any agreement between You and any \
Model provider. The Authors exercise no control \
over, and make no representations regarding, the \
content, capabilities, legality, safety, or \
licensing terms of any Model. The Authors are not \
responsible for any output generated by a Model, \
nor for any consequence arising from Your use, \
misuse, or redistribution of such output. Your \
use of any Model through the Application is \
conclusive evidence that You have independently \
reviewed and accepted that Model's license, and \
that You hold the Authors entirely harmless with \
respect to any claim arising therefrom.

**8. Indemnification.** You agree to indemnify, \
defend, and hold harmless the Authors from and \
against any and all claims, demands, actions, \
losses, liabilities, damages, costs, and expenses \
(including reasonable legal fees) arising out of \
or relating to: (a) Your use or misuse of the \
Application or any Model; (b) any output You \
generate, rely upon, or distribute; (c) Your \
violation of this Agreement; or (d) Your violation \
of any applicable law or third-party right.

**9. Severability.** If any provision of this \
Agreement is held invalid, illegal, or \
unenforceable, that provision shall be modified \
to the minimum extent necessary to make it \
enforceable, and the remaining provisions shall \
continue in full force and effect.

**10. Entire Agreement.** This Agreement constitutes \
the entire agreement between You and the Authors \
concerning the Application and supersedes all \
prior or contemporaneous representations, \
understandings, negotiations, or agreements, \
whether written or oral, relating to the same \
subject matter.

**11. Acceptance.** By clicking Accept, installing, \
or otherwise using the Application, You confirm \
that You have read, understood, and agree to be \
bound by every term of this Agreement. If You do \
not agree, do not install or use the Application.
"""
