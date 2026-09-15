import Foundation

public var isOS: Bool { return true }

enum Platform {

    static func protectUntilFirstUnlock(_ url: URL) {
        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
            ofItemAtPath: url.path)
    }

}
