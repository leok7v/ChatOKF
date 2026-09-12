import Foundation

enum AudioSession {
    static func beginRecording() { }
    static func endRecording() { }
    static func beginPlayback() { }
    static func endPlayback() { }
    static func describe() -> String { "macOS: no session" }
}
