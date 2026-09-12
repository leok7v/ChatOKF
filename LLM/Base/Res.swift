import Foundation

enum Res {

    static func url(_ name: String, _ ext: String?, dev devDir: URL) -> URL? {
        let full = ext.map { e in "\(name).\(e)" } ?? name
        var found = Bundle.main.url(forResource: name, withExtension: ext)
        if found == nil, let exe = Bundle.main.executableURL {
            let candidate = exe.deletingLastPathComponent()
                .appendingPathComponent(full)
            found = exists(candidate) ? candidate : nil
        }
        if found == nil {
            let candidate = devDir.appendingPathComponent(full)
            found = exists(candidate) ? candidate : nil
        }
        return found
    }

    private static func exists(_ u: URL) -> Bool {
        FileManager.default.fileExists(atPath: u.path)
    }
}
