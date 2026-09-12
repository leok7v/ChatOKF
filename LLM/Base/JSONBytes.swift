import Foundation

public enum JSONBytes {

    public static func reproducible<T: Encodable>(_ value: T) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = .sortedKeys
        return try e.encode(value)
    }
}
