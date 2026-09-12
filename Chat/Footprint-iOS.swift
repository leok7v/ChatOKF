import Foundation

extension Footprint {

    static func availableBytes() -> UInt64? {
        UInt64(os_proc_available_memory())
    }

}
