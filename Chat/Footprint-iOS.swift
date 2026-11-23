import Foundation

extension Footprint {

    static func availableBytes() -> UInt64? {
        Footprint.headroom(os_proc_available_memory())
    }

}
