import Foundation
import LLM
import MD

public enum Instrument {

    public static let launched = Date()

    private static func stampLaunch() { _ = launched }

    public static func install() -> Bool {
        stampLaunch()
        Diag.shared.report(.load, "[args] \(Flags.given)")
        MarkdownDiag.report = { s in Diag.shared.report(.perf, s) }
        MainQueueWatch.shared.start()
        Footprint.watch()
        Footprint.watchPressure()
        Diag.memory = { tag in Footprint.report(.load, tag) }
        Diag.memoryDetail = { tag in Footprint.report(.memory, tag) }
        return true
    }

    @discardableResult
    public static func timed<T>(_ label: @autoclosure () -> String,
                            over: Double = 12, _ body: () -> T) -> T {
        let t0 = DispatchTime.now()
        let result = body()
        let ms = Double(DispatchTime.now().uptimeNanoseconds
            - t0.uptimeNanoseconds) / 1e6
        if ms >= over {
            Diag.shared.report(.perf,
                               String(format: "[app] %.1fms %@",
                                      ms, label()))
        }
        return result
    }

    @MainActor private static var beats: [String: (n: Int, at: Date)] = [:]

    public static func note(_ gate: DiagGate, _ text: String) {
        Diag.shared.report(gate, text)
    }

    @MainActor public static func beat(_ label: String) {
        if DiagGate.perf.on {
            var b = beats[label] ?? (0, Date())
            b.n += 1
            let dt = Date().timeIntervalSince(b.at)
            if dt >= 1.0 {
                Diag.shared.report(.perf, "[beat] \(label): \(b.n) "
                    + "evals in " + String(format: "%.1fs", dt))
                b = (0, Date())
            }
            beats[label] = b
        }
    }
}

final class MainQueueWatch: @unchecked Sendable {

    static let shared = MainQueueWatch()

    private static let thresholdMs = 200.0
    private let queue = DispatchQueue(label: "io.github.leok7v.ChatOKF.watch")
    private var timer: DispatchSourceTimer?
    private var started = false

    func start() {
        queue.async { [weak self] in
            if let self, !self.started {
                self.started = true
                let t = DispatchSource.makeTimerSource(queue: self.queue)
                t.schedule(deadline: .now() + .milliseconds(100),
                           repeating: .milliseconds(100))
                t.setEventHandler { [weak self] in self?.ping() }
                self.timer = t
                t.resume()
            }
        }
    }

    private var drained = DispatchTime.now().uptimeNanoseconds
    private var mainThread: mach_port_t = 0
    private var mainCPUms = 0.0

    private func ping() {
        let sent = DispatchTime.now()
        DispatchQueue.main.async { [weak self] in
            let now = DispatchTime.now()
            let ms = Double(now.uptimeNanoseconds
                - sent.uptimeNanoseconds) / 1e6
            if let self {
                if self.mainThread == 0 {
                    self.mainThread = mach_thread_self()
                }
                let cpu = MainQueueWatch.cpuMs(self.mainThread)
                let burned = cpu - self.mainCPUms
                self.mainCPUms = cpu
                if ms >= MainQueueWatch.thresholdMs,
                   sent.uptimeNanoseconds >= self.drained {
                    self.drained = now.uptimeNanoseconds
                    if DiagGate.perf.on {
                        Diag.shared.report(.perf, String(format:
                            "[hang] main queue starved %.0fms, main thread "
                            + "used %.0fms CPU", ms, burned))
                    }
                }
            }
        }
    }

    private static func cpuMs(_ thread: mach_port_t) -> Double {
        var info = thread_basic_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<thread_basic_info_data_t>.size
                / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self,
                                capacity: Int(count)) { raw in
                thread_info(thread, thread_flavor_t(THREAD_BASIC_INFO),
                            raw, &count)
            }
        }
        var ms = 0.0
        if kr == KERN_SUCCESS {
            let user_s    = info.user_time.seconds
            let user_us   = info.user_time.microseconds
            let system_s  = info.system_time.seconds
            let system_us = info.system_time.microseconds
            ms = Double(user_s + system_s) * 1000 +
                 Double(user_us + system_us) / 1000
        }
        return ms
    }

}
