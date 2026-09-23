import Foundation
import Metal

final class MetalStopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false
    func raise() { lock.lock(); raised = true; lock.unlock() }
    func clear() { lock.lock(); raised = false; lock.unlock() }
    var raisedNow: Bool {
        lock.lock()
        defer { lock.unlock() }
        return raised
    }
}

public final class BackgroundGate: @unchecked Sendable {
    public static let shared = BackgroundGate()
    private let lock = NSLock()
    private var backgrounded = false

    public func setBackgrounded(_ v: Bool) {
        lock.lock(); backgrounded = v; lock.unlock()
    }

    private var isBackgrounded: Bool {
        lock.lock(); defer { lock.unlock() }; return backgrounded
    }

    public var parked: Bool { isBackgrounded }

    // Parks the calling thread while backgrounded so no GPU submit fires there;
    // returns at once in the foreground, and always on macOS.
    func waitForForeground() {
        while isBackgrounded { Thread.sleep(forTimeInterval: 1) }
    }
}

public struct GPUFault: Error, CustomStringConvertible {
    public let description: String
}

public final class GPUGate: @unchecked Sendable {
    public static let shared = GPUGate()
    public static let retries = 3
    static let criticalWait = 30.0
    private let lock = NSLock()
    private var submits = 0
    private var faultEvery = Flags.int("gpu-fault") ?? 0
    private var forced = GPUGate.parse(Flags.value("thermal") ?? "")

    static func parse(_ text: String) -> ProcessInfo.ThermalState? {
        var out: ProcessInfo.ThermalState? = nil
        switch text {
        case "nominal": out = .nominal
        case "fair": out = .fair
        case "serious": out = .serious
        case "critical": out = .critical
        default: out = nil
        }
        return out
    }

    static func label(_ state: ProcessInfo.ThermalState) -> String {
        let out: String
        switch state {
        case .nominal: out = "nominal"
        case .fair: out = "fair"
        case .serious: out = "serious"
        case .critical: out = "critical"
        @unknown default: out = "unknown"
        }
        return out
    }

    public var thermal: ProcessInfo.ThermalState {
        lock.lock()
        defer { lock.unlock() }
        return forced ?? ProcessInfo.processInfo.thermalState
    }

    public var hot: Bool {
        let state = thermal
        return state == .serious || state == .critical
    }

    public func simulate(thermal: ProcessInfo.ThermalState?, faultEvery: Int) {
        lock.lock()
        forced = thermal
        self.faultEvery = faultEvery
        submits = 0
        lock.unlock()
    }

    public func chunkCap(_ wanted: Int) -> Int {
        let out: Int
        switch thermal {
        case .critical: out = min(wanted, 32)
        case .serious: out = min(wanted, 128)
        default: out = wanted
        }
        return out
    }

    func pace() {
        BackgroundGate.shared.waitForForeground()
        var waited = 0.0
        while thermal == .critical && waited < GPUGate.criticalWait {
            Thread.sleep(forTimeInterval: 0.25)
            waited += 0.25
        }
        if hot { Thread.sleep(forTimeInterval: 0.02) }
    }

    func verdict(_ buffers: [MTLCommandBuffer], _ tag: String) -> Bool {
        lock.lock()
        submits += 1
        let staged = faultEvery > 0 && submits % faultEvery == 0
        lock.unlock()
        let fault = buffers.compactMap { cb in cb.error }.first
        let ok = fault == nil && !staged
        if !ok {
            Diag.shared.report("[gpu] \(tag) failed, thermal "
                + GPUGate.label(thermal) + ": "
                + (fault.map { err in "\(err)" } ?? "staged fault"))
        }
        return ok
    }

    func submit(_ cb: MTLCommandBuffer, _ tag: String) -> Bool {
        pace()
        cb.commit()
        cb.waitUntilCompleted()
        return verdict([cb], tag)
    }

    func backoff(_ attempt: Int) {
        Thread.sleep(forTimeInterval: 0.1 * pow(2, Double(attempt)))
    }

    func run(_ tag: String, _ encode: () -> MTLCommandBuffer) -> Bool {
        var ok = false
        var attempt = 0
        while !ok && attempt <= GPUGate.retries {
            if attempt > 0 { backoff(attempt - 1) }
            ok = submit(encode(), tag)
            attempt += 1
        }
        return ok
    }
}
