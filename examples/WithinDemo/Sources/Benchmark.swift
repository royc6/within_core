import Darwin
import Foundation
import WithinCore
import WithinCoreBase

/// The performance procedure behind the report's speed numbers: three 64-token warmups, then five
/// 512-token runs, each a new conversation with the same prompt (a clean KV cache), generating
/// exactly the stated tokens with stop tokens suppressed. A run starts once the phone's thermal state
/// is nominal. Speed of a run is 1000 / the median interval before tokens 33...512; first token is
/// request start to the first sampled token.
enum Benchmark {
    static let prompt =
        "Explain how a lighthouse works, from its lamp and lens to the way its light helps ships navigate safely at night."
    static let warmups = (count: 3, tokens: 64)
    static let runs = (count: 5, tokens: 512)
    static let measured = 33...512

    struct Run: Identifiable, Sendable {
        let id: Int
        let tokensPerSecond: Double
        let firstTokenMs: Double
        let thermalWaitSeconds: Double
    }

    struct Result: Sendable {
        let runs: [Run]
        let peakFootprintBytes: UInt64
        var medianTokensPerSecond: Double { median(runs.map(\.tokensPerSecond)) }
        var medianFirstTokenMs: Double { median(runs.map(\.firstTokenMs)) }
    }

    /// Runs the procedure; `onStatus` describes the step in progress.
    static func run(engine: Gemma4Engine, onStatus: @MainActor (String) -> Void) async throws -> Result {
        let turns = [ChatTurn(role: .user, text: prompt)]
        func options(_ tokens: Int) -> Gemma4Engine.Options {
            .init(
                temperature: 0.7, topK: 64, topP: 0.95, maxOutputTokens: tokens, thinking: false,
                exactOutputTokens: tokens)
        }
        var peak: UInt64 = 0
        for index in 1...warmups.count {
            await onStatus("Warmup \(index) of \(warmups.count)")
            _ = try await reply(from: engine, turns: turns, options: options(warmups.tokens))
            peak = max(peak, footprint())
        }
        var runs: [Run] = []
        for index in 1...self.runs.count {
            await onStatus("Waiting for the phone to cool down")
            let waited = try await waitForNominalThermalState()
            await onStatus("Run \(index) of \(self.runs.count)")
            _ = try await reply(from: engine, turns: turns, options: options(self.runs.tokens))
            peak = max(peak, footprint())
            guard let timings = await engine.lastTimings, timings.tokenMs.count == self.runs.tokens else {
                throw CocoaError(.coderInvalidValue)
            }
            runs.append(
                Run(
                    id: index, tokensPerSecond: 1000 / median(timings.intervals(measured)),
                    firstTokenMs: timings.tokenMs[0], thermalWaitSeconds: waited))
        }
        return Result(runs: runs, peakFootprintBytes: peak)
    }

    /// Waits until the thermal state is nominal, at most 15 minutes; returns the seconds waited.
    private static func waitForNominalThermalState() async throws -> Double {
        let start = ContinuousClock.now
        while ProcessInfo.processInfo.thermalState != .nominal, ContinuousClock.now - start < .seconds(900) {
            try await Task.sleep(for: .seconds(10))
        }
        let elapsed = (ContinuousClock.now - start).components
        return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    /// The process's physical footprint, the figure the system's memory limit applies to.
    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
