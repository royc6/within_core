import Foundation
import Synchronization
import UIKit
import WithinCore
import WithinCoreBase
import os

/// Runs the demo without taps, for measuring it from a Mac. Once the package is installed, `-benchmark` runs
/// the performance procedure and `-chat "<prompt>"` sends one message (`-max-tokens <n>` caps the reply; 1024 by
/// default, as in Chat). The screen stays on meanwhile. Each writes `Documents/benchmark.json` or
/// `Documents/chat.json` after every step, with the thermal state at each step and the engine's events.
enum LaunchCommand {
    case benchmark
    case chat(prompt: String, maxTokens: Int)

    /// The command in this launch's arguments, if any.
    static let requested: LaunchCommand? = {
        let arguments = ProcessInfo.processInfo.arguments
        func value(after flag: String) -> String? {
            guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
            return arguments[index + 1]
        }
        if arguments.contains("-benchmark") { return .benchmark }
        guard let prompt = value(after: "-chat") else { return nil }
        return .chat(prompt: prompt, maxTokens: value(after: "-max-tokens").flatMap { Int($0) } ?? 1024)
    }()

    @MainActor func run(store: ModelStore, holder: EngineHolder) async {
        let name =
            switch self {
            case .benchmark: "benchmark"
            case .chat: "chat"
            }
        let report = Report(
            url: URL.documentsDirectory.appending(path: name + ".json"), revision: store.manifest.revision)
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        do {
            guard store.state == .installed else { throw CocoaError(.fileNoSuchFile) }
            let events = report.events
            holder.events = { events.record($0, $1) }
            report.step("Loading the model")
            let engine = try await holder.engine(folder: store.folder)
            switch self {
            case .benchmark:
                let result = try await Benchmark.run(engine: engine) { report.step($0) }
                report.finish([
                    "medianTokensPerSecond": result.medianTokensPerSecond,
                    "medianFirstTokenMs": result.medianFirstTokenMs, "peakFootprintBytes": result.peakFootprintBytes,
                    "runs": result.runs.map {
                        [
                            "tokensPerSecond": $0.tokensPerSecond, "firstTokenMs": $0.firstTokenMs,
                            "thermalWaitSeconds": $0.thermalWaitSeconds,
                        ]
                    },
                ])
            case .chat(let prompt, let maxTokens):
                report.step("Sending")
                let options = Gemma4Engine.Options(
                    temperature: 0.7, topK: 64, maxOutputTokens: maxTokens, thinking: false)
                let text = try await reply(from: engine, turns: [ChatTurn(role: .user, text: prompt)], options: options)
                let timings = await engine.lastTimings
                report.finish([
                    "prompt": prompt, "maxTokens": maxTokens, "reply": text,
                    "inputTokens": timings?.inputTokens ?? 0, "tokenMs": timings?.tokenMs ?? [],
                ])
            }
        } catch {
            report.fail(error)
        }
    }
}

/// A launch command's result file, written again after every step so that a run that stops early still
/// leaves what it measured.
@MainActor private final class Report {
    let events = EventLog()
    private let url: URL
    private var fields: [String: Any]
    private var steps: [[String: Any]] = []

    init(url: URL, revision: String) {
        self.url = url
        var system = utsname()
        uname(&system)
        let device = withUnsafeBytes(of: system.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        #if DEBUG
        let build = "Debug"
        #else
        let build = "Release"
        #endif
        fields = [
            "revision": revision, "device": device, "build": build, "state": "running",
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    func step(_ status: String) {
        let thermal =
            switch ProcessInfo.processInfo.thermalState {
            case .nominal: "nominal"
            case .fair: "fair"
            case .serious: "serious"
            case .critical: "critical"
            @unknown default: "unknown"
            }
        steps.append(["status": status, "time": Date().timeIntervalSince1970, "thermalState": thermal])
        fields["steps"] = steps
        fields["events"] = events.all
        guard let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]) else { return }
        try? data.write(to: url, options: .atomic)
    }

    func finish(_ result: [String: Any]) {
        fields["result"] = result
        fields["state"] = "complete"
        step("Finished")
    }

    func fail(_ error: any Error) {
        fields["error"] = String(describing: error)
        fields["state"] = "failed"
        step("Failed")
    }
}

/// The engine's events, each with the time, the process footprint and the memory still available to it. They
/// stay in memory so that no file write lands inside a measured span.
private final class EventLog: Sendable {
    private struct Event: Sendable {
        let stage: String, fields: [String: String], time: Double, footprintBytes: UInt64, availableBytes: Int
    }
    private let events = Mutex<[Event]>([])

    func record(_ stage: String, _ fields: [String: String]) {
        let event = Event(
            stage: stage, fields: fields, time: Date().timeIntervalSince1970, footprintBytes: Benchmark.footprint(),
            availableBytes: os_proc_available_memory())
        events.withLock { $0.append(event) }
    }

    var all: [[String: Any]] {
        events.withLock { $0 }.map {
            [
                "stage": $0.stage, "fields": $0.fields, "time": $0.time, "footprintBytes": $0.footprintBytes,
                "availableBytes": $0.availableBytes,
            ]
        }
    }
}
