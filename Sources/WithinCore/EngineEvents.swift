import Foundation

/// Receives the engine's lifecycle and timing events: a stage name and string fields. The engine
/// stores nothing itself; an app can log, sample or drop them.
public typealias EngineEventHandler = @Sendable (_ stage: String, _ fields: [String: String]) -> Void

func milliseconds(_ duration: Duration) -> String {
    String(format: "%.3f", duration.milliseconds)
}

extension Duration {
    var milliseconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) * 1000 + Double(attoseconds) / 1e15
    }
}
