import Foundation
import Observation
import WithinCore
import WithinCoreBase

/// One engine for the app, loaded on first use. The first load after installing compiles every chunk
/// for the Neural Engine, which takes minutes; later loads come from the system's model cache.
@MainActor @Observable final class EngineHolder {
    private(set) var isLoading = false
    /// Receives the engine's load and timing events from the next load on; the launch commands record them.
    var events: EngineEventHandler?
    private var engine: Gemma4Engine?
    private var loading: Task<Gemma4Engine, any Error>?

    func engine(folder: URL) async throws -> Gemma4Engine {
        if let engine { return engine }
        if let loading { return try await loading.value }
        isLoading = true
        let events = events
        let task = Task { try await Gemma4Engine(folder: folder, events: events) }
        loading = task
        defer { isLoading = false; loading = nil }
        let loaded = try await task.value
        engine = loaded
        return loaded
    }

    /// Releases the model's memory; the next request loads it again.
    func release() {
        loading?.cancel()
        engine = nil
    }
}

/// Runs one generation and returns its final text, passing every partial reply to `onText`.
func reply(
    from engine: Gemma4Engine, turns: [ChatTurn], options: Gemma4Engine.Options,
    onText: @MainActor (String) -> Void = { _ in }
) async throws -> String {
    let (events, continuation) = AsyncStream<Gemma4Engine.Event>.makeStream()
    let generation = Task {
        defer { continuation.finish() }
        try await engine.generate(system: "", turns: turns, options: options) { continuation.yield($0) }
    }
    return try await withTaskCancellationHandler {
        var text = ""
        for await event in events {
            if case .text(let partial) = event {
                text = partial
                await onText(partial)
            }
        }
        try await generation.value
        return text
    } onCancel: {
        generation.cancel()
    }
}
