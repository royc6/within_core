import CoreAI
import CoreAILanguageModels
import Foundation
import WithinCoreBase

/// Only run() touches backing arrays. Its locked claim covers the entire await,
/// so @unchecked Sendable does not expose concurrently mutable NDArray storage.
/// This also keeps Core AI's non-Sendable views outside actor-isolated regions.
final class Gemma4StateStorage: @unchecked Sendable {
    private let backing: FixedNDArrayState
    private let lock = NSLock()
    private var running = false

    init(descriptors: [(name: String, descriptor: NDArrayDescriptor)]) {
        backing = FixedNDArrayState(states: descriptors)
    }

    /// The array bound for the index'th name, for reading after run().
    func array(at index: Int) -> NDArray { backing[stateIndex: index].array }

    /// Runs `function` with the arrays bound as its output views, so Core AI writes into
    /// them rather than allocating new outputs on every call.
    func run(function: InferenceFunction, inputs: [String: NDArray]) async throws -> InferenceFunction.Outputs {
        let claimed = lock.withLock {
            guard !running else { return false }
            running = true
            return true
        }
        guard claimed else { throw Gemma4RuntimeError.busy }
        defer { lock.withLock { running = false } }
        // A local strong owner gives bind's borrowed views a lifetime spanning
        // await; borrowing a stored property directly ends at that expression.
        let owner = backing
        defer { withExtendedLifetime(owner) {} }
        var views = InferenceFunction.MutableViews()
        owner.bind(into: &views)
        return try await function.run(inputs: inputs, outputViews: borrowStatesAcrossRun(consume views))
    }
}

// Same lifetime bridge as Apple's CoreAILanguageModels StateHandler.swift.
// Copyright 2026 Apple Inc. BSD 3-Clause License; see THIRD_PARTY_NOTICES.md.
// Its helper is internal to that module. The storage owner remains alive across
// run, serializes inference, and never replaces its arrays. Cancellation
// waits for the producer to finish before the engine releases this runtime.
@inline(__always)
@_unsafeNonescapableResult
@_lifetime(immortal)
func borrowStatesAcrossRun(_ views: consuming InferenceFunction.MutableViews) -> InferenceFunction.MutableViews {
    views
}
