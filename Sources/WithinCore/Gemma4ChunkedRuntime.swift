import CoreAI
import CoreAILanguageModels
import Foundation
import WithinCoreBase

/// The chunked decoder. Each chunk is its own asset; one call runs them in
/// order, passing hidden forward. Headless prefill runs only the KV-owner chunks.
/// Every owner layer's cache is held here, and consumer chunks get the donors'
/// cache plus fresh K/V.
///
/// Timing is measured in memory and written once per call, so the
/// diagnostics file sync never lands inside a measured span.
public actor Gemma4ChunkedRuntime {
    private struct Loaded {
        let spec: Gemma4ChunkContract.Chunk
        let functions: [String: InferenceFunction]
        let decodeOutputs: Gemma4StateStorage?  // step_1's outputs, reused by every token
    }
    private let configuration: Gemma4RuntimeConfiguration
    private let chunks: [Loaded]
    private let models: [AIModel]
    private let embeddings: Gemma4Embeddings
    private let perLayerEmbeddings: Gemma4Embeddings
    private let ring: Bool  // sliding-attention caches are rings of the window
    private var caches: [String: NDArray] = [:]
    private var position = 0
    private var working = false
    private var cancelled = false
    private let events: EngineEventHandler?

    /// Loads every chunk with Neural Engine specialization, prepares its functions, checks their
    /// inputs and outputs against the contract, and allocates zeroed host caches. A specialization
    /// cached by a different build can make preparation fail; that chunk's cache entries are dropped
    /// and it is compiled once more.
    public init(
        folder: URL, configuration c: Gemma4RuntimeConfiguration,
        contract: Gemma4ChunkContract, requireE4B: Bool, events: EngineEventHandler? = nil
    ) async throws {
        try contract.validate(configuration: c, requireE4B: requireE4B)
        configuration = c
        self.events = events
        ring = contract.slidingCacheLength != nil
        embeddings = try Gemma4Embeddings(folder: folder, descriptor: c.embeddings)
        perLayerEmbeddings = try Gemma4Embeddings(folder: folder, descriptor: c.perLayerEmbeddings)
        var loaded: [Loaded] = []
        var models: [AIModel] = []
        let clock = ContinuousClock()
        for (index, spec) in contract.chunks.enumerated() {
            try Task.checkCancellation()
            let chunk = String(index)
            events?(
                "model.specialize.begin",
                [
                    "kv": "chunked", "chunk": chunk,
                    "layers": "\(spec.start)..<\(spec.end)",
                ])
            let url = folder.appending(path: spec.file)
            var specialized: AIModel?
            var functions: [String: InferenceFunction] = [:]
            var retried = false
            // A specialization cached by a different build of the app can make loadFunction throw, as an
            // app update would for users. Drop this chunk's cache entries and compile it again, once.
            while specialized == nil {
                do {
                    let began = clock.now
                    let model: AIModel
                    do {
                        model = try await AIModel(
                            contentsOf: url, options: .init(preferredComputeUnitKind: .neuralEngine))
                    } catch {
                        events?(
                            "model.specialize.failed",
                            [
                                "chunk": chunk,
                                "detail": String(String(reflecting: error).prefix(2048)),
                            ])
                        throw error
                    }
                    events?(
                        "model.specialize.complete",
                        [
                            "chunk": chunk,
                            "ms": milliseconds(clock.now - began),
                        ])
                    functions = [:]
                    for (name, expected) in spec.functions.sorted(by: { $0.key < $1.key }) {
                        try Task.checkCancellation()
                        events?("function.prepare.begin", ["chunk": chunk, "function": name])
                        let prepared = clock.now
                        guard let function = try model.loadFunction(named: name) else {
                            throw Gemma4RuntimeError.invalidPackage
                        }
                        let descriptor = function.descriptor
                        guard Set(descriptor.inputNames) == Set(expected.inputs),
                            Set(descriptor.outputNames) == Set(expected.outputs),
                            descriptor.stateNames.isEmpty
                        else { throw Gemma4RuntimeError.invalidPackage }
                        functions[name] = function
                        events?(
                            "function.prepare.complete",
                            [
                                "chunk": chunk, "function": name,
                                "ms": milliseconds(clock.now - prepared),
                            ])
                    }
                    specialized = model
                } catch Gemma4RuntimeError.invalidPackage {
                    throw Gemma4RuntimeError.invalidPackage
                } catch  where !retried && !(error is CancellationError) {
                    retried = true
                    events?(
                        "model.cache.retry",
                        [
                            "chunk": chunk,
                            "detail": String(String(reflecting: error).prefix(512)),
                        ])
                    try? AIModelCache.default.deleteEntries(for: url)
                }
            }
            guard let model = specialized else { throw Gemma4RuntimeError.invalidPackage }
            // Decode writes into one reused set of step_1 outputs: fresh outputs per token are freed
            // only every few tokens and raise peak memory above the memory guard's floor.
            var decodeOutputs: Gemma4StateStorage?
            if let function = functions["step_1"], let expected = spec.functions["step_1"] {
                decodeOutputs = Gemma4StateStorage(
                    descriptors: try expected.outputs.map { name in
                        guard case .ndArray(let descriptor) = function.descriptor.outputDescriptor(of: name)
                        else { throw Gemma4RuntimeError.invalidPackage }
                        return (name, descriptor)
                    })
            }
            loaded.append(Loaded(spec: spec, functions: functions, decodeOutputs: decodeOutputs))
            models.append(model)
        }
        try Task.checkCancellation()
        chunks = loaded
        self.models = models
        // Zeros, not garbage: a masked slot still multiplies its value by 0. Allocated and cleared in
        // place, because building them from [Float16] arrays dominates a reload.
        for layer in c.cacheLayers {
            let shape = [
                1, c.keyValueHeads, c.headDimension(layer: layer), contract.cacheLength(layer: layer, configuration: c),
            ]
            for name in Gemma4ChunkContract.cacheNames(layer) {
                var cache = NDArray(shape: shape, scalarType: .float16)
                guard try Self.withContiguousBytes(of: &cache, { _ = memset($0, 0, $1) }) else {
                    throw Gemma4RuntimeError.invalidPackage
                }
                caches[name] = cache
            }
        }
    }

    /// Starts a new sequence at position 0. The masks hide every stale slot until it is overwritten.
    public func reset() throws {
        guard !working else { throw Gemma4RuntimeError.busy }
        position = 0  // Masks hide every stale slot until it is overwritten.
        cancelled = false
    }

    /// Makes the call in progress stop after its current chunk.
    public func cancel() { cancelled = true }

    /// Writes the host KV caches (sorted by name, contiguous FP16) to `url`, one cache at a time, and
    /// returns the position and byte count; nil when there is nothing to keep. A conversation can then
    /// outlive a model release: reading the caches back is far faster than prefilling it again.
    public func saveKV(to url: URL) throws -> (position: Int, bytes: Int)? {
        guard !working, position > 0 else { return nil }
        let file = Darwin.open(url.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        guard file >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(file) }
        var total = 0
        for name in caches.keys.sorted() {
            guard var cache = caches.removeValue(forKey: name) else { throw Gemma4RuntimeError.invalidOutput }
            defer { caches[name] = cache }
            var written = false
            guard
                try Self.withContiguousBytes(
                    of: &cache,
                    { base, bytes in
                        written = Self.transfer(bytes) { Darwin.write(file, base + $0, $1) }
                        total += bytes
                    }), written
            else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        return (position, total)
    }

    /// Reads back what saveKV wrote straight into the caches allocated at load. Reading through
    /// FileHandle keeps its Data buffers alive past the read, which puts the first decode steps after
    /// a restore over the memory guard's floor. Until every cache is back
    /// the position stays 0, so after a failure the masks hide whatever was restored and the prompt prefills in full.
    public func restoreKV(from url: URL, position: Int) throws {
        guard !working, (1...configuration.contextLimit).contains(position) else {
            throw Gemma4RuntimeError.invalidPackage
        }
        let file = Darwin.open(url.path, O_RDONLY)
        guard file >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(file) }
        self.position = 0
        for name in caches.keys.sorted() {
            guard var cache = caches.removeValue(forKey: name) else { throw Gemma4RuntimeError.invalidPackage }
            defer { caches[name] = cache }
            var filled = false
            guard
                try Self.withContiguousBytes(
                    of: &cache,
                    { base, bytes in
                        filled = Self.transfer(bytes) { Darwin.read(file, base + $0, $1) }
                    }), filled
            else { throw Gemma4RuntimeError.invalidPackage }
        }
        var extra: UInt8 = 0
        guard Darwin.read(file, &extra, 1) == 0 else { throw Gemma4RuntimeError.invalidPackage }
        self.position = position
        cancelled = false
    }

    /// Feeds `tokens` at the current position and returns the logits after the last one. Prompts
    /// run as headless prefill calls on the KV-owner chunks, then `step_1` on the held-back token.
    /// After a failure the runtime needs `reset()`.
    public func logits(for tokens: [Int]) async throws -> [Float16] {
        guard !working else { throw Gemma4RuntimeError.busy }
        guard position >= 0 else { throw Gemma4RuntimeError.resetRequired }
        guard tokens.allSatisfy({ (0..<configuration.vocabularySize).contains($0) }) else {
            throw Gemma4RuntimeError.invalidToken
        }
        guard !tokens.isEmpty, tokens.count <= configuration.contextLimit - position else {
            throw Gemma4RuntimeError.contextExceeded
        }
        working = true
        defer { working = false }
        do {
            var last: [Float16] = []
            let plan = try Gemma4PrefillPlan(tokenCount: tokens.count, prefillLength: configuration.prefillLength)
            for chunk in plan.chunks {
                try Task.checkCancellation()
                guard !cancelled else { throw CancellationError() }
                if let scores = try await step(Array(tokens[chunk.tokens]), plan: chunk) { last = scores }
            }
            return last
        } catch {
            position = -1  // A failing native call may already have mutated some states.
            throw error
        }
    }

    private func step(_ tokens: [Int], plan: Gemma4PrefillPlan.Chunk) async throws -> [Float16]? {
        let c = configuration, length = tokens.count, clock = ContinuousClock()
        let began = clock.now
        try c.validatePosition(step: position, length: length)
        let local = try c.rotary(step: position, length: length, global: false)
        let global = try c.rotary(step: position, length: length, global: true)
        let layers = c.layerTypes.count
        let identity = try perLayerEmbeddings.gather(
            tokens: tokens, layers: layers, scale: sqrt(Float(c.perLayerInputSize)))
        var common: [String: NDArray] = [
            "embedded": NDArray(
                scalars: try embeddings.gather(tokens: tokens, scale: sqrt(Float(c.hiddenSize))),
                shape: [1, c.hiddenSize, 1, length]),
            "local_cos": NDArray(scalars: local.cos, shape: [1, c.localHeadDimension, 1, length]),
            "local_sin": NDArray(scalars: local.sin, shape: [1, c.localHeadDimension, 1, length]),
            "global_cos": NDArray(scalars: global.cos, shape: [1, c.globalHeadDimension, 1, length]),
            "global_sin": NDArray(scalars: global.sin, shape: [1, c.globalHeadDimension, 1, length]),
        ]
        // Old slots at their positions (or ring slots), then this call's fresh K/V: [1, 1, L, slots + L].
        common["local_mask"] =
            ring
            ? NDArray(
                scalars: try c.ringMask(step: position, length: length),
                shape: [1, 1, length, c.slidingWindow + length])
            : NDArray(
                scalars: try c.mask(step: position, length: length, global: false),
                shape: [1, 1, length, c.contextLimit + length])
        common["global_mask"] = NDArray(
            scalars: try c.mask(step: position, length: length, global: true),
            shape: [1, 1, length, c.contextLimit + length])
        let inputsReady = clock.now
        // Packages have no prefill_1: a 1-token prefill tail runs step_1
        // on the owner chunks, and nothing reads the hidden it passes on.
        let name = plan.returnsLogits || plan.function == "prefill_1" ? "step_1" : plan.function
        var chunkTimes: [String] = []
        var logits: [NDArray] = []  // one array, or a head chunk's row slices in contract order
        var fresh: [String: NDArray] = [:]
        for loaded in chunks {
            if !plan.returnsLogits && (loaded.spec.owned ?? []).isEmpty { break }
            guard let expected = loaded.spec.functions[name] else {
                if plan.returnsLogits { throw Gemma4RuntimeError.invalidPackage }
                break  // Headless prefill stops after the last KV-owner chunk.
            }
            guard let function = loaded.functions[name] else { throw Gemma4RuntimeError.invalidPackage }
            var inputs: [String: NDArray] = [:]
            for input in expected.inputs {
                if input == "token_identity" {
                    let range = loaded.spec.start..<loaded.spec.end
                    inputs[input] = NDArray(
                        scalars: Gemma4Embeddings.sliceLayers(
                            identity, layers: layers, length: length,
                            range: range), shape: [1, c.perLayerInputSize, range.count, length])
                } else {
                    guard let value = common[input] ?? caches[input] ?? fresh[input] else {
                        throw Gemma4RuntimeError.invalidPackage
                    }
                    inputs[input] = value
                }
            }
            let started = clock.now
            let reused = name == "step_1" ? loaded.decodeOutputs : nil
            var outputs: InferenceFunction.Outputs
            if let reused {
                outputs = try await reused.run(function: function, inputs: inputs)
            } else {
                outputs = try await function.run(inputs: inputs)
            }
            chunkTimes.append(milliseconds(clock.now - started))
            try Task.checkCancellation()
            guard !cancelled else { throw CancellationError() }
            for (index, output) in expected.outputs.enumerated() {
                guard let value = reused?.array(at: index) ?? outputs.remove(output)?.ndArray
                else { throw Gemma4RuntimeError.invalidOutput }
                if output == "logits" || output.hasPrefix("logits_") {
                    logits.append(value)
                } else if output.hasPrefix("fresh_") {
                    fresh[output] = value
                } else {
                    common[output] = value
                }
            }
        }
        let computed = clock.now
        // Drop every other reference first so each cache mutates in place, not copy-on-write.
        common.removeAll()
        try write(fresh, length: length)
        let written = clock.now
        var scores: [Float16]?
        if plan.returnsLogits {
            var values: [Float16] = []
            values.reserveCapacity(c.vocabularySize)
            for part in logits {
                guard part.shape.count == 4, part.shape[0] == 1, part.shape[2] == 1, part.shape[3] == length,
                    part.scalarType == .float16
                else { throw Gemma4RuntimeError.invalidOutput }
                part.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
                    for row in 0..<shape[1] { values.append(pointer[row * strides[1] + (length - 1) * strides[3]]) }
                }
            }
            guard values.count == c.vocabularySize, values.allSatisfy(\.isFinite) else {
                throw Gemma4RuntimeError.invalidOutput
            }
            scores = values
        }
        let finished = clock.now
        position += length
        // "function" is the scheduled call (prefill_1 stays prefill_1); "graph" is what ran.
        events?(
            "inference.timing",
            [
                "function": plan.returnsLogits ? "step_1" : plan.function,
                "graph": name, "length": String(length),
                "position": String(position), "inputMs": milliseconds(inputsReady - began),
                "chunkMs": chunkTimes.joined(separator: ","), "writeMs": milliseconds(written - computed),
                "readbackMs": milliseconds(finished - written),
                "totalMs": milliseconds(finished - began),
            ])
        return scores
    }

    /// Runs `body` over a contiguous FP16 array's bytes; false when the layout is strided.
    private static func withContiguousBytes(
        of array: inout NDArray,
        _ body: (UnsafeMutableRawPointer, Int) throws -> Void
    ) throws -> Bool {
        var result: Result<Bool, any Error> = .success(false)
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { base, shape, strides in
            var count = 1, contiguous = true
            for axis in (0..<shape.count).reversed() {
                contiguous = contiguous && strides[axis] == count; count *= shape[axis]
            }
            guard contiguous else { return }
            result = Result {
                try body(UnsafeMutableRawPointer(base), count * MemoryLayout<Float16>.size); return true
            }
        }
        return try result.get()
    }

    /// Moves `bytes` through `step(offset, remaining)` (read or write), retrying short and interrupted calls.
    private static func transfer(_ bytes: Int, _ step: (Int, Int) -> Int) -> Bool {
        var done = 0
        while done < bytes {
            let moved = step(done, bytes - done)
            if moved < 0 && errno == EINTR { continue }
            guard moved > 0 else { return false }
            done += moved
        }
        return true
    }

    /// Fresh K/V [1, Hkv, D, L] into each owner cache at [position, position + L),
    /// modulo the slot count for a sliding-attention ring.
    private func write(_ fresh: [String: NDArray], length: Int) throws {
        for (output, array) in fresh {
            let name = String(output.dropFirst("fresh_".count))
            guard var cache = caches.removeValue(forKey: name), array.scalarType == .float16,
                array.shape == [cache.shape[0], cache.shape[1], cache.shape[2], length]
            else { throw Gemma4RuntimeError.invalidOutput }
            var finite = true
            let slots = cache.shape[3], start = position
            array.view(as: Float16.self).withUnsafePointer { source, shape, from in
                cache.mutableView(as: Float16.self).withUnsafeMutablePointer { target, _, to in
                    for head in 0..<shape[1] {
                        for dimension in 0..<shape[2] {
                            for token in 0..<length {
                                let value = source[head * from[1] + dimension * from[2] + token * from[3]]
                                finite = finite && value.isFinite
                                target[head * to[1] + dimension * to[2] + ((start + token) % slots) * to[3]] = value
                            }
                        }
                    }
                }
            }
            caches[name] = cache
            guard finite else { throw Gemma4RuntimeError.invalidOutput }
        }
    }
}
