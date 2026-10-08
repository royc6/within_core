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
    private let folder: URL
    private var chunks: [Loaded]
    private var models: [AIModel?]  // nil while a chunk is compiled again
    private let embeddings: Gemma4Embeddings
    private let perLayerEmbeddings: Gemma4Embeddings
    private let ring: Bool  // sliding-attention caches are rings of the window
    /// Whether old positions can be evicted and their slots reused (KV-SLIDE): the sliding-attention
    /// caches must be rings, which depend on the position only.
    public nonisolated let slides: Bool
    private var caches: [String: NDArray] = [:]
    /// Which position each full-attention cache slot holds; slot = position until something is evicted.
    private var table: Gemma4SlotTable
    /// The position the next token takes; -1 after a failure that needs `reset()`.
    private(set) var position = 0
    private var working = false
    private var cancelled = false
    /// Decode steps since the load or the last recompile, the chunk times of the sampled ones (nil once the
    /// check is over) and whether a recompile has been tried (Gemma4CompileCheck).
    private var decodeSteps = 0
    private var compileCheckSamples: [[Double]]? = []
    private var recompiled = false
    private let events: EngineEventHandler?

    /// Compiles one chunk for the Neural Engine ahead of its first load (COMPILE-ON-DOWNLOAD):
    /// the specialization is cached by content, so a chunk compiled where the download staged it is
    /// found when the installed package loads. The model is released on return. Returns the compile
    /// time (tens of seconds cold; milliseconds when the cache already holds it).
    public static func precompile(chunkAt url: URL) async throws -> Duration {
        let began = ContinuousClock.now
        _ = try await AIModel(contentsOf: url, options: .init(preferredComputeUnitKind: .neuralEngine))
        return ContinuousClock.now - began
    }

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
        self.folder = folder
        self.events = events
        ring = contract.slidingCacheLength != nil
        slides = contract.slidingCacheLength != nil
        table = Gemma4SlotTable(capacity: c.contextLimit)
        embeddings = try Gemma4Embeddings(folder: folder, descriptor: c.embeddings)
        perLayerEmbeddings = try Gemma4Embeddings(folder: folder, descriptor: c.perLayerEmbeddings)
        var loaded: [Loaded] = []
        var models: [AIModel?] = []
        let clock = ContinuousClock()
        for (index, spec) in contract.chunks.enumerated() {
            try Task.checkCancellation()
            let (model, functions) = try await Self.load(
                chunk: index, spec: spec, url: folder.appending(path: spec.file), clock: clock, events: events)
            loaded.append(
                Loaded(spec: spec, functions: functions, decodeOutputs: try Self.decodeOutputs(spec: spec, functions)))
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

    /// Specializes one chunk for the Neural Engine and prepares its functions. A specialization cached by a
    /// different build of the app can make loadFunction throw, as an app update would for users: that chunk's
    /// cache entries are dropped and it is compiled once more. `reason` names a compile after the load.
    private static func load(
        chunk index: Int, spec: Gemma4ChunkContract.Chunk, url: URL, reason: String? = nil,
        clock: ContinuousClock, events: EngineEventHandler?
    ) async throws -> (AIModel, [String: InferenceFunction]) {
        let chunk = String(index)
        var fields = ["kv": "chunked", "chunk": chunk, "layers": "\(spec.start)..<\(spec.end)"]
        if let reason { fields["reason"] = reason }
        events?("model.specialize.begin", fields)
        var retried = false
        while true {
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
                var functions: [String: InferenceFunction] = [:]
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
                return (model, functions)
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
    }

    /// Decode writes into one reused set of step_1 outputs: fresh outputs per token are freed only every
    /// few tokens and raise peak memory above the memory guard's floor.
    private static func decodeOutputs(
        spec: Gemma4ChunkContract.Chunk, _ functions: [String: InferenceFunction]
    ) throws -> Gemma4StateStorage? {
        guard let function = functions["step_1"], let expected = spec.functions["step_1"] else { return nil }
        return Gemma4StateStorage(
            descriptors: try expected.outputs.map { name in
                guard case .ndArray(let descriptor) = function.descriptor.outputDescriptor(of: name)
                else { throw Gemma4RuntimeError.invalidPackage }
                return (name, descriptor)
            })
    }

    /// The decode steps after a load, compared across the chunks (Gemma4CompileCheck): a chunk the Neural
    /// Engine compiler left degraded runs 3 to 130 times slower than its siblings, and the content-keyed cache
    /// keeps that result for every later load. Every degraded chunk has its cache entries dropped and is
    /// compiled again, once, and the steps after that are judged again: a chunk still degraded (the compiler
    /// failed the same way twice within seconds on the phone) has its entries dropped once more, so the next
    /// load compiles it afresh, and this load goes on with it. A compile that fails leaves the runtime without
    /// the chunk, so the error propagates and the next load compiles it afresh.
    private func recompileIfDegraded(steps: [[Double]]) async throws {
        let check = Gemma4CompileCheck(steps: steps)
        let degraded = check.degraded.filter { chunks.indices.contains($0) }
        events?(
            "model.compile.check",
            [
                "steps": steps.map { $0.map { String(format: "%.2f", $0) }.joined(separator: ",") }
                    .joined(separator: ";"),
                "medianMs": String(format: "%.2f", check.medianMs),
                "chunks": degraded.isEmpty ? "none" : degraded.map(String.init).joined(separator: ","),
                "round": recompiled ? "after recompile" : "first",
            ])
        guard !degraded.isEmpty else { return }
        if recompiled {
            for index in degraded {
                try? AIModelCache.default.deleteEntries(for: folder.appending(path: chunks[index].spec.file))
            }
            events?("model.compile.unresolved", ["chunks": degraded.map(String.init).joined(separator: ",")])
            return
        }
        recompiled = true
        for index in degraded {
            let spec = chunks[index].spec, url = folder.appending(path: spec.file)
            models[index] = nil  // release the Neural Engine's copy before compiling again
            chunks[index] = Loaded(spec: spec, functions: [:], decodeOutputs: nil)
            try? AIModelCache.default.deleteEntries(for: url)
            let (model, functions) = try await Self.load(
                chunk: index, spec: spec, url: url, reason: "degraded", clock: ContinuousClock(), events: events)
            chunks[index] = Loaded(
                spec: spec, functions: functions, decodeOutputs: try Self.decodeOutputs(spec: spec, functions))
            models[index] = model
        }
        decodeSteps = 0
        compileCheckSamples = []
    }

    /// Starts a new sequence at position 0. The masks hide every stale slot until it is overwritten.
    public func reset() throws {
        guard !working else { throw Gemma4RuntimeError.busy }
        position = 0  // Masks hide every stale slot until it is overwritten.
        table.reset()
        cancelled = false
    }

    /// Frees the full-attention slots of the tokens at `indices` of what the cache holds, oldest
    /// first (KV-SLIDE, REPORT.md Section 7.3): the masks hide them, in the sliding-attention
    /// rings too, and later tokens reuse the slots, while every other token keeps its position and
    /// its K/V.
    public func evict(_ indices: some Sequence<Int> & Sendable) throws {
        guard !working else { throw Gemma4RuntimeError.busy }
        guard slides, position >= 0, indices.allSatisfy({ (0..<table.held).contains($0) }) else {
            throw Gemma4RuntimeError.contextExceeded
        }
        table.evict(indices)
    }

    /// The position each full-attention slot holds (-1: free), to save beside `saveKV`'s file.
    public var slots: [Int32] { table.positions }

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
    /// `slots` is what `slots` returned when the caches were saved; nil when nothing had been evicted.
    public func restoreKV(from url: URL, position: Int, slots: [Int32]? = nil) throws {
        guard !working else { throw Gemma4RuntimeError.invalidPackage }
        var restored = Gemma4SlotTable(capacity: configuration.contextLimit)
        if let slots {
            restored = try Gemma4SlotTable(positions: slots)
            guard slides, restored.capacity == configuration.contextLimit, restored.held > 0,
                (1...Gemma4RuntimeConfiguration.maximumPosition).contains(position),
                slots.allSatisfy({ $0 < position })
            else { throw Gemma4RuntimeError.invalidPackage }
        } else {
            guard (1...configuration.contextLimit).contains(position) else { throw Gemma4RuntimeError.invalidPackage }
            _ = try restored.allocate(position: 0, count: position)
        }
        let file = Darwin.open(url.path, O_RDONLY)
        guard file >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(file) }
        self.position = 0
        table.reset()
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
        table = restored
        cancelled = false
    }

    /// PREFIX-KV (REPORT.md Section 7.5): writes the K/V of a prompt's first tokens to `url` when
    /// they are all the cache holds, counted from a reset. Each cache gives its first
    /// min(position, slots) slots, so the file grows with the prefix, not with the context.
    public func savePrefix(to url: URL) throws {
        let count = position
        guard !working, count > 0, table.held == count, (0..<count).allSatisfy({ table.positions[$0] == Int32($0) })
        else { throw Gemma4RuntimeError.resetRequired }
        let partial = url.appendingPathExtension("partial")
        let file = Darwin.open(partial.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        guard file >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        var closed = false
        defer { if !closed { Darwin.close(file) } }
        for name in caches.keys.sorted() {
            guard var cache = caches.removeValue(forKey: name) else { throw Gemma4RuntimeError.invalidOutput }
            defer { caches[name] = cache }
            let slots = cache.shape[3], kept = min(count, slots), rows = cache.shape[1] * cache.shape[2]
            var packed = [Float16](repeating: 0, count: rows * kept)
            guard
                try Self.withContiguousBytes(
                    of: &cache,
                    { base, _ in
                        let source = base.assumingMemoryBound(to: Float16.self)
                        for row in 0..<rows {
                            for slot in 0..<kept { packed[row * kept + slot] = source[row * slots + slot] }
                        }
                    }),
                packed.withUnsafeBytes({ bytes in
                    Self.transfer(bytes.count) { Darwin.write(file, bytes.baseAddress! + $0, $1) }
                })
            else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        closed = true
        guard Darwin.close(file) == 0, rename(partial.path, url.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    /// Reads `savePrefix`'s file for `count` tokens into a reset cache and continues after them.
    /// False when the file is missing, has another size or holds a value that is not finite (a
    /// masked slot still multiplies its value by 0); the cache then still counts as empty.
    public func restorePrefix(from url: URL, count: Int) throws -> Bool {
        guard !working, position == 0, table.held == 0, count > 0, count < configuration.contextLimit else {
            throw Gemma4RuntimeError.resetRequired
        }
        let file = Darwin.open(url.path, O_RDONLY)
        guard file >= 0 else { return false }
        defer { Darwin.close(file) }
        let expected = caches.values.reduce(0) {
            $0 + $1.shape[1] * $1.shape[2] * min(count, $1.shape[3]) * MemoryLayout<Float16>.size
        }
        var information = stat()
        guard fstat(file, &information) == 0, Int(information.st_size) == expected else { return false }
        for name in caches.keys.sorted() {
            guard var cache = caches.removeValue(forKey: name) else { throw Gemma4RuntimeError.invalidPackage }
            defer { caches[name] = cache }
            let slots = cache.shape[3], kept = min(count, slots), rows = cache.shape[1] * cache.shape[2]
            var packed = [Float16](repeating: 0, count: rows * kept)
            guard
                packed.withUnsafeMutableBytes({ bytes in
                    Self.transfer(bytes.count) { Darwin.read(file, bytes.baseAddress! + $0, $1) }
                }), packed.allSatisfy(\.isFinite),
                try Self.withContiguousBytes(
                    of: &cache,
                    { base, _ in
                        let target = base.assumingMemoryBound(to: Float16.self)
                        for row in 0..<rows {
                            for slot in 0..<kept { target[row * slots + slot] = packed[row * kept + slot] }
                        }
                    })
            else { return false }
        }
        _ = try table.allocate(position: 0, count: count)
        position = count
        return true
    }

    /// Feeds `tokens` at the current position and returns the logits after the last one. Prompts
    /// run as headless prefill calls on the KV-owner chunks, then `step_1` on the held-back token.
    /// `evicting` are the tokens the caller gives up for these, counted like `evict`'s (KV-SLIDE),
    /// each stretch as it stands once the one before it is gone: their slots are freed a block at
    /// a time, each when the next call needs the room, so every call still sees as much of the old
    /// context as fits. A call cancelled between two chunks throws `CancellationError` and keeps
    /// what the steps before it wrote: one token's call leaves the cache as it was and the next
    /// call continues; a prompt's leaves part of it fed and part of `evicting` freed, so its caller
    /// resets. After any other failure the runtime needs `reset()`.
    public func logits(for tokens: [Int], evicting: [Range<Int>] = []) async throws -> [Float16] {
        guard !working else { throw Gemma4RuntimeError.busy }
        guard position >= 0 else { throw Gemma4RuntimeError.resetRequired }
        guard tokens.allSatisfy({ (0..<configuration.vocabularySize).contains($0) }) else {
            throw Gemma4RuntimeError.invalidToken
        }
        // Slots hold the tokens; positions run on past the slot count once a package slides.
        let positions = (slides ? Gemma4RuntimeConfiguration.maximumPosition : configuration.contextLimit) - position
        var pending = evicting.filter { !$0.isEmpty }
        var giving = 0
        for stretch in pending {
            guard slides, stretch.lowerBound >= 0, stretch.upperBound <= table.held - giving else {
                throw Gemma4RuntimeError.contextExceeded
            }
            giving += stretch.count
        }
        guard !tokens.isEmpty, tokens.count <= table.free + giving, tokens.count <= positions else {
            throw Gemma4RuntimeError.contextExceeded
        }
        working = true
        defer { working = false }
        do {
            var last: [Float16] = []
            let plan = try Gemma4PrefillPlan(
                tokenCount: tokens.count, prefillLength: configuration.prefillLength, availableSlots: positions)
            // The embeddings of the next call are gathered while this call's chunks run on the
            // Neural Engine (REPORT.md Section 7.2); the masks and rotary tables, which depend on
            // what the earlier calls wrote, are built just before each call.
            var ahead: Task<Gathered, Error>?
            defer { ahead?.cancel() }
            for (index, chunk) in plan.chunks.enumerated() {
                try checkStopped()
                while table.free < chunk.tokens.count, var stretch = pending.first {
                    table.makeRoom(for: chunk.tokens.count, from: &stretch, block: KVSlidePlan.block)
                    if stretch.isEmpty { pending.removeFirst() } else { pending[0] = stretch }
                }
                let real = Array(tokens[chunk.tokens])
                let gathered: Gathered
                if let ready = ahead {
                    ahead = nil
                    gathered = try await ready.value
                } else {
                    gathered = try Self.gather(real, plan: chunk, configuration, embeddings, perLayerEmbeddings)
                }
                if index + 1 < plan.chunks.count {
                    let next = plan.chunks[index + 1], following = Array(tokens[next.tokens])
                    let c = configuration, embeddings = embeddings, identity = perLayerEmbeddings
                    ahead = Task.detached(priority: .userInitiated) {
                        try Self.gather(following, plan: next, c, embeddings, identity)
                    }
                }
                if let scores = try await step(real, plan: chunk, gathered: gathered) { last = scores }
            }
            for stretch in pending { table.evict(stretch) }  // given up, whether or not a call needed them
            return last
        } catch is Stopped {
            // A step writes its K/V and moves the position only once its last chunk has run, so
            // the caches hold whole steps and nothing needs a reset. The request to stop has been
            // served: left set, it would stop every call until the next reset.
            cancelled = false
            throw CancellationError()
        } catch {
            position = -1  // A failing native call may already have mutated some states.
            throw error
        }
    }

    /// Thrown where a call looks for cancellation: between two chunks, with nothing of the step
    /// in progress written. A `CancellationError` out of a native call is a failure like any other.
    private struct Stopped: Error {}

    private func checkStopped() throws {
        guard !Task.isCancelled, !cancelled else { throw Stopped() }
    }

    /// What one call reads from the embedding tables: its (padded) tokens' embeddings and
    /// per-layer embeddings, gathered off the actor while the previous call runs.
    struct Gathered: Sendable {
        var tokens: [Int]
        var embedded: [Float16]
        var identity: [Float16]
    }

    /// The tokens of one call: a short prefill remainder runs the static-width function padded
    /// with copies of its last token; the causal masks hide the padded keys from the real queries,
    /// and only the real tokens' K/V are written back (Gemma4PrefillPlan.paddedMinimum).
    private static func padded(
        _ real: [Int], plan: Gemma4PrefillPlan.Chunk, _ c: Gemma4RuntimeConfiguration
    ) throws -> [Int] {
        let padding = plan.function == "prefill_\(c.prefillLength)" ? c.prefillLength - real.count : 0
        guard padding >= 0, let last = real.last else { throw Gemma4RuntimeError.contextExceeded }
        return real + Array(repeating: last, count: padding)
    }

    private static func gather(
        _ real: [Int], plan: Gemma4PrefillPlan.Chunk, _ c: Gemma4RuntimeConfiguration,
        _ embeddings: Gemma4Embeddings, _ perLayerEmbeddings: Gemma4Embeddings
    ) throws -> Gathered {
        let tokens = try padded(real, plan: plan, c)
        return Gathered(
            tokens: tokens,
            embedded: try embeddings.gather(tokens: tokens, scale: sqrt(Float(c.hiddenSize))),
            identity: try perLayerEmbeddings.gather(
                tokens: tokens, layers: c.layerTypes.count, scale: sqrt(Float(c.perLayerInputSize))))
    }

    private func step(_ real: [Int], plan: Gemma4PrefillPlan.Chunk, gathered: Gathered) async throws -> [Float16]? {
        let c = configuration, clock = ContinuousClock()
        let began = clock.now
        let tokens = try Self.padded(real, plan: plan, c)
        guard gathered.tokens == tokens else { throw Gemma4RuntimeError.invalidToken }
        let length = tokens.count
        if !slides { try c.validatePosition(step: position, length: length) }
        let local = try c.rotary(step: position, length: length, global: false)
        let global = try c.rotary(step: position, length: length, global: true)
        let layers = c.layerTypes.count
        let identity = gathered.identity
        var common: [String: NDArray] = [
            "embedded": try Self.input(gathered.embedded, shape: [1, c.hiddenSize, 1, length]),
            "local_cos": try Self.input(local.cos, shape: [1, c.localHeadDimension, 1, length]),
            "local_sin": try Self.input(local.sin, shape: [1, c.localHeadDimension, 1, length]),
            "global_cos": try Self.input(global.cos, shape: [1, c.globalHeadDimension, 1, length]),
            "global_sin": try Self.input(global.sin, shape: [1, c.globalHeadDimension, 1, length]),
        ]
        // Old slots at their positions (or ring slots), then this call's fresh K/V: [1, 1, L, slots + L].
        common["local_mask"] =
            ring
            ? try Self.input(
                try c.ringMask(
                    step: position, length: length,
                    hidden: table.missing(max(0, position - c.slidingWindow)..<position)),
                shape: [1, 1, length, c.slidingWindow + length])
            : try Self.input(
                try c.mask(step: position, length: length, global: false),
                shape: [1, 1, length, c.contextLimit + length])
        common["global_mask"] = try Self.input(table.mask(length: length), shape: [1, 1, length, c.contextLimit + length])
        let inputsReady = clock.now
        // Packages have no prefill_1: a 1-token prefill tail runs step_1
        // on the owner chunks, and nothing reads the hidden it passes on.
        let name = plan.returnsLogits || plan.function == "prefill_1" ? "step_1" : plan.function
        var chunkTimes: [String] = []
        var stepMs: [Double] = []
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
                    inputs[input] = try Self.input(
                        Gemma4Embeddings.sliceLayers(identity, layers: layers, length: length, range: range),
                        shape: [1, c.perLayerInputSize, range.count, length])
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
            let elapsed = clock.now - started
            chunkTimes.append(milliseconds(elapsed))
            stepMs.append(elapsed.milliseconds)
            try checkStopped()
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
        try write(fresh, width: length, length: real.count)
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
        position += real.count
        // "function" is the scheduled call (prefill_1 stays prefill_1); "graph" is what ran.
        events?(
            "inference.timing",
            [
                "function": plan.returnsLogits ? "step_1" : plan.function,
                "graph": name, "length": String(real.count), "width": String(length),
                "position": String(position), "inputMs": milliseconds(inputsReady - began),
                "chunkMs": chunkTimes.joined(separator: ","), "writeMs": milliseconds(written - computed),
                "readbackMs": milliseconds(finished - written),
                "totalMs": milliseconds(finished - began),
            ])
        if name == "step_1", plan.returnsLogits, stepMs.count == chunks.count,
            var samples = compileCheckSamples
        {
            decodeSteps += 1
            if decodeSteps > Gemma4CompileCheck.coldSteps { samples.append(stepMs) }
            if samples.count < Gemma4CompileCheck.sampledSteps {
                compileCheckSamples = samples
            } else {
                compileCheckSamples = nil  // recompileIfDegraded reopens the sampling after a recompile
                try await recompileIfDegraded(steps: samples)
            }
        }
        return scores
    }

    /// An FP16 input holding `values` in `shape`, copied in one run. `NDArray(scalars:)` walks
    /// its sequence one value at a time, about 40 ns each: 20 ms for a 4K call's mask and 160 ms
    /// for a 32K one, most of what the host spent on a prefill call (REPORT.md Section 7.2).
    private static func input(_ values: [Float16], shape: [Int]) throws -> NDArray {
        guard !values.isEmpty, values.count == shape.reduce(1, *) else { throw Gemma4RuntimeError.invalidPackage }
        var array = NDArray(shape: shape, scalarType: .float16)
        let copied = try withContiguousBytes(of: &array) { base, bytes in
            values.withUnsafeBytes { _ = memcpy(base, $0.baseAddress, bytes) }
        }
        return copied ? array : NDArray(scalars: values, shape: shape)
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

    /// Writes the first `length` of a call's `width` fresh K/V columns [1, Hkv, D, L] for positions
    /// [position, position + L): into the slots the table gives them for a full-length cache, and
    /// modulo the slot count for a sliding-attention ring.
    private func write(_ fresh: [String: NDArray], width: Int, length: Int) throws {
        let assigned = try table.allocate(position: position, count: length)
        for (output, array) in fresh {
            let name = String(output.dropFirst("fresh_".count))
            guard var cache = caches.removeValue(forKey: name), array.scalarType == .float16,
                array.shape == [cache.shape[0], cache.shape[1], cache.shape[2], width], length <= width
            else { throw Gemma4RuntimeError.invalidOutput }
            var finite = true
            let slots = cache.shape[3], start = position, full = slots == table.capacity
            array.view(as: Float16.self).withUnsafePointer { source, shape, from in
                cache.mutableView(as: Float16.self).withUnsafeMutablePointer { target, _, to in
                    for head in 0..<shape[1] {
                        for dimension in 0..<shape[2] {
                            for token in 0..<length {
                                let value = source[head * from[1] + dimension * from[2] + token * from[3]]
                                finite = finite && value.isFinite
                                let slot = full ? assigned[token] : (start + token) % slots
                                target[head * to[1] + dimension * to[2] + slot * to[3]] = value
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
