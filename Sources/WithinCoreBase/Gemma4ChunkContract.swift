import Foundation

/// How the decoder is split into Core AI assets (`chunk-contract.json`): contiguous layer ranges,
/// then the vocabulary head, with host-held KV (`gemma4-chunked-hostkv-v1`). The host keeps every
/// owner layer's cache because the Neural Engine compiler rejects graphs that combine KV state with
/// compressed weights. The converter writes this file; the runtime checks it before loading a chunk.
public struct Gemma4ChunkContract: Codable, Sendable {
    /// The contract's file name in the package folder.
    public static let filename = "chunk-contract.json"
    /// The only contract this runtime runs: KV caches held by the host.
    public static let hostKV = "gemma4-chunked-hostkv-v1"
    /// One entry point of a chunk: `step_1` (decode) or `prefill_<n>` (n tokens at once).
    public struct Function: Codable, Equatable, Sendable {
        /// Tokens per call.
        public var length: Int
        /// Input names: shared inputs (hidden state, embeddings, rotary tables, masks) and KV caches.
        public var inputs: [String]
        /// Output names: the hidden state or logits, then fresh K/V for the layers the chunk owns.
        public var outputs: [String]
    }
    /// One Core AI asset running decoder layers `start..<end`, or the vocabulary head.
    public struct Chunk: Codable, Sendable {
        /// The asset's folder name in the package (`chunk-NN.aimodel`).
        public var file: String
        /// The first layer the chunk runs.
        public var start: Int
        /// One past the last layer the chunk runs.
        public var end: Int
        /// Entry points by name.
        public var functions: [String: Function]
        /// Layers whose KV this chunk produces, and the donors whose cache plus
        /// fresh K/V it reads.
        public var owned: [Int]?
        /// Donor layers outside this chunk whose cache and fresh K/V its shared layers read.
        public var externalDonors: [Int]?
        /// A head-only chunk after the last layer chunk, returning `parts` row
        /// slices `logits_00`... of `rows` each, concatenated by the host.
        public var head: Head?
        /// The vocabulary head's split into row slices.
        public struct Head: Codable, Equatable, Sendable {
            /// Row slices, each its own output.
            public var parts: Int
            /// Vocabulary rows per slice.
            public var rows: Int
        }
        /// The functions that run during headless prefill (owner chunks only).
        public var runsPrefill: Bool { functions.keys.contains { $0.hasPrefix("prefill_") } }
    }
    /// The contract identifier; must be `hostKV`.
    public var contract: String
    /// Tokens the full-attention caches hold.
    public var contextLimit: Int
    /// Tokens per `prefill_<n>` call.
    public var prefillLength: Int
    /// The first layer that reads another layer's KV instead of owning a cache.
    public var firstSharedLayer: Int
    /// How the chunk weights are stored, as recorded by the converter.
    public var weightFormat: String
    /// Layer chunks in order, then the head chunk.
    public var chunks: [Chunk]
    /// Sliding-attention caches are rings of this many slots (the window); only
    /// full-attention caches span `contextLimit`. Absent: every cache spans `contextLimit`.
    public var slidingCacheLength: Int?

    /// Slots in one layer's host-held cache.
    public func cacheLength(layer: Int, configuration c: Gemma4RuntimeConfiguration) -> Int {
        c.layerTypes[layer] == "sliding_attention" ? (slidingCacheLength ?? c.contextLimit) : c.contextLimit
    }

    static let commonInputs: Set<String> = [
        "hidden", "embedded", "token_identity", "local_cos", "local_sin",
        "global_cos", "global_sin", "local_mask", "global_mask",
    ]
}

extension Gemma4ChunkContract {
    /// Cache names: `key_<n>`/`value_<n>` [1, Hkv, D, C] read-only in,
    /// `fresh_key_<n>`/`fresh_value_<n>` [1, Hkv, D, L] out.
    public static func cacheNames(_ layer: Int, fresh: Bool = false) -> [String] {
        ["key", "value"].map { (fresh ? "fresh_" : "") + "\($0)_\(layer)" }
    }

    /// Output names of the head chunk's row slices: `logits_00`, `logits_01`, ….
    public static func logitNames(parts: Int) -> [String] {
        (0..<parts).map { String(format: "logits_%02d", $0) }
    }

    /// Checks the contract against the runtime configuration: contiguous layer ranges covering every
    /// layer, each owned layer's cache produced exactly once and before any chunk reads it, the inputs
    /// and outputs every function must have, and at most one head chunk, at the end.
    public func validate(configuration c: Gemma4RuntimeConfiguration, requireE4B: Bool = true) throws {
        try c.validate(requireE4B: requireE4B)
        guard contract == Self.hostKV else { throw Gemma4RuntimeError.invalidPackage }
        let cacheLayers = c.cacheLayers
        if let ring = slidingCacheLength {
            guard ring == c.slidingWindow, ring < c.contextLimit else { throw Gemma4RuntimeError.invalidPackage }
        }
        // Each attention type's donor: its last KV owner before sharing starts.
        var donors = Set<Int>()
        for kind in Set(c.layerTypes) {
            if let last = cacheLayers.last(where: { c.layerTypes[$0] == kind }) { donors.insert(last) }
        }
        // At most one head chunk, and only at the end.
        let headChunk = chunks.last.flatMap { $0.head == nil ? nil : $0 }
        let layerChunks = headChunk == nil ? chunks : Array(chunks.dropLast())
        guard contextLimit == c.contextLimit, prefillLength == c.prefillLength,
            firstSharedLayer == cacheLayers.upperBound,
            let first = layerChunks.first, let last = layerChunks.last,
            first.start == 0, last.end == c.layerTypes.count, layerChunks.count <= c.layerTypes.count,
            layerChunks.allSatisfy({ $0.head == nil })
        else { throw Gemma4RuntimeError.invalidPackage }
        var files = Set<String>()
        var produced = Set<Int>()
        let common = Self.commonInputs
        if let headChunk, let head = headChunk.head {
            guard head.parts > 0, head.rows > 0, head.parts * head.rows == c.vocabularySize,
                headChunk.start == c.layerTypes.count, headChunk.end == c.layerTypes.count,
                headChunk.owned == [], headChunk.externalDonors == [],
                headChunk.file.hasSuffix(".aimodel"), !headChunk.file.contains("/"),
                Set(headChunk.functions.keys) == ["step_1"], let step = headChunk.functions["step_1"],
                step.length == 1, step.inputs == ["hidden"], step.outputs == Self.logitNames(parts: head.parts)
            else { throw Gemma4RuntimeError.invalidPackage }
            files.insert(headChunk.file)
        }
        for (index, chunk) in layerChunks.enumerated() {
            let owned = Array(chunk.start..<max(chunk.start, min(chunk.end, firstSharedLayer)))
            guard chunk.start < chunk.end, chunk.end <= c.layerTypes.count else {
                throw Gemma4RuntimeError.invalidPackage
            }
            let consumers = min(max(chunk.start, firstSharedLayer), chunk.end)..<chunk.end
            let consumerKinds = Set(consumers.map { c.layerTypes[$0] })
            let external = donors.filter {
                consumerKinds.contains(c.layerTypes[$0]) && !(chunk.start..<chunk.end).contains($0)
            }.sorted()
            let fresh = owned.flatMap { Self.cacheNames($0, fresh: true) }
            let caches = Set(
                owned.flatMap { Self.cacheNames($0) }
                    + external.flatMap { Self.cacheNames($0) + Self.cacheNames($0, fresh: true) })
            let isLast = index == layerChunks.count - 1 && headChunk == nil
            guard chunk.start < chunk.end, index == 0 || layerChunks[index - 1].end == chunk.start,
                chunk.file.hasSuffix(".aimodel"), !chunk.file.contains("/"), files.insert(chunk.file).inserted,
                chunk.owned == owned, chunk.externalDonors == external,
                Set(external).isSubset(of: produced),
                let step = chunk.functions["step_1"], step.length == 1,
                step.inputs.contains("embedded"), step.inputs.contains("hidden") == (index > 0),
                caches.isSubset(of: Set(step.inputs)), Set(step.inputs).isSubset(of: common.union(caches)),
                step.outputs == (isLast ? ["logits"] : ["hidden"]) + fresh
            else { throw Gemma4RuntimeError.invalidPackage }
            let prefill = "prefill_\(prefillLength)"
            guard Set(chunk.functions.keys) == (owned.isEmpty ? ["step_1"] : ["step_1", prefill]) else {
                throw Gemma4RuntimeError.invalidPackage
            }
            if let function = chunk.functions[prefill] {
                guard function.length == prefillLength, function.inputs == step.inputs,
                    function.outputs == (chunk.end < firstSharedLayer ? ["hidden"] : []) + fresh
                else { throw Gemma4RuntimeError.invalidPackage }
            }
            produced.formUnion(owned)
        }
        guard produced == Set(cacheLayers) else { throw Gemma4RuntimeError.invalidPackage }
    }
}

extension Gemma4Embeddings {
    /// Layers [range] of an all-layer PLE gather, in the same [channel][layer][position] order.
    public static func sliceLayers(_ values: [Float16], layers: Int, length: Int, range: Range<Int>) -> [Float16] {
        let channels = values.count / (layers * length)
        var result = [Float16](repeating: 0, count: channels * range.count * length)
        values.withUnsafeBufferPointer { source in
            result.withUnsafeMutableBufferPointer { target in
                for channel in 0..<channels {
                    for (offset, layer) in range.enumerated() {
                        let from = (channel * layers + layer) * length
                        let to = (channel * range.count + offset) * length
                        for position in 0..<length { target[to + position] = source[from + position] }
                    }
                }
            }
        }
        return result
    }
}
