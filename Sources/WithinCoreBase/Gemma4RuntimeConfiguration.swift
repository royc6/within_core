import Foundation

/// The model shape the package was exported with (`gemma-runtime.json`), and the host-side
/// inputs computed from it for every call: rotary tables and attention masks. Shapes refer to the
/// exported package, not the upstream model's theoretical context limit.
public struct Gemma4RuntimeConfiguration: Codable, Sendable {
    /// The file format version; 1.
    public var schemaVersion: Int
    /// Width of the hidden state between layers.
    public var hiddenSize: Int
    /// Tokens in the vocabulary; rows of the embedding tables and the head.
    public var vocabularySize: Int
    /// Per layer: `sliding_attention` or `full_attention`.
    public var layerTypes: [String]
    /// Trailing layers that reuse an earlier layer's KV instead of keeping their own.
    public var sharedLayerCount: Int
    /// Width of one layer's slice of the per-layer embedding.
    public var perLayerInputSize: Int
    /// Key/value heads per layer.
    public var keyValueHeads: Int
    /// Head dimension of sliding-attention layers.
    public var localHeadDimension: Int
    /// Head dimension of full-attention layers.
    public var globalHeadDimension: Int
    /// Rotary base of sliding-attention layers.
    public var localTheta: Float
    /// Rotary base of full-attention layers.
    public var globalTheta: Float
    /// Share of a full-attention head's dimensions that rotate; the rest pass through.
    public var globalRotaryFraction: Float
    /// Tokens a sliding-attention layer sees, including the current one.
    public var slidingWindow: Int
    /// Tokens of prompt, history and reply the package holds.
    public var contextLimit: Int
    /// Tokens per prefill call.
    public var prefillLength: Int
    /// The token embedding table.
    public var embeddings: Gemma4EmbeddingDescriptor
    /// The per-layer embedding table, every layer's slice in one row.
    public var perLayerEmbeddings: Gemma4EmbeddingDescriptor

    /// Layers that keep a KV cache.
    public var cacheLayers: Range<Int> { 0..<(layerTypes.count - sharedLayerCount) }
    /// The head dimension of `layer`'s attention type.
    public func headDimension(layer: Int) -> Int {
        layerTypes[layer] == "sliding_attention" ? localHeadDimension : globalHeadDimension
    }

    /// Checks every value is in range and consistent; with `requireE4B`, that it is exactly Gemma 4 E4B.
    public func validate(requireE4B: Bool = true) throws {
        guard schemaVersion == 1, (4...262_144).contains(vocabularySize),
            (4...2560).contains(hiddenSize), (2...42).contains(layerTypes.count),
            (0..<layerTypes.count).contains(sharedLayerCount),
            layerTypes.allSatisfy({ ["sliding_attention", "full_attention"].contains($0) }),
            Set(layerTypes.prefix(layerTypes.count - sharedLayerCount)) == Set(layerTypes),
            (1...256).contains(perLayerInputSize), (1...8).contains(keyValueHeads),
            (4...512).contains(localHeadDimension), localHeadDimension.isMultiple(of: 2),
            (4...512).contains(globalHeadDimension), globalHeadDimension.isMultiple(of: 2),
            localTheta.isFinite, localTheta >= 1, globalTheta.isFinite, globalTheta >= 1,
            globalRotaryFraction.isFinite, (0...1).contains(globalRotaryFraction),
            (1...512).contains(slidingWindow), (8...32000).contains(contextLimit),
            (2...128).contains(prefillLength), prefillLength <= contextLimit,
            embeddings.rows == vocabularySize, embeddings.columns == hiddenSize,
            perLayerEmbeddings.rows == vocabularySize,
            perLayerEmbeddings.columns == layerTypes.count * perLayerInputSize
        else { throw Gemma4RuntimeError.invalidPackage }
        try embeddings.validate(); try perLayerEmbeddings.validate()
        if requireE4B {
            guard vocabularySize == 262_144, hiddenSize == 2560, layerTypes.count == 42,
                sharedLayerCount == 18, perLayerInputSize == 256, keyValueHeads == 2,
                localHeadDimension == 256, globalHeadDimension == 512,
                localTheta == 10_000, globalTheta == 1_000_000, globalRotaryFraction == 0.25,
                slidingWindow == 512,
                layerTypes.enumerated().allSatisfy({ index, kind in
                    kind == ((index + 1).isMultiple(of: 6) ? "full_attention" : "sliding_attention")
                })
            else { throw Gemma4RuntimeError.invalidPackage }
        }
    }

    /// BC1S full-dimension RoPE, including Gemma 4's unrotated global axes.
    public func rotary(step: Int, length: Int, global: Bool) throws -> (cos: [Float16], sin: [Float16]) {
        try validatePosition(step: step, length: length)
        let dimension = global ? globalHeadDimension : localHeadDimension
        let theta = global ? globalTheta : localTheta
        let rotated = global ? Int(globalRotaryFraction * Float(dimension) / 2) : dimension / 2
        var cosine = [Float16](repeating: 1, count: dimension * length)
        var sine = [Float16](repeating: 0, count: dimension * length)
        for axis in 0..<rotated {
            let inverse = 1 / pow(theta, Float(2 * axis) / Float(dimension))
            for token in 0..<length {
                let angle = Float(step + token) * inverse
                for channel in [axis, axis + dimension / 2] {
                    cosine[channel * length + token] = Float16(cos(angle))
                    sine[channel * length + token] = Float16(sin(angle))
                }
            }
        }
        return (cosine, sine)
    }

    /// Read-only cache occupies the first C slots; fresh tokens follow it.
    public func mask(step: Int, length: Int, global: Bool) throws -> [Float16] {
        try validatePosition(step: step, length: length)
        let width = contextLimit + length
        var values = [Float16](repeating: -10_000, count: length * width)
        for query in 0..<length {
            let position = step + query
            let first = global ? 0 : max(0, position - slidingWindow + 1)
            if first < step {
                for key in first..<step { values[query * width + key] = 0 }
            }
            for fresh in 0...query where step + fresh >= first {
                values[query * width + contextLimit + fresh] = 0
            }
        }
        return values
    }

    /// Host-held sliding-attention ring of `slidingWindow` slots plus this call's
    /// fresh K/V: [L, window + L]. Slot j holds the latest position p < step with
    /// p % window == j (older ones were overwritten); it is visible to query q
    /// when q - p < window. Fresh keys are causal within the same window.
    public func ringMask(step: Int, length: Int) throws -> [Float16] {
        try validatePosition(step: step, length: length)
        let window = slidingWindow, width = window + length
        var values = [Float16](repeating: -10_000, count: length * width)
        for query in 0..<length {
            let position = step + query
            if step > 0 {
                for slot in 0..<window {
                    let held = step - 1 - ((step - 1 - slot) % window + window) % window  // floor modulo
                    if held >= 0, position - held < window { values[query * width + slot] = 0 }
                }
            }
            for fresh in 0...query where query - fresh < window { values[query * width + window + fresh] = 0 }
        }
        return values
    }

    /// Throws unless `length` tokens starting at `step` fit one call and the context.
    public func validatePosition(step: Int, length: Int) throws {
        guard step >= 0, length > 0, length <= prefillLength,
            step <= contextLimit - length
        else { throw Gemma4RuntimeError.contextExceeded }
    }
}

/// Failures of the package checks and of the decoder.
public enum Gemma4RuntimeError: Error, LocalizedError {
    case invalidPackage, invalidToken, invalidOutput, contextExceeded, busy, resetRequired
    public var errorDescription: String? {
        switch self {
        case .invalidPackage: "This Gemma 4 model package is incompatible or incomplete."
        case .invalidToken: "The tokenizer returned a token outside this model's vocabulary."
        case .invalidOutput: "The model returned an invalid result. Please try again."
        case .contextExceeded: "This conversation exceeds the installed model's context limit."
        case .busy: "Please wait for the current reply to stop."
        case .resetRequired: "Please start the reply again to reset the model."
        }
    }
}
