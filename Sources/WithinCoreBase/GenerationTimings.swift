import Foundation

/// Wall-clock marks of one generation, for performance measurement.
public struct GenerationTimings: Sendable {
    /// The rendered prompt's token IDs.
    public var inputTokenIDs: [Int]
    /// Tokens in the rendered prompt.
    public var inputTokens: Int { inputTokenIDs.count }
    /// Prompt tokens already in the KV cache, which were not prefilled again.
    public var reusedTokens: Int
    /// Request start (before any model load) to the prompt's logits.
    public var prefillMs: Double
    /// Request start to each sampled token, in order; thought tokens included.
    public var tokenMs: [Double]
    /// Timings from explicit marks.
    public init(inputTokenIDs: [Int], reusedTokens: Int, prefillMs: Double, tokenMs: [Double]) {
        self.inputTokenIDs = inputTokenIDs; self.reusedTokens = reusedTokens; self.prefillMs = prefillMs;
        self.tokenMs = tokenMs
    }
    /// Intervals before tokens `range` (1-based): token n's interval is tokenMs[n-1] - tokenMs[n-2].
    public func intervals(_ range: ClosedRange<Int>) -> [Double] {
        range.compactMap { n in n >= 2 && n <= tokenMs.count ? tokenMs[n - 1] - tokenMs[n - 2] : nil }
    }
}
