import Foundation

/// The call schedule for one prompt: headless prefill of every token but the last in
/// `prefill_<prefillLength>` and `prefill_1` calls, then `step_1` on the held-back token
/// for its logits. Mirrors Apple's held-back-token prefill strategy for two static widths.
public struct Gemma4PrefillPlan: Sendable {
    /// One call: the prompt tokens it feeds and the function that runs them.
    public struct Chunk: Sendable, Equatable {
        /// Indices into the prompt.
        public let tokens: Range<Int>
        /// `prefill_<n>`, `prefill_1` or `step_1`.
        public let function: String
        /// Only the last call, on the held-back token, returns logits.
        public let returnsLogits: Bool
    }
    /// The calls in order.
    public let chunks: [Chunk]

    /// Plans `tokenCount` prompt tokens with `prefill_<prefillLength>` calls where they fit.
    public init(tokenCount: Int, prefillLength: Int) throws {
        guard tokenCount > 0, (2...128).contains(prefillLength) else {
            throw Gemma4RuntimeError.contextExceeded
        }
        var result: [Chunk] = []
        let prefixEnd = tokenCount - 1
        var cursor = 0
        while cursor < prefixEnd {
            let length = prefixEnd - cursor >= prefillLength ? prefillLength : 1
            result.append(Chunk(tokens: cursor..<cursor + length, function: "prefill_\(length)", returnsLogits: false))
            cursor += length
        }
        result.append(Chunk(tokens: prefixEnd..<tokenCount, function: "step_1", returnsLogits: true))
        chunks = result
    }
}
