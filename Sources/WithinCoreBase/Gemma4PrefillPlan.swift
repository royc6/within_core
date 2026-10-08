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
    /// A remainder of at least this many tokens runs as one padded `prefill_<prefillLength>` call
    /// (the runtime pads it to the static width and writes back only the real tokens) instead of
    /// one `prefill_1` call per token. A padded call costs about as much as a full one, which grows
    /// with the width (16: 35 ms, 64: 67 ms on the ANE of an iPhone 18 Pro, results/prefill.json), against about 20 ms
    /// per single-token call: width / 16 singles cost about the same, so fewer stay singles.
    public static func paddedMinimum(prefillLength: Int) -> Int { max(2, prefillLength / 16) }

    /// Plans `tokenCount` prompt tokens with `prefill_<prefillLength>` calls where they fit. A padded
    /// call is planned only while `availableSlots` (capacity left from the current position) holds
    /// the full static width, as the graph writes that many cache slots.
    public init(tokenCount: Int, prefillLength: Int, availableSlots: Int = .max) throws {
        guard tokenCount > 0, (2...256).contains(prefillLength) else {
            throw Gemma4RuntimeError.contextExceeded
        }
        var result: [Chunk] = []
        let prefixEnd = tokenCount - 1
        var cursor = 0
        while cursor < prefixEnd {
            let remaining = prefixEnd - cursor
            let length: Int
            let function: String
            if remaining >= prefillLength {
                (length, function) = (prefillLength, "prefill_\(prefillLength)")
            } else if remaining >= Self.paddedMinimum(prefillLength: prefillLength),
                cursor + prefillLength <= availableSlots
            {
                (length, function) = (remaining, "prefill_\(prefillLength)")
            } else {
                (length, function) = (1, "prefill_1")
            }
            result.append(Chunk(tokens: cursor..<cursor + length, function: function, returnsLogits: false))
            cursor += length
        }
        result.append(Chunk(tokens: prefixEnd..<tokenCount, function: "step_1", returnsLogits: true))
        chunks = result
    }
}
