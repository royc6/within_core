import Foundation

/// Gemma's thought channel is transient. Only answer tokens reach chat storage.
public struct GemmaOutputFilter: Sendable {
    /// Whether the model is inside a thought channel.
    public private(set) var isThinking = false
    private let channelStart: Int
    private let channelEnd: Int
    private let thinkMarker: Int

    /// Token IDs of `<|channel>`, `<channel|>` and `<|think|>`.
    public init(channelStart: Int, channelEnd: Int, thinkMarker: Int) {
        self.channelStart = channelStart; self.channelEnd = channelEnd; self.thinkMarker = thinkMarker
    }

    /// Whether `token` is answer text; control tokens and thought-channel tokens are not.
    public mutating func accepts(_ token: Int) -> Bool {
        if token == channelStart { isThinking = true; return false }
        if token == channelEnd { isThinking = false; return false }
        if token == thinkMarker { return false }
        return !isThinking
    }
}
