import Foundation

/// Teacher forcing on the device: the fixture carries token IDs only
/// (prompt rendered on the Mac with the frozen template, continuation from the
/// original model), so no tokenizer or sampling is involved in the score.
public struct TeacherForcingFixture: Codable, Sendable {
    /// The fixture format version.
    public static let version = "quant-quality-fixture-v1"
    /// One scored text: a rendered prompt and the reference continuation.
    public struct Item: Codable, Sendable {
        /// A stable identifier.
        public var id: String
        /// The part of the set the item belongs to, for per-subset results.
        public var subset: String
        /// Prompt token IDs, rendered with the chat template.
        public var prompt: [Int]
        /// Reference continuation token IDs; every one is scored.
        public var continuation: [Int]
    }
    /// Must equal `version`.
    public var version: String
    /// SHA-256 of the source set the items were made from.
    public var setSHA256: String
    /// SHA-256 of the reference continuations.
    public var referencesSHA256: String
    /// The items to score.
    public var items: [Item]

    /// Checks unique IDs, in-vocabulary tokens and that every item fits the context.
    public func validate(vocabularySize: Int, contextLimit: Int) throws {
        guard version == Self.version, !items.isEmpty, Set(items.map(\.id)).count == items.count else {
            throw TeacherForcingError.invalidFixture
        }
        for item in items {
            // The last continuation token is scored, never fed.
            guard !item.prompt.isEmpty, !item.continuation.isEmpty,
                item.prompt.count + item.continuation.count - 1 <= contextLimit,
                (item.prompt + item.continuation).allSatisfy({ (0..<vocabularySize).contains($0) })
            else { throw TeacherForcingError.invalidFixture }
        }
    }
}

/// A fixture or logits that cannot be scored.
public enum TeacherForcingError: Error, Equatable {
    case invalidFixture, invalidLogits
}

/// Summed negative log-likelihood of the tokens scored so far (natural log).
public struct TeacherForcingScore: Codable, Equatable, Sendable {
    /// Summed negative log-likelihood, in nats.
    public var nllSum = 0.0
    /// Tokens scored.
    public var tokens = 0
    /// Tokens that were also the most likely token.
    public var greedyMatches = 0

    /// An empty score.
    public init() {}

    /// Adds `-log softmax(logits)[token]`, computed in Double from the product's
    /// FP16 logits; a match means `token` is the first maximal logit.
    public mutating func add(logits: [Float16], token: Int) throws {
        guard logits.indices.contains(token) else { throw TeacherForcingError.invalidLogits }
        var maximum = -Float.infinity, best = 0
        for (index, value) in logits.enumerated() {
            let v = Float(value)
            guard v.isFinite else { throw TeacherForcingError.invalidLogits }
            if v > maximum { maximum = v; best = index }
        }
        var sum = 0.0
        for value in logits { sum += exp(Double(Float(value) - maximum)) }
        nllSum += log(sum) - Double(Float(logits[token]) - maximum)
        tokens += 1
        if best == token { greedyMatches += 1 }
    }
}
