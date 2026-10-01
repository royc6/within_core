import Foundation

/// Why a conversation cannot be fitted into the context.
public enum ContextBudgetError: Error, LocalizedError, Equatable {
    case missingUserMessage, inputTooLong, invalidConfiguration
    public var errorDescription: String? {
        switch self {
        case .missingUserMessage: "There is no message to send."
        case .inputTooLong: "This message is too long for the installed model. Please shorten it."
        case .invalidConfiguration: "The model's context configuration is invalid."
        }
    }
}

/// A conversation fitted into the context: the turns to render and the reply budget left.
public struct BudgetedContext: Sendable {
    /// Complete user–assistant pairs, then the latest user turn.
    public let messages: [ChatTurn]
    /// The reply budget: the requested output, lowered to what fits.
    public let outputTokens: Int
    /// Whether earlier turns were left out.
    public let trimmed: Bool
}

/// Fits a conversation into the context, keeping its start fixed for as long as it can so that
/// later prompts keep extending the KV cache instead of prefilling again.
public enum ContextBudget {
    /// The caller counts the actual rendered chat template, including role/control tokens.
    public static func fit(
        messages: [ChatTurn], contextLimit: Int, requestedOutput: Int,
        countRenderedTokens: ([ChatTurn]) throws -> Int
    ) throws -> BudgetedContext {
        guard contextLimit > 1, requestedOutput > 0 else { throw ContextBudgetError.invalidConfiguration }
        guard messages.last?.role == .user else { throw ContextBudgetError.missingUserMessage }
        let latest = [messages[messages.count - 1]]
        let minimumCount = try countRenderedTokens(latest)
        guard minimumCount < contextLimit else { throw ContextBudgetError.inputTooLong }
        let output = min(requestedOutput, contextLimit - minimumCount)
        var retained: [ChatTurn] = []
        var pendingUser: ChatTurn?
        // Failed or interrupted empty replies do not create adjacent user turns
        // in a model template that requires alternating roles.
        for message in messages.dropLast() {
            if message.role == .user {
                pendingUser = message
            } else if let user = pendingUser, !message.text.isEmpty {
                retained += [user, message]; pendingUser = nil
            }
        }
        retained += latest
        guard try countRenderedTokens(retained) + output > contextLimit else {
            return BudgetedContext(messages: retained, outputTokens: output, trimmed: retained.count != messages.count)
        }
        // Drop complete oldest turns, cutting only at checkpoints: the first turn
        // after each half-context of the conversation so far. Earlier turns never
        // change, so the cut stays put across later turns and their prompts keep
        // extending the KV cache; cutting one turn per request would force a full
        // prefill on every turn.
        let turns = stride(from: 0, to: retained.count - 1, by: 2).map { retained[$0..<$0 + 2] }
        var total = 0, checkpoints: [Int] = []
        for (index, turn) in turns.enumerated() {
            if index > 0, total / (contextLimit / 2) > checkpoints.count { checkpoints.append(index) }
            total += try countRenderedTokens(Array(turn))
        }
        checkpoints.append(turns.count)  // the latest turn alone
        for start in checkpoints {
            let candidate = turns[start...].flatMap { $0 } + latest
            if try countRenderedTokens(candidate) + output <= contextLimit {
                return BudgetedContext(messages: candidate, outputTokens: output, trimmed: true)
            }
        }
        throw ContextBudgetError.inputTooLong
    }
}
