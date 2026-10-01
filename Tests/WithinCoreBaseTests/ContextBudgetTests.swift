import Foundation
import Testing

@testable import WithinCoreBase

@Test func contextKeepsRecentCompleteTurnsAndTemplateBudget() throws {
    let messages = [ChatTurn.Role.user, .assistant, .user, .assistant, .user].enumerated().map {
        ChatTurn(role: $0.element, text: String($0.offset))
    }
    let result = try ContextBudget.fit(messages: messages, contextLimit: 45, requestedOutput: 10) { 5 + $0.count * 10 }
    #expect(result.messages.map(\.text) == ["2", "3", "4"])
    #expect(result.outputTokens == 10)
    #expect(result.trimmed)
}

@Test func tooLongLatestMessageIsNeverSilentlyCut() {
    let messages = [ChatTurn(role: .user, text: "long input")]
    #expect(throws: ContextBudgetError.inputTooLong) {
        try ContextBudget.fit(messages: messages, contextLimit: 10, requestedOutput: 4) { _ in 10 }
    }
}

@Test func trimmedConversationKeepsItsStartAcrossLaterTurns() throws {
    // Each message costs 10 tokens; 100-token context, 10 for the reply.
    var messages: [ChatTurn] = []
    var starts: [String] = []
    for turn in 0..<12 {
        messages.append(ChatTurn(role: .user, text: "u\(turn)"))
        let result = try ContextBudget.fit(messages: messages, contextLimit: 100, requestedOutput: 10) { $0.count * 10 }
        #expect(result.messages.count * 10 + 10 <= 100)
        #expect(result.messages.last?.text == "u\(turn)")
        starts.append(result.messages[0].text)
        messages.append(ChatTurn(role: .assistant, text: "a\(turn)"))
    }
    // Minimal trimming would move the start every turn once full (u4, u5, u6...);
    // checkpoints keep it for several turns so the KV cache prefix stays valid.
    #expect(starts == ["u0", "u0", "u0", "u0", "u0", "u3", "u3", "u3", "u5", "u5", "u8", "u8"])
}

@Test func failedEmptyRepliesDoNotLeaveAdjacentUserTurns() throws {
    let old = ChatTurn(role: .user, text: "Old request")
    let failed = ChatTurn(role: .assistant, text: "")
    let latest = ChatTurn(role: .user, text: "New request")
    let result = try ContextBudget.fit(messages: [old, failed, latest], contextLimit: 512, requestedOutput: 100) {
        messages in
        #expect(messages.last?.role == .user)
        for pair in zip(messages, messages.dropFirst()) { #expect(pair.0.role != pair.1.role) }
        return 20 * messages.count
    }
    #expect(result.messages == [latest])
    #expect(result.trimmed)
}
