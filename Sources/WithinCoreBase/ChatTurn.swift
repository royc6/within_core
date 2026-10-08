import Foundation

/// One message of a conversation as the model sees it. The system prompt is passed separately.
public struct ChatTurn: Equatable, Sendable {
    /// Who wrote the turn; the raw value is the chat template's role name.
    public enum Role: String, Sendable { case user, assistant }
    /// Who wrote it.
    public var role: Role
    /// The turn's text; an empty assistant turn is a reply that failed or was cut off.
    public var text: String
    /// AUTO-COMPACT: a user turn that asks for the summary of the conversation so far; the assistant
    /// turn after it is that summary. The engine keeps the latest such pair in the context while
    /// older turns leave it.
    public var isSummaryRequest: Bool
    /// The conversation was thinned when this assistant turn ended (REPORT.md Section 7.7): of the
    /// turns that end more than a stretch before it, the prompt renders only the user's short ones.
    public var thinned: Bool

    /// A turn from `role` with `text`.
    public init(role: Role, text: String, isSummaryRequest: Bool = false, thinned: Bool = false) {
        self.role = role; self.text = text; self.isSummaryRequest = isSummaryRequest; self.thinned = thinned
    }
}
