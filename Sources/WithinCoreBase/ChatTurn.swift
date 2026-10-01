import Foundation

/// One message of a conversation as the model sees it. The system prompt is passed separately.
public struct ChatTurn: Equatable, Sendable {
    /// Who wrote the turn; the raw value is the chat template's role name.
    public enum Role: String, Sendable { case user, assistant }
    /// Who wrote it.
    public var role: Role
    /// The turn's text; an empty assistant turn is a reply that failed or was cut off.
    public var text: String

    /// A turn from `role` with `text`.
    public init(role: Role, text: String) {
        self.role = role; self.text = text
    }
}
