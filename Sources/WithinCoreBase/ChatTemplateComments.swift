import Foundation

/// The pinned swift-jinja ignores a comment's `-#}` whitespace control, so the
/// Gemma 4 template renders `<bos>\n\n<|turn>user…` (one extra
/// token before every prompt) where Transformers renders `<bos><|turn>user…`.
/// Removing the comments first, with the whitespace handling Transformers'
/// Jinja applies to them (`{#-` / `-#}` trims; `lstrip_blocks` and `trim_blocks`
/// otherwise), gives the same text without relying on the renderer's comments.
public enum ChatTemplateComments {
    /// Returns `template` without its `{# … #}` comments, trimming whitespace as Transformers does.
    public static func strip(_ template: String) -> String {
        var output = ""
        var rest = Substring(template)
        while let open = rest.range(of: "{#") {
            output += rest[..<open.lowerBound]
            guard let close = rest[open.upperBound...].range(of: "#}") else {
                output += rest[open.lowerBound...]  // unterminated: leave it for the renderer to reject
                return output
            }
            let body = rest[open.upperBound..<close.lowerBound]
            if body.hasPrefix("-") {
                while let last = output.last, last.isWhitespace { output.removeLast() }
            } else if let lineStart = output.lastIndex(of: "\n").map({ output.index(after: $0) })
                ?? Optional(output.startIndex),
                output[lineStart...].allSatisfy({ $0 == " " || $0 == "\t" })
            {
                output.removeSubrange(lineStart...)  // lstrip_blocks
            }
            rest = rest[close.upperBound...]
            if body.hasSuffix("-") {
                rest = rest.drop(while: \.isWhitespace)
            } else if rest.first == "\n" {
                rest = rest.dropFirst()  // trim_blocks
            }
        }
        return output + rest
    }
}
