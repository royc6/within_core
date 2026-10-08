import Foundation

/// Keyword recall (REPORT.md Section 7.7): among the turns a conversation's cache no longer
/// holds, finds the ones that share the rarest words with the user's latest message, to be shown to
/// the model for that one reply. Nothing is indexed or kept, and no model runs.
public enum Recall {
    /// The words a text is matched by: runs of letters and digits (with `_ . / : -` inside them, so a
    /// path, a version or an identifier is one word), and each pair of neighbouring CJK characters.
    static func terms(_ text: String) -> [String] {
        var found: [String] = [], word = String.UnicodeScalarView(), pair: [Unicode.Scalar] = []
        let joins = Set("_./:-".unicodeScalars)
        func closeWord() {
            while let last = word.last, joins.contains(last) { word.removeLast() }
            if word.count > 1 { found.append(String(word)) }
            word.removeAll()
        }
        func closePair() {
            if pair.count == 1 { found.append(String(pair[0])) }
            for index in pair.indices.dropLast() { found.append(String(pair[index]) + String(pair[index + 1])) }
            pair.removeAll()
        }
        for scalar in text.lowercased().unicodeScalars {
            if wide(scalar) {
                closeWord(); pair.append(scalar)
            } else if scalar.properties.isAlphabetic || scalar.properties.numericType != nil {
                closePair(); word.append(scalar)
            } else if joins.contains(scalar), !word.isEmpty {
                word.append(scalar)
            } else {
                closeWord(); closePair()
            }
        }
        closeWord(); closePair()
        return found
    }

    /// CJK ideographs, kana and hangul: written without spaces, so matched by pairs of characters.
    private static func wide(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF, 0x3400...0x9FFF, 0xAC00...0xD7AF, 0xF900...0xFAFF: true
        default: false
        }
    }

    /// About how many tokens `text` takes: a CJK character is most of a token, other text about four
    /// characters to one.
    static func cost(_ text: String) -> Double {
        var total = 0.0
        for scalar in text.unicodeScalars { total += (0x3000...0xFFEF).contains(scalar.value) ? 0.75 : 0.25 }
        return total
    }

    /// `text` whole when it costs no more than `size`, otherwise cut at line ends (and a longer line
    /// within itself) into pieces of about that cost.
    static func pieces(_ text: String, size: Double) -> [String] {
        guard cost(text) > size else { return [text] }
        var found: [String] = [], current = ""
        func close() { if !current.isEmpty { found.append(current); current = "" } }
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            var rest = Substring(line)
            while cost(String(rest)) > size {
                close()
                var end = rest.startIndex, used = 0.0
                while end < rest.endIndex, used < size {
                    used += cost(String(rest[end])); end = rest.index(after: end)
                }
                found.append(String(rest[..<end])); rest = rest[end...]
            }
            if !current.isEmpty, cost(current) + cost(String(rest)) > size { close() }
            current = current.isEmpty ? String(rest) : current + "\n" + rest
        }
        close()
        return found.filter { !$0.allSatisfy(\.isWhitespace) }
    }

    /// The turns to show before the user's latest message `query`: a request to recall and, as the
    /// assistant's turn, what `old` holds that matches best. `old` are turns the cache no longer
    /// holds, in order. Each is searched whole, a long one in pieces of `piece` estimated tokens,
    /// ranked by BM25 on the words of `query`; up to `count` of them are quoted, in the order they
    /// were written, within `budget` estimated tokens. A match counts only when it scores at least
    /// what `least` words found in a single piece would, whatever the conversation's length. nil
    /// when nothing does.
    public static func turns(
        for query: String, among old: [ChatTurn], budget: Double = 700, count: Int = 2, piece: Double = 350,
        least: Double = 2.5
    ) -> [ChatTurn]? {
        var found: [Found] = []
        for turn in old {
            found += pieces(turn.text, size: piece).map { Found(key: $0, said: [(turn.role == .assistant, $0)]) }
        }
        return quoting(found, for: query, budget: budget, count: count, least: least)
    }

    /// The same for exchanges whose reply the cache no longer holds (REPORT.md Section 7.7): a reply
    /// is found by its own words and by the words of the message it answered, and both are quoted, the
    /// user's message first. The user's message may still be in the cache, far back; quoting it puts it
    /// next to the new one. A long message of the user's only helps to find the reply.
    public static func turns(
        for query: String, amongExchanges old: [(user: ChatTurn, assistant: ChatTurn)], budget: Double = 700,
        count: Int = 2, piece: Double = 350, least: Double = 2.5, common: Double = 1
    ) -> [ChatTurn]? {
        var found: [Found] = []
        for exchange in old {
            let asked = cost(exchange.user.text) <= piece / 2 ? [(mine: false, text: exchange.user.text)] : []
            found += pieces(exchange.assistant.text, size: piece).map {
                Found(key: exchange.user.text + "\n" + $0, said: asked + [(true, $0)])
            }
        }
        return quoting(found, for: query, budget: budget, count: count, least: least, common: common)
    }

    /// Something that can be recalled: the text it is found by, and what is quoted of it, each part
    /// the assistant's (`mine`) or the user's.
    private struct Found {
        var key: String
        var said: [(mine: Bool, text: String)]
    }

    /// A word held by more than the share `common` of what can be recalled does not count: it says
    /// nothing about which piece is meant (1 counts every word).
    private static func quoting(
        _ found: [Found], for query: String, budget: Double, count: Int, least: Double, common: Double = 1
    ) -> [ChatTurn]? {
        let asked = Set(terms(query))
        guard !asked.isEmpty, !found.isEmpty else { return nil }
        let words = found.map { terms($0.key) }
        var holders: [String: Int] = [:]  // how many pieces hold each word asked for
        for list in words { for word in Set(list) where asked.contains(word) { holders[word, default: 0] += 1 } }
        let total = Double(found.count), average = max(1, Double(words.reduce(0) { $0 + $1.count }) / total)
        let rarity = { (held: Double) in log(1 + (total - held + 0.5) / (held + 0.5)) }
        let unit = rarity(1)
        var ranked: [(score: Double, index: Int)] = []
        for (index, list) in words.enumerated() {
            var times: [String: Double] = [:]
            for word in list where asked.contains(word) && Double(holders[word] ?? 0) <= max(1, common * total) {
                times[word, default: 0] += 1
            }
            let length = 0.25 + 0.75 * Double(list.count) / average
            let score = times.reduce(0.0) { $0 + rarity(Double(holders[$1.key] ?? 0)) * $1.value * 2.2 / ($1.value + 1.2 * length) }
            if score / unit >= least { ranked.append((score, index)) }
        }
        ranked.sort { $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index }
        var chosen: [Int] = [], used = 0.0
        for match in ranked.prefix(count) {
            let price = found[match.index].said.reduce(0.0) { $0 + cost($1.text) }
            if used + price > budget { break }
            chosen.append(match.index); used += price
        }
        guard !chosen.isEmpty else { return nil }
        let chinese = query.unicodeScalars.contains { (0x3400...0x9FFF).contains($0.value) }
        let quoted = chosen.sorted().flatMap { index in
            found[index].said.map { part in
                (chinese ? (part.mine ? "我说过：" : "你写过：") : (part.mine ? "I said: " : "You wrote: ")) + part.text
            }
        }
        return [
            ChatTurn(
                role: .user,
                text: chinese
                    ? "（先回想一下这段对话里和我下一条消息有关的内容。）"
                    : "(First recall what in this conversation relates to my next message.)"),
            ChatTurn(
                role: .assistant,
                text: (chinese ? "这段对话里之前有这些：\n\n" : "Earlier in this conversation:\n\n")
                    + quoted.joined(separator: "\n\n")),
        ]
    }
}
