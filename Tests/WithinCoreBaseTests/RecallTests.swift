import Foundation
import Testing

@testable import WithinCoreBase

/// Keyword recall (REPORT.md Section 7.7): words are matched as written, CJK text by pairs of
/// characters, and an identifier, a path or a version stays one word.
@Test func recallMatchesWordsPathsAndCharacterPairs() {
    let words = Recall.terms("Jellyfin 映射到主机的端口是 31880，路径 /volume2/cinema，版本 10.9.11。OK?")
    #expect(words.contains("jellyfin") && words.contains("31880") && words.contains("volume2/cinema"))
    #expect(words.contains("10.9.11") && words.contains("ok") && words.contains("端口") && words.contains("映射"))
    #expect(!words.contains("是") && !words.contains("10.9.11。"))
    #expect(Recall.terms("fs.inotify.max_user_watches = 204800") == ["fs.inotify.max_user_watches", "204800"])
}

/// What the cache no longer holds is quoted for the question that asks about it, and only then.
@Test func recallQuotesTheOldTurnThatAnswersTheQuestion() throws {
    let old = [
        ChatTurn(role: .assistant, text: "入库不触发多半是 inotify 的监视数不够。建议把 fs.inotify.max_user_watches 调到 204800，改完重启 Jellyfin 容器。"),
        ChatTurn(role: .assistant, text: "周末请八个人吃饭，甜品可以提前做好放冰箱，杨枝甘露或者银耳雪梨羹都合适，当天只要分碗。"),
        ChatTurn(role: .user, text: "合同第七条：乙方退租并结清全部费用后，甲方应当在十二个工作日内退还押金，不计利息。"),
        ChatTurn(role: .assistant, text: "跑鞋跑到六百公里左右，中底的回弹就明显下降了，比赛前三周换新鞋比较稳妥。"),
    ]
    let pair = try #require(Recall.turns(for: "排查 Jellyfin 不自动入库的时候，你建议把 fs.inotify.max_user_watches 调到多少？", among: old))
    #expect(pair.map(\.role) == [.user, .assistant])
    #expect(pair[1].text.contains("我说过：") && pair[1].text.contains("204800") && !pair[1].text.contains("杨枝甘露"))
    // What the user pasted is quoted as theirs.
    let lease = try #require(Recall.turns(for: "合同里退租以后甲方多久退还押金？", among: old))
    #expect(lease[1].text.contains("你写过：") && lease[1].text.contains("十二个工作日"))
    // A question about something else recalls nothing: a shared common word is not a match.
    #expect(Recall.turns(for: "我的车牌号是多少？", among: old) == nil)
    #expect(Recall.turns(for: "跟我打个招呼，一句话就行。", among: old) == nil)
    #expect(Recall.turns(for: "排查 Jellyfin 入库的 inotify 设置", among: []) == nil)
    // English is labelled in English.
    let english = [
        ChatTurn(role: .assistant, text: "Cap the backup so the Wi-Fi chip survives it: add --bwlimit=2800 to the rsync command on the laptop."),
        ChatTurn(role: .assistant, text: "For the party, tell the guests to arrive early and keep the group chat muted."),
    ]
    let backup = try #require(Recall.turns(for: "What value did you give for the --bwlimit option of the rsync backup?", among: english))
    #expect(backup[0].text.hasPrefix("(First recall") && backup[1].text.contains("I said: Cap the backup"))
    #expect(!backup[1].text.contains("party"))
}

/// A long turn is searched and quoted in pieces, and nothing beyond the budget is quoted.
@Test func recallCutsLongTurnsAndKeepsToTheBudget() throws {
    let filler = (1...40).map { "第 \($0) 段：这里写的是一些和问题无关的普通内容，凑够长度用。" }
    let paste = (filler[..<20] + ["乌屿滩的那群塍鹬是去年十月初在退潮的时候看到的，一共数到一百三十多只。"] + filler[20...]).joined(separator: "\n")
    #expect(Recall.cost(paste) > 700)
    let parts = Recall.pieces(paste, size: 350)
    #expect(parts.count > 2 && parts.allSatisfy { Recall.cost($0) <= 350 } && parts.joined(separator: "\n") == paste)
    let pair = try #require(Recall.turns(for: "乌屿滩那群塍鹬是什么时候看到的？", among: [ChatTurn(role: .user, text: paste)]))
    #expect(pair[1].text.contains("去年十月初") && Recall.cost(pair[1].text) <= 720)
    // One line longer than a piece is cut within itself.
    let line = String(repeating: "很长的一行没有换行", count: 120)
    #expect(Recall.pieces(line, size: 350).allSatisfy { Recall.cost($0) <= 351 })
    // Nothing fits a budget smaller than the best piece.
    #expect(Recall.turns(for: "乌屿滩那群塍鹬是什么时候看到的？", among: [ChatTurn(role: .user, text: paste)], budget: 100) == nil)
}

/// A reply that does not say what it is about is found by the message it answered, and the two are
/// quoted together, the user's message first (REPORT.md Section 7.7).
@Test func recallFindsAReplyByTheMessageItAnswered() throws {
    let old: [(user: ChatTurn, assistant: ChatTurn)] = [
        (
            ChatTurn(role: .user, text: "我想约晏初见一面，一直定不下来。你随便替我定个时间和地方吧。"),
            ChatTurn(role: .assistant, text: "那就周三傍晚六点，花鸟市场门口。定了就别再改了。")
        ),
        (
            ChatTurn(role: .user, text: "周末请八个人吃饭，甜品做什么好？"),
            ChatTurn(role: .assistant, text: "甜品可以提前做好放冰箱，杨枝甘露或者银耳雪梨羹都合适，当天只要分碗。")
        ),
    ]
    let question = "你替我定的和晏初见面的地方是哪里？只回答地方。"
    #expect(Recall.turns(for: question, among: old.map(\.assistant)) == nil)
    let pair = try #require(Recall.turns(for: question, amongExchanges: old))
    #expect(pair[1].text.contains("你写过：我想约晏初见一面") && pair[1].text.contains("我说过：那就周三傍晚六点，花鸟市场门口"))
    #expect(!pair[1].text.contains("杨枝甘露"))
    #expect(Recall.turns(for: "我的车牌号是多少？", amongExchanges: old) == nil)
}

/// A measurement, not a check: WITHIN_RECALL_FILE names a JSON file with `exchanges` (user,
/// assistant), `queries`, `least`, `count` and `common`; what each query would be shown is written beside it
/// (`.out.json`, null when nothing is). DeviceChecks/compaction_set.py writes and reads the files.
@Test func recallOnAFile() throws {
    guard let path = ProcessInfo.processInfo.environment["WITHIN_RECALL_FILE"] else { return }
    struct Input: Decodable {
        struct Exchange: Decodable { let user: String; let assistant: String }
        let exchanges: [Exchange]; let queries: [String]; let least: Double; let count: Int; let common: Double
    }
    let input = try JSONDecoder().decode(Input.self, from: Data(contentsOf: URL(filePath: path)))
    let old = input.exchanges.map {
        (user: ChatTurn(role: .user, text: $0.user), assistant: ChatTurn(role: .assistant, text: $0.assistant))
    }
    let shown = input.queries.map {
        Recall.turns(for: $0, amongExchanges: old, count: input.count, least: input.least, common: input.common)?[1].text
    }
    try JSONEncoder().encode(shown).write(to: URL(filePath: path + ".out.json"))
}
