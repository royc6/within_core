import Foundation
import Testing

@testable import WithinCoreBase

@Test func thoughtChannelAndControlTokensNeverBecomeAnswerText() {
    var filter = GemmaOutputFilter(channelStart: 100, channelEnd: 101, thinkMarker: 98)
    let visible = [98, 100, 200, 201, 101, 300, 301, 100, 202, 101, 302].filter { filter.accepts($0) }
    #expect(visible == [300, 301, 302])
    #expect(!filter.isThinking)
    let accepted = filter.accepts(100)
    #expect(!accepted)
    #expect(filter.isThinking)
}
