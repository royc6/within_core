import Foundation
import Testing

@testable import WithinCoreBase

@Test func teacherForcedScoreMatchesLogSoftmax() throws {
    let logits: [Float16] = [1, 3, 2, -4]
    var score = TeacherForcingScore()
    try score.add(logits: logits, token: 2)
    try score.add(logits: logits, token: 1)
    let denominator = log([1.0, 3, 2, -4].map { exp($0) }.reduce(0, +))
    #expect(abs(score.nllSum - ((denominator - 2) + (denominator - 3))) < 1e-9)
    #expect(score.tokens == 2)
    #expect(score.greedyMatches == 1)
}

@Test func teacherForcedScoreRejectsUnusableLogits() {
    var score = TeacherForcingScore()
    #expect(throws: TeacherForcingError.invalidLogits) { try score.add(logits: [1, 2], token: 2) }
    #expect(throws: TeacherForcingError.invalidLogits) { try score.add(logits: [1, .infinity], token: 0) }
    #expect(score == TeacherForcingScore())
}

@Test func quantQualityFixtureMustFitTheContext() throws {
    let item = TeacherForcingFixture.Item(id: "en-01", subset: "en", prompt: [2, 5, 7], continuation: [9, 1])
    var fixture = TeacherForcingFixture(
        version: TeacherForcingFixture.version, setSHA256: "a", referencesSHA256: "b",
        items: [item])
    // Three prompt tokens plus one fed continuation token fill a four-slot context.
    try fixture.validate(vocabularySize: 10, contextLimit: 4)
    #expect(throws: TeacherForcingError.invalidFixture) { try fixture.validate(vocabularySize: 10, contextLimit: 3) }
    #expect(throws: TeacherForcingError.invalidFixture) { try fixture.validate(vocabularySize: 9, contextLimit: 4) }
    fixture.items.append(item)
    #expect(throws: TeacherForcingError.invalidFixture) { try fixture.validate(vocabularySize: 10, contextLimit: 4) }
}
