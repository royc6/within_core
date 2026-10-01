import Testing

@testable import WithinCoreBase

@Test func headlessPrefillCoversEveryTokenOnceAndHoldsBackExactlyOne() throws {
    for count in [1, 2, 3, 4, 5, 8, 9, 15, 511, 512, 513] {
        let plan = try Gemma4PrefillPlan(tokenCount: count, prefillLength: 4)
        #expect(plan.chunks.flatMap { Array($0.tokens) } == Array(0..<count))
        #expect(plan.chunks.filter(\.returnsLogits).count == 1)
        #expect(plan.chunks.last?.tokens == count - 1..<count)
        #expect(plan.chunks.last?.function == "step_1")
        for chunk in plan.chunks.dropLast() {
            #expect([1, 4].contains(chunk.tokens.count))
            #expect(chunk.function == "prefill_\(chunk.tokens.count)")
            #expect(!chunk.returnsLogits)
        }
    }
    let nine = try Gemma4PrefillPlan(tokenCount: 9, prefillLength: 4)
    #expect(nine.chunks.map(\.function) == ["prefill_4", "prefill_4", "step_1"])
    let four = try Gemma4PrefillPlan(tokenCount: 4, prefillLength: 4)
    #expect(four.chunks.map(\.function) == ["prefill_1", "prefill_1", "prefill_1", "step_1"])
}

@Test func prefillPlanRejectsEmptyAndInvalidStaticWidths() {
    for (count, width) in [(0, 4), (-1, 4), (8, 0), (8, 1), (8, 129)] {
        #expect(throws: Gemma4RuntimeError.self) {
            try Gemma4PrefillPlan(tokenCount: count, prefillLength: width)
        }
    }
}
