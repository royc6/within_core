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
            #expect((1...4).contains(chunk.tokens.count))
            #expect(chunk.function == (chunk.tokens.count == 1 ? "prefill_1" : "prefill_4"))
            #expect(!chunk.returnsLogits)
        }
    }
    let nine = try Gemma4PrefillPlan(tokenCount: 9, prefillLength: 4)
    #expect(nine.chunks.map(\.function) == ["prefill_4", "prefill_4", "step_1"])
    let four = try Gemma4PrefillPlan(tokenCount: 4, prefillLength: 4)
    #expect(four.chunks.map(\.function) == ["prefill_4", "step_1"])  // 3 tokens padded to the width
    #expect(four.chunks[0].tokens == 0..<3)
    let two = try Gemma4PrefillPlan(tokenCount: 2, prefillLength: 4)
    #expect(two.chunks.map(\.function) == ["prefill_1", "step_1"])
}

@Test func prefillPlanRejectsEmptyAndInvalidStaticWidths() {
    for (count, width) in [(0, 4), (-1, 4), (8, 0), (8, 1), (8, 257)] {
        #expect(throws: Gemma4RuntimeError.self) {
            try Gemma4PrefillPlan(tokenCount: count, prefillLength: width)
        }
    }
}

@Test func shortRemaindersRunAsOnePaddedWideCallWhenTheWidthFits() throws {
    // 1,024 tokens, width 64: 15 full calls, the 63-token remainder as one padded call, the held-back token.
    let padded = try Gemma4PrefillPlan(tokenCount: 1024, prefillLength: 64, availableSlots: 1024)
    #expect(padded.chunks.count == 17)
    #expect(padded.chunks.dropLast().allSatisfy { $0.function == "prefill_64" })
    #expect(padded.chunks[15].tokens == 960..<1023)
    #expect(padded.chunks.flatMap { Array($0.tokens) } == Array(0..<1024))
    // The padded call would write 64 slots from position 960: with one slot fewer it falls back to singles.
    let tight = try Gemma4PrefillPlan(tokenCount: 1024, prefillLength: 64, availableSlots: 1023)
    #expect(tight.chunks.filter { $0.function == "prefill_1" }.count == 63)
    #expect(tight.chunks.flatMap { Array($0.tokens) } == Array(0..<1024))
    // Fewer than paddedMinimum (width / 16, at least 2) remaining tokens stay single calls.
    #expect(Gemma4PrefillPlan.paddedMinimum(prefillLength: 16) == 2)
    #expect(Gemma4PrefillPlan.paddedMinimum(prefillLength: 64) == 4)
    #expect(Gemma4PrefillPlan.paddedMinimum(prefillLength: 128) == 8)
    let three = try Gemma4PrefillPlan(tokenCount: 64 + 3 + 1, prefillLength: 64)
    #expect(three.chunks.map(\.function) == ["prefill_64", "prefill_1", "prefill_1", "prefill_1", "step_1"])
    let four = try Gemma4PrefillPlan(tokenCount: 64 + 4 + 1, prefillLength: 64)
    #expect(four.chunks.map(\.function) == ["prefill_64", "prefill_64", "step_1"])
    #expect(four.chunks[1].tokens == 64..<68)
    let seven = try Gemma4PrefillPlan(tokenCount: 128 + 7 + 1, prefillLength: 128)
    #expect(seven.chunks.filter { $0.function == "prefill_1" }.count == 7)
    let eight = try Gemma4PrefillPlan(tokenCount: 128 + 8 + 1, prefillLength: 128)
    #expect(eight.chunks.map(\.function) == ["prefill_128", "prefill_128", "step_1"])
    // Width 4 (the plan tests' width) pads from 2 tokens: 4 + 2 + 1 tokens are two wide calls.
    let two = try Gemma4PrefillPlan(tokenCount: 4 + 2 + 1, prefillLength: 4)
    #expect(two.chunks.map(\.function) == ["prefill_4", "prefill_4", "step_1"])
}
