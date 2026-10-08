import Testing

@testable import WithinCoreBase

// The step_1 chunk times (ms) of the 1.1 model-card runs: Full, Balanced and Lite 32K at 64, Lite 4K at 128.
@Test func normalFirstDecodeStepsFlagNoChunk() {
    for steps in [
        [9.03, 7.63, 7.63, 7.63, 7.23, 7.23, 7.23, 3.46],
        [6.58, 5.64, 5.65, 5.65, 5.27, 5.26, 5.26, 3.42],
        [11.39, 9.86, 9.91, 9.9, 9.53, 9.51, 9.51, 2.53],
        [4.63, 4.0, 4.0, 4.0, 3.85, 3.85, 3.85, 2.2],
    ] {
        #expect(Gemma4CompileCheck(stepMs: steps).degraded == [])
    }
}

// The three degraded compiles of the width measurements: Full 64's chunk 0, Balanced 64's head, Lite 32K 64's chunk 0.
@Test func degradedCompilesOfTheMeasurementsAreFlagged() {
    #expect(Gemma4CompileCheck(stepMs: [24.01, 7.81, 7.57, 7.57, 7.16, 7.18, 7.19, 3.46]).degraded == [0])
    #expect(Gemma4CompileCheck(stepMs: [7.36, 5.57, 5.56, 5.56, 5.2, 5.2, 5.23, 71.8]).degraded == [7])
    let lite = Gemma4CompileCheck(stepMs: [234.02, 15.09, 11.49, 9.85, 9.49, 9.51, 9.5, 2.53])
    #expect(lite.degraded == [0])
    #expect(lite.medianMs == 9.85)
}

@Test func theWorstChunkIsNamedAndTheHeadHasItsOwnLimit() {
    #expect(Gemma4CompileCheck(stepMs: [30, 10, 10, 10, 10, 10, 10, 40]).degraded == [0, 7])  // both over their limits
    #expect(Gemma4CompileCheck(stepMs: [10, 10, 10, 10, 10, 10, 10, 19]).degraded == [])  // the head under 2×
    #expect(Gemma4CompileCheck(stepMs: [10, 10, 10, 10, 10, 10, 10, 21]).degraded == [7])
    #expect(Gemma4CompileCheck(stepMs: [24, 10, 10, 10, 10, 10, 10, 5]).degraded == [])  // a layer under 2.5×
}

// Lite 4K on the phone: the first decode step after the load is cold for the chunks the prefill calls did
// not use (chunks 4-6 at 20-25 ms), the step after a recompile is cold again, and from the third step on every
// chunk is near 4 ms. The medians of three steps are judged, not one step.
@Test func theColdFirstStepsAreNotJudgedAndTheMediansOfLaterStepsAre() {
    let first = [9.903, 5.778, 5.591, 6.110, 25.557, 20.654, 25.084, 3.334]
    #expect(Gemma4CompileCheck(stepMs: first).degraded == [4, 6])  // why the first steps are skipped
    let later = [
        [5.281, 4.651, 4.465, 4.296, 4.219, 4.137, 6.346, 2.556],
        [4.967, 4.242, 4.296, 4.313, 4.341, 4.047, 3.908, 2.554],
        [5.374, 4.225, 4.300, 4.306, 4.018, 3.948, 3.930, 2.582],
    ]
    let check = Gemma4CompileCheck(steps: later)
    #expect(check.degraded == [])
    // The layer chunks' medians: 5.281, 4.242, 4.300, 4.306, 4.219, 4.047, 3.930.
    #expect(abs(check.medianMs - 4.242) < 1e-9)
    #expect(Gemma4CompileCheck(steps: [[1, 1, 1, 1], [1, 1, 1]]).degraded == [])
    #expect(Gemma4CompileCheck(steps: []).degraded == [])
    let hiccup = [[5, 4, 4, 4, 4, 4, 4, 2.5], [5, 4, 40, 4, 4, 4, 4, 2.5], [5, 4, 4, 4, 4, 4, 4, 2.5]]
    #expect(Gemma4CompileCheck(steps: hiccup).degraded == [])  // one slow step is not a degraded compile
    let persistent = [[5, 4, 40, 4, 4, 4, 4, 2.5], [5, 4, 41, 4, 4, 4, 4, 2.5], [5, 4, 39, 4, 4, 4, 4, 2.5]]
    #expect(Gemma4CompileCheck(steps: persistent).degraded == [2])
}

// Lite 4K on the phone: chunks 6 and 7 compiled degraded in the same load (2.7 s instead of 22 for chunk 6; the head
// in its usual 10 s), 214 and 325 ms a step instead of 4 and 2.5, and the recompile of the head 7 s later was degraded
// again. Both are named.
@Test func twoChunksDegradedTogetherAreBothNamed() {
    let steps = [
        [7.739, 4.447, 4.336, 4.352, 4.379, 4.345, 214.452, 324.832],
        [12.686, 4.788, 4.782, 4.585, 4.159, 4.098, 213.636, 325.892],
        [5.748, 4.338, 4.280, 4.297, 4.275, 4.126, 214.093, 326.744],
    ]
    #expect(Gemma4CompileCheck(steps: steps).degraded == [6, 7])
}

@Test func tooFewChunksOrBadValuesFlagNothing() {
    #expect(Gemma4CompileCheck(stepMs: []).degraded == [])
    #expect(Gemma4CompileCheck(stepMs: [50, 1]).degraded == [])
    #expect(Gemma4CompileCheck(stepMs: [0, 0, 0, 5]).degraded == [])
    #expect(Gemma4CompileCheck(stepMs: [1, 1, 1, .nan]).degraded == [])
    #expect(Gemma4CompileCheck(stepMs: [1, -1, 1, 9]).degraded == [])
}
