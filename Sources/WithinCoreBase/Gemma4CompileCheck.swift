/// Whether the decode steps after a load say the Neural Engine compiler left one chunk degraded
/// (REPORT.md Section 7.2). A degraded chunk specializes in a fraction of its usual time, its functions
/// prepare in hundreds of milliseconds instead of about ten, and every step then runs it 3 to 20 times
/// slower than its siblings; the content-keyed cache keeps that result for every later load. On the
/// 1.1 measurements a layer chunk's step is within 1.5 times the median of the layer chunks (chunk 0
/// does the most) and the head chunk's under 0.6 times it, while the degraded ones were 3.2, 24 and,
/// for the head, 13 times the median. `stepMs` is one `step_1` call's time per chunk in contract
/// order; the last entry is the head (the contract allows one head chunk, at the end).
///
/// Not the first steps: a chunk's first call after a load pays for paging its weights in, so on the
/// first decode step the chunks the prefill calls did not use take 20 to 25 ms against 4 to 6 for the
/// others, and the step after a recompile is cold again. The runtime judges the medians of the third
/// to fifth steps (`init(steps:)`).
public struct Gemma4CompileCheck: Sendable {
    /// Decode steps the runtime skips after a load before it samples.
    public static let coldSteps = 2
    /// Steps sampled after the cold ones; the per-chunk medians are judged.
    public static let sampledSteps = 3

    /// The chunks over their limit, in contract order (two chunks degraded together once).
    public let degraded: [Int]
    /// The median of the layer chunks' times, 0 when there is no judgement.
    public let medianMs: Double

    /// Judges the per-chunk medians of several steps' times; steps of different lengths give no judgement.
    public init(steps: [[Double]], layerRatio: Double = 2.5, headRatio: Double = 2) {
        guard let first = steps.first, steps.allSatisfy({ $0.count == first.count }) else {
            self.init(stepMs: [], layerRatio: layerRatio, headRatio: headRatio)
            return
        }
        let medians = (0..<first.count).map { chunk -> Double in
            let values = steps.map { $0[chunk] }.sorted()
            let middle = values.count / 2
            return values.count % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2
        }
        self.init(stepMs: medians, layerRatio: layerRatio, headRatio: headRatio)
    }

    public init(stepMs: [Double], layerRatio: Double = 2.5, headRatio: Double = 2) {
        guard stepMs.count >= 3, stepMs.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            degraded = []
            medianMs = 0
            return
        }
        let layers = stepMs.dropLast().sorted()
        let middle = layers.count / 2
        let median = layers.count % 2 == 1 ? layers[middle] : (layers[middle - 1] + layers[middle]) / 2
        medianMs = median
        guard median > 0 else {
            degraded = []
            return
        }
        degraded = stepMs.indices.filter { index in
            stepMs[index] > median * (index == stepMs.count - 1 ? headRatio : layerRatio)
        }
    }
}
