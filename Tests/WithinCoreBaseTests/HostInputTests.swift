import Foundation
import Testing

@testable import WithinCoreBase

/// The host's per-call inputs (REPORT.md Section 7.2): the vectorised 2-bit gather and the mask
/// builders must give exactly what the one-value-at-a-time versions gave.
private func packedTable(rows: Int, columns: Int, columnsPerScale: Int, seed: UInt64) -> (
    codes: Data, scales: Data, descriptor: Gemma4EmbeddingDescriptor
) {
    var state = seed
    func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state >> 33
    }
    var codes = Data(count: rows * columns / 4)
    for index in codes.indices { codes[index] = UInt8(truncatingIfNeeded: next()) }
    var scales = Data()
    for _ in 0..<(rows * columns / columnsPerScale) {
        let value = Float(next() % 1000 + 1) / 4096
        withUnsafeBytes(of: value.bitPattern.littleEndian) { scales.append(contentsOf: $0) }
    }
    let descriptor = Gemma4EmbeddingDescriptor(
        file: "t.int2", format: .int2, rows: rows, columns: columns,
        scales: .init(file: "t.scales.f32", columnsPerScale: columnsPerScale))
    return (codes, scales, descriptor)
}

/// The gather as it was written for 1.1: one value at a time.
private func referenceGather(
    codes: Data, scales: Data, descriptor: Gemma4EmbeddingDescriptor, tokens: [Int], layers: Int, scale: Float
) -> [Float16] {
    let channels = descriptor.columns / layers, file = descriptor.scales!
    let groups = descriptor.columns / file.columnsPerScale
    var result = [Float16](repeating: 0, count: tokens.count * descriptor.columns)
    for (position, token) in tokens.enumerated() {
        for layer in 0..<layers {
            for channel in 0..<channels {
                let column = layer * channels + channel
                let index = token * descriptor.columns + column
                let code = Int((codes[index / 4] >> UInt8(2 * (index % 4))) & 3) - 2
                let offset = (token * groups + column / file.columnsPerScale) * 4
                let factor = Float(bitPattern: UInt32(littleEndian: scales.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }))
                result[(channel * layers + layer) * tokens.count + position] = Float16(Float(code) * factor * scale)
            }
        }
    }
    return result
}

@Test func twoBitGatherMatchesTheOneValueAtATimeVersion() throws {
    let folder = FileManager.default.temporaryDirectory.appending(path: "within-host-inputs-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for (columns, per, layers) in [(64, 64, 1), (42 * 8, 8, 42), (40, 10, 2)] {
        let (codes, scales, descriptor) = packedTable(rows: 13, columns: columns, columnsPerScale: per, seed: UInt64(columns))
        try codes.write(to: folder.appending(path: descriptor.file))
        try scales.write(to: folder.appending(path: descriptor.scales!.file))
        let table = try Gemma4Embeddings(folder: folder, descriptor: descriptor)
        for tokens in [[0], [12, 0, 7], Array(repeating: 5, count: 128), (0..<256).map { $0 % 13 }] {
            let expected = referenceGather(
                codes: codes, scales: scales, descriptor: descriptor, tokens: tokens, layers: layers, scale: 1.5)
            #expect(try table.gather(tokens: tokens, layers: layers, scale: 1.5) == expected)
        }
    }
}

private func referenceMask(_ c: Gemma4RuntimeConfiguration, step: Int, length: Int, global: Bool) -> [Float16] {
    let width = c.contextLimit + length
    var values = [Float16](repeating: -10_000, count: length * width)
    for query in 0..<length {
        let position = step + query
        let first = global ? 0 : max(0, position - c.slidingWindow + 1)
        if first < step { for key in first..<step { values[query * width + key] = 0 } }
        for fresh in 0...query where step + fresh >= first { values[query * width + c.contextLimit + fresh] = 0 }
    }
    return values
}

private func referenceTableMask(_ table: Gemma4SlotTable, length: Int) -> [Float16] {
    let width = table.capacity + length
    var values = [Float16](repeating: -10_000, count: length * width)
    for query in 0..<length {
        for slot in 0..<table.capacity where table.positions[slot] >= 0 { values[query * width + slot] = 0 }
        for fresh in 0...query { values[query * width + table.capacity + fresh] = 0 }
    }
    return values
}

@Test func masksMatchTheOneValueAtATimeVersions() throws {
    let c = Gemma4RuntimeConfiguration(
        schemaVersion: 1, hiddenSize: 32, vocabularySize: 64,
        layerTypes: (0..<42).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" },
        sharedLayerCount: 18, perLayerInputSize: 8, keyValueHeads: 2,
        localHeadDimension: 8, globalHeadDimension: 16, localTheta: 10_000, globalTheta: 1_000_000,
        globalRotaryFraction: 0.25, slidingWindow: 5, contextLimit: 64, prefillLength: 16,
        embeddings: .init(file: "embeddings.f16", format: .float16, rows: 64, columns: 32),
        perLayerEmbeddings: .init(file: "ple.f16", format: .float16, rows: 64, columns: 336))
    for step in [0, 1, 3, 7, 40, 48] {
        for length in [1, 2, 5, 16] where step + length <= 64 {
            for global in [false, true] {
                #expect(try c.mask(step: step, length: length, global: global) == referenceMask(c, step: step, length: length, global: global))
            }
        }
    }
    var table = Gemma4SlotTable(capacity: 40)
    for length in [1, 7, 16] {
        #expect(table.mask(length: length) == referenceTableMask(table, length: length))
    }
    _ = try table.allocate(position: 0, count: 30)
    table.evict([2, 3, 11, 29])
    _ = try table.allocate(position: 30, count: 6)
    for length in [1, 7, 16] {
        #expect(table.mask(length: length) == referenceTableMask(table, length: length))
    }
}
