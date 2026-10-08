import Foundation
import Testing

@testable import WithinCoreBase

private func configuration() throws -> Gemma4RuntimeConfiguration {
    try JSONDecoder().decode(
        Gemma4RuntimeConfiguration.self,
        from: Data(
            #"""
            {"schemaVersion":1,"hiddenSize":32,"vocabularySize":64,
             "layerTypes":["sliding_attention","full_attention","sliding_attention","full_attention","sliding_attention","full_attention"],
             "sharedLayerCount":2,"perLayerInputSize":8,"keyValueHeads":2,
             "localHeadDimension":8,"globalHeadDimension":16,"localTheta":10000,"globalTheta":1000000,
             "globalRotaryFraction":0.25,"slidingWindow":16,"contextLimit":32,"prefillLength":4,
             "embeddings":{"file":"embeddings.f16","format":"float16","rows":64,"columns":32},
             "perLayerEmbeddings":{"file":"ple.f16","format":"float16","rows":64,"columns":48}}
            """#.utf8))
}

@Test func gemmaContractRejectsWrongModelAndUnsafeShapes() throws {
    var config = try configuration()
    try config.validate(requireE4B: false)
    #expect(throws: Gemma4RuntimeError.self) { try config.validate() }
    config.sharedLayerCount = 6
    #expect(throws: Gemma4RuntimeError.self) { try config.validate(requireE4B: false) }
    config = try configuration(); config.embeddings.columns = Int.max
    #expect(throws: Gemma4RuntimeError.self) { try config.validate(requireE4B: false) }
    config = try configuration(); config.embeddings.file = "../weights"
    #expect(throws: Gemma4RuntimeError.self) { try config.validate(requireE4B: false) }
    config = try configuration(); config.globalTheta = .nan
    #expect(throws: Gemma4RuntimeError.self) { try config.validate(requireE4B: false) }
}

@Test func gemmaMasksExcludeFutureAndExpiredPositions() throws {
    let config = try configuration()
    let local = try config.mask(step: 17, length: 4, global: false)
    let global = try config.mask(step: 17, length: 4, global: true)
    // Query at 17 sees old keys 2...16 and its fresh key at physical slot 32.
    #expect(local.enumerated().filter { $0.offset < 36 && $0.element == 0 }.map(\.offset) == Array(2...16) + [32])
    // Last query (position 20) sees keys 5...16 and all four fresh keys.
    #expect(
        Array(local[108..<144]).enumerated().filter { $0.element == 0 }.map(\.offset) == Array(5...16) + Array(32...35))
    #expect(global[0] == 0 && local[0] < 0)
    #expect(global[17] < 0 && global[31] < 0 && global[33] < 0)
    #expect(throws: Gemma4RuntimeError.self) { try config.mask(step: 31, length: 4, global: true) }
}

@Test func gemmaGlobalRopePreservesUnrotatedAxes() throws {
    let config = try configuration()
    let rope = try config.rotary(step: 17, length: 4, global: true)
    // Proportional RoPE rotates 2 of the 8 axis pairs for global head_dim 16.
    for axis in [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15] {
        #expect(Array(rope.cos[axis * 4..<axis * 4 + 4]) == [1, 1, 1, 1])
        #expect(Array(rope.sin[axis * 4..<axis * 4 + 4]) == [0, 0, 0, 0])
    }
    #expect(rope.cos[0] == Float16(-0.275163338))
    #expect(rope.sin[0] == Float16(-0.961397492))
    #expect(Array(rope.cos[0..<8]) == Array(rope.cos[32..<40]))
}

@Test func gemmaInt2EmbeddingLookupDecodesGoogleQATCodes() throws {
    let folder = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    // Rows [-2, -1, 0, 1, 1, 0, -1, -2] and [0, 0, 0, 0, 1, 1, 1, 1]: code + 2, four per byte, low bits first.
    try Data([228, 27, 170, 255]).write(to: folder.appending(path: "table.int2"))
    let factors: [Float] = [0.5, 2, 1, 0.25]
    try factors.withUnsafeBytes { try Data($0).write(to: folder.appending(path: "table.scales.f32")) }
    let scales = Gemma4EmbeddingDescriptor.Scales(file: "table.scales.f32", columnsPerScale: 4)
    let descriptor = Gemma4EmbeddingDescriptor(file: "table.int2", format: .int2, rows: 2, columns: 8, scales: scales)
    let table = try Gemma4Embeddings(folder: folder, descriptor: descriptor)
    #expect(
        try table.gather(tokens: [1, 0], layers: 2, scale: 3)
            == [0, -3, 0.75, 6, 0, -1.5, 0.75, 0, 0, 0, 0.75, -6, 0, 1.5, 0.75, -12])
    #expect(throws: Gemma4RuntimeError.self) {
        try Gemma4EmbeddingDescriptor(file: "table.int2", format: .int2, rows: 2, columns: 8).validate()
    }
    #expect(throws: Gemma4RuntimeError.self) {
        try Gemma4EmbeddingDescriptor(file: "table.f16", format: .float16, rows: 2, columns: 8, scales: scales).validate()
    }
    try Data([0]).write(to: folder.appending(path: "table.scales.f32"), options: .atomic)
    #expect(throws: Gemma4RuntimeError.self) { try Gemma4Embeddings(folder: folder, descriptor: descriptor) }
}

private func temporaryFolder() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
}

@Test func gemmaEmbeddingLookupHandlesPLELayoutAndSubnormal() throws {
    let folder = try temporaryFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let values: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8, .leastNonzeroMagnitude, -1, -2, -3]
    try values.withUnsafeBytes { try Data($0).write(to: folder.appending(path: "table.f16")) }
    let descriptor = Gemma4EmbeddingDescriptor(file: "table.f16", format: .float16, rows: 3, columns: 4)
    let table = try Gemma4Embeddings(folder: folder, descriptor: descriptor)
    #expect(try table.gather(tokens: [1, 0], layers: 2, scale: 1) == [5, 1, 7, 3, 6, 2, 8, 4])
    #expect(try table.gather(tokens: [2], scale: 1).first == .leastNonzeroMagnitude)
    #expect(throws: Gemma4RuntimeError.self) { try table.gather(tokens: [-1], scale: 1) }
    #expect(throws: Gemma4RuntimeError.self) { try table.gather(tokens: [3], scale: 1) }
    try Data([0]).write(to: folder.appending(path: "table.f16"), options: .atomic)
    #expect(throws: Gemma4RuntimeError.self) { try Gemma4Embeddings(folder: folder, descriptor: descriptor) }
}
