import Foundation
import Testing

@testable import WithinCoreBase

/// The contract files of the package the app's catalog installs, byte for byte. If the
/// runtime's decoding or validation drifts from them, the installed model stops loading.
private func shipped<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
    let url = try #require(Bundle.module.url(forResource: "Fixtures/ShippedPackage/\(name)", withExtension: nil))
    return try JSONDecoder().decode(type, from: Data(contentsOf: url))
}

@Test func shippedPackageContractDecodesAndValidates() throws {
    let configuration = try shipped(Gemma4RuntimeConfiguration.self, "gemma-runtime.json")
    try configuration.validate(requireE4B: true)
    let contract = try shipped(Gemma4ChunkContract.self, Gemma4ChunkContract.filename)
    try contract.validate(configuration: configuration, requireE4B: true)

    #expect(contract.contract == Gemma4ChunkContract.hostKV)
    #expect(contract.contextLimit == configuration.contextLimit)
    #expect(contract.chunks.count == 8)
    #expect(contract.chunks.map(\.runsPrefill) == [true, true, true, true, false, false, false, false])
    #expect(contract.chunks.last?.head.map { $0.parts * $0.rows } == configuration.vocabularySize)
    // Sliding-attention caches are rings of the window; full-attention caches span the context.
    for layer in configuration.cacheLayers {
        let expected =
            configuration.layerTypes[layer] == "sliding_attention"
            ? configuration.slidingWindow : configuration.contextLimit
        #expect(contract.cacheLength(layer: layer, configuration: configuration) == expected)
    }
}
