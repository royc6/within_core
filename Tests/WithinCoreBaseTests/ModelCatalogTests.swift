import Foundation
import Testing

@testable import WithinCoreBase

private func manifest(_ revision: String, context: Int, files: [(String, Int64, String)]) -> ModelManifest {
    ModelManifest(
        schemaVersion: 1, modelID: "gemma-4-e4b", revision: revision, backend: .coreAIANE,
        baseURL: URL(string: "https://example.com/\(revision)/")!, contextLimit: context,
        files: files.map { ModelFile(path: $0.0, size: $0.1, sha256: $0.2) },
        licenseURL: URL(string: "https://example.com/LICENSE")!)
}
private let sha = String(repeating: "a", count: 64), shb = String(repeating: "b", count: 64)

@Test func v1CatalogIsOneFullPackageAndV2RoundTrips() throws {
    let full = manifest("lut6-c16384-20261001-01", context: 16384, files: [("chunk-00.aimodel/main.mlirb", 10, sha)])
    let v1 = try ModelCatalog.decode(JSONEncoder().encode(full))
    #expect(v1.catalogVersion == 1 && v1.defaultTier == .full && v1.packages == [CatalogPackage(tier: .full, manifest: full)])
    try v1.validate()
    let lite = manifest("qat-lut-srq-p128-c4096-20261002-01", context: 4096, files: [("embeddings.int2", 20, shb)])
    let v2 = ModelCatalog(defaultTier: .liteFast, packages: [.init(tier: .full, manifest: full), .init(tier: .liteFast, manifest: lite)])
    try v2.validate()
    let decoded = try ModelCatalog.decode(try v2.encoded())
    #expect(decoded == v2 && decoded.defaultPackage.manifest == lite && decoded.package(for: .full)?.manifest == full)
}

@Test func catalogRejectsDuplicateTiersMissingDefaultAndConflictingSizes() throws {
    let a = manifest("a-c4096-01", context: 4096, files: [("x", 10, sha)])
    let b = manifest("b-c32000-01", context: 32000, files: [("y", 11, sha)])  // same hash, other size
    #expect(throws: ManifestError.self) {
        try ModelCatalog(defaultTier: .liteFast, packages: [.init(tier: .liteFast, manifest: a), .init(tier: .liteFast, manifest: b)]).validate()
    }
    #expect(throws: ManifestError.self) {
        try ModelCatalog(defaultTier: .balance, packages: [.init(tier: .liteFast, manifest: a)]).validate()
    }
    #expect(throws: ManifestError.self) {
        try ModelCatalog(defaultTier: .liteFast, packages: [.init(tier: .liteFast, manifest: a), .init(tier: .liteLong, manifest: b)]).validate()
    }
    #expect(throws: ManifestError.self) { try ModelCatalog(defaultTier: .full, packages: []).validate() }
}
