import Foundation
import Testing

@testable import WithinCoreBase

@Test(arguments: [
    "../secret", "/absolute", "a/../b", "a//b", "a%2fb", "a\\b", "installed-manifest.json", "INSTALLED-MANIFEST.JSON",
])
func manifestRejectsUnsafePaths(path: String) {
    let manifest = ModelManifest(
        revision: "abc123", backend: .coreAIANE,
        baseURL: URL(string: "https://example.com/models/abc123/")!, contextLimit: 2048,
        files: [.init(path: path, size: 10, sha256: String(repeating: "a", count: 64))],
        licenseURL: URL(string: "https://example.com/license")!)
    #expect(throws: ManifestError.self) { try manifest.validate() }
}

@Test func manifestRejectsFileDirectoryCollision() {
    let manifest = ModelManifest(
        revision: "abc123", backend: .coreAIANE,
        baseURL: URL(string: "https://example.com/models/abc123/")!, contextLimit: 2048,
        files: ["model", "model/weights.bin"].map {
            .init(path: $0, size: 10, sha256: String(repeating: "a", count: 64))
        },
        licenseURL: URL(string: "https://example.com/license")!)
    #expect(throws: ManifestError.self) { try manifest.validate() }
}
