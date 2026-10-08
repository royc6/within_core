import Foundation

/// The package a device runs, one of four (REPORT.md Section 7.1): the v1 full package, the balanced 4K
/// re-export of the same weights, and the two lite packages from Google's QAT checkpoint.
public enum ModelTier: String, Codable, CaseIterable, Sendable {
    case full, balance, liteLong, liteFast
}

/// One downloadable package and the tier it serves. `manifest` is the package's own
/// `installed-manifest.json`; the installer still requires the two to be equal.
public struct CatalogPackage: Codable, Equatable, Sendable {
    public var tier: ModelTier
    public var manifest: ModelManifest
    public init(tier: ModelTier, manifest: ModelManifest) { self.tier = tier; self.manifest = manifest }
}

/// The app's bundled `ModelCatalog.json` (REPORT.md Section 7.1): `catalogVersion` 2 lists
/// one package per tier and names the default; a v1 catalog is a bare manifest and reads as one
/// full-tier package, so a v1 user's installed package stays equal to its catalog entry.
public struct ModelCatalog: Equatable, Sendable {
    public var catalogVersion: Int
    public var defaultTier: ModelTier
    public var packages: [CatalogPackage]

    public init(catalogVersion: Int = 2, defaultTier: ModelTier, packages: [CatalogPackage]) {
        self.catalogVersion = catalogVersion
        self.defaultTier = defaultTier
        self.packages = packages
    }

    private struct Version2: Codable {
        var catalogVersion: Int
        var defaultTier: ModelTier
        var packages: [CatalogPackage]
    }

    /// Reads a v2 catalog, or a v1 catalog (a bare manifest) as one full-tier package.
    public static func decode(_ data: Data) throws -> ModelCatalog {
        let decoder = JSONDecoder()
        if let v2 = try? decoder.decode(Version2.self, from: data) {
            return ModelCatalog(catalogVersion: v2.catalogVersion, defaultTier: v2.defaultTier, packages: v2.packages)
        }
        let manifest = try decoder.decode(ModelManifest.self, from: data)
        return ModelCatalog(catalogVersion: 1, defaultTier: .full, packages: [CatalogPackage(tier: .full, manifest: manifest)])
    }

    /// The v2 JSON text, sorted keys, as `package_model.py catalog` writes it.
    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(Version2(catalogVersion: catalogVersion, defaultTier: defaultTier, packages: packages))
    }

    /// 1–4 packages with distinct tiers and installation IDs, the default among them, every manifest
    /// valid, and one size per SHA-256 across packages (shared files are deduplicated by both).
    public func validate() throws {
        guard catalogVersion == 1 || catalogVersion == 2, (1...4).contains(packages.count),
            Set(packages.map(\.tier)).count == packages.count,
            Set(packages.map(\.manifest.installationID)).count == packages.count,
            packages.contains(where: { $0.tier == defaultTier })
        else { throw ManifestError.invalid }
        var sizes: [String: Int64] = [:]
        for package in packages {
            try package.manifest.validate()
            for file in package.manifest.files {
                if let size = sizes[file.sha256], size != file.size { throw ManifestError.invalid }
                sizes[file.sha256] = file.size
            }
        }
    }

    public func package(for tier: ModelTier) -> CatalogPackage? { packages.first { $0.tier == tier } }
    public var defaultPackage: CatalogPackage { package(for: defaultTier) ?? packages[0] }
}
