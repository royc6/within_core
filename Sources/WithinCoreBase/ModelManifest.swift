import Foundation

/// The runtime a package is built for: Core AI with the chunks specialized for the Neural Engine.
public enum ModelBackend: String, Codable, Sendable { case coreAIANE }

/// One file of the package and the size and SHA-256 it must have.
public struct ModelFile: Codable, Equatable, Sendable {
    /// Path relative to the package folder and to `baseURL`.
    public var path: String
    /// Size in bytes.
    public var size: Int64
    /// Lowercase hex SHA-256 of the contents.
    public var sha256: String
    /// A file entry.
    public init(path: String, size: Int64, sha256: String) { self.path = path; self.size = size; self.sha256 = sha256 }
}

/// The downloadable package: where its files are and what each must hash to. Installers check every
/// file against it before the engine loads the folder.
public struct ModelManifest: Codable, Equatable, Sendable {
    /// The manifest format version; 1.
    public var schemaVersion: Int
    /// The model, `gemma-4-e4b`.
    public var modelID: String
    /// The package revision, also its tag in the hosting repository.
    public var revision: String
    /// The runtime the package is built for.
    public var backend: ModelBackend
    /// HTTPS location the file paths are relative to, fixed to the revision.
    public var baseURL: URL
    /// Tokens of prompt, history and reply the package holds.
    public var contextLimit: Int
    /// Every file of the package.
    public var files: [ModelFile]
    /// The model license.
    public var licenseURL: URL

    /// The package size.
    public var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }
    /// A folder name for this model and revision.
    public var installationID: String { modelID + "-" + revision }

    /// A manifest from explicit values.
    public init(
        schemaVersion: Int = 1, modelID: String = "gemma-4-e4b", revision: String,
        backend: ModelBackend, baseURL: URL, contextLimit: Int, files: [ModelFile], licenseURL: URL
    ) {
        self.schemaVersion = schemaVersion; self.modelID = modelID; self.revision = revision
        self.backend = backend; self.baseURL = baseURL; self.contextLimit = contextLimit
        self.files = files; self.licenseURL = licenseURL
    }

    /// Checks the manifest is safe to install from: an HTTPS base without credentials, and plain,
    /// unique, relative file paths with sane sizes and hex digests.
    public func validate() throws {
        guard schemaVersion == 1, modelID == "gemma-4-e4b", contextLimit >= 512,
            !revision.isEmpty, revision.count <= 80,
            revision.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }),
            baseURL.scheme == "https", baseURL.host != nil, baseURL.user == nil, baseURL.password == nil,
            baseURL.query == nil, baseURL.fragment == nil, licenseURL.scheme == "https", !files.isEmpty
        else { throw ManifestError.invalid }
        var paths = Set<String>()
        var total: Int64 = 0
        for file in files {
            let components = file.path.split(separator: "/", omittingEmptySubsequences: false)
            guard !components.isEmpty, components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                !file.path.contains("\\"), !file.path.contains("%"), !file.path.contains(":"),
                !file.path.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
                file.size > 0, file.sha256.count == 64, file.sha256.allSatisfy(\.isHexDigit),
                paths.insert(file.path.precomposedStringWithCanonicalMapping.lowercased()).inserted,
                file.size < 30_000_000_000
            else { throw ManifestError.invalid }
            let sum = total.addingReportingOverflow(file.size)
            guard !sum.overflow, sum.partialValue < 40_000_000_000 else { throw ManifestError.invalid }
            total = sum.partialValue
        }
        let reserved = Set(["installed-manifest.json", "download-state.json"])
        guard paths.isDisjoint(with: reserved) else { throw ManifestError.invalid }
        // A file may not also be a parent directory of another manifest entry.
        for path in paths {
            let parts = path.split(separator: "/")
            for count in 1..<parts.count {
                guard !paths.contains(parts.prefix(count).joined(separator: "/")) else { throw ManifestError.invalid }
            }
        }
    }
}

/// Failures of a manifest or of the files installed from it.
public enum ManifestError: Error, LocalizedError {
    case invalid, integrityFailed, insufficientStorage
    public var errorDescription: String? {
        switch self {
        case .invalid: "The model download information is invalid."
        case .integrityFailed: "The model file could not be verified. Please download it again."
        case .insufficientStorage: "There is not enough free space to install this model."
        }
    }
}
