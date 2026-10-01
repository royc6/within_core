import CryptoKit
import Foundation
import Observation
import Synchronization
import WithinCoreBase

/// The model package on this iPhone: the bundled catalog names every file with its size and SHA-256,
/// and `download()` fetches the missing ones into Application Support, checking each before it is
/// kept. Nothing is fetched until the button is tapped, and cellular data is off unless allowed.
@MainActor @Observable final class ModelStore {
    enum State: Equatable {
        case notInstalled
        case downloading(file: String, fraction: Double)
        case checking(file: String)
        case installed
        case failed(String)
    }

    private(set) var state: State = .notInstalled
    var allowsCellular = false
    let manifest: ModelManifest
    /// Where the installed package lives; `Gemma4Engine` loads it from here.
    var folder: URL { root.appending(path: manifest.installationID, directoryHint: .isDirectory) }

    private let root = URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory)
    private var staging: URL { root.appending(path: manifest.installationID + ".partial", directoryHint: .isDirectory) }
    private var marker: URL { folder.appending(path: ".installed") }
    private var job: Task<Void, Never>?

    init() throws {
        guard let url = Bundle.main.url(forResource: "ModelCatalog", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        manifest = try JSONDecoder().decode(ModelManifest.self, from: Data(contentsOf: url))
        try manifest.validate()
        if FileManager.default.fileExists(atPath: marker.path) { state = .installed }
    }

    var totalBytes: Int64 { manifest.totalBytes }

    func download() {
        guard job == nil, state != .installed else { return }
        let allowsCellular = allowsCellular
        job = Task {
            defer { job = nil }
            do {
                try await fetchAll(allowsCellular: allowsCellular)
                state = .installed
            } catch is CancellationError {
                state = .notInstalled
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() { job?.cancel() }

    private func fetchAll(allowsCellular: Bool) async throws {
        let files = FileManager.default
        try files.createDirectory(at: staging, withIntermediateDirectories: true)
        var excluded = root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try excluded.setResourceValues(values)
        let free =
            try staging.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage ?? 0
        let missing = manifest.files.filter { !Self.hasSize($0, at: staging.appending(path: $0.path)) }
        let needed = missing.reduce(Int64(0)) { $0 + $1.size } + 1_000_000_000
        guard free > needed else {
            throw DownloadError.notEnoughSpace
        }
        let configuration = URLSessionConfiguration.default
        configuration.allowsCellularAccess = allowsCellular
        configuration.allowsExpensiveNetworkAccess = allowsCellular
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        for file in manifest.files {
            let destination = staging.appending(path: file.path)
            // A file kept from an earlier attempt was checked before it was kept.
            if Self.hasSize(file, at: destination) { continue }
            try files.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            state = .downloading(file: file.path, fraction: 0)
            let progress = Progress { [weak self] fraction in
                Task { @MainActor in self?.state = .downloading(file: file.path, fraction: fraction) }
            }
            let (temporary, response) = try await session.download(
                from: manifest.baseURL.appending(path: file.path), delegate: progress)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw DownloadError.server }
            state = .checking(file: file.path)
            guard try await Self.sha256(of: temporary) == file.sha256.lowercased() else {
                try? files.removeItem(at: temporary)
                throw DownloadError.integrity(file.path)
            }
            try? files.removeItem(at: destination)
            try files.moveItem(at: temporary, to: destination)
        }
        try? files.removeItem(at: folder)
        try files.moveItem(at: staging, to: folder)
        try Data().write(to: marker)
    }

    private static func hasSize(_ file: ModelFile, at url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) == file.size } ?? false
    }

    /// Hashes the file in 1 MB reads off the main actor.
    private static func sha256(of url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hash = SHA256()
            while true {
                try Task.checkCancellation()
                let chunk = try autoreleasepool { try handle.read(upToCount: 1 << 20) }
                guard let chunk, !chunk.isEmpty else { break }
                hash.update(data: chunk)
            }
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    enum DownloadError: LocalizedError {
        case notEnoughSpace, server
        case integrity(String)
        var errorDescription: String? {
            switch self {
            case .notEnoughSpace: "Not enough free space for the model."
            case .server: "The server did not return the file."
            case .integrity(let path): "\(path) did not match its checksum."
            }
        }
    }

    /// Reports a download's progress in steps of half a percent: the delegate is called for every
    /// few kilobytes written.
    private final class Progress: NSObject, URLSessionDownloadDelegate, Sendable {
        let report: @Sendable (Double) -> Void
        private let reported = Mutex(-1.0)
        init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
        ) {
            guard totalBytesExpectedToWrite > 0 else { return }
            let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            let due = reported.withLock { last in
                guard fraction - last >= 0.005 || fraction == 1 else { return false }
                last = fraction
                return true
            }
            if due { report(fraction) }
        }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL
        ) {}
    }
}
