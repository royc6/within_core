import Foundation

// Gemma 4's token embedding and per-layer embedding (PLE) tables stay outside the Core AI graphs:
// together they are 7 GB of FP16, and a decode step reads one row of each. The host maps the files
// and gathers rows into the chunks' `embedded` and `token_identity` inputs.

/// One FP16 embedding table in the package, as `gemma-runtime.json` describes it.
public struct Gemma4EmbeddingDescriptor: Codable, Sendable {
    /// The table's element type; packages store FP16.
    public enum Format: String, Codable, Sendable { case float16 }
    /// The table's file name in the package folder.
    public var file: String
    /// The element type.
    public var format: Format
    /// One row per vocabulary token.
    public var rows: Int
    /// Values per row: the hidden size, or every layer's PLE width for the per-layer table.
    public var columns: Int

    /// Checks the shape bounds and that `file` is a plain name inside the package folder.
    public func validate() throws {
        guard (1...262_144).contains(rows), (1...10_752).contains(columns), Self.safeName(file) else {
            throw Gemma4RuntimeError.invalidPackage
        }
    }
    private static func safeName(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && value.count < 120
            && value.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-" || $0 == "_") }
    }
}

/// A memory-mapped embedding table: only the rows a call needs are read, and no copy of the table
/// is made. Token embeddings and PLE both use it.
public struct Gemma4Embeddings: Sendable {
    private let descriptor: Gemma4EmbeddingDescriptor
    private let data: Data

    /// Maps `descriptor.file` in `folder` and checks its size against the descriptor.
    public init(folder: URL, descriptor: Gemma4EmbeddingDescriptor) throws {
        try descriptor.validate()
        self.descriptor = descriptor
        data = try Data(contentsOf: folder.appending(path: descriptor.file), options: .alwaysMapped)
        guard data.count == descriptor.rows * descriptor.columns * 2 else { throw Gemma4RuntimeError.invalidPackage }
    }

    /// The rows for `tokens`, each split into `layers` equal parts and multiplied by `scale` in FP16,
    /// as the Gemma embedding does: [row, layer, channel] on disk becomes [1, channel, layer, token]
    /// for Core AI.
    public func gather(tokens: [Int], layers: Int = 1, scale: Float) throws -> [Float16] {
        guard !tokens.isEmpty, tokens.count <= 128, layers > 0, layers <= 42,
            descriptor.columns.isMultiple(of: layers), scale.isFinite,
            tokens.allSatisfy({ (0..<descriptor.rows).contains($0) })
        else { throw Gemma4RuntimeError.invalidToken }
        let channels = descriptor.columns / layers
        var result = [Float16](repeating: 0, count: tokens.count * descriptor.columns)
        try data.withUnsafeBytes { bytes in
            for (position, token) in tokens.enumerated() {
                for layer in 0..<layers {
                    for channel in 0..<channels {
                        let index = token * descriptor.columns + layer * channels + channel
                        let value = Float16(
                            bitPattern: UInt16(
                                littleEndian: bytes.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)))
                        let scaled = value * Float16(scale)
                        guard scaled.isFinite else { throw Gemma4RuntimeError.invalidPackage }
                        result[(channel * layers + layer) * tokens.count + position] = scaled
                    }
                }
            }
        }
        return result
    }
}
