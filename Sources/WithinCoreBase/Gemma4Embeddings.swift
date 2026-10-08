import Foundation

// Gemma 4's token embedding and per-layer embedding (PLE) tables stay outside the Core AI graphs:
// together they are 7 GB of FP16 (0.92 GB as Google's 2-bit QAT codes), and a decode step reads one
// row of each. The host maps the files and gathers rows into the chunks' `embedded` and
// `token_identity` inputs.

/// One embedding table in the package, as `gemma-runtime.json` describes it.
public struct Gemma4EmbeddingDescriptor: Codable, Sendable {
    /// The table's element type: FP16, or (schema 2, v2 lite tier) Google's 2-bit QAT codes, four per
    /// byte with the low bits first, each 0...3 standing for -2...1, with F32 scales.
    public enum Format: String, Codable, Sendable { case float16, int2 }
    /// The F32 scales of an `int2` table: one per `columnsPerScale` values of a row, little-endian,
    /// `[rows, columns / columnsPerScale]`.
    public struct Scales: Codable, Sendable, Equatable {
        public var file: String
        public var columnsPerScale: Int
        public init(file: String, columnsPerScale: Int) { self.file = file; self.columnsPerScale = columnsPerScale }
    }
    /// The table's file name in the package folder.
    public var file: String
    /// The element type.
    public var format: Format
    /// One row per vocabulary token.
    public var rows: Int
    /// Values per row: the hidden size, or every layer's PLE width for the per-layer table.
    public var columns: Int
    /// Present exactly for `int2`.
    public var scales: Scales?

    public init(file: String, format: Format, rows: Int, columns: Int, scales: Scales? = nil) {
        self.file = file; self.format = format; self.rows = rows; self.columns = columns; self.scales = scales
    }

    /// Bytes of the table file.
    var tableBytes: Int { format == .float16 ? rows * columns * 2 : rows * columns / 4 }

    /// Checks the shape bounds and that every file is a plain name inside the package folder.
    public func validate() throws {
        guard (1...262_144).contains(rows), (1...10_752).contains(columns), Self.safeName(file) else {
            throw Gemma4RuntimeError.invalidPackage
        }
        switch format {
        case .float16:
            guard scales == nil else { throw Gemma4RuntimeError.invalidPackage }
        case .int2:
            guard let scales, columns.isMultiple(of: 4), (1...columns).contains(scales.columnsPerScale),
                columns.isMultiple(of: scales.columnsPerScale), Self.safeName(scales.file), scales.file != file
            else { throw Gemma4RuntimeError.invalidPackage }
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
    private let scales: Data

    /// Maps `descriptor.file` (and an `int2` table's scales) in `folder` and checks the sizes against
    /// the descriptor.
    public init(folder: URL, descriptor: Gemma4EmbeddingDescriptor) throws {
        try descriptor.validate()
        self.descriptor = descriptor
        data = try Data(contentsOf: folder.appending(path: descriptor.file), options: .alwaysMapped)
        guard data.count == descriptor.tableBytes else { throw Gemma4RuntimeError.invalidPackage }
        if let file = descriptor.scales {
            scales = try Data(contentsOf: folder.appending(path: file.file), options: .alwaysMapped)
            guard scales.count == descriptor.rows * (descriptor.columns / file.columnsPerScale) * 4 else {
                throw Gemma4RuntimeError.invalidPackage
            }
        } else {
            scales = Data()
        }
    }

    /// The rows for `tokens`, each split into `layers` equal parts and multiplied by `scale` (in FP16
    /// for FP16 tables; in F32 with the code's own scale for `int2`, then rounded once to FP16, as
    /// transformers' QuantizedEmbedding does), as the Gemma embedding does: [row, layer, channel] on
    /// disk becomes [1, channel, layer, token] for Core AI.
    public func gather(tokens: [Int], layers: Int = 1, scale: Float) throws -> [Float16] {
        guard !tokens.isEmpty, tokens.count <= 256, layers > 0, layers <= 42,  // the widest prefill call
            descriptor.columns.isMultiple(of: layers), scale.isFinite,
            tokens.allSatisfy({ (0..<descriptor.rows).contains($0) })
        else { throw Gemma4RuntimeError.invalidToken }
        let channels = descriptor.columns / layers
        var result = [Float16](repeating: 0, count: tokens.count * descriptor.columns)
        if let file = descriptor.scales {
            try gatherCodes(tokens: tokens, layers: layers, scale: scale, scales: file, into: &result)
            return result
        }
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

    /// The `int2` rows for `tokens`, laid out as `gather` describes, without touching each value
    /// one at a time: every code of a token and scale group is one of four values
    /// (`Float16(Float(code) * factor * scale)`, the same expression as before), so each group gets
    /// a four-entry table and a value is one lookup; the tokens of one channel are written in one
    /// run, and the four codes of a byte are read with one load. A 128-token prefill call reads
    /// 1.7 million values this way (REPORT.md Section 7.2).
    private func gatherCodes(
        tokens: [Int], layers: Int, scale: Float, scales file: Gemma4EmbeddingDescriptor.Scales,
        into result: inout [Float16]
    ) throws {
        let columns = descriptor.columns, channels = columns / layers, count = tokens.count
        let groups = columns / file.columnsPerScale, rowBytes = columns / 4
        // tables[(token * groups + group) * 4 + code + 2]
        var tables = [Float16](repeating: 0, count: count * groups * 4)
        try scales.withUnsafeBytes { factors in
            for (position, token) in tokens.enumerated() {
                for group in 0..<groups {
                    let factor = Float(
                        bitPattern: UInt32(
                            littleEndian: factors.loadUnaligned(
                                fromByteOffset: (token * groups + group) * 4, as: UInt32.self)))
                    for code in -2...1 {
                        let scaled = Float16(Float(code) * factor * scale)
                        guard scaled.isFinite else { throw Gemma4RuntimeError.invalidPackage }
                        tables[(position * groups + group) * 4 + code + 2] = scaled
                    }
                }
            }
        }
        data.withUnsafeBytes { codes in
            let bytes = codes.bindMemory(to: UInt8.self).baseAddress!
            tables.withUnsafeBufferPointer { tables in
                result.withUnsafeMutableBufferPointer { out in
                    for layer in 0..<layers {
                        for channel in 0..<channels {
                            let column = layer * channels + channel
                            let byte = column / 4, shift = UInt8(2 * (column % 4)), group = column / file.columnsPerScale
                            let row = (channel * layers + layer) * count
                            for position in 0..<count {
                                let code = Int((bytes[tokens[position] * rowBytes + byte] >> shift) & 3)
                                out[row + position] = tables[(position * groups + group) * 4 + code]
                            }
                        }
                    }
                }
            }
        }
    }
}
