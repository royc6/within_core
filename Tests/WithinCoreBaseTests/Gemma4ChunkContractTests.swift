import Foundation
import Testing

@testable import WithinCoreBase

private func chunkConfiguration() -> Gemma4RuntimeConfiguration {
    Gemma4RuntimeConfiguration(
        schemaVersion: 1, hiddenSize: 32, vocabularySize: 64,
        layerTypes: (0..<42).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" },
        sharedLayerCount: 18, perLayerInputSize: 8, keyValueHeads: 2,
        localHeadDimension: 8, globalHeadDimension: 16, localTheta: 10_000, globalTheta: 1_000_000,
        globalRotaryFraction: 0.25, slidingWindow: 5, contextLimit: 16, prefillLength: 4,
        embeddings: .init(file: "embeddings.f16", format: .float16, rows: 64, columns: 32),
        perLayerEmbeddings: .init(file: "ple.f16", format: .float16, rows: 64, columns: 336))
}

private func decode(_ payload: [String: Any]) throws -> Gemma4ChunkContract {
    try JSONDecoder().decode(Gemma4ChunkContract.self, from: JSONSerialization.data(withJSONObject: payload))
}

@Test func pleLayerSliceKeepsChannelLayerPositionOrder() {
    // channels 2, layers 3, positions 2; value = channel*100 + layer*10 + position
    var values: [Float16] = []
    for channel in 0..<2 {
        for layer in 0..<3 { for position in 0..<2 { values.append(Float16(channel * 100 + layer * 10 + position)) } }
    }
    let slice = Gemma4Embeddings.sliceLayers(values, layers: 3, length: 2, range: 1..<3)
    #expect(slice == [10, 11, 20, 21, 110, 111, 120, 121].map { Float16($0) })
}

/// Seven six-layer chunks, written out independently of the Python exporter: owners read
/// `key_<n>`/`value_<n>` and return `fresh_*`; consumer chunks read donors 22/23's cache plus fresh K/V.
private func hostPayload() -> [String: Any] {
    let common = ["token_identity", "local_cos", "local_sin", "global_cos", "global_sin", "local_mask", "global_mask"]
    let names = { (layer: Int, prefix: String) in ["\(prefix)key_\(layer)", "\(prefix)value_\(layer)"] }
    var chunks: [[String: Any]] = []
    for index in 0..<7 {
        let start = index * 6
        let owned = index < 4 ? Array(start..<start + 6) : []
        let donors = index < 4 ? [Int]() : [22, 23]
        let inputs =
            (index == 0 ? ["embedded"] : ["hidden", "embedded"]) + common
            + owned.flatMap { names($0, "") } + donors.flatMap { names($0, "") + names($0, "fresh_") }
        let fresh = owned.flatMap { names($0, "fresh_") }
        var functions: [String: Any] = [
            "step_1": [
                "length": 1, "inputs": inputs,
                "outputs": (index == 6 ? ["logits"] : ["hidden"]) + fresh,
            ]
        ]
        if !owned.isEmpty {
            functions["prefill_4"] = ["length": 4, "inputs": inputs, "outputs": (index < 3 ? ["hidden"] : []) + fresh]
        }
        chunks.append([
            "file": String(format: "chunk-%02d.aimodel", index), "start": start, "end": start + 6,
            "functions": functions, "owned": owned, "externalDonors": donors,
        ])
    }
    return [
        "contract": "gemma4-chunked-hostkv-v1", "contextLimit": 16, "prefillLength": 4, "firstSharedLayer": 24,
        "weightFormat": "int8-per-channel-symmetric-weight-only", "chunks": chunks,
    ]
}

private func hostMutated(_ index: Int, _ change: (inout [String: Any]) -> Void) -> [String: Any] {
    var payload = hostPayload()
    var chunks = payload["chunks"] as! [[String: Any]]
    change(&chunks[index])
    payload["chunks"] = chunks
    return payload
}

@Test func hostKVContractAcceptsSevenSixLayerChunks() throws {
    let contract = try decode(hostPayload())
    try contract.validate(configuration: chunkConfiguration(), requireE4B: false)
    #expect(contract.chunks.map(\.runsPrefill) == [true, true, true, true, false, false, false])
    #expect(contract.chunks[4].externalDonors == [22, 23])
}

@Test func hostKVContractRejectsBrokenTopology() throws {
    let c = chunkConfiguration()
    let broken: [[String: Any]] = [
        // A consumer chunk missing a donor's fresh K/V input.
        hostMutated(5) { chunk in
            var functions = chunk["functions"] as! [String: Any]
            var step = functions["step_1"] as! [String: Any]
            step["inputs"] = (step["inputs"] as! [String]).filter { $0 != "fresh_key_23" }
            functions["step_1"] = step; chunk["functions"] = functions
        },
        // An owner that stops returning one layer's fresh value.
        hostMutated(1) { chunk in
            var functions = chunk["functions"] as! [String: Any]
            var step = functions["step_1"] as! [String: Any]
            step["outputs"] = (step["outputs"] as! [String]).filter { $0 != "fresh_value_8" }
            functions["step_1"] = step; chunk["functions"] = functions
        },
        // Consumers never prefill.
        hostMutated(6) { chunk in
            var functions = chunk["functions"] as! [String: Any]
            functions["prefill_4"] = functions["step_1"]; chunk["functions"] = functions
        },
        // Ownership that disagrees with the layer range.
        hostMutated(2) { $0["owned"] = Array(12..<17) },
        // A layer gap between chunks.
        hostMutated(2) {
            $0["start"] = 13; $0["owned"] = Array(13..<18)
        },
    ]
    for payload in broken {
        #expect(throws: Gemma4RuntimeError.self) { try decode(payload).validate(configuration: c, requireE4B: false) }
    }
    // Only the host-held KV contract loads.
    var stateKV = hostPayload()
    stateKV["contract"] = "gemma4-chunked-kv-v1"
    #expect(throws: Gemma4RuntimeError.self) { try decode(stateKV).validate(configuration: c, requireE4B: false) }
}

/// Seven layer chunks whose last leaves the head to an eighth, head-only chunk.
private func headPayload(parts: Int = 4) -> [String: Any] {
    var payload = hostPayload()
    var chunks = payload["chunks"] as! [[String: Any]]
    var last = chunks[6]
    var functions = last["functions"] as! [String: Any]
    var step = functions["step_1"] as! [String: Any]
    step["outputs"] = ["hidden"]; functions["step_1"] = step; last["functions"] = functions
    chunks[6] = last
    chunks.append([
        "file": "chunk-07.aimodel", "start": 42, "end": 42, "owned": [Int](), "externalDonors": [Int](),
        "head": ["parts": parts, "rows": 64 / parts],
        "functions": [
            "step_1": [
                "length": 1, "inputs": ["hidden"],
                "outputs": (0..<parts).map { String(format: "logits_%02d", $0) },
            ]
        ],
    ])
    payload["chunks"] = chunks
    return payload
}

@Test func hostKVContractAcceptsTrailingHeadChunk() throws {
    let contract = try decode(headPayload())
    try contract.validate(configuration: chunkConfiguration(), requireE4B: false)
    #expect(contract.chunks.last?.head == .init(parts: 4, rows: 16))
    #expect(contract.chunks.map(\.runsPrefill) == [true, true, true, true, false, false, false, false])
}

@Test func hostKVContractRejectsBrokenHeadChunk() throws {
    let c = chunkConfiguration()
    var rows = headPayload()
    var chunks = rows["chunks"] as! [[String: Any]]
    chunks[7]["head"] = ["parts": 4, "rows": 15]  // does not cover the vocabulary
    rows["chunks"] = chunks
    var order = headPayload()
    chunks = order["chunks"] as! [[String: Any]]
    var functions = chunks[7]["functions"] as! [String: Any]
    functions["step_1"] = [
        "length": 1, "inputs": ["hidden"], "outputs": ["logits_01", "logits_00", "logits_02", "logits_03"],
    ]
    chunks[7]["functions"] = functions
    order["chunks"] = chunks
    var logits = headPayload()  // the last layer chunk still returning logits alongside a head chunk
    chunks = logits["chunks"] as! [[String: Any]]
    functions = chunks[6]["functions"] as! [String: Any]
    functions["step_1"] = [
        "length": 1, "inputs": ((functions["step_1"] as! [String: Any])["inputs"] as! [String]),
        "outputs": ["logits"],
    ]
    chunks[6]["functions"] = functions
    logits["chunks"] = chunks
    for payload in [rows, order, logits] {
        #expect(throws: Gemma4RuntimeError.self) { try decode(payload).validate(configuration: c, requireE4B: false) }
    }
}

@Test func ringMaskMatchesSimulatedRingSlots() throws {
    let c = chunkConfiguration()  // window 5, context 16
    let window = c.slidingWindow
    var held = [Int?](repeating: nil, count: window)  // position each ring slot holds
    var step = 0
    for length in [4, 4, 1, 1, 3, 1, 1] {
        let mask = try c.ringMask(step: step, length: length)
        let width = window + length
        for query in 0..<length {
            let position = step + query
            for slot in 0..<window {
                let visible = held[slot].map { position - $0 < window } ?? false
                #expect((mask[query * width + slot] == 0) == visible, "step \(step) query \(query) slot \(slot)")
            }
            for fresh in 0..<length {
                let visible = fresh <= query && query - fresh < window
                #expect((mask[query * width + window + fresh] == 0) == visible)
            }
        }
        for token in 0..<length { held[(step + token) % window] = step + token }
        step += length
    }
}

@Test func hostKVContractValidatesRingLength() throws {
    let c = chunkConfiguration()
    var payload = headPayload()
    payload["slidingCacheLength"] = 5
    let contract = try decode(payload)
    try contract.validate(configuration: c, requireE4B: false)
    #expect(contract.cacheLength(layer: 0, configuration: c) == 5)
    #expect(contract.cacheLength(layer: 5, configuration: c) == 16)
    payload["slidingCacheLength"] = 4  // not the window
    #expect(throws: Gemma4RuntimeError.self) { try decode(payload).validate(configuration: c, requireE4B: false) }
}
