import CoreAILanguageModels
import Foundation
import Tokenizers
import WithinCoreBase

/// A Gemma 4 package loaded for chat: the tokenizer and chat template, the chunked decoder, and the
/// generation loop (context budget, prefix reuse, sampling, the thought channel, thermal pacing).
/// Calls are serialized by the actor; the caller decides when to load, cancel and release it.
public actor Gemma4Engine {
    /// Sampling and length settings for one generation.
    public struct Options: Sendable {
        /// Softmax temperature; 0 samples greedily.
        public var temperature: Double
        /// Sample among this many most likely tokens.
        public var topK: Int
        /// Then among the smallest set reaching this probability.
        public var topP: Double
        /// The reply budget; the context budget may lower it to what fits.
        public var maxOutputTokens: Int
        /// Let the model write a thought channel before the answer.
        public var thinking: Bool
        /// Measurement only: generate exactly this many tokens, with stop tokens and thought-channel
        /// markers suppressed. Chats leave it nil.
        public var exactOutputTokens: Int?

        /// Options for one generation.
        public init(
            temperature: Double, topK: Int, topP: Double = 0.95, maxOutputTokens: Int, thinking: Bool,
            exactOutputTokens: Int? = nil
        ) {
            self.temperature = temperature; self.topK = topK; self.topP = topP
            self.maxOutputTokens = maxOutputTokens; self.thinking = thinking
            self.exactOutputTokens = exactOutputTokens
        }
    }

    /// What a generation reports, in order.
    public enum Event: Sendable {
        /// The reply so far (answer tokens only), decoded from the start.
        case text(String)
        case thinking(Bool)
        /// Earlier turns were left out to fit the context.
        case contextTrimmed
        case finished(lengthLimited: Bool)
    }

    /// Generation failures besides cancellation and decoder errors.
    public enum Failure: Error, Equatable {
        case incompatiblePackage
        /// The model ended without answer text.
        case emptyReply
        /// Thinking used the whole output budget before an answer.
        case thinkingLimit
    }

    /// Tokens of prompt, history and reply the package holds.
    public nonisolated let contextLimit: Int
    /// Tokens in the vocabulary.
    public nonisolated let vocabularySize: Int
    private let runtime: Gemma4ChunkedRuntime
    private let tokenizer: any Tokenizer
    /// The package's chat template with comments removed (ChatTemplateComments): renders exactly as
    /// Transformers does, which the pinned swift-jinja does not.
    private let chatTemplate: String
    private let stopTokens: Set<Int32>
    private let events: EngineEventHandler?
    /// Exactly the tokens the runtime's KV cache holds, in order; nil when unknown (after a failure or
    /// cancellation), which forces a reset.
    private var fed: [Int]?
    /// The last completed generation's timing marks.
    public private(set) var lastTimings: GenerationTimings?

    /// Loads the package in `folder`: `gemma-runtime.json`, `chunk-contract.json`, the chunks, the
    /// embedding tables, `tokenizer/` and `generation_config.json`. Reads installed files only.
    public init(folder: URL, requireE4B: Bool = true, events: EngineEventHandler? = nil) async throws {
        self.events = events
        let configuration = try JSONDecoder().decode(
            Gemma4RuntimeConfiguration.self,
            from: Data(contentsOf: folder.appending(path: "gemma-runtime.json")))
        try configuration.validate(requireE4B: requireE4B)
        events?(
            "engine.configuration.valid",
            [
                "layers": String(configuration.layerTypes.count),
                "context": String(configuration.contextLimit), "development": String(!requireE4B),
            ])
        // The convenience loader can contact Hugging Face. This path only reads installed files.
        events?("tokenizer.load.begin", [:])
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder.appending(path: "tokenizer"))
        let template = ChatTemplateComments.strip(
            try String(contentsOf: folder.appending(path: "tokenizer/chat_template.jinja"), encoding: .utf8))
        events?("tokenizer.load.complete", [:])
        try Task.checkCancellation()
        let contract = try JSONDecoder().decode(
            Gemma4ChunkContract.self,
            from: Data(contentsOf: folder.appending(path: Gemma4ChunkContract.filename)))
        runtime = try await Gemma4ChunkedRuntime(
            folder: folder, configuration: configuration, contract: contract, requireE4B: requireE4B,
            events: events)
        events?("engine.ready", [:])
        try Task.checkCancellation()
        var stopTokens = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stopTokens.insert(Int32(eos)) }
        // Packaging supplies all terminators from the checkpoint's generation config.
        let generationConfiguration = try JSONDecoder().decode(
            GenerationConfiguration.self,
            from: Data(contentsOf: folder.appending(path: "generation_config.json")))
        guard
            generationConfiguration.eosTokenIDs.allSatisfy({
                $0 >= 0 && $0 < configuration.vocabularySize && Int32(exactly: $0) != nil
            })
        else { throw Failure.incompatiblePackage }
        stopTokens.formUnion(generationConfiguration.eosTokenIDs.map(Int32.init))
        guard !stopTokens.isEmpty else { throw Failure.incompatiblePackage }
        self.tokenizer = tokenizer
        self.chatTemplate = template
        self.stopTokens = stopTokens
        contextLimit = configuration.contextLimit
        vocabularySize = configuration.vocabularySize
    }

    /// Generates a reply to the last turn, which must be the user's, and reports it through `onEvent`.
    /// A prompt that extends what the KV cache holds continues from it; anything else starts clean.
    /// `requestStart` anchors the recorded timings (by default, this call).
    public func generate(
        system: String, turns: [ChatTurn], options: Options,
        requestStart: ContinuousClock.Instant = .now,
        onEvent: @Sendable (Event) -> Void
    ) async throws {
        lastTimings = nil
        do {
            try await run(
                system: system, turns: turns, options: options, began: requestStart, onEvent: onEvent)
        } catch {
            fed = nil  // the failed or cancelled call may have written part of the cache
            throw error
        }
    }

    private func run(
        system: String, turns: [ChatTurn], options: Options, began: ContinuousClock.Instant,
        onEvent: @Sendable (Event) -> Void
    ) async throws {
        let template = ChatTemplateArgument.literal(chatTemplate)
        let render: ([ChatTurn]) throws -> [Int] = { turns in
            var messages: [Tokenizers.Message] = []
            if !system.isEmpty { messages.append(["role": "system", "content": system]) }
            messages += turns.map { ["role": $0.role.rawValue, "content": $0.text] }
            return try self.tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: template,
                addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
                additionalContext: ["enable_thinking": options.thinking])
        }
        let budget = try ContextBudget.fit(
            messages: turns, contextLimit: contextLimit,
            requestedOutput: options.exactOutputTokens ?? options.maxOutputTokens
        ) { try render($0).count }
        if budget.trimmed { onEvent(.contextTrimmed) }
        let tokens = try render(budget.messages)
        // A prompt that extends exactly what the cache holds (the previous turn's
        // prompt and reply, then the new user turn) continues from it. Anything
        // else (another chat, regeneration, trimming, a stripped thought channel)
        // starts from a clean cache, which keeps switching unambiguous.
        let reused = fed.map { tokens.count > $0.count && tokens.starts(with: $0) } ?? false
        let start = reused ? fed?.count ?? 0 : 0
        events?(
            "prompt.prepared",
            [
                "inputTokens": String(tokens.count),
                "reusedTokens": String(start), "outputBudget": String(budget.outputTokens),
            ])
        if !reused {
            fed = nil
            try await runtime.reset()
            fed = []
        }
        let sampling = SamplingConfiguration(
            temperature: options.temperature, topK: options.topK, topP: options.topP)
        guard let channelStart = tokenizer.convertTokenToId("<|channel>"),
            let channelEnd = tokenizer.convertTokenToId("<channel|>"),
            let thinkMarker = tokenizer.convertTokenToId("<|think|>")
        else { throw Failure.incompatiblePackage }
        var channel = GemmaOutputFilter(channelStart: channelStart, channelEnd: channelEnd, thinkMarker: thinkMarker)
        var generated: [Int] = []
        let clock = ContinuousClock()
        var stamp = clock.now
        var logits = try await runtime.logits(for: Array(tokens[start...]))
        fed?.append(contentsOf: tokens[start...])
        var logitsTime = clock.now - stamp
        var timings = GenerationTimings(
            inputTokenIDs: tokens, reusedTokens: start,
            prefillMs: (clock.now - began).milliseconds, tokenMs: [])
        events?(
            "generation.prefill.timing",
            [
                "inputTokens": String(tokens.count),
                "ms": milliseconds(logitsTime),
            ])
        var lengthLimited = true
        for index in 0..<budget.outputTokens {
            try Task.checkCancellation()
            let tokenStarted = clock.now
            if options.exactOutputTokens != nil {
                // Forced past its natural end the model otherwise opens a thought
                // channel, whose tokens the filter hides (an empty reply).
                for stop in stopTokens { logits[Int(stop)] = -Float16.greatestFiniteMagnitude }
                for marker in [channelStart, thinkMarker] { logits[marker] = -Float16.greatestFiniteMagnitude }
            }
            let token = sampling.fallbackSampler(from: &logits)
            let sampled = clock.now
            timings.tokenMs.append((sampled - began).milliseconds)
            var decodeTime = Duration.zero
            if stopTokens.contains(token) { lengthLimited = false; break }
            let wasThinking = channel.isThinking
            let accepted = channel.accepts(Int(token))
            if wasThinking != channel.isThinking { onEvent(.thinking(channel.isThinking)) }
            if accepted {
                generated.append(Int(token))
                let decoding = clock.now
                let decoded = tokenizer.decode(tokens: generated)
                decodeTime = clock.now - decoding
                // Byte-level tokenizers can temporarily end on an incomplete UTF-8 scalar.
                if !decoded.hasSuffix("\u{FFFD}") {
                    onEvent(.text(decoded))
                    events?("generation.text", ["generatedTokens": String(generated.count)])
                }
            }
            // Thought tokens also advance the model cache and count toward the budget.
            events?(
                "generation.token.timing",
                [
                    "index": String(index),
                    "previousLogitsMs": milliseconds(logitsTime), "sampleMs": milliseconds(sampled - tokenStarted),
                    "decodeMs": milliseconds(decodeTime),
                    "sinceTokenStartMs": milliseconds(clock.now - tokenStarted),
                ])
            if index + 1 < budget.outputTokens {
                // Pacing starts at .fair: once .serious is reached, even slow generation stays
                // there for minutes. A call after a pause also runs a few ms slower, so the pace
                // per token is set above the package's unpaced step time.
                let pace: Duration? =
                    switch ProcessInfo.processInfo.thermalState {
                    case .fair: .milliseconds(110)
                    case .serious, .critical: .milliseconds(200)
                    default: nil
                    }
                if let pace { try await Task.sleep(until: stamp + pace, clock: clock) }
                stamp = clock.now
                logits = try await runtime.logits(for: [Int(token)])
                fed?.append(Int(token))
                logitsTime = clock.now - stamp
            }
        }
        try Task.checkCancellation()
        let answer = tokenizer.decode(tokens: generated)
        guard !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if lengthLimited && options.thinking { throw Failure.thinkingLimit }
            throw Failure.emptyReply
        }
        onEvent(.thinking(false))
        onEvent(.text(answer))
        lastTimings = timings
        onEvent(.finished(lengthLimited: lengthLimited))
        events?("generation.complete", ["generatedTokens": String(generated.count)])
    }

    /// Teacher forcing: the summed negative log-likelihood of each `continuation` token given
    /// everything before it. Nothing is sampled; the cache is left holding these tokens.
    public func score(prompt: [Int], continuation: [Int]) async throws -> TeacherForcingScore {
        guard !prompt.isEmpty, !continuation.isEmpty, prompt.count + continuation.count - 1 <= contextLimit
        else { throw Failure.incompatiblePackage }
        fed = nil
        try await runtime.reset()
        var logits = try await runtime.logits(for: prompt)
        var score = TeacherForcingScore()
        for (index, token) in continuation.enumerated() {
            try Task.checkCancellation()
            try score.add(logits: logits, token: token)
            if index + 1 < continuation.count { logits = try await runtime.logits(for: [token]) }
        }
        return score
    }

    /// Writes the KV cache to `url` and returns the tokens it holds and its size; nil when it holds
    /// nothing. Reading it back is far faster than prefilling the conversation again.
    public func saveKV(to url: URL) async throws -> (tokens: [Int], bytes: Int)? {
        guard let fed, !fed.isEmpty else { return nil }
        guard let saved = try await runtime.saveKV(to: url), saved.position == fed.count else {
            throw CocoaError(.coderInvalidValue)
        }
        return (fed, saved.bytes)
    }

    /// Reads back what `saveKV` wrote; `tokens` are the tokens it returned. After a failure the
    /// cache counts as empty.
    public func restoreKV(from url: URL, tokens: [Int]) async throws {
        fed = nil
        try await runtime.restoreKV(from: url, position: tokens.count)
        fed = tokens
    }

    /// Stops the decoder call in flight; the caller cancels its own task too.
    public func cancel() async {
        await runtime.cancel()
    }
}

private struct GenerationConfiguration: Decodable {
    let eosTokenIDs: [Int]
    enum CodingKeys: String, CodingKey { case eosTokenIDs = "eos_token_id" }
    init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let values = try? container.decode([Int].self, forKey: .eosTokenIDs) {
            eosTokenIDs = values
        } else {
            eosTokenIDs = [try container.decode(Int.self, forKey: .eosTokenIDs)]
        }
    }
}
