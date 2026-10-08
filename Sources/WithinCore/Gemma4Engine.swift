import CoreAILanguageModels
import CryptoKit
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
        /// Thin the conversation when the reply ends with this many free slots or fewer
        /// (REPORT.md Section 7.7); nil never thins.
        public var thinAtFreeSlots: Int?
        /// Search the turns the cache no longer holds for the latest message's words and show the
        /// model what matches, for this reply only (REPORT.md Section 7.7).
        public var recall: Bool
        /// A variant of recall under test (REPORT.md Section 7.7): a reply that is gone is also
        /// found by the message it answered, and that message is quoted with it.
        public var recallExchanges: Bool
        /// Thinning keeps a user turn of up to this many tokens and takes a longer one for a paste.
        /// A conversation must be rendered with the same value from one prompt to the next.
        public var longestKeptTurn: Int

        /// Options for one generation.
        public init(
            temperature: Double, topK: Int, topP: Double = 0.95, maxOutputTokens: Int, thinking: Bool,
            exactOutputTokens: Int? = nil, thinAtFreeSlots: Int? = nil, recall: Bool = false,
            recallExchanges: Bool = false, longestKeptTurn: Int = 256
        ) {
            self.temperature = temperature; self.topK = topK; self.topP = topP
            self.maxOutputTokens = maxOutputTokens; self.thinking = thinking
            self.exactOutputTokens = exactOutputTokens; self.thinAtFreeSlots = thinAtFreeSlots
            self.recall = recall; self.recallExchanges = recallExchanges; self.longestKeptTurn = longestKeptTurn
        }
    }

    /// What a generation reports, in order.
    public enum Event: Sendable {
        /// The reply so far (answer tokens only), decoded from the start.
        case text(String)
        case thinking(Bool)
        /// Earlier turns were left out to fit the context.
        case contextTrimmed
        /// The conversation was thinned as this reply ended: later prompts mark its turn `thinned`.
        case thinned
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
    /// PREFIX-KV (REPORT.md Section 7.5): where the K/V of a prompt's head (`<bos>` and the system
    /// turn) is kept, written when a prompt with instructions starts clean and read back by the
    /// next one that starts with the same head; nil turns it off.
    private let prefixStore: URL?
    /// What the runtime's KV cache holds: the head, then one live window of the conversation's
    /// rendering (KV-SLIDE); nil when unknown (after a failure, or a cancellation the cache was not
    /// kept through), which forces a reset.
    private var slide: KVSlideState?
    /// The last completed generation's timing marks.
    public private(set) var lastTimings: GenerationTimings?

    /// Loads the package in `folder`: `gemma-runtime.json`, `chunk-contract.json`, the chunks, the
    /// embedding tables, `tokenizer/` and `generation_config.json`. Reads installed files only.
    /// `prefixStore` is a folder for this package alone (see `prefixStore`).
    public init(
        folder: URL, requireE4B: Bool = true, prefixStore: URL? = nil, events: EngineEventHandler? = nil
    ) async throws {
        self.events = events
        self.prefixStore = prefixStore
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
    /// A prompt that extends what the KV cache holds continues from it, evicting the oldest turns'
    /// slots when the package slides and they are needed; anything else starts clean.
    /// A reply cancelled while it is decoded keeps the cache up to its last whole token, for a
    /// prompt that renders what was reported of it; any other failure leaves the cache unknown.
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
        } catch is CacheKept {
            throw CancellationError()
        } catch {
            slide = nil  // the failed or cancelled call may have written part of the cache
            throw error
        }
    }

    /// A reply cancelled while it was decoded, its cache put in order for the next prompt (`run`):
    /// the caller sees a cancellation.
    private struct CacheKept: Error {}

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
        let requested = options.exactOutputTokens ?? options.maxOutputTokens
        guard requested > 0 else { throw ContextBudgetError.invalidConfiguration }
        guard let turnStart = tokenizer.convertTokenToId("<|turn>") else { throw Failure.incompatiblePackage }
        let complete = try ContextBudget.complete(turns)
        let full = try render(complete)
        let latest = try render([complete[complete.count - 1]]).count
        // A thinned conversation (REPORT.md Section 7.7) is rendered without the old replies and
        // long pastes: `shown` lists the tokens of `full` the cache follows, `view` is those tokens.
        // Turns are told apart by their opening token, one per turn and one for the generation prompt.
        let probe = try render([ChatTurn(role: .user, text: "a")])
        let head = runtime.slides ? KVSlidePlan.head(of: full, probe: probe, turnStart: turnStart) : nil
        let opens = head.map { head in full.indices.filter { $0 >= head && full[$0] == turnStart } } ?? []
        let thinnable = opens.count == complete.count + 1
        func thinned(at anchor: Int) -> [Int]? {
            guard thinnable, let head else { return nil }
            return KVSlidePlan.thinned(
                full: full, head: head, turnStart: turnStart, user: probe[head + 1], anchor: anchor,
                recent: contextLimit / Self.recentShare, longest: options.longestKeptTurn)
        }
        let shown = complete.dropLast().lastIndex(where: \.thinned).flatMap { thinned(at: opens[$0]) }
        let view = shown.map { $0.map { full[$0] } } ?? full
        // A prompt that continues what the cache holds (the previous turn's prompt and reply, then
        // the new user turn) goes in from there; once the slots run out the oldest tokens after the
        // head are evicted, a block at a time, and everything else keeps its K/V (KV-SLIDE,
        // REPORT.md Section 7.3).
        // Anything else (another chat, regeneration, a package that cannot slide and is full or
        // still holds a thought channel) starts from a clean cache with v1's trimming.
        var state = slide ?? KVSlideState()
        var tokens: [Int]
        let start: Int, outputTokens: Int
        var evicted = 0, trimmed = false
        var prefix: URL?  // where to keep the head's K/V once it is fed
        let budgeted: Int  // a clean start's reply budget
        var freed: [Range<Int>] = []  // cached tokens the prompt gives up; the runtime frees them as it needs room
        // The reply may take every slot but the head's and the latest turn's.
        let reply = min(requested, contextLimit - latest)
        if slide != nil, reply > 0,
            let step = KVSlidePlan.extend(
                state, full: view, capacity: contextLimit, reply: reply, slides: runtime.slides),
            state.position + step.feed.count + reply <= Gemma4RuntimeConfiguration.maximumPosition
        {
            freed = state.evict(step.evict)
            evicted = step.evict
            start = state.tokens.count
            tokens = state.tokens + step.feed
            outputTokens = reply
            trimmed = state.dropped > 0 || complete.count != turns.count || shown != nil
        } else {
            slide = nil
            try await runtime.reset()
            // A conversation that was compacted starts from the tokens the cache held then: the head,
            // the recent stretch, the summary turn, and what followed (summary turns, REPORT.md Section 7.7).
            if runtime.slides, reply > 0,
                let request = complete.dropLast().lastIndex(where: \.isSummaryRequest), let head,
                let rebuilt = try KVSlidePlan.rebuild(
                    full: full, head: head, fromSummary: render(Array(complete[request...])),
                    afterSummary: render(Array(complete[(request + 2)...])), latest: latest - head,
                    recent: contextLimit / Self.recentShare, capacity: contextLimit)
            {
                state = rebuilt
                tokens = rebuilt.tokens
                trimmed = rebuilt.dropped > 0 || complete.count != turns.count
                budgeted = reply
            } else if shown != nil, let head, reply > 0 {
                // A thinned conversation starts from the thinned rendering: the same tokens the cache
                // held, fed again. What does not fit goes from its oldest end, as when it slides.
                let over = max(0, view.count - contextLimit)
                tokens = Array(view[..<head]) + view[(head + over)...]
                state = KVSlideState(head: head, dropped: over)
                trimmed = true
                budgeted = reply
            } else {
                let budget = try ContextBudget.fit(
                    messages: turns, contextLimit: contextLimit, requestedOutput: requested
                ) { try render($0).count }
                trimmed = budget.trimmed
                tokens = try render(budget.messages)
                // The head stays while the rest slides; the turns left out here count as dropped.
                state = KVSlidePlan.start(tokens, full: full, probe: probe, turnStart: turnStart)
                budgeted = budget.outputTokens
            }
            // The head's K/V comes back from disk when a clean start with this head saved it.
            var restored = 0
            if let prefixStore, !system.isEmpty, state.head > 0 {
                let stamp = ContinuousClock.now
                let file = prefixStore.appending(path: Self.prefixName(tokens[..<state.head]))
                if (try? await runtime.restorePrefix(from: file, count: state.head)) == true {
                    restored = state.head
                    events?(
                        "prefix.restored",
                        ["tokens": String(restored), "ms": milliseconds(ContinuousClock.now - stamp)])
                } else {
                    prefix = file
                }
            }
            start = restored
            outputTokens = budgeted
        }
        // Recall (REPORT.md Section 7.7): the turns this cache no longer holds (thinned away or
        // slid out) are searched for the latest message's words. What matches goes in right before
        // the latest turn, as a turn of its own, and leaves the cache again when the reply ends.
        var recalled = 0
        let summarizing = complete[complete.count - 1].isSummaryRequest
        if options.recall, !summarizing, options.exactOutputTokens == nil, thinnable, let head, state.head == head,
            state.pinned.isEmpty, state.skipped == 0, shown != nil || state.dropped > 0
        {
            let searching = ContinuousClock.now
            let now = shown ?? Array(full.indices)
            let held = Set(now[min(now.count, head + state.dropped)...])
            let gone = complete.indices.dropLast().filter { !held.contains(opens[$0]) }
            let asked = complete[complete.count - 1].text
            let pair =
                options.recallExchanges
                ? Recall.turns(
                    for: asked,
                    amongExchanges: gone.filter {
                        $0 > 0 && complete[$0].role == .assistant && complete[$0 - 1].role == .user
                    }.map { (complete[$0 - 1], complete[$0]) },
                    // Set on the recall test conversations without the model: three short exchanges, found
                    // by a name alone, with the words a twenty-fifth of the pieces share left out.
                    count: 3, least: 1, common: 0.04)
                : Recall.turns(for: asked, among: gone.map { complete[$0] })
            if let pair {
                let rendered = try render(pair + [complete[complete.count - 1]])
                let extra = rendered[head..<(rendered.count - (latest - head))]
                if tokens.count + extra.count + Self.recallHeadroom <= contextLimit {
                    tokens.insert(contentsOf: extra, at: tokens.count - (latest - head))
                    recalled = extra.count
                }
            }
            events?(
                "recall",
                [
                    "searchedTurns": String(gone.count), "tokens": String(recalled),
                    "ms": milliseconds(ContinuousClock.now - searching),
                ])
        }
        if trimmed { onEvent(.contextTrimmed) }
        events?(
            "prompt.prepared",
            [
                "inputTokens": String(tokens.count),
                "reusedTokens": String(start), "outputBudget": String(outputTokens),
                "evictedTokens": String(evicted), "droppedTokens": String(state.dropped),
            ])
        let sampling = SamplingConfiguration(
            temperature: options.temperature, topK: options.topK, topP: options.topP)
        guard let channelStart = tokenizer.convertTokenToId("<|channel>"),
            let channelEnd = tokenizer.convertTokenToId("<channel|>"),
            let thinkMarker = tokenizer.convertTokenToId("<|think|>")
        else { throw Failure.incompatiblePackage }
        var channel = GemmaOutputFilter(channelStart: channelStart, channelEnd: channelEnd, thinkMarker: thinkMarker)
        var generated: [Int] = []
        var fed = 0, thought: [Int] = []  // reply tokens in the cache, and which of them the filter hid
        let clock = ContinuousClock()
        var stamp = clock.now
        var next = start
        if let prefix {
            // The head goes in alone and its K/V to disk, for the next clean start with this head.
            _ = try await runtime.logits(for: Array(tokens[..<state.head]))
            next = state.head
            let saving = clock.now
            if await savePrefix(to: prefix) {
                events?("prefix.saved", ["tokens": String(next), "ms": milliseconds(clock.now - saving)])
            }
        }
        var logits = try await runtime.logits(for: Array(tokens[next...]), evicting: freed)
        state.tokens = tokens
        slide = state
        var logitsTime = clock.now - stamp
        var timings = GenerationTimings(
            inputTokenIDs: tokens, reusedTokens: start,
            prefillMs: (clock.now - began).milliseconds, tokenMs: [], promptLogitsMs: logitsTime.milliseconds)
        events?(
            "generation.prefill.timing",
            [
                "inputTokens": String(tokens.count),
                "ms": milliseconds(logitsTime),
            ])
        var lengthLimited = true
        do {
            for index in 0..<outputTokens {
                try Task.checkCancellation()
                let tokenStarted = clock.now
                if options.exactOutputTokens != nil {
                    // Forced past its natural end the model otherwise opens a thought
                    // channel, whose tokens the filter hides (an empty reply).
                    for stop in stopTokens { logits[Int(stop)] = -Float16.greatestFiniteMagnitude }
                    for marker in [channelStart, thinkMarker] { logits[marker] = -Float16.greatestFiniteMagnitude }
                } else if summarizing {
                    // A summary is written straight away, whatever the conversation's Thinking setting.
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
                if index + 1 < outputTokens {
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
                    if let held = slide, held.tokens.count >= contextLimit {
                        // Every slot is held: a block of the oldest tokens after the head makes room, but
                        // none of the latest turn. The reply budget leaves at least one older token.
                        let older = held.evictable(keeping: latest - held.head + fed)
                        guard runtime.slides, held.head > 0, older > 0,
                            let oldest = slide?.evict(min(KVSlidePlan.block, older))
                        else { throw Gemma4RuntimeError.contextExceeded }
                        for stretch in oldest {
                            try await runtime.evict(stretch)
                            evicted += stretch.count
                        }
                        if !trimmed { onEvent(.contextTrimmed); trimmed = true }
                    }
                    logits = try await runtime.logits(for: [Int(token)])
                    slide?.tokens.append(Int(token))
                    if !accepted { thought.append(fed) }
                    fed += 1
                    logitsTime = clock.now - stamp
                }
            }
            try Task.checkCancellation()
        } catch let stop as CancellationError {
            // A reply stopped while it is decoded leaves the cache at its last whole token: the
            // runtime writes a step's K/V only when the step is done. The next prompt renders what
            // was shown of the reply like a finished one, so the cache stays, put in order as a
            // normal ending does. Not a summary, which counts only when it was written to the end;
            // not when the runtime counts another position than this state (a call failed inside a
            // step); not when a thought cannot leave the cache.
            var kept = false
            if !summarizing, runtime.slides || (thought.isEmpty && recalled == 0), let held = slide,
                await runtime.position == held.position
            {
                kept = (try? await hide(thought: thought, of: fed, recalled: recalled, latest: latest)) != nil
            }
            events?(
                "generation.cancelled",
                ["cacheKept": String(kept), "fedTokens": String(fed), "hiddenTokens": String(thought.count)])
            if kept { throw CacheKept() }
            throw stop
        }
        let answer = tokenizer.decode(tokens: generated)
        guard !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if lengthLimited && options.thinking { throw Failure.thinkingLimit }
            throw Failure.emptyReply
        }
        // The next prompt renders this reply without its thought channel. Those tokens leave the
        // cache (their slots are freed, their positions stay unused) and the answer keeps the K/V
        // computed with the thought in view, so the next turn continues the cache
        // (REPORT.md Section 7.4).
        // The recalled turns leave the same way: they were for this reply only, and the next prompt
        // does not render them (REPORT.md Section 7.7).
        try await hide(thought: thought, of: fed, recalled: recalled, latest: latest)
        // AUTO-COMPACT (REPORT.md Section 7.7): this reply is the summary the last turn asked for,
        // written with everything the cache held in view. The head, the most recent stretch before
        // the request and the summary turn stay (the turn pinned); all the older tokens leave.
        if summarizing, runtime.slides, var held = slide, held.head > 0 {
            let turn = latest - held.head + fed - thought.count
            for stretch in held.compact(turn: turn, recent: contextLimit / Self.recentShare) {
                try await runtime.evict(stretch)
                evicted += stretch.count
            }
            slide = held
        }
        // Thinning (REPORT.md Section 7.7): few slots are left. Of the turns that end a stretch
        // before this reply, the cache keeps only the user's short ones; the old replies and long
        // pastes leave it, in place and without a model call. The next prompts mark this reply's turn
        // and are rendered the same way.
        if let limit = options.thinAtFreeSlots, !summarizing, var held = slide, held.head > 0, held.head == head,
            contextLimit - held.tokens.count <= limit, let stay = thinned(at: opens[opens.count - 1]),
            let gone = held.thin(from: shown ?? Array(full.indices), to: stay), !gone.isEmpty
        {
            let thinning = clock.now
            try await runtime.evict(gone)
            slide = held
            evicted += gone.count
            onEvent(.thinned)
            events?(
                "context.thinned",
                [
                    "evictedTokens": String(gone.count), "heldTokens": String(held.tokens.count),
                    "ms": milliseconds(clock.now - thinning),
                ])
        }
        onEvent(.thinking(false))
        onEvent(.text(answer))
        lastTimings = timings
        onEvent(.finished(lengthLimited: lengthLimited))
        events?(
            "generation.complete",
            [
                "generatedTokens": String(generated.count), "evictedTokens": String(evicted),
                "hiddenTokens": String(thought.count),
            ])
    }

    /// Takes out of the cache what the next prompt does not render of the reply that ended or was
    /// stopped: its thought channel (`thought`: which of the `fed` reply tokens the filter hid) and
    /// the `recalled` tokens before the latest turn (`latest` tokens, the head counted in). A
    /// package that cannot slide keeps them.
    private func hide(thought: [Int], of fed: Int, recalled: Int, latest: Int) async throws {
        guard runtime.slides, let held = slide, !thought.isEmpty || recalled > 0 else { return }
        var hidden = thought.map { held.tokens.count - fed + $0 }
        if recalled > 0 {
            let end = held.tokens.count - fed - (latest - held.head)
            hidden += max(held.head, end - recalled)..<end
        }
        try await runtime.evict(hidden)
        slide?.hide(hidden)
    }

    /// Summary turns and thinning keep the most recent quarter of the window as it was.
    private static let recentShare = 4
    /// Recalled turns go in only when this many slots stay free after them.
    private static let recallHeadroom = 256

    /// What AUTO-COMPACT's trigger reads after a reply: the slots left for the conversation the
    /// cache holds. nil when the cache cannot be compacted: its contents or head are unknown, or
    /// the package does not slide.
    public var freeSlots: Int? {
        guard runtime.slides, let slide, slide.head > 0 else { return nil }
        return contextLimit - slide.tokens.count
    }

    /// The file name of a head's K/V: the tokens decide every value in it.
    private static func prefixName(_ head: ArraySlice<Int>) -> String {
        var hash = SHA256()
        hash.update(data: Data("prefix-kv-1".utf8))
        for token in head { withUnsafeBytes(of: Int32(token).littleEndian) { hash.update(bufferPointer: $0) } }
        return hash.finalize().map { String(format: "%02x", $0) }.joined() + ".kv"
    }

    /// Writes the head the cache holds to `file` and keeps the store at two files (one set of
    /// instructions, Thinking on and off). Failing costs the next clean start a prefill, no more.
    private func savePrefix(to file: URL) async -> Bool {
        let files = FileManager.default, folder = file.deletingLastPathComponent()
        do {
            try files.createDirectory(at: folder, withIntermediateDirectories: true)
            try await runtime.savePrefix(to: file)
        } catch { return false }
        let modified = { (url: URL) in
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
        }
        let others = ((try? files.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent != file.lastPathComponent }
            .sorted { modified($0) > modified($1) }
        for stale in others.dropFirst() { try? files.removeItem(at: stale) }
        return true
    }

    /// Teacher forcing: the summed negative log-likelihood of each `continuation` token given
    /// everything before it. Nothing is sampled; the cache is left holding these tokens.
    public func score(prompt: [Int], continuation: [Int]) async throws -> TeacherForcingScore {
        guard !prompt.isEmpty, !continuation.isEmpty, prompt.count + continuation.count - 1 <= contextLimit
        else { throw Failure.incompatiblePackage }
        slide = nil
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

    /// A saved cache: what it holds and, once positions were evicted, the position in each
    /// full-attention slot.
    public struct KVSnapshot: Codable, Sendable, Equatable {
        public var slide: KVSlideState
        public var slots: [Int32]?
        /// The tokens the cache holds.
        public var tokens: [Int] { slide.tokens }
    }

    /// Writes the KV cache to `url` and returns what it holds and its size; nil when it holds
    /// nothing. Reading it back is far faster than prefilling the conversation again.
    public func saveKV(to url: URL) async throws -> (snapshot: KVSnapshot, bytes: Int)? {
        guard let slide, !slide.tokens.isEmpty else { return nil }
        guard let saved = try await runtime.saveKV(to: url), saved.position == slide.position else {
            throw CocoaError(.coderInvalidValue)
        }
        return (KVSnapshot(slide: slide, slots: slide.slid > 0 ? await runtime.slots : nil), saved.bytes)
    }

    /// Reads back what `saveKV` wrote; `snapshot` is what it returned. After a failure the cache
    /// counts as empty.
    public func restoreKV(from url: URL, snapshot: KVSnapshot) async throws {
        slide = nil
        let held = snapshot.slots?.filter { $0 >= 0 }.count ?? snapshot.slide.tokens.count
        guard held == snapshot.slide.tokens.count, snapshot.slide.head <= held,
            (snapshot.slots == nil) == (snapshot.slide.slid == 0)
        else { throw CocoaError(.coderInvalidValue) }
        try await runtime.restoreKV(from: url, position: snapshot.slide.position, slots: snapshot.slots)
        slide = snapshot.slide
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
