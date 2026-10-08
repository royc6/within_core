import Foundation

/// What the KV cache holds while a conversation slides (KV-SLIDE, REPORT.md Section 7.3): the
/// head (everything before the first conversation turn, never evicted) and one contiguous window
/// of the conversation's full rendering.
public struct KVSlideState: Codable, Sendable, Equatable {
    /// The tokens the cache holds: the head, then the live window.
    public var tokens: [Int]
    /// Tokens of the head at the start of `tokens`; 0 when it is unknown, and then nothing is evicted.
    public var head: Int
    /// Tokens of the full rendering between the head and the live window: left out at the last
    /// clean start, or evicted since.
    public var dropped: Int
    /// Positions the cache gave up since the last clean start, evicted or hidden: the next token
    /// takes position `tokens.count + slid`.
    public var slid: Int
    /// AUTO-COMPACT (REPORT.md Section 7.7): where in `tokens` the latest summary turn sits (the
    /// request and the summary). Eviction passes over it; empty when there is none.
    public var pinned: Range<Int>
    /// Tokens of the full rendering evicted from right after the pinned turn.
    public var skipped: Int

    public init(
        tokens: [Int] = [], head: Int = 0, dropped: Int = 0, slid: Int = 0, pinned: Range<Int> = 0..<0, skipped: Int = 0
    ) {
        self.tokens = tokens; self.head = head; self.dropped = dropped; self.slid = slid
        self.pinned = pinned; self.skipped = skipped
    }

    /// The position the next token takes.
    public var position: Int { tokens.count + slid }

    /// The pinned turn, or an empty stretch right after the head.
    var pin: Range<Int> { pinned.isEmpty ? head..<head : pinned }

    /// How many tokens may be evicted when the last `kept` (the latest turn) must stay.
    public func evictable(keeping kept: Int) -> Int { tokens.count - head - pinned.count - kept }

    /// Removes the oldest `count` tokens after the head, passing over the pinned turn, and returns
    /// where in `tokens` they were: one or two stretches, each as it stands once the one before it
    /// is gone.
    public mutating func evict(_ count: Int) -> [Range<Int>] {
        let before = min(count, pin.lowerBound - head), after = count - before
        var removed: [Range<Int>] = []
        if before > 0 {
            removed.append(head..<head + before)
            tokens.removeSubrange(head..<head + before)
            pinned = (pinned.lowerBound - before)..<(pinned.upperBound - before)
            dropped += before
        }
        if after > 0 {
            let first = pin.upperBound
            removed.append(first..<first + after)
            tokens.removeSubrange(first..<first + after)
            if pinned.isEmpty { dropped += after } else { skipped += after }
        }
        slid += count
        return removed
    }

    /// Removes the tokens at `indices`, which the next prompt does not render (a reply's thought
    /// channel): what is left still reads as the rendered conversation.
    public mutating func hide(_ indices: [Int]) {
        let gone = Set(indices)
        let below = { (bound: Int) in gone.count { $0 < bound } }
        pinned = (pinned.lowerBound - below(pinned.lowerBound))..<(pinned.upperBound - below(pinned.upperBound))
        tokens = tokens.indices.filter { !gone.contains($0) }.map { tokens[$0] }
        slid += gone.count
    }

    /// Thinning (REPORT.md Section 7.7): the rendering the cache follows changes from the tokens
    /// `now` of the full rendering to the fewer tokens `stay` (both as indices into it, the head
    /// included). Removes what the cache holds of the difference and returns where in `tokens` that
    /// was, for the runtime to free; the reply after the prompt stays. nil when the cache does not
    /// hold `now` as one stretch after the head.
    public mutating func thin(from now: [Int], to stay: [Int]) -> [Int]? {
        guard pinned.isEmpty, skipped == 0, head > 0, now.count >= head + dropped,
            tokens.count >= now.count - dropped
        else { return nil }
        let cached = Array(now[..<head]) + now[(head + dropped)...]
        let staying = Set(stay)
        let gone = cached.indices.filter { !staying.contains(cached[$0]) }
        guard !gone.isEmpty else { return [] }
        // What stays but had already left the cache still counts as dropped.
        let first = cached.count > head ? cached[head] : Int.max
        hide(gone)
        dropped = stay.dropFirst(head).prefix { $0 < first }.count
        return gone
    }

    /// AUTO-COMPACT: the last `turn` tokens are a summary turn, written with everything the cache
    /// holds in view. Keeps the head, `recent` tokens before the turn, and the turn, which is pinned
    /// from now on; the turn pinned before is released with the rest. Returns what `evict` returns.
    public mutating func compact(turn: Int, recent: Int) -> [Range<Int>] {
        let start = tokens.count - turn
        // What stays is one stretch of the rendering: nothing from before an earlier gap.
        let first = max(head, start - recent, skipped > 0 ? pin.upperBound : head)
        let count = first - head
        tokens.removeSubrange(head..<first)
        dropped += count + skipped
        skipped = 0; slid += count
        pinned = (start - count)..<tokens.count
        return count > 0 ? [head..<first] : []
    }
}

/// How a cache continues with the next prompt.
public enum KVSlidePlan {
    /// Slots are freed this many at a time: a prompt's calls each free a block when
    /// they need room, a reply frees one when every slot is held. Freeing costs nothing, so a
    /// larger block would only leave more slots empty on average.
    public static let block = 128

    /// Continues `state` with `full`, the whole conversation rendered as the next prompt: possible
    /// when `full` still starts with the head, holds the live window where the cache says it is, and
    /// adds at least one token. Returns how many of the oldest live tokens to evict so the new ones
    /// fit `capacity` slots, and the tokens to feed; nil when the cache cannot be continued. Only a
    /// package that slides, with a known head, evicts; otherwise the new tokens and `reply` more
    /// must fit as they are, which with nothing dropped is the plain "the prompt extends the cache"
    /// of v1.
    public static func extend(
        _ state: KVSlideState, full: [Int], capacity: Int, reply: Int, slides: Bool
    ) -> (evict: Int, feed: ArraySlice<Int>)? {
        // The rendering holds the tokens up to the pinned turn's end `dropped` after the head, and
        // the rest `skipped` further on.
        let pin = state.pin
        guard state.head >= 0, state.dropped >= 0, state.skipped >= 0, state.head <= pin.lowerBound,
            pin.upperBound <= state.tokens.count
        else { return nil }
        let before = state.tokens[state.head..<pin.upperBound], after = state.tokens[pin.upperBound...]
        let start = state.head + state.dropped, resume = start + before.count + state.skipped
        guard full.count > resume + after.count,
            full[..<state.head].elementsEqual(state.tokens[..<state.head]),
            full[start..<start + before.count].elementsEqual(before),
            full[resume..<resume + after.count].elementsEqual(after)
        else { return nil }
        let feed = full[(resume + after.count)...]
        guard slides, state.head > 0 else {
            return state.tokens.count + feed.count + reply <= capacity ? (0, feed) : nil
        }
        // The new tokens, the head and the pinned turn must fit; the reply evicts as it goes.
        let evict = max(0, state.tokens.count + feed.count - capacity)
        return evict <= state.evictable(keeping: 0) ? (evict, feed) : nil
    }

    /// What a cache started clean with `tokens` holds, before they are fed: `tokens` renders the last
    /// turns of the conversation whose whole rendering is `full`. The head is everything before the
    /// first conversation turn. `probe` is the same rendering with one plain user turn, so its last
    /// two `turnStart` open that turn and the generation prompt, whatever the system prompt or the
    /// messages contain. The turns left out count as dropped. When the head cannot be told or
    /// `tokens` is not the head followed by the end of `full`, the head is unknown.
    public static func start(_ tokens: [Int], full: [Int], probe: [Int], turnStart: Int) -> KVSlideState {
        guard let head = head(of: full, probe: probe, turnStart: turnStart) else { return KVSlideState() }
        let kept = tokens.count - head
        guard kept > 0, tokens[head] == turnStart, full.count >= tokens.count,
            tokens[..<head].elementsEqual(probe[..<head]), full.suffix(kept).elementsEqual(tokens.suffix(kept))
        else { return KVSlideState() }
        return KVSlideState(head: head, dropped: full.count - tokens.count)
    }

    /// The length of the head `full` starts with (see `start`), or nil when it cannot be told.
    public static func head(of full: [Int], probe: [Int], turnStart: Int) -> Int? {
        let opens = probe.indices.filter { probe[$0] == turnStart }
        guard opens.count >= 2 else { return nil }
        let head = opens[opens.count - 2]
        guard head > 0, full.count > head, full[..<head].elementsEqual(probe[..<head]) else { return nil }
        return head
    }

    /// Thinning (REPORT.md Section 7.7): which tokens of `full` a thinned conversation renders,
    /// as indices. A turn opens at each `turnStart` from `head` on; `anchor` is where the reply opens
    /// that ended when the conversation was thinned. A turn that ends `recent` tokens or more before
    /// it stays only when it is the user's (the token after its `turnStart` is `user`) and no longer
    /// than `longest`: the user's own words stay, old replies and long pastes go. The head and every
    /// later turn stay whole. A later anchor keeps a subset of what an earlier one keeps.
    public static func thinned(
        full: [Int], head: Int, turnStart: Int, user: Int, anchor: Int, recent: Int, longest: Int
    ) -> [Int] {
        let opens = full.indices.filter { $0 >= head && full[$0] == turnStart }
        var kept = Array(0..<min(head, full.count))
        for (index, open) in opens.enumerated() {
            let end = index + 1 < opens.count ? opens[index + 1] : full.count
            let own = open + 1 < end && full[open + 1] == user && end - open <= longest
            if end > anchor - recent || own { kept += open..<end }
        }
        return kept
    }

    /// AUTO-COMPACT without the cache (REPORT.md Section 7.7): what a clean start feeds when the
    /// conversation holds a summary turn. `fromSummary` renders the conversation from the summary's
    /// request on and `afterSummary` from the turn after the summary on (both with the head), so
    /// their ends say where in `full` the turn sits. The prompt is the head, `recent` tokens before
    /// the turn, the turn (pinned) and what follows: the same tokens the cache held after the
    /// compaction. When that is more than `capacity`, the oldest of it goes, but none of the pinned
    /// turn or of the last `latest` tokens. Returns the state with the tokens to feed, or nil when
    /// the renderings do not line up or the prompt cannot fit.
    public static func rebuild(
        full: [Int], head: Int, fromSummary: [Int], afterSummary: [Int], latest: Int, recent: Int, capacity: Int
    ) -> KVSlideState? {
        let first = full.count - (fromSummary.count - head), end = full.count - (afterSummary.count - head)
        guard head > 0, head <= first, first < end, end <= full.count - latest,
            full[first...].elementsEqual(fromSummary[head...]), full[end...].elementsEqual(afterSummary[head...])
        else { return nil }
        let kept = max(head, first - recent)
        var state = KVSlideState(
            tokens: Array(full[..<head]) + full[kept...], head: head, dropped: kept - head,
            pinned: (head + first - kept)..<(head + end - kept))
        let over = state.tokens.count - capacity
        if over > 0 {
            guard over <= state.evictable(keeping: latest) else { return nil }
            _ = state.evict(over)
            state.slid = 0  // nothing was fed yet: positions start at 0
        }
        return state
    }
}
