import Foundation
import Testing

@testable import WithinCoreBase

private func slideConfiguration() -> Gemma4RuntimeConfiguration {
    Gemma4RuntimeConfiguration(
        schemaVersion: 1, hiddenSize: 32, vocabularySize: 64,
        layerTypes: (0..<42).map { $0 % 6 == 5 ? "full_attention" : "sliding_attention" },
        sharedLayerCount: 18, perLayerInputSize: 8, keyValueHeads: 2,
        localHeadDimension: 8, globalHeadDimension: 16, localTheta: 10_000, globalTheta: 1_000_000,
        globalRotaryFraction: 0.25, slidingWindow: 5, contextLimit: 16, prefillLength: 4,
        embeddings: .init(file: "embeddings.f16", format: .float16, rows: 64, columns: 32),
        perLayerEmbeddings: .init(file: "ple.f16", format: .float16, rows: 64, columns: 336))
}

/// Until something is evicted the table is v1: slot = position and the same mask, bit for bit.
@Test func slotTableIsTheV1LayoutUntilSomethingIsEvicted() throws {
    let c = slideConfiguration()
    var table = Gemma4SlotTable(capacity: c.contextLimit)
    var step = 0
    for length in [4, 4, 1, 1, 3, 2, 1] {
        #expect(table.mask(length: length) == (try c.mask(step: step, length: length, global: true)))
        #expect(try table.allocate(position: step, count: length) == Array(step..<step + length))
        step += length
    }
    #expect(table.free == 0 && table.held == 16)
    #expect(throws: Gemma4RuntimeError.self) { try table.allocate(position: step, count: 1) }
}

/// Evicted slots are reused by later positions, the head stays, and the mask shows exactly the held slots.
@Test func slotTableReusesEvictedSlotsAndMasksTheRest() throws {
    var table = Gemma4SlotTable(capacity: 16)
    var holds: [Int: Int] = [:]  // slot -> position, the model the table must match
    func check(_ length: Int) {
        let mask = table.mask(length: length), width = 16 + length
        for query in 0..<length {
            for slot in 0..<16 { #expect((mask[query * width + slot] == 0) == (holds[slot] != nil), "slot \(slot)") }
            for fresh in 0..<length { #expect((mask[query * width + 16 + fresh] == 0) == (fresh <= query)) }
        }
        #expect(table.held == holds.count && table.free == 16 - holds.count)
    }
    for (slot, position) in zip(try table.allocate(position: 0, count: 16), 0..<16) { holds[slot] = position }
    check(4)
    var step = 16
    // Head [0, 3); the oldest live tokens go first, a few at a time, far past the slot count.
    var cut = 3
    for (evicted, fed) in [(4, 4), (1, 1), (5, 3), (0, 2), (6, 6), (9, 9), (3, 1)] {
        table.evict(3..<3 + evicted)  // counted from the oldest position held
        holds = holds.filter { !(cut..<cut + evicted).contains($0.value) }
        cut += evicted
        check(max(1, fed))
        let slots = try table.allocate(position: step, count: fed)
        #expect(Set(slots).count == fed && slots.allSatisfy { holds[$0] == nil })
        for (slot, position) in zip(slots, step..<step + fed) { holds[slot] = position }
        step += fed
        #expect((0..<3).allSatisfy { holds[$0] == $0 })  // the head never moves
        #expect(Set(holds.values) == Set(0..<3).union(cut..<step))
    }
    #expect(step > 16 * 2)
    table.evict(0..<3)  // nothing protects a token but the caller
    #expect(table.held == step - cut && !table.positions.contains { (0..<3).contains($0) })
}

/// A prompt's calls free the positions it gives up a block at a time, each only when it needs room.
@Test func slotTableMakesRoomABlockAtATime() throws {
    var table = Gemma4SlotTable(capacity: 16)
    _ = try table.allocate(position: 0, count: 16)
    var pending = 2..<12, step = 16  // the head is [0, 2); the ten tokens after it may go
    table.makeRoom(for: 0, from: &pending, block: 4)
    #expect(pending == 2..<12 && table.free == 0)  // nothing needed, nothing freed
    for (freeBefore, pendingAfter) in [(4, 2..<8), (5, 2..<4), (4, 2..<2)] {
        table.makeRoom(for: 3, from: &pending, block: 4)  // a block, or what is left of what may go
        #expect(table.free == freeBefore && pending == pendingAfter)
        _ = try table.allocate(position: step, count: 3)
        step += 3
    }
    table.makeRoom(for: 3, from: &pending, block: 4)  // nothing left to give up
    #expect(table.free == 1 && Set(table.positions.filter { $0 >= 0 }) == Set([0, 1, 12, 13, 14, 15] + Array(16..<25)))
    // More than a block when one call needs it.
    var wide = Gemma4SlotTable(capacity: 16)
    _ = try wide.allocate(position: 0, count: 16)
    pending = 2..<12
    wide.makeRoom(for: 7, from: &pending, block: 4)
    #expect(wide.free == 7 && pending == 2..<5)
}

/// Tokens are evicted by their place among what the table holds, so a gap in the positions (a
/// thought channel taken out of a reply) does not move the count; the rings hide what is missing.
@Test func slotTableEvictsByOrderAcrossAGap() throws {
    var table = Gemma4SlotTable(capacity: 12)
    _ = try table.allocate(position: 0, count: 12)
    table.evict([5, 6, 7])  // a thought at positions 5, 6, 7
    #expect(table.missing(0..<12) == [5, 6, 7] && table.missing(8..<12).isEmpty)
    table.evict(2..<5)  // the three oldest after a head of two: positions 2, 3, 4
    #expect(Set(table.positions.filter { $0 >= 0 }) == Set([0, 1] + Array(8..<12)))
    table.evict(2..<4)  // the next two sit past the gap: positions 8 and 9
    #expect(Set(table.positions.filter { $0 >= 0 }) == [0, 1, 10, 11] && table.held == 4)
    var pending = 2..<4
    table.makeRoom(for: 9, from: &pending, block: 1)  // one more than is free: one of the two goes
    #expect(pending == 2..<3 && Set(table.positions.filter { $0 >= 0 }) == [0, 1, 11])
    let c = slideConfiguration(), window = c.slidingWindow
    let hidden: Set<Int> = [window - 2, window + 1]
    for (step, length) in [(window + 3, 1), (window + 3, 4)] {
        let plain = try c.ringMask(step: step, length: length)
        let mask = try c.ringMask(step: step, length: length, hidden: hidden), width = window + length
        for query in 0..<length {
            for slot in 0..<window {
                let position = step - 1 - (step - 1 - slot) % window  // the latest position the slot holds
                let expected = hidden.contains(position) ? -10_000 : plain[query * width + slot]
                #expect(mask[query * width + slot] == expected, "query \(query) slot \(slot)")
            }
            #expect(mask[(query * width + window)..<((query + 1) * width)] == plain[(query * width + window)..<((query + 1) * width)])
        }
        #expect(mask != plain)
    }
}

/// A table read back from a snapshot behaves as the one that was saved.
@Test func slotTableSnapshotContinuesWhereItStopped() throws {
    var table = Gemma4SlotTable(capacity: 8)
    _ = try table.allocate(position: 0, count: 8)
    table.evict(2..<5)
    _ = try table.allocate(position: 8, count: 2)
    var restored = try Gemma4SlotTable(positions: table.positions)
    #expect(restored.held == table.held && restored.mask(length: 2) == table.mask(length: 2))
    table.evict(2..<4); restored.evict(2..<4)  // positions 5 and 6
    #expect(try restored.allocate(position: 10, count: 3) == (try table.allocate(position: 10, count: 3)))
    #expect(throws: Gemma4RuntimeError.self) { try Gemma4SlotTable(positions: [0, 1, 1, -1]) }
    #expect(throws: Gemma4RuntimeError.self) { try Gemma4SlotTable(positions: [0, -2]) }
}

/// Rotary tables and the ring mask depend on the position only, so they run past the slot count,
/// up to the positions the model was trained on.
@Test func rotaryAndRingMaskRunPastTheSlotCount() throws {
    let c = slideConfiguration()
    #expect(throws: Gemma4RuntimeError.self) { try c.mask(step: 14, length: 4, global: true) }  // slots end at 16
    let near = try c.rotary(step: 3, length: 2, global: true), far = try c.rotary(step: 100_003, length: 2, global: true)
    #expect(near.cos.count == far.cos.count && near.cos != far.cos)
    _ = try c.rotary(step: Gemma4RuntimeConfiguration.maximumPosition - 4, length: 4, global: false)
    #expect(throws: Gemma4RuntimeError.self) {
        try c.rotary(step: Gemma4RuntimeConfiguration.maximumPosition - 3, length: 4, global: false)
    }
    let window = c.slidingWindow
    var held = [Int?](repeating: nil, count: window)
    var step = 0
    for length in [4, 4, 4, 3, 1, 4, 1, 1, 4, 2, 4, 4] {  // 36 positions on a 16-slot package
        let mask = try c.ringMask(step: step, length: length), width = window + length
        for query in 0..<length {
            for slot in 0..<window {
                let visible = held[slot].map { step + query - $0 < window } ?? false
                #expect((mask[query * width + slot] == 0) == visible, "step \(step) query \(query) slot \(slot)")
            }
        }
        for token in 0..<length { held[(step + token) % window] = step + token }
        step += length
    }
    #expect(step == 36)
}

/// The plan continues a cache when the new prompt holds the head and the live window where the cache
/// says they are, and evicts only what the new tokens need.
@Test func slidePlanContinuesTheLiveWindow() throws {
    func extend(_ state: KVSlideState, _ full: [Int], _ capacity: Int, reply: Int = 1, slides: Bool = true)
        -> (evict: Int, feed: ArraySlice<Int>)?
    { KVSlidePlan.extend(state, full: full, capacity: capacity, reply: reply, slides: slides) }
    // Nothing dropped: v1's "the prompt extends the cache".
    let fresh = KVSlideState(tokens: [1, 2, 3], head: 1)
    var plan = try #require(extend(fresh, [1, 2, 3, 4, 5], 8))
    #expect(plan.evict == 0 && Array(plan.feed) == [4, 5])
    #expect(extend(fresh, [1, 2, 3], 8) == nil)  // nothing new
    #expect(extend(fresh, [1, 2, 9, 4], 8) == nil)  // another conversation
    #expect(extend(fresh, [7, 2, 3, 4], 8) == nil)  // another head
    // Four slots: one of the old tokens goes, and the next prompt finds the window one token later.
    plan = try #require(extend(fresh, [1, 2, 3, 4, 5], 4))
    #expect(plan.evict == 1)
    var state = fresh
    #expect(state.evict(1) == [1..<2])
    state.tokens += plan.feed
    #expect(state == KVSlideState(tokens: [1, 3, 4, 5], head: 1, dropped: 1, slid: 1) && state.position == 5)
    plan = try #require(extend(state, [1, 2, 3, 4, 5, 6, 7], 4, reply: 100))  // the reply evicts as it goes
    #expect(plan.evict == 2 && Array(plan.feed) == [6, 7])
    #expect(state.evict(2) == [1..<3] && state.position == 5)  // places in the cache; positions stay
    #expect(extend(state, [1, 2, 3, 4, 5, 6, 7, 8, 9], 4) == nil)  // 4 new + head > 4 slots
    // The evicted part of the rendering is not checked, the live window is.
    #expect(extend(state, [1, 0, 0, 0, 5, 6], 4) != nil)
    #expect(extend(state, [1, 2, 3, 4, 6, 6], 4) == nil)
    // A clean start that left turns out: the window sits `dropped` tokens into the rendering.
    let trimmed = KVSlideState(tokens: [1, 5, 6], head: 1, dropped: 3)
    plan = try #require(extend(trimmed, [1, 2, 3, 4, 5, 6, 7], 8))
    #expect(plan.evict == 0 && Array(plan.feed) == [7] && trimmed.position == 3)
    // Evicting from it: positions count from the clean start, the window from the whole rendering.
    plan = try #require(extend(trimmed, [1, 2, 3, 4, 5, 6, 7, 8], 3))
    #expect(plan.evict == 2 && Array(plan.feed) == [7, 8])  // the whole live window goes
    state = trimmed
    #expect(state.evict(2) == [1..<3])
    state.tokens += plan.feed
    #expect(state == KVSlideState(tokens: [1, 7, 8], head: 1, dropped: 5, slid: 2) && state.position == 5)
    plan = try #require(extend(state, [1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
    #expect(plan.evict == 1 && Array(plan.feed) == [9])
}

/// A reply's thought channel leaves the cache: the next prompt, rendered without it, continues
/// what is left, and positions keep counting past the gap.
@Test func hiddenThoughtLetsTheNextPromptContinue() throws {
    // Head 1, a user turn 2 3, then the reply: thought 90 91 92, answer 4 5.
    var state = KVSlideState(tokens: [1, 2, 3, 90, 91, 92, 4, 5], head: 1)
    let next = [1, 2, 3, 4, 5, 6, 7]  // the conversation as the next prompt renders it
    #expect(KVSlidePlan.extend(state, full: next, capacity: 16, reply: 1, slides: true) == nil)
    state.hide([3, 4, 5])
    #expect(state == KVSlideState(tokens: [1, 2, 3, 4, 5], head: 1, slid: 3) && state.position == 8)
    var plan = try #require(KVSlidePlan.extend(state, full: next, capacity: 16, reply: 1, slides: true))
    #expect(plan.evict == 0 && Array(plan.feed) == [6, 7])
    // Evicting afterwards counts places in the cache, whatever positions the gap left.
    plan = try #require(KVSlidePlan.extend(state, full: next, capacity: 5, reply: 1, slides: true))
    #expect(plan.evict == 2 && state.evict(2) == [1..<3])
    #expect(state == KVSlideState(tokens: [1, 4, 5], head: 1, dropped: 2, slid: 5))
}

/// A reply stopped part-way stays in the cache up to its last whole token. The next prompt renders
/// what was shown of it and closes the turn: it continues the cache when the text ends where the
/// cache does or one token later (a token is shown before it is fed), and starts clean when the
/// cache holds a token the text does not.
@Test func stoppedReplyLetsTheNextPromptContinue() throws {
    let turn = 9, end = 8
    // Head 1, the user's turn and the generation prompt, then the reply as far as it was shown: 4 5 6.
    let prompt = [1, turn, 2, 3, end, turn]
    let next = prompt + [4, 5, 6] + [end, turn, 7, end, turn]
    for slides in [true, false] {
        func feed(_ cached: [Int]) -> [Int]? {
            KVSlidePlan.extend(
                KVSlideState(tokens: cached, head: 1), full: next, capacity: 32, reply: 4, slides: slides
            ).map { Array($0.feed) }
        }
        #expect(feed(prompt + [4, 5, 6]) == [end, turn, 7, end, turn])
        #expect(feed(prompt + [4, 5]) == [6, end, turn, 7, end, turn])  // stopped while 6 was fed
        #expect(feed(prompt + [4, 5, 6, 10]) == nil)  // the cache went one token past the text
    }
}

/// A reply stopped inside its thought channel gives the thought up as a closed one is at a normal
/// ending: the cache shows the prompt alone, in the rings too, and the next token takes the
/// position after the gap.
@Test func stoppedThoughtLeavesThePromptAndAGap() throws {
    let c = slideConfiguration(), window = c.slidingWindow
    // Head 1, the user's turn and the generation prompt 2 3 4, then the open thought 90 91 92.
    var state = KVSlideState(tokens: [1, 2, 3, 4, 90, 91, 92], head: 1)
    var table = Gemma4SlotTable(capacity: c.contextLimit)
    _ = try table.allocate(position: 0, count: state.tokens.count)
    let hidden = [4, 5, 6]
    table.evict(hidden)
    state.hide(hidden)
    #expect(state == KVSlideState(tokens: [1, 2, 3, 4], head: 1, slid: 3) && state.position == 7)
    #expect(table.held == state.tokens.count)
    let mask = table.mask(length: 1)
    #expect((0..<c.contextLimit).filter { mask[$0] == 0 } == [0, 1, 2, 3])
    // The ring's slots for positions 4, 5, 6 are hidden; of the prompt only position 3 is in reach.
    let gone = table.missing(max(0, state.position - window)..<state.position)
    let ring = try c.ringMask(step: state.position, length: 1, hidden: gone)
    #expect(gone == [4, 5, 6] && (0..<window).filter { ring[$0] == 0 } == [3])
    // What a prompt continuing these tokens feeds goes in at the position the state counts.
    let plan = try #require(KVSlidePlan.extend(state, full: [1, 2, 3, 4, 5, 6], capacity: 16, reply: 1, slides: true))
    #expect(plan.evict == 0 && Array(plan.feed) == [5, 6])
    _ = try table.allocate(position: state.position, count: plan.feed.count)
    state.tokens += plan.feed
    #expect(table.positions.filter { $0 >= 0 }.sorted() == [0, 1, 2, 3, 7, 8] && state.position == 9)
}

/// AUTO-COMPACT: after a summary turn the cache keeps the head, a recent stretch and the turn; the
/// turn is passed over by later evictions until the next summary releases it, and a clean start
/// from text feeds the same tokens.
@Test func compactionKeepsTheRecentStretchAndPinsTheSummary() throws {
    func extend(_ state: KVSlideState, _ full: [Int], _ capacity: Int = 64) -> (evict: Int, feed: ArraySlice<Int>)? {
        KVSlidePlan.extend(state, full: full, capacity: capacity, reply: 1, slides: true)
    }
    // Head 1; six old tokens 11...16; the request 21 22 and the summary 31 32.
    let old = [1, 11, 12, 13, 14, 15, 16], summary = [21, 22, 31, 32]
    var state = KVSlideState(tokens: old + summary, head: 1)
    #expect(state.compact(turn: 4, recent: 2) == [1..<5])
    #expect(state == KVSlideState(tokens: [1, 15, 16] + summary, head: 1, dropped: 4, slid: 4, pinned: 3..<7))
    // The next prompt, with two later turns 41...44 and the latest turn 51 52, continues it.
    let later = [41, 42, 43, 44], latest = [51, 52]
    let full = old + summary + later + latest
    var plan = try #require(extend(state, full))
    #expect(plan.evict == 0 && Array(plan.feed) == later + latest)
    // A clean start from the same conversation feeds exactly what the cache then holds.
    let rebuilt = try #require(
        KVSlidePlan.rebuild(
            full: full, head: 1, fromSummary: [1] + summary + later + latest, afterSummary: [1] + later + latest,
            latest: 2, recent: 2, capacity: 64))
    state.tokens += plan.feed
    #expect(rebuilt.tokens == state.tokens && rebuilt.pinned == state.pinned && rebuilt.dropped == state.dropped)
    #expect(rebuilt.slid == 0 && rebuilt.position == 13)
    // Short of room, the recent stretch goes first, then what follows the summary: never the
    // summary turn, never the latest turn.
    #expect(state.evictable(keeping: 2) == 6)
    #expect(state.evict(3) == [1..<3, 5..<6])  // 15 16, then 41 from behind the pinned turn
    #expect(state == KVSlideState(
        tokens: [1] + summary + [42, 43, 44] + latest, head: 1, dropped: 6, slid: 7, pinned: 1..<5, skipped: 1))
    plan = try #require(extend(state, full + [61]))
    #expect(Array(plan.feed) == [61])
    #expect(extend(state, old + summary + [41, 42, 43, 99] + latest + [61]) == nil)  // another conversation after the summary
    #expect(extend(state, old + [21, 22, 31, 99] + later + latest + [61]) == nil)  // another summary
    #expect(extend(state, full + [61], 6)?.evict == 5)  // all but the head and the summary turn
    #expect(extend(state, full + [61], 5) == nil)  // those two alone take five slots
    let tight = try #require(
        KVSlidePlan.rebuild(
            full: full, head: 1, fromSummary: [1] + summary + later + latest, afterSummary: [1] + later + latest,
            latest: 2, recent: 2, capacity: 10))
    #expect(tight == KVSlideState(
        tokens: [1] + summary + [42, 43, 44] + latest, head: 1, dropped: 6, pinned: 1..<5, skipped: 1))
    #expect(KVSlidePlan.rebuild(
        full: full, head: 1, fromSummary: [1] + summary + later + latest, afterSummary: [1] + later + latest,
        latest: 2, recent: 2, capacity: 6) == nil)  // the head, the summary and the latest turn alone are 7
    #expect(KVSlidePlan.rebuild(
        full: full, head: 1, fromSummary: [1, 21, 22, 31, 99] + later + latest, afterSummary: [1] + later + latest,
        latest: 2, recent: 2, capacity: 64) == nil)  // the renderings do not line up
    // The next summary (71 72, 81 82) was written with the old one in view: the old turn goes with
    // the rest, and nothing from before the gap behind it stays, however much may be kept.
    state.tokens += [61, 71, 72, 81, 82]
    var wide = state
    #expect(state.compact(turn: 4, recent: 3) == [1..<8])
    #expect(state == KVSlideState(
        tokens: [1, 51, 52, 61, 71, 72, 81, 82], head: 1, dropped: 14, slid: 14, pinned: 4..<8))
    #expect(wide.compact(turn: 4, recent: 100) == [1..<5])  // the old summary sat before the gap
    #expect(wide.tokens == [1, 42, 43, 44, 51, 52, 61, 71, 72, 81, 82] && wide.dropped == 11 && wide.skipped == 0)
    #expect(extend(wide, full + [61, 71, 72, 81, 82, 91]).map { Array($0.feed) } == [91])
    // A thought channel taken out of a later reply leaves the pinned turn where it is.
    state.tokens += [90, 91, 92]
    state.hide([8, 9])
    #expect(state.tokens.last == 92 && state.pinned == 4..<8)
}

/// Without a known head, or on a package that cannot slide, nothing is evicted: the cache continues
/// while the new tokens and the reply fit, also after a trim, and the caller starts over once they
/// do not.
@Test func slidePlanWithoutEvictionKeepsV1Reuse() throws {
    for (state, slides) in [
        (KVSlideState(tokens: [1, 5, 6], head: 0), true), (KVSlideState(tokens: [1, 5, 6], head: 1), false),
    ] {
        let full = [1, 5, 6, 7]
        let plan = try #require(KVSlidePlan.extend(state, full: full, capacity: 8, reply: 4, slides: slides))
        #expect(plan.evict == 0 && Array(plan.feed) == [7])
        #expect(KVSlidePlan.extend(state, full: full, capacity: 8, reply: 5, slides: slides) == nil)
        #expect(KVSlidePlan.extend(state, full: full + [8], capacity: 4, reply: 1, slides: slides) == nil)
    }
    // A package that cannot slide, trimmed at a clean start: later prompts keep extending the cache.
    let turn = 9, probe = [2, turn, 30, turn, 40]
    let full = [2, turn, 31, turn, 41, turn, 32, turn, 42, turn, 33, turn, 40]
    let tokens = [2] + full.suffix(8)
    var state = KVSlidePlan.start(tokens, full: full, probe: probe, turnStart: turn)
    #expect(state == KVSlideState(head: 1, dropped: 4))
    state.tokens = tokens + [43]  // fed, then the reply
    let next = full + [43, turn, 34, turn, 40]
    let plan = try #require(KVSlidePlan.extend(state, full: next, capacity: 16, reply: 2, slides: false))
    #expect(plan.evict == 0 && Array(plan.feed) == [turn, 34, turn, 40])
    #expect(KVSlidePlan.extend(state, full: next, capacity: 16, reply: 3, slides: false) == nil)
}

/// A clean start knows its head from the probe (one plain user turn under the same system prompt),
/// whatever the messages hold, and counts the turns it left out as dropped.
@Test func cleanStartFindsTheHeadAndWhatWasLeftOut() {
    let turn = 9
    func start(_ tokens: [Int], _ full: [Int], _ probe: [Int]) -> KVSlideState {
        KVSlidePlan.start(tokens, full: full, probe: probe, turnStart: turn)
    }
    let bare = [2, turn, 30, turn, 40]  // <bos>, a user turn, the generation prompt
    let system = [2, turn, 20, 21, turn, 30, turn, 40]
    #expect(start([2, turn, 31, 32, turn, 40], [2, turn, 31, 32, turn, 40], bare) == KVSlideState(head: 1))
    #expect(start([2, turn, 20, 21, turn, 31, turn, 40], [2, turn, 20, 21, turn, 31, turn, 40], system).head == 4)
    // The token itself inside a message, or inside the system prompt, does not move the head.
    #expect(start([2, turn, 31, turn, 32, turn, 40], [2, turn, 31, turn, 32, turn, 40], bare) == KVSlideState(head: 1))
    let quoting = [2, turn, 20, turn, 21, turn, 30, turn, 40]
    #expect(start([2, turn, 20, turn, 21, turn, 31, turn, 40], [2, turn, 20, turn, 21, turn, 31, turn, 40], quoting).head == 5)
    // Turns left out: the rendering is the head, then the end of the whole conversation.
    let full = [2, turn, 20, 21, turn, 31, turn, 41, turn, 32, turn, 40]
    #expect(start([2, turn, 20, 21, turn, 32, turn, 40], full, system) == KVSlideState(head: 4, dropped: 4))
    // Head unknown: another system prompt than the probe's, no turn after it, or not the end of `full`.
    #expect(start([2, turn, 22, 23, turn, 31, turn, 40], [2, turn, 22, 23, turn, 31, turn, 40], system) == KVSlideState())
    #expect(start([2, turn, 20, 21], [2, turn, 20, 21], system) == KVSlideState())
    #expect(start([2, turn, 20, 21, turn, 33, turn, 40], full, system) == KVSlideState())
    #expect(start([2, 30, 40], [2, 30, 40], [2, 30, 40]) == KVSlideState())
    #expect(start([turn, 31, turn, 40], [turn, 31, turn, 40], [turn, 30, turn, 40]) == KVSlideState())  // nothing before the turn
    #expect(start([2, 31, turn, 40], [2, 31, turn, 40], bare) == KVSlideState())  // no turn where the head ends
}

/// The complete turn list pairs every answered user turn and drops the unanswered ones before the latest.
@Test func completeTurnsSkipUnansweredOnes() throws {
    let turns = [
        ChatTurn(role: .user, text: "a"), ChatTurn(role: .assistant, text: "A"), ChatTurn(role: .user, text: "lost"),
        ChatTurn(role: .user, text: "b"), ChatTurn(role: .assistant, text: ""), ChatTurn(role: .user, text: "c"),
    ]
    #expect(try ContextBudget.complete(turns).map(\.text) == ["a", "A", "c"])
    #expect(throws: ContextBudgetError.missingUserMessage) { try ContextBudget.complete(Array(turns.prefix(2))) }
}

/// Thinning (REPORT.md Section 7.7): of the turns that end a stretch before the reply, only the
/// user's short ones stay; the cache gives up the rest in place and the next prompt, rendered the
/// same way, continues it.
@Test func thinningKeepsTheUsersShortTurnsAndTheNextPromptContinues() throws {
    let turn = 9, user = 7, model = 8
    let full: [Int] =
        [2] + [turn, user, 11, 12] + [turn, model, 21, 22, 23] + [turn, user, 31, 32, 33, 34, 35, 36]
        + [turn, model, 41, 42] + [turn, user, 51] + [turn, model, 61, 62] + [turn, user, 71] + [turn, model]
    func thinned(_ tokens: [Int], anchor: Int) -> [Int] {
        KVSlidePlan.thinned(full: tokens, head: 1, turnStart: turn, user: user, anchor: anchor, recent: 8, longest: 6)
    }
    // The reply opens at 32: the short user turn stays, the two replies and the long paste before
    // token 24 go, the turn that reaches past it stays whole.
    let stay = thinned(full, anchor: 32)
    #expect(stay == [0] + Array(1...4) + Array(22...33))
    let everything = Array(full.indices)
    var state = KVSlideState(tokens: full + [81, 82], head: 1)
    #expect(state.thin(from: everything, to: stay) == Array(5...21))
    #expect(state.tokens == [2, turn, user, 11, 12, turn, user, 51, turn, model, 61, 62, turn, user, 71, turn, model, 81, 82])
    #expect(state.dropped == 0 && state.position == full.count + 2)
    // The next prompt marks that reply; rendered the same way it continues the cache.
    let next = full + [81, 82] + [turn, user, 91] + [turn, model]
    let view = thinned(next, anchor: 32).map { next[$0] }
    let plan = try #require(KVSlidePlan.extend(state, full: view, capacity: 64, reply: 4, slides: true))
    #expect(plan.evict == 0 && Array(plan.feed) == [turn, user, 91, turn, model])
    // The plain rendering no longer continues it.
    #expect(KVSlidePlan.extend(state, full: next, capacity: 64, reply: 4, slides: true) == nil)
    // A cache that had slid six tokens: what stays of the part it no longer holds counts as dropped.
    var slid = KVSlideState(tokens: [2] + full[7...] + [81, 82], head: 1, dropped: 6, slid: 6)
    #expect(slid.thin(from: everything, to: stay) == Array(1...15))
    #expect(slid.tokens == [2, turn, user, 51, turn, model, 61, 62, turn, user, 71, turn, model, 81, 82])
    #expect(slid.dropped == 4)
    #expect(KVSlidePlan.extend(slid, full: view, capacity: 64, reply: 4, slides: true)?.feed.count == 5)
    // Thinned again later: a subset of what stayed, and nothing more to give up when nothing changed.
    let later = thinned(next, anchor: 39)
    #expect(Set(later).isSubset(of: Set(thinned(next, anchor: 32))) && later.contains(22) && !later.contains(25))
    var again = state
    #expect(again.thin(from: stay, to: stay) == [] && again == state)
    // A pinned summary turn and thinning do not mix.
    var pinned = KVSlideState(tokens: full, head: 1, pinned: 5..<10)
    #expect(pinned.thin(from: everything, to: stay) == nil)
}
