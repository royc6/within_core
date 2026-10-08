import Foundation

/// Which position each slot of a full-attention cache holds (KV-SLIDE, REPORT.md Section 7.3).
/// The rotary tables and the masks are computed on the host, so positions keep counting past the
/// slot count: a slot is reused once the position it held is evicted. Until something is evicted
/// slot s holds position s, and the mask equals `Gemma4RuntimeConfiguration.mask(global: true)`.
public struct Gemma4SlotTable: Sendable, Equatable {
    /// Per slot: the position it holds, or -1 when free.
    public private(set) var positions: [Int32]
    /// Slots that hold a position.
    public private(set) var held = 0
    private var cursor = 0  // where the search for a free slot resumes

    /// An empty table of `capacity` slots.
    public init(capacity: Int) { positions = Array(repeating: -1, count: capacity) }

    /// A table read back from a snapshot: each position at most once.
    public init(positions: [Int32]) throws {
        let live = positions.filter { $0 >= 0 }
        guard !positions.isEmpty, positions.allSatisfy({ $0 >= -1 }), Set(live).count == live.count else {
            throw Gemma4RuntimeError.invalidPackage
        }
        self.positions = positions
        held = live.count
        // New slots are searched after the one written last, as before the snapshot.
        if let latest = positions.indices.max(by: { positions[$0] < positions[$1] }), positions[latest] >= 0 {
            cursor = (latest + 1) % positions.count
        }
    }

    public var capacity: Int { positions.count }
    public var free: Int { capacity - held }

    /// Frees every slot.
    public mutating func reset() {
        for slot in positions.indices { positions[slot] = -1 }
        held = 0; cursor = 0
    }

    /// The slots of `count` new positions from `position` on, in order.
    public mutating func allocate(position: Int, count: Int) throws -> [Int] {
        guard position >= 0, count >= 0, count <= free, position + count <= Int(Int32.max) else {
            throw Gemma4RuntimeError.contextExceeded
        }
        var slots: [Int] = []
        slots.reserveCapacity(count)
        while slots.count < count {
            if positions[cursor] < 0 {
                positions[cursor] = Int32(position + slots.count)
                slots.append(cursor)
            }
            cursor = (cursor + 1) % capacity
        }
        held += count
        return slots
    }

    /// Frees the slots of the tokens at `indices`, counted from the oldest position held: a caller
    /// knows the tokens the cache holds in order, not which positions a gap has left unused.
    public mutating func evict(_ indices: some Sequence<Int>) {
        let order = positions.indices.filter { positions[$0] >= 0 }.sorted { positions[$0] < positions[$1] }
        for index in indices where positions[order[index]] >= 0 {
            positions[order[index]] = -1
            held -= 1
        }
    }

    /// Frees slots for `count` new positions out of `pending`, the tokens the caller gives up
    /// (counted like `evict`'s, oldest first): nothing while enough slots are free, otherwise at
    /// least `block` at a time, so a long prompt keeps seeing the old context until each of its
    /// calls needs the room. What is left of `pending` moves down to where it started.
    public mutating func makeRoom(for count: Int, from pending: inout Range<Int>, block: Int) {
        guard count > free, !pending.isEmpty else { return }
        let freed = min(max(count - free, block), pending.count)
        evict(pending.lowerBound..<pending.lowerBound + freed)
        pending = pending.lowerBound..<pending.upperBound - freed
    }

    /// The positions of `range` that no slot holds. Every position below the next one was written
    /// once, so these were evicted, and the sliding-attention rings hide them too.
    public func missing(_ range: Range<Int>) -> Set<Int> {
        var missing = Set(range)
        for position in positions where position >= range.lowerBound { missing.remove(Int(position)) }
        return missing
    }

    /// The additive mask of one call, [length, capacity + length]: every held slot is visible to
    /// every query (all of them were written before this call), then this call's fresh keys, causal.
    public func mask(length: Int) -> [Float16] {
        let width = capacity + length
        var values = [Float16](repeating: -10_000, count: length * width)
        // The slot part is the same for every query: build it once and copy it (32,000 slots
        // times 128 queries is 8 MB a call), then the causal fresh keys row by row.
        values.withUnsafeMutableBufferPointer { values in
            for slot in 0..<capacity where positions[slot] >= 0 { values[slot] = 0 }
            for query in 1..<max(1, length) {
                _ = UnsafeMutableBufferPointer(rebasing: values[(query * width)..<(query * width + capacity)])
                    .update(fromContentsOf: UnsafeBufferPointer(rebasing: values[0..<capacity]))
            }
            for query in 0..<length {
                UnsafeMutableBufferPointer(rebasing: values[(query * width + capacity)...(query * width + capacity + query)])
                    .update(repeating: 0)
            }
        }
        return values
    }
}
