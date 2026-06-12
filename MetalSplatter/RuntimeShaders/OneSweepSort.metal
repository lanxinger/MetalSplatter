#include <metal_stdlib>

using namespace metal;

// =============================================================================
// OneSweep radix sort (MSL 4.1)
// =============================================================================
//
// Single-dispatch-per-digit LSD radix sort using "chained scan with decoupled
// lookback" (Merrill & Garland's OneSweep). Replaces the legacy 6-dispatch
// per-pass scatter pipeline in Metal4AdvancedAtomics.metal, whose stable
// scatter ranks each element with an O(threadgroup size) serial loop over
// device memory and computes inter-threadgroup offsets with a serial
// O(threadgroup count) loop on 256 threads.
//
// Structure per sort (encoded by Metal4Sorter when OneSweep is available):
//   1. onesweep_global_histogram — all 4 digit histograms in ONE pass
//   2. onesweep_scan_histograms  — exclusive scan of each 256-bin histogram
//   3. 4 x (onesweep_reset_status + onesweep_digit_pass)
//
// The digit pass keeps elements stable via SIMD-group multi-split ranking
// (simd_ballot peer matching) and propagates inter-tile offsets with the
// decoupled-lookback protocol: each tile publishes its per-digit aggregate,
// then resolves its exclusive prefix by walking predecessor tiles.
//
// MSL 4.1: status words are published with memory_order_release and consumed
// with memory_order_acquire (new in Metal 4.1 for atomic operations), which is
// the correct formulation for cross-threadgroup handoff.
//
// Scheduling assumption: tiles are claimed through an atomic ticket, so tile
// order matches threadgroup scheduling order. A tile spinning in lookback only
// waits on tiles that began executing before it (the standard occupancy-bound
// execution assumption of decoupled lookback; same as CUDA's CUB/OneSweep).
//
// =============================================================================

#if __METAL_VERSION__ >= 410

// Matches advanced_atomics::SortingKey layout (Metal4AdvancedAtomics.metal)
struct OneSweepKey {
    float depth;          // sortable uint stored as float bits
    uint original_index;
};

constant constexpr uint ONESWEEP_RADIX = 256;          // 8-bit digits
constant constexpr uint ONESWEEP_PASSES = 4;
constant constexpr uint ONESWEEP_TG_SIZE = 256;
constant constexpr uint ONESWEEP_KEYS_PER_THREAD = 4;
constant constexpr uint ONESWEEP_TILE = ONESWEEP_TG_SIZE * ONESWEEP_KEYS_PER_THREAD;  // 1024
constant constexpr uint ONESWEEP_SIMD_GROUPS = ONESWEEP_TG_SIZE / 32;                 // 8 (Apple GPUs)

// Lookback status word: 2 flag bits + 30 value bits (supports up to 2^30 keys)
constant constexpr uint ONESWEEP_FLAG_AGGREGATE = 1u << 30;
constant constexpr uint ONESWEEP_FLAG_PREFIX    = 2u << 30;
constant constexpr uint ONESWEEP_FLAG_MASK      = 3u << 30;
constant constexpr uint ONESWEEP_VALUE_MASK     = ~ONESWEEP_FLAG_MASK;

// -----------------------------------------------------------------------------
// Pass 1: all-digit global histogram in a single sweep over the keys
// globalHist layout: [pass][digit] = [4][256] uints
// -----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(ONESWEEP_TG_SIZE)]]
void onesweep_global_histogram(
    constant OneSweepKey *keys [[buffer(0)]],
    device atomic_uint *globalHist [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint gridSize [[threads_per_grid]]
) {
    threadgroup atomic_uint localHist[ONESWEEP_PASSES * ONESWEEP_RADIX];

    for (uint i = tid; i < ONESWEEP_PASSES * ONESWEEP_RADIX; i += ONESWEEP_TG_SIZE) {
        atomic_store_explicit(&localHist[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = gid; i < count; i += gridSize) {
        uint bits = as_type<uint>(keys[i].depth);
        atomic_fetch_add_explicit(&localHist[0 * ONESWEEP_RADIX + ( bits         & 0xFF)], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&localHist[1 * ONESWEEP_RADIX + ((bits >> 8)   & 0xFF)], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&localHist[2 * ONESWEEP_RADIX + ((bits >> 16)  & 0xFF)], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&localHist[3 * ONESWEEP_RADIX + ((bits >> 24)  & 0xFF)], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < ONESWEEP_PASSES * ONESWEEP_RADIX; i += ONESWEEP_TG_SIZE) {
        uint v = atomic_load_explicit(&localHist[i], memory_order_relaxed);
        if (v != 0) {
            atomic_fetch_add_explicit(&globalHist[i], v, memory_order_relaxed);
        }
    }
}

// -----------------------------------------------------------------------------
// Pass 2: in-place exclusive scan of each pass's 256-bin histogram
// One threadgroup of 256 threads; 8 simdgroups of 32 lanes each.
// -----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(ONESWEEP_TG_SIZE)]]
void onesweep_scan_histograms(
    device uint *globalHist [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint laneID [[thread_index_in_simdgroup]],
    uint simdGroupID [[simdgroup_index_in_threadgroup]],
    uint simdGroupCount [[simdgroups_per_threadgroup]],
    uint simdSize [[threads_per_simdgroup]]
) {
    threadgroup uint partials[ONESWEEP_SIMD_GROUPS];

    for (uint pass = 0; pass < ONESWEEP_PASSES; pass++) {
        uint index = pass * ONESWEEP_RADIX + tid;
        uint value = globalHist[index];
        uint lanePrefix = simd_prefix_exclusive_sum(value);

        if (laneID == simdSize - 1) {
            partials[simdGroupID] = lanePrefix + value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdGroupID == 0) {
            uint groupSum = (laneID < simdGroupCount) ? partials[laneID] : 0;
            uint groupPrefix = simd_prefix_exclusive_sum(groupSum);
            if (laneID < simdGroupCount) {
                partials[laneID] = groupPrefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        globalHist[index] = partials[simdGroupID] + lanePrefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// -----------------------------------------------------------------------------
// Per-pass reset: clear lookback status words and the tile ticket counter
// -----------------------------------------------------------------------------
[[kernel]]
void onesweep_reset_status(
    device uint *status [[buffer(0)]],
    device uint *ticket [[buffer(1)]],
    constant uint &statusCount [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < statusCount) {
        status[gid] = 0;
    }
    if (gid == 0) {
        ticket[0] = 0;
    }
}

// -----------------------------------------------------------------------------
// Pass 3 (x4): stable digit binning with decoupled lookback
// -----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(ONESWEEP_TG_SIZE)]]
void onesweep_digit_pass(
    constant OneSweepKey *inputKeys [[buffer(0)]],
    device OneSweepKey *outputKeys [[buffer(1)]],
    constant uint *passOffsets [[buffer(2)]],   // scanned globalHist: [pass][digit]
    device atomic_uint *status [[buffer(3)]],   // [numTiles][256] lookback words
    device atomic_uint *ticket [[buffer(4)]],   // tile ticket counter
    constant uint &count [[buffer(5)]],
    constant uint &byteIndex [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint laneID [[thread_index_in_simdgroup]],
    uint simdGroupID [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint sharedTileId;
    threadgroup uint tgHist[ONESWEEP_RADIX];                              // running tile digit counts
    threadgroup uint sgCounts[ONESWEEP_SIMD_GROUPS * ONESWEEP_RADIX];     // per-simdgroup digit counts (per round)
    threadgroup uint tileExclusive[ONESWEEP_RADIX];                       // resolved global base per digit

    // Claim a tile through the ticket so tile order matches scheduling order
    // (keeps the lookback chain free of waits on unscheduled threadgroups).
    if (tid == 0) {
        sharedTileId = atomic_fetch_add_explicit(ticket, 1, memory_order_relaxed);
    }
    for (uint i = tid; i < ONESWEEP_RADIX; i += ONESWEEP_TG_SIZE) {
        tgHist[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tileId = sharedTileId;
    uint tileBase = tileId * ONESWEEP_TILE;
    uint shift = byteIndex * 8;

    OneSweepKey myKeys[ONESWEEP_KEYS_PER_THREAD];
    uint myRanks[ONESWEEP_KEYS_PER_THREAD];
    bool myValid[ONESWEEP_KEYS_PER_THREAD];

    // Rank the tile's keys round by round; index order (round-major, then
    // simdgroup, then lane) matches input order, which keeps the pass stable.
    for (uint r = 0; r < ONESWEEP_KEYS_PER_THREAD; r++) {
        for (uint i = tid; i < ONESWEEP_SIMD_GROUPS * ONESWEEP_RADIX; i += ONESWEEP_TG_SIZE) {
            sgCounts[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint index = tileBase + r * ONESWEEP_TG_SIZE + tid;
        bool valid = index < count;
        myValid[r] = valid;

        uint digit = 0;
        if (valid) {
            myKeys[r] = inputKeys[index];
            digit = (as_type<uint>(myKeys[r].depth) >> shift) & 0xFF;
        }

        // SIMD-group multi-split: find the lanes holding the same digit
        ulong validMask = static_cast<simd_vote::vote_t>(simd_ballot(valid));
        ulong peers = validMask;
        for (uint bit = 0; bit < 8; bit++) {
            bool isSet = (digit >> bit) & 1;
            ulong ballot = static_cast<simd_vote::vote_t>(simd_ballot(valid && isSet));
            peers &= isSet ? ballot : ~ballot;
        }

        ulong lanesBefore = (1ul << ulong(laneID)) - 1;
        uint rankInSimd = uint(popcount(peers & lanesBefore));

        // First peer lane publishes the simdgroup's count for this digit
        if (valid && rankInSimd == 0) {
            sgCounts[simdGroupID * ONESWEEP_RADIX + digit] = uint(popcount(peers));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint precedingCount = 0;
            for (uint sg = 0; sg < simdGroupID; sg++) {
                precedingCount += sgCounts[sg * ONESWEEP_RADIX + digit];
            }
            myRanks[r] = tgHist[digit] + precedingCount + rankInSimd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fold this round's counts into the running tile histogram
        for (uint d = tid; d < ONESWEEP_RADIX; d += ONESWEEP_TG_SIZE) {
            uint roundTotal = 0;
            for (uint sg = 0; sg < ONESWEEP_SIMD_GROUPS; sg++) {
                roundTotal += sgCounts[sg * ONESWEEP_RADIX + d];
            }
            tgHist[d] += roundTotal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Publish this tile's per-digit aggregate (tile 0 publishes its prefix
    // directly), then resolve the exclusive prefix via decoupled lookback.
    // One thread per digit; the status word carries both flag and value, and
    // release/acquire ordering makes the handoff between tiles well-defined.
    {
        uint digit = tid;
        uint aggregate = tgHist[digit];
        uint packed = (tileId == 0)
            ? (ONESWEEP_FLAG_PREFIX | aggregate)
            : (ONESWEEP_FLAG_AGGREGATE | aggregate);
        atomic_store_explicit(&status[tileId * ONESWEEP_RADIX + digit], packed,
                              memory_order_release, mem_flags::mem_device);

        uint exclusive = 0;
        if (tileId > 0) {
            uint look = tileId - 1;
            while (true) {
                uint word = atomic_load_explicit(&status[look * ONESWEEP_RADIX + digit],
                                                 memory_order_acquire, mem_flags::mem_device);
                uint flag = word & ONESWEEP_FLAG_MASK;
                if (flag == 0) {
                    continue;  // predecessor not published yet; spin
                }
                exclusive += word & ONESWEEP_VALUE_MASK;
                if (flag == ONESWEEP_FLAG_PREFIX) {
                    break;
                }
                look--;
            }
            atomic_store_explicit(&status[tileId * ONESWEEP_RADIX + digit],
                                  ONESWEEP_FLAG_PREFIX | (exclusive + aggregate),
                                  memory_order_release, mem_flags::mem_device);
        }

        tileExclusive[digit] = passOffsets[byteIndex * ONESWEEP_RADIX + digit] + exclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter the tile's keys to their final positions for this pass
    for (uint r = 0; r < ONESWEEP_KEYS_PER_THREAD; r++) {
        if (!myValid[r]) {
            continue;
        }
        uint digit = (as_type<uint>(myKeys[r].depth) >> shift) & 0xFF;
        outputKeys[tileExclusive[digit] + myRanks[r]] = myKeys[r];
    }
}

#endif  // __METAL_VERSION__ >= 410
