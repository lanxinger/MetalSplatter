#include <metal_stdlib>
using namespace metal;

#include "ShaderCommon.h"

// Interval metadata structure (must match IntervalManager.swift)
struct IntervalMetadata {
    uint sourceStart;
    uint sourceEnd;
    uint targetStart;
    uint lodLevel;
};

// Kernel to remap splat indices based on interval assignments.
// Takes visible indices from frustum culling and remaps them to contiguous output.
//
// This is Option B from the architecture: frustum culling runs first with global indices,
// then this kernel remaps to interval-local space.
//
// Input:
// - visibleIndices: Global splat indices that passed frustum culling
// - intervalTexture: 1D lookup texture mapping global index -> (intervalIdx, localOffset)
// - intervalMetadata: Array of interval properties
// - visibleCount: Number of visible splats
//
// Output:
// - remappedIndices: Indices in the output/render buffer after interval remapping
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void remapIndicesToIntervals(
    constant uint* visibleIndices [[buffer(0)]],
    texture1d<uint, access::read> intervalTexture [[texture(0)]],
    constant IntervalMetadata* intervalMetadata [[buffer(1)]],
    device uint* remappedIndices [[buffer(2)]],
    constant uint& visibleCount [[buffer(3)]],
    constant uint& intervalCount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= visibleCount) return;

    uint globalIndex = visibleIndices[gid];

    // Look up interval assignment from texture
    // RG32Uint: R = interval index, G = local offset within interval
    uint2 lookup = intervalTexture.read(globalIndex).rg;
    uint intervalIdx = lookup.x;
    uint localOffset = lookup.y;

    if (intervalIdx >= intervalCount) {
        // Index not in any active interval - skip (shouldn't happen with proper culling)
        remappedIndices[gid] = 0xFFFFFFFF;  // Invalid marker
        return;
    }

    // Compute remapped index: interval's target start + local offset
    IntervalMetadata interval = intervalMetadata[intervalIdx];
    uint remappedIndex = interval.targetStart + localOffset;

    remappedIndices[gid] = remappedIndex;
}

// Alternative kernel using buffer-based lookup (for devices without texture support)
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void remapIndicesToIntervalsBuffer(
    constant uint* visibleIndices [[buffer(0)]],
    constant uint2* intervalLookup [[buffer(1)]],  // Indexed by global splat index
    constant IntervalMetadata* intervalMetadata [[buffer(2)]],
    device uint* remappedIndices [[buffer(3)]],
    constant uint& visibleCount [[buffer(4)]],
    constant uint& intervalCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= visibleCount) return;

    uint globalIndex = visibleIndices[gid];

    // Look up interval assignment from buffer
    uint2 lookup = intervalLookup[globalIndex];
    uint intervalIdx = lookup.x;
    uint localOffset = lookup.y;

    if (intervalIdx >= intervalCount) {
        remappedIndices[gid] = 0xFFFFFFFF;
        return;
    }

    IntervalMetadata interval = intervalMetadata[intervalIdx];
    uint remappedIndex = interval.targetStart + localOffset;

    remappedIndices[gid] = remappedIndex;
}

// Kernel to filter and compact visible splats based on interval membership.
// Outputs only splats that are in active intervals.
//
// This is a compaction kernel that:
// 1. Reads global visible indices
// 2. Checks if each is in an active interval
// 3. Writes to output only if valid
// 4. Uses atomic counter for output position
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void compactActiveIntervalSplats(
    constant uint* visibleIndices [[buffer(0)]],
    texture1d<uint, access::read> intervalTexture [[texture(0)]],
    constant IntervalMetadata* intervalMetadata [[buffer(1)]],
    device int32_t* outputIndices [[buffer(2)]],  // Use int32_t to match sorted indices type
    device atomic_uint* outputCounter [[buffer(3)]],
    constant uint& visibleCount [[buffer(4)]],
    constant uint& intervalCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= visibleCount) return;

    uint globalIndex = visibleIndices[gid];

    // Look up interval assignment
    uint2 lookup = intervalTexture.read(globalIndex).rg;
    uint intervalIdx = lookup.x;

    // Check if in a valid active interval
    if (intervalIdx >= intervalCount) {
        return;  // Not in any active interval, skip
    }

    // Atomically claim output slot
    uint outputPos = atomic_fetch_add_explicit(outputCounter, 1, memory_order_relaxed);

    // Write the original global index (for rendering) at the compacted position
    outputIndices[outputPos] = int32_t(globalIndex);
}

// Kernel to build the interval lookup texture from interval metadata.
// Run once when intervals change, not every frame.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void buildIntervalLookupTexture(
    constant IntervalMetadata* intervals [[buffer(0)]],
    constant uint& intervalCount [[buffer(1)]],
    texture1d<uint, access::write> lookupTexture [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one splat position
    // Need to find which interval (if any) contains this global index

    uint globalIndex = gid;

    // Linear search through intervals (could be optimized with binary search for many intervals)
    for (uint i = 0; i < intervalCount; i++) {
        IntervalMetadata interval = intervals[i];
        if (globalIndex >= interval.sourceStart && globalIndex < interval.sourceEnd) {
            uint localOffset = globalIndex - interval.sourceStart;
            lookupTexture.write(uint4(i, localOffset, 0, 0), globalIndex);
            return;
        }
    }

    // Not in any interval - write invalid marker
    lookupTexture.write(uint4(0xFFFFFFFF, 0, 0, 0), globalIndex);
}
