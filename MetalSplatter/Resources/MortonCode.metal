#include "ShaderCommon.h"

// Morton code (Z-order curve) computation for spatial ordering of Gaussian splats.
// Morton ordering clusters spatially nearby 3D points together in memory,
// improving GPU cache coherency during rendering.
//
// Reference: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

// Expands a 10-bit integer into 30 bits by inserting 2 zeros between each bit.
// Input:  ---- ---- ---- ---- ---- --98 7654 3210
// Output: --9- -8-- 7--6 --5- -4-- 3--2 --1- -0--
inline uint expandBits(uint v) {
    uint x = v & 0x3FFu;  // Mask to 10 bits
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8))  & 0x0300F00Fu;
    x = (x | (x << 4))  & 0x030C30C3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}

// Encodes a 3D position (each component 10-bit) into a 30-bit Morton code.
// Bit interleaving: ...zyx zyx zyx
inline uint encodeMorton3(uint x, uint y, uint z) {
    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

// Morton code output structure
struct MortonCodeOutput {
    uint code;
    uint originalIndex;
};

// Parameters for Morton code computation
struct MortonParameters {
    float3 boundsMin;
    float3 invBoundsSize;
    uint splatCount;
    uint padding;
};

// Compute Morton codes for all splats
// Output: array of (mortonCode, originalIndex) pairs for sorting
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeMortonCodes(
    constant Splat* splats [[buffer(0)]],
    device MortonCodeOutput* output [[buffer(1)]],
    constant MortonParameters& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= params.splatCount) return;

    float3 pos = float3(splats[index].position);

    // Normalize position to [0, 1] within bounds
    float3 normalized = (pos - params.boundsMin) * params.invBoundsSize;

    // Quantize to 10-bit integers (0-1023)
    uint qx = uint(clamp(normalized.x * 1023.0f, 0.0f, 1023.0f));
    uint qy = uint(clamp(normalized.y * 1023.0f, 0.0f, 1023.0f));
    uint qz = uint(clamp(normalized.z * 1023.0f, 0.0f, 1023.0f));

    output[index].code = encodeMorton3(qx, qy, qz);
    output[index].originalIndex = index;
}

// Compute bounds (min/max) for all splats using parallel reduction
// Phase 1: Compute threadgroup-local bounds
struct BoundsReduction {
    float3 minBounds;
    float padding1;
    float3 maxBounds;
    float padding2;
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeBoundsForMorton(
    constant Splat* splats [[buffer(0)]],
    device atomic_uint* atomicMinX [[buffer(1)]],
    device atomic_uint* atomicMinY [[buffer(2)]],
    device atomic_uint* atomicMinZ [[buffer(3)]],
    device atomic_uint* atomicMaxX [[buffer(4)]],
    device atomic_uint* atomicMaxY [[buffer(5)]],
    device atomic_uint* atomicMaxZ [[buffer(6)]],
    constant uint& splatCount [[buffer(7)]],
    uint index [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]]
) {
    if (index >= splatCount) return;

    float3 pos = float3(splats[index].position);

    // Use SIMD reduction within each SIMD group
    float3 localMin = pos;
    float3 localMax = pos;

    localMin.x = simd_min(localMin.x);
    localMin.y = simd_min(localMin.y);
    localMin.z = simd_min(localMin.z);

    localMax.x = simd_max(localMax.x);
    localMax.y = simd_max(localMax.y);
    localMax.z = simd_max(localMax.z);

    // Only first lane of each SIMD group updates atomics
    if (simdLaneId == 0) {
        // Use atomic_fetch_min/max with floats encoded as uints
        // Note: This works correctly for positive floats; for negative floats,
        // we'd need a different approach. Most splat scenes have positive bounds.
        atomic_fetch_min_explicit(atomicMinX, as_type<uint>(localMin.x), memory_order_relaxed);
        atomic_fetch_min_explicit(atomicMinY, as_type<uint>(localMin.y), memory_order_relaxed);
        atomic_fetch_min_explicit(atomicMinZ, as_type<uint>(localMin.z), memory_order_relaxed);
        atomic_fetch_max_explicit(atomicMaxX, as_type<uint>(localMax.x), memory_order_relaxed);
        atomic_fetch_max_explicit(atomicMaxY, as_type<uint>(localMax.y), memory_order_relaxed);
        atomic_fetch_max_explicit(atomicMaxZ, as_type<uint>(localMax.z), memory_order_relaxed);
    }
}

// Reorder splats based on sorted Morton indices
// Uses double-buffering: reads from source, writes to destination
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void reorderSplatsByMorton(
    constant Splat* sourceSplats [[buffer(0)]],
    device Splat* destSplats [[buffer(1)]],
    constant uint* sortedIndices [[buffer(2)]],
    constant uint& splatCount [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= splatCount) return;

    uint sourceIndex = sortedIndices[index];
    destSplats[index] = sourceSplats[sourceIndex];
}
