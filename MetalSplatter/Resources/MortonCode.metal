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

// Compute bounds (min/max) for all splats using hierarchical parallel reduction
// Optimized: SIMD reduction -> Threadgroup reduction -> Single global atomic update
// Uses CAS-based float atomics that correctly handle negative values

// Atomic float min using compare-and-swap (handles negative floats correctly)
inline void atomicMinFloatMorton(device atomic_uint* addr, float val) {
    if (isnan(val)) return;
    uint newVal = as_type<uint>(val);
    uint prevVal = atomic_load_explicit(addr, memory_order_relaxed);
    while (val < as_type<float>(prevVal)) {
        if (atomic_compare_exchange_weak_explicit(addr, &prevVal, newVal,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            break;
        }
    }
}

// Atomic float max using compare-and-swap (handles negative floats correctly)
inline void atomicMaxFloatMorton(device atomic_uint* addr, float val) {
    if (isnan(val)) return;
    uint newVal = as_type<uint>(val);
    uint prevVal = atomic_load_explicit(addr, memory_order_relaxed);
    while (val > as_type<float>(prevVal)) {
        if (atomic_compare_exchange_weak_explicit(addr, &prevVal, newVal,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            break;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeBoundsForMorton(
    constant Splat* splats [[buffer(0)]],
    device atomic_uint* bounds [[buffer(1)]],  // Consolidated: [minX, minY, minZ, maxX, maxY, maxZ]
    constant uint& splatCount [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint localIndex [[thread_position_in_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup storage for SIMD-group results (8 SIMD groups of 32 threads = 256 threads)
    threadgroup float3 tgMin[8];
    threadgroup float3 tgMax[8];

    // Initialize thread-local bounds
    float3 threadMin = float3(INFINITY);
    float3 threadMax = float3(-INFINITY);

    if (index < splatCount) {
        float3 pos = float3(splats[index].position);
        threadMin = pos;
        threadMax = pos;
    }

    // Phase 1: SIMD-group reduction (32 threads -> 1 value)
    float3 simdMin = float3(
        simd_min(threadMin.x),
        simd_min(threadMin.y),
        simd_min(threadMin.z)
    );
    float3 simdMax = float3(
        simd_max(threadMax.x),
        simd_max(threadMax.y),
        simd_max(threadMax.z)
    );

    // First lane of each SIMD group writes to threadgroup memory
    if (simdLaneId == 0) {
        tgMin[simdGroupId] = simdMin;
        tgMax[simdGroupId] = simdMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Threadgroup reduction (8 SIMD groups -> 1 value)
    // All lanes in SIMD group 0 must participate uniformly in simd_min/simd_max
    // to avoid undefined behavior from divergent control flow.
    // Lanes >= 8 use neutral values that don't affect the reduction result.
    if (simdGroupId == 0) {
        // Lanes 0-7 load from threadgroup memory, lanes 8-31 use neutral values
        float3 tMin = (simdLaneId < 8) ? tgMin[simdLaneId] : float3(INFINITY);
        float3 tMax = (simdLaneId < 8) ? tgMax[simdLaneId] : float3(-INFINITY);

        // All 32 lanes participate uniformly in SIMD reduction
        tMin.x = simd_min(tMin.x);
        tMin.y = simd_min(tMin.y);
        tMin.z = simd_min(tMin.z);
        tMax.x = simd_max(tMax.x);
        tMax.y = simd_max(tMax.y);
        tMax.z = simd_max(tMax.z);

        // Only thread 0 updates global atomics (one update per threadgroup)
        if (simdLaneId == 0) {
            atomicMinFloatMorton(&bounds[0], tMin.x);
            atomicMinFloatMorton(&bounds[1], tMin.y);
            atomicMinFloatMorton(&bounds[2], tMin.z);
            atomicMaxFloatMorton(&bounds[3], tMax.x);
            atomicMaxFloatMorton(&bounds[4], tMax.y);
            atomicMaxFloatMorton(&bounds[5], tMax.z);
        }
    }
}

// Reset bounds buffer before computation
[[kernel]]
kernel void resetBoundsForMorton(device atomic_uint* bounds [[buffer(0)]]) {
    uint posInf = as_type<uint>(INFINITY);
    uint negInf = as_type<uint>(-INFINITY);

    atomic_store_explicit(&bounds[0], posInf, memory_order_relaxed);  // minX
    atomic_store_explicit(&bounds[1], posInf, memory_order_relaxed);  // minY
    atomic_store_explicit(&bounds[2], posInf, memory_order_relaxed);  // minZ
    atomic_store_explicit(&bounds[3], negInf, memory_order_relaxed);  // maxX
    atomic_store_explicit(&bounds[4], negInf, memory_order_relaxed);  // maxY
    atomic_store_explicit(&bounds[5], negInf, memory_order_relaxed);  // maxZ
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
