#include "ShaderCommon.h"

// =============================================================================
// Simplified Distance Computation Kernel
// =============================================================================
// Direct global memory access with coalesced reads/writes.
// Modern Apple GPUs have excellent L2 cache, making threadgroup caching
// counterproductive for simple linear access patterns like this.
//
// Benchmarks show this simple version often outperforms complex caching
// due to better occupancy and reduced register pressure.

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistances(uint index [[thread_position_in_grid]],
                                  constant Splat* splatArray [[buffer(0)]],
                                  device float* distances [[buffer(1)]],
                                  constant float3& cameraPosition [[buffer(2)]],
                                  constant float3& cameraForward [[buffer(3)]],
                                  constant bool& sortByDistance [[buffer(4)]],
                                  constant uint& splatCount [[buffer(5)]]) {
    if (index >= splatCount) return;

    // Direct coalesced read - GPU L2 cache handles this efficiently
    float3 splatPos = float3(splatArray[index].position);
    float3 delta = splatPos - cameraPosition;

    // Branchless selection between distance modes
    // Both values are cheap to compute; select avoids divergence
    float distanceSquared = dot(delta, delta);
    float depth = dot(delta, cameraForward);

    distances[index] = sortByDistance ? distanceSquared : depth;
}

// Legacy version with threadgroup caching (kept for A/B testing)
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistancesWithCaching(uint index [[thread_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             constant Splat* splatArray [[buffer(0)]],
                                             device float* distances [[buffer(1)]],
                                             constant float3& cameraPosition [[buffer(2)]],
                                             constant float3& cameraForward [[buffer(3)]],
                                             constant bool& sortByDistance [[buffer(4)]],
                                             constant uint& splatCount [[buffer(5)]]) {
    // Threadgroup memory for position caching
    threadgroup float3 positions[256];

    if (index < splatCount) {
        positions[tid] = float3(splatArray[index].position);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (index < splatCount) {
        float3 splatPos = positions[tid];
        float3 delta = splatPos - cameraPosition;

        float distanceSquared = dot(delta, delta);
        float depth = dot(delta, cameraForward);

        distances[index] = sortByDistance ? distanceSquared : depth;
    }
}

// =============================================================================
// SIMD-Group Parallel Bounds Computation
// =============================================================================
// Computes AABB bounds using SIMD-group reduction for 32x fewer atomic operations
// Each SIMD-group (32 threads) reduces to a single value before atomic update

// Helper: Atomic float min using compare-and-swap
// Metal doesn't have native atomic_fetch_min for floats, so we implement it with CAS
inline void atomicMinFloat(device atomic_uint* addr, float val) {
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

// Helper: Atomic float max using compare-and-swap
inline void atomicMaxFloat(device atomic_uint* addr, float val) {
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
kernel void computeBoundsParallel(uint index [[thread_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint simd_lane_id [[thread_index_in_simdgroup]],
                                  uint simd_group_id [[simdgroup_index_in_threadgroup]],
                                  constant Splat* splatArray [[buffer(0)]],
                                  constant uint& splatCount [[buffer(1)]],
                                  device atomic_uint* boundsMin [[buffer(2)]],  // float3 as 3 atomic_uint (for CAS)
                                  device atomic_uint* boundsMax [[buffer(3)]]) {
    
    // Initialize per-thread min/max with extreme values
    float3 threadMin = float3(INFINITY);
    float3 threadMax = float3(-INFINITY);
    
    // Each thread processes one splat
    if (index < splatCount) {
        float3 pos = float3(splatArray[index].position);
        threadMin = pos;
        threadMax = pos;
    }
    
    // SIMD-group reduction: 32 threads → 1 value
    // This is the key optimization - uses hardware SIMD lanes for parallel reduction
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
    
    // Only the first lane in each SIMD-group does the atomic update
    // This reduces atomic contention by 32x compared to per-thread atomics
    if (simd_lane_id == 0) {
        // Atomic min for bounds minimum (using CAS)
        atomicMinFloat(&boundsMin[0], simdMin.x);
        atomicMinFloat(&boundsMin[1], simdMin.y);
        atomicMinFloat(&boundsMin[2], simdMin.z);
        
        // Atomic max for bounds maximum (using CAS)
        atomicMaxFloat(&boundsMax[0], simdMax.x);
        atomicMaxFloat(&boundsMax[1], simdMax.y);
        atomicMaxFloat(&boundsMax[2], simdMax.z);
    }
}

// Kernel to reset bounds atomics before computation
[[kernel]]
kernel void resetBoundsAtomics(device atomic_uint* boundsMin [[buffer(0)]],
                               device atomic_uint* boundsMax [[buffer(1)]]) {
    // Reset min to +infinity, max to -infinity
    // Store as uint bit pattern for CAS-based atomic operations
    uint posInf = as_type<uint>(INFINITY);
    uint negInf = as_type<uint>(-INFINITY);
    
    atomic_store_explicit(&boundsMin[0], posInf, memory_order_relaxed);
    atomic_store_explicit(&boundsMin[1], posInf, memory_order_relaxed);
    atomic_store_explicit(&boundsMin[2], posInf, memory_order_relaxed);
    
    atomic_store_explicit(&boundsMax[0], negInf, memory_order_relaxed);
    atomic_store_explicit(&boundsMax[1], negInf, memory_order_relaxed);
    atomic_store_explicit(&boundsMax[2], negInf, memory_order_relaxed);
}