#include "ShaderCommon.h"

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistances(uint index [[thread_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 constant Splat* splatArray [[ buffer(0) ]],
                                 device float* distances [[ buffer(1) ]],
                                 constant float3& cameraPosition [[ buffer(2) ]],
                                 constant float3& cameraForward [[ buffer(3) ]],
                                 constant bool& sortByDistance [[ buffer(4) ]],
                                 constant uint& splatCount [[ buffer(5) ]]) {
    
    // Threadgroup memory for caching splat data (32 splats per cache)
    threadgroup Splat cachedSplats[32];
    threadgroup float3 cachedPositions[32];
    
    // Batch process splats using threadgroup memory for better coalescing
    uint batchStartIndex = tgid * 32;
    uint localBatchIndex = tid % 32;
    uint batchIndex = batchStartIndex + localBatchIndex;
    
    // Cooperative loading of splat batch into threadgroup memory
    if (batchIndex < splatCount && localBatchIndex < 32) {
        cachedSplats[localBatchIndex] = splatArray[batchIndex];
        cachedPositions[localBatchIndex] = float3(cachedSplats[localBatchIndex].position);
    }
    
    // Synchronize threadgroup before processing
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process multiple splats per thread from cached data
    uint splatStride = 256 / 32; // 8 splats per thread
    for (uint i = 0; i < splatStride && i < 32; i++) {
        uint processIndex = index + i * 256;
        uint cacheIndex = (processIndex - batchStartIndex) % 32;
        
        if (processIndex >= splatCount) break;
        
        // Use cached position data for better memory access patterns
        float3 splatPos;
        if (processIndex >= batchStartIndex && processIndex < batchStartIndex + 32) {
            splatPos = cachedPositions[cacheIndex];
        } else {
            splatPos = float3(splatArray[processIndex].position);
        }
        
        // Optimized distance calculation with SIMD operations
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            // Calculate length squared for distance sorting using SIMD
            float distanceSquared = dot(delta, delta);
            distances[processIndex] = distanceSquared;
        } else {
            // Project onto forward vector for depth sorting
            distances[processIndex] = dot(splatPos, cameraForward);
        }
    }
}

// Advanced compute distances kernel using simdgroup operations for better GPU utilization
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistancesSimdOptimized(uint index [[thread_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint simd_lane_id [[thread_index_in_simdgroup]],
                                              uint simd_group_id [[simdgroup_index_in_threadgroup]],
                                              constant Splat* splatArray [[ buffer(0) ]],
                                              device float* distances [[ buffer(1) ]],
                                              constant float3& cameraPosition [[ buffer(2) ]],
                                              constant float3& cameraForward [[ buffer(3) ]],
                                              constant bool& sortByDistance [[ buffer(4) ]],
                                              constant uint& splatCount [[ buffer(5) ]]) {
    
    // Threadgroup memory optimized for simdgroup access patterns
    threadgroup float3 positions[256];
    threadgroup float localDistances[256];
    
    // Load positions cooperatively within each simdgroup
    uint globalIndex = index;
    
    if (globalIndex < splatCount) {
        Splat splat = splatArray[globalIndex];
        positions[tid] = float3(splat.position);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process using simdgroup operations for vectorized computation
    if (globalIndex < splatCount) {
        float3 splatPos = positions[tid];
        
        // Vectorized distance calculations using simdgroup operations
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            float distanceSquared = dot(delta, delta);
            
            // Store computed distance
            localDistances[tid] = distanceSquared;
        } else {
            float depth = dot(splatPos, cameraForward);
            
            localDistances[tid] = depth;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Coalesce writes to global memory
    if (globalIndex < splatCount) {
        distances[globalIndex] = localDistances[tid];
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