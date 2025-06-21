#include "ShaderCommon.h"

struct FrustumPlane {
    float3 normal;
    float distance;
};

struct FrustumCullData {
    FrustumPlane planes[6]; // left, right, bottom, top, near, far
    float3 cameraPosition;
    float maxDistance; // Maximum culling distance
};

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void frustumCullSplats(uint index [[thread_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint tgid [[threadgroup_position_in_grid]],
                             constant Splat* inputSplats [[ buffer(0) ]],
                             device uint* visibleIndices [[ buffer(1) ]],
                             device atomic_uint* visibleCount [[ buffer(2) ]],
                             constant FrustumCullData& cullData [[ buffer(3) ]],
                             constant uint& splatCount [[ buffer(4) ]]) {
    
    // Threadgroup memory for cooperative loading and processing
    threadgroup Splat cachedSplats[64];
    threadgroup float3 cachedPositions[64];
    threadgroup uint localVisibleIndices[64];
    threadgroup atomic_uint localVisibleCount;
    
    // Initialize threadgroup counters
    if (tid == 0) {
        atomic_store_explicit(&localVisibleCount, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooperative loading into threadgroup memory
    uint batchSize = 64;
    uint batchStart = tgid * batchSize;
    uint localIndex = tid % batchSize;
    uint globalIndex = batchStart + localIndex;
    
    // Load splat data into threadgroup memory
    if (globalIndex < splatCount && localIndex < batchSize) {
        cachedSplats[localIndex] = inputSplats[globalIndex];
        cachedPositions[localIndex] = float3(cachedSplats[localIndex].position);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process splats from threadgroup memory
    if (index >= splatCount) return;
    
    float3 splatPos;
    if (index >= batchStart && index < batchStart + batchSize) {
        // Use cached data for better memory access
        splatPos = cachedPositions[index - batchStart];
    } else {
        // Fall back to direct memory access for out-of-batch splats
        splatPos = float3(inputSplats[index].position);
    }
    
    // Optimized distance culling with pre-computed squared distance
    float3 toCam = splatPos - cullData.cameraPosition;
    float distanceSquared = dot(toCam, toCam);
    float maxDistanceSquared = cullData.maxDistance * cullData.maxDistance;
    
    if (distanceSquared > maxDistanceSquared) {
        return;
    }
    
    // Frustum culling with early exit optimization
    // Approximate splat radius for culling (conservative estimate)
    float splatRadius = 2.0; // Could be calculated from covariance
    
    // Unrolled frustum plane tests for better performance
    bool visible = true;
    
    // Test planes in order of likelihood to fail (near plane first)
    if (visible) {
        FrustumPlane plane = cullData.planes[4]; // near plane
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    if (visible) {
        FrustumPlane plane = cullData.planes[5]; // far plane  
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    if (visible) {
        FrustumPlane plane = cullData.planes[0]; // left plane
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    if (visible) {
        FrustumPlane plane = cullData.planes[1]; // right plane
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    if (visible) {
        FrustumPlane plane = cullData.planes[2]; // bottom plane
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    if (visible) {
        FrustumPlane plane = cullData.planes[3]; // top plane
        visible = (dot(plane.normal, splatPos) + plane.distance) >= -splatRadius;
    }
    
    // Store visible indices locally first, then flush to global memory
    if (visible) {
        uint localIdx = atomic_fetch_add_explicit(&localVisibleCount, 1, memory_order_relaxed);
        if (localIdx < batchSize) {
            localVisibleIndices[localIdx] = index;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooperative flush of visible indices to global memory
    if (tid == 0) {
        uint localCount = atomic_load_explicit(&localVisibleCount, memory_order_relaxed);
        uint globalStartIdx = atomic_fetch_add_explicit(visibleCount, localCount, memory_order_relaxed);
        
        for (uint i = 0; i < min(localCount, batchSize); i++) {
            visibleIndices[globalStartIdx + i] = localVisibleIndices[i];
        }
    }
} 