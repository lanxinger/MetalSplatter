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