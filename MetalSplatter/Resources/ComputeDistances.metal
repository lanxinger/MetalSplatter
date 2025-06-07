#include "ShaderCommon.h"

// Optimized compute shader with SIMD vectorization and early culling
kernel void computeSplatDistances(uint index [[thread_position_in_grid]],
                                 constant Splat* splatArray [[ buffer(0) ]],
                                 device float* distances [[ buffer(1) ]],
                                 constant float3& cameraPosition [[ buffer(2) ]],
                                 constant float3& cameraForward [[ buffer(3) ]],
                                 constant bool& sortByDistance [[ buffer(4) ]],
                                 constant uint& splatCount [[ buffer(5) ]],
                                 constant float& maxCullDistance [[ buffer(6) ]],
                                 uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                                 uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
    
    if (index >= splatCount) return;
    
    // Load splat data once
    constant Splat& splat = splatArray[index];
    float3 splatPos = float3(splat.position);
    
    float distance;
    
    if (sortByDistance) {
        // Vectorized distance calculation
        float3 delta = splatPos - cameraPosition;
        distance = dot(delta, delta); // length squared
        
        // Early exit for very distant splats - mark as furthest distance
        if (distance > maxCullDistance * maxCullDistance) {
            distance = FLT_MAX;
        }
    } else {
        // Forward projection distance
        distance = dot(splatPos, cameraForward);
        
        // Early exit culling for splats behind camera or too far
        if (distance < 0.0 || distance > maxCullDistance) {
            distance = (distance < 0.0) ? -FLT_MAX : FLT_MAX;
        }
    }
    
    distances[index] = distance;
}

// Vectorized version that processes multiple splats per thread
kernel void computeSplatDistancesVectorized(uint index [[thread_position_in_grid]],
                                           constant Splat* splatArray [[ buffer(0) ]],
                                           device float* distances [[ buffer(1) ]],
                                           constant float3& cameraPosition [[ buffer(2) ]],
                                           constant float3& cameraForward [[ buffer(3) ]],
                                           constant bool& sortByDistance [[ buffer(4) ]],
                                           constant uint& splatCount [[ buffer(5) ]],
                                           constant float& maxCullDistance [[ buffer(6) ]]) {
    
    // Process 4 splats per thread for better SIMD utilization
    uint baseIndex = index * 4;
    if (baseIndex >= splatCount) return;
    
    float maxCullDistSquared = maxCullDistance * maxCullDistance;
    
    // Vectorized processing for up to 4 splats
    for (uint i = 0; i < 4 && (baseIndex + i) < splatCount; i++) {
        uint splatIndex = baseIndex + i;
        constant Splat& splat = splatArray[splatIndex];
        float3 splatPos = float3(splat.position);
        
        float distance;
        
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            distance = dot(delta, delta);
            
            // Early culling
            distance = (distance > maxCullDistSquared) ? FLT_MAX : distance;
        } else {
            distance = dot(splatPos, cameraForward);
            
            // Early culling  
            if (distance < 0.0 || distance > maxCullDistance) {
                distance = (distance < 0.0) ? -FLT_MAX : FLT_MAX;
            }
        }
        
        distances[splatIndex] = distance;
    }
} 