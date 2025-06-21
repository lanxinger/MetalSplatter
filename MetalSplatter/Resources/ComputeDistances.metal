#include "ShaderCommon.h"

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistances(uint index [[thread_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 constant Splat* splatArray [[ buffer(0) ]],
                                 device float* distances [[ buffer(1) ]],
                                 constant float3& cameraPosition [[ buffer(2) ]],
                                 constant float3& cameraForward [[ buffer(3) ]],
                                 constant bool& sortByDistance [[ buffer(4) ]],
                                 constant uint& splatCount [[ buffer(5) ]]) {
    
    if (index >= splatCount) return;
    
    Splat splat = splatArray[index];
    float3 splatPos = float3(splat.position);
    
    // Optimized distance calculation
    if (sortByDistance) {
        float3 delta = splatPos - cameraPosition;
        // Calculate length squared for distance sorting
        distances[index] = dot(delta, delta);
    } else {
        // Project onto forward vector for depth sorting
        distances[index] = dot(splatPos, cameraForward);
    }
} 