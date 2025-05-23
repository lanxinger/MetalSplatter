#include "ShaderCommon.h"

kernel void computeSplatDistances(uint index [[thread_position_in_grid]],
                                 constant Splat* splatArray [[ buffer(0) ]],
                                 device float* distances [[ buffer(1) ]],
                                 constant float3& cameraPosition [[ buffer(2) ]],
                                 constant float3& cameraForward [[ buffer(3) ]],
                                 constant bool& sortByDistance [[ buffer(4) ]],
                                 constant uint& splatCount [[ buffer(5) ]]) {
    
    if (index >= splatCount) return;
    
    Splat splat = splatArray[index];
    float3 splatPos = float3(splat.position);
    
    if (sortByDistance) {
        float3 delta = splatPos - cameraPosition;
        distances[index] = dot(delta, delta); // length squared
    } else {
        distances[index] = dot(splatPos, cameraForward);
    }
} 