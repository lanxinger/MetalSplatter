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

kernel void frustumCullSplats(uint index [[thread_position_in_grid]],
                             constant Splat* inputSplats [[ buffer(0) ]],
                             device uint* visibleIndices [[ buffer(1) ]],
                             device atomic_uint* visibleCount [[ buffer(2) ]],
                             constant FrustumCullData& cullData [[ buffer(3) ]],
                             constant uint& splatCount [[ buffer(4) ]]) {
    
    if (index >= splatCount) return;
    
    Splat splat = inputSplats[index];
    float3 splatPos = float3(splat.position);
    
    // Distance culling
    float3 toCam = splatPos - cullData.cameraPosition;
    if (dot(toCam, toCam) > cullData.maxDistance * cullData.maxDistance) {
        return;
    }
    
    // Frustum culling - test against all 6 planes
    // Approximate splat radius for culling (conservative estimate)
    float splatRadius = 2.0; // Could be calculated from covariance
    
    bool visible = true;
    for (uint i = 0; i < 6 && visible; i++) {
        FrustumPlane plane = cullData.planes[i];
        float distance = dot(plane.normal, splatPos) + plane.distance;
        if (distance < -splatRadius) {
            visible = false;
        }
    }
    
    if (visible) {
        uint visibleIndex = atomic_fetch_add_explicit(visibleCount, 1, memory_order_relaxed);
        visibleIndices[visibleIndex] = index;
    }
} 