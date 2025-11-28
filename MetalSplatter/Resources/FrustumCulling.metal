#include "ShaderCommon.h"

// Simplified frustum cull data using view-projection matrix directly
struct FrustumCullData {
    float4x4 viewProjectionMatrix;  // Combined view-projection for NDC transform
    float3 cameraPosition;
    float padding1;
    float maxDistance;
    float3 padding2;
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
    
    // Threadgroup memory for visible indices batching
    threadgroup uint localVisibleIndices[256];
    threadgroup atomic_uint localVisibleCount;
    
    // Initialize threadgroup counter
    if (tid == 0) {
        atomic_store_explicit(&localVisibleCount, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (index >= splatCount) return;
    
    // Load splat position
    float3 splatPos = float3(inputSplats[index].position);
    
    // Distance culling (optional, skip if maxDistance is very large)
    if (cullData.maxDistance < 10000.0) {
        float3 toCam = splatPos - cullData.cameraPosition;
        float distanceSquared = dot(toCam, toCam);
        float maxDistanceSquared = cullData.maxDistance * cullData.maxDistance;
        
        if (distanceSquared > maxDistanceSquared) {
            return;
        }
    }
    
    // === NDC-based frustum culling ===
    // Transform splat position to clip space
    float4 clipPos = cullData.viewProjectionMatrix * float4(splatPos, 1.0);
    
    bool visible = true;
    
    // Only cull if clearly behind camera (conservative)
    // Use a negative threshold to allow splats that are very close
    if (clipPos.w < -0.1) {
        visible = false;
    }
    
    // For splats in front of camera, check X/Y bounds only
    // Skip near/far plane culling - it's too aggressive for gaussian splats
    if (visible && clipPos.w > 0.001) {
        // Perspective divide for X/Y only
        float invW = 1.0 / clipPos.w;
        float ndcX = clipPos.x * invW;
        float ndcY = clipPos.y * invW;
        
        // Very conservative margin - splats can be quite large
        float margin = 1.5;
        
        // Only cull if clearly outside left/right/top/bottom
        visible = (ndcX >= -1.0 - margin) && (ndcX <= 1.0 + margin) &&
                  (ndcY >= -1.0 - margin) && (ndcY <= 1.0 + margin);
    }
    
    // Store visible indices
    if (visible) {
        uint localIdx = atomic_fetch_add_explicit(&localVisibleCount, 1, memory_order_relaxed);
        if (localIdx < 256) {
            localVisibleIndices[localIdx] = index;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Flush visible indices to global memory
    if (tid == 0) {
        uint localCount = min(atomic_load_explicit(&localVisibleCount, memory_order_relaxed), 256u);
        uint globalStartIdx = atomic_fetch_add_explicit(visibleCount, localCount, memory_order_relaxed);
        
        for (uint i = 0; i < localCount; i++) {
            visibleIndices[globalStartIdx + i] = localVisibleIndices[i];
        }
    }
}

// =============================================================================
// Indirect Draw Arguments Generation
// =============================================================================
// Converts visible count to MTLDrawIndexedPrimitivesIndirectArguments
// This enables fully GPU-driven rendering with zero CPU readback

struct DrawIndexedIndirectArguments {
    uint indexCount;
    uint instanceCount;
    uint indexStart;
    int  baseVertex;
    uint baseInstance;
};

[[kernel]]
kernel void generateIndirectDrawArguments(device DrawIndexedIndirectArguments* drawArgs [[buffer(0)]],
                                          device atomic_uint* visibleCount [[buffer(1)]],
                                          constant uint& indicesPerSplat [[buffer(2)]],   // 6 for triangles
                                          constant uint& maxIndexedSplatCount [[buffer(3)]]) {
    // Read the visible count (set by frustumCullSplats)
    uint visible = atomic_load_explicit(visibleCount, memory_order_relaxed);
    
    // Calculate instance count based on indexed splat batching
    uint indexedCount = min(visible, maxIndexedSplatCount);
    uint instanceCount = (visible + maxIndexedSplatCount - 1) / maxIndexedSplatCount;
    
    // Populate indirect draw arguments
    drawArgs->indexCount = indexedCount * indicesPerSplat;  // 6 indices per splat (2 triangles)
    drawArgs->instanceCount = instanceCount;
    drawArgs->indexStart = 0;
    drawArgs->baseVertex = 0;
    drawArgs->baseInstance = 0;
}

// Reset visible count before culling pass
[[kernel]]
kernel void resetVisibleCount(device atomic_uint* visibleCount [[buffer(0)]]) {
    atomic_store_explicit(visibleCount, 0, memory_order_relaxed);
}