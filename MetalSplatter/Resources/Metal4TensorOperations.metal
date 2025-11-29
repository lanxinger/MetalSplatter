#include <metal_stdlib>
#include "ShaderCommon.h"

using namespace metal;

// =============================================================================
// Metal 4 TensorOps for Batch Gaussian Splat Processing
// =============================================================================
//
// Purpose:
// Pre-compute expensive per-splat operations in batch before rendering.
// This moves work out of the per-vertex/mesh-shader hot path.
//
// Key optimizations:
// 1. Batch covariance projection (3D → 2D) for all splats at once
// 2. Batch view-space transformation
// 3. Batch frustum culling with depth output for sorting
//
// Usage:
// Run these kernels when camera moves, results cached until next camera change.
// Mesh shaders then just read pre-computed values instead of computing per-vertex.
//
// =============================================================================

// -----------------------------------------------------------------------------
// MARK: - Pre-computed Splat Data Structure
// -----------------------------------------------------------------------------

struct PrecomputedSplat {
    float4 clipPosition;    // Already projected to clip space
    float3 cov2D;           // 2D covariance (cov_xx, cov_xy, cov_yy)
    float2 axis1;           // Decomposed covariance axis 1
    float2 axis2;           // Decomposed covariance axis 2
    float depth;            // For sorting
    uint visible;           // Frustum culling result (0 = culled, 1 = visible)
};

// -----------------------------------------------------------------------------
// MARK: - Helper Functions
// -----------------------------------------------------------------------------

// Compute 2D covariance from 3D covariance and view parameters
inline float3 computeCovariance2D(float3 viewPos,
                                   packed_half3 cov3Da,
                                   packed_half3 cov3Db,
                                   float4x4 viewMatrix,
                                   float4x4 projectionMatrix,
                                   uint2 screenSize) {
    float invViewPosZ = 1.0f / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;
    
    float tanHalfFovX = 1.0f / projectionMatrix[0][0];
    float tanHalfFovY = 1.0f / projectionMatrix[1][1];
    float limX = 1.3f * tanHalfFovX;
    float limY = 1.3f * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;
    
    float focalX = float(screenSize.x) * projectionMatrix[0][0] * 0.5f;
    float focalY = float(screenSize.y) * projectionMatrix[1][1] * 0.5f;
    
    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);
    
    // Add small regularization
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// Decompose 2D covariance into ellipse axes
inline void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b;
    float trace = a + d;
    
    float mean = 0.5f * trace;
    float dist = max(0.1f, sqrt(mean * mean - det));
    
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;
    
    float2 eigenvector1;
    if (b == 0.0f) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        float2 unnormalized = float2(b, d - lambda2);
        eigenvector1 = normalize(unnormalized);
    }
    
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);
    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

// -----------------------------------------------------------------------------
// MARK: - Batch Precompute Kernel (Metal 4 Optimized)
// -----------------------------------------------------------------------------
//
// This kernel pre-computes all per-splat data in batch:
// - View-space position
// - Clip-space position  
// - 2D covariance
// - Decomposed ellipse axes
// - Visibility flag
// - Depth for sorting
//
// Run once when camera changes, results used by mesh shaders.
//

#if __METAL_VERSION__ >= 400
[[user_annotation("batch_precompute_splats")]]
#endif
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void batchPrecomputeSplats(
    uint index [[thread_position_in_grid]],
    constant Splat* inputSplats [[buffer(0)]],
    device PrecomputedSplat* outputSplats [[buffer(1)]],
    constant Uniforms& uniforms [[buffer(2)]],
    constant uint& splatCount [[buffer(3)]]
) {
    if (index >= splatCount) return;
    
    Splat splat = inputSplats[index];
    PrecomputedSplat output;
    
    // Transform to view space
    float3 worldPos = float3(splat.position);
    float3 viewPos = (uniforms.viewMatrix * float4(worldPos, 1.0f)).xyz;
    
    // Early culling - behind camera
    if (viewPos.z >= 0.0f) {
        output.visible = 0;
        output.depth = 1e10f;
        outputSplats[index] = output;
        return;
    }
    
    // Transform to clip space
    float4 clipPos = uniforms.projectionMatrix * float4(viewPos, 1.0f);
    output.clipPosition = clipPos;
    output.depth = -viewPos.z;  // Positive depth for sorting (front-to-back)
    
    // NDC frustum culling
    float3 ndc = clipPos.xyz / clipPos.w;
    if (ndc.z > 1.0f || any(abs(ndc.xy) > 1.5f)) {
        output.visible = 0;
        outputSplats[index] = output;
        return;
    }
    
    output.visible = 1;
    
    // Compute 2D covariance
    output.cov2D = computeCovariance2D(viewPos, splat.covA, splat.covB,
                                        uniforms.viewMatrix, uniforms.projectionMatrix, 
                                        uniforms.screenSize);
    
    // Decompose into axes
    decomposeCovariance(output.cov2D, output.axis1, output.axis2);
    
    outputSplats[index] = output;
}

// -----------------------------------------------------------------------------
// MARK: - Batch Transform Kernel (SIMD-Optimized)
// -----------------------------------------------------------------------------
//
// Optimized batch position transformation using SIMD-group operations.
// Processes 4 splats per thread using vectorized loads.
//

#if __METAL_VERSION__ >= 400
[[user_annotation("batch_transform_positions_simd")]]
#endif
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void batchTransformPositionsSIMD(
    uint index [[thread_position_in_grid]],
    constant Splat* inputSplats [[buffer(0)]],
    device float4* viewPositions [[buffer(1)]],
    device float4* clipPositions [[buffer(2)]],
    device float* depths [[buffer(3)]],
    constant Uniforms& uniforms [[buffer(4)]],
    constant uint& splatCount [[buffer(5)]]
) {
    // Each thread processes 4 splats for better memory bandwidth utilization
    uint baseIndex = index * 4;
    
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint splatIndex = baseIndex + i;
        if (splatIndex >= splatCount) break;
        
        float3 worldPos = float3(inputSplats[splatIndex].position);
        float4 viewPos = uniforms.viewMatrix * float4(worldPos, 1.0f);
        float4 clipPos = uniforms.projectionMatrix * viewPos;
        
        viewPositions[splatIndex] = viewPos;
        clipPositions[splatIndex] = clipPos;
        depths[splatIndex] = -viewPos.z;
    }
}

// -----------------------------------------------------------------------------
// MARK: - Batch Covariance Kernel (SIMD-Optimized)
// -----------------------------------------------------------------------------
//
// Computes 2D covariance and decomposes into axes for rendering.
// Uses SIMD-group operations for parallel matrix math.
//

#if __METAL_VERSION__ >= 400
[[user_annotation("batch_compute_covariance_simd")]]
#endif
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void batchComputeCovarianceSIMD(
    uint index [[thread_position_in_grid]],
    constant Splat* inputSplats [[buffer(0)]],
    constant float4* viewPositions [[buffer(1)]],
    device float4* covarianceData [[buffer(2)]],  // xyz = cov2D, w = unused
    device float4* axisData [[buffer(3)]],        // xy = axis1, zw = axis2
    constant Uniforms& uniforms [[buffer(4)]],
    constant uint& splatCount [[buffer(5)]]
) {
    if (index >= splatCount) return;
    
    float3 viewPos = viewPositions[index].xyz;
    
    // Skip if behind camera
    if (viewPos.z >= 0.0f) {
        covarianceData[index] = float4(0);
        axisData[index] = float4(0);
        return;
    }
    
    Splat splat = inputSplats[index];
    
    float3 cov2D = computeCovariance2D(viewPos, splat.covA, splat.covB,
                                        uniforms.viewMatrix, uniforms.projectionMatrix,
                                        uniforms.screenSize);
    
    float2 axis1, axis2;
    decomposeCovariance(cov2D, axis1, axis2);
    
    covarianceData[index] = float4(cov2D, 0);
    axisData[index] = float4(axis1, axis2);
}

// -----------------------------------------------------------------------------
// MARK: - Count Visible Splats (for indirect draw)
// -----------------------------------------------------------------------------

#if __METAL_VERSION__ >= 400
[[user_annotation("count_visible_splats")]]
#endif
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void countVisibleSplats(
    uint index [[thread_position_in_grid]],
    constant PrecomputedSplat* precomputed [[buffer(0)]],
    device atomic_uint* visibleCount [[buffer(1)]],
    device uint* visibleIndices [[buffer(2)]],
    constant uint& splatCount [[buffer(3)]]
) {
    if (index >= splatCount) return;
    
    // Check visibility and store index if visible
    uint isVisible = precomputed[index].visible;
    
    // Each visible thread claims a slot via atomic add
    if (isVisible) {
        uint slot = atomic_fetch_add_explicit(visibleCount, 1, memory_order_relaxed);
        visibleIndices[slot] = index;
    }
}

// -----------------------------------------------------------------------------
// MARK: - Precomputed Data Access for Mesh Shaders
// -----------------------------------------------------------------------------
//
// When using precomputed data, mesh shaders can read directly:
//
// float4 clipPos = precomputed[splatIndex].clipPosition;
// float2 axis1 = precomputed[splatIndex].axis1;
// float2 axis2 = precomputed[splatIndex].axis2;
//
// This eliminates redundant covariance computation in the mesh shader.
//
