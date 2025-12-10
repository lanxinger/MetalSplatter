#include <metal_stdlib>
#include "ShaderCommon.h"
#include "SplatProcessing.h"
using namespace metal;

// =============================================================================
// Metal 4 Optimized Mesh Shaders for Gaussian Splat Rendering
// - User annotations for GPU profiling
// - SIMD-group reductions (faster than atomics)
// - Optimized threadgroup limits
// - Proper covariance computation for accurate Gaussians
// =============================================================================

// =============================================================================
// MARK: - Configuration
// =============================================================================

// Increased from 32 to 64 for better GPU occupancy and reduced dispatch overhead.
// Metal mesh shaders have a max of 256 vertices per mesh output, so with 4 vertices
// per splat, the maximum is 64 splats per meshlet (256/4 = 64).
// PlayCanvas uses 128 splats per instance but WebGL has different constraints.
constant constexpr uint SPLATS_PER_MESHLET = 64;
constant constexpr uint VERTICES_PER_SPLAT = 4;
constant constexpr uint TRIANGLES_PER_SPLAT = 2;
constant constexpr uint MAX_VERTICES = SPLATS_PER_MESHLET * VERTICES_PER_SPLAT;    // 256 (Metal limit)
constant constexpr uint MAX_PRIMITIVES = SPLATS_PER_MESHLET * TRIANGLES_PER_SPLAT; // 128
constant constexpr uint MAX_MESH_THREADGROUPS = 256;

// =============================================================================
// MARK: - Payload Structure (Object → Mesh)
// =============================================================================

struct MeshletPayload {
    uint visibleCount;
    uint meshletStartIndex;
    float3 viewPositions[SPLATS_PER_MESHLET];
    uint visibleLocalIndices[SPLATS_PER_MESHLET];
};

// =============================================================================
// MARK: - Mesh Vertex Output
// =============================================================================

struct MeshVertexOutput {
    float4 position [[position]];
    half2 relativePosition;
    half4 color;
    half lodBand;
    uint debugFlags;
};

// Mesh type alias
using SplatMeshType = metal::mesh<MeshVertexOutput, void, MAX_VERTICES, MAX_PRIMITIVES, metal::topology::triangle>;

// =============================================================================
// MARK: - Helper Functions
// =============================================================================

inline float3 meshCalcCovariance2D(float3 viewPos,
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

    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

inline void meshDecomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
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

// =============================================================================
// MARK: - Object Shader (Metal 4 Optimized)
// - User annotation for GPU profiling
// - SIMD-group reduction instead of atomics (faster)
// - max_total_threadgroups_per_mesh_grid for dispatch optimization
// =============================================================================

#if __METAL_VERSION__ >= 400
[[user_annotation("splat_object_shader_metal4")]]
#endif
[[object,
  max_total_threads_per_threadgroup(SPLATS_PER_MESHLET),
  max_total_threadgroups_per_mesh_grid(MAX_MESH_THREADGROUPS)]]
void splatObjectShader(
    uint threadIndex [[thread_index_in_threadgroup]],
    uint simdLaneId [[thread_index_in_simdgroup]],
    uint simdGroupId [[simdgroup_index_in_threadgroup]],
    uint meshletIndex [[threadgroup_position_in_grid]],
    constant Splat* splatArray [[buffer(BufferIndexSplat)]],
    constant UniformsArray& uniformsArray [[buffer(BufferIndexUniforms)]],
    constant int32_t* sortedIndices [[buffer(BufferIndexSortedIndices)]],
    object_data MeshletPayload& payload [[payload]],
    mesh_grid_properties meshGridProps
) {
    Uniforms uniforms = uniformsArray.uniforms[0];
    
    uint meshletStartIndex = meshletIndex * SPLATS_PER_MESHLET;
    uint localIndex = threadIndex;
    uint globalSortedIndex = meshletStartIndex + localIndex;
    
    payload.meshletStartIndex = meshletStartIndex;
    
    // Check validity and visibility
    bool isValid = globalSortedIndex < uniforms.splatCount;
    bool isVisible = false;
    float3 viewPos = float3(0);
    
    if (isValid) {
        uint actualSplatIndex = uint(sortedIndices[globalSortedIndex]);
        Splat splat = splatArray[actualSplatIndex];
        
        float3 splatPos = float3(splat.position);
        viewPos = uniforms.viewMatrix[0].xyz * splatPos.x +
                  uniforms.viewMatrix[1].xyz * splatPos.y +
                  uniforms.viewMatrix[2].xyz * splatPos.z +
                  uniforms.viewMatrix[3].xyz;
        
        // Cull if behind camera
        if (viewPos.z < 0.0f) {
            float4 projectedCenter = uniforms.projectionMatrix * float4(viewPos, 1.0f);
            float3 ndc = projectedCenter.xyz / projectedCenter.w;
            
            // Frustum check with margin
            if (ndc.z <= 1.0f && all(abs(ndc.xy) <= 1.5f)) {
                isVisible = true;
            }
        }
    }
    
    // =======================================================================
    // SIMD-group reduction for visible count (faster than atomics!)
    // Uses simd_sum to count visible splats within each SIMD-group
    // Then uses threadgroup memory to combine SIMD-group results
    // =======================================================================
    
    uint isVisibleUint = isVisible ? 1u : 0u;
    
    // SIMD-group prefix sum to get local index within SIMD-group
    uint simdVisibleCount = simd_sum(isVisibleUint);
    uint simdPrefixSum = simd_prefix_exclusive_sum(isVisibleUint);
    
    // Threadgroup memory to accumulate SIMD-group results
    // With 64 threads and SIMD width 32, we have 2 SIMD-groups
    // (Max 4 SIMD-groups if SIMD width is 16 on some devices)
    threadgroup uint simdGroupCounts[4];
    threadgroup uint simdGroupOffsets[4];
    
    // First lane of each SIMD-group writes its count
    if (simdLaneId == 0) {
        simdGroupCounts[simdGroupId] = simdVisibleCount;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Thread 0 computes prefix sums for SIMD-group offsets
    uint totalVisible = 0;
    if (threadIndex == 0) {
        uint offset = 0;
        for (uint i = 0; i < 4; i++) {
            simdGroupOffsets[i] = offset;
            offset += simdGroupCounts[i];
        }
        totalVisible = offset;
        payload.visibleCount = totalVisible;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each visible thread writes to payload
    if (isVisible) {
        uint globalSlot = simdGroupOffsets[simdGroupId] + simdPrefixSum;
        if (globalSlot < SPLATS_PER_MESHLET) {
            payload.visibleLocalIndices[globalSlot] = localIndex;
            payload.viewPositions[globalSlot] = viewPos;
        }
    }
    
    // First thread dispatches mesh shader
    if (threadIndex == 0) {
        if (payload.visibleCount > 0) {
            meshGridProps.set_threadgroups_per_grid(uint3(1, 1, 1));
        } else {
            meshGridProps.set_threadgroups_per_grid(uint3(0, 0, 0));
        }
    }
}

// =============================================================================
// MARK: - Mesh Shader (Metal 4 Optimized)
// - User annotation for GPU profiling
// - Proper covariance computation (1x per splat, not 4x!)
// =============================================================================

#if __METAL_VERSION__ >= 400
[[user_annotation("splat_mesh_shader_metal4")]]
#endif
[[mesh, max_total_threads_per_threadgroup(SPLATS_PER_MESHLET)]]
void splatMeshShader(
    SplatMeshType outputMesh,
    const object_data MeshletPayload& payload [[payload]],
    constant Splat* splatArray [[buffer(BufferIndexSplat)]],
    constant UniformsArray& uniformsArray [[buffer(BufferIndexUniforms)]],
    constant int32_t* sortedIndices [[buffer(BufferIndexSortedIndices)]],
    uint threadIndex [[thread_index_in_threadgroup]]
) {
    Uniforms uniforms = uniformsArray.uniforms[0];
    
    // First thread sets primitive count
    if (threadIndex == 0) {
        outputMesh.set_primitive_count(payload.visibleCount * TRIANGLES_PER_SPLAT);
    }
    
    if (threadIndex >= payload.visibleCount) {
        return;
    }
    
    uint localIndex = payload.visibleLocalIndices[threadIndex];
    uint globalSortedIndex = payload.meshletStartIndex + localIndex;
    uint actualSplatIndex = uint(sortedIndices[globalSortedIndex]);
    
    Splat splat = splatArray[actualSplatIndex];
    float3 viewPos = payload.viewPositions[threadIndex];
    
    // Compute 2D covariance ONCE per splat (key optimization vs vertex shader)
    float3 cov2D = meshCalcCovariance2D(viewPos, splat.covA, splat.covB,
                                         uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    
    float2 axis1, axis2;
    meshDecomposeCovariance(cov2D, axis1, axis2);
    
    float4 projectedCenter = uniforms.projectionMatrix * float4(viewPos, 1.0f);

    // Pre-compute scale factor ONCE (saves 3 divisions per splat)
    // scaleFactor = (2 * kBoundsRadius) / screenSize
    half2 scaleFactor = (2.0h * kBoundsRadius) / half2(uniforms.screenSize);

    // Pre-scale axes by the scale factor to avoid per-vertex division
    half2 scaledAxis1 = half2(axis1) * scaleFactor;
    half2 scaledAxis2 = half2(axis2) * scaleFactor;

    // Pre-compute common vertex attributes
    half4 splatColor = splat.color;
    uint debugFlags = uniforms.debugFlags;
    float projW = projectedCenter.w;

    uint vertexBase = threadIndex * VERTICES_PER_SPLAT;

    // Generate 4 vertices for this splat's quad (unrolled)
    // Each vertex now uses pre-scaled axes - just multiply, no division

    // Vertex 0: bottom-left (-1, -1)
    {
        half2 screenDelta = -scaledAxis1 - scaledAxis2;  // (-1) * axis1 + (-1) * axis2

        MeshVertexOutput v0;
        v0.position = float4(projectedCenter.x + float(screenDelta.x) * projW,
                             projectedCenter.y + float(screenDelta.y) * projW,
                             projectedCenter.z, projW);
        v0.relativePosition = half2(-kBoundsRadius, -kBoundsRadius);
        v0.color = splatColor;
        v0.lodBand = half(0);
        v0.debugFlags = debugFlags;
        outputMesh.set_vertex(vertexBase + 0, v0);
    }

    // Vertex 1: top-left (-1, 1)
    {
        half2 screenDelta = -scaledAxis1 + scaledAxis2;  // (-1) * axis1 + (1) * axis2

        MeshVertexOutput v1;
        v1.position = float4(projectedCenter.x + float(screenDelta.x) * projW,
                             projectedCenter.y + float(screenDelta.y) * projW,
                             projectedCenter.z, projW);
        v1.relativePosition = half2(-kBoundsRadius, kBoundsRadius);
        v1.color = splatColor;
        v1.lodBand = half(0);
        v1.debugFlags = debugFlags;
        outputMesh.set_vertex(vertexBase + 1, v1);
    }

    // Vertex 2: bottom-right (1, -1)
    {
        half2 screenDelta = scaledAxis1 - scaledAxis2;  // (1) * axis1 + (-1) * axis2

        MeshVertexOutput v2;
        v2.position = float4(projectedCenter.x + float(screenDelta.x) * projW,
                             projectedCenter.y + float(screenDelta.y) * projW,
                             projectedCenter.z, projW);
        v2.relativePosition = half2(kBoundsRadius, -kBoundsRadius);
        v2.color = splatColor;
        v2.lodBand = half(0);
        v2.debugFlags = debugFlags;
        outputMesh.set_vertex(vertexBase + 2, v2);
    }

    // Vertex 3: top-right (1, 1)
    {
        half2 screenDelta = scaledAxis1 + scaledAxis2;  // (1) * axis1 + (1) * axis2

        MeshVertexOutput v3;
        v3.position = float4(projectedCenter.x + float(screenDelta.x) * projW,
                             projectedCenter.y + float(screenDelta.y) * projW,
                             projectedCenter.z, projW);
        v3.relativePosition = half2(kBoundsRadius, kBoundsRadius);
        v3.color = splatColor;
        v3.lodBand = half(0);
        v3.debugFlags = debugFlags;
        outputMesh.set_vertex(vertexBase + 3, v3);
    }
    
    // Generate 2 triangles for this splat's quad
    uint triangleBase = threadIndex * TRIANGLES_PER_SPLAT;
    
    // Triangle 0: vertices 0, 1, 2
    outputMesh.set_index(triangleBase * 3 + 0, vertexBase + 0);
    outputMesh.set_index(triangleBase * 3 + 1, vertexBase + 1);
    outputMesh.set_index(triangleBase * 3 + 2, vertexBase + 2);
    
    // Triangle 1: vertices 1, 2, 3
    outputMesh.set_index(triangleBase * 3 + 3, vertexBase + 1);
    outputMesh.set_index(triangleBase * 3 + 4, vertexBase + 2);
    outputMesh.set_index(triangleBase * 3 + 5, vertexBase + 3);
}

// =============================================================================
// MARK: - Fragment Shader (Metal 4 Optimized)
// =============================================================================

#if __METAL_VERSION__ >= 400
[[user_annotation("splat_mesh_fragment_metal4")]]
#endif
fragment half4 meshSplatFragmentShader(MeshVertexOutput in [[stage_in]]) {
    half distanceSquared = dot(in.relativePosition, in.relativePosition);
    if (distanceSquared > kBoundsRadiusSquared) {
        discard_fragment();
    }
    
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    half3 color = in.color.rgb;
    
    if ((in.debugFlags & DebugFlagLodTint) != 0) {
        color = mix(color, lodTintForBand(in.lodBand), half(0.5));
    }
    
    return half4(color * alpha, alpha);
}
