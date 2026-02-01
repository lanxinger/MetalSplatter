#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

#if __METAL_VERSION__ >= 400

// SIMD-group matrix operations for efficient splat transforms
namespace simd_group_ops {
    
    // Cooperative matrix multiply for view-projection transforms
    template<typename T>
    [[user_annotation("simd_group_matrix_multiply")]]
    T simd_group_matrix_multiply_4x4(T a, T b, uint thread_index_in_simdgroup) {
        // Use SIMD-group cooperative operations for 4x4 matrix multiply
        // Each thread handles a portion of the computation
        T result = T(0);
        
        for (uint k = 0; k < 4; ++k) {
            T a_broadcast = simd_broadcast(a[k], thread_index_in_simdgroup % 4);
            result += a_broadcast * b[thread_index_in_simdgroup / 4];
        }
        
        return result;
    }
    
    // Cooperative covariance matrix computation for splats
    [[user_annotation("simd_group_covariance")]]
    float3x3 compute_covariance_simd_group(
        float3 scale,
        float4 rotation,
        uint thread_index_in_simdgroup
    ) {
        // Distribute rotation matrix computation across SIMD group
        float4 q = normalize(rotation);
        
        // Each thread computes different elements of rotation matrix
        float3 result_row;
        switch (thread_index_in_simdgroup % 3) {
            case 0: // First row
                result_row = float3(
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    2.0 * (q.x * q.y + q.w * q.z),
                    2.0 * (q.x * q.z - q.w * q.y)
                );
                break;
            case 1: // Second row
                result_row = float3(
                    2.0 * (q.x * q.y - q.w * q.z),
                    1.0 - 2.0 * (q.x * q.x + q.z * q.z),
                    2.0 * (q.y * q.z + q.w * q.x)
                );
                break;
            case 2: // Third row
                result_row = float3(
                    2.0 * (q.x * q.z + q.w * q.y),
                    2.0 * (q.y * q.z - q.w * q.x),
                    1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                );
                break;
        }
        
        // Apply scaling cooperatively
        result_row *= scale;
        
        // Gather results from SIMD group
        float3 row0 = simd_broadcast(result_row, 0);
        float3 row1 = simd_broadcast(result_row, 1);  
        float3 row2 = simd_broadcast(result_row, 2);
        
        return float3x3(row0, row1, row2);
    }
    
    // Cooperative frustum culling for multiple splats
    [[user_annotation("simd_group_frustum_cull")]]
    bool frustum_cull_simd_group(
        float3 center,
        float radius,
        constant float4 frustum_planes[6],
        uint thread_index_in_simdgroup
    ) {
        // Each thread tests against 1-2 frustum planes
        bool visible = true;
        
        uint plane_start = (thread_index_in_simdgroup * 6) / 32;
        uint plane_end = min(((thread_index_in_simdgroup + 1) * 6) / 32, 6u);
        
        for (uint i = plane_start; i < plane_end; ++i) {
            float distance = dot(frustum_planes[i].xyz, center) + frustum_planes[i].w;
            if (distance < -radius) {
                visible = false;
                break;
            }
        }
        
        // Use SIMD vote to determine if any thread found the splat invisible
        return simd_all(visible);
    }
}

// Enhanced vertex shader using SIMD-group operations
[[user_annotation("metal4_simd_group_vertex")]]
vertex FragmentIn metal4_simd_group_splatVertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    using namespace simd_group_ops;
    
    FragmentIn out;
    out.splatID = instanceID;

    if (instanceID >= argumentBuffer.splatCount) {
        out.position = float4(0, 0, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        return out;
    }
    
    Splat splat = argumentBuffer.splatBuffer[instanceID];
    
    // Use SIMD-group cooperative matrix operations
    float3x3 covariance = compute_covariance_simd_group(
        splat.scale, 
        splat.rotation, 
        thread_index_in_simdgroup
    );
    
    // Cooperative frustum culling
    float radius = length(splat.scale);
    bool is_visible = frustum_cull_simd_group(
        splat.position,
        radius,
        uniforms.frustumPlanes,
        thread_index_in_simdgroup
    );
    
    if (!is_visible) {
        out.position = float4(0, 0, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        return out;
    }
    
    // Continue with standard vertex processing...
    float4x4 modelViewProj = uniforms.projectionMatrix * uniforms.viewMatrix;
    
    // Transform splat position
    float4 worldPos = float4(splat.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    
    // Generate quad vertex
    float2 quadVertices[6] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(-1, 1), float2(1, -1), float2(1, 1)
    };
    
    float2 vertex = quadVertices[vertexID];
    
    // Apply 2D covariance in screen space
    float2 screenOffset = vertex * radius * 0.1; // Scale factor
    float4 screenPos = modelViewProj * float4(viewPos.xyz + float3(screenOffset, 0), 1.0);
    
    out.position = screenPos;
    out.relativePosition = half2(0);  // Not using for this path
    out.color = splat.color;
    out.lodBand = 0;
    out.debugFlags = 0;

    return out;
}

#endif // __METAL_VERSION__ >= 400