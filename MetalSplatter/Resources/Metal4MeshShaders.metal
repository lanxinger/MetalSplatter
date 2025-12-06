#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

#if __METAL_VERSION__ >= 400

// Mesh shader structs for splat rendering
struct MeshOutput {
    float4 position [[position]];
    float4 color;
    float2 uv;
    uint splat_id;
};

// Increased from 32 to 64 for better GPU occupancy.
// Metal mesh shaders have max 256 vertices per output, with 4 vertices/splat = 64 max.
constant constexpr uint MESHLET_SIZE = 64;

struct ObjectPayload {
    uint visible_splat_count;
    uint splat_indices[64]; // Max 64 splats per meshlet (Metal limit: 256 vertices / 4 = 64)
};

// Object shader for splat culling and LOD
[[user_annotation("metal4_object_shader_splats")]]
[[max_total_threads_per_threadgroup(MESHLET_SIZE)]]
object void splatObjectShader(
    constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]],
    object_data ObjectPayload& payload [[payload]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint meshlet_id = threadgroup_position_in_grid.x;
    uint thread_id = thread_position_in_threadgroup.x;
    uint splats_per_meshlet = MESHLET_SIZE;
    uint splat_start = meshlet_id * splats_per_meshlet;
    uint splat_id = splat_start + thread_id;
    
    payload.visible_splat_count = 0;
    
    // Early exit if beyond splat range
    if (splat_id >= argumentBuffer.splatCount) {
        return;
    }
    
    // Load splat data
    Splat splat = argumentBuffer.splatBuffer[splat_id];
    
    // Frustum culling
    float radius = length(splat.scale);
    bool is_visible = true;
    
    // Test against all frustum planes
    for (uint i = 0; i < 6; ++i) {
        float distance = dot(uniforms.frustumPlanes[i].xyz, splat.position) + uniforms.frustumPlanes[i].w;
        if (distance < -radius) {
            is_visible = false;
            break;
        }
    }
    
    // Distance-based LOD culling
    float4 view_pos = uniforms.viewMatrix * float4(splat.position, 1.0);
    float distance_to_camera = length(view_pos.xyz);
    float max_distance = 100.0; // Configurable LOD distance
    
    if (distance_to_camera > max_distance) {
        is_visible = false;
    }
    
    // Size-based culling for tiny splats
    float projected_size = radius / distance_to_camera;
    if (projected_size < 0.001) { // Less than 1 pixel
        is_visible = false;
    }
    
    // Add to visible list if passes all culling tests
    if (is_visible) {
        uint index = atomic_fetch_add_explicit(&payload.visible_splat_count, 1, memory_order_relaxed);
        if (index < MESHLET_SIZE) {
            payload.splat_indices[index] = splat_id;
        }
    }

    // Generate meshlets based on visible splat count
    uint num_meshlets = (payload.visible_splat_count + MESHLET_SIZE - 1) / MESHLET_SIZE; // Round up
    mesh_grid_properties properties;
    properties.set_threadgroups_per_grid(uint3(num_meshlets, 1, 1));
    set_mesh_grid_properties(properties);
}

// Mesh shader for generating splat quads
[[user_annotation("metal4_mesh_shader_splats")]]
[[max_total_threads_per_threadgroup(MESHLET_SIZE)]]
[[max_total_threadgroups_per_mesh_grid(256)]]
mesh void splatMeshShader(
    constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]],
    const object_data ObjectPayload& payload [[payload]],
    mesh<MeshOutput, void, 6, 4> output_mesh,
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint thread_id = thread_position_in_threadgroup.x;

    // Early exit if no visible splats
    if (payload.visible_splat_count == 0) {
        output_mesh.set_primitive_count(0);
        return;
    }

    // Calculate number of primitives to generate
    uint primitive_count = min(payload.visible_splat_count, (uint)MESHLET_SIZE);
    output_mesh.set_primitive_count(primitive_count);
    
    if (thread_id >= primitive_count) {
        return;
    }
    
    // Get splat data for this thread
    uint splat_id = payload.splat_indices[thread_id];
    Splat splat = argumentBuffer.splatBuffer[splat_id];
    
    // Transform splat to screen space
    float4x4 viewProj = uniforms.projectionMatrix * uniforms.viewMatrix;
    float4 center_screen = viewProj * float4(splat.position, 1.0);
    
    // Calculate splat size in screen space
    float4 view_pos = uniforms.viewMatrix * float4(splat.position, 1.0);
    float distance = length(view_pos.xyz);
    float scale_factor = max(splat.scale.x, max(splat.scale.y, splat.scale.z));
    float screen_scale = scale_factor / distance * 100.0; // Scale factor
    
    // Generate quad vertices for this splat
    float2 offsets[4] = {
        float2(-screen_scale, -screen_scale),
        float2( screen_scale, -screen_scale),
        float2(-screen_scale,  screen_scale),
        float2( screen_scale,  screen_scale)
    };
    
    float2 uvs[4] = {
        float2(0, 0),
        float2(1, 0), 
        float2(0, 1),
        float2(1, 1)
    };
    
    // Generate 4 vertices for the quad
    uint vertex_base = thread_id * 4;
    for (uint i = 0; i < 4; ++i) {
        uint vertex_id = vertex_base + i;
        if (vertex_id < output_mesh.max_vertex_count) {
            MeshOutput vertex;
            
            // Apply screen space offset
            float4 screen_pos = center_screen;
            screen_pos.xy += offsets[i] * screen_pos.w;
            
            vertex.position = screen_pos;
            vertex.color = splat.color;
            vertex.uv = uvs[i];
            vertex.splat_id = splat_id;
            
            output_mesh.set_vertex(vertex_id, vertex);
        }
    }
    
    // Generate 2 triangles for the quad
    uint triangle_base = thread_id * 2;
    if (triangle_base < output_mesh.max_primitive_count) {
        // First triangle: 0, 1, 2
        output_mesh.set_index(triangle_base * 3 + 0, vertex_base + 0);
        output_mesh.set_index(triangle_base * 3 + 1, vertex_base + 1);
        output_mesh.set_index(triangle_base * 3 + 2, vertex_base + 2);
        
        // Second triangle: 2, 1, 3  
        if (triangle_base + 1 < output_mesh.max_primitive_count) {
            output_mesh.set_index((triangle_base + 1) * 3 + 0, vertex_base + 2);
            output_mesh.set_index((triangle_base + 1) * 3 + 1, vertex_base + 1);
            output_mesh.set_index((triangle_base + 1) * 3 + 2, vertex_base + 3);
        }
    }
}

// Fragment shader for mesh-rendered splats
[[user_annotation("metal4_mesh_fragment_splats")]]
fragment float4 splatMeshFragment(
    MeshOutput in [[stage_in]],
    constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]]
) {
    // Get splat data
    if (in.splat_id >= argumentBuffer.splatCount) {
        discard_fragment();
    }
    
    Splat splat = argumentBuffer.splatBuffer[in.splat_id];
    
    // Gaussian falloff based on UV coordinates
    float2 uv_centered = in.uv * 2.0 - 1.0; // Map to [-1, 1]
    float gaussian = exp(-dot(uv_centered, uv_centered));
    
    // Apply gaussian alpha
    float4 color = in.color;
    color.a *= gaussian;
    
    // Alpha test for performance
    if (color.a < 0.01) {
        discard_fragment();
    }
    
    return color;
}

// Alternative high-performance mesh shader variant for dense scenes
[[user_annotation("metal4_mesh_shader_dense")]]
[[max_total_threads_per_threadgroup(64)]]
mesh void denseSplatMeshShader(
    constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]],
    const object_data ObjectPayload& payload [[payload]],
    mesh<MeshOutput, void, 6, 4> output_mesh,
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    // Higher density version - pack more splats per meshlet
    uint thread_id = thread_position_in_threadgroup.x;
    uint splats_per_thread = 2; // Process 2 splats per thread
    
    uint primitive_count = min(payload.visible_splat_count * splats_per_thread, 64u);
    output_mesh.set_primitive_count(primitive_count);
    
    if (thread_id * splats_per_thread >= payload.visible_splat_count) {
        return;
    }
    
    // Process multiple splats per thread for better utilization
    for (uint splat_offset = 0; splat_offset < splats_per_thread; ++splat_offset) {
        uint splat_index = thread_id * splats_per_thread + splat_offset;
        if (splat_index >= payload.visible_splat_count) break;
        
        uint splat_id = payload.splat_indices[splat_index];
        Splat splat = argumentBuffer.splatBuffer[splat_id];
        
        // Similar quad generation logic but optimized for density
        // ... (similar implementation with optimizations for batch processing)
    }
}

#endif // __METAL_VERSION__ >= 400