#import "ShaderCommon.h"

// Vertex shader for AABB wireframe rendering
struct AABBVertexOut {
    float4 position [[position]];
    half4 color;
};

vertex AABBVertexOut aabbVertexShader(
    const device packed_float3* vertices [[buffer(0)]],
    constant UniformsArray& uniformsArray [[buffer(1)]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]])
{
    constant Uniforms& uniforms = uniformsArray.uniforms[instanceID];

    float3 worldPos = float3(vertices[vertexID]);
    float4 clipPos = uniforms.projectionMatrix * uniforms.viewMatrix * float4(worldPos, 1.0);

    AABBVertexOut out;
    out.position = clipPos;
    out.color = half4(0.0h, 1.0h, 1.0h, 1.0h); // Cyan wireframe
    return out;
}

// Fragment shader for AABB wireframe rendering
fragment half4 aabbFragmentShader(AABBVertexOut in [[stage_in]])
{
    return in.color;
}
