#include <metal_stdlib>
using namespace metal;

// Import our common structures
#include "SplatProcessing.h"

// Real Metal 4.0 argument buffer structure (not custom abstraction)
// This corresponds to the MTLArgumentEncoder setup in Metal4ArgumentBufferManager
struct SplatArgumentBuffer {
    device Splat *splatBuffer [[id(0)]];
    constant UniformsArray &uniformsArray [[id(1)]];
};

// Metal 4.0 vertex shader using real argument buffers
#if __METAL_VERSION__ >= 400
[[user_annotation("metal4_bindless_vertex")]]
#endif
vertex FragmentIn metal4_splatVertex(uint vertexID [[vertex_id]],
                                     uint instanceID [[instance_id]],
                                     ushort amplificationID [[amplification_id]],
                                     constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]]) {
    
    // Access resources through argument buffer (bindless)
    device Splat *splatArray = argumentBuffer.splatBuffer;
    constant UniformsArray &uniformsArray = argumentBuffer.uniformsArray;
    
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount - 1)];
    
    uint logicalSplatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (logicalSplatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        out.splatID = 0;
        return out;
    }

    Splat splat = splatArray[logicalSplatID];
    return splatVertex(splat, uniforms, vertexID % 4, logicalSplatID);
}

// Metal 4.0 fragment shader (can also access argument buffer if needed)
#if __METAL_VERSION__ >= 400
[[user_annotation("metal4_bindless_fragment")]]
#endif
fragment half4 metal4_splatFragment(FragmentIn in [[stage_in]]) {
    return shadeSplat(in);
}

// Alternative compute kernel using argument buffer
#if __METAL_VERSION__ >= 400
[[user_annotation("metal4_bindless_compute")]]
#endif
kernel void metal4_splatCompute(constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    // Example of compute shader accessing bindless resources
    // Note: Resources accessed through argument buffer for demonstration
    // In actual implementation, these would be used for processing
    
    // Demonstrate bindless access pattern without unused variable warnings
    if (gid.x == 0 && gid.y == 0) {
        // Access resources to demonstrate bindless pattern
        device Splat *splats = argumentBuffer.splatBuffer;
        constant UniformsArray &uniforms = argumentBuffer.uniformsArray;
        
        // Placeholder: Would perform actual compute operations here
        (void)splats;   // Suppress unused variable warning
        (void)uniforms; // Suppress unused variable warning
    }
}
