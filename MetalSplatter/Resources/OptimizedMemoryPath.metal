#include "SplatProcessing.h"

// Buffer indices for optimized path
enum OptimizedBufferIndex: int32_t
{
    OptimizedBufferIndexUniforms = 0,
    OptimizedBufferIndexSplat    = 1,
    OptimizedBufferIndexColor    = 2,
};

// Helper function to unpack RGBA8888 color
half4 unpackColor(uint packedColor) {
    half4 color;
    color.r = half((packedColor >> 24) & 0xFF) / 255.0;
    color.g = half((packedColor >> 16) & 0xFF) / 255.0;
    color.b = half((packedColor >> 8) & 0xFF) / 255.0;
    color.a = half(packedColor & 0xFF) / 255.0;
    return color;
}

vertex FragmentIn optimizedSplatVertexShader(uint vertexID [[vertex_id]],
                                            uint instanceID [[instance_id]],
                                            ushort amplificationID [[amplification_id]],
                                            constant SplatOptimized* splatArray [[ buffer(OptimizedBufferIndexSplat) ]],
                                            constant PackedColor* colorArray [[ buffer(OptimizedBufferIndexColor) ]],
                                            constant UniformsArray & uniformsArray [[ buffer(OptimizedBufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];
    
    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }
    
    SplatOptimized splatOpt = splatArray[splatID];
    PackedColor packedColor = colorArray[splatID];
    
    // Create a standard Splat structure for the existing vertex function
    Splat splat;
    splat.position = splatOpt.position;
    splat.covA = splatOpt.covA;
    splat.covB = splatOpt.covB;
    splat.color = unpackColor(packedColor.rgba);
    
    return splatVertex(splat, uniforms, vertexID % 4);
}

// Fragment shader remains the same
fragment half4 optimizedSplatFragmentShader(FragmentIn in [[stage_in]]) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    return half4(alpha * in.color.rgb, alpha);
}