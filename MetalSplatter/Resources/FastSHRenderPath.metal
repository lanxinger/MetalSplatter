#include "ShaderCommon.h"
#include "SplatProcessing.h"

// Extended Splat structure for SH support
typedef struct {
    packed_float3 position;
    packed_half4 baseColor;      // Base color (DC term) + opacity
    packed_half3 covA;
    packed_half3 covB;
    uint shPaletteIndex;         // Index into SH palette (for SOGS)
    ushort shDegree;             // SH degree (0-3)
} SplatSH;

// Convert SplatSH to regular Splat using pre-evaluated SH
Splat evaluateSplatWithSH(SplatSH splatSH, 
                         device const float4* evaluatedSH,
                         float3 viewDirection) {
    Splat splat;
    splat.position = splatSH.position;
    splat.covA = splatSH.covA;
    splat.covB = splatSH.covB;
    
    // Use pre-evaluated SH color if available (including degree 0)
    if (splatSH.shPaletteIndex != 0) {
        float4 shColor = evaluatedSH[splatSH.shPaletteIndex];
        splat.color = packed_half4(half4(half3(shColor.rgb), half(splatSH.baseColor.a)));
    } else {
        // Fall back to base color for invalid index
        splat.color = splatSH.baseColor;
    }
    
    return splat;
}

// Fast SH vertex shader using pre-evaluated colors
vertex FragmentIn fastSHSplatVertexShader(uint vertexID [[vertex_id]],
                                         uint instanceID [[instance_id]],
                                         ushort amplificationID [[amplification_id]],
                                         constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                         constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                         device const float4* evaluatedSH [[ buffer(3) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];
    
    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }
    
    SplatSH splatSH = splatArray[splatID];
    
    // Convert to regular splat using pre-evaluated SH
    float3 viewDirection = normalize(float3(uniforms.viewMatrix[0][2], 
                                           uniforms.viewMatrix[1][2], 
                                           uniforms.viewMatrix[2][2]));
    Splat splat = evaluateSplatWithSH(splatSH, evaluatedSH, viewDirection);
    
    return splatVertex(splat, uniforms, vertexID % 4);
}

// Fragment shader with early discard optimization
fragment half4 fastSHSplatFragmentShader(FragmentIn in [[stage_in]]) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);

    // Early fragment discard for transparent fragments - saves blending work
    if (alpha < 0.01h) {
        discard_fragment();
    }

    return half4(alpha * in.color.rgb, alpha);
}

// Texture-based SH evaluation for better edge accuracy
vertex FragmentIn textureSHSplatVertexShader(uint vertexID [[vertex_id]],
                                            uint instanceID [[instance_id]],
                                            ushort amplificationID [[amplification_id]],
                                            constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                            constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                            texture2d<float> evaluatedSHTexture [[ texture(0) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];
    
    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }
    
    SplatSH splatSH = splatArray[splatID];
    Splat splat;
    splat.position = splatSH.position;
    splat.covA = splatSH.covA;
    splat.covB = splatSH.covB;
    
    // Sample pre-evaluated SH from texture
    if (splatSH.shDegree > 0 && splatSH.shPaletteIndex != 0) {
        // Convert palette index to texture coordinates
        uint textureWidth = evaluatedSHTexture.get_width();
        uint2 texCoord = uint2(splatSH.shPaletteIndex % textureWidth,
                              splatSH.shPaletteIndex / textureWidth);
        
        float4 shColor = evaluatedSHTexture.read(texCoord);
        splat.color = packed_half4(half4(half3(shColor.rgb), half(splatSH.baseColor.a)));
    } else {
        splat.color = splatSH.baseColor;
    }
    
    return splatVertex(splat, uniforms, vertexID % 4);
}