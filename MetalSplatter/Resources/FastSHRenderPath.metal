#include "ShaderCommon.h"
#include "SplatProcessing.h"

// Forward declaration from spherical_harmonics_evaluate.metal
float4 evaluateSH(float3 dir, device const float3* sh_coeffs, uint degree);
// Coefficients follow the Graphdeco/gsplat ordering documented in spherical_harmonics_evaluate.metal.

struct FastSHParams {
    uint coeffsPerEntry;
    uint paletteSize;
    uint degree;
    uint padding;
};

// Extended Splat structure for SH support
typedef struct {
    packed_float3 position;
    packed_half4 baseColor;      // Base color (DC term) + opacity
    float4 rotation;             // Quaternion (x,y,z,w)
    packed_half3 covA;
    packed_half3 covB;
    uint shPaletteIndex;         // Index into SH palette (for SOGS)
    ushort shDegree;             // SH degree (0-3)
    ushort padding;
} SplatSH;

// Convert SplatSH to regular Splat using pre-evaluated SH
inline float3 cameraWorldPosition(matrix_float4x4 viewMatrix) {
    float3x3 rotation = float3x3(viewMatrix[0].xyz,
                                 viewMatrix[1].xyz,
                                 viewMatrix[2].xyz);
    float3 translation = viewMatrix[3].xyz;
    return -(transpose(rotation) * translation);
}

inline float3 rotateVectorByQuaternion(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    float3 t = 2.0f * cross(u, v);
    return v + s * t + cross(u, t);
}

Splat evaluateSplatWithSH(SplatSH splatSH,
                         Uniforms uniforms,
                         device const float3* shPalette,
                         constant FastSHParams& params) {
    Splat splat;
    splat.position = splatSH.position;
    splat.covA = splatSH.covA;
    splat.covB = splatSH.covB;

    const uint invalidIndex = 0xffffffffu;
    bool hasSH = (splatSH.shDegree > 0) &&
                 (params.coeffsPerEntry > 0) &&
                 (splatSH.shPaletteIndex != invalidIndex) &&
                 (splatSH.shPaletteIndex < params.paletteSize);

    if (hasSH) {
        device const float3* coeffs = shPalette + params.coeffsPerEntry * splatSH.shPaletteIndex;

    float3 worldPosition = float3(splatSH.position);
    float3 cameraPosition = cameraWorldPosition(uniforms.viewMatrix);
    float3 viewDirection = normalize(cameraPosition - worldPosition);

    // Rotate view direction into the Gaussian's local frame
    float4 q = splatSH.rotation;
    float4 qConjugate = float4(-q.xyz, q.w);

    float3 localDirection = normalize(rotateVectorByQuaternion(qConjugate, viewDirection));

        float4 shColor = evaluateSH(localDirection, coeffs, params.degree);
        float3 rgb = shColor.rgb;
        splat.color = packed_half4(half4(half3(rgb), splatSH.baseColor.w));
    } else {
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
                                         device const float3* shPalette [[ buffer(3) ]],
                                         constant FastSHParams& params [[ buffer(4) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];
    
    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        return out;
    }
    
    SplatSH splatSH = splatArray[splatID];
    // Evaluate in Gaussian local frame
    Splat splat = evaluateSplatWithSH(splatSH, uniforms, shPalette, params);
    
    return splatVertex(splat, uniforms, vertexID % 4);
}

// Fragment shader remains the same
fragment half4 fastSHSplatFragmentShader(FragmentIn in [[stage_in]]) {
    return shadeSplat(in);
}

// Texture-based SH evaluation for better edge accuracy
vertex FragmentIn textureSHSplatVertexShader(uint vertexID [[vertex_id]],
                                            uint instanceID [[instance_id]],
                                            ushort amplificationID [[amplification_id]],
                                            constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                            constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                            device const float3* shPalette [[ buffer(3) ]],
                                            constant FastSHParams& params [[ buffer(4) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];
    
    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        return out;
    }
    
    SplatSH splatSH = splatArray[splatID];
    Splat splat = evaluateSplatWithSH(splatSH, uniforms, shPalette, params);
    return splatVertex(splat, uniforms, vertexID % 4);
}
