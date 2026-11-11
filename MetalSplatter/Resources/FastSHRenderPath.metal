#include "ShaderCommon.h"
#include "SplatProcessing.h"

// Forward declarations from spherical_harmonics_evaluate.metal
// Legacy version with runtime degree parameter (float precision)
float4 evaluateSH(float3 dir, device const float3* sh_coeffs, uint degree);
// Specialized version using function constants (float precision)
float4 evaluateSHSpecialized(float3 dir, device const float3* sh_coeffs);
// Half-precision versions — read half3 palette, compute in half
half4 evaluateSHHalf(float3 dir, half3 sh_dc, device const half3* sh_bands, uint degree);
half4 evaluateSHSpecializedHalf(float3 dir, device const half3* sh_coeffs);
// Coefficients follow the Graphdeco/gsplat ordering documented in spherical_harmonics_evaluate.metal.

// Import function constants from spherical_harmonics_evaluate.metal
// These are shared across all shaders that include this file
extern constant uint SH_DEGREE [[function_constant(0)]];
extern constant bool HAS_SH_BAND_1;
extern constant bool HAS_SH_BAND_2;
extern constant bool HAS_SH_BAND_3;

constant constexpr int FastSHBufferIndexPalette = 7;
constant constexpr int FastSHBufferIndexParams = 8;

struct FastSHParams {
    uint coeffsPerEntry;
    uint paletteSize;
    uint degree;
    uint skipSHEvaluation; // Explicit DC-only mode; evaluated colors are not cached
};

// Extended Splat structure for SH support
typedef struct {
    packed_float3 position;
    uint packedBaseColor;        // Base color (DC term) + opacity, RGBA8 unorm
    float4 tintColor;            // Animation tint/alpha multiplier, display-encoded
    float4 rotation;             // Quaternion (x,y,z,w)
    packed_half3 covA;
    packed_half3 covB;
    uint shPaletteIndex;         // Index into SH palette (for SOGS)
    ushort shDegree;             // SH degree (0-3)
    packed_half3 shDC;           // DC coefficient, which the palette doesn't hold
    float linearOpacity;
} SplatSH;

// SH is expressed in the scene frame, independently of the covariance quaternion.
// A model/edit transform carries appearance with the object. Normalize its columns
// to remove scale before taking the camera-to-surface direction back to that frame.
inline float3 sphericalHarmonicsDirection(float3 position,
                                         float4x4 appearanceViewMatrix,
                                         bool isOrthographic) {
    float3x3 rotation = float3x3(normalize(appearanceViewMatrix[0].xyz),
                               normalize(appearanceViewMatrix[1].xyz),
                               normalize(appearanceViewMatrix[2].xyz));
    float3 viewDirection = isOrthographic
        ? float3(0, 0, -1)
        : (appearanceViewMatrix * float4(position, 1)).xyz;
    return normalize(transpose(rotation) * viewDirection);
}

// The colour a splat draws with, display-encoded. SH colour isn't packed into RGBA8 or limited to 1:
// the trainer floors it at 0 and nothing more, so a faint splat may carry a value above 1, and clamping
// those dims a trained model's bright areas.
half4 fastSHSplatColor(SplatSH splatSH,
                       Uniforms uniforms,
                       float4x4 appearanceViewMatrix,
                       device const half3* shPalette,
                       constant FastSHParams& params) {
    half4 baseColor = unpackSplatColor(splatSH.packedBaseColor);
    baseColor.a = half(splatSH.linearOpacity);
    half4 tintColor = half4(splatSH.tintColor);

    const uint invalidIndex = 0xffffffffu;
    // Explicitly disabled SH uses the original base color, not a cached SH result.
    bool hasSH = (params.skipSHEvaluation == 0) &&
                 (splatSH.shDegree > 0) &&
                 (params.coeffsPerEntry > 0) &&
                 (splatSH.shPaletteIndex != invalidIndex) &&
                 (splatSH.shPaletteIndex < params.paletteSize);
    if (!hasSH) {
        return baseColor * tintColor;
    }

    device const half3* bands = shPalette + params.coeffsPerEntry * splatSH.shPaletteIndex;
    float3 direction = sphericalHarmonicsDirection(float3(splatSH.position),
                                                    appearanceViewMatrix,
                                                    uniforms.isOrthographic != 0);
    half4 shColor = evaluateSHHalf(direction, half3(splatSH.shDC), bands, params.degree);
    // Preserve original alpha from base color, combine with SH-evaluated RGB
    return half4(shColor.rgb * tintColor.rgb, baseColor.a * tintColor.a);
}

// Fast SH vertex shader evaluating view-dependent colors per splat
// GPU-only sorted rendering: uses sorted indices buffer to access splats in depth order
inline FragmentIn fastSHSplatVertex(uint vertexID,
                                    uint instanceID,
                                    ushort amplificationID,
                                    constant SplatSH* splatArray,
                                    constant UniformsArray &uniformsArray,
                                    constant int32_t* sortedIndices,
                                    const device uchar *editStates,
                                    const device ushort *transformIndices,
                                    const device float4x4 *transformPalette,
                                    device const half3* shPalette,
                                    constant FastSHParams& params) {
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

    // Use sorted index to access splat in depth-sorted order
    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    // Defensive guard for transient stale/corrupt sorted indices
    if (actualSplatID >= uniforms.totalSplatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        out.splatID = 0;
        return out;
    }
    SplatSH splatSH = splatArray[actualSplatID];
    float4x4 appearanceViewMatrix = uniforms.viewMatrix;
    if (transformIndices != nullptr && transformPalette != nullptr) {
        uint transformIndex = transformIndices[actualSplatID];
        if (transformIndex != 0u) {
            appearanceViewMatrix = appearanceViewMatrix * transformPalette[transformIndex];
        }
    }
    Splat splat;
    splat.position = splatSH.position;
    splat.packedColor = 0;
    splat.linearOpacity = splatSH.linearOpacity;
    splat.covA = splatSH.covA;
    splat.covB = splatSH.covB;
    half4 color = fastSHSplatColor(splatSH, uniforms, appearanceViewMatrix, shPalette, params);
    uint editState = editStates != nullptr ? editStates[actualSplatID] : 0u;

    return splatVertex(splat,
                       color,
                       uniforms,
                       vertexID % 4,
                       actualSplatID,
                       editState,
                       transformIndices,
                       transformPalette);
}

vertex FragmentIn fastSHSplatVertexShader(uint vertexID [[vertex_id]],
                                          uint instanceID [[instance_id]],
                                          ushort amplificationID [[amplification_id]],
                                          constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                          constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                          constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                          device const half3* shPalette [[ buffer(FastSHBufferIndexPalette) ]],
                                          constant FastSHParams& params [[ buffer(FastSHBufferIndexParams) ]]) {
    return fastSHSplatVertex(vertexID,
                             instanceID,
                             amplificationID,
                             splatArray,
                             uniformsArray,
                             sortedIndices,
                             nullptr,
                             nullptr,
                             nullptr,
                             shPalette,
                             params);
}

vertex FragmentIn fastSHSplatVertexShaderEditing(uint vertexID [[vertex_id]],
                                                 uint instanceID [[instance_id]],
                                                 ushort amplificationID [[amplification_id]],
                                                 constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                                 constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                                 constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                                 const device uchar *editStates [[ buffer(BufferIndexEditState) ]],
                                                 const device ushort *transformIndices [[ buffer(BufferIndexTransformIndex) ]],
                                                 const device float4x4 *transformPalette [[ buffer(BufferIndexTransformPalette) ]],
                                                 device const half3* shPalette [[ buffer(FastSHBufferIndexPalette) ]],
                                                 constant FastSHParams& params [[ buffer(FastSHBufferIndexParams) ]]) {
    return fastSHSplatVertex(vertexID,
                             instanceID,
                             amplificationID,
                             splatArray,
                             uniformsArray,
                             sortedIndices,
                             editStates,
                             transformIndices,
                             transformPalette,
                             shPalette,
                             params);
}

// Fragment shader remains the same
fragment half4 fastSHSplatFragmentShader(FragmentIn in [[stage_in]]) {
    return shadeSplat(in);
}

// Texture-based SH evaluation for better edge accuracy
// GPU-only sorted rendering: uses sorted indices buffer to access splats in depth order
vertex FragmentIn textureSHSplatVertexShader(uint vertexID [[vertex_id]],
                                            uint instanceID [[instance_id]],
                                            ushort amplificationID [[amplification_id]],
                                            constant SplatSH* splatArray [[ buffer(BufferIndexSplat) ]],
                                            constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                            constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                            const device uchar *editStates [[ buffer(BufferIndexEditState) ]],
                                            const device ushort *transformIndices [[ buffer(BufferIndexTransformIndex) ]],
                                            const device float4x4 *transformPalette [[ buffer(BufferIndexTransformPalette) ]],
                                            device const half3* shPalette [[ buffer(FastSHBufferIndexPalette) ]],
                                            constant FastSHParams& params [[ buffer(FastSHBufferIndexParams) ]]) {
    return fastSHSplatVertex(vertexID,
                             instanceID,
                             amplificationID,
                             splatArray,
                             uniformsArray,
                             sortedIndices,
                             editStates,
                             transformIndices,
                             transformPalette,
                             shPalette,
                             params);
}
