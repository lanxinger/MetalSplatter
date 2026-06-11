#include "SplatProcessing.h"

// GPU-only sorted rendering: uses sorted indices buffer to access splats in depth order
// This avoids CPU readback and reordering, keeping all sort data on GPU
vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                               constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                               const device uint *editStates [[ buffer(BufferIndexEditState) ]],
                                               const device uint *transformIndices [[ buffer(BufferIndexTransformIndex) ]],
                                               const device float4x4 *transformPalette [[ buffer(BufferIndexTransformPalette) ]]) {
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
    // sortedIndices maps logical draw order → actual splat index in buffer
    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    // Defensive check for transient stale/corrupt sorted indices.
    if (actualSplatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        out.splatID = 0;
        return out;
    }
    Splat splat = splatArray[actualSplatID];

    return splatVertex(splat,
                       uniforms,
                       vertexID % 4,
                       actualSplatID,
                       editStates[actualSplatID],
                       transformIndices,
                       transformPalette);
}

// Precomputed-path vertex shader (PlayCanvas-style pipeline): projection,
// covariance decomposition, and frustum/sub-pixel culling run once per splat in
// the batchPrecomputeSplats compute pass; this shader only expands quad corners
// from the cached data instead of redoing the projection for all 4 vertices.
// Single-view only: the precomputed data is valid for uniforms[0]'s camera.
vertex FragmentIn singleStageSplatVertexShaderPrecomputed(uint vertexID [[vertex_id]],
                                                          uint instanceID [[instance_id]],
                                                          constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                                          constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                                          constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                                          const device uint *editStates [[ buffer(BufferIndexEditState) ]],
                                                          constant PrecomputedSplat* precomputed [[ buffer(BufferIndexPrecomputed) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[0];

    FragmentIn out;
    out.position = float4(1, 1, 0, 1);
    out.relativePosition = half2(0);
    out.color = half4(0);
    out.lodBand = 0;
    out.debugFlags = uniforms.debugFlags;
    out.splatID = 0;

    uint logicalSplatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (logicalSplatID >= uniforms.splatCount) {
        return out;
    }
    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    if (actualSplatID >= uniforms.splatCount) {
        return out;
    }
    out.splatID = actualSplatID;

    PrecomputedSplat pre = precomputed[actualSplatID];
    if (pre.visible == 0) {
        return out;
    }

    uint editState = editStates[actualSplatID];
    if ((editState & ((1u << 1) | (1u << 3))) != 0u) {
        return out;
    }

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[vertexID % 4];

    half2 axisContribution = relativeCoordinates.x * half2(pre.axis1)
                           + relativeCoordinates.y * half2(pre.axis2);
    half2 projectedScreenDelta = axisContribution * (2.0h * kBoundsRadius) / half2(uniforms.screenSize);

    float4 clipPosition = pre.clipPosition;
    out.position = float4(clipPosition.x + projectedScreenDelta.x * clipPosition.w,
                          clipPosition.y + projectedScreenDelta.y * clipPosition.w,
                          clipPosition.z,
                          clipPosition.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;

    out.color = unpackSplatColor(splatArray[actualSplatID].packedColor);
    out.color.a = min(out.color.a * half(pre.opacityScale), half(1.0));
    if ((editState & (1u << 2)) != 0u) {
        constexpr half4 lockedTint = half4(1.0h, 0.72h, 0.22h, 0.65h);
        out.color.rgb = mix(out.color.rgb, lockedTint.rgb, lockedTint.a);
    }
    bool isSelected = (editState & (1u << 0)) != 0u;
    if (uniforms.editingEnabled != 0u && isSelected) {
        out.color.rgb = mix(out.color.rgb, half3(uniforms.selectionTintColor.xyz), half(uniforms.selectionTintColor.w));
    }

    if ((uniforms.debugFlags & DebugFlagLodTint) != 0) {
        float distance = pre.depth;
        float3 thresholds = uniforms.lodThresholds;
        if (distance > thresholds.z) {
            out.lodBand = 3;
        } else if (distance > thresholds.y) {
            out.lodBand = 2;
        } else if (distance > thresholds.x) {
            out.lodBand = 1;
        }
    }

    return out;
}

fragment half4 singleStageSplatFragmentShader(FragmentIn in [[stage_in]]) {
    return shadeSplat(in);
}

// Dithered (stochastic) transparency fragment shader.
// Uses order-independent transparency via stochastic alpha testing.
// No sorting required - best paired with TAA for noise reduction.
fragment half4 singleStageSplatFragmentShaderDithered(FragmentIn in [[stage_in]]) {
    return shadeSplatDithered(in, in.position.xy);
}

vertex FragmentIn selectedOutlineVertexShader(uint vertexID [[vertex_id]],
                                              uint instanceID [[instance_id]],
                                              ushort amplificationID [[amplification_id]],
                                              constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                              constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                              const device uint *editStates [[ buffer(BufferIndexEditState) ]],
                                              const device uint *transformIndices [[ buffer(BufferIndexTransformIndex) ]],
                                              const device float4x4 *transformPalette [[ buffer(BufferIndexTransformPalette) ]]) {
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

    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    if (actualSplatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        out.splatID = 0;
        return out;
    }

    Splat splat = splatArray[actualSplatID];
    return selectedOutlineVertex(splat,
                                 uniforms,
                                 vertexID % 4,
                                 actualSplatID,
                                 editStates[actualSplatID],
                                 transformIndices,
                                 transformPalette);
}

fragment half4 selectedOutlineFragmentShader(FragmentIn in [[stage_in]]) {
    return shadeSplatOutline(in);
}
