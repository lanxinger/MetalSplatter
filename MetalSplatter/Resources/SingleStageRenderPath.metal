#include "SplatProcessing.h"

// GPU-only sorted rendering: uses sorted indices buffer to access splats in depth order
// This avoids CPU readback and reordering, keeping all sort data on GPU
vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                               constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]],
                                               constant PackedColor* packedColors [[ buffer(BufferIndexPackedColors) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    uint logicalSplatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (logicalSplatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        out.relativePosition = half2(0);
        out.color = half4(0);
        out.lodBand = 0;
        out.debugFlags = 0;
        return out;
    }

    // Use sorted index to access splat in depth-sorted order
    // sortedIndices maps logical draw order → actual splat index in buffer
    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    Splat splat = splatArray[actualSplatID];

    // Override color with packed buffer if enabled (via function constant)
    splat.color = getSplatColor(actualSplatID, splatArray, packedColors);

    return splatVertex(splat, uniforms, vertexID % 4);
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
