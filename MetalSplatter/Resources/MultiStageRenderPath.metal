#include "SplatProcessing.h"

typedef struct
{
    half4 color [[raster_order_group(0)]];
    float depth [[raster_order_group(0)]];
} FragmentValues;

typedef struct
{
    FragmentValues values [[imageblock_data]];
} FragmentStore;

typedef struct
{
    half4 color [[color(0)]];
    float depth [[depth(any)]];
} FragmentOut;

[[kernel, max_total_threads_per_threadgroup(1024)]]
kernel void initializeFragmentStore(imageblock<FragmentValues, imageblock_layout_explicit> blockData,
                                    ushort2 localThreadID [[thread_position_in_threadgroup]],
                                    ushort2 threadgroupID [[threadgroup_position_in_grid]]) {
    threadgroup_imageblock FragmentValues *values = blockData.data(localThreadID);
    
    // Optimized initialization with SIMD-friendly operations
    values->color = half4(0);
    values->depth = 0.0f;
}

// GPU-only sorted rendering: uses sorted indices buffer to access splats in depth order
vertex FragmentIn multiStageSplatVertexShader(uint vertexID [[vertex_id]],
                                              uint instanceID [[instance_id]],
                                              ushort amplificationID [[amplification_id]],
                                              constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                              constant int32_t* sortedIndices [[ buffer(BufferIndexSortedIndices) ]]) {
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
    uint actualSplatID = uint(sortedIndices[logicalSplatID]);
    Splat splat = splatArray[actualSplatID];

    return splatVertex(splat, uniforms, vertexID % 4);
}

fragment FragmentStore multiStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                                     FragmentValues previousFragmentValues [[imageblock_data]]) {
    FragmentStore out;

    half4 shaded = shadeSplat(in);
    half alpha = shaded.a;
    half4 colorWithPremultipliedAlpha = half4(shaded.rgb, alpha);

    half oneMinusAlpha = 1 - alpha;

    half4 previousColor = previousFragmentValues.color;
    out.values.color = previousColor * oneMinusAlpha + colorWithPremultipliedAlpha;

    float previousDepth = previousFragmentValues.depth;
    float depth = in.position.z;
    out.values.depth = previousDepth * oneMinusAlpha + depth * alpha;

    return out;
}

/// Generate a single triangle covering the entire screen
vertex FragmentIn postprocessVertexShader(uint vertexID [[vertex_id]]) {
    FragmentIn out;

    float4 position;
    position.x = (vertexID == 2) ? 3.0 : -1.0;
    position.y = (vertexID == 0) ? -3.0 : 1.0;
    position.zw = 1.0;

    out.position = position;
    out.relativePosition = half2(0);
    out.color = half4(0);
    out.lodBand = 0;
    out.debugFlags = 0;
    return out;
}

fragment FragmentOut postprocessFragmentShader(FragmentValues fragmentValues [[imageblock_data]]) {
    FragmentOut out;
    out.depth = (fragmentValues.color.a == 0) ? 0 : fragmentValues.depth / fragmentValues.color.a;
    out.color = fragmentValues.color;
    return out;
}

fragment half4 postprocessFragmentShaderNoDepth(FragmentValues fragmentValues [[imageblock_data]]) {
    return fragmentValues.color;
}
