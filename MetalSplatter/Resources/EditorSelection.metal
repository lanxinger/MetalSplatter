#include "ShaderCommon.h"

using namespace metal;

constant uint EditableSplatStateHidden = 1u << 1;
constant uint EditableSplatStateLocked = 1u << 2;
constant uint EditableSplatStateDeleted = 1u << 3;

constant uint SelectionModePoint = 0u;
constant uint SelectionModeRect = 1u;
constant uint SelectionModeMask = 2u;
constant uint SelectionModeSphere = 3u;
constant uint SelectionModeBox = 4u;

typedef struct
{
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float2 point;
    float pointRadius;
    uint mode;
    uint splatCount;
    float4 rect;
    float3 sphereCenter;
    float sphereRadius;
    float3 boxCenter;
    float padding0;
    float3 boxExtents;
    float padding1;
    uint2 maskSize;
    uint2 padding2;
} SelectionQueryParameters;

typedef struct
{
    int index;
    float distanceSquared;
} NearestSelectionCandidate;

inline float4x4 editorTransformForIndex(const device uint *transformIndices,
                                        const device float4x4 *transformPalette,
                                        uint index) {
    uint transformIndex = transformIndices[index];
    return transformIndex == 0u ? float4x4(1.0f) : transformPalette[transformIndex];
}

inline bool isSelectableEditorState(uint state) {
    return (state & (EditableSplatStateHidden | EditableSplatStateLocked | EditableSplatStateDeleted)) == 0u;
}

inline bool projectEditableSplat(float3 worldPosition,
                                 constant SelectionQueryParameters &params,
                                 thread float2 &normalized,
                                 thread float &distanceSquared) {
    float4 clipPosition = params.projectionMatrix * (params.viewMatrix * float4(worldPosition, 1.0));
    if (abs(clipPosition.w) <= kDivisionEpsilon) {
        return false;
    }

    float2 ndc = clipPosition.xy / clipPosition.w;
    normalized = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    if (!all(normalized >= 0.0f) || !all(normalized <= 1.0f)) {
        return false;
    }

    float2 delta = normalized - params.point;
    distanceSquared = dot(delta, delta);
    return true;
}

kernel void selectEditableSplats(constant Splat *splats [[buffer(0)]],
                                 const device uint *states [[buffer(1)]],
                                 const device uint *transformIndices [[buffer(2)]],
                                 const device float4x4 *transformPalette [[buffer(3)]],
                                 device uint *outputIndices [[buffer(4)]],
                                 device atomic_uint *outputCount [[buffer(5)]],
                                 constant SelectionQueryParameters &params [[buffer(6)]],
                                 texture2d<float, access::read> maskTexture [[texture(0)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= params.splatCount) {
        return;
    }

    uint state = states[gid];
    if (!isSelectableEditorState(state)) {
        return;
    }

    float4x4 editTransform = editorTransformForIndex(transformIndices, transformPalette, gid);
    float4 worldPosition4 = editTransform * float4(float3(splats[gid].position), 1.0);
    float3 worldPosition = worldPosition4.xyz;

    bool selected = false;

    switch (params.mode) {
        case SelectionModePoint:
        case SelectionModeRect:
        case SelectionModeMask: {
            float2 normalized;
            float distanceSquared = 0.0f;
            if (projectEditableSplat(worldPosition, params, normalized, distanceSquared)) {
                if (params.mode == SelectionModePoint) {
                    selected = distanceSquared <= params.pointRadius * params.pointRadius;
                } else if (params.mode == SelectionModeRect) {
                    selected = normalized.x >= params.rect.x &&
                               normalized.y >= params.rect.y &&
                               normalized.x <= params.rect.z &&
                               normalized.y <= params.rect.w;
                } else if (all(params.maskSize > 0u)) {
                    uint2 pixel = uint2(
                        min(uint(normalized.x * float(params.maskSize.x)), params.maskSize.x - 1),
                        min(uint(normalized.y * float(params.maskSize.y)), params.maskSize.y - 1)
                    );
                    selected = maskTexture.read(pixel).r > 0.5f;
                }
            }
            break;
        }
        case SelectionModeSphere: {
            selected = distance(worldPosition, params.sphereCenter) <= params.sphereRadius;
            break;
        }
        case SelectionModeBox: {
            float3 delta = abs(worldPosition - params.boxCenter);
            selected = all(delta <= params.boxExtents);
            break;
        }
        default:
            selected = false;
            break;
    }

    if (selected) {
        uint outputIndex = atomic_fetch_add_explicit(outputCount, 1, memory_order_relaxed);
        outputIndices[outputIndex] = gid;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void pickNearestEditableSplat(constant Splat *splats [[buffer(0)]],
                                     const device uint *states [[buffer(1)]],
                                     const device uint *transformIndices [[buffer(2)]],
                                     const device float4x4 *transformPalette [[buffer(3)]],
                                     device NearestSelectionCandidate *candidates [[buffer(4)]],
                                     constant SelectionQueryParameters &params [[buffer(5)]],
                                     uint gid [[thread_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint threadCount [[threads_per_threadgroup]]) {
    threadgroup float localDistances[256];
    threadgroup int localIndices[256];

    float bestDistanceSquared = INFINITY;
    int bestIndex = -1;

    if (gid < params.splatCount) {
        uint state = states[gid];
        if (isSelectableEditorState(state)) {
            float4x4 editTransform = editorTransformForIndex(transformIndices, transformPalette, gid);
            float3 worldPosition = (editTransform * float4(float3(splats[gid].position), 1.0)).xyz;
            float2 normalized;
            float distanceSquared = 0.0f;
            if (projectEditableSplat(worldPosition, params, normalized, distanceSquared) &&
                distanceSquared <= params.pointRadius * params.pointRadius) {
                bestDistanceSquared = distanceSquared;
                bestIndex = int(gid);
            }
        }
    }

    localDistances[tid] = bestDistanceSquared;
    localIndices[tid] = bestIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint reductionWidth = 1;
    while (reductionWidth * 2 <= threadCount) {
        reductionWidth *= 2;
    }

    for (uint stride = reductionWidth / 2; stride > 0; stride >>= 1) {
        if (tid < stride && localDistances[tid + stride] < localDistances[tid]) {
            localDistances[tid] = localDistances[tid + stride];
            localIndices[tid] = localIndices[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        candidates[tgid].index = localIndices[0];
        candidates[tgid].distanceSquared = localDistances[0];
    }
}
