#include "SplatProcessing.h"

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
    uint2 screenSize;
    float focalX;
    float focalY;
    float tanHalfFovX;
    float tanHalfFovY;
    float covarianceBlur;
    uint renderMode;
    uint2 padding3;
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

inline float visibleFootprintExtent(float3x3 covariance3D, float3 planeNormal) {
    float normalLength = length(planeNormal);
    float3 normalizedPlaneNormal = normalLength > kDivisionEpsilon
        ? planeNormal / normalLength
        : float3(0.0f, 1.0f, 0.0f);
    float variance = dot(normalizedPlaneNormal, covariance3D * normalizedPlaneNormal);
    return 3.0f * sqrt(max(variance, 0.0f));
}

inline bool rectContainsCircle(float2 center, float radius, float4 rect) {
    return center.x + radius >= rect.x &&
           center.y + radius >= rect.y &&
           center.x - radius <= rect.z &&
           center.y - radius <= rect.w;
}

inline bool maskContainsProjectedCircle(texture2d<float, access::read> maskTexture,
                                        constant SelectionQueryParameters &params,
                                        float2 normalized,
                                        float radius) {
    if (!all(params.maskSize > 0u)) {
        return false;
    }

    const float2 offsets[9] = {
        float2(0.0f, 0.0f),
        float2(radius, 0.0f),
        float2(-radius, 0.0f),
        float2(0.0f, radius),
        float2(0.0f, -radius),
        float2(radius, radius),
        float2(radius, -radius),
        float2(-radius, radius),
        float2(-radius, -radius)
    };

    for (uint sampleIndex = 0; sampleIndex < 9; ++sampleIndex) {
        float2 sample = clamp(normalized + offsets[sampleIndex], 0.0f, 0.999999f);
        uint2 pixel = uint2(
            min(uint(sample.x * float(params.maskSize.x)), params.maskSize.x - 1),
            min(uint(sample.y * float(params.maskSize.y)), params.maskSize.y - 1)
        );
        if (maskTexture.read(pixel).r > 0.5f) {
            return true;
        }
    }

    return false;
}

inline void editableWorldState(constant Splat &splat,
                               const device uint *transformIndices,
                               const device float4x4 *transformPalette,
                               uint index,
                               thread float3 &worldPosition,
                               thread packed_half3 &covA,
                               thread packed_half3 &covB) {
    worldPosition = float3(splat.position);
    covA = splat.covA;
    covB = splat.covB;

    uint transformIndex = transformIndices[index];
    if (transformIndex == 0u) {
        return;
    }

    float4x4 editTransform = transformPalette[transformIndex];
    worldPosition = (editTransform * float4(worldPosition, 1.0f)).xyz;

    float3x3 transform3x3 = float3x3(editTransform[0].xyz, editTransform[1].xyz, editTransform[2].xyz);
    float3x3 covariance3D = float3x3(
        covA.x, covA.y, covA.z,
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z
    );
    float3x3 transformedCovariance = transform3x3 * covariance3D * transpose(transform3x3);
    covA = packed_half3(half(transformedCovariance[0][0]),
                        half(transformedCovariance[0][1]),
                        half(transformedCovariance[0][2]));
    covB = packed_half3(half(transformedCovariance[1][1]),
                        half(transformedCovariance[1][2]),
                        half(transformedCovariance[2][2]));
}

inline bool projectEditableSplat(float3 worldPosition,
                                 packed_half3 covA,
                                 packed_half3 covB,
                                 constant SelectionQueryParameters &params,
                                 thread float2 &normalized,
                                 thread float &distanceSquared,
                                 thread float &projectedRadius) {
    float4 viewPosition4 = params.viewMatrix * float4(worldPosition, 1.0);
    float3 viewPosition = viewPosition4.xyz;
    if (viewPosition.z >= 0.0f) {
        return false;
    }

    float4 clipPosition = params.projectionMatrix * float4(viewPosition, 1.0);
    if (abs(clipPosition.w) <= kDivisionEpsilon) {
        return false;
    }

    float2 ndc = clipPosition.xy / clipPosition.w;
    normalized = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);

    float opacityScale = 1.0f;
    float3 cov2D = calcCovariance2D(viewPosition,
                                    covA,
                                    covB,
                                    params.viewMatrix,
                                    params.focalX,
                                    params.focalY,
                                    params.tanHalfFovX,
                                    params.tanHalfFovY,
                                    params.covarianceBlur,
                                    params.renderMode,
                                    opacityScale);
    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);
    float pixelRadius = max(length(axis1 + axis2), length(axis1 - axis2)) * kBoundsRadius;
    projectedRadius = max(pixelRadius / float(min(params.screenSize.x, params.screenSize.y)), 0.008f);

    if (normalized.x + projectedRadius < 0.0f ||
        normalized.y + projectedRadius < 0.0f ||
        normalized.x - projectedRadius > 1.0f ||
        normalized.y - projectedRadius > 1.0f) {
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

    float3 worldPosition;
    packed_half3 covA;
    packed_half3 covB;
    editableWorldState(splats[gid], transformIndices, transformPalette, gid, worldPosition, covA, covB);

    bool selected = false;

    switch (params.mode) {
        case SelectionModePoint:
        case SelectionModeRect:
        case SelectionModeMask: {
            float2 normalized;
            float distanceSquared = 0.0f;
            float projectedRadius = 0.0f;
            if (projectEditableSplat(worldPosition, covA, covB, params, normalized, distanceSquared, projectedRadius)) {
                if (params.mode == SelectionModePoint) {
                    float effectiveRadius = params.pointRadius + projectedRadius;
                    selected = distanceSquared <= effectiveRadius * effectiveRadius;
                } else if (params.mode == SelectionModeRect) {
                    selected = rectContainsCircle(normalized, projectedRadius, params.rect);
                } else {
                    selected = maskContainsProjectedCircle(maskTexture, params, normalized, projectedRadius);
                }
            }
            break;
        }
        case SelectionModeSphere: {
            float3x3 covariance3D = float3x3(
                covA.x, covA.y, covA.z,
                covA.y, covB.x, covB.y,
                covA.z, covB.y, covB.z
            );
            float extent = visibleFootprintExtent(covariance3D, worldPosition - params.sphereCenter);
            selected = distance(worldPosition, params.sphereCenter) <= params.sphereRadius + extent;
            break;
        }
        case SelectionModeBox: {
            float3x3 covariance3D = float3x3(
                covA.x, covA.y, covA.z,
                covA.y, covB.x, covB.y,
                covA.z, covB.y, covB.z
            );
            float extentX = visibleFootprintExtent(covariance3D, float3(1, 0, 0));
            float extentY = visibleFootprintExtent(covariance3D, float3(0, 1, 0));
            float extentZ = visibleFootprintExtent(covariance3D, float3(0, 0, 1));
            float3 delta = abs(worldPosition - params.boxCenter);
            selected = all(delta <= params.boxExtents + float3(extentX, extentY, extentZ));
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
            float3 worldPosition;
            packed_half3 covA;
            packed_half3 covB;
            editableWorldState(splats[gid], transformIndices, transformPalette, gid, worldPosition, covA, covB);
            float2 normalized;
            float distanceSquared = 0.0f;
            float projectedRadius = 0.0f;
            if (projectEditableSplat(worldPosition, covA, covB, params, normalized, distanceSquared, projectedRadius)) {
                float effectiveRadius = params.pointRadius + projectedRadius;
                if (distanceSquared <= effectiveRadius * effectiveRadius) {
                    bestDistanceSquared = distanceSquared;
                    bestIndex = int(gid);
                }
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
