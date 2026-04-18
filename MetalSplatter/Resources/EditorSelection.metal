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

inline float4x4 editorTransformForIndex(const device uint *transformIndices,
                                        const device float4x4 *transformPalette,
                                        uint index) {
    uint transformIndex = transformIndices[index];
    return transformIndex == 0u ? float4x4(1.0f) : transformPalette[transformIndex];
}

kernel void selectEditableSplats(constant Splat *splats [[buffer(0)]],
                                 const device uint *states [[buffer(1)]],
                                 const device uint *transformIndices [[buffer(2)]],
                                 const device float4x4 *transformPalette [[buffer(3)]],
                                 device uchar *outputMask [[buffer(4)]],
                                 constant SelectionQueryParameters &params [[buffer(5)]],
                                 texture2d<float, access::read> maskTexture [[texture(0)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= params.splatCount) {
        return;
    }

    uint state = states[gid];
    if ((state & (EditableSplatStateHidden | EditableSplatStateLocked | EditableSplatStateDeleted)) != 0u) {
        outputMask[gid] = 0;
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
            float4 clipPosition = params.projectionMatrix * (params.viewMatrix * float4(worldPosition, 1.0));
            if (abs(clipPosition.w) > kDivisionEpsilon) {
                float2 ndc = clipPosition.xy / clipPosition.w;
                float2 normalized = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);

                if (all(normalized >= 0.0f) && all(normalized <= 1.0f)) {
                    if (params.mode == SelectionModePoint) {
                        selected = distance(normalized, params.point) <= params.pointRadius;
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

    outputMask[gid] = selected ? 1 : 0;
}
