#include <metal_stdlib>

using namespace metal;

struct SceneOcclusionUniforms {
    float4x4 inverseProjection;
    float3x3 viewToDepthTransform;
    uint2 viewportSize;
    float occlusionBias;
    float confidenceThreshold;
    float2 padding;
};

inline float3 sampleCameraColor(texture2d<float, access::sample> cameraY,
                                texture2d<float, access::sample> cameraCbCr,
                                sampler linearSampler,
                                float2 uv) {
    float y = cameraY.sample(linearSampler, uv).r;
    float2 cbcr = cameraCbCr.sample(linearSampler, uv).rg - float2(0.5, 0.5);

    float3 rgb;
    rgb.r = y + 1.5748 * cbcr.y;
    rgb.g = y - 0.1873 * cbcr.x - 0.4681 * cbcr.y;
    rgb.b = y + 1.8556 * cbcr.x;
    return saturate(rgb);
}

kernel void gaussian_splat_scene_occlusion(
    texture2d<float, access::sample> sceneDepthTexture        [[texture(0)]],
    texture2d<float, access::sample> sceneConfidenceTexture   [[texture(1)]],
    texture2d<float, access::sample> splatDepthTexture        [[texture(2)]],
    texture2d<float, access::read_write> colorTexture         [[texture(3)]],
    texture2d<float, access::sample> cameraTextureY           [[texture(4)]],
    texture2d<float, access::sample> cameraTextureCbCr        [[texture(5)]],
    constant SceneOcclusionUniforms& uniforms                 [[buffer(0)]],
    sampler linearSampler                                     [[sampler(0)]],
    uint2 gid                                                 [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.viewportSize.x || gid.y >= uniforms.viewportSize.y) {
        return;
    }

    float virtualDepthSample = splatDepthTexture.read(gid).r;
    if (virtualDepthSample <= 0.0f || virtualDepthSample >= 1.0f) {
        return;
    }

    float2 viewportSize = float2(uniforms.viewportSize);
    float2 uv = (float2(gid) + 0.5f) / viewportSize;

    float2 clipXY = uv * 2.0f - 1.0f;
    float clipZ = virtualDepthSample * 2.0f - 1.0f;
    float4 clipPosition = float4(clipXY, clipZ, 1.0f);
    float4 viewPosition = uniforms.inverseProjection * clipPosition;
    viewPosition /= viewPosition.w;
    float virtualDepthMeters = -viewPosition.z;
    if (virtualDepthMeters <= 0.0f) {
        return;
    }

    float3 depthUVH = uniforms.viewToDepthTransform * float3(uv, 1.0f);
    float2 depthUV = depthUVH.xy;
    if (any(depthUV < float2(0.0f)) || any(depthUV > float2(1.0f))) {
        return;
    }

    float sceneDepthMeters = sceneDepthTexture.sample(linearSampler, depthUV).r;
    if (sceneDepthMeters <= 0.0f) {
        return;
    }

    float confidence = sceneConfidenceTexture.sample(linearSampler, depthUV).r;
    if (confidence < uniforms.confidenceThreshold) {
        return;
    }

    if (sceneDepthMeters + uniforms.occlusionBias < virtualDepthMeters) {
        float3 cameraColor = sampleCameraColor(cameraTextureY, cameraTextureCbCr, linearSampler, depthUV);
        colorTexture.write(float4(cameraColor, 1.0f), gid);
    }
}
