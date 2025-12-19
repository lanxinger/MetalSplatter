#import "ShaderCommon.h"

// Debug flags
constant uint DebugFlagOverdraw = 1u;
constant uint DebugFlagLodTint = 2u;

// Render mode flags (passed via uniforms.renderModeFlags)
constant uint RenderModeDitheredTransparency = 1u;

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex);

inline float unpackUnorm(uint value, uint bits) {
    uint mask = (1u << bits) - 1u;
    return float(value & mask) / float(mask);
}

inline float3 unpack111011(uint value) {
    float x = unpackUnorm(value >> 21, 11);
    float y = unpackUnorm(value >> 11, 10);
    float z = unpackUnorm(value, 11);
    return float3(x, y, z);
}

inline float4 unpackRotation2101010(uint value) {
    float norm = sqrt(2.0f);
    float a = (unpackUnorm(value >> 20, 10) - 0.5f) * norm;
    float b = (unpackUnorm(value >> 10, 10) - 0.5f) * norm;
    float c = (unpackUnorm(value, 10) - 0.5f) * norm;
    float m = sqrt(max(0.0f, 1.0f - (a * a + b * b + c * c)));

    switch (value >> 30) {
        case 0: return float4(a, b, c, m); // w is largest
        case 1: return float4(m, b, c, a); // x is largest
        case 2: return float4(b, m, c, a); // y is largest
        default: return float4(b, c, m, a); // z is largest
    }
}

inline float3 decodePackedPosition(PackedSplat packed, PackedSplatChunk chunk) {
    float3 t = unpack111011(packed.data.x);
    float3 minPos = float3(chunk.minPosition);
    float3 maxPos = float3(chunk.maxPosition);
    return mix(minPos, maxPos, t);
}

inline float3 decodePackedScale(PackedSplat packed, PackedSplatChunk chunk) {
    float3 t = unpack111011(packed.data.z);
    float3 minScale = float3(chunk.minScale);
    float3 maxScale = float3(chunk.maxScale);
    float3 logScale = mix(minScale, maxScale, t);
    return exp(logScale);
}

inline half4 decodePackedColor(PackedSplat packed, PackedSplatChunk chunk) {
    float4 colorUnorm = float4(unpackUnorm(packed.data.w >> 24, 8),
                               unpackUnorm(packed.data.w >> 16, 8),
                               unpackUnorm(packed.data.w >> 8, 8),
                               unpackUnorm(packed.data.w, 8));
    float3 minColor = float3(chunk.minColor);
    float3 maxColor = float3(chunk.maxColor);
    float3 rgb = mix(minColor, maxColor, colorUnorm.xyz);
    return half4(half3(rgb), half(colorUnorm.w));
}

inline float3x3 quaternionToMatrix(float4 q) {
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return float3x3(
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy),
        2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
        2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy)
    );
}

inline Splat decodePackedSplat(PackedSplat packed, PackedSplatChunk chunk) {
    Splat splat;
    float3 position = decodePackedPosition(packed, chunk);
    float3 scale = decodePackedScale(packed, chunk);
    float4 rotation = unpackRotation2101010(packed.data.y);

    float3x3 rotMat = quaternionToMatrix(rotation);
    float3x3 scaleMat = float3x3(
        float3(scale.x, 0.0f, 0.0f),
        float3(0.0f, scale.y, 0.0f),
        float3(0.0f, 0.0f, scale.z)
    );
    float3x3 transform = rotMat * scaleMat;
    float3x3 cov3D = transform * transpose(transform);

    splat.position = packed_float3(position);
    splat.color = decodePackedColor(packed, chunk);
    splat.covA = packed_half3(half3(cov3D[0][0], cov3D[0][1], cov3D[0][2]));
    splat.covB = packed_half3(half3(cov3D[1][1], cov3D[1][2], cov3D[2][2]));
    return splat;
}

// Inline helper functions
inline half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    // Use fast exponential for significant performance improvement
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? half(0) : fast::exp(half(0.5) * negativeMagnitudeSquared) * splatAlpha;
}

inline half3 lodTintForBand(half band) {
    switch (int(band)) {
        case 0: return half3(0.4h, 1.0h, 0.6h);   // near
        case 1: return half3(1.0h, 0.85h, 0.4h);  // mid
        case 2: return half3(1.0h, 0.45h, 0.35h); // far
        default: return half3(0.6h, 0.7h, 1.0h);  // very far
    }
}

inline half4 shadeSplat(FragmentIn in) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    half3 rgb = in.color.rgb;

    if ((in.debugFlags & DebugFlagLodTint) != 0) {
        rgb = lodTintForBand(in.lodBand);
    }
    if ((in.debugFlags & DebugFlagOverdraw) != 0) {
        half intensity = clamp(alpha + 0.05h, 0.05h, 1.0h);
        rgb = half3(intensity);
    }

    return half4(alpha * rgb, alpha);
}

// Stochastic (dithered) transparency shading.
// Instead of alpha blending, uses a screen-space hash to stochastically accept/reject fragments.
// This enables order-independent transparency - no sorting required.
// Best used with TAA (temporal anti-aliasing) to reduce noise.
inline half4 shadeSplatDithered(FragmentIn in, float2 screenPos) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    half3 rgb = in.color.rgb;

    if ((in.debugFlags & DebugFlagLodTint) != 0) {
        rgb = lodTintForBand(in.lodBand);
    }
    if ((in.debugFlags & DebugFlagOverdraw) != 0) {
        half intensity = clamp(alpha + 0.05h, 0.05h, 1.0h);
        rgb = half3(intensity);
    }

    // Screen-space hash for stochastic test
    // Uses a simple but effective hash function for temporal stability
    float hash = fract(sin(dot(screenPos, float2(12.9898, 78.233))) * 43758.5453);

    // Stochastic alpha test: discard if alpha < random threshold
    if (float(alpha) < hash) {
        discard_fragment();
    }

    // Output opaque fragment (no blending needed)
    return half4(rgb, 1.0h);
}
