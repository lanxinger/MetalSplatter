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
                        float focalX,
                        float focalY,
                        float tanHalfFovX,
                        float tanHalfFovY);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       uint splatID);

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

// 8x8 Bayer matrix for ordered dithering (normalized to [0,1])
// Better visual quality than hash-based dithering with TAA
constant float bayerMatrix[64] = {
     0/64.0, 32/64.0,  8/64.0, 40/64.0,  2/64.0, 34/64.0, 10/64.0, 42/64.0,
    48/64.0, 16/64.0, 56/64.0, 24/64.0, 50/64.0, 18/64.0, 58/64.0, 26/64.0,
    12/64.0, 44/64.0,  4/64.0, 36/64.0, 14/64.0, 46/64.0,  6/64.0, 38/64.0,
    60/64.0, 28/64.0, 52/64.0, 20/64.0, 62/64.0, 30/64.0, 54/64.0, 22/64.0,
     3/64.0, 35/64.0, 11/64.0, 43/64.0,  1/64.0, 33/64.0,  9/64.0, 41/64.0,
    51/64.0, 19/64.0, 59/64.0, 27/64.0, 49/64.0, 17/64.0, 57/64.0, 25/64.0,
    15/64.0, 47/64.0,  7/64.0, 39/64.0, 13/64.0, 45/64.0,  5/64.0, 37/64.0,
    63/64.0, 31/64.0, 55/64.0, 23/64.0, 61/64.0, 29/64.0, 53/64.0, 21/64.0
};

// Bayer dithering with temporal noise based on splat ID
// Uses & 7 instead of % 8 to handle negative screen coords safely
inline float bayerDither(float2 screenPos, uint splatID) {
    int2 pos = int2(floor(screenPos)) & 7;
    float threshold = bayerMatrix[pos.y * 8 + pos.x];
    // Temporal noise based on splat ID (improves TAA integration)
    threshold += fract(float(splatID) * 0.013) * 0.1;
    // Clamp to [0, 1) to prevent full discard when bayerMatrix (max 63/64) + noise (max 0.1) > 1
    return fract(threshold);
}

// Stochastic (dithered) transparency shading.
// Uses Bayer matrix dithering with temporal noise for better visual quality.
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

    // Bayer matrix dithering with temporal noise from splat ID
    float threshold = bayerDither(screenPos, in.splatID);

    // Stochastic alpha test: discard if alpha < threshold
    if (float(alpha) < threshold) {
        discard_fragment();
    }

    // Output opaque fragment (no blending needed)
    return half4(rgb, 1.0h);
}
