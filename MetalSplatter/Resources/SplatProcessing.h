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
