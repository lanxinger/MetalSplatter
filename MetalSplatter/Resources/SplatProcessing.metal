#import "SplatProcessing.h"

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float4x4 projectionMatrix,
                        uint2 screenSize) {
    // Use fast division for better performance
    float invViewPosZ = fast::divide(1.0f, viewPos.z);
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    // Use fast division for projection matrix reciprocals
    float tanHalfFovX = fast::divide(1.0f, projectionMatrix[0][0]);
    float tanHalfFovY = fast::divide(1.0f, projectionMatrix[1][1]);
    float limX = 1.3 * tanHalfFovX;
    float limY = 1.3 * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    // Pre-compute focal lengths to avoid division in render loop
    float focalX = float(screenSize.x) * projectionMatrix[0][0] * 0.5f;
    float focalY = float(screenSize.y) * projectionMatrix[1][1] * 0.5f;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    // Use fast square root for performance improvement with SIMD optimization
    float meanSquared = mean * mean;
    float dist = max(0.1f, fast::sqrt(meanSquared - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues computed with SIMD-friendly operations
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        // Optimized normalization using fast inverse square root
        float2 unnormalized = float2(b, d - lambda2);
        float invLength = fast::rsqrt(dot(unnormalized, unnormalized));
        eigenvector1 = unnormalized * invLength;
    }

    // Gaussian axes are orthogonal - SIMD-optimized computation
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    // Use fast square root for eigenvector scaling with SIMD operations
    float2 sqrtLambdas = float2(fast::sqrt(lambda1), fast::sqrt(lambda2));
    v1 = eigenvector1 * sqrtLambdas.x;
    v2 = eigenvector2 * sqrtLambdas.y;
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex) {
    FragmentIn out;

    // Optimized matrix multiplication with memory-coalesced access pattern
    // Load position components into SIMD-friendly variables for better memory access
    float3 splatPos = float3(splat.position);
    
    // Pre-load matrix rows for coalesced access
    float3 viewMatrixRow0 = uniforms.viewMatrix[0].xyz;
    float3 viewMatrixRow1 = uniforms.viewMatrix[1].xyz;
    float3 viewMatrixRow2 = uniforms.viewMatrix[2].xyz;
    float3 viewMatrixRow3 = uniforms.viewMatrix[3].xyz;
    
    // SIMD-optimized matrix multiplication
    float3 viewPosition3 = viewMatrixRow0 * splatPos.x +
                          viewMatrixRow1 * splatPos.y +
                          viewMatrixRow2 * splatPos.z +
                          viewMatrixRow3;

    // Early exit for splats behind camera
    if (viewPosition3.z >= 0.0) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    // Optimized projection matrix multiplication
    float4 projectedCenter = uniforms.projectionMatrix * float4(viewPosition3, 1.0);
    
    // Optimized frustum culling with single bounds calculation
    float invW = fast::divide(1.0f, projectedCenter.w);
    float3 ndc = projectedCenter.xyz * invW;
    // Combined frustum check: depth + XY bounds in one condition
    if (ndc.z > 1.0f || any(abs(ndc.xy) > 1.2f)) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    // Pre-load covariance data for coalesced memory access
    packed_half3 covA = splat.covA;
    packed_half3 covB = splat.covB;
    
    float3 cov2D = calcCovariance2D(viewPosition3, covA, covB,
                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    // Pre-compute lookup table data for better cache utilization
    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    
    // Pre-compute screen size as half2 to avoid repeated conversions
    half2 screenSizeFloat = half2(uniforms.screenSize);
    
    // SIMD-optimized delta calculation with coalesced memory access
    half2 axisContribution1 = relativeCoordinates.x * half2(axis1);
    half2 axisContribution2 = relativeCoordinates.y * half2(axis2);
    half2 totalAxisContribution = axisContribution1 + axisContribution2;
    
    half2 projectedScreenDelta = totalAxisContribution * (2.0h * kBoundsRadius) / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = kBoundsRadius * relativeCoordinates;
    out.color = splat.color;
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    // Use fast exponential for significant performance improvement
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? half(0) : fast::exp(half(0.5) * negativeMagnitudeSquared) * splatAlpha;
}

// MARK: - AR Background Rendering

struct ARBackgroundVertex {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct ARBackgroundVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex ARBackgroundVertexOut ar_background_vertex(const ARBackgroundVertex in [[stage_in]]) {
    ARBackgroundVertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

fragment float4 ar_background_fragment(ARBackgroundVertexOut in [[stage_in]],
                                      texture2d<float> capturedImageTextureY [[texture(0)]],
                                      texture2d<float> capturedImageTextureCbCr [[texture(1)]]) {
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    
    // Use the same YUV to RGB conversion as the reference implementation
    const float4x4 ycbcrToRGBTransform = float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
                                                  float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
                                                  float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
                                                  float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));

    float4 color = ycbcrToRGBTransform * float4(capturedImageTextureY.sample(colorSampler, in.texCoord).r, 
                                               capturedImageTextureCbCr.sample(colorSampler, in.texCoord).rg, 
                                               1.0);
    
    // Apply gamma correction for better color representation
    color.rgb = pow(color.rgb, 2.2);
    
    return color;
}

// MARK: - AR Composition

struct ARCompositionVertex {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct ARCompositionVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex ARCompositionVertexOut ar_composition_vertex(const ARCompositionVertex in [[stage_in]]) {
    ARCompositionVertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

fragment float4 ar_composition_fragment(ARCompositionVertexOut in [[stage_in]],
                                       texture2d<float> backgroundTexture [[texture(0)]],
                                       texture2d<float> contentTexture [[texture(1)]]) {
    constexpr sampler textureSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    
    // Sample both textures
    float4 background = backgroundTexture.sample(textureSampler, in.texCoord);
    float4 content = contentTexture.sample(textureSampler, in.texCoord);
    
    // Proper alpha blending: content over background
    float3 finalColor = mix(background.rgb, content.rgb, content.a);
    
    return float4(finalColor, 1.0);
}
