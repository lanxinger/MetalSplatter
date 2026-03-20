#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 3;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

// Small epsilon to prevent division by zero in projection calculations
constant static const float kDivisionEpsilon = 1e-6f;

enum BufferIndex: int32_t
{
    BufferIndexUniforms       = 0,
    BufferIndexSplat          = 1,
    BufferIndexSortedIndices  = 2,  // GPU-side sorted indices for indirect rendering
    BufferIndexPrecomputed    = 3,  // Precomputed splat data (Metal 4 TensorOps)
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;

    // Precomputed values for covariance projection (derived from projectionMatrix and screenSize)
    float focalX;                  // screenSize.x * projectionMatrix[0][0] / 2
    float focalY;                  // screenSize.y * projectionMatrix[1][1] / 2
    float tanHalfFovX;             // 1 / projectionMatrix[0][0]
    float tanHalfFovY;             // 1 / projectionMatrix[1][1]

    /*
     The first N splats are represented as as 2N primitives and 4N vertex indices. The remained are represented
     as instanced of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;
    uint indexedSplatCount;
    uint debugFlags;
    float3 lodThresholds;
    float covarianceBlur;       // Low-pass filter for 2D covariance (0.3 default, 0.1 for mip splatting)
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

// Compact splat struct — 28 bytes per splat.
// Color stored as RGBA8 (4 bytes) instead of half4 (8 bytes) for 12.5% less memory.
typedef struct
{
    packed_float3 position;     // 12 bytes
    uint packedColor;           // 4 bytes — RGBA8 unorm
    packed_half3 covA;          // 6 bytes
    packed_half3 covB;          // 6 bytes
} Splat;                        // Total: 28 bytes

// Unpack RGBA8 color from Splat struct
inline half4 unpackSplatColor(uint packed) {
    return unpack_unorm4x8_to_half(packed);
}

// Pre-computed splat data for Metal 4 TensorOps optimization
// Must match layout in Metal4TensorOperations.metal
typedef struct
{
    float4 clipPosition;    // 16 bytes - already projected to clip space
    float3 cov2D;           // 12 bytes + 4 padding - 2D covariance (cov_xx, cov_xy, cov_yy)
    float2 axis1;           // 8 bytes - decomposed covariance axis 1
    float2 axis2;           // 8 bytes - decomposed covariance axis 2
    float depth;            // 4 bytes - for sorting
    uint visible;           // 4 bytes - frustum culling result (0 = culled, 1 = visible)
} PrecomputedSplat;         // Total: 64 bytes aligned

// Function constant for 2DGS rendering mode
// When enabled, uses simplified screen-space quads instead of full 3D covariance projection
// Using index 12 to avoid conflict with SH (0-3)
constant bool use2DGS [[function_constant(12)]];

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
    half lodBand;
    uint debugFlags;
    uint splatID [[flat]];  // For temporal noise in Bayer dithering

    // 2DGS ray-splat intersection data (flat-interpolated, same for all 4 quad vertices)
    float3 viewCenter [[flat]];     // Splat center in view space
    float3 viewNormal [[flat]];     // Splat normal in view space (smallest eigenvector of 3D cov)
    float3 viewTangentU [[flat]];   // Tangent U axis in view space (scaled by 1/sigma_u)
    float3 viewTangentV [[flat]];   // Tangent V axis in view space (scaled by 1/sigma_v)
} FragmentIn;
