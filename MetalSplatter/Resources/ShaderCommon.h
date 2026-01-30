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
    BufferIndexPackedColors   = 4,  // Optional packed colors (snorm10a2)
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;

    /*
     The first N splats are represented as as 2N primitives and 4N vertex indices. The remained are represented
     as instanced of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;
    uint indexedSplatCount;
    uint debugFlags;
    float3 lodThresholds;
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

typedef struct
{
    packed_float3 position;
    packed_half4 color;
    packed_half3 covA;
    packed_half3 covB;
} Splat;

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

// Packed color for bandwidth optimization (snorm10a2)
// Use function constant to select between packed/unpacked paths
typedef struct
{
    uint packedColor;       // 4 bytes - RGB10 + A2 (snorm10a2 format)
} PackedColor;

// Function constants for packed color path
// Using indices 10-11 to avoid conflict with SH function constants (0-3)
// Set via MTLFunctionConstantValues when creating pipeline
constant bool usePackedColors [[function_constant(10)]];
constant bool hasPackedColorsBuffer [[function_constant(11)]];

// Helper to unpack snorm10a2 to half4
// Format: [A:2][B:10][G:10][R:10] (standard snorm10a2 layout)
inline half4 unpackSnorm10a2ToHalf(uint packed) {
    // Extract components (10 bits each for RGB, 2 bits for A)
    int r = int(packed & 0x3FF);
    int g = int((packed >> 10) & 0x3FF);
    int b = int((packed >> 20) & 0x3FF);
    int a = int((packed >> 30) & 0x3);

    // Convert signed 10-bit to -1..1 range
    // For snorm: if value >= 512, it's negative (two's complement for 10 bits)
    float rf = (r >= 512) ? float(r - 1024) / 511.0f : float(r) / 511.0f;
    float gf = (g >= 512) ? float(g - 1024) / 511.0f : float(g) / 511.0f;
    float bf = (b >= 512) ? float(b - 1024) / 511.0f : float(b) / 511.0f;
    // Alpha is 2-bit unsigned: 0, 1, 2, 3 -> 0.0, 0.33, 0.67, 1.0
    float af = float(a) / 3.0f;

    return half4(half(rf), half(gf), half(bf), half(af));
}

// Helper to get splat color, using packed buffer if available
inline half4 getSplatColor(uint splatIndex,
                           constant Splat* splats,
                           constant PackedColor* packedColors) {
    if (usePackedColors && hasPackedColorsBuffer) {
        return unpackSnorm10a2ToHalf(packedColors[splatIndex].packedColor);
    } else {
        return splats[splatIndex].color;
    }
}

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
    half lodBand;
    uint debugFlags;
} FragmentIn;
