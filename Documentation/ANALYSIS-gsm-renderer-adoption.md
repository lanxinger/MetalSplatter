# Analysis: Adopting gsm-renderer Techniques into MetalSplatter

This document provides a comprehensive comparison of gsm-renderer and MetalSplatter, identifying techniques from gsm-renderer that could benefit MetalSplatter.

---

## Table of Contents

1. [Project Overview Comparison](#1-project-overview-comparison)
2. [Architecture Philosophy](#2-architecture-philosophy)
3. [Data Structures Comparison](#3-data-structures-comparison)
4. [Spherical Harmonics Implementation](#4-spherical-harmonics-implementation)
5. [Sorting Algorithms](#5-sorting-algorithms)
6. [Tile-Based Rendering](#6-tile-based-rendering)
7. [Performance Optimizations Comparison](#7-performance-optimizations-comparison)
8. [File Format Support](#8-file-format-support)
9. [Platform-Specific Features](#9-platform-specific-features)
10. [Adoptable Techniques](#10-adoptable-techniques-from-gsm-renderer)
11. [Implementation Recommendations](#11-implementation-recommendations)

---

## 1. Project Overview Comparison

| Aspect | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| **Status** | Work-in-progress | Production-ready |
| **Platform** | macOS 14+, iOS 17+ | macOS 14+, iOS 17+, **visionOS 1+** |
| **Swift Version** | Swift 6.0+ | Swift 5.x |
| **Module Count** | 2 (Renderer, RendererTypes) | 5 (MetalSplatter, SplatIO, PLYIO, SampleApp, SplatConverter) |
| **Metal Shader Lines** | ~3,018 | ~3,599 |
| **Production Apps** | None | MetalSplatter Viewer, OverSoul |

---

## 2. Architecture Philosophy

### gsm-renderer: Three Specialized Renderers

Provides **three distinct rendering strategies** optimized for different scenarios:

1. **GlobalRenderer** - Global radix sort
   - Best for high overdraw scenes
   - Single sort handles all tiles
   - Supports ~16M tile assignments
   - Memory: O(total assignments)

2. **LocalRenderer** - Per-tile bitonic sort
   - Fixed memory footprint
   - Cache-friendly (sorting in threadgroup memory)
   - Fixed capacity: 2,048 Gaussians per tile
   - Memory: O(tiles × maxPerTile)

3. **DepthFirstRenderer** - Hybrid depth-first
   - Depth sort before tile expansion
   - Stable sort preserves depth order within tiles
   - Best correctness guarantees

Each renderer is a complete, independent implementation with its own resources and shaders.

### MetalSplatter: Single Flexible Renderer

Uses **one unified `SplatRenderer`** with configurable rendering paths:

- Single-stage pipeline (fast, simple)
- Multi-stage pipeline (high-quality depth for Vision Pro)
- Dithered transparency (order-independent)
- Mesh shaders (optional, Metal 3+)

**Trade-off**: MetalSplatter favors runtime flexibility; gsm-renderer favors algorithmic specialization.

---

## 3. Data Structures Comparison

### Gaussian Storage

| Property | gsm-renderer | MetalSplatter |
|----------|-------------|---------------|
| **Float32 size** | 48 bytes | N/A |
| **Float16 size** | 32 bytes | 28 bytes |
| **Position** | float32 (12 bytes) | float32 (12 bytes) |
| **Color** | Computed from SH | half4 (8 bytes) |
| **Covariance** | Computed from quaternion+scale | Stored directly (12 bytes) |

### gsm-renderer Structure (48 bytes float32, 32 bytes float16)

```c
// Float32 precision (48 bytes)
struct PackedWorldGaussian {
    float px, py, pz;           // 12 bytes - World position
    float opacity;              // 4 bytes
    float sx, sy, sz;           // 12 bytes - Scale (log-space)
    float _pad;                 // 4 bytes
    simd_float4 rotation;       // 16 bytes - Unit quaternion
};

// Float16 precision (32 bytes)
struct PackedWorldGaussianHalf {
    float px, py, pz;           // 12 bytes - Position stays float32
    half opacity;               // 2 bytes
    half sx, sy, sz;            // 6 bytes - Scale
    half rx, ry, rz, rw;        // 8 bytes - Rotation quaternion
    half _pad0, _pad1;          // 4 bytes
};
```

### MetalSplatter Structure (28 bytes)

```c
struct Splat {
    float3 position;            // 12 bytes - Packed float3
    packed_half4 color;         // 8 bytes  - RGBA as Float16
    packed_half3 covA;          // 6 bytes  - Upper covariance
    packed_half3 covB;          // 6 bytes  - Lower covariance
};
```

### Trade-offs

| Aspect | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| **Memory per Gaussian** | 32-48 bytes | 28 bytes |
| **Original data fidelity** | Preserves quaternion/scale | Loses rotation/scale |
| **Computation** | Covariance computed per-frame | Covariance precomputed |
| **Flexibility** | Can recompute with different params | Fixed at load time |

---

## 4. Spherical Harmonics Implementation

### gsm-renderer Approach

**File**: `Sources/Renderer/Shared/GaussianShared.h:30-116`

Uses **Metal function constants** for compile-time specialization:

```metal
// Compile-time constants
constant uint SH_DEGREE [[function_constant(0)]];
constant bool SH_DEGREE_0 = (SH_DEGREE == 0);
constant bool SH_DEGREE_1 = (SH_DEGREE == 1);
constant bool SH_DEGREE_2 = (SH_DEGREE == 2);
constant bool SH_DEGREE_3 = (SH_DEGREE == 3);

// Usage with compile-time branching
if (SH_DEGREE_1) {
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        color.x += float(harmonics[base + i]) * shBasis[i];
        color.y += float(harmonics[base + 4 + i]) * shBasis[i];
        color.z += float(harmonics[base + 8 + i]) * shBasis[i];
    }
}
```

**Key features**:
- 4 specialized pipeline states compiled at init
- `#pragma unroll` for loop unrolling
- Zero runtime branches in hot path
- SH evaluated per-Gaussian during projection

### MetalSplatter Approach

**File**: `MetalSplatter/Resources/spherical_harmonics_evaluate.metal`

Uses **runtime degree checking**:

```metal
float4 evaluateSH(float3 dir, device const float3* sh_coeffs, uint degree) {
    float3 result = float3(0.5f) + SH_C0 * sh_coeffs[0];

    if (degree >= 1) {
        result += SH_C1 * d.y * sh_coeffs[1];
        // ... band 1
    }

    if (degree >= 2) {
        // ... band 2
    }

    if (degree >= 3) {
        // ... band 3
    }

    return float4(clamp(result, 0.0f, 1.0f), 1.0f);
}
```

**Key features**:
- Single pipeline state
- Runtime `if` checks for each band
- Cached per-frame with direction threshold (`shDirectionEpsilon`)
- Palette-based evaluation

### Comparison

| Aspect | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| **Branching** | Compile-time | Runtime |
| **Pipeline states** | 4 (one per degree) | 1 |
| **Loop unrolling** | Explicit `#pragma unroll` | Compiler-dependent |
| **Evaluation timing** | Per-Gaussian in projection | Cached per-frame |
| **Direction handling** | Per-Gaussian direction | Single view direction |

---

## 5. Sorting Algorithms

### gsm-renderer Sorting

| Renderer | Algorithm | Complexity | Key Size | Capacity |
|----------|-----------|------------|----------|----------|
| GlobalRenderer | Radix sort | O(n) | 32-bit | ~16M |
| LocalRenderer | Bitonic sort | O(n log²n) | 16-bit | 2K/tile |
| DepthFirstRenderer | Dual radix sort | O(n) | 32-bit | ~16M |

**Radix Sort Implementation** (`RadixSortHelpers.h`):
- Block size: 256 threads
- Grain size: 4 elements per thread
- Radix: 256 (8-bit digits)
- 4 passes for 32-bit keys
- SIMD-optimized prefix scan

**Key code** (`RadixSortHelpers.h:91-109`):
```metal
template <int SCAN_TYPE, typename BinaryOp, typename T>
static inline T radix_simdgroup_scan(T value, ushort local_id, BinaryOp Op) {
    const ushort lane_id = local_id % 32;
    T temp = simd_shuffle_up(value, 1);
    if (lane_id >= 1) value = Op(value, temp);
    temp = simd_shuffle_up(value, 2);
    if (lane_id >= 2) value = Op(value, temp);
    // ... continues for 4, 8, 16
}
```

### MetalSplatter Sorting

| Sorter | Algorithm | Complexity | Key Size |
|--------|-----------|------------|----------|
| CountingSorter | Counting/histogram | O(n) | 16-bit |
| BinnedSorter | Camera-relative bins | O(n) | 32 bins |
| MPS ArgSort | Radix sort | O(n log n) | 32-bit |

**Counting Sort Implementation** (`CountingSort.metal`):
- 65,536 bins (16-bit depth quantization)
- Three passes: histogram → prefix sum → scatter
- Simple single-thread prefix sum (or basic Hillis-Steele)

**Key code** (`CountingSort.metal:64-80`):
```metal
// Simple single-thread prefix sum
if (tid != 0) return;
uint sum = 0;
for (uint i = 0; i < binCount; i++) {
    prefixSum[i] = sum;
    sum += histogram[i];
}
```

### Comparison

| Aspect | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| **Prefix scan** | SIMD shuffle optimized | Simple/Hillis-Steele |
| **Sort strategies** | 3 specialized algorithms | 3 general algorithms |
| **Per-tile sorting** | Yes (LocalRenderer) | No |
| **Key precision** | 32-bit (can use 16-bit) | 16-bit |

---

## 6. Tile-Based Rendering

### gsm-renderer Tile System

**Tile size**: 16×16 pixels

**GlobalRenderer** (`GlobalShaders.metal:138-161`):
```metal
int minTileX = int(floor(xmin / float(params.tileWidth)));
int maxTileX = int(ceil(xmax / float(params.tileWidth))) - 1;
int minTileY = int(floor(ymin / float(params.tileHeight)));
int maxTileY = int(ceil(ymax / float(params.tileHeight))) - 1;
```

**Features**:
- Explicit tile assignment passes with prefix sums
- Active tile tracking for sparse dispatch
- Gaussian-tile intersection test (exact ellipse test)
- OBB (Oriented Bounding Box) extents for tighter bounds

**LocalRenderer**:
- Fixed 2,048 Gaussians per tile maximum
- Bitonic sort within threadgroup memory
- Atomic scatter into per-tile slots

### MetalSplatter Tile System

**Tile size**: 32×32 pixels (multi-stage path)

**Features**:
- Hardware tile memory for high-quality depth blending
- No explicit tile assignment
- Relies on GPU rasterization for tile dispatch

### Comparison

| Aspect | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| **Tile size** | 16×16 | 32×32 |
| **Assignment** | Explicit GPU passes | Hardware rasterization |
| **Per-tile capacity** | 2,048 (LocalRenderer) | Unlimited |
| **Intersection test** | Exact ellipse | Bounding box |

---

## 7. Performance Optimizations Comparison

| Optimization | gsm-renderer | MetalSplatter |
|--------------|-------------|---------------|
| **Function constants** | ✓ (SH degree) | ✗ |
| **Indirect dispatch** | ✓ | ✓ |
| **Buffer pooling** | ✗ | ✓ |
| **Morton ordering** | ✗ | ✓ |
| **Frustum culling** | ✓ (in projection) | ✓ (dedicated pass) |
| **LOD system** | ✗ (planned) | ✓ |
| **Adaptive quality** | ✗ | ✓ (interaction mode) |
| **Color-only updates** | ✗ | ✓ |
| **Dithered transparency** | ✗ | ✓ |
| **Metal 4 bindless** | ✗ | ✓ |
| **Mesh shaders** | ✗ | ✓ |
| **OBB extents** | ✓ | ✗ |
| **Total ink culling** | ✓ | ✗ |
| **SIMD prefix scan** | ✓ | ✗ |
| **Fused projection** | ✓ | ✗ |

---

## 8. File Format Support

| Format | gsm-renderer | MetalSplatter |
|--------|-------------|---------------|
| PLY (ASCII) | ✓ | ✓ |
| PLY (Binary) | ✓ | ✓ |
| .splat | ✗ | ✓ |
| SPX | ✗ | ✓ |
| SPZ | ✗ (planned) | ✓ |
| SOGS | ✗ | ✓ (v1, v2, optimized) |

MetalSplatter has significantly broader format support through its modular `SplatIO` layer.

---

## 9. Platform-Specific Features

### Vision Pro / visionOS

| Feature | gsm-renderer | MetalSplatter |
|---------|-------------|---------------|
| **visionOS support** | ✗ | ✓ |
| **Stereo rendering** | Basic (separate passes) | Amplification ID |
| **High-quality depth** | ✗ | ✓ (multi-stage) |
| **Reprojection-friendly** | ✗ | ✓ |

### AR Integration

| Feature | gsm-renderer | MetalSplatter |
|---------|-------------|---------------|
| **ARKit** | ✗ | ✓ |
| **Auto-placement** | ✗ | ✓ |
| **Background video** | ✗ | ✓ |

---

## 10. Adoptable Techniques from gsm-renderer

### 10.1 Function Constants for SH Degree Specialization ⭐⭐⭐

**Impact**: High
**Effort**: Low
**Estimated Speedup**: 5-15% for SH evaluation

**gsm-renderer Implementation** (`GaussianShared.h:30-35`):
```metal
constant uint SH_DEGREE [[function_constant(0)]];
constant bool SH_DEGREE_0 = (SH_DEGREE == 0);
constant bool SH_DEGREE_1 = (SH_DEGREE == 1);
constant bool SH_DEGREE_2 = (SH_DEGREE == 2);
constant bool SH_DEGREE_3 = (SH_DEGREE == 3);
```

**Benefits**:
- Eliminates runtime branches in hot SH evaluation path
- Compiler can fully unroll loops and optimize register usage
- Creates 4 specialized pipeline states (SH0, SH1, SH2, SH3)

**Implementation for MetalSplatter**:
```metal
// Add to spherical_harmonics_evaluate.metal
constant uint SH_DEGREE [[function_constant(0)]];
constant bool HAS_SH1 = (SH_DEGREE >= 1);
constant bool HAS_SH2 = (SH_DEGREE >= 2);
constant bool HAS_SH3 = (SH_DEGREE >= 3);

float4 evaluateSH(float3 dir, device const float3* sh_coeffs) {
    float3 result = float3(0.5f) + SH_C0 * sh_coeffs[0];

    if (HAS_SH1) {  // Compile-time constant
        #pragma unroll
        result += SH_C1 * d.y * sh_coeffs[1];
        // ...
    }
    // ...
}
```

Then create 4 pipeline states at initialization with different function constant values.

---

### 10.2 Fused Projection + Tile Bounds Kernel ⭐⭐⭐

**Impact**: High
**Effort**: Medium
**Estimated Speedup**: 10-20% reduction in projection phase

**gsm-renderer Implementation** (`GlobalShaders.metal:19-163`):

Single `projectGaussiansFused` kernel performs:
1. Projects 3D→2D
2. Computes covariance/conic
3. Evaluates spherical harmonics
4. Calculates tile bounds
5. Performs all culling (scale, depth, opacity, off-screen, "total ink")

All with a **single coalesced read** of gaussian data.

**Key code**:
```metal
template <typename PackedWorldT, typename HarmonicT, typename RenderDataT>
kernel void projectGaussiansFused(
    const device PackedWorldT* worldGaussians [[buffer(0)]],
    const device HarmonicT* harmonics [[buffer(1)]],
    device RenderDataT* outRenderData [[buffer(2)]],
    device int4* outBounds [[buffer(3)]],
    device uchar* outMask [[buffer(4)]],
    constant CameraUniforms& camera [[buffer(5)]],
    constant TileBinningParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // SINGLE coalesced read for all core gaussian data
    PackedWorldT g = worldGaussians[gid];

    // All processing in one kernel...
    // Early culls, projection, SH eval, bounds calc

    // Single write of all outputs
    outRenderData[gid] = renderData;
    outBounds[gid] = int4(minTileX, maxTileX, minTileY, maxTileY);
    outMask[gid] = 1;
}
```

**Benefits**:
- Reduces kernel launch overhead
- Improves memory locality - gaussian data read once
- Single write of projected data vs multiple buffer round-trips

---

### 10.3 OBB (Oriented Bounding Box) Extents ⭐⭐

**Impact**: Medium
**Effort**: Low
**Estimated Speedup**: 5-10% fillrate savings for elongated splats

**gsm-renderer Implementation** (`GaussianShared.h:236-261`):
```metal
inline float2 computeOBBExtents(float2x2 cov, float sigma_multiplier) {
    float a = cov[0][0], b = cov[0][1], d = cov[1][1];
    float det = a * d - b * b;
    float mid = 0.5f * (a + d);
    float disc = max(mid * mid - det, 1e-6f);
    float sqrtDisc = sqrt(disc);

    float lambda1 = mid + sqrtDisc;
    float lambda2 = max(mid - sqrtDisc, 1e-6f);

    float e1 = sigma_multiplier * sqrt(max(lambda1, 1e-6f));
    float e2 = sigma_multiplier * sqrt(max(lambda2, 1e-6f));

    float2 v1;
    if (abs(b) > 1e-6f) {
        float vx = b, vy = lambda1 - a;
        float vlen = sqrt(vx * vx + vy * vy);
        v1 = float2(vx, vy) / max(vlen, 1e-6f);
    } else {
        v1 = (a >= d) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    }

    float xExtent = abs(v1.x) * e1 + abs(v1.y) * e2;
    float yExtent = abs(v1.y) * e1 + abs(v1.x) * e2;
    return float2(xExtent, yExtent);
}
```

**Current MetalSplatter**: Uses circular radius bounds (3σ from max eigenvalue)

**Benefits**:
- Tighter screen-space bounds for elongated Gaussians
- Reduces overdraw for non-circular splats
- Better frustum culling accuracy

---

### 10.4 "Total Ink" Culling ⭐⭐

**Impact**: Medium
**Effort**: Low
**Estimated Speedup**: 5-15% reduction in rendered splats

**gsm-renderer Implementation** (`GlobalShaders.metal:99-107`):
```metal
float det = conic.x * conic.z - conic.y * conic.y;
float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
float depthFactor = 1.0 - pow(saturate((farPlane - depth) / (farPlane - nearPlane)), 2.0);
float adjustedThreshold = depthFactor * params.totalInkThreshold;
if (totalInk < adjustedThreshold) {
    // Cull this Gaussian
}
```

**Logic**:
- `totalInk = opacity × 2π / √det` = opacity × screen area
- `depthFactor` makes threshold stricter for distant objects
- Adapts culling based on actual **screen contribution**

**Current MetalSplatter**: Fixed opacity threshold + LOD-based skip factors

**Benefits**:
- More intelligent culling than simple opacity threshold
- Adapts to view distance automatically
- Could replace/supplement the LOD system

---

### 10.5 SIMD-Optimized Prefix Scan Templates ⭐⭐

**Impact**: Medium
**Effort**: Medium
**Estimated Speedup**: 2-5x for prefix sum operations

**gsm-renderer Implementation** (`RadixSortHelpers.h:91-109`):
```metal
template <int SCAN_TYPE, typename BinaryOp, typename T>
static inline T radix_simdgroup_scan(T value, ushort local_id, BinaryOp Op) {
    const ushort lane_id = local_id % 32;
    T temp = simd_shuffle_up(value, 1);
    if (lane_id >= 1) value = Op(value, temp);
    temp = simd_shuffle_up(value, 2);
    if (lane_id >= 2) value = Op(value, temp);
    temp = simd_shuffle_up(value, 4);
    if (lane_id >= 4) value = Op(value, temp);
    temp = simd_shuffle_up(value, 8);
    if (lane_id >= 8) value = Op(value, temp);
    temp = simd_shuffle_up(value, 16);
    if (lane_id >= 16) value = Op(value, temp);
    if (SCAN_TYPE == RADIX_SCAN_TYPE_EXCLUSIVE) {
        temp = simd_shuffle_up(value, 1);
        value = (lane_id == 0) ? 0 : temp;
    }
    return value;
}
```

**Current MetalSplatter** (`CountingSort.metal:64-80`):
```metal
// Simple single-thread prefix sum
if (tid != 0) return;
uint sum = 0;
for (uint i = 0; i < binCount; i++) {
    prefixSum[i] = sum;
    sum += histogram[i];
}
```

**Benefits**:
- SIMD shuffle operations are significantly faster than threadgroup memory
- Better GPU occupancy during scan operations
- Reusable templates for any reduction operation

---

### 10.6 Per-Tile Bitonic Sort Option ⭐

**Impact**: Medium (for specific scenes)
**Effort**: High
**Benefit**: Cache-friendly sorting for uniform distributions

**gsm-renderer LocalRenderer**:
- Bitonic sort within threadgroup memory
- Fixed 2,048 Gaussians per 16×16 tile
- 16-bit depth keys (sufficient for per-tile ordering)
- No global memory traffic during sort

**Benefits**:
- Cache-friendly - sorting happens entirely in L1/threadgroup memory
- Deterministic memory usage - no dynamic allocation
- Better for scenes with uniform splat distribution
- Could be offered as an alternative sorting strategy

---

### 10.7 Gaussian-Tile Intersection Test ⭐

**Impact**: Low (unless adopting tile-based rendering)
**Effort**: Low

**gsm-renderer Implementation** (`GaussianShared.h:313-358`):
```metal
inline bool gaussianTileIntersectsEllipse(
    int2 pix_min, int2 pix_max,
    float2 center, float3 conic, float power
) {
    float w = 2.0f * power;
    float dx, dy, a, b, c;

    // Test horizontal edge intersection
    if (center.x * 2.0f < float(pix_min.x + pix_max.x)) {
        dx = center.x - float(pix_min.x);
    } else {
        dx = center.x - float(pix_max.x);
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;
    if (gaussianSegmentIntersectEllipse(a, b, c, center.y, float(pix_min.y), float(pix_max.y))) {
        return true;
    }
    // ... similar for vertical edge
}

inline bool gaussianIntersectsTile(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    return gaussianTileContainsCenter(pix_min, pix_max, center) ||
           gaussianTileIntersectsEllipse(pix_min, pix_max, center, conic, power);
}
```

**Benefits**:
- More precise tile assignment than bounding box
- Reduces false-positive tile assignments
- Useful if MetalSplatter adopts tile-based rendering

---

### 10.8 Template-Based Gaussian Accessor Functions ⭐

**Impact**: Low (code quality)
**Effort**: Low

**gsm-renderer Implementation** (`GlobalShaders.metal:9-15`):
```metal
inline half2 getMean(const device GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
inline half4 getConic(const device GaussianRenderData& g) { return half4(g.conicA, g.conicB, g.conicC, g.conicD); }
inline half3 getColor(const device GaussianRenderData& g) { return half3(g.colorR, g.colorG, g.colorB); }

// Both device and thread overloads
inline half2 getMean(const thread GaussianRenderData& g) { return half2(g.meanX, g.meanY); }
```

**Benefits**:
- Cleaner shader code
- Easier to change data layouts
- Consistent access patterns across all kernels

---

## 11. Implementation Recommendations

### Priority Matrix

| Priority | Feature | Effort | Impact | Estimated Gain |
|----------|---------|--------|--------|----------------|
| **1** | Function constants for SH degree | Low | High | 5-15% SH speedup |
| **2** | Fused projection kernel | Medium | High | 10-20% projection speedup |
| **3** | SIMD prefix scan templates | Medium | Medium | 2-5x scan speedup |
| **4** | OBB extents for bounds | Low | Medium | 5-10% fillrate |
| **5** | Total ink culling | Low | Medium | 5-15% fewer splats |
| **6** | Per-tile sort option | High | Medium | Scene-dependent |

### Quick Wins (Can Implement Today)

#### 1. Function Constants for SH

Add to `spherical_harmonics_evaluate.metal`:
```metal
constant uint SH_DEGREE [[function_constant(0)]];
constant bool HAS_SH1 = (SH_DEGREE >= 1);
constant bool HAS_SH2 = (SH_DEGREE >= 2);
constant bool HAS_SH3 = (SH_DEGREE >= 3);
```

Create 4 specialized pipelines at init time with:
```swift
let constants = MTLFunctionConstantValues()
constants.setConstantValue(&degree, type: .uint, index: 0)
let function = try library.makeFunction(name: "evaluateSH", constantValues: constants)
```

#### 2. OBB Extents

Drop in gsm-renderer's `computeOBBExtents()` function - it's self-contained:
```metal
inline float2 computeOBBExtents(float2x2 cov, float sigma_multiplier) {
    // ... copy from GaussianShared.h:236-261
}
```

#### 3. Total Ink Culling

Add to existing frustum culling pass:
```metal
float det = covA.x * covA.z - covA.y * covA.y;  // From conic
float totalInk = opacity * 6.283185f / sqrt(max(det, 1e-6f));
float depthFactor = 1.0 - pow(saturate((far - depth) / (far - near)), 2.0);
if (totalInk < depthFactor * totalInkThreshold) {
    // Cull
}
```

### Medium-Term Improvements

#### 4. SIMD Prefix Scan

Replace `countingSortPrefixSum` with SIMD-optimized version:
```metal
// Copy radix_simdgroup_scan template from RadixSortHelpers.h
// Integrate into CountingSort.metal
```

#### 5. Fused Projection Kernel

Combine distance computation + SH evaluation + culling into single kernel:
```metal
kernel void fusedProjectAndCull(
    device const Splat* splats,
    device const float3* shCoeffs,
    device ProjectedSplat* output,
    device uint* visibleCount,
    // ...
) {
    // Single read of splat data
    Splat s = splats[gid];

    // All culling checks
    // Distance computation
    // SH evaluation
    // Output write
}
```

### Long-Term Considerations

#### 6. Alternative Sorting Strategy

Consider adding LocalRenderer-style per-tile sorting as an option:
- Best for scenes with uniform splat distribution
- Predictable memory usage
- Would require tile assignment infrastructure

---

## Summary

The most impactful adoptions from gsm-renderer would be:

1. **Function constants** - Eliminates SH runtime branches with minimal code changes
2. **Fused projection** - Reduces memory bandwidth and kernel launches
3. **SIMD prefix scan** - Speeds up sorting infrastructure

These three changes alone could yield **15-25% overall performance improvement** for MetalSplatter, particularly on complex scenes with higher SH degrees.

The gsm-renderer project demonstrates that algorithmic specialization (multiple renderer strategies) can be powerful, but MetalSplatter's strength is its runtime flexibility and platform breadth. The best approach is to adopt gsm-renderer's low-level optimizations while keeping MetalSplatter's architectural flexibility.
