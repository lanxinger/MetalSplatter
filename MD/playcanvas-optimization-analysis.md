# PlayCanvas Optimization Analysis for MetalSplatter

This document captures optimization techniques from the [PlayCanvas engine](https://github.com/playcanvas/engine) Gaussian splatting implementation that could improve MetalSplatter's performance.

**Analysis Date:** 2025-12-02
**PlayCanvas Version Analyzed:** v2.13+
**MetalSplatter Commit:** fa6f181

---

## Executive Summary

PlayCanvas has a mature, web-optimized Gaussian splatting implementation. While MetalSplatter already incorporates several similar techniques (binned sorting, GPU acceleration, async processing), there are valuable optimizations worth adopting, particularly **Morton order data layout** and **octree-based LOD streaming**.

---

## Current State Comparison

### Features MetalSplatter Already Has

| Feature | MetalSplatter | PlayCanvas | Notes |
|---------|--------------|------------|-------|
| **Binned Sorting** | ✅ 32 bins, camera-relative | ✅ 32 bins, histogram | Similar approach |
| **GPU Sorting** | ✅ MPS ArgSort (radix) | ⚠️ CPU Web Worker | Metal advantage |
| **Async Sorting** | ✅ Double-buffered, compute queue | ✅ Web Worker | Similar |
| **Frustum Culling** | ✅ GPU compute shader | ✅ Basic back-face | MetalSplatter more thorough |
| **LOD System** | ✅ Distance bands (10/25/50m) | ✅ Octree streaming | PlayCanvas more sophisticated |
| **Mesh Shaders** | ✅ Metal 3+ | ❌ N/A (WebGL) | Metal advantage |
| **Compression** | ✅ SPZ, SOGS | ✅ SOGS, SOG | Comparable |
| **Multi-Stage Depth** | ✅ Tile memory blending | ❌ N/A | Metal advantage |
| **SIMD Optimization** | ✅ simdgroup_min/max | ❌ N/A | Metal advantage |

### Features to Consider Adopting

| Feature | PlayCanvas | MetalSplatter | Priority |
|---------|-----------|---------------|----------|
| **Morton Order Layout** | ✅ | ❌ | High |
| **Counting Sort** | ✅ | ❌ | Medium |
| **Octree LOD Streaming** | ✅ | ❌ | High |
| **Stochastic Transparency** | ✅ | ❌ | Medium |
| **Work Buffer Atlas** | ✅ | ❌ | Medium |
| **SH Update Thresholds** | ✅ | Partial | Low |

---

## Optimization Recommendations

### 1. Morton Order Data Layout

**Priority:** High
**Effort:** Medium
**Impact:** Significant GPU cache improvement

#### Description

PlayCanvas pre-orders splats using Morton codes (Z-order curve) for spatial coherency. This clusters nearby 3D points together in memory, dramatically improving GPU cache hit rates during rendering.

#### PlayCanvas Implementation

Location: `src/scene/gsplat/gsplat-data.js:365-410`

```javascript
calcMortonOrder() {
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    const encodeMorton3 = (x, y, z) => {
        const expandBits = (v) => {
            v = (v | (v << 16)) & 0x030000FF;
            v = (v | (v <<  8)) & 0x0300F00F;
            v = (v | (v <<  4)) & 0x030C30C3;
            v = (v | (v <<  2)) & 0x09249249;
            return v;
        };
        return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    };

    // Quantize positions to 10-bit integers within bounding box
    // Sort by Morton code
    // Return reordering indices
}
```

#### Proposed MetalSplatter Implementation

**Files to modify:**
- `SplatIO/Sources/SplatSceneReader.swift` - Add reordering pass after loading
- `MetalSplatter/Sources/SplatRenderer.swift` - Option to reorder during `add(splat:)`

**Algorithm:**
1. Compute axis-aligned bounding box of all splats
2. Normalize positions to [0, 1023] (10-bit)
3. Compute Morton code for each splat
4. Sort splat indices by Morton code
5. Reorder splat buffer according to sorted indices

**Metal Shader (optional GPU implementation):**
```metal
kernel void computeMortonCodes(
    constant Splat* splats [[buffer(0)]],
    device uint* mortonCodes [[buffer(1)]],
    constant float3& boundsMin [[buffer(2)]],
    constant float3& boundsInvSize [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    float3 pos = float3(splats[index].position);
    float3 normalized = (pos - boundsMin) * boundsInvSize;
    uint3 quantized = uint3(clamp(normalized * 1023.0, 0.0, 1023.0));
    mortonCodes[index] = encodeMorton3(quantized.x, quantized.y, quantized.z);
}
```

#### Expected Benefits
- 20-40% improvement in rendering throughput for large scenes
- Better texture cache utilization
- Reduced memory bandwidth

---

### 2. Counting Sort Alternative

**Priority:** Medium
**Effort:** Medium
**Impact:** Potentially faster for very large datasets (>5M splats)

#### Description

PlayCanvas uses O(n) counting sort with histogram binning instead of O(n log n) comparison-based sorts. This is particularly effective when the key space is bounded.

#### PlayCanvas Implementation

Location: `src/scene/gsplat/gsplat-sort-worker.js`

```javascript
// Two-pass counting sort:
// Pass 1: Build histogram of splats per depth bucket
// Pass 2: Scatter splats to final sorted positions

// Adaptive bit allocation based on dataset size
const compareBits = Math.max(10, Math.min(20, Math.round(Math.log2(numVertices / 4))));
```

#### Proposed MetalSplatter Implementation

**Files to add:**
- `MetalSplatter/Sources/CountingSorter.swift`
- `MetalSplatter/Resources/CountingSort.metal`

**Algorithm:**
```metal
// Pass 1: Count (parallel histogram)
kernel void countingSort_count(
    constant uint* keys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    uint bucket = keys[index] >> (32 - compareBits);
    atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
}

// Pass 2: Prefix sum (exclusive scan)
// Pass 3: Scatter to sorted positions
```

#### Considerations
- MPS ArgSort is highly optimized on Apple Silicon
- Benchmark before switching - may only win for >5M splats
- Could offer as alternative sorting backend

---

### 3. Octree-Based LOD Streaming

**Priority:** High
**Effort:** High
**Impact:** Critical for large scenes (>10M splats)

#### Description

PlayCanvas implements sophisticated octree-based LOD streaming that loads/unloads splat data based on camera position, enabling rendering of massive scenes that don't fit in GPU memory.

#### PlayCanvas Implementation

Locations:
- `src/scene/gsplat-unified/gsplat-octree.js`
- `src/scene/gsplat-unified/gsplat-octree-instance.js`
- `src/framework/parsers/gsplat-octree.js`

Key features:
- Hierarchical octree with LOD levels per node
- Camera distance-based LOD selection
- Behind-camera penalty for deprioritization
- Prefetching of finer LODs
- Cooldown timers to prevent thrashing
- Underfill settings for quality vs. performance

#### Proposed MetalSplatter Implementation

**Files to add:**
- `MetalSplatter/Sources/SplatOctree.swift` - Octree data structure
- `MetalSplatter/Sources/SplatLODStreamer.swift` - Streaming controller
- `SplatIO/Sources/OctreeSplatReader.swift` - Octree file format support

**Data Structure:**
```swift
struct OctreeNode {
    var bounds: AABB
    var lodLevels: [LODLevel]  // Multiple quality levels
    var children: [OctreeNode?]  // 8 children (or nil for leaf)
    var loadState: LoadState
    var lastAccessTime: TimeInterval
}

struct LODLevel {
    var splatRange: Range<Int>
    var fileOffset: UInt64
    var compressedSize: Int
    var quality: Float  // 0.0 = coarse, 1.0 = full
}
```

**Streaming Logic:**
```swift
func updateVisibleNodes(camera: Camera) {
    // 1. Traverse octree, compute screen-space error per node
    // 2. Select appropriate LOD level based on distance
    // 3. Queue loads for nodes entering view
    // 4. Queue unloads for nodes leaving view (with cooldown)
    // 5. Prefetch adjacent/finer nodes
}
```

#### Expected Benefits
- Render arbitrarily large scenes
- Reduced memory footprint
- Progressive loading experience

---

### 4. Stochastic Transparency (Dithering)

**Priority:** Medium
**Effort:** Low
**Impact:** Order-independent transparency option

#### Description

PlayCanvas offers dithered/stochastic transparency as an alternative to sorted alpha blending. This eliminates the need for depth sorting entirely for certain use cases.

#### PlayCanvas Implementation

Location: `src/scene/gsplat/gsplat-material.js`

```javascript
// Blend mode switches based on dithering
if (ditherEnum !== DITHER_NONE) {
    material.blendType = BLEND_NONE;
    material.depthWrite = true;
} else {
    material.blendType = BLEND_PREMULTIPLIED;
}
```

#### Proposed MetalSplatter Implementation

**Files to modify:**
- `MetalSplatter/Resources/SingleStageRenderPath.metal`
- `MetalSplatter/Sources/SplatRenderer.swift` - Add `useDitheredTransparency` option

**Shader Implementation:**
```metal
fragment half4 singleStageSplatFragmentShader_Dithered(
    SplatVertexOut in [[stage_in]],
    uint sampleIndex [[sample_id]]
) {
    float alpha = computeGaussianAlpha(in);

    // Stochastic test using screen-space hash
    float2 screenPos = in.position.xy;
    float hash = fract(sin(dot(screenPos, float2(12.9898, 78.233))) * 43758.5453);

    if (alpha < hash) {
        discard_fragment();
    }

    return half4(in.color.rgb, 1.0);
}
```

#### Use Cases
- Scenes with TAA (temporal anti-aliasing)
- VR applications where sorting latency is problematic
- Artistic/stylized rendering

---

### 5. Work Buffer Atlas

**Priority:** Medium
**Effort:** Medium
**Impact:** Better batching for multi-source scenes

#### Description

PlayCanvas packs multiple splat sources into a unified "work buffer" atlas texture, reducing draw calls and enabling efficient interval-based updates.

#### PlayCanvas Implementation

Locations:
- `src/scene/gsplat-unified/gsplat-work-buffer.js`
- `src/scene/gsplat-unified/gsplat-world-state.js`

Features:
- MRT textures (color + covariance + intervals)
- Row-based packing of splat sources
- Sparse interval remapping
- Incremental color-only updates

#### Proposed MetalSplatter Implementation

**Files to add:**
- `MetalSplatter/Sources/SplatAtlas.swift`
- `MetalSplatter/Resources/AtlasPacking.metal`

**Concept:**
```swift
class SplatAtlas {
    var packedSplats: MTLBuffer      // All sources concatenated
    var sourceIntervals: [(start: Int, count: Int)]
    var atlasTexture: MTLTexture     // Packed attributes

    func addSource(_ splats: [Splat]) -> SourceHandle
    func removeSource(_ handle: SourceHandle)
    func updateSource(_ handle: SourceHandle, splats: [Splat])
}
```

#### Expected Benefits
- Single draw call for multiple splat sources
- Reduced CPU overhead
- Efficient partial updates

---

### 6. Spherical Harmonics Update Thresholds

**Priority:** Low
**Effort:** Low
**Impact:** Minor performance improvement

#### Description

PlayCanvas only re-evaluates spherical harmonics when camera movement exceeds a threshold, reducing computation for static or slowly-moving cameras.

#### PlayCanvas Implementation

Location: `src/scene/gsplat-unified/gsplat-manager.js`

```javascript
// Only update SH when camera moves significantly
if (cameraMovement > shUpdateThreshold) {
    evaluateSphericalHarmonics();
}
```

#### Proposed MetalSplatter Implementation

MetalSplatter already has similar thresholds for sorting (`sortPositionEpsilon`, `sortDirectionEpsilon`). Consider adding specific SH evaluation thresholds.

**Files to modify:**
- `MetalSplatter/Sources/SplatRenderer.swift`

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

- [ ] **Morton Order Layout** - Implement in `SplatSceneReader`
- [ ] **Stochastic Transparency** - Add dithering option to fragment shader
- [ ] **SH Update Thresholds** - Add camera movement checks

### Phase 2: Core Optimizations (2-4 weeks)

- [ ] **Counting Sort** - Implement as alternative backend
- [ ] **Work Buffer Atlas** - Design and implement packing system

### Phase 3: Advanced Features (4-8 weeks)

- [ ] **Octree LOD Streaming** - Full implementation with file format
- [ ] **Progressive Loading** - Streaming decompression and upload

---

## Benchmarking Plan

### Test Datasets

| Dataset | Splat Count | Expected Benefit |
|---------|-------------|------------------|
| Small (room) | 100K-500K | Baseline |
| Medium (building) | 1M-5M | Morton order |
| Large (outdoor) | 10M-50M | LOD streaming |
| Massive (city) | 100M+ | Full streaming |

### Metrics to Track

1. **Frame time** (ms) - Total render time
2. **Sort time** (ms) - Sorting pass duration
3. **GPU memory** (MB) - VRAM usage
4. **Cache hit rate** - GPU L2 cache efficiency
5. **Draw calls** - CPU overhead

### Profiling Tools

- Xcode GPU Frame Debugger
- Metal System Trace
- Instruments (Metal performance counters)

---

## References

### PlayCanvas Resources

- [PlayCanvas Engine GitHub](https://github.com/playcanvas/engine)
- [PlayCanvas Gaussian Splatting Docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/)
- [PlayCanvas SOG Format Blog](https://blog.playcanvas.com/playcanvas-open-sources-sog-format-for-gaussian-splatting/)
- [PlayCanvas v2.13 Release Notes](https://radiancefields.com/playcanvas-engine-2-13-expands-unified-gsplat-performance-and-customization)

### Technical References

- [Morton Codes (Fabian Giesen)](https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/)
- [Radix Sort for WebGPU](https://shi-yan.github.io/webgpuunleashed/Compute/radix_sort.html)
- [GPU Sorting Algorithms (Linebender)](https://linebender.org/wiki/gpu/sorting/)

### Related MetalSplatter Files

| Component | File | Lines |
|-----------|------|-------|
| Main Renderer | `MetalSplatter/Sources/SplatRenderer.swift` | 46-2067 |
| Binned Sorting | `MetalSplatter/Sources/BinnedSorter.swift` | 1-172 |
| Distance Compute | `MetalSplatter/Resources/ComputeDistances.metal` | 1-212 |
| Binned Distance | `MetalSplatter/Resources/ComputeDistancesBinned.metal` | 1-166 |
| MPS Sort | `MetalSplatter/Sources/MPSArgSort.swift` | 1-68 |
| Render Pipeline | `MetalSplatter/Resources/SplatProcessing.metal` | 1-200+ |
| Frustum Culling | `MetalSplatter/Resources/FrustumCulling.metal` | 1-135 |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-02 | Initial analysis document created |
