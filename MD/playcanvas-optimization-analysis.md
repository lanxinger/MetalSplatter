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

| Feature | PlayCanvas | MetalSplatter | Priority | Status |
|---------|-----------|---------------|----------|--------|
| **Morton Order Layout** | ✅ | ✅ | High | **IMPLEMENTED** |
| **Counting Sort** | ✅ | ❌ | Medium | Pending |
| **Octree LOD Streaming** | ✅ | ❌ | High | Pending |
| **Stochastic Transparency** | ✅ | ✅ | Medium | **IMPLEMENTED** |
| **Work Buffer Atlas** | ✅ | ❌ | Medium | Pending |
| **SH Update Thresholds** | ✅ | ✅ | Low | **IMPLEMENTED** |

---

## Optimization Recommendations

### 1. Morton Order Data Layout ✅ IMPLEMENTED

**Priority:** High
**Effort:** Medium
**Impact:** Significant GPU cache improvement
**Status:** **COMPLETED** (2025-12-02)

#### Description

PlayCanvas pre-orders splats using Morton codes (Z-order curve) for spatial coherency. This clusters nearby 3D points together in memory, dramatically improving GPU cache hit rates during rendering.

#### MetalSplatter Implementation

**Files added/modified:**
- `SplatIO/Sources/MortonOrder.swift` - Core Morton code computation and reordering utilities
- `SplatIO/Sources/SplatSceneReaderExtension.swift` - Added `readSceneWithMortonOrdering()` method
- `MetalSplatter/Sources/SplatRenderer.swift` - Added `mortonOrderingEnabled` option (default: true)
- `MetalSplatter/Resources/MortonCode.metal` - GPU-accelerated Morton code computation kernel
- `SplatIO/Tests/SplatIOTests.swift` - Comprehensive unit tests

**Usage:**

```swift
// Option 1: Automatic (default behavior)
// Morton ordering is enabled by default in SplatRenderer
renderer.mortonOrderingEnabled = true  // default
renderer.add(points)

// Option 2: Explicit reader method
let reader = try AutodetectSceneReader(url)
let orderedPoints = try reader.readSceneWithMortonOrdering()

// Option 3: Manual reordering
let orderedPoints = MortonOrder.reorder(points)
// Or for large datasets (>100K points):
let orderedPoints = MortonOrder.reorderParallel(points)
```

**API:**
- `MortonOrder.encode(x, y, z)` - Encode 10-bit coordinates to 30-bit Morton code
- `MortonOrder.computeMortonCodes(points)` - Compute codes for all points
- `MortonOrder.reorder(points)` - Reorder points by Morton code
- `MortonOrder.reorderParallel(points)` - Parallel version for large datasets
- `MortonOrder.computeStatistics(points)` - Get distribution statistics

**Configuration:**
- `SplatRenderer.mortonOrderingEnabled` - Enable/disable (default: true)
- `SplatRenderer.mortonParallelThreshold` - Threshold for parallel processing (default: 100,000)

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

### 4. Stochastic Transparency (Dithering) ✅ IMPLEMENTED

**Priority:** Medium
**Effort:** Low
**Impact:** Order-independent transparency option
**Status:** **COMPLETED** (2025-12-02)

#### Description

PlayCanvas offers dithered/stochastic transparency as an alternative to sorted alpha blending. This eliminates the need for depth sorting entirely for certain use cases.

#### MetalSplatter Implementation

**Files added/modified:**
- `MetalSplatter/Resources/SplatProcessing.h` - Added `shadeSplatDithered()` inline function
- `MetalSplatter/Resources/SingleStageRenderPath.metal` - Added `singleStageSplatFragmentShaderDithered` fragment shader
- `MetalSplatter/Sources/SplatRenderer.swift` - Added `useDitheredTransparency` option, dithered pipeline state

**Usage:**

```swift
// Enable dithered (stochastic) transparency
renderer.useDitheredTransparency = true

// Benefits:
// - No sorting overhead - significant performance improvement
// - Order-independent - no popping artifacts from sort order changes
// - Better for VR where sorting latency is problematic

// Trade-offs:
// - Produces noise/stippling pattern (best paired with TAA)
// - May look grainy without temporal anti-aliasing
// - Different visual aesthetic than smooth alpha blending
```

**Shader Implementation:**
```metal
// Screen-space hash for stochastic test
float hash = fract(sin(dot(screenPos, float2(12.9898, 78.233))) * 43758.5453);

// Stochastic alpha test: discard if alpha < random threshold
if (float(alpha) < hash) {
    discard_fragment();
}

// Output opaque fragment (no blending needed)
return half4(rgb, 1.0h);
```

**Pipeline Configuration:**
- Blending: Disabled (fragments are opaque or discarded)
- Depth test: Less (proper occlusion since order-independent)
- Depth write: Enabled

#### Use Cases
- Scenes with TAA (temporal anti-aliasing)
- VR applications where sorting latency is problematic
- Artistic/stylized rendering
- Performance-critical scenarios where sorting overhead is prohibitive

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

### 6. Spherical Harmonics Update Thresholds ✅ IMPLEMENTED

**Priority:** Low
**Effort:** Low
**Impact:** Minor performance improvement
**Status:** **COMPLETED** (2025-12-03)

#### Description

PlayCanvas only re-evaluates spherical harmonics when camera movement exceeds a threshold, reducing computation for static or slowly-moving cameras.

#### MetalSplatter Implementation

**Files modified:**
- `MetalSplatter/Sources/SplatRenderer.swift` - Added threshold properties and `shouldUpdateSHForCurrentCamera()` method
- `MetalSplatter/Sources/SplatRenderer+FastSH.swift` - Integrated threshold check into render path
- `MetalSplatter/Resources/FastSHRenderPath.metal` - Added `skipSHEvaluation` flag to skip per-splat SH computation

**Usage:**

```swift
// Configure SH update thresholds (similar to sort thresholds)
renderer.shDirectionEpsilon = 0.001    // ~2.5° rotation threshold (default)
renderer.minimumSHUpdateInterval = 0   // No minimum interval (default)

// For lower quality but better performance during interaction:
renderer.shDirectionEpsilon = 0.01     // ~8° rotation threshold
```

**API:**
- `SplatRenderer.shDirectionEpsilon` - Direction change threshold for SH re-evaluation (default: 0.001, ~2.5° rotation)
- `SplatRenderer.minimumSHUpdateInterval` - Minimum time between SH updates in seconds (default: 0)
- `SplatRenderer.shouldUpdateSHForCurrentCamera()` - Returns true if SH needs re-evaluation
- `SplatRenderer.didUpdateSHForCurrentCamera()` - Called after SH evaluation to update cache

**How it works:**
1. When rendering with FastSH, the system checks if camera direction has changed beyond `shDirectionEpsilon`
2. If threshold not exceeded, shader receives `skipSHEvaluation = 1` and uses cached base colors
3. If threshold exceeded, shader performs full SH evaluation and cache is updated

#### Expected Benefits
- Reduced GPU computation when camera is stationary or moving slowly
- Maintains visual quality while skipping redundant SH calculations
- Configurable trade-off between visual fidelity and performance

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

- [x] **Morton Order Layout** - ✅ COMPLETED (2025-12-02)
- [x] **Stochastic Transparency** - ✅ COMPLETED (2025-12-02)
- [x] **SH Update Thresholds** - ✅ COMPLETED (2025-12-03)

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
| 2025-12-02 | **Morton Order Layout IMPLEMENTED** - Added `MortonOrder.swift`, GPU kernel, SplatRenderer integration, and unit tests |
| 2025-12-02 | **Stochastic Transparency IMPLEMENTED** - Added `useDitheredTransparency` option, dithered fragment shader, order-independent transparency pipeline |
| 2025-12-03 | **SH Update Thresholds IMPLEMENTED** - Added `shDirectionEpsilon` and `minimumSHUpdateInterval` properties, `shouldUpdateSHForCurrentCamera()` method, shader skip flag for camera-movement-based SH caching |
