# MetalSplatter vs PlayCanvas GSplat: Optimization Analysis

A comprehensive comparison of the MetalSplatter (Metal/Swift) and PlayCanvas (WebGL/WebGPU) Gaussian splatting implementations, identifying optimization opportunities to achieve browser-level performance in native Metal.

---

## Executive Summary

PlayCanvas achieves impressive Gaussian splatting performance in a web browser through several key optimizations that MetalSplatter could adopt:

1. **O(n) Counting Sort** vs O(n log n) radix sort
2. **Aggressive data compression** (32 bytes → ~12 bytes per splat)
3. **Chunk-based histogram caching** for sorting acceleration
4. **Camera movement throttling** to skip unnecessary re-sorts
5. **Unified multi-splat rendering** with global depth ordering

---

## Detailed Comparison

### 1. Sorting Algorithm

| Aspect | PlayCanvas | MetalSplatter |
|--------|------------|---------------|
| **Algorithm** | Counting sort O(n) | MPS argSort (radix) O(n log n) |
| **Precision** | 16-20 bits dynamic | Full float precision |
| **Chunking** | 256-splat chunks with pre-computed histograms | No chunking |
| **Throttling** | Epsilon 0.001 threshold skips re-sort | Always re-sorts |
| **Culling** | Binary search for front-plane culling | Full scene sorted |

#### PlayCanvas Counting Sort Implementation

From `src/scene/gsplat/gsplat-sort-worker.js`:

```javascript
// 32-bin camera-relative precision allocation
const binWeights = [
    40,  // Camera bin - highest precision
    20, 20,  // Adjacent bins (±1)
    8, 8, 8, 8,  // Nearby bins (±2-3)
    3, 3, 3, 3, 3, 3,  // Medium distance
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  // Far bins
];

// Counting sort with histogram
for (let i = 0; i < numSplats; i++) {
    const bin = getBin(distances[i]);
    histogram[bin]++;
}
// Prefix sum for final positions
for (let i = 1; i < 32; i++) {
    histogram[i] += histogram[i - 1];
}
```

#### Recommendation

Implement counting sort in Metal compute shader:

```metal
kernel void countingSortPass1(
    device const float* distances [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float d = distances[tid];
    uint bin = uint((d - params.minDist) / params.binSize);
    bin = clamp(bin, 0u, 31u);
    atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
}
```

**Expected improvement: 2-3× sorting speedup**

---

### 2. Data Compression

| Data | PlayCanvas | MetalSplatter | Savings |
|------|------------|---------------|---------|
| **Position** | 11+10+11 bits (4 bytes) | 3× float32 (12 bytes) | 3× |
| **Rotation** | 2+10+10+10 bits (4 bytes) | half4 (8 bytes) | 2× |
| **Scale** | 11+10+11 bits log-space (4 bytes) | 3× float32 (12 bytes) | 3× |
| **Color** | RGBA8 or SH-normalized (4 bytes) | half4 (8 bytes) | 2× |
| **Total** | ~16 bytes | 32 bytes | 2× |

#### PlayCanvas Compression Scheme

From `src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCompressedData.js`:

```glsl
// Position: 11+10+11 bits packed into uint32
vec3 unpackPosition(uint packed, vec3 chunkMin, vec3 chunkMax) {
    vec3 normalized = vec3(
        float(packed >> 21) / 2047.0,           // 11 bits [0-2047]
        float((packed >> 11) & 0x3FF) / 1023.0, // 10 bits [0-1023]
        float(packed & 0x7FF) / 2047.0          // 11 bits [0-2047]
    );
    return mix(chunkMin, chunkMax, normalized);
}

// Rotation: 2+10+10+10 bits (largest component index + 3 others)
vec4 unpackRotation(uint packed) {
    uint largest = packed >> 30;  // 2 bits: which component is largest
    float x = float((packed >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    float y = float((packed >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    float z = float(packed & 0x3FF) / 1023.0 * 2.0 - 1.0;
    // Reconstruct 4th component from unit quaternion constraint
    float w = sqrt(max(0.0, 1.0 - x*x - y*y - z*z));
    // Reorder based on largest index...
}

// Scale: logarithmic encoding for better precision distribution
vec3 unpackScale(uint packed, vec3 chunkMinScale, vec3 chunkMaxScale) {
    vec3 normalized = vec3(
        float(packed >> 21) / 2047.0,
        float((packed >> 11) & 0x3FF) / 1023.0,
        float(packed & 0x7FF) / 2047.0
    );
    return exp(mix(log(chunkMinScale), log(chunkMaxScale), normalized));
}
```

#### Recommendation

Implement compressed splat format in Metal:

```metal
struct CompressedSplat {
    uint32_t position;      // 11+10+11 bits
    uint32_t rotation;      // 2+10+10+10 bits
    uint32_t scale;         // 11+10+11 bits log-space
    uint32_t color;         // RGBA8
};  // 16 bytes total

struct ChunkBounds {
    float3 posMin, posMax;
    float3 scaleMin, scaleMax;
};  // Shared per 256 splats
```

**Expected improvement: 50% memory bandwidth reduction**

---

### 3. Chunk-Based Histogram Caching

PlayCanvas pre-computes histograms for 256-splat chunks, allowing:

- Skip re-processing static chunks
- Faster AABB projection for depth range estimation
- Parallel histogram aggregation

From `src/scene/gsplat/gsplat-sort-worker.js`:

```javascript
// Pre-computed chunk data
const chunkSize = 256;
const numChunks = Math.ceil(numSplats / chunkSize);

// Each chunk stores:
// - AABB (min/max position)
// - Histogram contribution per bin
// - Last computed camera position

function updateChunkHistogram(chunkIndex, cameraPos) {
    const chunk = chunks[chunkIndex];
    if (distance(chunk.lastCameraPos, cameraPos) < epsilon) {
        return chunk.cachedHistogram;  // Skip recomputation
    }
    // Recompute only this chunk's histogram...
}
```

#### Recommendation

Add chunk metadata buffer:

```metal
struct ChunkMetadata {
    float3 aabbMin;
    float3 aabbMax;
    float3 lastCameraPos;
    uint histogramCache[32];
};
```

---

### 4. Camera Movement Throttling

PlayCanvas skips re-sorting when camera movement is below threshold:

```javascript
// From gsplat-sorter.js
const epsilon = 0.001;

function shouldSort(newCamera, oldCamera) {
    const positionDelta = distance(newCamera.position, oldCamera.position);
    const rotationDelta = quaternionAngle(newCamera.rotation, oldCamera.rotation);

    return positionDelta > epsilon || rotationDelta > epsilon;
}
```

#### Recommendation

Add to MetalSplatter's `SplatRenderer.swift`:

```swift
private var lastSortCameraPosition: SIMD3<Float> = .zero
private var lastSortCameraRotation: simd_quatf = .init()
private let sortEpsilon: Float = 0.001

func shouldResort(camera: Camera) -> Bool {
    let positionDelta = simd_distance(camera.position, lastSortCameraPosition)
    let rotationDelta = simd_angle(camera.rotation, lastSortCameraRotation)
    return positionDelta > sortEpsilon || rotationDelta > sortEpsilon
}
```

**Expected improvement: 0ms sort time for static camera**

---

### 5. Instancing Strategy

| Aspect | PlayCanvas | MetalSplatter |
|--------|------------|---------------|
| **Splats per Instance** | 128 | 32 (meshlet) |
| **Vertex Layout** | 4×16 grid | Per-splat |
| **Draw Calls** | Single instanced | Per-model |

From `src/scene/gsplat/gsplat-resource-base.js`:

```javascript
const numSplatsPerInstance = 128;  // 4x more than MetalSplatter

// Vertex buffer: 4 corners × 128 splats = 512 vertices per instance
// But only 4 unique corner offsets, rest is instanced
```

#### Recommendation

Increase meshlet size in `Metal4MeshShaders.metal`:

```metal
// Current
[[max_total_threads_per_threadgroup(32)]]

// Recommended
[[max_total_threads_per_threadgroup(128)]]
```

---

### 6. Unified Work Buffer Architecture

PlayCanvas aggregates all splats across multiple models into a single unified buffer:

From `src/scene/gsplat-unified/gsplat-work-buffer.js`:

```javascript
class GSplatWorkBuffer {
    constructor() {
        // Single buffer for ALL scene splats
        this.colorTexture = null;      // RGBA16F
        this.centerTexture = null;     // RG32F (screen position)
        this.covarianceTexture = null; // RGBA32F
        this.orderBuffer = null;       // Sorted indices
    }

    // All splats rendered with single draw call
    render() {
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.totalSplatCount);
    }
}
```

#### Benefits
- Single draw call for entire scene
- Global depth sorting across all models
- Reduced pipeline state changes
- Better GPU occupancy

#### Recommendation

Implement unified splat manager:

```swift
class UnifiedSplatManager {
    var allSplats: MTLBuffer          // All scene splats concatenated
    var sortedIndices: MTLBuffer      // Global sort order
    var splatRanges: [(start: Int, count: Int)]  // Per-model ranges

    func renderAll(encoder: MTLRenderCommandEncoder) {
        encoder.setVertexBuffer(allSplats, offset: 0, index: 0)
        encoder.setVertexBuffer(sortedIndices, offset: 0, index: 1)
        encoder.drawPrimitives(type: .triangleStrip,
                               vertexStart: 0,
                               vertexCount: 4,
                               instanceCount: totalSplatCount)
    }
}
```

---

### 7. Color-Only Update Path

PlayCanvas separates geometry and color updates:

From `src/scene/gsplat-unified/gsplat-work-buffer.js`:

```javascript
// Track what changed
const updateFlags = {
    GEOMETRY: 1,  // Position, rotation, scale
    COLOR: 2,     // SH evaluation, lighting
    ALL: 3
};

function update(flags) {
    if (flags & GEOMETRY) {
        this.uploadGeometry();  // Expensive
    }
    if (flags & COLOR) {
        this.uploadColors();    // Cheaper, separate pass
    }
}
```

#### Recommendation

Add dirty flags to `SplatRenderer.swift`:

```swift
struct DirtyFlags: OptionSet {
    let rawValue: UInt8
    static let geometry = DirtyFlags(rawValue: 1 << 0)
    static let color = DirtyFlags(rawValue: 1 << 1)
    static let sorting = DirtyFlags(rawValue: 1 << 2)
}

var dirtyFlags: DirtyFlags = []

func updateIfNeeded() {
    if dirtyFlags.contains(.geometry) {
        updateGeometryBuffers()
    }
    if dirtyFlags.contains(.color) {
        updateColorBuffers()  // SH evaluation only
    }
    if dirtyFlags.contains(.sorting) {
        performSort()
    }
    dirtyFlags = []
}
```

---

### 8. Interval Texture for LOD

PlayCanvas uses GPU prefix sums for efficient subset rendering:

From `src/scene/gsplat-unified/gsplat-interval-texture.js`:

```javascript
// GPU-side remapping for LOD transitions
class IntervalTexture {
    // Maps visible splat indices to sorted order
    // Allows rendering subsets without full re-sort

    computePrefixSum(intervals) {
        // Parallel prefix sum on GPU
        // O(log n) passes for n elements
    }
}
```

This enables:
- Smooth LOD transitions
- Subset rendering without re-sorting
- Efficient culling result application

---

## Implementation Priority

### Phase 1: High Impact (Week 1-2)

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Camera movement throttling | 10-30% for static views | Low |
| Counting sort implementation | 20-40% sorting speedup | Medium |
| Increase meshlet size to 128 | 10-20% vertex processing | Low |

### Phase 2: Medium Impact (Week 3-4)

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Data compression (32→16 bytes) | 30-50% bandwidth | Medium |
| Chunk histogram caching | 15-25% sorting | Medium |
| Color-only update path | Variable (SH scenes) | Low |

### Phase 3: Architecture (Week 5-6)

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| Unified multi-splat rendering | 20-40% draw overhead | High |
| Interval texture for LOD | Smoother LOD transitions | Medium |
| Global depth sorting | Correct multi-model blending | High |

---

## Reference Files

### PlayCanvas (This Repository)

**Sorting:**
- `src/scene/gsplat/gsplat-sort-worker.js` - Counting sort implementation
- `src/scene/gsplat-unified/gsplat-unified-sort-worker.js` - Chunk histograms

**Compression:**
- `src/scene/gsplat/gsplat-compressed-data.js` - Data structures
- `src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCompressedData.js` - GPU unpacking

**Unified Rendering:**
- `src/scene/gsplat-unified/gsplat-work-buffer.js` - Work buffer system
- `src/scene/gsplat-unified/gsplat-manager.js` - Multi-splat aggregation
- `src/scene/gsplat-unified/gsplat-renderer.js` - Single-call rendering

**Instancing:**
- `src/scene/gsplat/gsplat-resource-base.js` - 128 splats per instance

### MetalSplatter

**Sorting:**
- `MetalSplatter/Sources/BinnedSorter.swift` - Current binned sorting
- `MetalSplatter/Sources/MPSArgSort.swift` - MPS radix sort
- `MetalSplatter/Resources/ComputeDistancesBinned.metal` - Distance compute

**Rendering:**
- `MetalSplatter/Sources/SplatRenderer.swift` - Main renderer
- `MetalSplatter/Resources/Metal4MeshShaders.metal` - Mesh shader pipeline
- `MetalSplatter/Resources/SplatProcessing.metal` - Core shaders

**Data:**
- `SplatIO/Sources/DotSplatEncodedPoint.swift` - Current 32-byte format

---

## Conclusion

PlayCanvas achieves excellent browser performance through:

1. **Algorithmic efficiency**: O(n) counting sort vs O(n log n) radix
2. **Memory efficiency**: 2× smaller splat data through compression
3. **Temporal coherence**: Skipping work when nothing changes
4. **Batching**: Single draw call for all scene splats

Implementing these optimizations in MetalSplatter should yield **50-100% performance improvement** on bandwidth-limited scenarios (mobile GPUs) and **20-40% improvement** on compute-limited scenarios (desktop GPUs).

The most impactful single change is likely the **counting sort with camera throttling**, as it eliminates the dominant per-frame cost for most scenes.
