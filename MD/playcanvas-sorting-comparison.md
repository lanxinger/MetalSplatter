# PlayCanvas vs MetalSplatter: Gaussian Splat Sorting Comparison

## Executive Summary

This document compares the Gaussian splat sorting implementations between the PlayCanvas WebGL engine and MetalSplatter, analyzing their approaches and identifying optimization opportunities.

## Architecture Comparison

### PlayCanvas (Web Worker + Counting Sort)

**Location:** `engine/src/scene/gsplat-unified/`

**Key Files:**
- [gsplat-unified-sorter.js](../../engine/src/scene/gsplat-unified/gsplat-unified-sorter.js) - Main sorter coordinator
- [gsplat-unified-sort-worker.js](../../engine/src/scene/gsplat-unified/gsplat-unified-sort-worker.js) - Worker thread implementation
- [gsplat-manager.js](../../engine/src/scene/gsplat-unified/gsplat-manager.js) - Scene/camera management

**Approach:**
1. Maintains centers cache in Web Worker (indexed by resource ID)
2. Computes effective min/max distance from AABB 8-corner projection
3. Maps distances to 32 camera-relative bins with weighted precision
4. Performs counting sort in worker thread
5. Transfers sorted indices back to main thread
6. Throttles jobs (max 3 in-flight) to prevent worker backlog

**Precision Weighting:**
```javascript
const weightTiers = [
    { maxDistance: 0, weight: 40.0 },   // Camera bin
    { maxDistance: 2, weight: 20.0 },   // Adjacent bins
    { maxDistance: 5, weight: 8.0 },    // Nearby bins
    { maxDistance: 10, weight: 3.0 },   // Medium distance
    { maxDistance: Infinity, weight: 1.0 }  // Far bins
];
```

### MetalSplatter (GPU + MPSArgSort)

**Location:** `MetalSplatter/Sources/`

**Key Files:**
- [SplatRenderer.swift](../MetalSplatter/Sources/SplatRenderer.swift) - Main renderer
- [MPSArgSort.swift](../MetalSplatter/Sources/MPSArgSort.swift) - GPU sort wrapper
- [ComputeDistances.metal](../MetalSplatter/Resources/ComputeDistances.metal) - Distance compute shader

**Approach (Standard):**
1. GPU compute shader calculates per-splat distances
2. MPSGraph argSort performs GPU-accelerated sorting
3. CPU reads back sorted indices
4. CPU reorders splat buffer

**Approach (Binned - New):**
1. CPU computes AABB bounds (TODO: move to GPU)
2. GPU kernel sets up camera-relative bin parameters
3. GPU kernel computes binned distances (uint32 values)
4. MPSGraph argSort on binned distances
5. CPU reads back sorted indices
6. CPU reorders splat buffer

## Sort Throttling Comparison

### PlayCanvas
```javascript
// From gsplat-manager.js:437-458
testCameraMovedForSort() {
    const epsilon = 0.001;

    if (this.scene.gsplat.radialSorting) {
        // For radial: only position matters
        return this.lastSortCameraPos.distance(currentPos) > epsilon;
    }

    // For directional: only forward direction matters
    const dot = this.lastSortCameraFwd.dot(currentFwd);
    const angle = Math.acos(dot);
    return angle > epsilon;
}
```

**Features:**
- Separate epsilon checks for radial vs directional sorting
- Hard-coded epsilon (0.001)
- No time-based rate limiting
- Job queue tracking prevents worker overload

### MetalSplatter
```swift
// From SplatRenderer.swift:765-780
private func shouldResortForCurrentCamera() -> Bool {
    if sortDirtyDueToData {
        return true
    }
    let now = CFAbsoluteTimeGetCurrent()
    if minimumSortInterval > 0 && (now - lastSortTime) < minimumSortInterval {
        return false
    }
    guard let lastPos = lastSortedCameraPosition,
          let lastFwd = lastSortedCameraForward else {
        return true
    }
    let positionDelta = simd_distance(sortCameraPosition, lastPos)
    let forwardDelta = 1 - simd_dot(simd_normalize(sortCameraForward), simd_normalize(lastFwd))
    return positionDelta > sortPositionEpsilon || forwardDelta > sortDirectionEpsilon
}
```

**Features:**
- Configurable epsilon values (`sortPositionEpsilon`, `sortDirectionEpsilon`)
- Optional time-based rate limiting (`minimumSortInterval`)
- Data-dirty flag for forced re-sort when geometry changes
- Checks both position AND direction (not mode-specific)

## Key Differences

| Feature | PlayCanvas | MetalSplatter (Before) | MetalSplatter (After) |
|---------|-----------|----------------------|---------------------|
| **Execution Location** | CPU (Worker) | GPU (Compute) | GPU (Compute) |
| **Sort Algorithm** | Counting Sort | MPSGraph ArgSort | MPSGraph ArgSort |
| **Distance Precision** | 32 weighted bins | Uniform float32 | 32 weighted bins (optional) |
| **AABB Optimization** | 8-corner projection | None | CPU-side (TODO: GPU) |
| **Job Throttling** | 3 in-flight max | None | None |
| **Sort Throttling** | Hard-coded 0.001 | Configurable | Configurable |
| **Mode-Specific Check** | Yes (radial vs directional) | No | No |
| **Buffer Reuse** | Manual arrays | MetalBufferPool | MetalBufferPool |
| **Time Rate Limiting** | No | Optional | Optional |

## Performance Analysis

### PlayCanvas Bottlenecks
1. **CPU-bound** - Counting sort runs on single worker thread
2. **Transfer overhead** - Must copy centers to worker, results to main thread
3. **Worker latency** - Communication overhead between threads
4. **Limited parallelism** - Max 3 concurrent sorts

### PlayCanvas Strengths
1. **Binned precision** - Better quality for large scenes
2. **AABB optimization** - Avoids per-splat distance initially
3. **Job throttling** - Prevents worker overload
4. **Buffer reuse** - Efficient memory management

### MetalSplatter Bottlenecks
1. **CPU readback** - Sorted indices must be read back to CPU
2. **CPU reorder** - Splat buffer reordering happens on CPU
3. **No job throttling** - Could queue up multiple sorts
4. **Full precision always** - (Before) No option for binned precision

### MetalSplatter Strengths
1. **GPU parallelism** - Distance computation fully parallel
2. **MPSGraph optimization** - Highly optimized sort implementation
3. **Integrated pipeline** - No worker communication overhead
4. **Buffer pooling** - Automatic memory management with MetalBufferPool
5. **Flexible throttling** - Configurable epsilon and time limits

## Implemented Optimizations

### ✅ Binned Precision Sorting
- **Status:** Implemented as optional feature
- **Files:** `ComputeDistancesBinned.metal`, `BinnedSorter.swift`
- **Impact:** Improved visual quality for large scenes
- **Usage:** `renderer.useBinnedSorting = true`

### ✅ Sort Throttling (Already Present)
- **Status:** Already implemented, well-tuned
- **Properties:** `sortPositionEpsilon`, `sortDirectionEpsilon`, `minimumSortInterval`
- **Impact:** Reduces unnecessary re-sorts
- **Recommendation:** Already optimal, but could add mode-specific checks

## Recommended Future Optimizations

### High Priority

1. **GPU AABB Bounds Computation**
   - Move `computeDistanceBounds` from CPU to GPU
   - Use parallel reduction to find min/max
   - Eliminates CPU readback before sort

2. **Mode-Specific Throttling**
   - Only check position delta for radial sorting
   - Only check direction delta for directional sorting
   - Minor optimization but matches PlayCanvas approach

3. **Sort Job Queue**
   - Track in-flight sorts (like PlayCanvas)
   - Skip sort if already busy
   - Prevents GPU overload on heavy scenes

### Medium Priority

4. **Adaptive Binning**
   - Adjust bin count based on splat count
   - More bins for larger scenes
   - Fewer bins for small scenes

5. **LOD-Aware Binning**
   - Coarser binning for distant LODs
   - Saves computation for low-importance splats

6. **Bin Visualization**
   - Debug mode showing bin assignments
   - Helps tune bin weights

### Low Priority

7. **Counting Sort on GPU**
   - Implement PlayCanvas algorithm on GPU
   - Compare performance vs MPSArgSort
   - Likely not faster, but worth testing

8. **Sort Quality Metrics**
   - Automated testing of sort quality
   - Quantify improvements from binning
   - Regression testing

## Code Examples

### Enabling Binned Sorting

```swift
// In your app initialization
let renderer = try SplatRenderer(
    device: device,
    colorFormat: .bgra8Unorm,
    depthFormat: .depth32Float,
    sampleCount: 1,
    maxViewCount: 2,
    maxSimultaneousRenders: 3
)

// Enable PlayCanvas-inspired binned sorting
renderer.useBinnedSorting = true

// Tune sort throttling for your scene
renderer.sortPositionEpsilon = 0.02  // Resort on 2cm movement
renderer.sortDirectionEpsilon = 0.0005  // Resort on ~1.5° rotation
renderer.minimumSortInterval = 0.016  // Max 60 sorts/second
```

### Comparing Sort Modes

```swift
// For testing, toggle binned sorting at runtime
func toggleBinnedSorting() {
    renderer.useBinnedSorting.toggle()
    print("Binned sorting: \(renderer.useBinnedSorting ? "ON" : "OFF")")
}

// Monitor sort performance
renderer.onSortComplete = { duration in
    print("Sort completed in \(duration * 1000)ms")
}
```

## Conclusion

MetalSplatter now incorporates the key sorting optimizations from PlayCanvas:

1. **✅ Binned Precision Sorting** - Optional feature for improved quality
2. **✅ Sort Throttling** - Already well-implemented with configurable parameters

The implementation adapts PlayCanvas's web worker approach to Metal's GPU compute model, maintaining the benefits while leveraging GPU parallelism. The binned sorting is optional and disabled by default for backward compatibility.

Future work should focus on moving AABB bounds computation to GPU and adding mode-specific throttling checks for maximum efficiency.

## References

- [PlayCanvas Engine Repository](https://github.com/playcanvas/engine)
- [gaussian-splat-analysis.md](../../engine/gaussian-splat-analysis.md) - Detailed PlayCanvas analysis
- [binned-precision-sorting.md](binned-precision-sorting.md) - Implementation documentation
- [metal-splatter-gaussian-analysis.md](metal-splatter-gaussian-analysis.md) - MetalSplatter architecture
