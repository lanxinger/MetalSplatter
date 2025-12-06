# Camera-Relative Binned Precision Sorting

## Overview

MetalSplatter now includes an advanced sorting optimization inspired by the PlayCanvas engine's approach to Gaussian splat rendering. This feature provides improved sort quality for large scenes by allocating more sorting precision to splats near the camera, where visual quality matters most.

## Key Concepts

### Standard Sorting (Default)
- Computes raw float distances for each splat (either Euclidean distance squared or forward-vector projection)
- Sorts using full 32-bit float precision uniformly across all distances
- Works well for most scenes but can exhibit artifacts in large scenes with extreme depth ranges

### Binned Precision Sorting (New)
- Divides the distance range into 32 bins
- Allocates variable sorting precision to each bin based on distance from camera
- Camera bin: 40x precision weight
- Adjacent bins (1-2 away): 20x weight
- Nearby bins (3-5 away): 8x weight
- Medium distance (6-10 away): 3x weight
- Far bins (10+ away): 1x weight

This weighted allocation means that splats near the camera get much finer sorting granularity, reducing visual artifacts where they're most noticeable.

## Implementation Details

### Components

1. **ComputeDistancesBinned.metal**
   - `setupCameraRelativeBins`: One-time kernel that computes bin boundaries and precision allocation
   - `computeSplatDistancesBinned`: Main kernel that maps each splat to a binned distance value

2. **BinnedSorter.swift**
   - Swift wrapper that manages Metal pipelines and buffers
   - `computeDistanceBounds()`: CPU-side AABB distance computation (future: move to GPU)
   - `setupBins()`: Invokes GPU kernel to set up bin parameters
   - `computeBinnedDistances()`: Invokes GPU kernel to compute binned distances

3. **SplatRenderer Integration**
   - Optional feature controlled by `useBinnedSorting` property
   - Falls back to standard sorting if binned sorter initialization fails
   - Works with both radial (distance-based) and directional (forward-vector) sorting modes

### Sort Throttling

Sort throttling is already implemented in MetalSplatter to avoid unnecessary re-sorts:

```swift
public var sortPositionEpsilon: Float = 0.01       // Minimum camera movement to trigger resort (meters)
public var sortDirectionEpsilon: Float = 0.0001    // Minimum rotation to trigger resort (~0.5-1°)
public var minimumSortInterval: TimeInterval = 0   // Minimum time between sorts (seconds)
```

The renderer automatically checks if the camera has moved enough to warrant re-sorting:
- **Position-based**: Triggers when camera moves > `sortPositionEpsilon` meters
- **Rotation-based**: Triggers when camera rotates > `sortDirectionEpsilon` radians
- **Time-based**: Optionally rate-limits sorts with `minimumSortInterval`
- **Directional sorting mode**: Only checks rotation (position doesn't affect order)
- **Radial sorting mode**: Only checks position (rotation doesn't affect order)

## Usage

### Enabling Binned Sorting

```swift
let renderer = try SplatRenderer(
    device: device,
    colorFormat: .bgra8Unorm,
    depthFormat: .depth32Float,
    sampleCount: 1,
    maxViewCount: 2,
    maxSimultaneousRenders: 3
)

// Enable binned precision sorting
renderer.useBinnedSorting = true
```

### Adjusting Sort Throttling

```swift
// More aggressive throttling (fewer sorts, better performance)
renderer.sortPositionEpsilon = 0.05  // Resort only when camera moves 5cm+
renderer.sortDirectionEpsilon = 0.001  // Resort only on larger rotations (~2-3°)
renderer.minimumSortInterval = 0.016  // Limit to 60 sorts/sec max

// Less aggressive throttling (more frequent sorts, better quality)
renderer.sortPositionEpsilon = 0.001  // Resort on 1mm camera movement
renderer.sortDirectionEpsilon = 0.0001  // Resort on ~0.5° rotation
renderer.minimumSortInterval = 0  // No time-based rate limiting
```

## Performance Characteristics

### Binned Sorting
- **Pros:**
  - Improved visual quality for large scenes
  - Better precision near camera where it matters
  - Same O(n) GPU compute complexity as standard sorting
  - Uses same MPSArgSort backend (just different input encoding)

- **Cons:**
  - Additional bin setup kernel dispatch per sort
  - CPU-side AABB computation (currently) - could be moved to GPU
  - Slightly more complex distance encoding

### When to Use

- **Enable binned sorting when:**
  - Scene has large depth range (e.g., landscape with distant mountains)
  - Close-up details are important (e.g., character faces, product visualization)
  - You notice sorting artifacts near the camera in standard mode

- **Use standard sorting when:**
  - Scene depth range is small (e.g., single room, small objects)
  - Performance is critical and every millisecond counts
  - Sorting artifacts aren't noticeable in your content

### Sort Throttling Tuning

- **For static or slow-moving scenes:** Increase epsilon values significantly
- **For fast-moving cameras:** Keep epsilon values small but increase `minimumSortInterval`
- **For VR/AR:** Be conservative with throttling; incorrect sort order is very noticeable
- **For large scenes:** Consider larger epsilon values since sort errors are less noticeable at distance

## Comparison with PlayCanvas

MetalSplatter's implementation adapts PlayCanvas's CPU-based web worker sorting to Metal GPU compute:

| Aspect | PlayCanvas | MetalSplatter |
|--------|-----------|---------------|
| **Execution** | CPU (Web Worker) | GPU (Metal Compute) |
| **Algorithm** | Counting sort | MPSArgSort (Radix-like) |
| **Binning** | 32 bins, weighted | 32 bins, same weights |
| **Distance Calc** | Per-splat in worker | GPU parallel compute |
| **AABB Bounds** | Per-splat projection | CPU-side (for now) |
| **Job Throttling** | Max 3 in-flight | Implicit via epsilon checks |
| **Buffer Reuse** | Manual pooling | MetalBufferPool automatic |

## Future Optimizations

1. **GPU AABB Bounds**: Move `computeDistanceBounds` to a GPU kernel using parallel reduction
2. **Adaptive Bin Count**: Adjust number of bins based on scene complexity
3. **LOD Integration**: Use coarser binning for distant LOD levels
4. **Bin Visualization**: Debug mode to visualize bin assignments
5. **Sort Quality Metrics**: Quantify improvements with automated testing

## References

- PlayCanvas Engine: [gsplat-unified-sort-worker.js](https://github.com/playcanvas/engine/blob/main/src/scene/gsplat-unified/gsplat-unified-sort-worker.js)
- PlayCanvas Analysis: [gaussian-splat-analysis.md](../../engine/gaussian-splat-analysis.md)
- MetalSplatter Gaussian Analysis: [metal-splatter-gaussian-analysis.md](metal-splatter-gaussian-analysis.md)
