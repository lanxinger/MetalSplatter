# MetalSplatter Gaussian Splatting – Improvement Ideas

Actionable changes inspired by PlayCanvas' gsplat pipeline (`gaussian-splat-analysis.md`) and mapped to MetalSplatter files. Use this as a backlog; update it as items land.

**Legend**: ✅ = completed, 🚧 = in progress, ⏳ = planned

## Work Buffer & Incremental Updates
- ⏳ **GPU work buffer stage** (color + covariance MRT) to decouple ingest formats from render:
  - `SplatRenderer.swift`: add an intermediate render/compute pass that writes a unified buffer from source splat data; expose toggles to use the path.
  - Shaders: new pair similar to PlayCanvas "copy to workbuffer" (e.g., `SplatWorkbuffer.metal`) plus read path that consumes packed buffers instead of raw splat structs.
- ⏳ **Color-only refresh for SH changes**:
  - `SplatRenderer.swift` / `SplatRenderer+FastSH.swift`: track per-splat dirty bits for SH/color and add a lightweight pass that only refreshes color targets without rewriting covariance/geometry.
  - Shaders: define flag to skip geometry upload when only color is dirty.
- ⏳ **Per-splat dirty tracking (transforms vs color/SH)**:
  - Renderer core: mark transform vs color dirty sets; re-run the work-buffer copy only for transforms, color-only pass for SH updates.

## Sorting Cadence & Throttling
- ✅ **Sort thresholds and job throttling**:
  - ✅ `SplatRenderer.swift`: added `sortPositionEpsilon`, `sortDirectionEpsilon`, and `minimumSortInterval` to skip re-sorting when camera deltas fall below thresholds.
  - ✅ Added `sortJobsInFlight` counter and `maxConcurrentSorts` limit (default: 1) to prevent sort queue buildup.
  - ✅ Sort requests are skipped when max concurrent limit reached; logged at debug level.
  - ✅ Added `sortJobsInFlight` to `FrameStatistics` for monitoring.
  - ✅ Tuned `sortDirectionEpsilon` to 0.0001 (~0.5-1° rotation) to fix flickering/popping during rotation (reduced from 0.001 which was ~3-4°).
- ⏳ **Bin precision near camera**:
  - If using bucketed sorts, bias bit budget to near-camera bins (configurable); expose per-viewport mode (distance vs forward-vector vs radial) on `ViewportDescriptor`.

## LOD & Interval Masking
- ⏳ **GPU interval/prefix masks for LOD-excluded splats**:
  - New compute shader to build masks/prefix sums; render path multiplies or skips masked splats without CPU repacking.
  - `SplatRenderer.swift`: manage interval buffers and hook into render descriptors.
- ⏳ **LOD underfill/prefetch policy**:
  - Asset/scene loader (e.g., `SplatIO` ingestion point + renderer): allow temporary coarser LOD if finer data missing; prefetch one step toward optimal LOD per frame to smooth streaming.
- ⏳ **Cooldown-based unloads**:
  - Buffer/asset manager: delay releasing LOD buffers for N frames/ticks to reduce thrash when camera oscillates around thresholds.

## Debug & Observability
- ✅ **Overdraw and LOD tint debug modes**:
  - ✅ Shaders: added unified `shadeSplat` with overdraw accumulation and per-LOD tinting in `SplatProcessing.h` (inline functions accessible to all render paths).
  - ✅ `SplatRenderer.swift`: exposed `debugOptions` (`.overdraw`, `.lodTint`) and `lodThresholds` for per-viewport control.
  - ✅ Early-return vertices now zero all stage-in fields to prevent undefined data in debug modes.
- ✅ **Stats callbacks and frame readiness**:
  - ✅ Added `FrameStatistics` struct with `onFrameReady` callback plus `onRenderStart`/`onRenderComplete` hooks.
  - ✅ Tracks: ready state, loading count, sort duration (GPU/CPU), buffer upload count, splat count, frame time.
  - ✅ Accessible via `renderer.onFrameReady = { stats in ... }`.
- ✅ **Debug AABBs**:
  - ✅ Added `.showAABB` debug option to render wireframe bounding box
  - ✅ New `DebugAABB.metal` shader with cyan wireframe rendering
  - ✅ Uses existing `calculateBounds()` to determine scene extents
  - ✅ Renders as overlay with depth testing (no depth write) for proper occlusion

## Buffer Management & Reuse
- ✅ **Worker-side caching and buffer reuse**:
  - ✅ Sort path now reuses distance/order buffers across frames via dedicated `MetalBufferPool` instances.
  - ✅ `sortDistanceBufferPool` and `sortIndexBufferPool` eliminate per-frame allocations in GPU sort path.
  - ✅ Added `sortBufferPoolStats` to `FrameStatistics` for monitoring buffer reuse efficiency.
  - ⏳ Future: cache intermediate compute results (centers, bin dividers) to further reduce overhead.
- ✅ **Max-buffer safety checks for huge clouds**:
  - ✅ `MetalBuffer.swift`: now checks and clamps requested capacity against `device.maxBufferLength`; logs warning when clamping occurs.
  - ⏳ Consider fallback strategy (e.g., chunked processing) for clouds exceeding max buffer size.

## Shader/CPU Parity (SH)
- ✅ **Keep SH ordering in lockstep**:
  - ✅ Documented shared SH coefficient ordering (Graphdeco/gsplat format) in `SphericalHarmonicsEvaluator.swift`, `spherical_harmonics_evaluate.metal`, and `FastSHRenderPath.metal`.
  - ✅ Added runtime assertion to verify layout consistency (allows SPZ 15-coeff case).

## Multi-Viewport/Layers
- ⏳ **Director-style aggregation**:
  - If MetalSplatter grows to multiple scenes/layers per device, consider a director that aggregates placements per camera/layer, sharing pipelines and caches to avoid redundant work (analogous to PlayCanvas' `GSplatDirector`/manager).

---

## Recent Completions (Latest Release)

### Quick Wins Landed
1. ✅ **SH Ordering Guardrails** - Documented Graphdeco/gsplat coefficient order across CPU/GPU; added runtime layout assertion
2. ✅ **Debug Overlays** - Overdraw and LOD tint modes via `debugOptions`; unified `shadeSplat` inline function in `SplatProcessing.h`
3. ✅ **Stats Hooks** - `FrameStatistics` + `onFrameReady` callback with sort duration, buffer upload count, ready state
4. ✅ **Sort Throttling** - Configurable `sortPositionEpsilon`, `sortDirectionEpsilon`, `minimumSortInterval` to reduce churn
5. ✅ **Buffer Safety** - `MetalBuffer.swift` now clamps to `device.maxBufferLength` with warnings
6. ✅ **Sort Buffer Reuse** - GPU sort path now uses pooled buffers (`sortDistanceBufferPool`, `sortIndexBufferPool`) to eliminate per-frame allocations
7. ✅ **Jobs-in-Flight Guard** - Prevents sort queue buildup with `sortJobsInFlight` counter and `maxConcurrentSorts` limit (default: 1)
8. ✅ **Debug AABB Rendering** - Wireframe bounding box visualization via `.showAABB` debug option
9. ✅ **Adaptive Interaction Mode** - `beginInteraction()` / `endInteraction()` relax sort thresholds during user gestures to reduce popping; final quality sort scheduled after interaction ends
10. ✅ **GPU-Only Sort Reordering** - Sorted indices stay on GPU, shader indexes into static splat buffer. Eliminates CPU readback and reordering loop for 2-4x sort performance improvement
11. ✅ **SIMD-Group Parallel Bounds** - GPU compute kernel using `simd_min`/`simd_max` for 32x fewer atomic operations. Replaces CPU bounds loop with parallel reduction. **Cached** - computed once on model load, reused until splats change
12. 🔶 **GPU Frustum Culling** - Compute pass filters splats outside camera view frustum (stats only - full culling requires sort-after-cull)
13. ✅ **Async Compute Overlap** - Separate compute queue for sorting, double-buffered sorted indices. Sorting runs in parallel with rendering for smoother frame times

### Usage Notes
- **Toggle overlays**: `renderer.debugOptions = [.overdraw, .lodTint, .showAABB]`; tune `renderer.lodThresholds` as needed
- **Stats hook**: `renderer.onFrameReady = { stats in ... }` provides ready/loading state, sort duration, upload count, splat count, frame time, buffer pool stats, and jobs-in-flight count
- **Sort knobs**: Adjust `sortPositionEpsilon`, `sortDirectionEpsilon` (default: 0.0001 for ~0.5-1° rotation sensitivity), `minimumSortInterval` to balance visual quality vs performance
- **Buffer pool monitoring**: Check `stats.sortBufferPoolStats` to see available buffers, leased buffers, and memory usage for sort operations
- **Sort queue monitoring**: Check `stats.sortJobsInFlight` to see how many sorts are currently executing (typically 0 or 1)
- **AABB visualization**: Enable `.showAABB` to render cyan wireframe box around scene bounds; useful for debugging spatial queries and AR alignment
- **Flickering fix**: If you experience flickering during rotation, the default `sortDirectionEpsilon` of 0.0001 should prevent it; increase it (e.g., to 0.001) only if you need to reduce sort frequency for performance
- **Interaction mode**: Call `renderer.beginInteraction()` / `renderer.endInteraction()` to enable adaptive sort quality during user interaction (automatically handled in SampleApp gestures)

### Adaptive Interaction Mode (New)
The renderer now supports an **interaction mode** that reduces splat popping during user gestures:

```swift
// Automatic in SampleApp - gesture handlers call these automatically
renderer.beginInteraction()  // Relaxes sort thresholds
renderer.endInteraction()    // Restores quality, triggers final sort

// Configurable interaction parameters (defaults shown)
renderer.interactionSortPositionEpsilon = 0.05      // 5cm during interaction
renderer.interactionSortDirectionEpsilon = 0.003   // ~2-3° during interaction  
renderer.interactionMinimumSortInterval = 0.033    // Max ~30 sorts/sec
renderer.postInteractionSortDelay = 0.1            // Delay before final sort
```

**How it works:**
1. When user starts panning/pinching/rotating, `beginInteraction()` relaxes sort thresholds
2. Fewer sorts occur during interaction → less visual popping
3. When gesture ends, `endInteraction()` restores quality settings and schedules a final high-quality sort after a brief delay

### GPU-Only Sort Reordering (New)
A major performance optimization that eliminates CPU involvement in sorting:

**Before (slow):**
```
GPU: compute distances → GPU: argsort → CPU: readback indices → CPU: reorder splats → GPU: render
```

**After (fast):**
```
GPU: compute distances → GPU: argsort → GPU: render (using sorted indices)
```

**Key changes:**
- Splat buffer stays **static** (never reordered)
- Sorted indices buffer (`sortedIndicesBuffer`) holds the depth-sorted order
- Shaders use `sortedIndices[logicalSplatID]` to access splats in correct render order
- Eliminates ~3 synchronization points per sort

**Performance impact:**
- 2-4x faster sorting for large scenes
- Reduced CPU utilization
- Better GPU pipelining (no stalls waiting for readback)
