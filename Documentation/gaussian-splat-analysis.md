# PlayCanvas Gaussian Splatting – Implementation & Optimizations

## Architecture
- Public surface is `GSplatComponent` / `GSplatComponentSystem` with optional unified mode (`src/framework/components/gsplat`); non-unified uses a per-entity sorter, unified funnels everything through a global director/manager (`src/scene/gsplat-unified`).
- Core resources live in `src/scene/gsplat` and serve both paths; unified renderer adds work-buffer, global sort, and octree LOD streaming.

## Data Representation & Upload
- Base resource (`src/scene/gsplat/gsplat-resource-base.js`) builds a quad mesh that packs 128 splats per instance to raise VS occupancy and keeps a shared instancing buffer; caches work-buffer render info per define set to avoid shader churn.
- Standard data (`gsplat-resource.js`): colors stored as RGBA16F with sigmoid opacity; transforms packed into RGBA32U/16F (position + quat.xy, scale + quat.z) to minimize bandwidth; SH coefficients quantized into 21/10-bit fields in RGBA32U textures (bands 1–3) and normalized per-splat.
- Compressed format (`gsplat-compressed-resource.js`): packed vertex data and chunk metadata textures; optional SH textures; chunkTexture padding fixed if min/max absent. Still exposes centers for sorting.
- SOGS format (`gsplat-sogs-data.js` + `gsplat-sogs-resource.js`): GPU reorders raw per-attribute textures into tightly packed RGBA32U/RGBA8 via fullscreen passes, supports versioned codebooks and optional SH centroids; optional `minimalMemory` drops source textures after packing; repacks on device restore.
- Instance meshes/aabbs are shared and ref-counted; per-resource `workBufferRenderInfos` are cached keyed by defines (intervals, color-only, color format).

## Unified Renderer Pipeline
- Director (`gsplat-director.js`) tracks cameras/layers, creating a `GSplatManager` per camera+layer and firing `material:created` events for shader customization.
- Manager (`gsplat-manager.js`) aggregates placements (plain + octree instances) into versioned `GSplatWorldState`s. World state binary-searches the smallest square texture that fits active splats and assigns contiguous line ranges to each splat, tracking padding and total used pixels for sorter/renderer.
- Work buffer (`gsplat-work-buffer.js`): MRT of color (RGBA16F fallback to RGBA16U) + two integer textures holding centers/covariance; uses `UploadStream` for non-blocking uploads; WebGPU uses a storage buffer for order, WebGL uses R32U texture. Has a color-only pass to update SH without rewriting geometry.
- Renderer (`gsplat-renderer.js`) draws from the work buffer using a single instanced mesh per camera/layer; caches viewport per camera (VR-aware) and uses a custom isVisibleFunc to keep splats camera-specific. Supports overdraw visualization with additive blend.
- Incremental updates: transforms that changed re-render only those splats; SH color updates can render color-only to save bandwidth; camera tracking avoids redundant updates.
- Frame readiness: manager emits `frame:ready` with `ready` and `loadingCount` (octree loads) for capture pipelines.

## Sorting Strategy
- Unified sorter (`gsplat-unified-sorter.js` + worker) maintains a centers cache keyed by resource id; reuses order buffers to avoid allocations; throttles jobs (`jobsInFlight`) and only dispatches when a new world version or no pending job.
- Worker (`gsplat-unified-sort-worker.js`) performs counting sort in a memory/bandwidth-friendly way:
  - Computes effective min/max distance per splat from transformed AABBs (linear) or radial camera distance (cubemap mode).
  - Uses 32 camera-relative bins with weighted bit budgets to increase precision near the camera; bin bases/dividers are derived per frame from camera position/bin.
  - Supports radial or directional sorting; intervals let it skip excluded splats (half-open ranges built in `gsplat-info.js` and GPU remapped via `gsplat-interval-texture.js`).
  - Counting sort runs over “used pixels” only (excluding padding) and returns an order buffer transferred back to the main thread.
- Sort triggers are throttled: only when camera moved/rotated past epsilon (radial vs directional), splats moved, or parameters changed; if workers are backed up, the manager delays full updates.

## LOD Streaming & Octree Handling
- Octree resources (`gsplat-octree.js`) map files to LOD levels, keep per-file refcounts, and use cooldown ticks before unloading to avoid thrash; environment resources tracked separately.
- Instances (`gsplat-octree-instance.js`) select LOD per node using camera-local distance with optional behind-camera penalty and clamped range; underfill allows temporarily coarser LODs if finer data not resident; prefetches advance one LOD toward optimal to avoid bursty requests.
- Pending transitions track old/new file indices so refcounts stay correct even if LOD targets change mid-load; pending-visible adds avoid dropping current data until replacements arrive.
- Files load via `GSplatAssetLoader` queue with retry/max-concurrency and staged release via `pendingReleases` once a new world state is sorted; `updateCooldownTick` decrements refcounts after the configured cooldown.

## Color / SH Update Policy
- `GSplatInfo` tracks camera movement accumulation; SH-enabled splats update color only when translation/rotation thresholds are exceeded, scaled per LOD (`colorUpdateDistanceLodScale`, `colorUpdateAngleLodScale`). Accumulators start at random fractions to stagger work.
- Full re-renders reset accumulators; color-only updates use the dedicated work-buffer pass. Managers batch both sets per frame to amortize draw overhead.

## Work-Buffer/Shader Details
- Work-buffer copy pass (`gsplat-work-buffer-render-pass.js` + `gsplatCopyToWorkbuffer` shader) converts source formats (raw, compressed, SOGS) into a unified layout, with optional interval remapping and per-LOD debug tinting.
- Renderer shader reads packed covariance from integer textures, supports RGBA16U color fallback via defines; `GSPLAT_WORKBUFFER_DATA` path removes source-format conditionals. Overdraw debug uses an optional color ramp texture.
- Interval textures are GPU-generated (prefix sums + remap) to avoid CPU-side per-splat remapping.

## Parameter Knobs / Debugging
- Global params (`gsplat-params.js`) cover radial sorting, LOD update thresholds, behind-camera penalty, allowed LOD range/underfill, overdraw color ramp, and debug colorization (LOD or color-update visualization). Changing params flips a dirty flag to rebuild world state.
- Debug draws: optional AABBs per splat or octree node, and per-LOD colors; `frame:ready` helps deterministic captures.

## Legacy Per-Entity Path
- Non-unified mode (`gsplat-instance.js` + `gsplat-sorter.js`) keeps a dedicated order texture and sorter worker per splat, still uses instancing and order updates but lacks global sorting/LOD streaming. Useful for simple scenes or when unified sorting is unnecessary.
