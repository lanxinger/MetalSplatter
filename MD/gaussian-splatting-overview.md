# Gaussian Splatting Support in PlayCanvas

This document captures the current state (Codex analysis, 2024-XX-XX) of every engine module involved in Gaussian Splatting (GSplat) rendering. Use it as a baseline for future updates.

---

## 1. API Surface (Components & Systems)

- `src/framework/components/gsplat/component.js`  
  *Legacy vs unified rendering, asset wiring, per-entity controls.*  
  - `GSplatComponent#unified` switches between simple instances and the unified pipeline, and is only mutable while the component is disabled (`:393-412`).  
  - Handles `material`, `highQualitySH`, `castShadows`, custom AABBs, and `lodDistances`, while keeping `_instance` vs `_placement` mutually exclusive (`:248-384`).  
  - Accepts gsplat assets only via the asset system; manual creation is blocked in favor of `AssetReference` callbacks (`:169-226`).  

- `src/framework/components/gsplat/system.js`  
  *Owns the runtime system, installs shader chunks, and exposes events.*  
  - Registers GLSL/WGSL chunk collections (`gsplatChunksGLSL/WGSL`) during construction.  
  - Emits `material:created` and `frame:ready` (docstrings at `:41-88`).  
  - Stores a `GSplatDirector` instance on `app.renderer` and forwards `getGSplatMaterial(camera, layer)` to unified managers (`:101-190`).

- `src/scene/layer.js`  
  *Layers track splat placements so unified rendering can gather them.*  
  - Maintains `gsplatPlacements` array with dirty flag, plus `addGSplatPlacement` / `removeGSplatPlacement` helpers (`:179-518`).

## 2. Asset Ingestion & Data Formats

- `src/framework/handlers/gsplat.js`  
  *Routes file extensions to specific parsers (`ply`, `sog`, `json`, `lod-meta.json`).*

- Parsers  
  - `src/framework/parsers/ply.js` — Reads raw `.ply` splats (chunk data, packed vertices, optional SH coefficients).  
  - `src/framework/parsers/sogs.js` — Downloads meta JSON + texture atlases, handles legacy format upgrade, creates `GSplatSogsData`.  
  - `src/framework/parsers/sog-bundle.js` — Decompresses `.sog` zip bundles, inflates files, parses bundled `meta.json`.  
  - `src/framework/parsers/gsplat-octree.js` — Fetches streaming LOD metadata (`lod-meta.json`) and instantiates `GSplatOctreeResource`.

- `src/framework/components/gsplat/gsplat-asset-loader.js`  
  *Programmatic loader with concurrency limits, retries, and bookkeeping for the unified director.*

## 3. Data Containers & GPU Resources

- `src/scene/gsplat/gsplat-data.js`, `gsplat-compressed-data.js`, `gsplat-sogs-data.js`  
  *Per-format iterators, AABB calculation, decompression helpers, and center extraction.*

- `src/scene/gsplat/gsplat-resource-base.js`  
  - Creates instancing meshes, shared GPU textures, and caches `WorkBufferRenderInfo` objects used by the unified work buffer.  
  - Concrete resources:  
    - `gsplat-resource.js` (raw data, color + transform + SH textures).  
    - `gsplat-compressed-resource.js` (quantized packed textures + chunk LUT).  
    - `gsplat-sogs-resource.js` (SOGS GPU packing, meta uniform setup).

## 4. Classic (Non-Unified) Rendering Path

- `src/scene/gsplat/gsplat-instance.js`  
  *Owns a `MeshInstance`, per-camera `GSplatSorter`, and optional `GSplatResolveSH` evaluator.*  
  - Sorter worker updates the order texture whenever camera position/forward changes.  
  - `setHighQualitySH` toggles SH resolve passes for SOGS data.

- `src/scene/gsplat/gsplat-sorter.js` & `gsplat-sort-worker.js`  
  *Texture-backed counting-sort implementation with binning, mapping, and chunk hints.*

- `src/scene/gsplat/gsplat-resolve-sh.js`  
  *Evaluates SH palettes into RGBA8 textures for SOGS resources and injects shader chunks.*

- `src/scene/gsplat/gsplat-material.js`  
  *Utility to build standalone splat materials with custom defines/chunks.*

## 5. Unified Rendering & Work Buffer

- `src/scene/gsplat-unified/gsplat-director.js`  
  *Tracks all cameras/layers, spawns `GSplatManager`s, and fires stats events.*

- `src/scene/gsplat-unified/gsplat-manager.js`  
  *Central brain: reconciles placements, builds `GSplatWorldState`s, schedules sorts/renders, handles SH/color update thresholds, and drives octree streaming.*

- Supporting modules:  
  - `gsplat-world-state.js` — Packs splats into atlas rows/lines for the work buffer and keeps versioned states.  
  - `gsplat-work-buffer.js` & `gsplat-work-buffer-render-pass.js` — MRT textures (color + covariance + intervals) plus color-only refresh path and GPU upload helpers.  
  - `gsplat-interval-texture.js` — GPU remapping of sparse intervals into dense target indices.  
  - `gsplat-renderer.js` — Consumes the work buffer, applies dithering/overdraw modes, and pushes the instanced quad to the layer.

- Sorting stack:  
  - `gsplat-unified-sorter.js` — Manages a worker pool, provides `setSortParameters`/`setSortParams`, tracks pending jobs.  
  - `gsplat-unified-sort-worker.js` — Camera-relative bucket allocation, directional vs radial sorting, interval/padding aware counting-sort.

## 6. LOD Streaming & Octree Management

- `src/scene/gsplat-unified/gsplat-placement.js`  
  *Wraps a resource + node + per-node intervals (used by LOD nodes and classic splats).*

- `src/scene/gsplat-unified/gsplat-octree.resource.js` & `gsplat-octree.js`  
  *Octree metadata, per-file refcounts/cooldowns, environment resource handling, and debug tracing.*

- `src/scene/gsplat-unified/gsplat-octree-instance.js`  
  - Chooses desired LOD based on camera distance, behind-camera penalties, and underfill settings.  
  - Prefetches finer LODs step-by-step, tracks pending loads, and updates placements once resources land.  
  - Emits debug AABBs per node when requested.

- `src/scene/gsplat-unified/gsplat-asset-loader-base.js`  
  *Abstract API consumed by the director and octree instances for loading/unloading streamed GSplat files.*

- `src/scene/gsplat-unified/gsplat-params.js`  
  *Global knobs (LOD distance/angle thresholds, radial sorting, overdraw visualization, SH color update cadence, debug colorization).*

## 7. Shader Infrastructure & Scripts

- Shader chunks registered via `gsplatChunksGLSL/WGSL` cover:  
  - Vertex/fragment programs (`gsplatVS/PS`).  
  - Compression readers, SH helpers, unified work-buffer copy, interval remapping, SOGS palette evaluation, etc. (see `src/scene/shader-lib/{glsl,wgsl}/chunks/gsplat/**`).

- Scripts & Effects (`scripts/esm/gsplat/`)  
  - `gsplat-shader-effect.mjs` — Base class for attaching GLSL/WGSL overrides to gsplat components. Works in both unified and non-unified modes by listening for `material:created` and retrying shader application per camera/layer.  
  - Effect implementations (reveal-grid-eruption, reveal-radial, reveal-rain, shader-effect-box) derive from the base class and provide concrete shader sources/uniform updates.

## 8. References & Next Steps

- **Testing/Examples:** engine examples linked in `GSplatComponent` JSDoc (`simple`, `global-sorting`, `lod`, `lod-instances`, `lod-streaming`, `lod-streaming-sh`, `multi-splat`, `multi-view`, `picking`, `reveal`, `shader-effects`, `spherical-harmonics`).  
- **Where to watch for changes:**  
  - Unified pipeline: `src/scene/gsplat-unified/**/*`.  
  - Classic pipeline: `src/scene/gsplat/**/*`.  
  - Asset ingestion: `src/framework/{handlers,parsers}/`.  
  - Shader chunks: `src/scene/shader-lib/*/chunks/gsplat/`.  
  - Scripts/creator tooling: `scripts/esm/gsplat/`.

Keep this document updated whenever GSplat modules change (new shader chunks, pipeline adjustments, loader behaviors, etc.) to preserve institutional knowledge and simplify regression tracking.
