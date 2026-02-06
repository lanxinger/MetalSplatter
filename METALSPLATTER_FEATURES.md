# MetalSplatter — Feature Overview (Customer‑Facing)

MetalSplatter is a high‑performance Swift/Metal library for rendering 3D Gaussian Splats on Apple platforms. It’s engineered for real‑time performance, broad format compatibility, and high‑quality depth—ideal for immersive spatial apps, AR experiences, and professional visualization workflows.

## Highlights
- Native Swift + Metal renderer tuned for Apple GPUs across iOS, macOS, and visionOS.
- Multi‑stage, tile‑memory rendering pipeline for high‑quality depth blending (ideal for Vision Pro reprojection).
- GPU‑accelerated sorting options: O(n) counting sort, camera‑relative binning, MPSGraph arg sort fallback, and Metal 4 stable radix sort for large scenes.
- Multiple file formats with auto‑detection: PLY (ASCII/binary), `.splat`, SPZ, SPX, SOGS v1/v2/ZIP.
- Optional order‑independent transparency (dithered) and a faster 2DGS rendering mode.
- Spherical harmonics support (degrees 0–3), plus a fast SH pipeline with palette optimization.
- ARKit renderer that composites splats with the camera feed and supports auto placement and plane detection.
- Metal 3 mesh shaders and Metal 4 bindless/tensor/SIMD‑group paths on supported hardware.
- Strong tooling: CLI converter, data validation, Morton reordering, and GPU profiling utilities.

## What MetalSplatter Enables
MetalSplatter gives teams an end‑to‑end foundation for Gaussian Splat visualization, including I/O, GPU rendering, advanced sorting, and platform‑specific rendering paths. It can be integrated into existing apps or used as a standalone renderer with its own file pipeline and tooling.

## Modules and What Each One Provides
- `MetalSplatter`: Core GPU renderer, sorting systems, LOD/streaming, performance optimizations, debug overlays.
- `SplatIO`: Gaussian splat data model plus readers/writers for multiple formats, validation, compression helpers.
- `PLYIO`: Standalone PLY reader/writer (ASCII and binary).
- `SplatConverter`: Command‑line conversion and inspection tool.
- `SampleBoxRenderer`: Lightweight Metal renderer for integration tests and minimal demo rendering.
- `SampleApp`: Cross‑platform demo app with camera controls, gestures, and format loading.

## Core Rendering Engine
MetalSplatter’s `SplatRenderer` is the main rendering engine and exposes a multi‑viewport, multi‑frame‑in‑flight API suitable for mono or stereo rendering.

Key rendering capabilities include:
- Single‑stage pipeline for high‑throughput rendering.
- Multi‑stage pipeline using tile memory for correct depth blending and stable reprojection depth.
- Configurable depth path with `highQualityDepth` for Vision Pro use cases.
- Dithered transparency option for order‑independent blending without costly sorting.
- 2DGS mode that simplifies splats to circular 2D Gaussian footprints for faster rendering.
- Packed color option (snorm10a2) to reduce bandwidth versus half‑float colors.

## Sorting and Visual Correctness
The renderer provides multiple sorting strategies, selectable at runtime, designed to balance quality and performance:
- Radial sort for rotation‑heavy views.
- Linear sort for translation‑heavy movement.
- Auto mode that selects based on recent camera motion.
- O(n) counting sort with histogram + prefix sum + scatter.
- Camera‑relative binning for higher precision near the camera.
- MPSGraph arg sort fallback when needed.
- Metal 4 stable radix sort path for very large scenes.

## Performance and Scalability
MetalSplatter is built around GPU and memory efficiency for large point clouds:
- Morton ordering to improve GPU cache locality (available on load and in the I/O pipeline).
- Asynchronous sorting on a dedicated compute queue so rendering can overlap.
- Frustum culling on GPU with indirect draw argument generation.
- Configurable LOD thresholds and skip factors to reduce draw load at distance.
- Adaptive interaction mode that relaxes sort thresholds while the user is manipulating the camera.
- Buffer pools that reuse Metal buffers to reduce allocation overhead.
- Command buffer management with Metal 4 pooling and memory‑pressure cleanup hooks.

## Lighting and Appearance
The library supports advanced shading models for splats:
- Spherical harmonics color representation with degrees 0–3.
- Threshold‑based SH updates to avoid recomputing lighting every frame.
- Fast SH renderer that uses SH palette indices for large datasets.
- Option to disable SH evaluation entirely for performance‑first rendering.

## Visibility, LOD, and Streaming
MetalSplatter supports large scenes with visibility‑driven streaming and LOD:
- `SplatOctree` for spatial queries, frustum visibility, and LOD selection.
- `StreamingLODManager` to asynchronously load and unload octree nodes with a memory budget.
- `IntervalManager` to remap active splat ranges for partial scene updates.
- Screen‑space error metrics to choose LOD levels dynamically per node.

## File Formats and Data Pipeline
MetalSplatter includes a full I/O stack with auto‑detection and conversion support.

| Format | Read | Write | Notes |
|---|---|---|---|
| PLY (`.ply`) | Yes | Yes | ASCII and binary, SH supported |
| SPLAT (`.splat`) | Yes | Yes | Compact binary format |
| SPZ (`.spz`, `.spz.gz`) | Yes | Yes | Gzip‑compressed |
| SPX (`.spx`) | Yes | Yes | Extensible format with optional compression |
| SOGS v1 (`.sogs`) | Yes | No | WebP‑based folders |
| SOGS v2 (`.sog`) | Yes | No | Bundled archive with codebook compression |
| SOGS ZIP (`.zip`) | Yes | No | Legacy ZIP bundle |

Additional pipeline capabilities:
- Autodetection of format via `AutodetectSceneReader`.
- Morton reordering during reads for spatial locality.
- Chunked compression format utilities with 256‑splat blocks and ~3.25:1 compression.
- Safe gzip decompression with size limits to prevent decompression bombs.
- Data validation utilities with strict/lenient/safety modes.

## SOGS v2 Enhancements
The library includes a dedicated SOGS v2 reader and metadata support, enabling:
- Codebook compression for scales and colors.
- Bundled `.sog` archives for easier distribution.
- Optional pre‑ordering of splats to reduce runtime sorting.
- Metadata for versioning and scene statistics.

## AR and visionOS Support
MetalSplatter offers native Apple‑platform integrations:
- `ARSplatRenderer` for ARKit sessions on iOS.
- Camera feed compositing via `ARBackgroundRenderer`.
- Auto placement and raycast‑based surface detection for AR scenes.
- Adaptive quality controls for AR tracking conditions.
- Vision Pro rendering through CompositorServices with stereo via vertex amplification.
- Support for `MTLRasterizationRateMap` to enable foveated rendering.

## Metal 3 and Metal 4 Optimizations
MetalSplatter exposes advanced GPU paths when hardware and OS versions allow:
- Mesh shader rendering (Metal 3, Apple GPU Family 7+).
- Metal 4 compute preprocessing with SIMD‑group and tensor paths.
- Bindless argument buffer architecture to reduce per‑draw binding overhead.
- Optional residency tracking and background resource population.
- Metal 4 GPU radix sort path for very large splat counts.

## Debugging and Observability
The library surfaces instrumentation for tuning and diagnostics:
- Debug overlays for overdraw, LOD tinting, and AABB visualization.
- Frame statistics callbacks, including sort duration and buffer upload counts.
- Buffer pool statistics for memory monitoring.
- GPU profiling utility for measuring memory bandwidth and kernel performance.

## Developer Experience and Extensibility
MetalSplatter is designed for clean integration into production apps:
- Swift Package Manager with modular targets.
- Simple API to load splats and render into a Metal command buffer.
- `SplatScenePoint` model with multiple color, opacity, and scale encodings.
- Clear extension points for new file readers/writers and rendering paths.

## Tooling
The included CLI utility enables data pipeline workflows:
- `SplatConverter` converts between formats and inspects splat data.
- Morton reordering option for cache‑friendly output.
- Verbose timing output for benchmarking conversions.

## Platform and Build Requirements
- Swift tools version: 6.2.
- iOS 17+, macOS 14+, visionOS 1+.
- Metal 3+ recommended for mesh shader path.
- Metal 4 optimizations available on Apple GPU Family 9+ with iOS 26+/macOS 26+/visionOS 26+.
- Intel macOS simulator targets are not supported for runtime rendering.

## Typical Use Cases
- Vision Pro and spatial apps needing high‑quality splat rendering.
- Mobile AR visualization with real‑time placement and camera feed compositing.
- Mac desktop visualization tools for 3DGS datasets.
- Research and visualization pipelines requiring multi‑format support and conversion tools.
