# MetalSplatter Gaussian Splatting – Implementation Analysis

Deep dive into how the MetalSplatter Swift package implements Gaussian splatting across its renderer, buffers, shaders, and tooling. The goal is to keep a single markdown summary that mirrors the current architecture of `MetalSplatter/Sources` and its shader resources.

## High-Level Flow
- **Ingest**: `SplatRenderer.read(from:)` streams any supported scene via `SplatIO.AutodetectSceneReader` into `SplatMemoryBuffer`, so huge point clouds avoid full in-memory expansion.
- **Stage**: Points are packed into double-buffered `MetalBuffer<Splat>` pools (`SplatRenderer.swift`) that can grow exponentially but are capped against `device.maxBufferLength`. Optional Fast SH metadata rides alongside in its own pooled buffers.
- **Sort**: Distances are computed on GPU (`ComputeDistances.metal`) or via `MPSArgSort` (`MPSArgSort.swift`) and ordered either by Euclidean distance or forward-vector dot product. LOD constants (`maxRenderDistance`, `lodSkipFactors`) gate how many sorted splats survive to the draw.
- **Render**: A single-stage pipeline (`SingleStageRenderPath.metal`) handles the fast path, while `MultiStageRenderPath.metal` uses imageblock memory to improve depth quality when depth writes and `highQualityDepth` are enabled. Stereo/multi-view is driven by `ViewportDescriptor` instances.
- **Present/AR**: `ARSplatRenderer.swift` composites splats with `ARBackgroundRenderer` output and manages camera pose, placement, and SOGS-v2 coordinate calibration before passing control back to the shared renderer.

## Renderer Core (`MetalSplatter/Sources/MetalSplatter`)
- `SplatRenderer.swift` encapsulates device setup, render targets, and optional callbacks (`onSortStart`, `onRenderComplete`, etc.). It lazy-builds pipeline state objects (`buildSingleStagePipelineState`, `buildInitializePipelineState`, `buildMultiStagePipelineState`) so cost is paid only when needed.
- Sorting is double-buffered: `prepareForSorting` swaps the active `MetalBuffer<Splat>`, `appendSplatForSorting` fills the write buffer, and the render path consumes the previous frame’s buffer. This avoids synchronization stalls between CPU ingestion and GPU consumption.
- LOD and culling live alongside sorting; `maxRenderDistance` drops far splats up front, while directional (forward-vector) sorting keeps front-most fragments early in the list to reduce overdraw.
- The renderer owns a `ViewportDescriptor` per output, packing projection/view matrices and render target sizes. That drives vertex amplification for stereo and feeds into compute culling kernels (`FrustumCulling.metal`).

## Metal Infrastructure & Performance Paths
- **Buffer management**: `MetalBuffer.swift` and `MetalBufferPool.swift` provide CPU-mappable rings with safety checks and automatic growth. Pools recycle allocations by age and heed memory-pressure callbacks to release stale buffers.
- **Command buffers**: `CommandBufferManager` fronts either the legacy allocator or `Metal4CommandBufferPool.swift`, which reuses command buffers on Metal 4 hardware and surfaces stats/hooks for memory pressure handling.
- **Argument buffers / bindless**: `Metal4ArgumentBufferManager.swift` builds `MTLArgumentEncoder` layouts for splat/uniform/index buffers; `Metal4BindlessArchitecture.swift` tracks residency, updates handle tables asynchronously, and binds nothing in the draw loop. Integration layers (`SplatRenderer+Metal4Simple.swift`, `SplatRenderer+BindlessIntegration.swift`, `SplatRenderer+Metal4Integration.swift`) gate everything behind `#available` and `supportsFamily`.
- **Compute preprocessing**: On Apple GPU family 9+, `SplatRenderer+Metal4Integration.swift` adds a compute stage (`SplatProcessing.metal`, `Metal4SIMDGroupOperations.metal`, `Metal4TensorOperations.metal`) to massage splat data (distance, visibility) before rasterization. `Metal4MeshShaders.metal` and `Metal4AdvancedAtomics.metal` are experimental hooks for mesh-shader style amplification and fine-grained depth handling.
- **Profiling**: `GPUPerformanceProfiler.swift` issues synthetic kernels to measure sort, distance, and memory-bandwidth costs so Metal 4 experiments can be validated quantitatively.

## Shader Paths & Depth Handling (`MetalSplatter/Resources`)
- **Single-stage** (`SingleStageRenderPath.metal`): Minimal draw loop that consumes the sorted buffer, expands quads, evaluates SH/color, and blends with the target.
- **Multi-stage** (`MultiStageRenderPath.metal`): Uses imageblock memory to accumulate contributions per tile, enabling better depth precision when depth writes are on and `highQualityDepth` is requested. An initialization pass zeroes imageblock state before accumulation.
- **Preprocess & cull**: `SplatProcessing.metal` and `ComputeDistances.metal` compute distances and visibility; `FrustumCulling.metal` can prune splats per viewport before the render pass.
- **Bindless/argument-buffer variants**: `Metal4BindlessShaders.metal`, `Metal4ArgumentBuffer.metal`, and `Metal4AdvancedAtomics.metal` provide shader-side entry points that match the CPU argument-buffer/bindless setup.
- **Fast SH**: `FastSHRenderPath.metal` shares constants/structs from `ShaderCommon.h` and `SplatProcessing.h`. CPU-side `SphericalHarmonicsEvaluator.swift` mirrors `spherical_harmonics_evaluate.metal` to keep palette evaluation identical on both sides.
- **AR**: `ARMetal4Enhancements.metal` layers AR-specific paths (camera feed composition, depth tweaks) on top of the base shaders.

## Fast SH & Palette-Based Color
- `SplatRenderer+FastSH.swift` introduces `FastSHSplatRenderer`, adding palette buffers and per-splat SH metadata ingestion. The renderer shadows the buffer swap (via overridden `prepareForSorting`) so SH data stays aligned with the active splat buffer.
- `FastSHExample.swift` demonstrates loading `SplatScenePoint` arrays (with SH coefficients) and driving multi-viewport renders, providing a template for app integration.
- CPU/GPU parity is enforced by reusing the same band ordering in `SphericalHarmonicsEvaluator.swift` and `spherical_harmonics_evaluate.metal`; any coefficient-order change must be mirrored in both files.

## Data Ingestion & Format Coverage (SplatIO / PLYIO)
- `SplatIO` supports DotSplat (`DotSplatSceneReader/Writer`), PLY (ASCII/binary via `SplatPLYSceneReader/Writer`), SOGS v1/v2 (`SplatSOGSSceneReader`, `SplatSOGSSceneReaderOptimized`, `SplatSOGSSceneReaderV2`), packed NeRF exports (`SPZ*`, `SPX*`), and zip/bundle helpers (`SplatSOGSZipReader`, `SplatSOGSSceneReader`).
- Readers stream points to `SplatMemoryBuffer` through `SplatSceneReaderDelegate`, enabling the renderer to begin sorting/upload without waiting for full file decode.
- Writers mirror the same coverage (`SplatSceneWriter`, `DotSplatSceneWriter`, PLY writers) so MetalSplatter can round-trip scenes or emit filtered subsets (see `SplatConverter/Sources/SplatConverter.swift` for CLI entry points).

## AR Integration
- `ARSplatRenderer.swift` owns an `ARSession`, feeds camera textures through `ARBackgroundRenderer.swift`, and derives projection/view matrices via `ARPerspectiveCamera.swift`.
- It exposes placement helpers (position/scale/rotation), optional auto-placement heuristics for SOGS v2 scenes, and toggles for Metal 4 bindless/MPP acceleration so AR paths keep feature parity with the main renderer.

## Instrumentation, Debugging, and Lifecycle
- Logging flows through `os.Logger` in buffer pools, bindless managers, and renderer callbacks for observability.
- Public hooks on the renderer (`onSortStart`, `onRenderComplete`, `onError`) enable capture pipelines and regression tracking without forking shader code.
- `SampleBoxRenderer/Sources/SampleBoxRenderer.swift` mirrors the API but renders a cube with its own shader set, making it a drop-in for integration debugging when splat assets are unavailable.

## Practical Takeaways
- Keep CPU/GPU SH evaluation locked in step; adjust `SphericalHarmonicsEvaluator.swift` and `spherical_harmonics_evaluate.metal` together.
- When enabling bindless/argument-buffer paths, ensure availability checks (`supportsFamily`) stay aligned with shader feature sets in `Metal4BindlessShaders.metal` and `Metal4ArgumentBuffer.metal`.
- Multi-stage rendering should be reserved for depth-critical views; otherwise the single-stage path keeps latency and bandwidth lower.
- For enormous clouds, prefer GPU sorting (`MPSArgSort`) and verify buffer pool ceilings against `device.maxBufferLength` to avoid growth failures.
