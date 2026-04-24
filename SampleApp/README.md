# SampleApp: iOS/iPadOS Gaussian Splat Editor Demo

`SampleApp` is the reference iPhone and iPad app for loading, rendering, editing, and exporting gaussian splat scenes with `MetalSplatter`.

The app still contains some shared rendering infrastructure for other Apple platforms, but the current interactive editing workflow is intentionally `iOS/iPadOS` first.

## What It Demonstrates

- `SplatRenderer` integrated into a touch-driven `MTKView`
- `SplatEditor` layered on top of the renderer for editing state and undo/redo
- GPU-assisted selection for 2D and 3D selection queries
- Direct-touch transform editing for move, rotate, and scale
- Round-trip export of edited splat scenes through `SplatIO`

## Editing Tooling

The iOS/iPadOS demo exposes these tools in the editing toolbar:

- Selection: point, rect, brush, lasso, flood, eyedropper/color-match, sphere, box, polygon, and measure
- Editing: move, rotate, scale, hide, unhide all, lock, unlock all, delete, restore deleted, duplicate, separate, undo, redo, and export
- Selection utilities: replace/add/subtract combine modes plus all/none/invert helpers

Renderer feedback in the demo includes:

- selection tint
- outline pass for selected splats
- locked-splat tinting
- the same edit-state and preview-transform rendering on both standard and Fast SH scenes

## Architecture

### Main pieces

- `Scene/MetalKitSceneRenderer.swift`
  Owns Metal setup, file loading, camera state, and the active `SplatRenderer` / `SplatEditor`.
- `Scene/MetalKitSceneView.swift`
  Hosts the SwiftUI UI, editing toolbar, parameter panels, and the overlay used for rect/brush/lasso/polygon/measure interaction.
- `MetalSplatter/Sources/SplatEditor.swift`
  Provides the editor actor used by the app for selection, transforms, visibility changes, history, and export.

### Editing flow

1. Load a supported splat file into `SplatRenderer`.
2. Create one `SplatEditor` for the active scene.
3. Route touch gestures or overlay shapes into editor selection queries.
4. Use preview transforms for move/rotate/scale, then commit or cancel.
5. Export the visible edited points through `SplatIO`.

Fast SH scenes do not use a separate editor path. The sample app applies edit-state buffers, preview transforms, and animation overlays through the same optimized renderer that draws the base scene.

## File Format Support In The Demo

The sample app can load the same formats supported by `AutodetectSceneReader`:

| Format | Extensions | Notes |
|--------|------------|-------|
| PLY | `.ply` | ASCII and binary |
| SPLAT | `.splat` | Compact binary format |
| SPZ | `.spz`, `.spz.gz` | Compressed splat container |
| SPX | `.spx` | Alternative binary format |
| glTF | `.gltf` | `KHR_gaussian_splatting` |
| GLB | `.glb` | Binary `KHR_gaussian_splatting` |
| SOGS v1 | `.sogs`, `meta.json` | Read-only in the library |
| SOGS v2 | `.sog` | Read/write in the library |
| SOGS ZIP | `.zip` | Legacy read-only archive |

The export action in the editing UI writes the currently visible edited scene using the library’s scene writers.

## Touch Interaction Model

### Camera

- One-finger drag: orbit camera when no transform tool is active
- Two-finger drag: pan camera
- Pinch: zoom camera, or scale selection when the scale tool is active
- Rotation gesture: roll camera, or rotate selection when the rotate tool is active
- Double-tap: reset camera

### Editing

- Tap: point-select, flood-select, eyedropper/color-match, or measure depending on the active tool
- Drag in overlay: rect, brush, or lasso selection
- Polygon: tap vertices, then close the shape from the first point or toolbar
- Move tool: drag selected splats in the camera plane
- Sphere/box: use toolbar parameters centered on the current selection bounds, or visible bounds if nothing is selected

## Running The Demo

1. Open `SampleApp/MetalSplatter_SampleApp.xcodeproj`.
2. Select an iPhone or iPad destination.
3. Use `Release` configuration for meaningful rendering performance.
4. Build and run.
5. Load a supported splat asset from Files.
6. Open the editing toolbar and start selecting or transforming splats.

## Notes

- The demo is optimized for local editing workflows, not scene-authoring parity with every desktop splat editor.
- `Separate` currently operates within the active editing session rather than turning one file into multiple in-scene objects.
- Some renderer-backed tests are skipped in plain SwiftPM environments because Metal shader libraries are not always available there; the Xcode app path remains the best place to validate the full editing UX.
