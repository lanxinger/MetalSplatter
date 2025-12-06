# SampleApp Integration: Binned Sorting Toggle

## Overview

The SampleApp now includes a UI toggle for the new camera-relative binned precision sorting feature.

## Changes Made

### Files Modified

1. **[RenderSettings.swift](../SampleApp/Scene/RenderSettings.swift)**
   - Added `@Binding var binnedSortingEnabled: Bool` parameter
   - Added toggle UI for binned sorting in the settings panel
   - Wired up the binding throughout the view hierarchy

2. **[MetalKitSceneRenderer.swift](../SampleApp/Scene/MetalKitSceneRenderer.swift)**
   - Added `setBinnedSorting(_ enabled: Bool)` method
   - Updates `renderer.useBinnedSorting` property when toggled
   - Logs enable/disable events

## User Interface

The settings panel now includes three main toggles:

1. **Fast Spherical Harmonics** - Optimized SH evaluation
2. **Camera-Relative Binned Sorting** ⭐ NEW
   - Description: "Improved sort quality for large scenes (PlayCanvas-inspired)"
   - Default: **Disabled** (for backward compatibility)
3. **Metal 4 Bindless Resources** - Advanced GPU features

## Usage

### In the App

1. Open the SampleApp
2. Tap the **gear icon** (⚙️) in the top-right corner
3. Toggle **"Camera-Relative Binned Sorting"** on/off
4. The change applies immediately to the active renderer

### Default Behavior

```swift
@State private var binnedSortingEnabled = false // Default: OFF
```

- **Disabled by default** to maintain backward compatibility
- Users must explicitly enable it to test
- Recommended for:
  - Large scenes with extreme depth ranges
  - Scenes where near-camera detail is critical
  - Comparing visual quality vs standard sorting

## Testing Workflow

### A/B Comparison

1. Load a large Gaussian splat scene
2. Move camera close to a detailed area
3. Open settings and toggle binned sorting on/off
4. Observe visual quality differences near the camera
5. Compare sort artifacts between modes

### Performance Profiling

Enable Xcode's Metal Frame Capture:
1. Run with binned sorting **disabled**
2. Capture frame, note sort duration
3. Enable binned sorting
4. Capture frame, compare timings
5. Check GPU compute time for distance calculation

### Recommended Test Scenes

- **Landscape scenes**: Test with mountains/distant objects
- **Indoor scenes**: Close-up details (faces, furniture)
- **Mixed depth**: Foreground + far background
- **High splat count**: >1M splats to stress test

## Code Flow

```
User toggles switch
    ↓
RenderSettings.binnedSortingEnabled changes
    ↓
MetalKitRendererViewEnhanced.updateSettings() called
    ↓
MetalKitSceneRenderer.setBinnedSorting(_:) invoked
    ↓
SplatRenderer.useBinnedSorting property updated
    ↓
Next frame: GPU sort uses binned precision
```

## Known Limitations

- Toggle applies to **SplatRenderer** only (not AR or FastSH variants)
- Requires a loaded model to take effect
- No visual indicator of whether binned sorter initialized successfully
- No performance metrics shown in UI (use Xcode profiling)

## Future Enhancements

1. **Status Indicator**: Show if binned sorter is active/available
2. **Performance Stats**: Display sort time in-app
3. **Adaptive Toggle**: Auto-enable for scenes with >500k splats
4. **AR Support**: Extend to `ARSplatRenderer`
5. **Debug Visualization**: Show bin assignments on-screen

## Related Documentation

- [binned-precision-sorting.md](binned-precision-sorting.md) - Technical implementation
- [playcanvas-sorting-comparison.md](playcanvas-sorting-comparison.md) - Comparison analysis
- [RenderSettings.swift](../SampleApp/Scene/RenderSettings.swift) - Settings UI code
- [MetalKitSceneRenderer.swift](../SampleApp/Scene/MetalKitSceneRenderer.swift) - Renderer integration
