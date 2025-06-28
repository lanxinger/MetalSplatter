# Fast SH Integration Guide

This guide explains how to use the Fast Spherical Harmonics (SH) implementation in the MetalSplatter SampleApp.

## What is Fast SH?

Fast SH is an optimization technique that pre-computes spherical harmonics lighting once per frame using the camera direction, instead of evaluating it for each gaussian splat individually. This provides significant performance improvements with minimal visual quality loss.

## Key Benefits

- **Performance**: Up to 24% faster rendering (e.g., 45ms → 34ms GPU time)
- **Memory Efficiency**: Reduces SH data from 45MB to ~5MB for 1M splats using palette compression
- **Quality**: Most accurate at screen center, acceptable quality at edges
- **Compatibility**: Seamlessly works with existing SOGS files

## How to Use

### 1. Loading Files with Fast SH

The SampleApp now automatically uses Fast SH for compatible files:

```swift
// Fast SH is enabled by default for SOGS files and large PLY files
// The system automatically detects SH data and applies recommended settings
```

### 2. Settings Interface

When viewing a 3D scene, you'll see:

- **Settings Gear Icon**: Tap to access Fast SH configuration
- **Fast SH Status Indicator**: Shows when Fast SH is active with performance info

### 3. Configuration Options

In the settings sheet, you can control:

- **Enable Fast SH**: Toggle the optimization on/off
- **Use Texture Evaluation**: Better edge accuracy (uses more GPU memory)
- **Update Frequency**: How often to recalculate SH (1-10 frames)
- **Max Palette Size**: Maximum unique SH coefficient sets (1K-128K)

### 4. Performance Information

The status indicator shows:
- Active/inactive status
- Palette size (number of unique SH sets)
- SH degree (0-3, higher = more complex lighting)
- Estimated performance gain

## Recommended Settings

The app automatically applies recommended settings based on your model:

### Small Models (< 10K splats)
- Fast SH: Disabled (minimal benefit)
- Uses traditional per-splat evaluation

### Medium Models (10K-500K splats)
- Fast SH: Enabled
- Buffer-based evaluation
- Update every frame

### Large Models (> 500K splats)
- Fast SH: Enabled
- Texture-based evaluation for better accuracy
- Update every 2 frames for performance

### SOGS Files
- Always recommended to enable Fast SH
- Leverages existing palette compression
- Significant memory and performance benefits

## File Format Support

### Fully Supported
- **SOGS**: Best performance, uses palette compression
- **PLY with SH data**: Works with multi-band spherical harmonics

### Partial Support
- **PLY without SH**: Falls back to regular rendering
- **SPLAT/SPZ**: Limited SH support depending on format variant

## Technical Details

### SH Evaluation Modes

1. **Fast Buffer Mode**
   - Pre-computes all palette entries into a buffer
   - Single camera direction per frame
   - Best for most use cases

2. **Texture Mode**
   - Stores evaluations in a texture
   - Better edge accuracy
   - Higher GPU memory usage

3. **Disabled Mode**
   - Traditional per-splat evaluation
   - Highest quality, lowest performance
   - Fallback for compatibility

### Performance Characteristics

- **Memory Usage**: Dramatically reduced for large models
- **GPU Compute**: Trades per-splat work for per-frame work
- **Quality Trade-off**: Minimal at screen center, slight reduction at edges
- **Update Cost**: Configurable frequency balances quality vs performance

## Troubleshooting

### Fast SH Not Activating
- Ensure the model has spherical harmonics data
- Check that Fast SH is enabled in settings
- Verify the model has sufficient splat count (>10K recommended)

### Performance Issues
- Try increasing update frequency (update less often)
- Switch from texture to buffer mode
- Reduce max palette size for memory-constrained devices

### Visual Artifacts
- Disable Fast SH for highest quality
- Increase update frequency to every frame
- Switch to texture mode for better edge accuracy

## Developer Integration

To integrate Fast SH in your own app:

```swift
// Create Fast SH renderer
let renderer = try FastSHSplatRenderer(device: device, ...)

// Configure settings
renderer.fastSHConfig.enabled = true
renderer.fastSHConfig.useTextureEvaluation = false
renderer.fastSHConfig.updateFrequency = 1

// Load with SH support
try await renderer.loadSplatsWithSH(splats)

// Use FastSHSettings for UI binding
let settings = FastSHSettings()
settings.connect(to: renderer)
```

## Future Enhancements

Planned improvements include:
- Per-pixel view direction evaluation
- Adaptive quality based on viewing distance
- Multi-view SH evaluation for stereo rendering
- Integration with level-of-detail systems