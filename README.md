# MetalSplatter

[![Swift 6.0+](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2017%2B%20%7C%20macOS%2014%2B%20%7C%20visionOS%201%2B-blue.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance Swift/Metal library for rendering 3D Gaussian Splats on Apple platforms (iOS, macOS, and visionOS).

![A greek-style bust of a woman made of metal, wearing aviator-style goggles while gazing toward colorful abstract metallic blobs floating in space](http://metalsplatter.com/hero.640.jpg)

MetalSplatter implements GPU-accelerated rendering of scenes captured via [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Load PLY, SPLAT, SPZ, or SOGS files and visualize them with real-time performance across all Apple platforms, including stereo rendering on Vision Pro.

## Features

- **Multi-Platform Support**: iOS/iPadOS, macOS, and visionOS with platform-optimized rendering paths
- **Multiple File Formats**: PLY (ASCII/binary), SPLAT, SPZ (compressed), SOGS (WebP-based)
- **Advanced Rendering Pipeline**: Single-stage and multi-stage pipelines with tile memory for high-quality depth blending
- **GPU-Accelerated Sorting**: O(n) counting sort with camera-relative binning for optimal visual quality
- **Spherical Harmonics**: Full SH support (degrees 0-3) for view-dependent lighting effects
- **Level of Detail**: Distance-based LOD with configurable thresholds and skip factors
- **Metal 4 Support**: Bindless rendering, tensor operations, and SIMD-group optimizations on supported hardware
- **AR Integration**: ARKit support on iOS for augmented reality experiences
- **Vision Pro Stereo**: Vertex amplification for efficient stereo rendering via CompositorServices

## Modules

| Module | Description |
|--------|-------------|
| **MetalSplatter** | Core Metal rendering engine for gaussian splats |
| **PLYIO** | Standalone PLY file reader/writer (ASCII and binary) |
| **SplatIO** | Interprets PLY/SPLAT/SPZ/SOGS files as gaussian splat point clouds |
| **SplatConverter** | Command-line tool for format conversion and inspection |
| **SampleApp** | Cross-platform demo application |
| **SampleBoxRenderer** | Debug renderer for integration testing |

## Installation

### Swift Package Manager

Add MetalSplatter to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/scier/MetalSplatter.git", from: "1.0.0")
]
```

Then add the modules you need to your target:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "MetalSplatter", package: "MetalSplatter"),
        .product(name: "SplatIO", package: "MetalSplatter"),
    ]
)
```

### Xcode Project

1. File → Add Package Dependencies
2. Enter the repository URL: `https://github.com/scier/MetalSplatter.git`
3. Select the modules you need

## Quick Start

### Loading and Rendering Splats

```swift
import MetalSplatter
import SplatIO

// Initialize the renderer
let renderer = try SplatRenderer(
    device: device,
    colorFormat: .bgra8Unorm,
    depthFormat: .depth32Float,
    sampleCount: 1,
    maxViewCount: 2,           // 2 for stereo, 1 for mono
    maxSimultaneousRenders: 3
)

// Load splats from file (auto-detects format)
let reader = try AutodetectSceneReader(url)
let points = try reader.read()
try renderer.add(points)

// In your render loop
try renderer.render(
    viewports: [viewportDescriptor],
    colorTexture: drawable.texture,
    colorStoreAction: .store,
    depthTexture: depthTexture,
    depthStoreAction: .dontCare,
    rasterizationRateMap: nil,
    renderTargetArrayLength: 0,
    to: commandBuffer
)
```

### Reading Specific File Formats

```swift
// PLY files
let plyReader = try SplatPLYSceneReader(url)

// Binary .splat files
let splatReader = try DotSplatSceneReader(url)

// Compressed SPZ files
let spzReader = try SPZSceneReader(url)

// SOGS format (WebP-based)
let sogsReader = try SplatSOGSSceneReaderV2(url)
```

### Writing Splat Files

```swift
// Write to PLY format
let writer = SplatPLYSceneWriter(format: .binary)
try writer.write(points, to: outputURL)

// Write to SPZ format
let spzWriter = SPZSceneWriter()
try spzWriter.write(points, to: outputURL)
```

## Sample App

Try the included sample application to see MetalSplatter in action:

1. Clone the repository
2. Open `SampleApp/MetalSplatter_SampleApp.xcodeproj`
3. Select your target device and set your development team (iOS/visionOS only)
4. **Important**: Use Release configuration for best performance (Debug is >10x slower)
5. Build and run
6. Load a PLY or SPLAT file to visualize

> **Tip**: For best framerate, run without the debugger attached (stop in Xcode, then launch from Home screen).

## Command-Line Tools

### SplatConverter

Convert between splat file formats and inspect splat data:

```bash
# Build the converter
swift build -c release

# Convert PLY to binary SPLAT format
swift run SplatConverter input.ply -o output.splat

# Convert to ASCII PLY
swift run SplatConverter input.ply -f ply-ascii -o output.ply

# Reorder by Morton code for better GPU cache coherency
swift run SplatConverter input.ply -o output.splat --morton-order

# Inspect splat data
swift run SplatConverter input.ply --describe --start 0 --count 10 -v
```

**Options:**
- `-o, --output-file`: Output file path
- `-f, --output-format`: Format (`dotSplat`, `ply`, `ply-binary`, `ply-ascii`)
- `-m, --morton-order`: Reorder splats by Morton code for spatial locality
- `--describe`: Print splat details
- `--start`: First splat index (default: 0)
- `--count`: Maximum splats to process
- `-v, --verbose`: Verbose output with timing

## File Format Support

| Format | Extensions | Read | Write | Notes |
|--------|------------|:----:|:-----:|-------|
| PLY | `.ply` | ✓ | ✓ | ASCII and binary, full SH support |
| SPLAT | `.splat` | ✓ | ✓ | Compact binary format |
| SPZ | `.spz`, `.spz.gz` | ✓ | ✓ | Gzip-compressed format |
| SPX | `.spx` | ✓ | - | Alternative binary format |
| SOGS v1 | `.sogs` | ✓ | - | WebP-based with metadata |
| SOGS v2 | `.sog` | ✓ | - | Bundled archive format |
| SOGS ZIP | `.zip` | ✓ | - | Legacy ZIP archive |

Use `AutodetectSceneReader` for automatic format detection based on file extension and content.

## Configuration

### Rendering Options

```swift
// Multi-stage pipeline for high-quality depth blending
renderer.useMultiStagePipeline = true

// High-quality depth for Vision Pro frame reprojection
renderer.highQualityDepth = true

// Order-independent transparency (no sorting required)
renderer.useDitheredTransparency = true

// Metal 3+ mesh shaders
renderer.meshShaderEnabled = true

// Metal 4 bindless rendering
renderer.useMetal4Bindless = true
```

### Sorting & Performance

```swift
// O(n) counting sort (recommended for large scenes)
renderer.useCountingSort = true

// Camera-relative bin weighting for better near-field precision
renderer.useCameraRelativeBinning = true

// Morton code reordering for GPU cache optimization
renderer.mortonOrderingEnabled = true

// Sorting thresholds (camera movement before re-sorting)
renderer.sortPositionEpsilon = 0.01      // meters
renderer.sortDirectionEpsilon = 0.0001   // ~0.5-1 degree
```

### Interactive Mode

Reduce sorting frequency during user interaction for smoother response:

```swift
// Begin interaction (e.g., on gesture start)
renderer.beginInteraction()

// During interaction, sorting thresholds are relaxed:
// - sortPositionEpsilon: 0.01 → 0.05
// - sortDirectionEpsilon: 0.0001 → 0.003
// - minimumSortInterval: 0 → 0.033 (~30 sorts/sec max)

// End interaction (high-quality sort triggered after delay)
renderer.endInteraction()
```

### Level of Detail

```swift
// Distance thresholds for LOD levels
renderer.lodThresholds = SIMD3<Float>(10, 25, 50)

// Skip factors per LOD level (1 = all, 2 = half, etc.)
renderer.lodSkipFactors = [1, 2, 4, 8]

// Maximum render distance
renderer.maxRenderDistance = 100.0
```

### Spherical Harmonics

```swift
// Update threshold for view-dependent lighting
renderer.shDirectionEpsilon = 0.001   // ~2.5 degree rotation

// Minimum time between SH updates
renderer.minimumSHUpdateInterval = 0.016  // ~60 updates/sec max
```

## Debug & Profiling

### Debug Overlays

```swift
// Visualize overdraw (coverage issues)
renderer.debugOptions.insert(.overdraw)

// Visualize LOD bands
renderer.debugOptions.insert(.lodTint)

// Show axis-aligned bounding box
renderer.debugOptions.insert(.showAABB)
```

### Frame Statistics

```swift
renderer.onFrameReady = { stats in
    print("Ready: \(stats.ready)")
    print("Splat count: \(stats.splatCount)")
    print("Sort duration: \(stats.sortDuration ?? 0)ms")
    print("Frame time: \(stats.frameTime)ms")
    print("Buffer uploads: \(stats.bufferUploadCount)")
}

renderer.onSortComplete = { duration in
    print("Sort completed in \(duration)ms")
}

renderer.onRenderStart = { }
renderer.onRenderComplete = { }
```

## Platform-Specific Notes

### iOS/macOS

- Uses `MTKView` with `MTKViewDelegate` pattern
- Full gesture support (pinch zoom, rotation, panning)
- AR support via ARKit on iOS

### visionOS

- Uses CompositorServices with spatial rendering
- Automatic stereo via vertex amplification
- World tracking integration
- Optimized for Vision Pro display characteristics

### Simulator

The iOS Simulator on Intel Macs (x86_64) is not supported due to Metal limitations.

## Building from Source

```bash
# Clone the repository
git clone https://github.com/scier/MetalSplatter.git
cd MetalSplatter

# Build all targets (release mode recommended)
swift build -c release

# Run tests
swift test

# Build specific target
swift build --target MetalSplatter -c release
```

## Showcase

Apps and projects using MetalSplatter:

- **[MetalSplatter Viewer](https://apps.apple.com/us/app/metalsplatter/id6476895334)** - Official Vision Pro app with camera controls and splat gallery
- **[OverSoul](https://apps.apple.com/app/id6475262918)** - Spatial photos, 3D models, and immersive spaces for Vision Pro

Using MetalSplatter in your project? [Let us know!](https://github.com/scier/MetalSplatter/issues)

## Resources

### Getting Splat Files

- **Capture your own**: Use a camera or drone, then train with [Nerfstudio](https://docs.nerf.studio/nerfology/methods/splat.html)
- **Luma AI**: Capture with the [iPhone app](https://apps.apple.com/us/app/luma-ai/id1615849914), export in "splat" format
- **Original paper data**: [Scene data from the original paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

### Learning More

- [RadianceFields.com](https://radiancefields.com) - News and articles about 3DGS and NeRFs
- [MrNeRF's Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) - Comprehensive research list

### Other Implementations

- [Kevin Kwok's WebGL implementation](https://github.com/antimatter15/splat) ([demo](https://antimatter15.com/splat/))
- [Mark Kellogg's three.js implementation](https://github.com/mkkellogg/GaussianSplats3D) ([demo](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php))
- [Aras Pranckevičius's Unity implementation](https://github.com/aras-p/UnityGaussianSplatting) and blog posts: [1](https://aras-p.info/blog/2023/09/05/Gaussian-Splatting-is-pretty-cool/), [2](https://aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/), [3](https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/)
- [Original reference implementation](https://github.com/graphdeco-inria/gaussian-splatting)

## License

MIT License - Copyright 2023 Sean Cier

See [LICENSE](LICENSE) for details.
