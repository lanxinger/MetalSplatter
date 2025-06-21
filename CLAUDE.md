# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetalSplatter is a Swift/Metal library for rendering 3D Gaussian Splats on Apple platforms (iOS, macOS, visionOS). It uses Metal for GPU-accelerated rendering and supports multiple splat file formats.

## Common Development Commands

### Building

```bash
# Build all targets (use release mode for performance - debug is >10x slower)
swift build -c release

# Build specific target
swift build --target MetalSplatter -c release

# Open sample app in Xcode
open SampleApp/MetalSplatter_SampleApp.xcodeproj

# Build via xcodebuild
xcodebuild -project SampleApp/MetalSplatter_SampleApp.xcodeproj -scheme "MetalSplatter SampleApp" -configuration Release
```

### Testing

```bash
# Run all tests
swift test

# Run specific test targets
swift test --filter PLYIOTests
swift test --filter SplatIOTests
```

### Tools

```bash
# Build and run the converter tool
swift run SplatConverter --help

# Convert between formats
swift run SplatConverter input.ply output.splat
```

## Architecture Overview

### Core Design Patterns

1. **Protocol-Oriented Architecture**: The `ModelRenderer` protocol is the central abstraction for all renderers. New renderer types implement this protocol rather than inheriting from base classes.

2. **Layered Module Structure**:
   - `PLYIO`: Standalone PLY file I/O
   - `SplatIO`: Interprets PLY/splat files as gaussian splats (depends on PLYIO)
   - `MetalSplatter`: Core Metal rendering engine
   - `SampleApp`: Cross-platform demonstration app

3. **Platform Abstraction**:
   - iOS/macOS use `MetalKitSceneRenderer` with `MTKView`
   - visionOS uses `VisionSceneRenderer` with `CompositorServices`
   - Platform-specific code is isolated using conditional compilation

### Key Components

- **SplatRenderer**: Main renderer implementing GPU-accelerated gaussian splat rendering
- **MetalBuffer<T>**: Type-safe wrapper for Metal buffers
- **SceneReader/Writer**: Protocol-based file I/O system supporting PLY, SPLAT, SPZ, and SOGS formats
- **Shader System**: Modular Metal shaders with shared CPU/GPU definitions in `ShaderCommon.h`

### Performance Considerations

- Always use Release builds for testing with large files
- Render throttling with semaphore (max 3 concurrent frames)
- GPU-accelerated sorting using Metal Performance Shaders
- Double buffering for async operations without blocking renders

### Adding New Features

- **New file formats**: Implement `SplatSceneReader`/`SplatSceneWriter` protocols
- **New renderers**: Implement `ModelRenderer` protocol
- **New platforms**: Add conditional compilation following existing patterns
- **Shader modifications**: Update both Metal shaders and `ShaderCommon.h` for CPU/GPU consistency

### Important Technical Details

- Minimum requirements: Swift 5.9+, iOS 17.0+, macOS 14.0+, visionOS 1.0+
- WebP support (for SOGS format) requires iOS 14+/macOS 11+
- Uses hybrid indexing/instancing to balance performance and memory usage
- Implements frustum culling and distance-based sorting for optimization