# SampleApp: iOS Gaussian Splat Renderer

A sophisticated iOS sample application demonstrating high-performance Gaussian splat rendering using Apple's Metal graphics API. Supports multiple file formats (PLY, SPLAT, SPZ, SOGS) with advanced camera controls and cross-platform compatibility.

## Architecture Overview

### Core Design Patterns
- **Protocol-Oriented Rendering**: `ModelRenderer` protocol enables multiple renderer implementations
- **Multi-Platform Support**: iOS, macOS, and visionOS with platform-specific optimizations
- **MVC + Coordinator Pattern**: Clean separation between UI, rendering, and state management

### Key Components

```
SampleApp/
├── App/                    # Application entry point and constants
├── Model/                  # Rendering protocols and implementations
├── Scene/                  # UI views and Metal scene renderers
└── Util/                   # Mathematical utilities and helpers
```

## Gaussian Splat Rendering Pipeline

### 1. Core Rendering Interface

```swift
public protocol ModelRenderer {
    func render(viewports: [ModelRendererViewportDescriptor],
               colorTexture: MTLTexture,
               colorStoreAction: MTLStoreAction,
               depthTexture: MTLTexture?,
               rasterizationRateMap: MTLRasterizationRateMap?,
               renderTargetArrayLength: Int,
               to commandBuffer: MTLCommandBuffer) throws
}
```

### 2. SplatRenderer Implementation
- Extends MetalSplatter's `SplatRenderer` to conform to `ModelRenderer`
- Handles viewport descriptor adaptation for splat-specific rendering
- Manages multiple file format loading and processing

### 3. Platform-Specific Renderers

#### MetalKitSceneRenderer (iOS/macOS)
- **Metal Infrastructure**: Device, command queue, and render pipeline management
- **Camera System**: Advanced orbital camera with smooth animations
- **Gesture Controls**: Multi-touch support for rotation, zoom, pan, and roll
- **Performance**: Render throttling with semaphore-based concurrency control

#### VisionSceneRenderer (visionOS)
- **ARKit Integration**: World tracking and device anchor positioning
- **Spatial Computing**: Immersive space rendering with stereo viewports
- **CompositorServices**: Native visionOS rendering framework integration

## File Format Support

| Format | Description | Features |
|--------|-------------|----------|
| **PLY** | Standard 3D Gaussian splat format | ASCII/Binary point cloud data |
| **SPLAT** | Optimized binary format | Compressed, fast loading |
| **SPZ** | Compressed splat format | Space-efficient storage |
| **SOGS v1** | Compressed format with WebP textures | Folder-based, metadata JSON |
| **SOGS v2** | Enhanced compressed format | Bundled .sog files, codebook compression |

### SOGS v1 Format Structure
```
model_folder/
├── meta.json           # Configuration and metadata  
├── means_l.webp       # Position data (low bytes)
├── means_u.webp       # Position data (high bytes)
├── scales.webp        # Scale data
├── quats.webp         # Orientation data
├── sh0.webp           # Color/opacity data
├── shN_centroids.webp # Optional: SH centroids
└── shN_labels.webp    # Optional: SH labels
```

### SOGS v2 Format Structure
```
model.sog              # Single bundled ZIP file containing:
├── meta.json          # v2 metadata with codebooks
├── means_l.webp       # Position data (low bytes)
├── means_u.webp       # Position data (high bytes)
├── scales.webp        # Scale indices (codebook-based)
├── quats.webp         # Orientation data
├── sh0.webp           # Color indices (codebook-based)
├── shN_centroids.webp # Optional: SH centroids
└── shN_labels.webp    # Optional: SH palette indices
```

### SOGS v2 Improvements
- **Codebook Compression**: 256-entry k-means codebooks for scales and colors
- **Single File Distribution**: Bundled .sog ZIP archives for easy sharing
- **Morton Code Ordering**: Optimized spatial ordering eliminates runtime sorting
- **Enhanced Metadata**: Version tracking, splat count, antialiasing flags

## Interactive Camera Controls

### Gesture Recognition System
- **Single-finger drag**: Orbital rotation around model center
- **Two-finger drag**: Panning/translation in screen space
- **Pinch gesture**: Zoom with constraints (0.2x to 5.0x range)
- **Two-finger rotation**: Roll rotation around Z-axis
- **Double-tap**: Animated reset to default viewing position

### Advanced Camera Features
- **Auto-rotation**: Configurable automatic model rotation when idle
- **Smooth Animations**: Eased transitions for view reset operations
- **State Persistence**: Camera position maintained across interactions
- **Interpolated Reset**: Smooth animated return to default state

## Performance Optimizations

### Concurrency Management
```swift
// Render throttling prevents GPU overload
private let renderSemaphore = DispatchSemaphore(value: 3)

// Background loading prevents UI blocking
Task.detached {
    let splat = try SplatRenderer(device: device, ...)
    try await splat.read(from: url)
    await MainActor.run {
        self.modelRenderer = splat
    }
}
```

### Memory Management
- **Security Scoped Resources**: Proper file access in sandboxed environments
- **MTLTexture Reuse**: Efficient texture management for render targets
- **Automatic Cleanup**: Resource cleanup after access timeouts

### Platform-Specific Optimizations
- **iOS**: Volume button integration, gesture recognizer chaining
- **visionOS**: Foveated rendering support, layered rendering modes
- **macOS**: Window-based rendering with display scaling

## Mathematical Foundations

### Matrix Transformations
```swift
// Perspective projection matrix
func perspectiveMatrix(fov: Float, aspect: Float, near: Float, far: Float) -> float4x4

// View transformation matrix combining rotation, translation, and zoom
let transformMatrix = translationMatrix * rotationMatrix * zoomMatrix
```

### Camera State Management
```swift
struct CameraState {
    var rotation: Angle = .zero
    var verticalRotation: Float = 0.0
    var rollRotation: Float = 0.0
    var zoom: Float = 1.0
    var translation: SIMD2<Float> = .zero
}
```

## Integration with MetalSplatter

### Dependencies
- **MetalSplatter**: Core Gaussian splat rendering engine
- **SplatIO/PLYIO**: File format I/O libraries
- **Metal/MetalKit**: Apple's graphics framework

### Renderer Adaptation
```swift
extension SplatRenderer: ModelRenderer {
    public func render(viewports: [ModelRendererViewportDescriptor], ...) throws {
        // Convert viewport descriptors to SplatRenderer format
        let splatViewports = viewports.map { viewport in
            SplatRenderer.ViewportDescriptor(
                viewport: viewport.viewport,
                projectionMatrix: viewport.projectionMatrix,
                worldToCameraTransform: viewport.viewMatrix.inverse,
                screenSize: viewport.screenSize
            )
        }
        
        // Delegate to underlying SplatRenderer
        try render(viewports: splatViewports, ...)
    }
}
```

## Usage Examples

### Basic Setup
```swift
// Initialize Metal infrastructure
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Create and configure MetalKit view
let metalKitView = MTKView()
metalKitView.device = device
metalKitView.colorPixelFormat = .bgra8Unorm_srgb
metalKitView.depthStencilPixelFormat = .depth32Float

// Create scene renderer
let sceneRenderer = MetalKitSceneRenderer(device: device, view: metalKitView)
metalKitView.delegate = sceneRenderer
```

### Loading Gaussian Splat Models
```swift
// Async model loading
Task {
    do {
        let splat = try SplatRenderer(device: device, ...)
        try await splat.read(from: modelURL)
        
        await MainActor.run {
            sceneRenderer.modelRenderer = splat
        }
    } catch {
        print("Failed to load model: \(error)")
    }
}
```

### Gesture Integration
```swift
// Pan gesture for orbital rotation
let panGesture = UIPanGestureRecognizer { gesture in
    let translation = gesture.translation(in: view)
    sceneRenderer.updateRotation(deltaX: Float(translation.x),
                                deltaY: Float(translation.y))
}

// Pinch gesture for zoom
let pinchGesture = UIPinchGestureRecognizer { gesture in
    sceneRenderer.updateZoom(scale: Float(gesture.scale))
}
```

## Build Requirements

### Xcode Configuration
- **iOS Deployment Target**: iOS 17.0+
- **visionOS Support**: visionOS 1.0+
- **Metal Feature Set**: iOS GPU Family 4 or higher
- **SwiftUI**: Required for UI components

### Package Dependencies
```swift
dependencies: [
    .package(path: "../"), // MetalSplatter package
]
```

## Performance Considerations

### GPU Memory Management
- Models are loaded into GPU memory once and reused
- Texture atlases optimize memory usage for complex splats
- Automatic LOD (Level of Detail) based on viewing distance

### Rendering Optimization
- **Frustum Culling**: Only render visible splats
- **Depth Testing**: Proper Z-buffer usage for correct occlusion
- **Batched Rendering**: Multiple splats rendered in single draw calls

### Battery Efficiency
- **Adaptive Frame Rate**: Reduces rendering frequency when static
- **Thermal Management**: Automatic quality reduction under thermal pressure
- **Background Handling**: Pauses rendering when app backgrounded

## Future Enhancements

### Potential Features
- **Real-time Lighting**: Dynamic lighting effects on Gaussian splats
- **Animation Support**: Temporal Gaussian splats for video content
- **AR Integration**: Placement of splats in real-world environments
- **Editing Tools**: Runtime modification of splat properties
- **Networking**: Remote model loading and streaming

### Architecture Extensions
- **Plugin System**: Support for custom renderer implementations
- **Scripting Interface**: Runtime behavior modification
- **Analytics Integration**: Performance monitoring and optimization
- **Export Capabilities**: Convert between different splat formats

---

This sample application demonstrates production-ready Gaussian splat rendering with sophisticated camera controls, multi-platform support, and optimized Metal performance. The architecture is extensible and can be easily integrated into larger applications requiring 3D Gaussian splat visualization capabilities.