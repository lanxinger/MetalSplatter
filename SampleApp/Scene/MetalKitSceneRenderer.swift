#if os(iOS) || os(macOS)

@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalSplatter
import os
import SampleBoxRenderer
import simd
import SplatIO
import SwiftUI

// Fast SH is only available when MetalSplatter includes it
#if canImport(MetalSplatter)
// FastSHSettings and FastSHSplatRenderer should be available
#endif

// Helper function for linear interpolation
private func lerp<T: FloatingPoint>(_ a: T, _ b: T, _ t: T) -> T {
    return a + (b - a) * t
}

// Helper function for Angle interpolation
private func lerp(_ a: Angle, _ b: Angle, _ t: Double) -> Angle {
    .radians(lerp(a.radians, b.radians, t))
}

// Helper function for SIMD2<Float> interpolation
private func lerp(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ t: Float) -> SIMD2<Float> {
    return a + (b - a) * t
}

// Simple ease-in-out easing function
private func smoothStep<T: FloatingPoint>(_ t: T) -> T {
    return t * t * (3 - 2 * t)
}

@MainActor
class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let commandBufferManager: CommandBufferManager

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?
    
    // Track last logged model to avoid spam
    private var lastLoggedModel: String?
    
    // Fast SH Support
    var fastSHSettings = FastSHSettings()
    
    // Metal 4 Bindless Support
    var useMetal4Bindless: Bool = true // Default to enabled

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero
    var zoom: Float = 1.0
    private var userIsInteracting = false
    // Add vertical rotation (pitch)
    var verticalRotation: Float = 0.0
    // Add roll rotation (2-finger twist)
    var rollRotation: Float = 0.0
    // Add translation for panning
    var translation: SIMD2<Float> = .zero
    private var modelScale: Float = 1.0
    private var autoFitEnabled: Bool = true

    var drawableSize: CGSize = .zero

    // Animation State for Reset
    private var isAnimatingReset: Bool = false
    private var animationStartTime: Date? = nil
    private let animationDuration: TimeInterval = 0.3 // seconds
    private var startRotation: Angle = .zero
    private var startVerticalRotation: Float = 0.0
    private var startRollRotation: Float = 0.0
    private var startZoom: Float = 1.0
    private var startTranslation: SIMD2<Float> = .zero

    init?(_ metalKitView: MTKView) {
        guard let device = metalKitView.device else { return nil }
        self.device = device
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.commandBufferManager = CommandBufferManager(commandQueue: queue)
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        
        switch model {
        case .gaussianSplat(let url):
            // Get cached model data
            let cachedModel = try await ModelCache.shared.getModel(.gaussianSplat(url))
            
            // Capture needed values from main actor context
            let deviceRef = device
            let colorPixelFormat = metalKitView.colorPixelFormat
            let depthStencilPixelFormat = metalKitView.depthStencilPixelFormat
            let sampleCount = metalKitView.sampleCount
            let useFastSH = fastSHSettings.enabled
            
            // Create and load splat entirely in nonisolated context
            let points = cachedModel.points // Explicit copy for isolation
            let splat = try await Task {
                if useFastSH {
                    // Use Fast SH renderer with cached data
                    let renderer = try FastSHSplatRenderer(device: deviceRef,
                                                         colorFormat: colorPixelFormat,
                                                         depthFormat: depthStencilPixelFormat,
                                                         sampleCount: sampleCount,
                                                         maxViewCount: 1,
                                                         maxSimultaneousRenders: Constants.maxSimultaneousRenders)
                    try await renderer.loadSplatsWithSH(points)
                    return renderer as SplatRenderer // Cast to base class
                } else {
                    // Use regular renderer with cached data
                    let renderer = try SplatRenderer(device: deviceRef,
                                                    colorFormat: colorPixelFormat,
                                                    depthFormat: depthStencilPixelFormat,
                                                    sampleCount: sampleCount,
                                                    maxViewCount: 1,
                                                    maxSimultaneousRenders: Constants.maxSimultaneousRenders)
                    try renderer.add(points)
                    return renderer
                }
            }.value
            
            modelRenderer = splat
            
            // Initialize Metal 4 bindless resources if available and enabled
            if useMetal4Bindless {
                if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
                    do {
                        try splat.initializeMetal4Bindless()
                        Self.log.info("Initialized Metal 4 bindless resources for Gaussian Splat model")
                    } catch {
                        Self.log.warning("Failed to initialize Metal 4 bindless resources: \(error.localizedDescription)")
                        // Continue with traditional rendering
                    }
                } else {
                    Self.log.info("Metal 4 bindless resources not available on this platform (requires iOS 26+/macOS 26+)")
                }
            }
            
            // Configure Fast SH if using FastSHSplatRenderer
            if let fastRenderer = splat as? FastSHSplatRenderer {
                // Apply settings from FastSHSettings object
                fastRenderer.fastSHConfig.enabled = fastSHSettings.enabled
                fastRenderer.fastSHConfig.maxPaletteSize = fastSHSettings.maxPaletteSize
                // Note: updateFrequency is deprecated; use shDirectionEpsilon for threshold-based updates

                // Analyze model and update settings
                let splatCount = fastRenderer.splatCount
                let uniqueShSets = fastRenderer.shCoefficients.count
                let shDegree = fastRenderer.shDegree

                await MainActor.run {
                    fastSHSettings.analyzeAndConfigure(
                        splatCount: splatCount,
                        uniqueShSets: uniqueShSets,
                        shDegree: shDegree
                    )
                }

                // Apply any updated settings back to renderer
                fastRenderer.fastSHConfig.enabled = fastSHSettings.enabled
                fastRenderer.fastSHConfig.maxPaletteSize = fastSHSettings.maxPaletteSize

                print("Fast SH configured for \(url.lastPathComponent): enabled=\(fastSHSettings.enabled), palette=\(uniqueShSets), degree=\(shDegree)")
            }
            
            if autoFitEnabled {
                await optimizeViewportForModel(splat)
            }
        case .sampleBox:
            do {
                modelRenderer = try SampleBoxRenderer(device: device,
                                                     colorFormat: metalKitView.colorPixelFormat,
                                                     depthFormat: metalKitView.depthStencilPixelFormat,
                                                     sampleCount: metalKitView.sampleCount,
                                                     maxViewCount: 1,
                                                     maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            } catch {
                Self.log.error("Failed to create SampleBoxRenderer: \(error)")
                return
            }
        case .none:
            break
        }
    }

    private var viewport: ModelRendererViewportDescriptor {
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(Constants.fovy.radians) / zoom,
                                                             aspectRatio: Float(drawableSize.width / drawableSize.height),
                                                             nearZ: 0.1,
                                                             farZ: 100.0)

        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        // Add vertical rotation (pitch) around X axis
        let verticalMatrix = matrix4x4_rotation(radians: verticalRotation, axis: SIMD3<Float>(1, 0, 0))
        // Add roll rotation (2-finger twist) around Z axis
        let rollMatrix = matrix4x4_rotation(radians: rollRotation, axis: SIMD3<Float>(0, 0, 1))
        // Add translation for panning
        let panMatrix = matrix4x4_translation(translation.x, translation.y, 0)
        let scaleMatrix = matrix4x4_scale(modelScale, modelScale, modelScale)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        // Coordinate system calibration based on file format
        // SOG coordinate system: x=right, y=up, z=back (−z is forward)
        //
        // Camera position difference from web/Plinth viewers:
        // - Web/Plinth: camera at -Z looking toward origin (after framing)
        // - MetalSplatter: camera at origin looking toward -Z (model at z=-8)
        //
        // Since the camera orientations differ by 180° around Y, we need:
        // - Web/Plinth use 180° Z rotation
        // - MetalSplatter needs 180° Z + 180° Y = 180° X rotation
        // This produces the same visual result from the user's perspective.
        let modelDescription = model?.description ?? ""
        let descriptionLowercased = modelDescription.lowercased()
        let isSOGS = descriptionLowercased.contains("meta.json") || descriptionLowercased.contains(".zip")
        let isSPZ = descriptionLowercased.contains(".spz") || descriptionLowercased.contains(".spx")
        let isSOGSv2 = descriptionLowercased.contains(".sog")
        
        // SPZ files are already correctly oriented like SOGS v1 files
        // SOGS v2 (.sog) files need 180° X rotation (equivalent to web's 180° Z with our camera setup)
        // PLY files need 180° rotation around Z axis to be right-side up
        let commonUpCalibration: simd_float4x4
        if isSOGSv2 {
            commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0)) // 180° around X for SOGS v2
        } else if isSOGS || isSPZ {
            commonUpCalibration = matrix_identity_float4x4 // No rotation for SOGS v1 and SPZ
        } else {
            commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1)) // 180° around Z for PLY
        }
        
        // Log coordinate calibration decision only when model changes
        if lastLoggedModel != modelDescription {
            lastLoggedModel = modelDescription
            print("MetalKitSceneRenderer: model='\(modelDescription)', isSOGS=\(isSOGS), isSOGSv2=\(isSOGSv2), isSPZ=\(isSPZ), rotation=\(isSOGSv2 ? "180°X" : (isSOGS || isSPZ ? "none" : "180°Z"))")
        }

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * panMatrix * rotationMatrix * verticalMatrix * rollMatrix * scaleMatrix * commonUpCalibration,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    private func updateRotation() {
        guard !userIsInteracting else { return }
        guard Constants.rotationPerSecond.degrees != 0 else { return }
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func draw(in view: MTKView) {
        guard let modelRenderer else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        guard let commandBuffer = commandBufferManager.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { @Sendable (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        // --- Handle Reset Animation ---
        if isAnimatingReset {
            guard let startTime = animationStartTime else {
                // Should not happen, but safety check
                isAnimatingReset = false
                return
            }
            let timeElapsed = Date().timeIntervalSince(startTime)
            let progress = min(timeElapsed / animationDuration, 1.0)
            let t = smoothStep(Float(progress)) // Eased progress

            // Interpolate view properties
            rotation = lerp(startRotation, .zero, Double(t))
            verticalRotation = lerp(startVerticalRotation, 0.0, t)
            rollRotation = lerp(startRollRotation, 0.0, t)
            zoom = lerp(startZoom, 1.0, t)
            translation = lerp(startTranslation, .zero, t)

            if progress >= 1.0 {
                // Animation finished
                isAnimatingReset = false
                animationStartTime = nil
                userIsInteracting = false // Allow auto-rotate again
                lastRotationUpdateTimestamp = nil // Reset timestamp for smooth auto-rotate start
            } else {
                // Request next frame
                #if os(macOS)
                metalKitView.setNeedsDisplay(metalKitView.bounds)
                #else
                metalKitView.setNeedsDisplay()
                #endif
            }
        } else {
             // Only update auto-rotation if not animating reset and not interacting
            updateRotation()
        }
        // --- End Animation Handling ---

        renderTraditional(modelRenderer: modelRenderer,
                        viewport: viewport,
                        view: view,
                        drawable: drawable,
                        commandBuffer: commandBuffer)

        commandBuffer.present(drawable)

        commandBuffer.commit()
    }
    
    // MARK: - Render Methods
    
    /// Traditional full-resolution rendering
    private func renderTraditional(modelRenderer: any ModelRenderer,
                                 viewport: ModelRendererViewportDescriptor,
                                 view: MTKView,
                                 drawable: CAMetalDrawable,
                                 commandBuffer: MTLCommandBuffer) {
        do {
            try modelRenderer.render(viewports: [viewport],
                                   colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                   colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                   depthTexture: view.depthStencilTexture,
                                   rasterizationRateMap: nil,
                                   renderTargetArrayLength: 0,
                                   to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    // MARK: - Metal 4 Configuration
    
    /// Enable or disable Metal 4 bindless rendering
    func setMetal4Bindless(_ enabled: Bool) {
        useMetal4Bindless = enabled

        // If we already have a renderer, try to initialize Metal 4
        if enabled, let splat = modelRenderer as? SplatRenderer {
            if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
                do {
                    try splat.initializeMetal4Bindless()
                    Self.log.info("Enabled Metal 4 bindless resources for current model")
                } catch {
                    Self.log.warning("Failed to enable Metal 4 bindless: \(error.localizedDescription)")
                }
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Check if Metal 4 bindless is available on this device
    var isMetal4BindlessAvailable: Bool {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return device.supportsFamily(.apple9) // Requires Apple 9 GPU family
        }
        return false
    }
    
    /// Enable or disable debug AABB visualization
    /// This tests the GPU SIMD-group parallel bounds computation
    func setDebugAABB(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            if enabled {
                splat.debugOptions.insert(.showAABB)
            } else {
                splat.debugOptions.remove(.showAABB)
            }
            Self.log.info("Debug AABB \(enabled ? "enabled" : "disabled")")
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    /// Enable or disable GPU frustum culling
    /// When enabled, splats outside the camera's view frustum are filtered out before rendering
    func setFrustumCulling(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            splat.frustumCullingEnabled = enabled
            Self.log.info("Frustum culling \(enabled ? "enabled" : "disabled")")
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    /// Enable or disable mesh shader rendering (Metal 3+)
    /// When enabled, geometry is generated entirely on GPU - 1 computation per splat instead of 4
    func setMeshShader(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            if enabled && !splat.isMeshShaderSupported {
                Self.log.info("Mesh shaders not supported on this device")
                return
            }
            splat.meshShaderEnabled = enabled
            if enabled {
                Self.log.info("Mesh shader rendering enabled - geometry generated on GPU")
            } else {
                Self.log.info("Mesh shader rendering disabled - using vertex shader path")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    /// Enable or disable Metal 4 TensorOps batch precompute
    /// Pre-computes covariance/transforms for all splats when camera changes
    /// Best for large scenes (50k+ splats) where camera movement is intermittent
    func setBatchPrecompute(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            splat.batchPrecomputeEnabled = enabled
            if enabled {
                Self.log.info("Metal 4 TensorOps batch precompute enabled")
            } else {
                Self.log.info("Metal 4 TensorOps batch precompute disabled")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Enable or disable dithered (stochastic) transparency
    /// When enabled, uses order-independent transparency via stochastic alpha testing
    /// Best paired with TAA for noise reduction - eliminates need for depth sorting
    func setDitheredTransparency(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            splat.useDitheredTransparency = enabled
            if enabled {
                Self.log.info("Dithered transparency enabled - order-independent, no sorting needed")
            } else {
                Self.log.info("Dithered transparency disabled - using sorted alpha blending")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    // MARK: - User Interaction API
    #if os(iOS)
    
    /// Notify renderer that user interaction has started (for adaptive sort quality)
    private func notifyInteractionBegan() {
        if let splat = modelRenderer as? SplatRenderer {
            splat.beginInteraction()
        }
    }
    
    /// Notify renderer that user interaction has ended (triggers quality sort)
    private func notifyInteractionEnded() {
        if let splat = modelRenderer as? SplatRenderer {
            splat.endInteraction()
        }
    }
    
    func setUserRotation(_ newRotation: Angle, vertical: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        rotation = newRotation
        verticalRotation = vertical
        lastRotationUpdateTimestamp = nil // Reset timestamp for smooth resumption of auto-rotation
    }

    func setUserZoom(_ newZoom: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        zoom = newZoom
        lastRotationUpdateTimestamp = nil
    }

    func setUserRollRotation(_ rollRotation: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        self.rollRotation = rollRotation
        lastRotationUpdateTimestamp = nil
    }

    func setUserTranslation(_ translation: SIMD2<Float>) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        self.translation = translation
        lastRotationUpdateTimestamp = nil
    }

    func endUserInteraction() {
        if userIsInteracting {
            notifyInteractionEnded()
        }
        userIsInteracting = false
        lastRotationUpdateTimestamp = Date()
    }
    
    /// Reset view to default state with smooth animation
    func resetView() {
        // If already animating, do nothing
        guard !isAnimatingReset else { return }
        
        // Store starting values for animation
        startRotation = rotation
        startVerticalRotation = verticalRotation
        startRollRotation = rollRotation
        startZoom = zoom
        startTranslation = translation
        
        // Start animation
        isAnimatingReset = true
        animationStartTime = Date()
        userIsInteracting = true // Prevent auto-rotate during animation
        
        // Trigger first animation frame
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    #endif

    /// Optimizes the viewport (scale, position) to fit the loaded model
    private func optimizeViewportForModel(_ renderer: any ModelRenderer) async {
        // Calculate model bounds
        guard let bounds = await calculateModelBounds(renderer) else {
            return
        }
        
        // Calculate model dimensions
        let size = bounds.max - bounds.min
        let maxDimension = max(size.x, size.y, size.z)
        
        // Adaptive scaling strategy based on model size
        let newScale: Float
        
        if maxDimension > 0 {
            // Only scale up small models - never scale down anything
            if maxDimension < 0.5 {
                // Very tiny models - scale up significantly
                let targetSize: Float = 3.0
                newScale = targetSize / maxDimension
                modelScale = max(1.0, min(newScale, 25.0))
            } else if maxDimension < 2.0 {
                // Small models - scale up moderately  
                let targetSize: Float = 4.0
                newScale = targetSize / maxDimension
                modelScale = max(1.0, min(newScale, 8.0))
            } else {
                // Everything else (>= 2 units) - leave completely unchanged
                modelScale = 1.0
            }
        } else {
            modelScale = 1.0
        }
        
        // Trigger redraw with new scale
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    private func calculateModelBounds(_ renderer: any ModelRenderer) async -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        if let splatRenderer = renderer as? SplatRenderer {
            return splatRenderer.calculateBounds()
        } else if let fastSHRenderer = renderer as? FastSHSplatRenderer {
            return fastSHRenderer.calculateBounds()
        }
        
        return nil
    }
}

#endif // os(iOS) || os(macOS)
