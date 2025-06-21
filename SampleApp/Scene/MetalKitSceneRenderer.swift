#if os(iOS) || os(macOS)

@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalSplatter
import os
import SampleBoxRenderer
import simd
import SwiftUI

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
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

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
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
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
            // Capture needed values from main actor context
            let deviceRef = device
            let colorPixelFormat = metalKitView.colorPixelFormat
            let depthStencilPixelFormat = metalKitView.depthStencilPixelFormat
            let sampleCount = metalKitView.sampleCount
            
            // Create and read splat entirely in nonisolated context
            let splat = try await Task.detached {
                let renderer = try SplatRenderer(device: deviceRef,
                                                colorFormat: colorPixelFormat,
                                                depthFormat: depthStencilPixelFormat,
                                                sampleCount: sampleCount,
                                                maxViewCount: 1,
                                                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
                try await renderer.read(from: url)
                return renderer
            }.value
            
            // Enable optimizations for testing
            splat.useOptimizedMemoryLayout = true
            splat.useGPURadixSort = false // Disable for now - needs work for large datasets
            
            Self.log.info("SplatRenderer loaded - Splat count: \(splat.splatCount)")
            Self.log.info("Memory optimization: \(splat.useOptimizedMemoryLayout ? "ENABLED" : "DISABLED")")
            Self.log.info("GPU radix sort: \(splat.useGPURadixSort ? "ENABLED" : "DISABLED")")
            
            modelRenderer = splat
            if autoFitEnabled {
                await optimizeViewportForModel(splat)
            }
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: metalKitView.colorPixelFormat,
                                                   depthFormat: metalKitView.depthStencilPixelFormat,
                                                   sampleCount: metalKitView.sampleCount,
                                                   maxViewCount: 1,
                                                   maxSimultaneousRenders: Constants.maxSimultaneousRenders)
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
        // Turn common 3D GS PLY files rightside-up. This isn't generally meaningful, it just
        // happens to be a useful default for the most common datasets at the moment.
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

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

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
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

        commandBuffer.present(drawable)

        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    // MARK: - User Interaction API
    #if os(iOS)
    func setUserRotation(_ newRotation: Angle, vertical: Float) {
        userIsInteracting = true
        rotation = newRotation
        verticalRotation = vertical
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    func setUserZoom(_ newZoom: Float) {
        userIsInteracting = true
        zoom = newZoom
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    func setUserTranslation(_ newTranslation: SIMD2<Float>) {
        userIsInteracting = true
        translation = newTranslation
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    func setUserRollRotation(_ newRoll: Float) {
        userIsInteracting = true
        rollRotation = newRoll
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    func resetView() {
        // Start animation instead of setting directly
        guard !isAnimatingReset else { return } // Don't restart if already animating

        isAnimatingReset = true
        animationStartTime = Date()
        userIsInteracting = true // Prevent auto-rotate during animation

        // Store starting state
        startRotation = rotation
        startVerticalRotation = verticalRotation
        startRollRotation = rollRotation
        startZoom = zoom
        startTranslation = translation

        // No need to set final values here, draw() will handle it.
        // No need for metalKitView.setNeedsDisplay() here, draw() will trigger redraws.
    }
    
    func toggleAutoFit() {
        autoFitEnabled.toggle()
        if autoFitEnabled, let modelRenderer = modelRenderer {
            Task {
                await optimizeViewportForModel(modelRenderer)
            }
        } else {
            modelScale = 1.0
            #if os(macOS)
            metalKitView.setNeedsDisplay(metalKitView.bounds)
            #else
            metalKitView.setNeedsDisplay()
            #endif
        }
    }

    /// Call this when user gestures (drag, pinch) end.
    func endUserInteraction() {
        // If user interacts during reset animation, cancel the animation
        if isAnimatingReset {
            isAnimatingReset = false
            animationStartTime = nil
        }
        userIsInteracting = false
        // Reset timestamp to avoid jump in auto-rotation after interaction
        lastRotationUpdateTimestamp = nil
    }
    #endif
    
    // MARK: - Viewport Optimization
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
        }
        
        return nil
    }
}

#endif // os(iOS) || os(macOS)
