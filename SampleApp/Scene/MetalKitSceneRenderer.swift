#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
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
    // Add translation for panning
    var translation: SIMD2<Float> = .zero

    var drawableSize: CGSize = .zero

    // Animation State for Reset
    private var isAnimatingReset: Bool = false
    private var animationStartTime: Date? = nil
    private let animationDuration: TimeInterval = 0.3 // seconds
    private var startRotation: Angle = .zero
    private var startVerticalRotation: Float = 0.0
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
            let splat = try await SplatRenderer(device: device,
                                                colorFormat: metalKitView.colorPixelFormat,
                                                depthFormat: metalKitView.depthStencilPixelFormat,
                                                sampleCount: metalKitView.sampleCount,
                                                maxViewCount: 1,
                                                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            try await splat.read(from: url)
            modelRenderer = splat
        case .sampleBox:
            modelRenderer = try! await SampleBoxRenderer(device: device,
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
        // Add translation for panning
        let panMatrix = matrix4x4_translation(translation.x, translation.y, 0)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        // Turn common 3D GS PLY files rightside-up. This isn't generally meaningful, it just
        // happens to be a useful default for the most common datasets at the moment.
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * panMatrix * rotationMatrix * verticalMatrix * commonUpCalibration,
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
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
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
    func resetView() {
        // Start animation instead of setting directly
        guard !isAnimatingReset else { return } // Don't restart if already animating

        isAnimatingReset = true
        animationStartTime = Date()
        userIsInteracting = true // Prevent auto-rotate during animation

        // Store starting state
        startRotation = rotation
        startVerticalRotation = verticalRotation
        startZoom = zoom
        startTranslation = translation

        // No need to set final values here, draw() will handle it.
        // No need for metalKitView.setNeedsDisplay() here, draw() will trigger redraws.
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
}

#endif // os(iOS) || os(macOS)
