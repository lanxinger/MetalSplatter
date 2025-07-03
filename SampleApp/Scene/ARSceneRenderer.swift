#if os(iOS)

import ARKit
import Foundation
@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalSplatter
import os
import simd
import SwiftUI

@MainActor
class ARSceneRenderer: NSObject, MTKViewDelegate {
    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
        category: "ARSceneRenderer"
    )
    
    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var model: ModelIdentifier?
    private var arSplatRenderer: ARSplatRenderer?
    
    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)
    
    var drawableSize: CGSize = .zero
    
    // AR Session state
    var isARSessionActive = false
    
    init?(_ metalKitView: MTKView) {
        guard let device = metalKitView.device else { return nil }
        self.device = device
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.metalKitView = metalKitView
        
        // Configure MTKView for AR
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.invalid // AR doesn't use depth
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalKitView.framebufferOnly = false // Allow texture reading for composition
    }
    
    func load(_ model: ModelIdentifier?) async throws {
        Self.log.info("AR: Loading model \(String(describing: model))")
        guard model != self.model else { 
            Self.log.info("AR: Model already loaded, skipping")
            return 
        }
        self.model = model
        
        // Stop existing AR session
        stopARSession()
        arSplatRenderer = nil
        
        switch model {
        case .gaussianSplat(let url):
            Self.log.info("AR: Loading gaussian splat from \(url.lastPathComponent)")
            // Capture needed values from main actor context
            let deviceRef = device
            let colorPixelFormat = metalKitView.colorPixelFormat
            let depthStencilPixelFormat = metalKitView.depthStencilPixelFormat
            let sampleCount = metalKitView.sampleCount
            
            // Create AR splat renderer entirely in nonisolated context
            let renderer = try await Task.detached {
                let arRenderer = try ARSplatRenderer(
                    device: deviceRef,
                    colorFormat: colorPixelFormat,
                    depthFormat: depthStencilPixelFormat,
                    sampleCount: sampleCount,
                    maxViewCount: 1,
                    maxSimultaneousRenders: Constants.maxSimultaneousRenders
                )
                try await arRenderer.read(from: url)
                return arRenderer
            }.value
            
            arSplatRenderer = renderer
            Self.log.info("AR: Successfully created AR splat renderer")
            
        case .sampleBox:
            Self.log.info("AR: Creating sample box for AR")
            // Create a simple AR renderer without a model for demonstration
            let deviceRef = device
            let colorPixelFormat = metalKitView.colorPixelFormat
            let depthStencilPixelFormat = metalKitView.depthStencilPixelFormat
            let sampleCount = metalKitView.sampleCount
            
            let renderer = try await Task.detached {
                return try ARSplatRenderer(
                    device: deviceRef,
                    colorFormat: colorPixelFormat,
                    depthFormat: depthStencilPixelFormat,
                    sampleCount: sampleCount,
                    maxViewCount: 1,
                    maxSimultaneousRenders: Constants.maxSimultaneousRenders
                )
            }.value
            
            arSplatRenderer = renderer
            Self.log.info("AR: Successfully created AR renderer for sample box")
            
        case .none:
            Self.log.info("AR: No model specified")
            break
        }
    }
    
    func startARSession() {
        guard let arSplatRenderer = arSplatRenderer else {
            Self.log.error("Cannot start AR session without a loaded splat model")
            return
        }
        
        arSplatRenderer.startARSession()
        isARSessionActive = true
        Self.log.info("AR session started")
    }
    
    func stopARSession() {
        arSplatRenderer?.stopARSession()
        isARSessionActive = false
        Self.log.info("AR session stopped")
    }
    
    func draw(in view: MTKView) {
        guard let arSplatRenderer = arSplatRenderer else { 
            Self.log.error("No AR splat renderer available")
            return 
        }
        
        guard isARSessionActive else { 
            Self.log.warning("AR session is not active")
            return 
        }
        
        guard let drawable = view.currentDrawable else { 
            Self.log.error("No drawable available")
            return 
        }
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        let semaphore = inFlightSemaphore
        defer {
            semaphore.signal()
        }
        
        do {
            try arSplatRenderer.render(to: drawable, viewportSize: drawableSize)
            // Removed excessive per-frame logging
        } catch {
            Self.log.error("Unable to render AR scene: \(error.localizedDescription)")
            // Fallback: Clear to a visible color so we know rendering is working
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = drawable.texture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0.5, blue: 0, alpha: 1) // Green screen
            
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.endEncoding()
            }
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }
    
    // MARK: - AR Session Management
    
    func pauseARSession() {
        arSplatRenderer?.stopARSession()
        isARSessionActive = false
    }
    
    func resumeARSession() {
        guard arSplatRenderer != nil else { return }
        startARSession()
    }
    
    func isARTrackingNormal() -> Bool {
        return arSplatRenderer?.isARTrackingNormal ?? false
    }
    
    // MARK: - Touch Handling for AR Interaction
    
    // MARK: - Gesture Handlers
    
    func handleTap(at location: CGPoint) {
        guard let arSplatRenderer = arSplatRenderer else { 
            Self.log.error("No AR splat renderer for tap gesture")
            return 
        }
        
        // Get the view bounds in points (not pixels)
        let viewBounds = metalKitView.bounds.size
        
        Self.log.info("AR tap-to-place at location: \(location.x), \(location.y), view bounds: \(viewBounds.width)x\(viewBounds.height)")
        arSplatRenderer.placeSplatAtScreenPoint(location, viewportSize: viewBounds)
    }
    
    func handlePan(translation: CGPoint, velocity: CGPoint) {
        guard let arSplatRenderer = arSplatRenderer else { return }
        
        // Convert screen pan to world movement
        // Scale based on distance from camera for more intuitive movement
        let scale: Float = 0.0005 // Reduced sensitivity for more precise control
        let deltaX = Float(translation.x) * scale
        let deltaZ = Float(translation.y) * scale // Y translation affects Z movement (forward/back)
        
        arSplatRenderer.moveSplat(by: SIMD3<Float>(deltaX, 0, deltaZ))
    }
    
    func handlePinch(scale: CGFloat, velocity: CGFloat) {
        guard let arSplatRenderer = arSplatRenderer else { return }
        
        let scaleFactor = Float(scale)
        arSplatRenderer.scaleSplat(factor: scaleFactor)
    }
    
    func handleRotation(rotation: CGFloat, velocity: CGFloat) {
        guard let arSplatRenderer = arSplatRenderer else { return }
        
        let angle = Float(rotation)
        let yAxis = SIMD3<Float>(0, 1, 0) // Rotate around Y axis (vertical)
        arSplatRenderer.rotateSplat(by: angle, axis: yAxis)
        Self.log.info("AR: Rotating splat by \(angle) radians around Y axis")
    }
}

#endif // os(iOS)