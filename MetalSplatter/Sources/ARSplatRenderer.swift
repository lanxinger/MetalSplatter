#if os(iOS)

import ARKit
import Foundation
import Metal
import MetalKit
import SplatIO
import simd
import UIKit

public class ARSplatRenderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // AR components
    public let session: ARSession
    public let arCamera: ARPerspectiveCamera
    private let arBackgroundRenderer: ARBackgroundRenderer
    
    // Core splat renderer
    private var splatRenderer: SplatRenderer
    
    // Note: Removed offscreen textures and composition pipeline for single-pass rendering
    
    // Current viewport size
    private var viewportSize = CGSize.zero
    
    // Splat transform properties for AR interaction
    public var splatPosition: SIMD3<Float> = SIMD3<Float>(0, 0, -1.5) // 1.5 meters in front of camera
    public var splatScale: Float = 1.0 // Default scale for properly sized models
    public var splatRotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    private var hasBeenPlaced = false // Track if user has placed the splat
    private var arSessionStartTime: CFTimeInterval = 0 // Track when AR session started
    private var isWaitingForARTracking = true // Track if we're waiting for AR to stabilize
    
    // Track the loaded file URL for coordinate calibration
    private var loadedFileURL: URL?
    
    // AR session state
    public var isARTrackingNormal: Bool {
        guard let frame = session.currentFrame else { return false }
        return frame.camera.trackingState == .normal
    }
    
    public func isWaitingForSurfaceDetection() -> Bool {
        return isWaitingForARTracking && !hasBeenPlaced && splatRenderer.splatCount > 0
    }
    
    // Note: Removed CompositionVertex struct for single-pass rendering
    
    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int = 1,
                maxSimultaneousRenders: Int = 3) throws {
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw ARSplatRendererError.failedToCreateCommandQueue
        }
        self.commandQueue = commandQueue
        
        // Use the MetalSplatter bundle's library instead of the default library
        do {
            let metalSplatterBundle = Bundle.module
            let library = try device.makeDefaultLibrary(bundle: metalSplatterBundle)
            
            print("ARSplatRenderer: Successfully created Metal library from MetalSplatter bundle")
            print("ARSplatRenderer: Available functions: \(library.functionNames.sorted())")
            
            // Check if AR shader functions are available
            if library.makeFunction(name: "ar_background_vertex") != nil {
                print("ARSplatRenderer: ✅ Found ar_background_vertex")
            } else {
                print("ARSplatRenderer: ❌ ar_background_vertex function not found")
            }
            
            if library.makeFunction(name: "ar_background_fragment") != nil {
                print("ARSplatRenderer: ✅ Found ar_background_fragment")
            } else {
                print("ARSplatRenderer: ❌ ar_background_fragment function not found")
            }
            
            self.library = library
        } catch {
            print("ARSplatRenderer: Error creating Metal library: \(error)")
            throw ARSplatRendererError.failedToCreateLibrary
        }
        
        // Initialize AR session
        self.session = ARSession()
        self.arCamera = ARPerspectiveCamera(session: session)
        
        print("ARSplatRenderer: Creating ARBackgroundRenderer...")
        do {
            self.arBackgroundRenderer = try ARBackgroundRenderer(device: device, session: session)
            print("ARSplatRenderer: Successfully created ARBackgroundRenderer")
        } catch {
            print("ARSplatRenderer: Failed to create ARBackgroundRenderer: \(error)")
            throw error
        }
        
        // Initialize core splat renderer
        self.splatRenderer = try SplatRenderer(
            device: device,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: sampleCount,
            maxViewCount: maxViewCount,
            maxSimultaneousRenders: maxSimultaneousRenders
        )
        
        // Note: Removed composition pipeline creation for single-pass rendering
        
        super.init()
        
        // Log device AR capabilities
        logARCapabilities()
        
        // Configure AR session
        setupARSession()
    }
    
    public func read(from url: URL) async throws {
        try await splatRenderer.read(from: url)
        print("ARSplatRenderer: Loaded \(splatRenderer.splatCount) splats from \(url.lastPathComponent)")
        
        // Store the file URL for coordinate calibration
        loadedFileURL = url
        
        // Reset placement state when loading new splats
        hasBeenPlaced = false
        isWaitingForARTracking = true
        
        // Reset AR session start time to give new model time to stabilize
        arSessionStartTime = CACurrentMediaTime()
        
        // Auto-place splat in front of camera when first loaded (will be done in render loop)
    }
    
    private func autoPlaceSplatInFrontOfCamera() {
        // Only try auto-placement if we're still waiting
        guard isWaitingForARTracking else { return }
        
        // Wait for AR session to be properly initialized with tracking state
        guard let frame = session.currentFrame else {
            return // Don't mark as placed yet, try again next frame
        }
        
        // Give AR session more time to detect surfaces before auto-placing
        let timeSinceStart = CACurrentMediaTime() - arSessionStartTime
        guard timeSinceStart > 3.0 else {
            return // Wait for AR session to stabilize and detect surfaces
        }
        
        // Check if camera tracking is working properly
        switch frame.camera.trackingState {
        case .normal:
            // AR is tracking properly, but check if we have detected surfaces
            let detectedAnchors = session.currentFrame?.anchors.filter { $0 is ARPlaneAnchor } ?? []
            
            if detectedAnchors.isEmpty && timeSinceStart < 10.0 {
                // No planes detected yet, wait longer (up to 10 seconds)
                if Int(timeSinceStart) % 2 == 0 && timeSinceStart - floor(timeSinceStart) < 0.1 {
                    print("ARSplatRenderer: Waiting for surface detection... (\(String(format: "%.1f", timeSinceStart))s)")
                }
                return
            }
            
            // Try to auto-place on a detected surface
            if let placementPosition = findAutoPlacementPosition(frame: frame) {
                splatPosition = placementPosition
                splatScale = 1.0
                hasBeenPlaced = true
                isWaitingForARTracking = false
                print("ARSplatRenderer: ✅ Auto-placed splat on detected surface at \(splatPosition) with scale \(splatScale) after \(String(format: "%.1f", timeSinceStart))s")
            } else if timeSinceStart > 10.0 {
                // After 10 seconds, fall back to fixed distance even without surface detection
                let cameraTransform = frame.camera.transform
                let forward = -cameraTransform.columns.2.xyz // Camera looks down negative Z
                splatPosition = cameraTransform.columns.3.xyz + forward * 1.5
                splatScale = 1.0
                hasBeenPlaced = true
                isWaitingForARTracking = false
                print("ARSplatRenderer: ⚠️ Timeout: No surface detected after 10s, placed at fixed distance \(splatPosition)")
            } else {
                // Still waiting for better surface detection
                return
            }
        case .limited(let reason):
            // Only log once per second to avoid spam
            if Int(timeSinceStart) % 2 == 0 && timeSinceStart - floor(timeSinceStart) < 0.1 {
                print("ARSplatRenderer: AR tracking limited (\(reason)), waiting...")
            }
            return
        case .notAvailable:
            // Only log once per second
            if Int(timeSinceStart) % 2 == 0 && timeSinceStart - floor(timeSinceStart) < 0.1 {
                print("ARSplatRenderer: AR tracking not available, waiting...")
            }
            return
        }
    }
    
    private func updateSplatPositionRelativeToCamera() {
        guard let frame = session.currentFrame else {
            return
        }
        
        let cameraTransform = frame.camera.transform
        let forward = -cameraTransform.columns.2.xyz // Camera looks down negative Z
        splatPosition = cameraTransform.columns.3.xyz + forward * 1.5 // 1.5 meters in front
        splatScale = 0.1 // Much smaller initial scale
    }
    
    public func startARSession() {
        print("ARSplatRenderer: Starting AR session (hasBeenPlaced=\(hasBeenPlaced), isWaitingForARTracking=\(isWaitingForARTracking))...")
        let configuration = ARWorldTrackingConfiguration()
        
        // Enable all plane detection for maximum surface coverage
        configuration.planeDetection = [.horizontal, .vertical]
        
        // Enable scene reconstruction if available (LiDAR devices)
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
            print("ARSplatRenderer: ✅ LiDAR scene reconstruction enabled")
        } else {
            print("ARSplatRenderer: ⚠️ LiDAR not available, using visual-inertial tracking")
        }
        
        // Enable frame semantics for better understanding
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            configuration.frameSemantics.insert(.personSegmentationWithDepth)
            print("ARSplatRenderer: ✅ Person segmentation with depth enabled")
        }
        
        // Enable automatic image stabilization for better tracking
        configuration.isAutoFocusEnabled = true
        
        // Use the highest quality video format available
        let videoFormats = ARWorldTrackingConfiguration.supportedVideoFormats
        if let highestResFormat = videoFormats.max(by: { 
            $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height 
        }) {
            configuration.videoFormat = highestResFormat
            print("ARSplatRenderer: ✅ High resolution camera enabled (\(Int(highestResFormat.imageResolution.width))x\(Int(highestResFormat.imageResolution.height)))")
        }
        
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        arSessionStartTime = CACurrentMediaTime() // Record when we started the session
        isWaitingForARTracking = true // Reset waiting state
        hasBeenPlaced = false // Allow auto-placement to happen again
        
        print("ARSplatRenderer: AR session started with optimized configuration")
        print("ARSplatRenderer: - Plane detection: horizontal + vertical")
        print("ARSplatRenderer: - Scene reconstruction: \(configuration.sceneReconstruction)")
        print("ARSplatRenderer: - Frame semantics: \(configuration.frameSemantics)")
        print("ARSplatRenderer: - Auto focus: \(configuration.isAutoFocusEnabled)")
    }
    
    public func stopARSession() {
        print("ARSplatRenderer: Stopping AR session and resetting state")
        session.pause()
        
        // Reset placement state so auto-placement can happen again
        hasBeenPlaced = false
        isWaitingForARTracking = true
        
        // Reset session start time for next session
        arSessionStartTime = 0
        
        // Clear any cached AR textures in the background renderer
        arBackgroundRenderer.clearCachedTextures()
        
        print("ARSplatRenderer: AR session stopped and state reset")
    }
    
    // MARK: - AR Interaction Methods
    
    public func placeSplatAtScreenPoint(_ screenPoint: CGPoint, viewportSize: CGSize) {
        print("ARSplatRenderer: placeSplatAtScreenPoint called with \(screenPoint) in viewport \(viewportSize)")
        
        guard let frame = session.currentFrame else { 
            print("ARSplatRenderer: No AR frame available for tap-to-place")
            return 
        }
        
        // Use raycast for more accurate placement (iOS 13+)
        // First try horizontal planes (most common for object placement)
        var raycastQuery = ARRaycastQuery(
            origin: frame.camera.transform.columns.3.xyz,
            direction: screenPointToWorldDirection(screenPoint, frame: frame, viewportSize: viewportSize),
            allowing: .existingPlaneGeometry,
            alignment: .horizontal
        )
        
        var results = session.raycast(raycastQuery)
        
        if let result = results.first {
            // Place the splat directly on the surface with minimal offset to avoid Z-fighting
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.1.xyz // Y column is up vector
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            splatPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            hasBeenPlaced = true
            print("ARSplatRenderer: ✅ Placed splat on existing horizontal plane at: \(splatPosition) (offset: \(offsetDistance))")
            return
        }
        
        // Try vertical planes (walls)
        raycastQuery = ARRaycastQuery(
            origin: frame.camera.transform.columns.3.xyz,
            direction: screenPointToWorldDirection(screenPoint, frame: frame, viewportSize: viewportSize),
            allowing: .existingPlaneGeometry,
            alignment: .vertical
        )
        
        results = session.raycast(raycastQuery)
        
        if let result = results.first {
            // Place the splat directly on the vertical surface with minimal offset
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.2.xyz // Z column is forward/normal for vertical
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            splatPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            hasBeenPlaced = true
            print("ARSplatRenderer: ✅ Placed splat on existing vertical plane at: \(splatPosition) (offset: \(offsetDistance))")
            return
        }
        
        // Try estimated planes if no existing geometry
        raycastQuery = ARRaycastQuery(
            origin: frame.camera.transform.columns.3.xyz,
            direction: screenPointToWorldDirection(screenPoint, frame: frame, viewportSize: viewportSize),
            allowing: .estimatedPlane,
            alignment: .horizontal
        )
        
        results = session.raycast(raycastQuery)
        
        if let result = results.first {
            // Place the splat directly on the estimated surface with minimal offset
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.1.xyz // Y column is up vector
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            splatPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            hasBeenPlaced = true
            print("ARSplatRenderer: ✅ Placed splat on estimated plane at: \(splatPosition) (offset: \(offsetDistance))")
            return
        }
        
        // Fallback: try legacy hit test for compatibility with more types
        let hitTestResults = frame.hitTest(screenPoint, types: [.existingPlaneUsingExtent, .estimatedHorizontalPlane, .featurePoint])
        
        if let result = hitTestResults.first {
            // Place the splat directly on the detected surface with minimal offset
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.1.xyz // Y column is up vector for most surfaces
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            splatPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            hasBeenPlaced = true
            print("ARSplatRenderer: ✅ Placed splat using legacy hit test at: \(splatPosition) (offset: \(offsetDistance))")
            return
        }
        
        // Final fallback: place along ray at fixed distance
        let direction = screenPointToWorldDirection(screenPoint, frame: frame, viewportSize: viewportSize)
        splatPosition = frame.camera.transform.columns.3.xyz + normalize(direction) * 1.5
        hasBeenPlaced = true
        print("ARSplatRenderer: ⚠️ No surface found, placed splat along ray at: \(splatPosition)")
    }
    
    private func screenPointToWorldDirection(_ screenPoint: CGPoint, frame: ARFrame, viewportSize: CGSize) -> SIMD3<Float> {
        // Use the same transformation as ARBackgroundRenderer but in reverse
        let interfaceOrientation = getInterfaceOrientation()
        
        // Get the inverse display transform to go from view coordinates to camera image coordinates
        let displayToCameraTransform = frame.displayTransform(for: interfaceOrientation, viewportSize: viewportSize).inverted()
        
        // Convert screen point to normalized coordinates [0,1] in view space
        let normalizedViewPoint = CGPoint(
            x: screenPoint.x / viewportSize.width,
            y: screenPoint.y / viewportSize.height
        )
        
        // Apply the inverse transform to get normalized camera coordinates
        let normalizedCameraPoint = normalizedViewPoint.applying(displayToCameraTransform)
        
        // Convert to camera image pixel coordinates
        let imageResolution = frame.camera.imageResolution
        let imagePoint = CGPoint(
            x: normalizedCameraPoint.x * imageResolution.width,
            y: normalizedCameraPoint.y * imageResolution.height
        )
        
        // Unproject using camera intrinsics
        let intrinsics = frame.camera.intrinsics
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        
        // Convert to camera coordinate system (normalized device coordinates)
        let x = (Float(imagePoint.x) - cx) / fx
        let y = (Float(imagePoint.y) - cy) / fy
        let z: Float = -1.0  // Camera looks down negative Z
        
        let cameraDirection = SIMD3<Float>(x, y, z)
        
        // Transform from camera space to world space
        let cameraTransform = frame.camera.transform
        let worldDirection = cameraTransform.upperLeft3x3 * cameraDirection
        
        let normalizedDirection = normalize(worldDirection)
        
        print("ARSplatRenderer: screenPoint=\(screenPoint) → normalizedView=\(normalizedViewPoint) → normalizedCamera=\(normalizedCameraPoint) → imagePoint=\(imagePoint) → worldDirection=\(normalizedDirection)")
        
        return normalizedDirection
    }
    
    
    private func getInterfaceOrientation() -> UIInterfaceOrientation {
        // Get the current interface orientation
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            return windowScene.interfaceOrientation
        }
        return .portrait // fallback
    }
    
    private func findAutoPlacementPosition(frame: ARFrame) -> SIMD3<Float>? {
        // Try to find a surface near the center of the screen for auto-placement
        let cameraTransform = frame.camera.transform
        let forward = -cameraTransform.columns.2.xyz // Camera looks down negative Z
        
        // Log detected planes for debugging
        let planeAnchors = frame.anchors.compactMap { $0 as? ARPlaneAnchor }
        print("ARSplatRenderer: Auto-placement attempting with \(planeAnchors.count) detected planes")
        
        // Try raycasting from camera center forward
        let raycastQuery = ARRaycastQuery(
            origin: cameraTransform.columns.3.xyz,
            direction: forward,
            allowing: .existingPlaneGeometry,
            alignment: .horizontal
        )
        
        var results = session.raycast(raycastQuery)
        print("ARSplatRenderer: Horizontal plane raycast returned \(results.count) results")
        
        if let result = results.first {
            // Found horizontal surface - place directly on it
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.1.xyz // Y column is up vector
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            let finalPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            print("ARSplatRenderer: Auto-placement found horizontal surface at \(surfacePosition), placing at \(finalPosition)")
            return finalPosition
        }
        
        // Try estimated planes if no existing geometry
        let estimatedQuery = ARRaycastQuery(
            origin: cameraTransform.columns.3.xyz,
            direction: forward,
            allowing: .estimatedPlane,
            alignment: .horizontal
        )
        
        results = session.raycast(estimatedQuery)
        print("ARSplatRenderer: Estimated plane raycast returned \(results.count) results")
        
        if let result = results.first {
            // Found estimated surface - place directly on it
            let surfacePosition = result.worldTransform.columns.3.xyz
            let surfaceNormal = result.worldTransform.columns.1.xyz // Y column is up vector
            let offsetDistance: Float = 0.001 // Minimal 1mm offset to avoid Z-fighting
            let finalPosition = surfacePosition + normalize(surfaceNormal) * offsetDistance
            print("ARSplatRenderer: Auto-placement found estimated surface at \(surfacePosition), placing at \(finalPosition)")
            return finalPosition
        }
        
        // No surface found with raycast methods
        print("ARSplatRenderer: ⚠️ Auto-placement could not find any surfaces via raycast")
        return nil // No surface found
    }
    
    public func scaleSplat(factor: Float) {
        splatScale = max(0.01, min(10.0, splatScale * factor)) // Clamp between 0.01x and 10x
        print("ARSplatRenderer: Scaled splat to: \(splatScale)")
    }
    
    public func moveSplat(by delta: SIMD3<Float>) {
        // Transform delta relative to camera orientation for more intuitive movement
        guard let frame = session.currentFrame else {
            splatPosition += delta
            return
        }
        
        let cameraTransform = frame.camera.transform
        let right = cameraTransform.columns.0.xyz // Camera right vector
        let up = cameraTransform.columns.1.xyz // Camera up vector
        let forward = -cameraTransform.columns.2.xyz // Camera forward vector
        
        // Apply movement relative to camera orientation
        let worldDelta = right * delta.x + up * delta.y + forward * delta.z
        splatPosition += worldDelta
        print("ARSplatRenderer: Moved splat by \(delta) in camera space, world delta: \(worldDelta), new position: \(splatPosition)")
    }
    
    public func rotateSplat(by angle: Float, axis: SIMD3<Float>) {
        let rotation = simd_quatf(angle: angle, axis: normalize(axis))
        splatRotation = simd_mul(rotation, splatRotation)
        print("ARSplatRenderer: Rotated splat by \(angle) radians around \(axis)")
    }
    
    
    public func render(to drawable: CAMetalDrawable, viewportSize: CGSize) throws {
        // Update viewport size if needed
        if self.viewportSize != viewportSize {
            self.viewportSize = viewportSize
            arBackgroundRenderer.resize(viewportSize)
        }
        
        // Update AR camera
        arCamera.update(viewportSize: viewportSize)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ARSplatRendererError.failedToCreateCommandBuffer
        }
        
        // Single-pass rendering: render AR background directly to drawable
        arBackgroundRenderer.render(to: drawable.texture, with: commandBuffer)
        
        // Render splats on top with proper blending (if we have splats loaded)
        if splatRenderer.splatCount > 0 {
            // Auto-place splat once when first loaded
            if !hasBeenPlaced {
                autoPlaceSplatInFrontOfCamera()
            }
            
            // Only render splats if they've been placed (either auto or manually)
            if hasBeenPlaced {
                // Render splats directly to drawable with alpha blending
                try renderSplatsToDrawable(drawable, commandBuffer: commandBuffer)
            }
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func setupARSession() {
        session.delegate = self
    }
    
    private func logARCapabilities() {
        print("ARSplatRenderer: 📱 Device AR Capabilities:")
        print("  - World Tracking: \(ARWorldTrackingConfiguration.isSupported)")
        print("  - Scene Reconstruction: \(ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh))")
        print("  - Person Segmentation: \(ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth))")
        
        let videoFormats = ARWorldTrackingConfiguration.supportedVideoFormats
        print("  - Video Formats Available: \(videoFormats.count)")
        
        // Find and log the highest resolution format
        if let highestResFormat = videoFormats.max(by: { 
            $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height 
        }) {
            print("  - Max Resolution: \(Int(highestResFormat.imageResolution.width))x\(Int(highestResFormat.imageResolution.height)) @ \(highestResFormat.framesPerSecond)fps")
        }
        
        // Log first few formats
        for format in videoFormats.prefix(3) {
            print("    • \(Int(format.imageResolution.width))x\(Int(format.imageResolution.height)) @ \(format.framesPerSecond)fps")
        }
        
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            print("  🚀 LiDAR detected - enabling advanced features!")
        } else {
            print("  📷 Using visual-inertial tracking")
        }
    }
    
    // Note: Removed clearTexture method for single-pass rendering
    
    // Note: Removed renderTestPattern method for single-pass rendering
    
    // Note: Removed renderSimpleTestShape method for single-pass rendering
    
    // Note: Removed createSolidColorTexture method for single-pass rendering
    
    // Note: Removed createOffscreenTextures method for single-pass rendering
    
    private func renderSplatsToDrawable(_ drawable: CAMetalDrawable, commandBuffer: MTLCommandBuffer) throws {
        // Create transform matrix for splat positioning in AR space
        let translationMatrix = matrix4x4_translation(splatPosition.x, splatPosition.y, splatPosition.z)
        let rotationMatrix = matrix4x4_rotation(splatRotation)
        let scaleMatrix = matrix4x4_scale(splatScale, splatScale, splatScale)
        
        // Combine transformations: Translation * Rotation * Scale
        // Note: AR coordinate system doesn't need the same calibration as MetalKitSceneRenderer
        let modelMatrix = translationMatrix * rotationMatrix * scaleMatrix
        
        // Apply AR camera view matrix to the transformed splat
        let viewMatrix = arCamera.viewMatrix
        
        let viewport = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(drawable.texture.width), height: Double(drawable.texture.height), znear: 0, zfar: 1),
            projectionMatrix: arCamera.projectionMatrix,
            viewMatrix: viewMatrix * modelMatrix,
            screenSize: SIMD2(x: drawable.texture.width, y: drawable.texture.height)
        )
        
        try splatRenderer.render(
            viewports: [viewport],
            colorTexture: drawable.texture,
            colorLoadAction: .load, // Preserve AR background
            colorStoreAction: .store,
            depthTexture: nil, // AR doesn't use depth buffer for composition
            rasterizationRateMap: nil,
            renderTargetArrayLength: 0,
            to: commandBuffer
        )
    }
    
    private func renderSplatsDirectlyToTexture(_ texture: MTLTexture, commandBuffer: MTLCommandBuffer) throws {
        // Create transform matrix for splat positioning in AR space
        let translationMatrix = matrix4x4_translation(splatPosition.x, splatPosition.y, splatPosition.z)
        let rotationMatrix = matrix4x4_rotation(splatRotation)
        let scaleMatrix = matrix4x4_scale(splatScale, splatScale, splatScale)
        
        // Combine transformations: Translation * Rotation * Scale
        let modelMatrix = translationMatrix * rotationMatrix * scaleMatrix
        
        // Apply AR camera view matrix to the transformed splat
        let viewMatrix = arCamera.viewMatrix
        
        let viewport = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(texture.width), height: Double(texture.height), znear: 0, zfar: 1),
            projectionMatrix: arCamera.projectionMatrix,
            viewMatrix: viewMatrix * modelMatrix,
            screenSize: SIMD2(x: texture.width, y: texture.height)
        )
        
        try splatRenderer.render(
            viewports: [viewport],
            colorTexture: texture,
            colorLoadAction: .load, // Preserve existing content
            colorStoreAction: .store,
            depthTexture: nil, // AR doesn't use depth buffer for composition
            rasterizationRateMap: nil,
            renderTargetArrayLength: 0,
            to: commandBuffer
        )
    }
    
    // Note: Removed compositeToDrawable method for single-pass rendering
    
    // Note: Removed createCompositionPipelineState method for single-pass rendering
}

// MARK: - ARSessionDelegate

extension ARSplatRenderer: ARSessionDelegate {
    public func session(_ session: ARSession, didFailWithError error: Error) {
        print("AR Session failed with error: \(error)")
    }
    
    public func sessionWasInterrupted(_ session: ARSession) {
        print("AR Session was interrupted")
    }
    
    public func sessionInterruptionEnded(_ session: ARSession) {
        print("AR Session interruption ended")
    }
    
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // This will be called frequently when AR is working
        // Uncomment this line only for debugging, it will spam the console:
        // print("AR Session: Received frame update")
    }
    
    // MARK: - Public Interface for SampleApp Integration
    
    public func renderAsModelRenderer(viewports: [Any], // Will be ModelRendererViewportDescriptor from SampleApp
                                     colorTexture: MTLTexture,
                                     colorStoreAction: MTLStoreAction,
                                     depthTexture: MTLTexture?,
                                     rasterizationRateMap: MTLRasterizationRateMap?,
                                     renderTargetArrayLength: Int,
                                     to commandBuffer: MTLCommandBuffer) throws {
        // For AR, we ignore the provided viewports and use AR camera data
        let viewportSize = CGSize(width: colorTexture.width, height: colorTexture.height)
        
        guard let drawable = colorTexture as? CAMetalDrawable else {
            // If not a drawable, fall back to direct rendering to texture
            try renderSplatsDirectlyToTexture(colorTexture, commandBuffer: commandBuffer)
            return
        }
        
        try render(to: drawable, viewportSize: viewportSize)
    }
}

public enum ARSplatRendererError: Error {
    case failedToCreateCommandQueue
    case failedToCreateLibrary
    case failedToCreateVertexBuffer
    case failedToCreateIndexBuffer
    case failedToCreateShaderFunctions
    case failedToCreateOffscreenTextures
    case failedToCreateCommandBuffer
}

// MARK: - Matrix Extensions

extension simd_float4x4 {
    var upperLeft3x3: simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z),
            SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z),
            SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z)
        )
    }
}

extension simd_float4 {
    var xyz: SIMD3<Float> {
        return SIMD3<Float>(x, y, z)
    }
}

#endif // os(iOS)