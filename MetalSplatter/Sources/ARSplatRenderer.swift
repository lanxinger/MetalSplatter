#if os(iOS)

import ARKit
import Foundation
import Metal
import MetalKit
import SplatIO
import simd

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
    
    // Offscreen textures for compositing
    private var backgroundTexture: MTLTexture?
    private var contentTexture: MTLTexture?
    private var updateTextures = true
    
    // Composition pipeline
    private let compositionPipelineState: MTLRenderPipelineState
    private let compositionVertexBuffer: MTLBuffer
    private let compositionIndexBuffer: MTLBuffer
    
    // Current viewport size
    private var viewportSize = CGSize.zero
    
    // Splat transform properties for AR interaction
    public var splatPosition: SIMD3<Float> = SIMD3<Float>(0, 0, -1.5) // 1.5 meters in front of camera
    public var splatScale: Float = 0.1 // Start small for AR
    public var splatRotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    private var hasBeenPlaced = false // Track if user has placed the splat
    
    struct CompositionVertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
    }
    
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
        
        // Create composition pipeline
        self.compositionPipelineState = try Self.createCompositionPipelineState(device: device, library: library, colorFormat: colorFormat)
        
        // Create composition geometry
        let vertices = [
            CompositionVertex(position: SIMD2<Float>(-1, -1), texCoord: SIMD2<Float>(0, 1)),
            CompositionVertex(position: SIMD2<Float>( 1, -1), texCoord: SIMD2<Float>(1, 1)),
            CompositionVertex(position: SIMD2<Float>(-1,  1), texCoord: SIMD2<Float>(0, 0)),
            CompositionVertex(position: SIMD2<Float>( 1,  1), texCoord: SIMD2<Float>(1, 0))
        ]
        
        let indices: [UInt16] = [0, 1, 2, 1, 3, 2]
        
        guard let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<CompositionVertex>.stride, options: []) else {
            throw ARSplatRendererError.failedToCreateVertexBuffer
        }
        self.compositionVertexBuffer = vertexBuffer
        
        guard let indexBuffer = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.stride, options: []) else {
            throw ARSplatRendererError.failedToCreateIndexBuffer
        }
        self.compositionIndexBuffer = indexBuffer
        
        super.init()
        
        // Configure AR session
        setupARSession()
    }
    
    public func read(from url: URL) async throws {
        try await splatRenderer.read(from: url)
        print("ARSplatRenderer: Loaded \(splatRenderer.splatCount) splats from \(url.lastPathComponent)")
        
        // Reset placement state when loading new splats
        hasBeenPlaced = false
        
        // Auto-place splat in front of camera when first loaded (will be done in render loop)
    }
    
    private func autoPlaceSplatInFrontOfCamera() {
        print("ARSplatRenderer: Auto-placing splat in front of camera...")
        
        // Wait a bit for AR session to be ready
        guard let frame = session.currentFrame else {
            print("ARSplatRenderer: No AR frame available yet, will try again next frame")
            return // Don't mark as placed yet, try again next frame
        }
        
        let cameraTransform = frame.camera.transform
        let forward = -cameraTransform.columns.2.xyz // Camera looks down negative Z
        splatPosition = cameraTransform.columns.3.xyz + forward * 1.5
        splatScale = 0.1
        hasBeenPlaced = true // Mark as placed so it stops updating
        print("ARSplatRenderer: Auto-placed splat at \(splatPosition) with scale \(splatScale)")
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
        print("ARSplatRenderer: Starting AR session...")
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        session.run(configuration)
        print("ARSplatRenderer: AR session started with configuration")
    }
    
    public func stopARSession() {
        session.pause()
    }
    
    // MARK: - AR Interaction Methods
    
    public func placeSplatAtScreenPoint(_ screenPoint: CGPoint, viewportSize: CGSize) {
        print("ARSplatRenderer: placeSplatAtScreenPoint called with \(screenPoint)")
        
        guard let frame = session.currentFrame else { 
            print("ARSplatRenderer: No AR frame available for tap-to-place")
            return 
        }
        
        print("ARSplatRenderer: AR frame available, proceeding with placement")
        print("ARSplatRenderer: Tap-to-place at screen point: \(screenPoint) in viewport: \(viewportSize)")
        
        // Create raycast query for existing plane geometry
        guard let orientation = getOrientation() else {
            print("ARSplatRenderer: Cannot get orientation for raycast, falling back to camera placement")
            // Fall back to placing in front of camera
            let cameraTransform = frame.camera.transform
            let forward = -cameraTransform.columns.2.xyz
            splatPosition = cameraTransform.columns.3.xyz + forward * 1.0
            hasBeenPlaced = true
            print("ARSplatRenderer: Placed splat in front of camera (orientation fallback) at: \(splatPosition)")
            return
        }
        
        print("ARSplatRenderer: Got orientation: \(orientation), proceeding with raycast")
        
        // Debug screen point conversion
        print("ARSplatRenderer: Screen point: \(screenPoint), Viewport: \(viewportSize)")
        
        // Convert screen point to normalized coordinates first
        let normalizedPoint = CGPoint(
            x: screenPoint.x / viewportSize.width,
            y: screenPoint.y / viewportSize.height
        )
        print("ARSplatRenderer: Normalized point: \(normalizedPoint)")
        
        // Calculate world direction
        let worldDirection = screenPointToWorldDirection(normalizedPoint, frame: frame, viewportSize: viewportSize)
        print("ARSplatRenderer: World direction: \(worldDirection)")
        
        // Try horizontal planes first (tables, floors) - most common for AR placement
        let horizontalQuery = ARRaycastQuery(
            origin: frame.camera.transform.columns.3.xyz,
            direction: worldDirection,
            allowing: .existingPlaneGeometry,
            alignment: .horizontal
        )
        
        let results = session.raycast(horizontalQuery)
        print("ARSplatRenderer: Raycast for existing planes returned \(results.count) results")
        
        if let result = results.first {
            // Place splat at the detected plane location
            splatPosition = result.worldTransform.columns.3.xyz
            hasBeenPlaced = true
            print("ARSplatRenderer: Placed splat on detected plane at: \(splatPosition)")
        } else {
            // Try with estimated planes if no existing geometry found
            let estimatedQuery = ARRaycastQuery(
                origin: frame.camera.transform.columns.3.xyz,
                direction: worldDirection,
                allowing: .estimatedPlane,
                alignment: .horizontal
            )
            
            let estimatedResults = session.raycast(estimatedQuery)
            print("ARSplatRenderer: Raycast for estimated planes returned \(estimatedResults.count) results")
            
            if let result = estimatedResults.first {
                splatPosition = result.worldTransform.columns.3.xyz
                hasBeenPlaced = true
                print("ARSplatRenderer: Placed splat on estimated plane at: \(splatPosition)")
            } else {
                // Fall back to placing in front of camera
                let cameraTransform = frame.camera.transform
                let forward = -cameraTransform.columns.2.xyz
                splatPosition = cameraTransform.columns.3.xyz + forward * 1.0
                hasBeenPlaced = true
                print("ARSplatRenderer: No plane detected, placed splat in front of camera at: \(splatPosition)")
            }
        }
    }
    
    public func scaleSplat(factor: Float) {
        splatScale = max(0.1, min(10.0, splatScale * factor)) // Clamp between 0.1x and 10x
        print("ARSplatRenderer: Scaled splat to: \(splatScale)")
    }
    
    public func moveSplat(by delta: SIMD3<Float>) {
        splatPosition += delta
        print("ARSplatRenderer: Moved splat to: \(splatPosition)")
    }
    
    public func rotateSplat(by angle: Float, axis: SIMD3<Float>) {
        let rotation = simd_quatf(angle: angle, axis: normalize(axis))
        splatRotation = simd_mul(rotation, splatRotation)
        print("ARSplatRenderer: Rotated splat")
    }
    
    private func screenPointToWorldDirection(_ normalizedPoint: CGPoint, frame: ARFrame, viewportSize: CGSize) -> SIMD3<Float> {
        guard let orientation = getOrientation() else {
            return SIMD3<Float>(0, 0, -1)
        }
        
        // Convert to camera coordinates
        let intrinsics = frame.camera.intrinsics
        let imageResolution = frame.camera.imageResolution
        
        // Account for display transform
        let displayTransform = frame.displayTransform(for: orientation, viewportSize: viewportSize)
        let transformedPoint = normalizedPoint.applying(displayTransform.inverted())
        
        // Convert to camera pixel coordinates
        let pixelX = Float(transformedPoint.x) * Float(imageResolution.width)
        let pixelY = Float(transformedPoint.y) * Float(imageResolution.height)
        
        // Unproject to camera space
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        
        let x = (pixelX - cx) / fx
        let y = (pixelY - cy) / fy
        let z: Float = -1.0 // Camera looks down negative Z
        
        // Transform to world space
        let cameraDirection = SIMD3<Float>(x, y, z)
        let cameraTransform = frame.camera.transform
        let worldDirection = cameraTransform.upperLeft3x3 * cameraDirection
        
        return normalize(worldDirection)
    }
    
    private func getOrientation() -> UIInterfaceOrientation? {
        return UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.interfaceOrientation
    }
    
    public func render(to drawable: CAMetalDrawable, viewportSize: CGSize) throws {
        // Update viewport size if needed
        if self.viewportSize != viewportSize {
            self.viewportSize = viewportSize
            arBackgroundRenderer.resize(viewportSize)
            updateTextures = true
        }
        
        // Update AR camera
        arCamera.update(viewportSize: viewportSize)
        
        // Create or update offscreen textures
        if updateTextures {
            createOffscreenTextures(size: viewportSize)
            updateTextures = false
        }
        
        guard let backgroundTexture = backgroundTexture,
              let contentTexture = contentTexture else {
            throw ARSplatRendererError.failedToCreateOffscreenTextures
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw ARSplatRendererError.failedToCreateCommandBuffer
        }
        
        // Render AR background to offscreen texture
        arBackgroundRenderer.render(to: backgroundTexture, with: commandBuffer)
        
        // Render splats to offscreen texture (if we have splats loaded)
        if splatRenderer.splatCount > 0 {
            // Auto-place splat once when first loaded
            if !hasBeenPlaced {
                autoPlaceSplatInFrontOfCamera()
            }
            
            // Removed excessive per-frame logging
            
            // Render actual splats
            try renderSplatsToTexture(contentTexture, commandBuffer: commandBuffer)
        } else {
            print("ARSplatRenderer: No splats loaded, clearing content texture")
            // Clear content texture if no splats
            clearTexture(contentTexture, commandBuffer: commandBuffer)
        }
        
        // Composite both textures to final drawable
        compositeToDrawable(drawable, backgroundTexture: backgroundTexture, contentTexture: contentTexture, commandBuffer: commandBuffer)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func setupARSession() {
        session.delegate = self
    }
    
    private func clearTexture(_ texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0) // Transparent
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.endEncoding()
        }
    }
    
    private func renderTestPattern(_ texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 0.5) // Magenta with transparency
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.endEncoding()
        }
        print("ARSplatRenderer: Rendered magenta test pattern to content texture")
    }
    
    private func renderSimpleTestShape(_ texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        // Just clear the content texture to a visible color to test composition
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 0, alpha: 0.3) // Lower alpha for blending
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            renderEncoder.endEncoding()
        }
        print("ARSplatRenderer: Rendered simple red test shape")
    }
    
    private func createSolidColorTexture(red: Float, green: Float, blue: Float, alpha: Float) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.width = 1
        descriptor.height = 1
        descriptor.pixelFormat = .bgra8Unorm_srgb
        descriptor.usage = [.shaderRead]
        
        let texture = device.makeTexture(descriptor: descriptor)!
        let color = [UInt8(red * 255), UInt8(green * 255), UInt8(blue * 255), UInt8(alpha * 255)]
        texture.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: 1, height: 1, depth: 1)), 
                       mipmapLevel: 0, 
                       withBytes: color, 
                       bytesPerRow: 4)
        return texture
    }
    
    private func createOffscreenTextures(size: CGSize) {
        let width = Int(size.width)
        let height = Int(size.height)
        
        guard width > 0, height > 0 else { return }
        
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .bgra8Unorm_srgb
        descriptor.width = width
        descriptor.height = height
        descriptor.textureType = .type2D
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        
        backgroundTexture = device.makeTexture(descriptor: descriptor)
        backgroundTexture?.label = "AR Background Texture"
        
        // Content texture with alpha for blending
        descriptor.pixelFormat = .bgra8Unorm_srgb
        contentTexture = device.makeTexture(descriptor: descriptor)
        contentTexture?.label = "AR Content Texture"
    }
    
    private func renderSplatsToTexture(_ texture: MTLTexture, commandBuffer: MTLCommandBuffer) throws {
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
            colorStoreAction: .store,
            depthTexture: nil, // AR doesn't use depth buffer for composition
            rasterizationRateMap: nil,
            renderTargetArrayLength: 0,
            to: commandBuffer
        )
    }
    
    private func compositeToDrawable(_ drawable: CAMetalDrawable, backgroundTexture: MTLTexture, contentTexture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(compositionPipelineState)
        renderEncoder.setVertexBuffer(compositionVertexBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(backgroundTexture, index: 0)
        renderEncoder.setFragmentTexture(contentTexture, index: 1)
        
        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                           indexCount: 6,
                                           indexType: .uint16,
                                           indexBuffer: compositionIndexBuffer,
                                           indexBufferOffset: 0)
        
        renderEncoder.endEncoding()
    }
    
    private static func createCompositionPipelineState(device: MTLDevice, library: MTLLibrary, colorFormat: MTLPixelFormat) throws -> MTLRenderPipelineState {
        guard let vertexFunction = library.makeFunction(name: "ar_composition_vertex"),
              let fragmentFunction = library.makeFunction(name: "ar_composition_fragment") else {
            throw ARSplatRendererError.failedToCreateShaderFunctions
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = colorFormat
        
        // Enable alpha blending for splats over background
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .zero
        
        // Vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<CompositionVertex>.stride
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        descriptor.vertexDescriptor = vertexDescriptor
        
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }
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
            // If not a drawable, fall back to direct rendering
            try renderSplatsToTexture(colorTexture, commandBuffer: commandBuffer)
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