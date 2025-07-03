#if os(iOS)

import ARKit
import Foundation
import Metal
import MetalKit
import simd

public class ARBackgroundRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private let renderPipelineState: MTLRenderPipelineState
    
    // Captured image texture cache
    private var capturedImageTextureCache: CVMetalTextureCache!
    private var viewportSize = CGSize(width: 0, height: 0)
    private var updateGeometry = true
    
    // Vertex buffer for fullscreen quad
    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    
    public private(set) var capturedImageTextureY: CVMetalTexture?
    public private(set) var capturedImageTextureCbCr: CVMetalTexture?
    
    unowned let session: ARSession
    
    struct Vertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
    }
    
    public init(device: MTLDevice, session: ARSession) throws {
        self.device = device
        self.session = session
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw ARBackgroundRendererError.failedToCreateCommandQueue
        }
        self.commandQueue = commandQueue
        
        // Use the MetalSplatter bundle's library
        do {
            let metalSplatterBundle = Bundle.module
            let library = try device.makeDefaultLibrary(bundle: metalSplatterBundle)
            
            self.library = library
            print("ARBackgroundRenderer: Successfully created Metal library from MetalSplatter bundle")
        } catch {
            print("ARBackgroundRenderer: Error creating Metal library: \(error)")
            throw ARBackgroundRendererError.failedToCreateLibrary
        }
        
        // Create initial quad vertices
        let vertices = [
            Vertex(position: SIMD2<Float>(-1, -1), texCoord: SIMD2<Float>(0, 1)),
            Vertex(position: SIMD2<Float>( 1, -1), texCoord: SIMD2<Float>(1, 1)),
            Vertex(position: SIMD2<Float>(-1,  1), texCoord: SIMD2<Float>(0, 0)),
            Vertex(position: SIMD2<Float>( 1,  1), texCoord: SIMD2<Float>(1, 0))
        ]
        
        let indices: [UInt16] = [0, 1, 2, 1, 3, 2]
        
        guard let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Vertex>.stride, options: []) else {
            throw ARBackgroundRendererError.failedToCreateVertexBuffer
        }
        self.vertexBuffer = vertexBuffer
        
        guard let indexBuffer = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.stride, options: []) else {
            throw ARBackgroundRendererError.failedToCreateIndexBuffer
        }
        self.indexBuffer = indexBuffer
        
        // Create render pipeline
        print("ARBackgroundRenderer: Creating render pipeline state...")
        do {
            self.renderPipelineState = try Self.createRenderPipelineState(device: device, library: library)
            print("ARBackgroundRenderer: Successfully created render pipeline state")
        } catch {
            print("ARBackgroundRenderer: Failed to create render pipeline state: \(error)")
            throw error
        }
        
        // Setup texture cache
        setupTextureCache()
        
        // Setup orientation change notification
        NotificationCenter.default.addObserver(
            forName: UIDevice.orientationDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateGeometry = true
        }
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    public func render(to texture: MTLTexture, with commandBuffer: MTLCommandBuffer) {
        update()
        
        // Only render if we have valid camera textures
        guard let textureY = capturedImageTextureY,
              let textureCbCr = capturedImageTextureCbCr else {
            // No camera data available yet - just clear to black and return
            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = texture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.endEncoding()
            }
            return
        }
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        // Set captured image textures (guaranteed to exist due to guard above)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureY), index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(textureCbCr), index: 1)
        
        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                           indexCount: 6,
                                           indexType: .uint16,
                                           indexBuffer: indexBuffer,
                                           indexBufferOffset: 0)
        
        renderEncoder.endEncoding()
    }
    
    public func resize(_ size: CGSize) {
        viewportSize = size
        updateGeometry = true
    }
    
    public func clearCachedTextures() {
        // Clear cached textures to prevent purple flash on restart
        capturedImageTextureY = nil
        capturedImageTextureCbCr = nil
        updateGeometry = true
    }
    
    private func update() {
        guard let frame = session.currentFrame else { return }
        
        updateTextures(frame)
        
        if updateGeometry {
            updateGeometry(frame)
            updateGeometry = false
        }
    }
    
    private func updateGeometry(_ frame: ARFrame) {
        guard let interfaceOrientation = getOrientation() else { return }
        
        // Update texture coordinates based on display transform
        let displayToCameraTransform = frame.displayTransform(for: interfaceOrientation, viewportSize: viewportSize).inverted()
        
        let vertices = [
            Vertex(position: SIMD2<Float>(-1, -1), texCoord: SIMD2<Float>(0, 1)),
            Vertex(position: SIMD2<Float>( 1, -1), texCoord: SIMD2<Float>(1, 1)),
            Vertex(position: SIMD2<Float>(-1,  1), texCoord: SIMD2<Float>(0, 0)),
            Vertex(position: SIMD2<Float>( 1,  1), texCoord: SIMD2<Float>(1, 0))
        ]
        
        // Transform texture coordinates
        var transformedVertices = vertices
        for i in 0..<transformedVertices.count {
            let uv = transformedVertices[i].texCoord
            let textureCoord = CGPoint(x: CGFloat(uv.x), y: CGFloat(uv.y))
            let transformedCoord = textureCoord.applying(displayToCameraTransform)
            transformedVertices[i] = Vertex(
                position: transformedVertices[i].position,
                texCoord: SIMD2<Float>(Float(transformedCoord.x), Float(transformedCoord.y))
            )
        }
        
        // Update vertex buffer
        let contents = vertexBuffer.contents().bindMemory(to: Vertex.self, capacity: 4)
        for i in 0..<4 {
            contents[i] = transformedVertices[i]
        }
    }
    
    private func updateTextures(_ frame: ARFrame) {
        guard CVPixelBufferGetPlaneCount(frame.capturedImage) == 2 else { return }
        
        capturedImageTextureY = createTexture(
            fromPixelBuffer: frame.capturedImage,
            pixelFormat: .r8Unorm,
            planeIndex: 0
        )
        
        capturedImageTextureCbCr = createTexture(
            fromPixelBuffer: frame.capturedImage,
            pixelFormat: .rg8Unorm,
            planeIndex: 1
        )
    }
    
    private func setupTextureCache() {
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        capturedImageTextureCache = textureCache
    }
    
    private func createTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, capturedImageTextureCache, pixelBuffer, nil, pixelFormat, width, height, planeIndex, &texture
        )
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    private func getOrientation() -> UIInterfaceOrientation? {
        return UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.interfaceOrientation
    }
    
    private static func createRenderPipelineState(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        print("ARBackgroundRenderer: Looking for shader functions...")
        print("ARBackgroundRenderer: Available function names: \(library.functionNames.sorted())")
        
        guard let vertexFunction = library.makeFunction(name: "ar_background_vertex") else {
            print("ARBackgroundRenderer: ar_background_vertex function not found!")
            throw ARBackgroundRendererError.failedToCreateShaderFunctions
        }
        print("ARBackgroundRenderer: Found ar_background_vertex")
        
        guard let fragmentFunction = library.makeFunction(name: "ar_background_fragment") else {
            print("ARBackgroundRenderer: ar_background_fragment function not found!")
            throw ARBackgroundRendererError.failedToCreateShaderFunctions
        }
        print("ARBackgroundRenderer: Found ar_background_fragment")
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        descriptor.colorAttachments[0].isBlendingEnabled = false
        
        // Vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        descriptor.vertexDescriptor = vertexDescriptor
        
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

public enum ARBackgroundRendererError: Error {
    case failedToCreateCommandQueue
    case failedToCreateLibrary
    case failedToCreateVertexBuffer
    case failedToCreateIndexBuffer
    case failedToCreateShaderFunctions
}

#endif // os(iOS)