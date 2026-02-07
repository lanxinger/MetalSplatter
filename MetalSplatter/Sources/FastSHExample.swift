import Foundation
import Metal
import MetalKit
import SplatIO

// Remove duplicate definition - will use the one from SplatRenderer+FastSH

/// Example demonstrating how to use the Fast SH implementation for SOGS files
public class FastSHExample {
    
    public static func demonstrateUsage() async throws {
        // Get Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SplatRendererError.metalDeviceUnavailable
        }
        
        // Create Fast SH renderer
        let renderer = try FastSHSplatRenderer(
            device: device,
            colorFormat: .bgra8Unorm,
            depthFormat: .depth32Float,
            sampleCount: 1,
            maxViewCount: 2,
            maxSimultaneousRenders: 3
        )
        
        // Configure fast SH options
        renderer.fastSHConfig.enabled = true
        renderer.shDirectionEpsilon = 0.0 // Update every frame
        renderer.fastSHConfig.maxPaletteSize = 65536 // Support up to 64K unique SH sets
        
        // Load SOGS file with SH data
        let sogsURL = URL(fileURLWithPath: "path/to/your/file.sogs")
        let reader = try SplatSOGSSceneReader(sogsURL)
        
        var splats: [SplatScenePoint] = []
        // Read all splats using SplatMemoryBuffer approach
        var buffer = SplatMemoryBuffer()
        try await buffer.read(from: reader)
        splats = buffer.points
        
        // Load splats with SH support
        try await renderer.loadSplatsWithSH(splats)
        
        print("Loaded \(splats.count) splats with fast SH support")
        
        // Example render loop
        // In a real application, this would be called from your render loop
        guard let commandQueue = device.makeCommandQueue() else {
            throw SplatRendererError.failedToCreateBuffer(length: 0)
        }
        
        // Create viewport using SplatRenderer's ViewportDescriptor
        let viewport = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: 1920, height: 1080, znear: 0, zfar: 1),
            projectionMatrix: matrix_perspective_right_hand(fovyRadians: .pi / 3, aspectRatio: 16.0/9.0, nearZ: 0.1, farZ: 100.0),
            viewMatrix: matrix_identity_float4x4,
            screenSize: SIMD2<Int>(1920, 1080)
        )
        
        // Render with fast SH (using SplatRenderer's internal render method)
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            let colorTexture = try createDummyTexture(device: device)
            
            try renderer.render(
                viewports: [viewport],
                colorTexture: colorTexture,
                colorStoreAction: .store,
                depthTexture: nil,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
            
            commandBuffer.commit()
        }
    }
    
    /// Example of switching between fast and accurate SH modes
    public static func demonstrateModes(renderer: FastSHSplatRenderer) {
        // Fast mode - single direction evaluation
        renderer.fastSHConfig.enabled = true
        renderer.shDirectionEpsilon = 0.08 // Less frequent updates for performance

        // Disable fast SH - fall back to CPU evaluation
        renderer.fastSHConfig.enabled = false
    }
    
    private static func createDummyTexture(device: MTLDevice) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 1920,
            height: 1080,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw SplatRendererError.failedToCreateBuffer(length: 1920 * 1080 * 4)
        }
        return texture
    }
}

// Matrix helper (would normally come from your math library)
private func matrix_perspective_right_hand(fovyRadians: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovyRadians * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    
    return matrix_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, nearZ * zs, 0)
    ))
}
