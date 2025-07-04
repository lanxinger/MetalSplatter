import Foundation
import Metal
import simd

/// Manages fast spherical harmonics evaluation for SOGS format
public class SphericalHarmonicsEvaluator {
    private let device: MTLDevice
    private let computePipeline: MTLComputePipelineState
    private let directionalPipeline: MTLComputePipelineState?
    
    /// Structure matching the Metal shader parameters
    private struct SHEvaluateParams {
        let viewDirection: SIMD3<Float>
        let paletteSize: UInt32
        let degree: UInt32
        let padding: UInt32 = 0  // Ensure 16-byte alignment
    }
    
    public enum Mode {
        /// Evaluate SH once per frame using camera forward direction
        case fast
        /// Evaluate SH with per-pixel accuracy (more expensive)
        case accurate
    }
    
    public init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        
        // Create compute pipeline for palette evaluation
        guard let function = library.makeFunction(name: "evaluateSphericalHarmonicsPalette") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "evaluateSphericalHarmonicsPalette")
        }
        self.computePipeline = try device.makeComputePipelineState(function: function)
        
        // Create optional directional pipeline
        if let directionalFunction = library.makeFunction(name: "evaluateSphericalHarmonicsDirectional") {
            self.directionalPipeline = try? device.makeComputePipelineState(function: directionalFunction)
        } else {
            self.directionalPipeline = nil
        }
    }
    
    /// Pre-evaluate spherical harmonics for a palette of coefficients
    /// - Parameters:
    ///   - shPalette: Buffer containing SH coefficients for each palette entry
    ///   - paletteSize: Number of unique SH coefficient sets
    ///   - degree: SH degree (0-3)
    ///   - viewDirection: Camera view direction (normalized)
    ///   - commandBuffer: Command buffer to encode into
    /// - Returns: Buffer containing evaluated RGB colors for each palette entry
    public func evaluatePalette(
        shPalette: MTLBuffer,
        paletteSize: Int,
        degree: Int,
        viewDirection: SIMD3<Float>,
        commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer? {
        // Calculate buffer size for output
        let outputSize = paletteSize * MemoryLayout<SIMD4<Float>>.stride
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModePrivate) else {
            return nil
        }
        outputBuffer.label = "SH Evaluated Colors"
        
        // Create parameters
        var params = SHEvaluateParams(
            viewDirection: normalize(viewDirection),
            paletteSize: UInt32(paletteSize),
            degree: UInt32(degree)
        )
        
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<SHEvaluateParams>.stride,
            options: .storageModeShared
        ) else {
            return nil
        }
        
        // Encode compute command
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        computeEncoder.label = "SH Palette Evaluation"
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(shPalette, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        
        // Calculate thread groups
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (paletteSize + 255) / 256,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        
        computeEncoder.endEncoding()
        
        return outputBuffer
    }
    
    /// Pre-evaluate spherical harmonics into a texture for more accurate edge rendering
    /// - Parameters:
    ///   - shPalette: Buffer containing SH coefficients for each palette entry
    ///   - textureSize: Size of the output texture (must fit paletteSize entries)
    ///   - degree: SH degree (0-3)
    ///   - viewDirection: Camera view direction (normalized)
    ///   - commandBuffer: Command buffer to encode into
    /// - Returns: Texture containing evaluated RGB colors
    public func evaluateToTexture(
        shPalette: MTLBuffer,
        textureSize: MTLSize,
        degree: Int,
        viewDirection: SIMD3<Float>,
        commandBuffer: MTLCommandBuffer
    ) -> MTLTexture? {
        guard let pipeline = directionalPipeline else { return nil }
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2D
        textureDescriptor.pixelFormat = .rgba16Float
        textureDescriptor.width = textureSize.width
        textureDescriptor.height = textureSize.height
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.storageMode = .private
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        texture.label = "SH Evaluated Texture"
        
        let paletteSize = textureSize.width * textureSize.height
        
        // Create parameters
        var params = SHEvaluateParams(
            viewDirection: normalize(viewDirection),
            paletteSize: UInt32(paletteSize),
            degree: UInt32(degree)
        )
        
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<SHEvaluateParams>.stride,
            options: .storageModeShared
        ) else {
            return nil
        }
        
        // Encode compute command
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        computeEncoder.label = "SH Texture Evaluation"
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(shPalette, offset: 0, index: 0)
        computeEncoder.setTexture(texture, index: 0)
        computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        
        // Calculate thread groups
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (textureSize.width + 15) / 16,
            height: (textureSize.height + 15) / 16,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        
        computeEncoder.endEncoding()
        
        return texture
    }
}

// Helper function to determine SH degree from coefficient count
extension SphericalHarmonicsEvaluator {
    public static func degreeFromCoefficientCount(_ count: Int) -> Int {
        switch count {
        case 1: return 0   // DC only
        case 4: return 1   // DC + band 1
        case 9: return 2   // DC + bands 1,2
        case 15: return 3  // SPZ format: DC + 14 additional coeffs
        case 16: return 3  // Standard format: DC + bands 1,2,3
        default: return 0
        }
    }
    
    public static func coefficientCountForDegree(_ degree: Int) -> Int {
        return (degree + 1) * (degree + 1)
    }
}