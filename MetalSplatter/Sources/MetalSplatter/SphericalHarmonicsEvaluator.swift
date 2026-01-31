import Foundation
import Metal
import os
import simd

fileprivate let log = Logger(subsystem: Bundle.module.bundleIdentifier ?? "com.metalsplatter.unknown",
                             category: "SphericalHarmonicsEvaluator")

/// Manages fast spherical harmonics evaluation for SOGS format
public class SphericalHarmonicsEvaluator {
    private let device: MTLDevice

    /// Legacy pipeline (runtime degree parameter) - kept for compatibility
    private let computePipeline: MTLComputePipelineState
    private let directionalPipeline: MTLComputePipelineState?

    /// Specialized pipelines using function constants (compile-time degree)
    /// Index corresponds to SH degree (0-3)
    private var specializedPipelines: [MTLComputePipelineState] = []
    private var specializedDirectionalPipelines: [MTLComputePipelineState] = []

    /// Function constant index for SH_DEGREE (matches [[function_constant(0)]] in shader)
    private static let shDegreeConstantIndex: Int = 0

    /// SH coefficient order (Graphdeco/gsplat layout) shared with spherical_harmonics_evaluate.metal and FastSHRenderPath.metal.
    /// Index 0 is the DC term, followed by Y bands: y, z, x, xy, yz, 2zz-xx-yy, xz, xx-yy, then band 3.
    public static let coefficientOrder: [String] = [
        "L0,0 (dc)",
        "L1,-1 (y)", "L1,0 (z)", "L1,1 (x)",
        "L2,-2 (xy)", "L2,-1 (yz)", "L2,0 (2zz-xx-yy)", "L2,1 (xz)", "L2,2 (xx-yy)",
        "L3,-3 (y*(3xx-yy))", "L3,-2 (xy*z)", "L3,-1 (y*(4zz-xx-yy))",
        "L3,0 (z*(2zz-3xx-3yy))", "L3,1 (x*(4zz-xx-yy))", "L3,2 (z*(xx-yy))", "L3,3 (x*(xx-3yy))"
    ]

    /// Validates that the SH coefficient count matches the expected count for the given degree.
    /// - Returns: `true` if the layout is valid, `false` otherwise (logs a warning on mismatch).
    @inline(__always)
    @discardableResult
    public static func validateLayout(degree: Int, coefficientCount: Int) -> Bool {
        let expected = coefficientCountForDegree(degree)
        let matchesSPZ = degree == 3 && coefficientCount == 15 // SPZ packs 15 coeffs; treated as degree 3
        let isValid = coefficientCount == expected || matchesSPZ
        if !isValid {
            log.warning("SH coefficient count \(coefficientCount) does not match degree \(degree); expected \(expected). SH evaluation will be skipped.")
        }
        return isValid
    }

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

        // Create legacy compute pipeline for palette evaluation (runtime degree)
        guard let function = library.makeFunction(name: "evaluateSphericalHarmonicsPalette") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "evaluateSphericalHarmonicsPalette")
        }
        self.computePipeline = try device.makeComputePipelineState(function: function)

        // Create optional legacy directional pipeline
        if let directionalFunction = library.makeFunction(name: "evaluateSphericalHarmonicsDirectional") {
            self.directionalPipeline = try? device.makeComputePipelineState(function: directionalFunction)
        } else {
            self.directionalPipeline = nil
        }

        // Create specialized pipelines for each SH degree (0-3)
        // These use function constants for compile-time branch elimination
        try createSpecializedPipelines(library: library)
    }

    /// Creates 4 specialized pipeline states (one per SH degree) using function constants
    private func createSpecializedPipelines(library: MTLLibrary) throws {
        for degree in 0...3 {
            // Create function constant values for this degree
            let constants = MTLFunctionConstantValues()
            var degreeValue = UInt32(degree)
            constants.setConstantValue(&degreeValue, type: .uint, index: Self.shDegreeConstantIndex)

            // Create specialized palette evaluation pipeline
            if let function = try? library.makeFunction(name: "evaluateSphericalHarmonicsPaletteSpecialized",
                                                        constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: function)
                specializedPipelines.append(pipeline)
            }

            // Create specialized directional pipeline
            if let function = try? library.makeFunction(name: "evaluateSphericalHarmonicsDirectionalSpecialized",
                                                        constantValues: constants) {
                let pipeline = try device.makeComputePipelineState(function: function)
                specializedDirectionalPipelines.append(pipeline)
            }
        }

        if !specializedPipelines.isEmpty {
            print("SphericalHarmonicsEvaluator: Created \(specializedPipelines.count) specialized pipelines (function constants enabled)")
        }
    }

    /// Returns the specialized pipeline for the given degree, or nil if not available
    private func specializedPipeline(for degree: Int) -> MTLComputePipelineState? {
        guard degree >= 0 && degree < specializedPipelines.count else { return nil }
        return specializedPipelines[degree]
    }

    /// Returns the specialized directional pipeline for the given degree, or nil if not available
    private func specializedDirectionalPipeline(for degree: Int) -> MTLComputePipelineState? {
        guard degree >= 0 && degree < specializedDirectionalPipelines.count else { return nil }
        return specializedDirectionalPipelines[degree]
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
        if paletteSize > 0 {
            let coeffCount = (shPalette.length / MemoryLayout<SIMD3<Float>>.stride) / max(paletteSize, 1)
            guard Self.validateLayout(degree: degree, coefficientCount: coeffCount) else {
                return nil
            }
        }

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

        // Use specialized pipeline if available (function constants for compile-time optimization)
        // Falls back to legacy pipeline with runtime branching if specialized version unavailable
        let pipeline: MTLComputePipelineState
        if let specializedPipeline = specializedPipeline(for: degree) {
            pipeline = specializedPipeline
            computeEncoder.label = "SH Palette Evaluation (Specialized Degree \(degree))"
        } else {
            pipeline = computePipeline
            computeEncoder.label = "SH Palette Evaluation (Legacy)"
        }

        computeEncoder.setComputePipelineState(pipeline)
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
        // Use specialized pipeline if available, otherwise fall back to legacy
        let pipeline: MTLComputePipelineState
        let isSpecialized: Bool
        if let specializedPipeline = specializedDirectionalPipeline(for: degree) {
            pipeline = specializedPipeline
            isSpecialized = true
        } else if let legacyPipeline = directionalPipeline {
            pipeline = legacyPipeline
            isSpecialized = false
        } else {
            return nil
        }

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

        computeEncoder.label = isSpecialized
            ? "SH Texture Evaluation (Specialized Degree \(degree))"
            : "SH Texture Evaluation (Legacy)"
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
