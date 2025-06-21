import Foundation
import simd

/**
 * Utility for validating splat scene data to ensure numerical stability and prevent crashes
 */
public struct SplatDataValidator {
    
    // MARK: - Validation Mode
    
    public enum ValidationMode {
        case strict    // Enforce all range checks
        case lenient   // Only check for NaN/infinity and critical issues
        case safety    // Only check for crash-causing issues (NaN/infinity, bounds)
    }
    
    // MARK: - Constants
    
    /// Maximum reasonable position values (in world units)
    public static let maxPositionMagnitude: Float = 10_000_000.0  // More permissive for large scenes
    
    /// Maximum reasonable scale values 
    public static let maxScaleMagnitude: Float = 1000.0
    public static let minScaleMagnitude: Float = 1e-10  // Much more permissive - only catch truly problematic values
    
    /// Valid opacity range
    public static let opacityRange: ClosedRange<Float> = 0.0...1.0
    
    /// Valid color component range
    public static let colorRange: ClosedRange<Float> = 0.0...1.0
    
    /// Maximum reasonable spherical harmonics coefficient magnitude
    public static let maxSHMagnitude: Float = 100.0
    
    // MARK: - SIMD3<Float> Validation
    
    /// Validates that a SIMD3<Float> contains finite values (no NaN or infinity)
    public static func validateFinite(_ vector: SIMD3<Float>, name: String) throws {
        if !vector.x.isFinite || !vector.y.isFinite || !vector.z.isFinite {
            throw SplatValidationError.invalidDataRange("\(name) contains NaN or infinity values")
        }
    }
    
    /// Validates that a float value is finite
    public static func validateFinite(_ value: Float, name: String) throws {
        if !value.isFinite {
            throw SplatValidationError.invalidDataRange("\(name) contains NaN or infinity")
        }
    }
    
    // MARK: - Position Validation
    
    /// Validates position data for reasonable values and finite numbers
    public static func validatePosition(_ position: SIMD3<Float>, mode: ValidationMode = .lenient) throws {
        try validateFinite(position, name: "Position")
        
        // Only check magnitude in strict mode
        if mode == .strict {
            let magnitude = length(position)
            if magnitude > maxPositionMagnitude {
                throw SplatValidationError.invalidPosition(position, reason: "Position magnitude \(magnitude) exceeds maximum \(maxPositionMagnitude)")
            }
        }
    }
    
    // MARK: - Scale Validation
    
    /// Validates scale data for reasonable values and finite numbers
    public static func validateScale(_ scale: SIMD3<Float>, mode: ValidationMode = .lenient) throws {
        try validateFinite(scale, name: "Scale")
        
        // Always check for negative scales (can cause rendering issues)
        if scale.x < 0 || scale.y < 0 || scale.z < 0 {
            throw SplatValidationError.invalidScale(scale, reason: "Scale components must be non-negative")
        }
        
        // Only check scale ranges in strict mode
        if mode == .strict {
            // Check for very small scales that might cause numerical issues
            if scale.x < minScaleMagnitude || scale.y < minScaleMagnitude || scale.z < minScaleMagnitude {
                throw SplatValidationError.invalidScale(scale, reason: "Scale components \(scale) are too small (minimum: \(minScaleMagnitude))")
            }
            
            // Check for unreasonably large scales
            let maxComponent = max(scale.x, scale.y, scale.z)
            if maxComponent > maxScaleMagnitude {
                throw SplatValidationError.invalidScale(scale, reason: "Scale component \(maxComponent) exceeds maximum \(maxScaleMagnitude)")
            }
        }
    }
    
    // MARK: - Rotation Validation
    
    /// Validates quaternion rotation data
    public static func validateRotation(_ rotation: simd_quatf, mode: ValidationMode = .lenient) throws {
        let vector = rotation.vector
        try validateFinite(SIMD3<Float>(vector.x, vector.y, vector.z), name: "Rotation vector")
        try validateFinite(vector.w, name: "Rotation scalar")
        
        // Only validate quaternion properties in strict or lenient mode
        if mode != .safety {
            let magnitude = length(vector)
            if magnitude < 0.001 {
                throw SplatValidationError.invalidRotation(rotation, reason: "Quaternion magnitude \(magnitude) is too small (near-zero)")
            }
            
            // Only check normalization in strict mode
            if mode == .strict {
                let expectedMagnitude: Float = 1.0
                let tolerance: Float = 0.5  // More permissive tolerance
                if abs(magnitude - expectedMagnitude) > tolerance {
                    throw SplatValidationError.invalidRotation(rotation, reason: "Quaternion magnitude \(magnitude) deviates significantly from unit length")
                }
            }
        }
    }
    
    // MARK: - Opacity Validation
    
    /// Validates opacity values
    public static func validateOpacity(_ opacity: Float, mode: ValidationMode = .lenient) throws {
        try validateFinite(opacity, name: "Opacity")
        
        // Only check range in strict mode
        if mode == .strict && !opacityRange.contains(opacity) {
            throw SplatValidationError.invalidOpacity(opacity, reason: "Opacity \(opacity) is outside valid range \(opacityRange)")
        }
    }
    
    // MARK: - Color Validation
    
    /// Validates color components
    public static func validateColor(_ color: SIMD3<Float>, mode: ValidationMode = .lenient) throws {
        try validateFinite(color, name: "Color")
        
        // Only check range in strict mode
        if mode == .strict {
            if color.x < colorRange.lowerBound || color.x > colorRange.upperBound ||
               color.y < colorRange.lowerBound || color.y > colorRange.upperBound ||
               color.z < colorRange.lowerBound || color.z > colorRange.upperBound {
                throw SplatValidationError.invalidColor(reason: "Color components \(color) are outside valid range \(colorRange)")
            }
        }
    }
    
    // MARK: - Spherical Harmonics Validation
    
    /// Validates spherical harmonics coefficients
    public static func validateSphericalHarmonics(_ sh: [SIMD3<Float>]) throws {
        // Validate expected SH coefficient counts (1, 4, 9, or 16)
        let validCounts = [1, 4, 9, 16]
        if !validCounts.contains(sh.count) {
            throw SplatValidationError.invalidSphericalHarmonics(sh, reason: "Invalid SH coefficient count \(sh.count), expected one of: \(validCounts)")
        }
        
        // Validate each coefficient
        for (index, coefficient) in sh.enumerated() {
            try validateFinite(coefficient, name: "SH coefficient \(index)")
            
            let magnitude = length(coefficient)
            if magnitude > maxSHMagnitude {
                throw SplatValidationError.invalidSphericalHarmonics(sh, reason: "SH coefficient \(index) magnitude \(magnitude) exceeds maximum \(maxSHMagnitude)")
            }
        }
    }
    
    // MARK: - Complete Point Validation
    
    /// Validates a complete SplatScenePoint for all potential issues
    public static func validatePoint(_ point: SplatScenePoint, mode: ValidationMode = .lenient) throws {
        // Validate position
        try validatePosition(point.position, mode: mode)
        
        // Validate scale based on the scale type
        let scaleVector = point.scale.asLinearFloat
        try validateScale(scaleVector, mode: mode)
        
        // Validate rotation
        try validateRotation(point.rotation, mode: mode)
        
        // Validate opacity based on opacity type
        let opacityValue = point.opacity.asLinearFloat
        try validateOpacity(opacityValue, mode: mode)
        
        // Validate color based on color type
        let colorVector = point.color.asLinearFloat
        try validateColor(colorVector, mode: mode)
        
        // Validate spherical harmonics if present (only in strict mode)
        if mode == .strict {
            let shCoefficients = point.color.asSphericalHarmonic
            try validateSphericalHarmonics(shCoefficients)
        }
    }
    
    // MARK: - Batch Validation
    
    /// Validates an array of points, collecting all errors before throwing
    public static func validatePoints(_ points: [SplatScenePoint], mode: ValidationMode = .lenient) throws {
        var errors: [SplatValidationError] = []
        
        // Only sample a subset for performance on large datasets
        let sampleSize = min(points.count, 1000)
        let step = max(1, points.count / sampleSize)
        
        for i in stride(from: 0, to: points.count, by: step) {
            do {
                try validatePoint(points[i], mode: mode)
            } catch let error as SplatValidationError {
                errors.append(error)
                
                // Stop after collecting some errors to avoid memory issues
                if errors.count >= 50 {
                    break
                }
            }
        }
        
        // Only throw if error rate is very high
        let errorThreshold: Float = mode == .safety ? 0.5 : (mode == .lenient ? 0.2 : 0.1)
        let errorRate = Float(errors.count) / Float(sampleSize)
        if errorRate > errorThreshold {
            throw SplatValidationError.corruptedData("High validation error rate: \(Int(errorRate * 100))% of sampled points failed validation. First error: \(errors[0].localizedDescription)")
        }
    }
    
    // MARK: - Data Bounds Checking
    
    /// Validates that data access is within bounds
    public static func validateDataBounds(data: Data, offset: Int, size: Int) throws {
        guard offset >= 0 else {
            throw SplatValidationError.dataOutOfBounds(offset: offset, size: size, available: data.count)
        }
        
        guard offset + size <= data.count else {
            throw SplatValidationError.dataOutOfBounds(offset: offset, size: size, available: data.count)
        }
    }
    
    /// Safe data access with bounds checking
    public static func safeDataAccess<T>(data: Data, offset: Int, type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        try validateDataBounds(data: data, offset: offset, size: size)
        
        return data.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: offset, as: T.self)
        }
    }
}