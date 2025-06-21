import Foundation
import simd

/**
 * Comprehensive validation errors for splat scene data
 */
public enum SplatValidationError: LocalizedError {
    case invalidPosition(SIMD3<Float>, reason: String)
    case invalidScale(SIMD3<Float>, reason: String)
    case invalidRotation(simd_quatf, reason: String)
    case invalidOpacity(Float, reason: String)
    case invalidColor(reason: String)
    case invalidSphericalHarmonics([SIMD3<Float>], reason: String)
    case dataOutOfBounds(offset: Int, size: Int, available: Int)
    case invalidDataRange(String)
    case corruptedData(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidPosition(let pos, let reason):
            return "Invalid position (\(pos.x), \(pos.y), \(pos.z)): \(reason)"
        case .invalidScale(let scale, let reason):
            return "Invalid scale (\(scale.x), \(scale.y), \(scale.z)): \(reason)"
        case .invalidRotation(let quat, let reason):
            return "Invalid rotation (\(quat.vector.x), \(quat.vector.y), \(quat.vector.z), \(quat.vector.w)): \(reason)"
        case .invalidOpacity(let opacity, let reason):
            return "Invalid opacity \(opacity): \(reason)"
        case .invalidColor(let reason):
            return "Invalid color: \(reason)"
        case .invalidSphericalHarmonics(let sh, let reason):
            return "Invalid spherical harmonics (count: \(sh.count)): \(reason)"
        case .dataOutOfBounds(let offset, let size, let available):
            return "Data access out of bounds: trying to read \(size) bytes at offset \(offset), but only \(available) bytes available"
        case .invalidDataRange(let reason):
            return "Invalid data range: \(reason)"
        case .corruptedData(let reason):
            return "Corrupted data: \(reason)"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .invalidPosition, .invalidScale, .invalidRotation, .invalidOpacity, .invalidColor, .invalidSphericalHarmonics:
            return "Check the source data for NaN, infinity, or out-of-range values"
        case .dataOutOfBounds, .invalidDataRange, .corruptedData:
            return "Verify the file format and integrity of the source data"
        }
    }
}