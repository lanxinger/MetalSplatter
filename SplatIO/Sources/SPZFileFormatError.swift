import Foundation

/**
 * Errors that can occur when reading or writing SPZ format files
 * Aligned with reference SPZ implementation error handling
 */
public enum SplatFileFormatError: Error {
    case invalidHeader
    case unsupportedVersion
    case tooManyPoints
    case unsupportedSHDegree
    case readError
    case writeError
    case compressionError
    case decompressionError
    case decompressionOutputTooLarge
    case fileTooLarge(String)
    case invalidData
    case invalidFormat(String)
    case custom(String)

    // Legacy aliases for compatibility
    static let compressionFailed = compressionError
    static let decompressionFailed = decompressionError
}
