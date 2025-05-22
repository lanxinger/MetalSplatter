import Foundation

/**
 * Errors that can occur when reading or writing SPZ format files
 */
public enum SplatFileFormatError: Error {
    case invalidHeader
    case unsupportedVersion
    case invalidData
    case decompressionFailed
    case compressionFailed
}
