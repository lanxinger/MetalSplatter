import Foundation

/**
 * Errors that can occur when reading or writing SPX format files
 */
public enum SPXFileFormatError: Error {
    case invalidHeader
    case invalidMagicNumber
    case unsupportedVersion
    case invalidBlockFormat
    case invalidBlockLength
    case decompressionFailed
    case compressionFailed
    case invalidDataBlock
    case insufficientData
    case invalidBoundingBox
}