import Foundation
import zlib

/// Extension for Data to add gzip decompression capabilities
extension Data {
    /// Maximum allowed decompressed size (512 MB) - prevents decompression bombs
    static let maxDecompressedSize: Int = 512 * 1024 * 1024

    /// Decompress gzipped Data with optional size limit
    /// - Parameter maxOutputSize: Maximum allowed output size in bytes (default: 512 MB)
    /// - Returns: Decompressed data
    /// - Throws: SplatFileFormatError.decompressionOutputTooLarge if output exceeds limit
    func gunzipped(maxOutputSize: Int = Data.maxDecompressedSize) throws -> Data {
        guard self.count > 2 else {
            throw SplatFileFormatError.decompressionFailed
        }

        // Check for gzip signature
        guard self[startIndex] == 0x1F && self[startIndex + 1] == 0x8B else {
            throw SplatFileFormatError.decompressionFailed
        }

        return try inflated(windowBits: Data.gzipOrZlibWindowBits, maxOutputSize: maxOutputSize)
    }

    /// zlib `windowBits` that accept a gzip or zlib header.
    static let gzipOrZlibWindowBits: Int32 = MAX_WBITS + 32
    /// zlib `windowBits` for a raw deflate stream without any header.
    static let rawDeflateWindowBits: Int32 = -MAX_WBITS

    /// Inflates a deflate stream, never producing more than `maxOutputSize` bytes.
    /// - Parameters:
    ///   - windowBits: `gzipOrZlibWindowBits` or `rawDeflateWindowBits`.
    ///   - truncating: Return the first `maxOutputSize` bytes instead of throwing when the stream
    ///     holds more, for reading a header.
    /// - Throws: `SplatFileFormatError.decompressionOutputTooLarge` past the limit, or
    ///   `decompressionFailed` for invalid data. A truncated stream returns what it holds.
    func inflated(windowBits: Int32, maxOutputSize: Int, truncating: Bool = false) throws -> Data {
        guard !isEmpty, maxOutputSize > 0 else {
            throw SplatFileFormatError.decompressionFailed
        }

        var stream = z_stream()
        guard inflateInit2_(&stream, windowBits, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw SplatFileFormatError.decompressionFailed
        }
        defer { _ = inflateEnd(&stream) }

        return try self.withUnsafeBytes { (inputPointer: UnsafeRawBufferPointer) -> Data in
            guard let baseAddress = inputPointer.bindMemory(to: Bytef.self).baseAddress else {
                throw SplatFileFormatError.decompressionFailed
            }
            stream.next_in = UnsafeMutablePointer<Bytef>(mutating: baseAddress)
            stream.avail_in = uInt(self.count)

            let bufferSize = Swift.min(self.count > (10 * 1024 * 1024) ? (256 * 1024) : (128 * 1024), maxOutputSize + 1)
            let buffer = UnsafeMutablePointer<Bytef>.allocate(capacity: bufferSize)
            defer { buffer.deallocate() }

            var result = Data()
            var status: Int32
            repeat {
                stream.next_out = buffer
                stream.avail_out = uInt(bufferSize)

                status = inflate(&stream, Z_NO_FLUSH)
                guard status != Z_NEED_DICT &&
                      status != Z_DATA_ERROR &&
                      status != Z_MEM_ERROR else {
                    throw SplatFileFormatError.decompressionFailed
                }

                let bytesDecompressed = bufferSize - Int(stream.avail_out)
                if bytesDecompressed > maxOutputSize - result.count {
                    guard truncating else {
                        throw SplatFileFormatError.decompressionOutputTooLarge
                    }
                    result.append(buffer, count: maxOutputSize - result.count)
                    return result
                }
                if bytesDecompressed > 0 {
                    result.append(buffer, count: bytesDecompressed)
                }
            } while status != Z_STREAM_END && stream.avail_out == 0

            return result
        }
    }
}

// Helper constants from zlib.h
private let MAX_WBITS: Int32 = 15
private let Z_NO_FLUSH: Int32 = 0
private let Z_OK: Int32 = 0
private let Z_STREAM_END: Int32 = 1
private let Z_NEED_DICT: Int32 = 2
private let Z_DATA_ERROR: Int32 = -3
private let Z_MEM_ERROR: Int32 = -4

/// Import zlib functions
@_silgen_name("inflateInit2_")
private func inflateInit2_(_ strm: UnsafeMutablePointer<z_stream>,
                          _ windowBits: Int32,
                          _ version: UnsafePointer<Int8>,
                          _ stream_size: Int32) -> Int32

@_silgen_name("inflate")
private func inflate(_ strm: UnsafeMutablePointer<z_stream>, _ flush: Int32) -> Int32

@_silgen_name("inflateEnd")
private func inflateEnd(_ strm: UnsafeMutablePointer<z_stream>) -> Int32
