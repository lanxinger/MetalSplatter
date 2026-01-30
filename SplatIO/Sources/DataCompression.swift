import Foundation
import zlib

/// Extension for Data to add gzip decompression capabilities
extension Data {
    /// Decompress gzipped Data
    func gunzipped() throws -> Data {
        guard self.count > 2 else { 
            throw SplatFileFormatError.decompressionFailed 
        }
        
        // Check for gzip signature
        guard self[0] == 0x1F && self[1] == 0x8B else {
            throw SplatFileFormatError.decompressionFailed
        }
        
        // Prepare for zlib inflation
        var stream = z_stream()
        var status: Int32
        
        // Initialize zlib
        status = inflateInit2_(&stream, MAX_WBITS + 32, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
        guard status == Z_OK else {
            throw SplatFileFormatError.decompressionFailed
        }
        
        // Setup the source buffer
        return try self.withUnsafeBytes { (inputPointer: UnsafeRawBufferPointer) -> Data in
            guard let baseAddress = inputPointer.bindMemory(to: Bytef.self).baseAddress else {
                throw SplatFileFormatError.decompressionFailed
            }
            stream.next_in = UnsafeMutablePointer<Bytef>(mutating: baseAddress)
            stream.avail_in = uInt(self.count)
            
            // Prepare for output
            // Increased from 16KB to 128KB for better decompression performance
            let bufferSize = self.count > (10 * 1024 * 1024) ? (256 * 1024) : (128 * 1024)
            let buffer = UnsafeMutablePointer<Bytef>.allocate(capacity: bufferSize)
            defer { buffer.deallocate() }
            
            // Setup the destination buffer
            var result = Data()
            repeat {
                stream.next_out = buffer
                stream.avail_out = uInt(bufferSize)
                
                // Inflate and check for errors
                status = inflate(&stream, Z_NO_FLUSH)
                
                guard status != Z_NEED_DICT &&
                      status != Z_DATA_ERROR &&
                      status != Z_MEM_ERROR else {
                    _ = inflateEnd(&stream) // Use result to avoid warning
                    throw SplatFileFormatError.decompressionFailed
                }
                
                // Calculate how much data we got
                let bytesDecompressed = bufferSize - Int(stream.avail_out)
                if bytesDecompressed > 0 {
                    result.append(buffer, count: bytesDecompressed)
                }
                
            } while status != Z_STREAM_END && stream.avail_out == 0
            
            // Clean up and return
            _ = inflateEnd(&stream) // Use result to avoid warning
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
