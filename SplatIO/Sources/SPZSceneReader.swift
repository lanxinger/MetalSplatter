import Foundation
import Compression
import simd
import os

#if canImport(Metal)
import Metal
#endif

// Logger for SPZ scene reading (uses .debug level for hot-path logs)
private let spzLog = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.splatIO", category: "SPZSceneReader")

/**
 * Reader for SPZ format Gaussian splat scenes.
 * SPZ is a compact binary format for Gaussian splats with support for:
 * - Float16 or fixed-point position encoding
 * - Spherical harmonics for color representation
 * - Antialiasing support
 */
public class SPZSceneReader: SplatSceneReader {
    private var data: Data
    
    public init(data: Data) {
        self.data = data
    }
    
    public convenience init(contentsOf url: URL) throws {
        spzLog.debug("Trying to load file: \(url.path)")
        spzLog.debug("File extension: \(url.pathExtension)")
        
        // Make sure the file exists and is readable
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            spzLog.debug(" File is not readable: \(url.path)")
            throw SplatFileFormatError.invalidData
        }
        
        do {
            let fileData = try Data(contentsOf: url)
            spzLog.debug(" Successfully read data, size: \(fileData.count) bytes")
            
            // Initialize with original data first
            self.init(data: fileData)
            
            // Special handling for iOS files from Downloads folder (they're often gzipped)
            if url.path.contains("/Containers/Shared/AppGroup/") && url.path.contains("/File Provider Storage/Downloads/") {
                spzLog.debug(" Detected iOS downloads file - using specialized handling")
                if processIOSDownloadFile(fileData) {
                    return
                }
            }
            
            // First attempt - try to load it as uncompressed SPZ
            spzLog.debug(" First attempt - trying to load as uncompressed SPZ")
            
            // Check for SPZ magic number
            if fileData.count >= 4 {
                // Safer way to load UInt32 from potentially unaligned data
                var magic: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                    fileData.prefix(4).copyBytes(to: magicPtr)
                }
                spzLog.debug(" No SPZ magic number found at start: 0x\(String(format: "%08X", magic))")
            }
            
            // Check if the file is gzipped
            if Self.isGzipped(fileData) {
                spzLog.debug(" Is gzipped: true")
                if let decompressedData = Self.decompressGzipped(fileData) {
                    spzLog.debug(" Successfully decompressed data: \(decompressedData.count) bytes")
                    self.data = decompressedData
                }
            }
        } catch {
            spzLog.debug(" Error reading file: \(error)")
            throw error
        }
    }
    
    // Special function to handle iOS downloaded files (often problematic gzipped files)
    private func processIOSDownloadFile(_ fileData: Data) -> Bool {
        spzLog.debug(" Trying iOS-specific handling for downloaded file")
        
        // Try the GZipArchive approach (used by many iOS apps)
        if let decompressed = Self.decompressIOSGzippedFile(fileData) {
            spzLog.debug(" Successfully decompressed iOS file: \(decompressed.count) bytes")
            self.data = decompressed
            return true
        }
        
        // Try to find a magic number in the file and extract from there
        if fileData.count > 100 {
            for offset in stride(from: 0, to: min(fileData.count - 20, 10000), by: 1) {
                let currentBytes = fileData[offset..<min(offset+4, fileData.count)]
                if currentBytes.count == 4 {
                    // Safer way to load UInt32 from potentially unaligned data
                    var magic: UInt32 = 0
                    _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                        currentBytes.copyBytes(to: magicPtr)
                    }
                    
                    // Check for NGSP in little endian
                    if magic == 0x5053474E {
                        spzLog.debug(" Found SPZ magic number at offset \(offset)")
                        let extractedData = fileData.subdata(in: offset..<fileData.count)
                        spzLog.debug(" Extracted \(extractedData.count) bytes starting from magic number")
                        self.data = extractedData
                        return true
                    }
                }
            }
        }
        
        spzLog.debug(" iOS-specific handling did not find a solution")
        return false
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        spzLog.debug("Attempting to deserialize data, size: \(self.data.count) bytes")
        
        // Try standard deserialization first
        do {
            let packedGaussians = try PackedGaussians.deserialize(data)
            spzLog.debug(" Successfully deserialized \(packedGaussians.numPoints) points")
            let points = unpackGaussians(packedGaussians)
            spzLog.debug(" Successfully unpacked \(points.count) points")
            return points
        } catch let deserializationError {
            spzLog.debug(" Standard deserialization failed: \(deserializationError)")
            
            // If the standard approach fails, try a more aggressive fallback
            spzLog.debug(" Trying alternative deserialization approach")
            
            // Check if we can find the SPZ magic number anywhere in the file
            if data.count > 20 {
                for offset in stride(from: 0, to: min(1024, data.count - 16), by: 4) {
                    guard offset + 4 <= data.count else { break }
                    
                    let magicBytes = data[offset..<(offset+4)]
                    // Safer way to load UInt32 from potentially unaligned data
                    var magic: UInt32 = 0
                    _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                        magicBytes.copyBytes(to: magicPtr)
                    }
                    
                    // Check for NGSP magic (0x5053474E in little endian)
                    if magic == 0x5053474E {
                        spzLog.debug(" Found SPZ magic at offset \(offset), trying to parse from there")
                        
                        // Create a new data object starting from the magic number
                        let offsetData = data.subdata(in: offset..<data.count)
                        
                        do {
                            let packedGaussians = try PackedGaussians.deserialize(offsetData)
                            spzLog.debug(" Successfully deserialized \(packedGaussians.numPoints) points from offset \(offset)")
                            let points = unpackGaussians(packedGaussians)
                            spzLog.debug(" Successfully unpacked \(points.count) points")
                            return points
                        } catch {
                            spzLog.debug(" Failed to parse from offset \(offset): \(error)")
                            // Continue searching for another magic number
                        }
                    }
                }
            }
            
            // If all fallbacks fail, rethrow the original error
            spzLog.debug(" All deserialization approaches failed")
            throw deserializationError
        }
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        do {
            delegate.didStartReading(withPointCount: nil)
            let points = try readScene()
            delegate.didRead(points: points)
            delegate.didFinishReading()
        } catch {
            delegate.didFailReading(withError: error)
        }
    }
    
    // MARK: - Private helpers
    
    private func unpackGaussians(_ packedGaussians: PackedGaussians) -> [SplatScenePoint] {
        // Implementation based on the original C++ code with SIMD optimizations
        var results = [SplatScenePoint]()
        
        // Safety checks matching C++ reference implementation  
        let maxPointsToRead = 10000000  // C++ constant: constexpr int32_t maxPointsToRead = 10000000;
        if packedGaussians.numPoints > maxPointsToRead {
            spzLog.debug(" Too many points: \(packedGaussians.numPoints), capping at \(maxPointsToRead)")
        }
        if packedGaussians.shDegree > 3 {
            spzLog.debug(" Unsupported SH degree: \(packedGaussians.shDegree), SPZ spec allows 0-3")
        }
        let safeNumPoints = min(packedGaussians.numPoints, maxPointsToRead)
        spzLog.debug(" Unpacking \(safeNumPoints) points with SH degree \(packedGaussians.shDegree)")
        
        // Determine data layout based on format
        let positionStride = packedGaussians.usesFloat16 ? 6 : 9 // bytes per position
        let rotationStride = packedGaussians.usesQuaternionSmallestThree ? 4 : 3 // bytes per rotation
        let shStride = 3 // RGB components per SH coefficient
        
        // Validate the data format matches the C++ implementation expectations
        let positionBytesPerPoint = positionStride
        let expectedPositionBytes = safeNumPoints * positionBytesPerPoint
        let expectedScaleBytes = safeNumPoints * 3
        let expectedRotationBytes = safeNumPoints * rotationStride
        let expectedAlphaBytes = safeNumPoints
        let expectedColorBytes = safeNumPoints * 3
        let shDim = shDimForDegree(packedGaussians.shDegree)  // Use correct Niantic formula
        let expectedSHBytes = safeNumPoints * shDim * 3
        
        // Log actual vs expected data sizes
        spzLog.debug(" Data size validation:")
        spzLog.debug("  Positions: \(packedGaussians.positions.count)/\(expectedPositionBytes) bytes")
        spzLog.debug("  Scales: \(packedGaussians.scales.count)/\(expectedScaleBytes) bytes")
        spzLog.debug("  Rotations: \(packedGaussians.rotations.count)/\(expectedRotationBytes) bytes")
        spzLog.debug("  Alphas: \(packedGaussians.alphas.count)/\(expectedAlphaBytes) bytes")
        spzLog.debug("  Colors: \(packedGaussians.colors.count)/\(expectedColorBytes) bytes")
        spzLog.debug("  SH: \(packedGaussians.sh.count)/\(expectedSHBytes) bytes")
        
        // Check if we have any data at all
        guard !packedGaussians.positions.isEmpty && 
              !packedGaussians.scales.isEmpty && 
              !packedGaussians.rotations.isEmpty && 
              !packedGaussians.colors.isEmpty && 
              !packedGaussians.alphas.isEmpty else {
            spzLog.debug(" Missing essential component data")
            return results
        }
        
        // Process in smaller batches for better memory management and parallelization
        let chunkSize = 10000
        let chunks = stride(from: 0, to: safeNumPoints, by: chunkSize).map { startIdx -> Range<Int> in
            let endIdx = min(startIdx + chunkSize, safeNumPoints)
            return startIdx..<endIdx
        }
        
        // Use concurrent processing for large datasets
        let useParallelProcessing = safeNumPoints > 100000

        // Create a container for results that will be populated concurrently
        var chunkResults = Array<[SplatScenePoint]?>(repeating: nil, count: chunks.count)

        if useParallelProcessing {
            // Use withUnsafeMutableBufferPointer + concurrentPerform for thread-safe parallel writes
            chunkResults.withUnsafeMutableBufferPointer { buffer in
                DispatchQueue.concurrentPerform(iterations: chunks.count) { chunkIndex in
                    let chunkRange = chunks[chunkIndex]
                    buffer[chunkIndex] = self.processPointChunk(packedGaussians: packedGaussians,
                                                                range: chunkRange,
                                                                positionStride: positionStride,
                                                                rotationStride: rotationStride,
                                                                shDim: shDim,
                                                                shStride: shStride)
                }
            }
        } else {
            // Process sequentially for smaller datasets
            for (chunkIndex, chunkRange) in chunks.enumerated() {
                chunkResults[chunkIndex] = self.processPointChunk(packedGaussians: packedGaussians,
                                                                  range: chunkRange,
                                                                  positionStride: positionStride,
                                                                  rotationStride: rotationStride,
                                                                  shDim: shDim,
                                                                  shStride: shStride)

                // Progress reporting for sequential processing
                if chunkRange.lowerBound > 0 && chunkRange.lowerBound % 100000 == 0 {
                    spzLog.debug(" Unpacked \(chunkRange.lowerBound) points...")
                }
            }
        }
        
        // Combine results from all chunks
        results.reserveCapacity(safeNumPoints)
        for chunkResult in chunkResults {
            if let points = chunkResult {
                results.append(contentsOf: points)
            }
        }
        
        spzLog.debug(" Successfully unpacked \(results.count) points")
        return results
    }
    
    // Process a chunk of points using SIMD operations for better performance
    // Uses C++ reference coordinate conversion: RUB (SPZ internal) -> target coordinate system
    private func processPointChunk(packedGaussians: PackedGaussians,
                                 range: Range<Int>,
                                 positionStride: Int,
                                 rotationStride: Int,
                                 shDim: Int,
                                 shStride: Int) -> [SplatScenePoint] {
        var chunkResults = [SplatScenePoint]()
        chunkResults.reserveCapacity(range.count)

        // SPZ uses RUB coordinate system internally, convert to target system
        // For now, assume we want RDF (PLY) coordinate system for compatibility
        let coordinateConverter = CoordinateConverter.converter(from: .rub, to: .rdf)

        // Color scale factor from original C++ implementation
        let colorScale: Float = 0.15

        // Fixed-point scale for non-float16 positions
        let fixedPointScale = 1.0 / Float(1 << packedGaussians.fractionalBits)

        // PERF: Hoist withUnsafeBytes outside the loop to avoid per-point closure overhead.
        // Get raw pointers once, use them for all points in the chunk.
        packedGaussians.positions.withUnsafeBytes { positionsBuffer in
            packedGaussians.rotations.withUnsafeBytes { rotationsBuffer in
                let positionsBase = positionsBuffer.baseAddress
                let rotationsBase = rotationsBuffer.baseAddress
                let positionsCount = packedGaussians.positions.count
                let rotationsCount = packedGaussians.rotations.count

                for i in range {
                    // Calculate offsets
                    let posOffset = i * positionStride
                    let colorOffset = i * 3
                    let scaleOffset = i * 3
                    let rotOffset = i * rotationStride
                    let shOffset = i * shDim * shStride

                    // Check if all essential components are in bounds
                    guard i < packedGaussians.alphas.count &&
                          colorOffset + 2 < packedGaussians.colors.count &&
                          scaleOffset + 2 < packedGaussians.scales.count &&
                          rotOffset + (rotationStride - 1) < rotationsCount &&
                          posOffset + (positionStride - 1) < positionsCount else {
                        continue
                    }

                    // Extract position using proper decoding based on the format
                    var position = SIMD3<Float>(0, 0, 0)

                    if packedGaussians.usesFloat16 {
                        // Decode float16 positions using hoisted raw pointer
                        if posOffset + 5 < positionsCount, let base = positionsBase?.advanced(by: posOffset) {
                            // Use loadUnaligned to safely read potentially misaligned UInt16 values
                            let x = float16ToFloat32(base.loadUnaligned(as: UInt16.self))
                            let y = float16ToFloat32(base.advanced(by: 2).loadUnaligned(as: UInt16.self))
                            let z = float16ToFloat32(base.advanced(by: 4).loadUnaligned(as: UInt16.self))
                            position = SIMD3<Float>(
                                x * coordinateConverter.flipP[0],
                                y * coordinateConverter.flipP[1],
                                z * coordinateConverter.flipP[2]
                            )
                        }
                    } else {
                        // Decode fixed-point positions
                        if posOffset + 8 < positionsCount {
                            // Process all three position components
                            for j in 0..<3 {
                                let byteOffset = posOffset + j * 3
                                var fixed32: Int32 = Int32(packedGaussians.positions[byteOffset])
                                fixed32 |= Int32(packedGaussians.positions[byteOffset + 1]) << 8
                                fixed32 |= Int32(packedGaussians.positions[byteOffset + 2]) << 16

                                // Apply sign extension for negative values
                                if (fixed32 & 0x800000) != 0 {
                                    fixed32 |= Int32(bitPattern: 0xFF000000)
                                }

                                position[j] = Float(fixed32) * fixedPointScale * coordinateConverter.flipP[j]
                            }
                        } else {
                            // Use a fallback position if we can't decode properly
                            position = SIMD3<Float>(
                                Float(i % 100) * 0.1,
                                Float(i / 100 % 100) * 0.1,
                                Float(i / 10000) * 0.1
                            )
                        }
                    }

                    // Extract scale using SIMD operations
                    var scale = SIMD3<Float>(-5, -5, -5) // Default to a small scale
                    if scaleOffset + 2 < packedGaussians.scales.count {
                        // Convert all scale values in one operation
                        let scaleBytes = SIMD3<Float>(
                            Float(packedGaussians.scales[scaleOffset]),
                            Float(packedGaussians.scales[scaleOffset + 1]),
                            Float(packedGaussians.scales[scaleOffset + 2])
                        )
                        scale = scaleBytes / 16.0 - 10.0
                    }

                    // Extract rotation using hoisted raw pointer (zero-allocation)
                    var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) // Default identity quaternion
                    if rotOffset + (rotationStride - 1) < rotationsCount,
                       let base = rotationsBase?.advanced(by: rotOffset) {
                        // Use appropriate unpacking function based on format
                        if packedGaussians.usesQuaternionSmallestThree {
                            unpackQuaternionSmallestThreeUnsafe(&rotation, base, coordinateConverter)
                        } else {
                            unpackQuaternionFirstThreeUnsafe(&rotation, base, coordinateConverter)
                        }
                    }

                    // Extract alpha (apply logit transformation)
                    let alpha: Float
                    if i < packedGaussians.alphas.count {
                        alpha = logit(Float(packedGaussians.alphas[i]) / 255.0)
                    } else {
                        alpha = 0.0 // Default fully transparent
                    }

                    // Extract colors and SH coefficients using SIMD operations
                    var sphericalHarmonics = [SIMD3<Float>()]

                    // First extract the DC term (from colors or first SH coefficient)
                    if colorOffset + 2 < packedGaussians.colors.count {
                        // Extract all color components at once
                        let colorBytes = SIMD3<Float>(
                            Float(packedGaussians.colors[colorOffset]),
                            Float(packedGaussians.colors[colorOffset + 1]),
                            Float(packedGaussians.colors[colorOffset + 2])
                        )

                        // Apply the full transformation in one SIMD operation
                        let colorVector = (colorBytes / 255.0 - 0.5) / colorScale
                        sphericalHarmonics[0] = colorVector
                    }

                    // Extract additional SH coefficients if present and needed
                    // SH data is organized with color channel as inner axis, coefficient as outer axis
                    // For degree 1: sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, sh10_g, sh10_b, sh1p1_r, sh1p1_g, sh1p1_b
                    if shDim > 1 && shOffset + (shDim-1)*3 < packedGaussians.sh.count {
                        // We already have the DC term (first coefficient), so start from the second
                        for j in 1..<min(shDim, 15) { // Limit to 15 max coefficients per channel
                            let idx = shOffset + j * 3
                            if idx + 2 < packedGaussians.sh.count {
                                // Process all three SH components at once (R, G, B for coefficient j)
                                let shBytes = SIMD3<Float>(
                                    Float(packedGaussians.sh[idx]),     // R component
                                    Float(packedGaussians.sh[idx + 1]), // G component
                                    Float(packedGaussians.sh[idx + 2])  // B component
                                )

                                // Use SIMD to unquantize all components at once and apply coordinate conversion
                                let shCoeffs = (shBytes - 128.0) / 128.0
                                let flip = coordinateConverter.flipSh[j]
                                sphericalHarmonics.append(shCoeffs * flip)
                            }
                        }
                    }

                    // Create the point with properly decoded values
                    let point = SplatScenePoint(
                        position: position,
                        color: .sphericalHarmonic(sphericalHarmonics),
                        opacity: .logitFloat(alpha),
                        scale: .exponent(scale),
                        rotation: rotation
                    )

                    // Add to results
                    chunkResults.append(point)
                } // end for i in range
            } // end rotations.withUnsafeBytes
        } // end positions.withUnsafeBytes

        return chunkResults
    }
    
    // Helper function to convert sigmoid to logit
    private func logit(_ x: Float) -> Float {
        let safe_x = max(0.0001, min(0.9999, x)) // Clamp to avoid log(0) or log(1)
        return log(safe_x / (1.0 - safe_x))
    }
    
    private static func isGzipped(_ data: Data) -> Bool {
        return data.count >= 2 && data[0] == 0x1F && data[1] == 0x8B
    }

    // Special method for iOS files that don't decompress with standard methods
    private static func decompressIOSGzippedFile(_ data: Data) -> Data? {
        guard data.count > 10 else { return nil } // Need at least the gzip header

        spzLog.debug(" Attempting iOS-specific gzip decompression")

        let maxDecompressedSize = Data.maxDecompressedSize

        // First try to use the Compression framework directly
        do {
            var decompressed = Data()
            var tooLarge = false
            // Skip the first 10 bytes (gzip header)
            let compressedData = data.subdata(in: 10..<data.count)

            // Initialize a decoder for zlib with size limit check
            let outputFilter = try OutputFilter(.decompress, using: .zlib) { (chunk: Data?) -> Void in
                if let chunk = chunk, !tooLarge {
                    decompressed.append(chunk)
                    if decompressed.count > maxDecompressedSize {
                        tooLarge = true
                        decompressed.removeAll()  // Free memory
                    }
                }
            }

            // Process the data
            try outputFilter.write(compressedData)
            try outputFilter.finalize()

            // Check if decompression was aborted due to size
            if tooLarge {
                spzLog.debug(" iOS-specific decompression aborted - output too large")
                return nil
            }

            // If we got something reasonable, return it
            if decompressed.count > 1000 {
                spzLog.debug(" iOS-specific decompression succeeded with \(decompressed.count) bytes")
                return decompressed
            }
        } catch {
            spzLog.debug(" iOS-specific decompression failed: \(error)")
        }

        // Second approach: try several offsets into the file
        let possibleOffsets = [0, 10, 16, 18, 20, 24, 32]

        for offset in possibleOffsets {
            guard offset < data.count else { continue }

            let compressedData = data.subdata(in: offset..<data.count)
            do {
                let decompressed = try (compressedData as NSData).decompressed(using: .zlib) as Data
                // Post-check size (limited protection - NSData already allocated)
                guard decompressed.count <= maxDecompressedSize else {
                    spzLog.debug(" iOS decompression at offset \(offset) exceeded size limit")
                    continue
                }
                if decompressed.count > 1000 {
                    spzLog.debug(" iOS decompression succeeded at offset \(offset) with \(decompressed.count) bytes")
                    return decompressed
                }
            } catch {
                // Just try the next offset
            }
        }

        return nil
    }
    
    private static func decompressGzipped(_ data: Data) -> Data? {
        // Simple check for gzip header
        guard data.count >= 2, data[0] == 0x1F, data[1] == 0x8B else {
            spzLog.debug(" Not a gzip file (wrong magic bytes)")
            return nil
        }

        spzLog.debug(" Attempting to decompress gzipped data")

        let maxDecompressedSize = Data.maxDecompressedSize

        // First try our streaming gunzipped() which has built-in size limits
        do {
            let decompressed = try data.gunzipped(maxOutputSize: maxDecompressedSize)
            spzLog.debug(" Successfully decompressed with streaming gunzipped: \(decompressed.count) bytes")
            return decompressed
        } catch SplatFileFormatError.decompressionOutputTooLarge {
            spzLog.debug(" Decompression aborted - output too large")
            return nil
        } catch {
            spzLog.debug(" Streaming gunzipped failed: \(error)")
        }

        // Fall back to built-in methods with post-check size limits
        spzLog.debug(" Trying built-in decompression methods")

        // Try each compression algorithm with size check
        do {
            let decompressed = try (data as NSData).decompressed(using: .zlib) as Data
            guard decompressed.count <= maxDecompressedSize else {
                spzLog.debug(" .zlib decompression exceeded size limit")
                return nil
            }
            spzLog.debug(" Successfully decompressed with .zlib algorithm")
            return decompressed
        } catch {
            spzLog.debug(" Decompression with .zlib failed")
        }

        do {
            let decompressed = try (data as NSData).decompressed(using: .lzfse) as Data
            guard decompressed.count <= maxDecompressedSize else {
                spzLog.debug(" .lzfse decompression exceeded size limit")
                return nil
            }
            spzLog.debug(" Successfully decompressed with .lzfse algorithm")
            return decompressed
        } catch {
            spzLog.debug(" Decompression with .lzfse failed")
        }

        if #available(iOS 13.0, macOS 10.15, *) {
            do {
                let decompressed = try (data as NSData).decompressed(using: .lz4) as Data
                guard decompressed.count <= maxDecompressedSize else {
                    spzLog.debug(" .lz4 decompression exceeded size limit")
                    return nil
                }
                spzLog.debug(" Successfully decompressed with .lz4 algorithm")
                return decompressed
            } catch {
                spzLog.debug(" Decompression with .lz4 failed")
            }
        }

        // Try to decompress with gzip header skipped as a last resort
        spzLog.debug(" Attempting gzip payload decoder")
        do {
            // Skip the gzip header (10 bytes) and try to decompress the payload
            if data.count > 10 {
                let payloadData = data.subdata(in: 10..<data.count)
                let decompressed = try (payloadData as NSData).decompressed(using: .zlib) as Data
                guard decompressed.count <= maxDecompressedSize else {
                    spzLog.debug(" Gzip payload decompression exceeded size limit")
                    return nil
                }
                spzLog.debug(" Successfully decompressed gzip payload: \(decompressed.count) bytes")
                return decompressed
            }
        } catch {
            spzLog.debug(" Gzip payload decoder failed: \(error)")
        }

        spzLog.debug(" All decompression attempts failed")
        return nil
    }
}
