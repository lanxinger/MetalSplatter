import Foundation
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
    public private(set) var isAntialiased = false
    
    /// Reads SPZ from memory: a raw (v4) stream, or a gzip-compressed (v1–v3) one.
    public init(data: Data) {
        if Self.isGzipped(data), let decompressed = Self.decompressGzipped(data) {
            self.data = decompressed
        } else {
            self.data = data
        }
        refreshMetadata()
    }

    /// The antialiased flag of an SPZ stream, read from its header without decoding the splats.
    static func headerIsAntialiased(_ data: Data) -> Bool {
        let headerBytes = Self.isGzipped(data)
            ? Self.gzipPrefix(data, count: PackedGaussiansHeader.size)
            : data.prefix(PackedGaussiansHeader.size)
        guard let headerBytes, let header = try? PackedGaussiansHeader(data: Data(headerBytes)) else { return false }
        return (header.flags & PackedGaussiansHeader.FlagAntialiased) != 0
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

            // init(data:) already gunzips v1–v3 files, with a size limit.
            self.init(data: fileData)
            defer { self.refreshMetadata() }
            if Self.isGzipped(fileData) {
                return
            }
            
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
            
        } catch {
            spzLog.debug(" Error reading file: \(error)")
            throw error
        }
    }
    
    // Special function to handle iOS downloaded files (often problematic gzipped files)
    private func processIOSDownloadFile(_ fileData: Data) -> Bool {
        spzLog.debug(" Trying iOS-specific handling for downloaded file")
        
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

    private func refreshMetadata() {
        guard let header = try? PackedGaussiansHeader(data: data) else {
            isAntialiased = false
            return
        }
        isAntialiased = (header.flags & PackedGaussiansHeader.FlagAntialiased) != 0
    }
    
    private func unpackGaussians(_ packedGaussians: PackedGaussians) -> [SplatScenePoint] {
        // Implementation based on the original C++ code with SIMD optimizations
        var results = [SplatScenePoint]()
        
        // The point count is bounded by the data actually decoded (deserialize trims it to the
        // bytes present), so every point is unpacked rather than truncating at a fixed limit.
        if packedGaussians.shDegree > 4 {
            spzLog.debug(" Unsupported SH degree: \(packedGaussians.shDegree), SPZ spec allows 0-4")
        }
        let safeNumPoints = packedGaussians.numPoints
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
        
        // Process chunks sequentially to satisfy strict-concurrency checks.
        var chunkResults = Array<[SplatScenePoint]?>(repeating: nil, count: chunks.count)

        for (chunkIndex, chunkRange) in chunks.enumerated() {
            chunkResults[chunkIndex] = self.processPointChunk(packedGaussians: packedGaussians,
                                                              range: chunkRange,
                                                              positionStride: positionStride,
                                                              rotationStride: rotationStride,
                                                              shDim: shDim,
                                                              shStride: shStride)

            if chunkRange.lowerBound > 0 && chunkRange.lowerBound % 100000 == 0 {
                spzLog.debug(" Unpacked \(chunkRange.lowerBound) points...")
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

        // Data tagged with the Adobe coordinate_system extension is converted to PLY axes (RDF),
        // which the other readers produce. Untagged data is kept as stored: gsbox, SplatKit and
        // earlier versions of this reader treat SPZ as PLY axes, although nianticlabs/spz
        // assumes RUB.
        let conversion = SPZCoordinateConversion(from: packedGaussians.coordinateSystem)
        let coordinateConverter = conversion.flips

        // Color scale factor from original C++ implementation
        let colorScale: Float = 0.15

        // Fixed-point scale for non-float16 positions
        let fixedPointScale = spzFixedPointScale(fractionalBits: packedGaussians.fractionalBits)

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
                    if shDim > 0 && shOffset + (shDim * 3) <= packedGaussians.sh.count {
                        for j in 0..<min(shDim, 24) { // Limit to 24 non-DC coefficients
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
                    var point = SplatScenePoint(
                        position: position,
                        color: .sphericalHarmonic(sphericalHarmonics),
                        opacity: .logitFloat(alpha),
                        scale: .exponent(scale),
                        rotation: rotation
                    )
                    conversion.rotate(&point)

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
        return data.count >= 2 && data[data.startIndex] == 0x1F && data[data.startIndex + 1] == 0x8B
    }

    /// Gunzips a legacy (v1–v3) SPZ file, never producing more than its header allows.
    /// A gzip header zlib can't parse (such as a compression method other than deflate) is
    /// retried as raw deflate after the fixed 10-byte gzip header, with the same limit.
    private static func decompressGzipped(_ data: Data) -> Data? {
        guard isGzipped(data) else {
            spzLog.debug(" Not a gzip file (wrong magic bytes)")
            return nil
        }

        let attempts: [(windowBits: Int32, input: Data)] = [
            (Data.gzipOrZlibWindowBits, data),
            (Data.rawDeflateWindowBits, data.count > 10 ? data.subdata(in: (data.startIndex + 10)..<data.endIndex) : Data()),
        ]
        for attempt in attempts where !attempt.input.isEmpty {
            guard let prefix = try? attempt.input.inflated(windowBits: attempt.windowBits,
                                                            maxOutputSize: PackedGaussiansHeader.size,
                                                            truncating: true) else { continue }
            let limit = legacyDecompressedSizeLimit(header: prefix, compressedSize: data.count)
            do {
                let decompressed = try attempt.input.inflated(windowBits: attempt.windowBits, maxOutputSize: limit)
                spzLog.debug(" Decompressed \(decompressed.count) bytes (limit \(limit))")
                return decompressed
            } catch {
                spzLog.debug(" Gzip decompression failed: \(error)")
            }
        }
        return nil
    }

    /// The first `count` decompressed bytes of a gzip stream.
    private static func gzipPrefix(_ data: Data, count: Int) -> Data? {
        try? data.inflated(windowBits: Data.gzipOrZlibWindowBits, maxOutputSize: count, truncating: true)
    }

    /// Upper bound on a legacy SPZ stream's decompressed size, from its header: the payload the
    /// header describes plus room for extension records. Like nianticlabs/spz, a header claiming
    /// more points than the compressed file could hold (positions take 9 bytes per point and
    /// deflate stays under 1032:1) gets no more than that. Data without an SPZ header keeps the
    /// general limit.
    static func legacyDecompressedSizeLimit(header: Data, compressedSize: Int) -> Int {
        guard let header = try? PackedGaussiansHeader(data: header), (1...3).contains(header.version) else {
            return Data.maxDecompressedSize
        }
        let maximumPoints = compressedSize.multipliedReportingOverflow(by: 1024).partialValue / 9
        let points = min(Int(header.numPoints), maximumPoints)
        let bytesPerPoint = (header.version == 1 ? 6 : 9) + 1 + 3 + 3 + (header.version >= 3 ? 4 : 3)
            + shDimForDegree(Int(header.shDegree)) * 3
        let extensionBytes = (header.flags & PackedGaussiansHeader.FlagHasExtensions) != 0 ? 16 * 1024 * 1024 : 0
        let (payload, overflow) = points.multipliedReportingOverflow(by: bytesPerPoint)
        guard !overflow else { return Data.maxDecompressedSize }
        return PackedGaussiansHeader.size + payload + extensionBytes
    }
}
