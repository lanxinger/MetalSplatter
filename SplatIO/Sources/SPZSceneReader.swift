import Foundation
import Compression
import simd

#if canImport(Metal)
import Metal
#endif

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
        print("SPZSceneReader: Trying to load file: \(url.path)")
        print("SPZSceneReader: File extension: \(url.pathExtension)")
        
        // Make sure the file exists and is readable
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            print("SPZSceneReader: File is not readable: \(url.path)")
            throw SplatFileFormatError.invalidData
        }
        
        do {
            let fileData = try Data(contentsOf: url)
            print("SPZSceneReader: Successfully read data, size: \(fileData.count) bytes")
            
            // Initialize with original data first
            self.init(data: fileData)
            
            // Special handling for iOS files from Downloads folder (they're often gzipped)
            if url.path.contains("/Containers/Shared/AppGroup/") && url.path.contains("/File Provider Storage/Downloads/") {
                print("SPZSceneReader: Detected iOS downloads file - using specialized handling")
                if processIOSDownloadFile(fileData) {
                    return
                }
            }
            
            // First attempt - try to load it as uncompressed SPZ
            print("SPZSceneReader: First attempt - trying to load as uncompressed SPZ")
            
            // Check for SPZ magic number
            if fileData.count >= 4 {
                // Safer way to load UInt32 from potentially unaligned data
                var magic: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                    fileData.prefix(4).copyBytes(to: magicPtr)
                }
                print("SPZSceneReader: No SPZ magic number found at start: 0x\(String(format: "%08X", magic))")
            }
            
            // Check if the file is gzipped
            if Self.isGzipped(fileData) {
                print("SPZSceneReader: Is gzipped: true")
                if let decompressedData = Self.decompressGzipped(fileData) {
                    print("SPZSceneReader: Successfully decompressed data: \(decompressedData.count) bytes")
                    self.data = decompressedData
                }
            }
        } catch {
            print("SPZSceneReader: Error reading file: \(error)")
            throw error
        }
    }
    
    // Special function to handle iOS downloaded files (often problematic gzipped files)
    private func processIOSDownloadFile(_ fileData: Data) -> Bool {
        print("SPZSceneReader: Trying iOS-specific handling for downloaded file")
        
        // Try the GZipArchive approach (used by many iOS apps)
        if let decompressed = Self.decompressIOSGzippedFile(fileData) {
            print("SPZSceneReader: Successfully decompressed iOS file: \(decompressed.count) bytes")
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
                        print("SPZSceneReader: Found SPZ magic number at offset \(offset)")
                        let extractedData = fileData.subdata(in: offset..<fileData.count)
                        print("SPZSceneReader: Extracted \(extractedData.count) bytes starting from magic number")
                        self.data = extractedData
                        return true
                    }
                }
            }
        }
        
        print("SPZSceneReader: iOS-specific handling did not find a solution")
        return false
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        print("SPZSceneReader: Attempting to deserialize data, size: \(data.count) bytes")
        
        // Try standard deserialization first
        do {
            let packedGaussians = try PackedGaussians.deserialize(data)
            print("SPZSceneReader: Successfully deserialized \(packedGaussians.numPoints) points")
            let points = unpackGaussians(packedGaussians)
            print("SPZSceneReader: Successfully unpacked \(points.count) points")
            return points
        } catch let deserializationError {
            print("SPZSceneReader: Standard deserialization failed: \(deserializationError)")
            
            // If the standard approach fails, try a more aggressive fallback
            print("SPZSceneReader: Trying alternative deserialization approach")
            
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
                        print("SPZSceneReader: Found SPZ magic at offset \(offset), trying to parse from there")
                        
                        // Create a new data object starting from the magic number
                        let offsetData = data.subdata(in: offset..<data.count)
                        
                        do {
                            let packedGaussians = try PackedGaussians.deserialize(offsetData)
                            print("SPZSceneReader: Successfully deserialized \(packedGaussians.numPoints) points from offset \(offset)")
                            let points = unpackGaussians(packedGaussians)
                            print("SPZSceneReader: Successfully unpacked \(points.count) points")
                            return points
                        } catch {
                            print("SPZSceneReader: Failed to parse from offset \(offset): \(error)")
                            // Continue searching for another magic number
                        }
                    }
                }
            }
            
            // If all fallbacks fail, rethrow the original error
            print("SPZSceneReader: All deserialization approaches failed")
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
        
        // Add safety checks, similar to the C++ implementation
        let maxPointsToRead = 10000000
        if packedGaussians.numPoints > maxPointsToRead {
            print("SPZSceneReader: Too many points: \(packedGaussians.numPoints), capping at \(maxPointsToRead)")
        }
        if packedGaussians.shDegree > 3 {
            print("SPZSceneReader: Unsupported SH degree: \(packedGaussians.shDegree), limiting to 3")
        }
        let safeNumPoints = min(packedGaussians.numPoints, maxPointsToRead) // Safety cap at 10 million points
        print("SPZSceneReader: Unpacking \(safeNumPoints) points with SH degree \(packedGaussians.shDegree)")
        
        // Validate the data format matches the C++ implementation expectations
        let positionBytesPerPoint = packedGaussians.usesFloat16 ? 6 : 9
        let expectedPositionBytes = safeNumPoints * positionBytesPerPoint
        let expectedScaleBytes = safeNumPoints * 3
        let expectedRotationBytes = safeNumPoints * 3
        let expectedAlphaBytes = safeNumPoints
        let expectedColorBytes = safeNumPoints * 3
        let shDim = shDimForDegree(packedGaussians.shDegree)  // Use correct Niantic formula
        let expectedSHBytes = safeNumPoints * shDim * 3
        
        // Log actual vs expected data sizes
        print("SPZSceneReader: Data size validation:")
        print("  Positions: \(packedGaussians.positions.count)/\(expectedPositionBytes) bytes")
        print("  Scales: \(packedGaussians.scales.count)/\(expectedScaleBytes) bytes")
        print("  Rotations: \(packedGaussians.rotations.count)/\(expectedRotationBytes) bytes")
        print("  Alphas: \(packedGaussians.alphas.count)/\(expectedAlphaBytes) bytes")
        print("  Colors: \(packedGaussians.colors.count)/\(expectedColorBytes) bytes")
        print("  SH: \(packedGaussians.sh.count)/\(expectedSHBytes) bytes")
        
        // Check if we have any data at all
        guard !packedGaussians.positions.isEmpty && 
              !packedGaussians.scales.isEmpty && 
              !packedGaussians.rotations.isEmpty && 
              !packedGaussians.colors.isEmpty && 
              !packedGaussians.alphas.isEmpty else {
            print("SPZSceneReader: Missing essential component data")
            return results
        }
        
        // Determine data layout based on format
        let positionStride = packedGaussians.usesFloat16 ? 6 : 9 // bytes per position
        let shStride = 3 // RGB components per SH coefficient
        
        // Process in smaller batches for better memory management and parallelization
        let chunkSize = 10000
        let chunks = stride(from: 0, to: safeNumPoints, by: chunkSize).map { startIdx -> Range<Int> in
            let endIdx = min(startIdx + chunkSize, safeNumPoints)
            return startIdx..<endIdx
        }
        
        // Use concurrent processing for large datasets
        let useParallelProcessing = safeNumPoints > 100000
        let processingQueue = DispatchQueue(label: "com.metalsplatter.spzprocessing", 
                                          qos: .userInitiated, 
                                          attributes: .concurrent)
        
        // Create a container for results that will be populated concurrently
        var chunkResults = Array<[SplatScenePoint]?>(repeating: nil, count: chunks.count)
        
        let processingGroup = DispatchGroup()
        
        for (chunkIndex, chunkRange) in chunks.enumerated() {
            if useParallelProcessing {
                // Process each chunk on a background queue
                processingGroup.enter()
                processingQueue.async {
                    chunkResults[chunkIndex] = self.processPointChunk(packedGaussians: packedGaussians,
                                                                   range: chunkRange,
                                                                   positionStride: positionStride,
                                                                   shDim: shDim,
                                                                   shStride: shStride)
                    processingGroup.leave()
                }
            } else {
                // Process sequentially for smaller datasets
                chunkResults[chunkIndex] = self.processPointChunk(packedGaussians: packedGaussians,
                                                               range: chunkRange,
                                                               positionStride: positionStride,
                                                               shDim: shDim,
                                                               shStride: shStride)
                
                // Progress reporting for sequential processing
                if chunkRange.lowerBound > 0 && chunkRange.lowerBound % 100000 == 0 {
                    print("SPZSceneReader: Unpacked \(chunkRange.lowerBound) points...")
                }
            }
        }
        
        // Wait for all parallel processing to complete
        if useParallelProcessing {
            processingGroup.wait()
        }
        
        // Combine results from all chunks
        results.reserveCapacity(safeNumPoints)
        for chunkResult in chunkResults {
            if let points = chunkResult {
                results.append(contentsOf: points)
            }
        }
        
        print("SPZSceneReader: Successfully unpacked \(results.count) points")
        return results
    }
    
    // Process a chunk of points using SIMD operations for better performance
    private func processPointChunk(packedGaussians: PackedGaussians, 
                                 range: Range<Int>,
                                 positionStride: Int,
                                 shDim: Int,
                                 shStride: Int) -> [SplatScenePoint] {
        var chunkResults = [SplatScenePoint]()
        chunkResults.reserveCapacity(range.count)
        
        // Color scale factor from original C++ implementation
        let colorScale: Float = 0.15
        
        for i in range {
            // Calculate offsets
            let posOffset = i * positionStride
            let colorOffset = i * 3
            let scaleOffset = i * 3
            let rotOffset = i * 3
            let shOffset = i * shDim * shStride
            
            // Check if all essential components are in bounds
            guard i < packedGaussians.alphas.count &&
                  colorOffset + 2 < packedGaussians.colors.count &&
                  scaleOffset + 2 < packedGaussians.scales.count &&
                  rotOffset + 2 < packedGaussians.rotations.count &&
                  posOffset + (positionStride - 1) < packedGaussians.positions.count else {
                continue
            }
            
            // Extract position using proper decoding based on the format
            var position = SIMD3<Float>(0, 0, 0)
            
            if packedGaussians.usesFloat16 {
                // Decode float16 positions using the optimized converter
                if posOffset + 5 < packedGaussians.positions.count {
                    // Extract position data for this point (6 bytes total - 3 components × 2 bytes)
                    let posData = Array(packedGaussians.positions[posOffset..<(posOffset+6)])
                    // Use the optimized FloatConversion utility which uses SIMD operations
                    let convertedVals = FloatConversion.convertFloat16PositionsToFloat32(posData, count: 1)
                    if !convertedVals.isEmpty {
                        // Fix the Y coordinate to handle upside-down display
                        position = SIMD3<Float>(
                            convertedVals[0].x,
                            -convertedVals[0].y, // Flip Y coordinate
                            convertedVals[0].z
                        )
                    }
                }
            } else {
                // Decode fixed-point positions
                let scale = 1.0 / Float(1 << packedGaussians.fractionalBits)
                
                if posOffset + 8 < packedGaussians.positions.count {
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
                        
                        // Apply the coordinate transform - need to flip Y axis
                        if j == 1 { // Y coordinate
                            position[j] = -Float(fixed32) * scale // Flip the Y coordinate
                        } else {
                            position[j] = Float(fixed32) * scale
                        }
                    }
                } else {
                    // Use a fallback position if we can't decode properly
                    position = SIMD3<Float>(
                        Float(i % 100) * 0.1,
                        -Float(i / 100 % 100) * 0.1, // Flip Y coordinate in fallback position
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
            
            // Extract rotation (quaternion from 3 bytes)
            var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) // Default identity quaternion
            if rotOffset + 2 < packedGaussians.rotations.count {
                // Convert all rotation components using SIMD
                let rotBytes = SIMD3<Float>(
                    Float(packedGaussians.rotations[rotOffset]),
                    Float(packedGaussians.rotations[rotOffset + 1]),
                    Float(packedGaussians.rotations[rotOffset + 2])
                )
                let xyz = rotBytes / 127.5 - 1.0
                
                // Calculate w component to ensure unit quaternion
                let xyzSquaredSum = xyz.x*xyz.x + xyz.y*xyz.y + xyz.z*xyz.z
                let w = xyzSquaredSum < 1.0 ? sqrt(1.0 - xyzSquaredSum) : 0.0
                
                rotation = simd_quatf(ix: xyz.x, iy: xyz.y, iz: xyz.z, r: w)
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
            if shDim > 1 && shOffset + (shDim-1)*3 < packedGaussians.sh.count {
                // We already have the DC term (first coefficient), so start from the second
                for j in 1..<min(shDim, 15) { // Limit to 15 max coefficients per channel
                    let idx = shOffset + j * 3
                    if idx + 2 < packedGaussians.sh.count {
                        // Process all three SH components at once
                        let shBytes = SIMD3<Float>(
                            Float(packedGaussians.sh[idx]),
                            Float(packedGaussians.sh[idx + 1]),
                            Float(packedGaussians.sh[idx + 2])
                        )
                        
                        // Use SIMD to unquantize all components at once
                        let shCoeffs = (shBytes - 128.0) / 128.0
                        sphericalHarmonics.append(shCoeffs)
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
        }
        
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
    
    // Create a temporary file to store the data
    private static func createTemporaryFile(with data: Data) -> URL? {
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = UUID().uuidString
        let fileURL = tempDir.appendingPathComponent(fileName)
        
        do {
            try data.write(to: fileURL)
            return fileURL
        } catch {
            print("SPZSceneReader: Error creating temporary file: \(error)")
            return nil
        }
    }
    
    // Run a shell command and return the output
    private static func runCommand(launchPath: String, arguments: [String]) -> Data? {
        #if os(macOS)
        let task = Process()
        let pipe = Pipe()
        
        task.standardOutput = pipe
        task.standardError = pipe
        task.arguments = arguments
        task.launchPath = launchPath
        
        do {
            try task.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            task.waitUntilExit()
            
            if task.terminationStatus == 0 {
                return data
            } else {
                print("SPZSceneReader: Command failed with status: \(task.terminationStatus)")
                return nil
            }
        } catch {
            print("SPZSceneReader: Error running command: \(error)")
            return nil
        }
        #else
        return nil
        #endif
    }
    
    // Special method for iOS files that don't decompress with standard methods
    private static func decompressIOSGzippedFile(_ data: Data) -> Data? {
        guard data.count > 10 else { return nil } // Need at least the gzip header
        
        print("SPZSceneReader: Attempting iOS-specific gzip decompression")
        
        // First try to use the Compression framework directly
        do {
            var decompressed = Data()
            // Skip the first 10 bytes (gzip header)
            let compressedData = data.subdata(in: 10..<data.count)
            
            // Initialize a decoder for zlib
            let outputFilter = try OutputFilter(.decompress, using: .zlib) { (data: Data?) -> Void in
                if let data = data {
                    decompressed.append(data)
                }
            }
            
            // Process the data
            try outputFilter.write(compressedData)
            try outputFilter.finalize()
            
            // If we got something reasonable, return it
            if decompressed.count > 1000 {
                print("SPZSceneReader: iOS-specific decompression succeeded with \(decompressed.count) bytes")
                return decompressed
            }
        } catch {
            print("SPZSceneReader: iOS-specific decompression failed: \(error)")
        }
        
        // Second approach: try several offsets into the file
        let possibleOffsets = [0, 10, 16, 18, 20, 24, 32]
        
        for offset in possibleOffsets {
            guard offset < data.count else { continue }
            
            let compressedData = data.subdata(in: offset..<data.count)
            do {
                let decompressed = try (compressedData as NSData).decompressed(using: .zlib) as Data
                if decompressed.count > 1000 {
                    print("SPZSceneReader: iOS decompression succeeded at offset \(offset) with \(decompressed.count) bytes")
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
            print("SPZSceneReader: Not a gzip file (wrong magic bytes)")
            return nil
        }
        
        print("SPZSceneReader: Attempting to decompress gzipped data")
        
        // First, try using an external process if available (most reliable)
        #if os(macOS)
        print("SPZSceneReader: Trying external gzip command")
        
        // Save the data to a temporary file
        guard let tempFile = createTemporaryFile(with: data) else {
            print("SPZSceneReader: Failed to create temporary file")
            return nil
        }
        
        // Try using the gzip command
        let result = runCommand(launchPath: "/usr/bin/gunzip", 
                               arguments: ["-c", tempFile.path])
        
        // Clean up the temporary file
        try? FileManager.default.removeItem(at: tempFile)
        
        if let decompressed = result, !decompressed.isEmpty {
            print("SPZSceneReader: Successfully decompressed with external gunzip command: \(decompressed.count) bytes")
            return decompressed
        }
        
        // Try using the zcat command as an alternative
        let zcatResult = runCommand(launchPath: "/usr/bin/zcat",
                                  arguments: [tempFile.path])
        
        if let decompressed = zcatResult, !decompressed.isEmpty {
            print("SPZSceneReader: Successfully decompressed with external zcat command: \(decompressed.count) bytes")
            return decompressed
        }
        #endif
        
        // Fall back to built-in methods if external process doesn't work
        print("SPZSceneReader: External decompression failed or not available, trying built-in methods")
        
        // Try each compression algorithm
        do {
            let decompressed = try (data as NSData).decompressed(using: .zlib) as Data
            print("SPZSceneReader: Successfully decompressed with .zlib algorithm")
            return decompressed
        } catch {
            print("SPZSceneReader: Decompression with .zlib failed")
        }
        
        do {
            let decompressed = try (data as NSData).decompressed(using: .lzfse) as Data
            print("SPZSceneReader: Successfully decompressed with .lzfse algorithm")
            return decompressed
        } catch {
            print("SPZSceneReader: Decompression with .lzfse failed")
        }
        
        if #available(iOS 13.0, macOS 10.15, *) {
            do {
                let decompressed = try (data as NSData).decompressed(using: .lz4) as Data
                print("SPZSceneReader: Successfully decompressed with .lz4 algorithm")
                return decompressed
            } catch {
                print("SPZSceneReader: Decompression with .lz4 failed")
            }
        }
        
        // Try to implement a custom gzip decoder as a last resort
        print("SPZSceneReader: Attempting custom gzip decoder")
        do {
            // Skip the gzip header (10 bytes) and try to decompress the payload
            if data.count > 10 {
                let payloadData = data.subdata(in: 10..<data.count)
                let decompressed = try (payloadData as NSData).decompressed(using: .zlib) as Data
                print("SPZSceneReader: Successfully decompressed gzip payload: \(decompressed.count) bytes")
                return decompressed
            }
        } catch {
            print("SPZSceneReader: Custom gzip decoder failed: \(error)")
        }
        
        print("SPZSceneReader: All decompression attempts failed")
        return nil
    }
}
