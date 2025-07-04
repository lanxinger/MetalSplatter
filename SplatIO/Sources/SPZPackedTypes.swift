import Foundation
import simd
#if canImport(Metal)
import Metal
#endif

/**
 * Represents the header structure for SPZ format files
 */
struct PackedGaussiansHeader {
    static let magic: UInt32 = 0x5053474e  // NGSP = Niantic gaussian splat
    static let version: UInt32 = 2
    
    var magic: UInt32 = PackedGaussiansHeader.magic
    var version: UInt32 = PackedGaussiansHeader.version
    var numPoints: UInt32 = 0
    var shDegree: UInt8 = 0
    var fractionalBits: UInt8 = 0
    var flags: UInt8 = 0
    var reserved: UInt8 = 0
    
    // Size of the header in bytes
    static let size = 16 // 4 + 4 + 4 + 1 + 1 + 1 + 1
    
    // Constants for flag bits (matching C++ implementation)
    static let FlagAntialiased: UInt8 = 0x01
    static let FlagUsesFloat16: UInt8 = 0x02
    
    // Flags bit meanings (matching C++ implementation):
    // bit 0: antialiased (whether gaussians should be rendered with mip-splat antialiasing)
    // bit 1: usesFloat16 (whether positions are stored as float16 or fixed-point)
    
    init() {}
    
    // Helper to calculate SH coefficient count based on degree (matching Niantic reference)
    var shCoeffCount: Int {
        return shDimForDegree(Int(shDegree))
    }
    
    init(data: Data) throws {
        guard data.count >= PackedGaussiansHeader.size else {
            throw SplatFileFormatError.invalidHeader
        }
        
        // Validate bounds for header data access
        try SplatDataValidator.validateDataBounds(data: data, offset: 0, size: PackedGaussiansHeader.size)
        
        // Safely load UInt32 values using copyBytes to avoid alignment issues
        _ = withUnsafeMutableBytes(of: &magic) { ptr in
            data[0..<4].copyBytes(to: ptr)
        }
        _ = withUnsafeMutableBytes(of: &version) { ptr in
            data[4..<8].copyBytes(to: ptr)
        }
        _ = withUnsafeMutableBytes(of: &numPoints) { ptr in
            data[8..<12].copyBytes(to: ptr)
        }
        
        // Validate magic number
        guard magic == PackedGaussiansHeader.magic else {
            throw SplatFileFormatError.invalidHeader
        }
        
        // Validate version
        guard version == PackedGaussiansHeader.version else {
            throw SplatFileFormatError.unsupportedVersion
        }
        
        shDegree = data[12]
        fractionalBits = data[13]
        flags = data[14]
        reserved = data[15]
        
        // Validate reasonable values
        guard shDegree <= 3 else {
            throw SplatFileFormatError.invalidData
        }
        
        guard fractionalBits <= 16 else {
            throw SplatFileFormatError.invalidData
        }
    }
    
    func serialize() -> Data {
        var data = Data(capacity: PackedGaussiansHeader.size)
        data.append(contentsOf: withUnsafeBytes(of: magic) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: version) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: numPoints) { Data($0) })
        data.append(shDegree)
        data.append(fractionalBits)
        data.append(flags)
        data.append(reserved)
        return data
    }
}

/**
 * Represents a single unpacked gaussian with full precision
 */
struct UnpackedGaussian {
    var position: SIMD3<Float>
    var rotation: simd_quatf
    var scale: SIMD3<Float>
    var color: SIMD3<Float>
    var alpha: Float
    var shR: [Float]
    var shG: [Float]
    var shB: [Float]
    
    init() {
        position = .zero
        rotation = simd_quatf()
        scale = .zero
        color = .zero
        alpha = 0
        shR = Array(repeating: 0, count: 15)
        shG = Array(repeating: 0, count: 15)
        shB = Array(repeating: 0, count: 15)
    }
}

/**
 * Represents a single packed gaussian with compressed representation
 */
struct PackedGaussian {
    var position: [UInt8]
    var rotation: [UInt8]
    var scale: [UInt8]
    var color: [UInt8]
    var alpha: UInt8
    var shR: [UInt8]
    var shG: [UInt8]
    var shB: [UInt8]
    
    init() {
        position = Array(repeating: 0, count: 9) // 3 positions × 3 bytes each for 24-bit fixed-point (or 6 bytes for float16)
        rotation = Array(repeating: 0, count: 3)
        scale = Array(repeating: 0, count: 3)
        color = Array(repeating: 0, count: 3)
        alpha = 0
        shR = Array(repeating: 0, count: 15)
        shG = Array(repeating: 0, count: 15)
        shB = Array(repeating: 0, count: 15)
    }
    
    func unpack(usesFloat16: Bool, fractionalBits: Int) -> UnpackedGaussian {
        var result = UnpackedGaussian()
        
        // Unpack position based on format
        if usesFloat16 {
            for i in 0..<3 {
                let idx = i * 2
                if idx + 1 < position.count {
                    let halfValue = UInt16(position[idx]) | (UInt16(position[idx + 1]) << 8)
                    result.position[i] = float16ToFloat32(halfValue)
                }
            }
        } else {
            let scale = 1.0 / Float(1 << fractionalBits)
            for i in 0..<3 {
                let idx = i * 3
                if idx + 2 < position.count {
                    var fixed32: Int32 = Int32(position[idx])
                    fixed32 |= Int32(position[idx + 1]) << 8
                    fixed32 |= Int32(position[idx + 2]) << 16
                    if (fixed32 & 0x800000) != 0 {
                        fixed32 |= Int32(bitPattern: 0xFF000000) // Sign extension
                    }
                    result.position[i] = Float(fixed32) * scale
                }
            }
        }
        
        // Unpack scale
        for i in 0..<3 {
            if i < scale.count {
                result.scale[i] = Float(scale[i]) / 16.0 - 10.0
            }
        }
        
        // Unpack rotation
        if rotation.count >= 3 {
            // Convert back from 8-bit to normalized values
            var xyz = SIMD3<Float>(
                Float(rotation[0]) / 127.5 - 1.0,
                Float(rotation[1]) / 127.5 - 1.0,
                Float(rotation[2]) / 127.5 - 1.0
            )
            
            // Normalize the vector to ensure it's valid
            let length = simd_length(xyz)
            if length > 0 {
                xyz /= length
            }
            
            // Reconstruct the w component
            let w = sqrt(max(0.0, 1.0 - simd_dot(xyz, xyz)))
            
            // Create quaternion
            result.rotation = simd_quatf(vector: SIMD4<Float>(xyz.x, xyz.y, xyz.z, w))
        }
        
        // Unpack alpha using sigmoid
        result.alpha = logit(Float(alpha) / 255.0)
        
        // Unpack color (matching Niantic reference with colorScale = 0.15)
        for i in 0..<3 {
            if i < color.count {
                result.color[i] = unquantizeColor(color[i])
            }
        }
        
        // Copy SH coefficients
        for i in 0..<min(shR.count, 15) {
            result.shR[i] = unquantizeSH(shR[i])
            result.shG[i] = unquantizeSH(shG[i])
            result.shB[i] = unquantizeSH(shB[i])
        }
        
        return result
    }
    
    // Helper function to convert quantized SH coefficient to float
    private func unquantizeSH(_ value: UInt8) -> Float {
        return (Float(value) - 128.0) / 128.0
    }
    
    // Convert from [0, 255] to [-1, 1] for rotation components
    private func unquantizeRotation(_ value: UInt8) -> Float {
        return Float(value) / 127.5 - 1.0
    }
    
    // Convert from [0, 255] to proper SH coefficient range
    // The scale factor matches the C++ implementation's normalization
    private func unquantizeColor(_ value: UInt8, colorScale: Float = 0.15) -> Float {
        return ((Float(value) / 255.0) - 0.5) / colorScale
    }
    
    // Convert from [0, 255] to scale value with the mapping used in the C++ version
    private func unquantizeScale(_ value: UInt8) -> Float {
        return Float(value) / 16.0 - 10.0 // Maps [0, 255] to [-10, 5.9375]
    }
    
    // Inverse sigmoid function (logit)
    private func logit(_ x: Float) -> Float {
        let clamped = max(0.0001, min(0.9999, x))
        return log(clamped / (1.0 - clamped))
    }
}

/**
 * Represents a full set of packed gaussians
 */
struct PackedGaussians {
    var numPoints: Int = 0
    var shDegree: Int = 0
    var fractionalBits: Int = 0
    var antialiased: Bool = false
    
    var positions: [UInt8] = []
    var scales: [UInt8] = []
    var rotations: [UInt8] = []
    var alphas: [UInt8] = []
    var colors: [UInt8] = []
    var sh: [UInt8] = []
    
    var usesFloat16: Bool {
        // C++ reference: version 1 uses float16, version 2+ uses fixed-point
        // For compatibility, also check data size as fallback
        return positions.count == numPoints * 3 * 2
    }
    
    // Helper to calculate SH coefficient count (matching Niantic reference)
    var shCoeffCount: Int {
        return shDimForDegree(shDegree)
    }
    
    func at(_ index: Int) -> PackedGaussian {
        var result = PackedGaussian()
        let positionBits = usesFloat16 ? 6 : 9
        let start3 = index * 3
        let posStart = index * positionBits
        
        // Verify index is in bounds for all arrays
        guard index >= 0 && index < numPoints else {
            print("PackedGaussians.at: Index \(index) out of bounds (numPoints: \(numPoints))")
            return result
        }
        
        // Additional safety checks for array bounds
        if start3 + 2 >= colors.count || start3 + 2 >= scales.count || 
           start3 + 2 >= rotations.count || index >= alphas.count {
            print("PackedGaussians.at: Array bounds issue for index \(index)")
            print("  Colors: \(colors.count), needed: \(start3 + 3)")
            print("  Scales: \(scales.count), needed: \(start3 + 3)")
            print("  Rotations: \(rotations.count), needed: \(start3 + 3)")
            print("  Alphas: \(alphas.count), needed: \(index + 1)")
            // Return empty result rather than crashing
            return result
        }
        
        // Copy position bytes
        let positionSize = usesFloat16 ? 6 : 9
        if posStart + positionSize <= positions.count {
            result.position = Array(positions[posStart..<posStart + positionSize])
        }
        
        // Copy scale bytes
        let scaleStart = start3
        if scaleStart + 3 <= scales.count {
            result.scale = Array(scales[scaleStart..<scaleStart + 3])
        }
        
        // Copy rotation bytes
        let rotStart = start3
        if rotStart + 3 <= rotations.count {
            result.rotation = Array(rotations[rotStart..<rotStart + 3])
        }
        
        // Copy alpha
        if index < alphas.count {
            result.alpha = alphas[index]
        }
        
        // Copy color bytes
        let colorStart = start3
        if colorStart + 3 <= colors.count {
            result.color = Array(colors[colorStart..<colorStart + 3])
        }
        
        // Calculate SH dimension based on degree (matching Niantic reference)
        let shDim = shDimForDegree(shDegree)
        let shStart = index * shDim * 3
        
        // Verify SH array bounds
        if shStart + (shDim * 3) > sh.count {
            // If we don't have all SH data, use what we have
            let availableDims = (sh.count - shStart) / 3
            print("PackedGaussians.at: SH data truncated. Using \(availableDims) of \(shDim) dimensions for point \(index)")
            
            // Copy what SH data we have
            for j in 0..<min(availableDims, shDim) {
                let idx = shStart + j * 3
                guard idx + 2 < sh.count else { break }
                result.shR[j] = sh[idx]
                result.shG[j] = sh[idx + 1]
                result.shB[j] = sh[idx + 2]
            }
            
            // Fill remaining with neutral values
            for j in min(availableDims, shDim)..<15 {
                result.shR[j] = 128
                result.shG[j] = 128
                result.shB[j] = 128
            }
            
            return result
        }
        
        // Copy SH data - matches C++ reference load-spz.cc:363-373
        // Data layout: color channel is inner axis, coefficient is outer axis
        for j in 0..<shDim {
            let idx = shStart + j * 3
            guard j < result.shR.count && (idx + 2) < sh.count else {
                break
            }
            result.shR[j] = sh[idx]
            result.shG[j] = sh[idx + 1] 
            result.shB[j] = sh[idx + 2]
        }
        
        // Fill remaining coefficients with neutral value (128 = 0 after unquantization)
        for j in shDim..<15 {
            result.shR[j] = 128
            result.shG[j] = 128
            result.shB[j] = 128
        }
        
        return result
    }
    
    func serialize() -> Data {
        var header = PackedGaussiansHeader()
        header.numPoints = UInt32(numPoints)
        header.shDegree = UInt8(shDegree)
        header.fractionalBits = UInt8(fractionalBits)
        // Set the antialiased flag if needed
        header.flags = (antialiased ? PackedGaussiansHeader.FlagAntialiased : 0x0)
        
        // Set the usesFloat16 flag if positions are stored in float16 format
        if positions.count == numPoints * 3 * 2 {
            header.flags |= PackedGaussiansHeader.FlagUsesFloat16
        }
        
        var data = header.serialize()
        
        // Append data in non-interleaved order
        data.append(contentsOf: positions)
        data.append(contentsOf: alphas)
        data.append(contentsOf: colors)
        data.append(contentsOf: scales)
        data.append(contentsOf: rotations)
        data.append(contentsOf: sh)
        
        return data
    }
    
    static func deserialize(_ data: Data) throws -> PackedGaussians {
        // Enable more detailed debug logging for troubleshooting
        let debug = true
        
        func debugPrint(_ message: String) {
            if debug {
                print("PackedGaussians.deserialize: \(message)")
            }
        }
        
        debugPrint("Data size: \(data.count) bytes")
        if data.count >= 32 {
            let hexString = data.prefix(32).map { String(format: "%02X", $0) }.joined(separator: " ")
            print("PackedGaussians.deserialize: First bytes: \(hexString)")
            
            // Check if this is a gzipped file
            if data.count >= 2 && data[0] == 0x1F && data[1] == 0x8B {
                print("PackedGaussians.deserialize: This is a gzipped file. Decompression should have been handled earlier.")
                print("PackedGaussians.deserialize: Trying to skip gzip header and find the SPZ magic number...")
                
                // Try to find the SPZ magic number (NGSP = 0x5053474E) in the first 2KB
                if data.count > 20 {
                    for offset in stride(from: 0, to: min(2048, data.count - 16), by: 1) {
                        guard offset + 4 <= data.count else { break }
                        
                        let magicBytes = data[offset..<(offset+4)]
                        // Safer way to load UInt32 from potentially unaligned data
                        var magic: UInt32 = 0
                        _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                            magicBytes.copyBytes(to: magicPtr)
                        }
                        
                        // Check for NGSP magic (0x5053474E in little endian)
                        if magic == 0x5053474E {
                            print("PackedGaussians.deserialize: Found SPZ magic at offset \(offset), trying to parse from there")
                            
                            // Create a new data object starting from the magic number
                            return try deserialize(data.subdata(in: offset..<data.count))
                        }
                    }
                }
            }
        }
        
        // Parse the header
        let header: PackedGaussiansHeader
        do {
            header = try PackedGaussiansHeader(data: data)
            print("PackedGaussians.deserialize: Header parsed successfully")
            print("PackedGaussians.deserialize: Magic: 0x\(String(format: "%08X", header.magic))")
            print("PackedGaussians.deserialize: Version: \(header.version)")
            print("PackedGaussians.deserialize: NumPoints: \(header.numPoints)")
            print("PackedGaussians.deserialize: SH Degree: \(header.shDegree)")
            print("PackedGaussians.deserialize: Fractional Bits: \(header.fractionalBits)")
            print("PackedGaussians.deserialize: Flags: 0x\(String(format: "%02X", header.flags))")
            
            // Check for correct magic number, but be lenient if it's a known variant
            if header.magic != PackedGaussiansHeader.magic {
                print("PackedGaussians.deserialize: Unexpected magic number: 0x\(String(format: "%08X", header.magic)) vs expected: 0x\(String(format: "%08X", PackedGaussiansHeader.magic))")
                
                // Check for known variants and alternate encodings
                let possibleMagics: [UInt32] = [
                    0x5053474E,  // NGSP (little endian)
                    0x4E475350,  // NGSP (big endian)
                    0x5350474E,  // SPGN variant
                    0x4E475053   // NGPS variant
                ]
                
                if possibleMagics.contains(header.magic) {
                    print("PackedGaussians.deserialize: Found an acceptable alternative magic number: 0x\(String(format: "%08X", header.magic))")
                    // Continue processing with this variant
                } else {
                    // The magic number is completely wrong, so throw an error
                    throw SplatFileFormatError.invalidHeader
                }
            }
        } catch {
            print("PackedGaussians.deserialize: Error parsing header: \(error)")
            throw error
        }
        
        // Extract header fields with C++ reference validation
        let numPoints = Int(header.numPoints)
        print("PackedGaussians.deserialize: Number of points: \(numPoints)")
        // C++ check: maxPointsToRead = 10000000
        if numPoints <= 0 || numPoints > 10000000 {
            print("PackedGaussians.deserialize: Invalid point count: \(numPoints), must be 1-10M")
            throw SplatFileFormatError.invalidData
        }
        
        let shDegree = Int(header.shDegree)
        print("PackedGaussians.deserialize: SH degree: \(shDegree)")
        if shDegree < 0 || shDegree > 3 { // SPZ spec: SH degree must be between 0 and 3 (inclusive)
            print("PackedGaussians.deserialize: Invalid SH degree: \(shDegree). SPZ spec requires degree 0-3.")
            throw SplatFileFormatError.invalidData
        }
        
        let shDim = shDimForDegree(shDegree)
        // C++ reference: version 1 uses float16, version 2+ uses fixed-point with flags
        let usesFloat16 = (header.version == 1) || (header.flags & PackedGaussiansHeader.FlagUsesFloat16) != 0
        print("PackedGaussians.deserialize: Uses Float16: \(usesFloat16) (version: \(header.version), flags: 0x\(String(format: "%02X", header.flags)))")
        
        // Calculate component sizes
        let positionBytes = usesFloat16 ? (numPoints * 3 * 2) : (numPoints * 3 * 3)
        let colorBytes = numPoints * 3
        let scaleBytes = numPoints * 3
        let rotationBytes = numPoints * 3
        let alphaBytes = numPoints
        let shBytes = numPoints * shDim * 3
        
        // Print component sizes for debugging
        print("PackedGaussians.deserialize: Position bytes: \(positionBytes)")
        print("PackedGaussians.deserialize: Color bytes: \(colorBytes)")
        print("PackedGaussians.deserialize: Scale bytes: \(scaleBytes)")
        print("PackedGaussians.deserialize: Rotation bytes: \(rotationBytes)")
        print("PackedGaussians.deserialize: Alpha bytes: \(alphaBytes)")
        print("PackedGaussians.deserialize: SH bytes: \(shBytes)")
        
        // Calculate offsets
        var offset = PackedGaussiansHeader.size
        let positionOffset = offset
        offset += positionBytes
        
        let alphaOffset = offset
        offset += alphaBytes
        
        let colorOffset = offset
        offset += colorBytes
        
        let scaleOffset = offset
        offset += scaleBytes
        
        let rotationOffset = offset
        offset += rotationBytes
        
        let shOffset = offset
        offset += shBytes
        
        let expectedSize = offset
        
        print("PackedGaussians.deserialize: Expected size: \(expectedSize), Actual data size: \(data.count)")
        
        // Be more lenient with the size check - as long as we have the header, try to extract what we can
        if data.count < PackedGaussiansHeader.size {
            print("PackedGaussians.deserialize: Data too small to contain header")
            throw SplatFileFormatError.invalidData
        }
        
        // For troubleshooting, dump the first bytes as possible magic values
        if data.count >= 16 {
            print("PackedGaussians.deserialize: Possible magic values:")
            for i in 0...12 {
                if i + 4 <= data.count {
                    let magicBytes = data[i..<(i+4)]
                    // Safer way to load UInt32 from potentially unaligned data
                    var magic: UInt32 = 0
                    _ = withUnsafeMutableBytes(of: &magic) { magicPtr in
                        magicBytes.copyBytes(to: magicPtr)
                    }
                    print("  Offset \(i): 0x\(String(format: "%08X", magic)) (\(String(bytes: magicBytes, encoding: .ascii) ?? "non-ASCII"))")
                }
            }
        }
        
        // Issue a warning but continue if there's not enough data for all points
        if data.count < expectedSize {
            print("PackedGaussians.deserialize: Warning - Data size mismatch. Expected \(expectedSize) bytes, got \(data.count)")
            // We'll proceed anyway and just take what we can get
        }
        
        var result = PackedGaussians()
        result.numPoints = numPoints
        result.shDegree = Int(header.shDegree)
        result.fractionalBits = Int(header.fractionalBits)
        result.antialiased = (header.flags & PackedGaussiansHeader.FlagAntialiased) != 0
        
        // Extract component data (safely handling truncated files)
        // For each component, check if we have enough data and adjust if needed
        let safePositionBytes = positionOffset < data.count ? min(positionBytes, data.count - positionOffset) : 0
        let safeAlphaBytes = alphaOffset < data.count ? min(alphaBytes, data.count - alphaOffset) : 0
        let safeColorBytes = colorOffset < data.count ? min(colorBytes, data.count - colorOffset) : 0
        let safeScaleBytes = scaleOffset < data.count ? min(scaleBytes, data.count - scaleOffset) : 0
        let safeRotationBytes = rotationOffset < data.count ? min(rotationBytes, data.count - rotationOffset) : 0
        let safeSHBytes = shOffset < data.count ? min(shBytes, data.count - shOffset) : 0
        
        print("PackedGaussians.deserialize: Safe data sizes after truncation check:")
        print("  Positions: \(safePositionBytes)/\(positionBytes) bytes")
        print("  Colors: \(safeColorBytes)/\(colorBytes) bytes")
        print("  Scales: \(safeScaleBytes)/\(scaleBytes) bytes")
        print("  Rotations: \(safeRotationBytes)/\(rotationBytes) bytes")
        print("  Alphas: \(safeAlphaBytes)/\(alphaBytes) bytes")
        print("  SH: \(safeSHBytes)/\(shBytes) bytes")
        
        // Only read the data we actually have, with bounds checking
        do {
            if safePositionBytes > 0 && positionOffset + safePositionBytes <= data.count {
                result.positions = Array(data[positionOffset..<(positionOffset + safePositionBytes)])
            } else {
                result.positions = []
                print("PackedGaussians.deserialize: Warning - Cannot read position data safely")
            }
            
            if safeAlphaBytes > 0 && alphaOffset + safeAlphaBytes <= data.count {
                result.alphas = Array(data[alphaOffset..<(alphaOffset + safeAlphaBytes)])
            } else {
                result.alphas = []
                print("PackedGaussians.deserialize: Warning - Cannot read alpha data safely")
            }
            
            if safeColorBytes > 0 && colorOffset + safeColorBytes <= data.count {
                result.colors = Array(data[colorOffset..<(colorOffset + safeColorBytes)])
            } else {
                result.colors = []
                print("PackedGaussians.deserialize: Warning - Cannot read color data safely")
            }
            
            if safeScaleBytes > 0 && scaleOffset + safeScaleBytes <= data.count {
                result.scales = Array(data[scaleOffset..<(scaleOffset + safeScaleBytes)])
            } else {
                result.scales = []
                print("PackedGaussians.deserialize: Warning - Cannot read scale data safely")
            }
            
            if safeRotationBytes > 0 && rotationOffset + safeRotationBytes <= data.count {
                result.rotations = Array(data[rotationOffset..<(rotationOffset + safeRotationBytes)])
            } else {
                result.rotations = []
                print("PackedGaussians.deserialize: Warning - Cannot read rotation data safely")
            }
            
            if safeSHBytes > 0 && shOffset + safeSHBytes <= data.count {
                result.sh = Array(data[shOffset..<(shOffset + safeSHBytes)])
            } else {
                result.sh = []
                print("PackedGaussians.deserialize: Warning - Cannot read SH data safely")
            }
        } catch {
            print("PackedGaussians.deserialize: Error extracting data: \(error)")
            throw SplatFileFormatError.invalidData
        }
        
        // Update point count based on what we actually read
        var constraints: [Int] = [numPoints]
        
        // Add constraints based on available data
        if safePositionBytes > 0 {
            constraints.append(safePositionBytes / (usesFloat16 ? 6 : 9))
        }
        if safeAlphaBytes > 0 {
            constraints.append(safeAlphaBytes)
        }
        if safeColorBytes > 0 {
            constraints.append(safeColorBytes / 3)
        }
        if safeScaleBytes > 0 {
            constraints.append(safeScaleBytes / 3)
        }
        if safeRotationBytes > 0 {
            constraints.append(safeRotationBytes / 3)
        }
        if safeSHBytes > 0 && shDim > 0 {
            constraints.append(safeSHBytes / (shDim * 3))
        }
        
        let actualPointCount = constraints.min() ?? 0
        
        print("PackedGaussians.deserialize: Adjusted point count from \(numPoints) to \(actualPointCount)")
        
        // Ensure we have a valid point count
        if actualPointCount <= 0 {
            print("PackedGaussians.deserialize: Error - Cannot determine valid point count")
            throw SplatFileFormatError.invalidData
        }
        
        result.numPoints = actualPointCount
        
        return result
    }
}

/**
 * Utility function to convert a 16-bit half-precision float to a 32-bit float.
 * Uses Metal for hardware-accelerated conversion when available,
 * with a software fallback implementation for all platforms.
 */
func float16ToFloat32(_ half: UInt16) -> Float {
    // Software implementation for cross-platform compatibility
    let sign = (half & 0x8000) != 0
    let exponent = Int((half & 0x7C00) >> 10)
    let mantissa = Int(half & 0x03FF)
    
    let signMul: Float = sign ? -1.0 : 1.0
    
    if exponent == 0 {
        // Zero or denormalized
        if mantissa == 0 {
            return 0.0 * signMul
        }
        
        // Denormalized
        return signMul * pow(2.0, -14.0) * (Float(mantissa) / 1024.0)
    }
    
    if exponent == 31 {
        // Infinity or NaN
        return mantissa != 0 ? Float.nan : Float.infinity * signMul
    }
    
    // Normalized
    return signMul * pow(2.0, Float(exponent - 15)) * (1.0 + Float(mantissa) / 1024.0)
}

// MARK: - Coordinate System Support (matching Niantic reference)

public enum CoordinateSystem: Int {
    case unspecified = 0
    case ldb = 1  // Left Down Back
    case rdb = 2  // Right Down Back
    case lub = 3  // Left Up Back
    case rub = 4  // Right Up Back, Three.js coordinate system
    case ldf = 5  // Left Down Front
    case rdf = 6  // Right Down Front, PLY coordinate system
    case luf = 7  // Left Up Front, GLB coordinate system
    case ruf = 8  // Right Up Front, Unity coordinate system
}

public struct CoordinateConverter {
    let flipP: SIMD3<Float>  // x, y, z flips for positions
    let flipQ: SIMD3<Float>  // x, y, z flips for quaternions (w is never flipped)
    let flipSh: [Float]      // Flips for the 15 spherical harmonics coefficients
    
    public static func converter(from: CoordinateSystem, to: CoordinateSystem) -> CoordinateConverter {
        let (xMatch, yMatch, zMatch) = axesMatch(from, to)
        let x: Float = xMatch ? 1.0 : -1.0
        let y: Float = yMatch ? 1.0 : -1.0
        let z: Float = zMatch ? 1.0 : -1.0
        
        return CoordinateConverter(
            flipP: SIMD3<Float>(x, y, z),
            flipQ: SIMD3<Float>(y * z, x * z, x * y),
            flipSh: [
                y,          // 0
                z,          // 1
                x,          // 2
                x * y,      // 3
                y * z,      // 4
                1.0,        // 5
                x * z,      // 6
                1.0,        // 7
                y,          // 8
                x * y * z,  // 9
                y,          // 10
                z,          // 11
                x,          // 12
                z,          // 13
                x,          // 14
            ]
        )
    }
    
    private static func axesMatch(_ a: CoordinateSystem, _ b: CoordinateSystem) -> (Bool, Bool, Bool) {
        let aNum = a.rawValue - 1
        let bNum = b.rawValue - 1
        
        if aNum < 0 || bNum < 0 {
            return (true, true, true)
        }
        
        return (
            ((aNum >> 0) & 1) == ((bNum >> 0) & 1),
            ((aNum >> 1) & 1) == ((bNum >> 1) & 1),
            ((aNum >> 2) & 1) == ((bNum >> 2) & 1)
        )
    }
}

// MARK: - Spherical Harmonics Utilities (matching Niantic reference)

/// Calculate SH coefficient count for a given degree (matching C++ dimForDegree)
func shDimForDegree(_ degree: Int) -> Int {
    switch degree {
    case 0: return 0
    case 1: return 3
    case 2: return 8
    case 3: return 15
    default:
        print("Warning: Unsupported SH degree: \(degree)")
        return 0
    }
}

/// Calculate SH degree for a given coefficient count
func shDegreeForDim(_ dim: Int) -> Int {
    if dim < 3 { return 0 }
    if dim < 8 { return 1 }
    if dim < 15 { return 2 }
    return 3
}
