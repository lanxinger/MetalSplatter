import Foundation
import Compression
import simd

/**
 * Reader for SPX format Gaussian splat scenes.
 * SPX is a flexible, extensible format for 3D Gaussian Splatting models with:
 * - 128-byte header with metadata and bounding box
 * - Variable data blocks with different format types
 * - Support for gzip compression
 * - Spherical harmonics support
 */
public class SPXSceneReader: SplatSceneReader {
    private var data: Data
    private var header: SPXHeader?
    private var dataBlocks: [SPXDataBlock] = []
    
    public init(data: Data) throws {
        self.data = data
        try parseFile()
    }
    
    public convenience init(contentsOf url: URL) throws {
        print("SPXSceneReader: Loading file: \(url.path)")
        
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            print("SPXSceneReader: File is not readable: \(url.path)")
            throw SPXFileFormatError.invalidDataBlock
        }
        
        do {
            let fileData = try Data(contentsOf: url)
            print("SPXSceneReader: Read \(fileData.count) bytes")
            
            // Check if the file is gzipped and decompress if needed
            let processedData: Data
            if Self.isGzipped(fileData) {
                print("SPXSceneReader: File is gzipped, decompressing...")
                guard let decompressed = Self.decompressGzipped(fileData) else {
                    throw SPXFileFormatError.decompressionFailed
                }
                processedData = decompressed
                print("SPXSceneReader: Decompressed to \(processedData.count) bytes")
            } else {
                processedData = fileData
            }
            
            try self.init(data: processedData)
        } catch {
            print("SPXSceneReader: Error reading file: \(error)")
            throw error
        }
    }
    
    private func parseFile() throws {
        print("SPXSceneReader: Parsing SPX file...")
        
        // Parse header
        guard data.count >= SPXHeader.headerSize else {
            throw SPXFileFormatError.invalidHeader
        }
        
        header = try SPXHeader(data: data)
        print("SPXSceneReader: Header parsed - Version: \(header!.version), Total points: \(header!.splatCount), SH degree: \(header!.shDegree)")
        if let h = header {
            print("SPXSceneReader.parseFile: BBox immediately after parse - minX: \(h.minX), maxX: \(h.maxX)")
        }
        
        // Parse data blocks
        var offset = SPXHeader.headerSize
        while offset < data.count {
            guard let block = try parseDataBlock(at: &offset) else {
                break
            }
            dataBlocks.append(block)
        }
        
        print("SPXSceneReader: Parsed \(dataBlocks.count) data blocks")
    }
    
    private func parseDataBlock(at offset: inout Int) throws -> SPXDataBlock? {
        // Need at least 4 bytes for block length
        guard offset + 4 <= data.count else {
            return nil
        }
        
        // Parse block length first, ensuring correct endianness
        var blockLengthLE: Int32 = 0
        _ = withUnsafeMutableBytes(of: &blockLengthLE) {
            data[offset..<(offset+4)].copyBytes(to: $0)
        }
        let blockLength = Int32(littleEndian: blockLengthLE)
        offset += 4
        
        // Determine if compressed and actual block size
        let isCompressed = blockLength < 0
        let actualBlockSize = isCompressed ? Int(-blockLength) : Int(blockLength)
        
        print("SPXSceneReader: Raw block length: \(blockLength), actual size: \(actualBlockSize), compressed: \(isCompressed)")
        
        // Read the entire block data
        guard offset + actualBlockSize <= data.count else {
            throw SPXFileFormatError.insufficientData
        }
        
        var blockBytes = data.subdata(in: offset..<(offset+actualBlockSize))
        offset += actualBlockSize
        
        // Decompress if needed BEFORE parsing the block content
        if isCompressed {
            guard let decompressed = Self.decompressData(Data(blockBytes)) else {
                throw SPXFileFormatError.decompressionFailed
            }
            blockBytes = decompressed
        }
        
        // Now parse the decompressed block content header
        guard blockBytes.count >= 8 else {
            throw SPXFileFormatError.insufficientData
        }
        
        var gaussianCountLE: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &gaussianCountLE) {
            blockBytes[0..<4].copyBytes(to: $0)
        }
        let gaussianCount = UInt32(littleEndian: gaussianCountLE)
        
        var formatID_LE: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &formatID_LE) {
            blockBytes[4..<8].copyBytes(to: $0)
        }
        let formatID = UInt32(littleEndian: formatID_LE)
        
        // Extract the actual data (everything after the 8-byte header)
        let blockData = blockBytes.subdata(in: 8..<blockBytes.count)
        
        if gaussianCount > 1000 { // Only log for large blocks
            print("SPXSceneReader: Block - Gaussians: \(gaussianCount), Format: \(formatID), Data: \(blockData.count) bytes")
        }
        
        return SPXDataBlock(
            blockLength: blockLength,
            gaussianCount: gaussianCount,
            formatID: formatID,
            data: Data(blockData)
        )
    }
    
    // MARK: - SplatSceneReader Implementation
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        guard let header = header else {
            print("SPXSceneReader: No header parsed")
            return
        }
        
        print("SPXSceneReader: Starting read with delegate")
        delegate.didStartReading(withPointCount: UInt32(header.splatCount))
        
        var allPoints: [SplatScenePoint] = []
        // This will be populated with all non-DC SH coefficients.
        var shCoefficients: [[SIMD3<Float>]] = []
        
        // Track the next index to write to for the two groups of SH coefficients.
        var nextSHIndex = (sh1_2: 0, sh3: 0)
        
        for (blockIdx, block) in dataBlocks.enumerated() {
            do {
                if block.formatID != 20 { // Log for non-basic blocks
                    print("SPXSceneReader: Processing block \(blockIdx + 1)/\(dataBlocks.count), format: \(block.formatID)")
                }
                
                switch block.formatID {
                case 0, 20:
                    // Basic data block - creates new points
                    let newPoints = try parseFormat20Points(block.data, count: Int(block.gaussianCount))
                    allPoints.append(contentsOf: newPoints)
                    // Ensure the SH coefficients array has space for the new points.
                    shCoefficients.append(contentsOf: Array(repeating: [], count: newPoints.count))
                    if newPoints.count > 1000 {
                        print("SPXSceneReader: Added \(newPoints.count) basic points, total: \(allPoints.count)")
                    }
                    
                case 1, 2:
                    try applySHData(block.data,
                                    count: Int(block.gaussianCount),
                                    to: &shCoefficients,
                                    shOrder: Int(block.formatID),
                                    startIndex: nextSHIndex.sh1_2)
                    nextSHIndex.sh1_2 += Int(block.gaussianCount)
                    
                case 3:
                    try applySHData(block.data,
                                    count: Int(block.gaussianCount),
                                    to: &shCoefficients,
                                    shOrder: Int(block.formatID),
                                    startIndex: nextSHIndex.sh3)
                    nextSHIndex.sh3 += Int(block.gaussianCount)
                    
                default:
                    print("SPXSceneReader: Unknown format \(block.formatID), skipping block")
                }
                
            } catch {
                print("SPXSceneReader: Error parsing block \(blockIdx): \(error)")
                delegate.didFailReading(withError: error)
                return
            }
        }
        
        // After parsing all blocks, combine the base color (DC term) with the other SH coefficients.
        for i in 0..<allPoints.count {
            if !shCoefficients[i].isEmpty {
                // The first SH coefficient is the DC term (the base color).
                var finalCoeffs = allPoints[i].color.asSphericalHarmonic
                finalCoeffs.append(contentsOf: shCoefficients[i])
                allPoints[i].color = .sphericalHarmonic(finalCoeffs)
            }
        }

        // Send all points in chunks
        let chunkSize = 10000
        for chunk in allPoints.chunked(into: chunkSize) {
            delegate.didRead(points: chunk)
        }
        
        print("SPXSceneReader: Read complete - \(allPoints.count) points")
        delegate.didFinishReading()
    }
    
    private func parseDataBlockPoints(_ block: SPXDataBlock) throws -> [SplatScenePoint] {
        var points: [SplatScenePoint] = []
        let gaussianCount = Int(block.gaussianCount)
        
        print("SPXSceneReader: Parsing block with format ID: \(block.formatID), gaussians: \(gaussianCount), data size: \(block.data.count)")
        
        switch block.formatID {
        case 0:
            // Format 0: Treat as basic data (common variant)
            print("SPXSceneReader: Treating format 0 as basic data format")
            try points = parseFormat20(block.data, count: gaussianCount)
            
        case 20:
            // Format 20: Basic data
            try points = parseFormat20(block.data, count: gaussianCount)
            
        case 1, 2, 3:
            // Format 1-3: Spherical harmonics
            try points = parseFormatSH(block.data, count: gaussianCount, formatID: UInt8(block.formatID))
            
        default:
            print("SPXSceneReader: Unsupported format ID: \(block.formatID)")
            // For unknown formats, try to parse as basic data as fallback
            print("SPXSceneReader: Attempting to parse unknown format as basic data")
            try points = parseFormat20(block.data, count: gaussianCount)
        }
        
        return points
    }
    
    private func parseFormat20(_ data: Data, count: Int) throws -> [SplatScenePoint] {
        // SPX Format 20 expected size: count * 20 bytes total
        // (3+3+3) bytes for positions + 3 bytes for scales + 4 bytes for colors + 4 bytes for rotations
        let expectedSize = count * 20
        print("SPXSceneReader: Format 20 parsing - Count: \(count), Expected size: \(expectedSize), Actual size: \(data.count), Bytes per gaussian: 20")
        
        guard data.count >= expectedSize else {
            print("SPXSceneReader: Insufficient data for format 20. Need \(expectedSize) bytes, have \(data.count) bytes")
            
            // Try to parse as many points as we can with available data
            let availablePoints = data.count / 20
            print("SPXSceneReader: Attempting to parse \(availablePoints) points instead of \(count)")
            
            if availablePoints == 0 {
                throw SPXFileFormatError.insufficientData
            }
            
            return try parseFormat20Points(data, count: availablePoints)
        }
        
        return try parseFormat20Points(data, count: count)
    }
    
    private func parseFormat20Points(_ data: Data, count: Int) throws -> [SplatScenePoint] {
        var newPoints: [SplatScenePoint] = []
        newPoints.reserveCapacity(count)

        guard let header = header else {
            throw SPXFileFormatError.invalidHeader
        }

        // Bounding box for coordinate transformation
        let boundsMin = SIMD3<Float>(header.minX, header.minY, header.minZ)
        let boundsMax = SIMD3<Float>(header.maxX, header.maxY, header.maxZ)
        let boundsSize = boundsMax - boundsMin
        let boundsCenter = (boundsMin + boundsMax) * 0.5
        
        // SPX Format 20 data layout (based on Go reference implementation):
        // Positions: count * 3 bytes each for X, Y, Z (stored as arrays, not interleaved)
        // Scales: count * 1 byte each for X, Y, Z
        // Colors: count * 1 byte each for R, G, B, A
        // Rotations: count * 1 byte each for W, X, Y, Z
        
        for i in 0..<count {
            // Position (3 bytes each for X, Y, Z - stored in separate arrays)
            let posX = decodeSpxPosition(from: data, at: i * 3)
            let posY = decodeSpxPosition(from: data, at: count * 3 + i * 3)
            let posZ = decodeSpxPosition(from: data, at: count * 6 + i * 3)
            var position = SIMD3<Float>(posX, posY, posZ)
            
            // Scale (1 byte each for X, Y, Z - stored in separate arrays)
            // SPX stores scales in log space, convert to linear space with exp()
            let scaleXLog = decodeSpxScale(data[count * 9 + i])
            let scaleYLog = decodeSpxScale(data[count * 10 + i])
            let scaleZLog = decodeSpxScale(data[count * 11 + i])
            
            // Convert from log space and clamp to reasonable values
            let scaleX = max(0.0001, min(100.0, exp(scaleXLog)))
            let scaleY = max(0.0001, min(100.0, exp(scaleYLog)))
            let scaleZ = max(0.0001, min(100.0, exp(scaleZLog)))
            let scale = SIMD3<Float>(scaleX, scaleY, scaleZ)
            
            // Color (1 byte each for R, G, B, A - stored in separate arrays)
            let r = Float(data[count * 12 + i]) / 255.0
            let g = Float(data[count * 13 + i]) / 255.0
            let b = Float(data[count * 14 + i]) / 255.0
            let opacity = Float(data[count * 15 + i]) / 255.0
            let color = SIMD3<Float>(r, g, b)
            
            // Rotation (1 byte each for W, X, Y, Z - stored in separate arrays)
            let rotW = data[count * 16 + i]
            let rotX = data[count * 17 + i]
            let rotY = data[count * 18 + i]
            let rotZ = data[count * 19 + i]
            let rotation = normalizeRotations(rotW, rotX, rotY, rotZ)
            
            if i == 0 { // First point in the block
                print("SPXSceneReader: Header bounds - X:[\(header.minX), \(header.maxX)], Y:[\(header.minY), \(header.maxY)], Z:[\(header.minZ), \(header.maxZ)]")
                print("SPXSceneReader: Raw position 0: (\(posX), \(posY), \(posZ))")
            }
            
            // Bounding box transformation logic
            let boundsSizeValid = boundsSize.x.isFinite && boundsSize.y.isFinite && boundsSize.z.isFinite &&
                                  abs(boundsSize.x) > 1e-6 && abs(boundsSize.y) > 1e-6 && abs(boundsSize.z) > 1e-6
            let boundsCenterValid = boundsCenter.x.isFinite && boundsCenter.y.isFinite && boundsCenter.z.isFinite
            
            if boundsSizeValid && boundsCenterValid {
                position = position * boundsSize * 0.5 + boundsCenter
                if i == 0 {
                    print("SPXSceneReader: Applied bounding box transformation")
                }
            } else if i == 0 {
                let defaultScale: Float = 1.0
                position *= defaultScale
                print("SPXSceneReader: Invalid bounding box (NaN or zero size), applying default scale factor \(defaultScale)")
                print("SPXSceneReader: Bounds size: \(boundsSize)")
                print("SPXSceneReader: Bounds center: \(boundsCenter)")
                print("SPXSceneReader: BoundsSize finite: (\(boundsSize.x.isFinite), \(boundsSize.y.isFinite), \(boundsSize.z.isFinite))")
                print("SPXSceneReader: BoundsCenter finite: (\(boundsCenter.x.isFinite), \(boundsCenter.y.isFinite), \(boundsCenter.z.isFinite))")
                print("SPXSceneReader: BoundsSize > minSize: (\(abs(boundsSize.x) > 1e-6), \(abs(boundsSize.y) > 1e-6), \(abs(boundsSize.z) > 1e-6))")
            }
            
            if i == 0 {
                print("SPXSceneReader: Final position 0: \(position)")
                print("SPXSceneReader: Point 0 - Pos: \(position), Scale: \(scale), Color: (\(Int(r*255)), \(Int(g*255)), \(Int(b*255)), \(Int(opacity*255))), Opacity: \(opacity)")
            }

            let point = SplatScenePoint(
                position: position,
                color: .linearFloat(color),
                opacity: .linearFloat(opacity),
                scale: .linearFloat(scale),
                rotation: rotation
            )
            newPoints.append(point)
        }
        
        return newPoints
    }
    
    // MARK: - SH Data Application Functions
    
    private func applySHData(_ data: Data, count: Int, to shCoefficients: inout [[SIMD3<Float>]], shOrder: Int, startIndex: Int) throws {
        // A block might have a gaussian count but no actual data.
        if count == 0 { return }
        guard !data.isEmpty else {
            print("SPXSceneReader: Warning - applySHData called with empty data for \(count) points. This may indicate a malformed block.")
            return
        }

        var coeffsPerPoint = 0
        switch shOrder {
        case 1: coeffsPerPoint = 3  // SH Degree 1 adds 3 coefficients
        case 2: coeffsPerPoint = 8  // SH Degrees 1 & 2 add 8 coefficients
        case 3: coeffsPerPoint = 7  // SH Degree 3 adds 7 coefficients
        default: return
        }

        let expectedDataSize = count * coeffsPerPoint * 3 // Each coefficient has 3 channels (RGB)
        guard data.count >= expectedDataSize else {
            print("SPXSceneReader: Warning - SH data is insufficient. Expected \(expectedDataSize), got \(data.count). Skipping SH block.")
            return
        }
        
        var shDataByteIndex = 0
        for i in 0..<count {
            let pointIndex = startIndex + i
            guard pointIndex < shCoefficients.count else { break }

            var pointSHCoeffs: [SIMD3<Float>] = []
            pointSHCoeffs.reserveCapacity(coeffsPerPoint)

            for _ in 0..<coeffsPerPoint {
                let r = (Float(data[shDataByteIndex    ]) - 128.0) / 128.0
                let g = (Float(data[shDataByteIndex + 1]) - 128.0) / 128.0
                let b = (Float(data[shDataByteIndex + 2]) - 128.0) / 128.0
                pointSHCoeffs.append(SIMD3<Float>(r, g, b))
                shDataByteIndex += 3
            }

            // Format 3 (SH3) is appended to existing SH1/2 data.
            // Formats 1 and 2 replace any previous (non-DC) SH data.
            if shOrder == 3 {
                shCoefficients[pointIndex].append(contentsOf: pointSHCoeffs)
            } else {
                shCoefficients[pointIndex] = pointSHCoeffs
            }
        }
    }
    
    private func parseFormatSH(_ data: Data, count: Int, formatID: UInt8) throws -> [SplatScenePoint] {
        guard let header = header else {
            throw SPXFileFormatError.invalidHeader
        }
        
        let shDegree = Int(header.shDegree)
        let shDim = shDimForDegree(shDegree)
        let pointSize = SPXBasicGaussian.byteSize + (shDim * 3) // Basic data + SH coefficients
        
        let expectedSize = count * pointSize
        guard data.count >= expectedSize else {
            throw SPXFileFormatError.insufficientData
        }
        
        var points: [SplatScenePoint] = []
        points.reserveCapacity(count)
        
        var offset = 0
        for _ in 0..<count {
            // Parse basic data first (same as Format 20)
            guard offset + SPXBasicGaussian.byteSize <= data.count else {
                break
            }
            
            var gaussian = SPXSHGaussian()
            
            // Parse basic gaussian data
            let x = UInt32(data[offset]) | (UInt32(data[offset+1]) << 8) | (UInt32(data[offset+2]) << 16)
            let y = UInt32(data[offset+3]) | (UInt32(data[offset+4]) << 8) | (UInt32(data[offset+5]) << 16)
            let z = UInt32(data[offset+6]) | (UInt32(data[offset+7]) << 8) | (UInt32(data[offset+8]) << 16)
            offset += 9
            
            let scaleX = data[offset]
            let scaleY = data[offset+1]
            let scaleZ = data[offset+2]
            offset += 3
            
            let colorR = data[offset]
            let colorG = data[offset+1]
            let colorB = data[offset+2]
            let colorA = data[offset+3]
            offset += 4
            
            gaussian.baseData.position = SIMD3<UInt32>(x, y, z)
            gaussian.baseData.scale = SIMD3<UInt8>(scaleX, scaleY, scaleZ)
            gaussian.baseData.color = SIMD4<UInt8>(colorR, colorG, colorB, colorA)
            
            // Parse SH coefficients
            gaussian.shCoefficients.reserveCapacity(shDim)
            for _ in 0..<shDim {
                guard offset + 3 <= data.count else {
                    break
                }
                
                let shR = data[offset]
                let shG = data[offset+1]
                let shB = data[offset+2]
                offset += 3
                
                gaussian.shCoefficients.append(SIMD3<UInt8>(shR, shG, shB))
            }
            
            points.append(gaussian.toSplatScenePoint(shDegree: shDegree))
        }
        
        return points
    }
    
    // MARK: - Compression Support
    
    private static func isGzipped(_ data: Data) -> Bool {
        return data.starts(with: [0x1f, 0x8b])
    }
    
    private static func decompressData(_ data: Data) -> Data? {
        do {
            return try data.gunzipped()
        } catch {
            print("Error decompressing data block: \(error)")
            return nil
        }
    }
    
    private static func decompressGzipped(_ data: Data) -> Data? {
        do {
            return try data.gunzipped()
        } catch {
            print("Error decompressing gzipped data: \(error)")
            return nil
        }
    }
}

// MARK: - Array Extension for Chunking

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

// MARK: - Format-specific Parsers
private extension SPXSceneReader {
    func decodeSpxPosition(from data: Data, at offset: Int) -> Float {
        let b1 = data[offset]
        let b2 = data[offset + 1]
        let b3 = data[offset + 2]
        return SplatIO.decodeSpxPosition(b1, b2, b3)
    }

    func decodeSpxScale(_ byte: UInt8) -> Float {
        return Float(byte) / 16.0 - 10.0
    }

    func normalizeRotations(_ w: UInt8, _ x: UInt8, _ y: UInt8, _ z: UInt8) -> simd_quatf {
        // Match Go implementation: cmn.NormalizeRotations
        var r0 = Double(w) / 128.0 - 1.0
        var r1 = Double(x) / 128.0 - 1.0
        var r2 = Double(y) / 128.0 - 1.0
        var r3 = Double(z) / 128.0 - 1.0
        
        if r0 < 0 {
            r0 = -r0
            r1 = -r1
            r2 = -r2
            r3 = -r3
        }
        
        let qlen = sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3)
        
        return simd_quatf(
            ix: Float(r1 / qlen),
            iy: Float(r2 / qlen),
            iz: Float(r3 / qlen),
            r: Float(r0 / qlen)
        )
    }
}

private func decodeSpxPosition(_ packed: UInt32) -> Float {
    let sign = (packed >> 23) & 1
    let value = packed & 0x7FFFFF
    var floatValue = Float(value) / Float(1 << 22)
    if sign != 0 {
        floatValue -= 1.0
    }
    return floatValue
}