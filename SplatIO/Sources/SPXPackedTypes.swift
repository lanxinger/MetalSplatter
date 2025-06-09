import Foundation
import simd

/**
 * Represents the 128-byte header structure for SPX format files
 */
struct SPXHeader {
    static let magic: [UInt8] = [0x73, 0x70, 0x78] // "spx"
    static let headerSize: Int = 128
    
    var magic: [UInt8] = SPXHeader.magic
    var version: UInt8 = 1
    var splatCount: UInt32 = 0
    var minX: Float = 0
    var maxX: Float = 0
    var minY: Float = 0
    var maxY: Float = 0
    var minZ: Float = 0
    var maxZ: Float = 0
    var minTopY: Float = 0
    var maxTopY: Float = 0
    var createDate: UInt32 = 0
    var createrId: UInt32 = 0
    var exclusiveId: UInt32 = 0
    var shDegree: UInt8 = 0
    var flag1: UInt8 = 0
    var flag2: UInt8 = 0
    var flag3: UInt8 = 0
    var reserve1: UInt32 = 0
    var reserve2: UInt32 = 0
    var comment: [UInt8] = Array(repeating: 0, count: 60)
    var hash: UInt32 = 0
    
    init() {}
    
    init(data: Data) throws {
        guard data.count >= SPXHeader.headerSize else {
            throw SPXFileFormatError.invalidHeader
        }
        
        let magicBytes = Array(data[0..<3])
        guard magicBytes == SPXHeader.magic else {
            throw SPXFileFormatError.invalidMagicNumber
        }
        self.magic = magicBytes
        
        self.version = data[3]
        
        // Use withUnsafeBytes for safe, alignment-unaware memory access.
        self.splatCount = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self).littleEndian }
        
        self.minX = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 8, as: UInt32.self).littleEndian) }
        self.maxX = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 12, as: UInt32.self).littleEndian) }
        self.minY = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 16, as: UInt32.self).littleEndian) }
        self.maxY = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 20, as: UInt32.self).littleEndian) }
        self.minZ = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 24, as: UInt32.self).littleEndian) }
        self.maxZ = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 28, as: UInt32.self).littleEndian) }
        
        print("SPXHeader.init: Parsed BBox - minX: \(self.minX), maxX: \(self.maxX), minY: \(self.minY), maxY: \(self.maxY), minZ: \(self.minZ), maxZ: \(self.maxZ)")
        
        self.minTopY = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 32, as: UInt32.self).littleEndian) }
        self.maxTopY = data.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 36, as: UInt32.self).littleEndian) }
        
        self.createDate = data.withUnsafeBytes { $0.load(fromByteOffset: 40, as: UInt32.self).littleEndian }
        self.createrId = data.withUnsafeBytes { $0.load(fromByteOffset: 44, as: UInt32.self).littleEndian }
        self.exclusiveId = data.withUnsafeBytes { $0.load(fromByteOffset: 48, as: UInt32.self).littleEndian }
        
        self.shDegree = data[52]
        self.flag1 = data[53]
        self.flag2 = data[54]
        self.flag3 = data[55]
        
        self.reserve1 = data.withUnsafeBytes { $0.load(fromByteOffset: 56, as: UInt32.self).littleEndian }
        self.reserve2 = data.withUnsafeBytes { $0.load(fromByteOffset: 60, as: UInt32.self).littleEndian }
        
        self.comment = Array(data[64..<124])
        self.hash = data.withUnsafeBytes { $0.load(fromByteOffset: 124, as: UInt32.self).littleEndian }
    }
    
    func serialize() -> Data {
        var data = Data(capacity: SPXHeader.headerSize)
        
        // Magic (3 bytes)
        data.append(contentsOf: magic)
        
        // Version (1 byte)
        data.append(version)
        
        // Splat count (4 bytes)
        var splatCountLE = splatCount.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &splatCountLE) { Data($0) })
        
        // Bounding box (24 bytes)
        var minXLE = minX.bitPattern.littleEndian
        var maxXLE = maxX.bitPattern.littleEndian
        var minYLE = minY.bitPattern.littleEndian
        var maxYLE = maxY.bitPattern.littleEndian
        var minZLE = minZ.bitPattern.littleEndian
        var maxZLE = maxZ.bitPattern.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &minXLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &maxXLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &minYLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &maxYLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &minZLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &maxZLE) { Data($0) })
        
        // Top Y height (8 bytes)
        var minTopYLE = minTopY.bitPattern.littleEndian
        var maxTopYLE = maxTopY.bitPattern.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &minTopYLE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &maxTopYLE) { Data($0) })
        
        // Create date (4 bytes)
        var createDateLE = createDate.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &createDateLE) { Data($0) })
        
        // Creater ID (4 bytes)
        var createrIdLE = createrId.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &createrIdLE) { Data($0) })
        
        // Exclusive ID (4 bytes)
        var exclusiveIdLE = exclusiveId.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &exclusiveIdLE) { Data($0) })
        
        // SH degree (1 byte)
        data.append(shDegree)
        
        // Flags (3 bytes)
        data.append(flag1)
        data.append(flag2)
        data.append(flag3)
        
        // Reserve fields (8 bytes)
        var reserve1LE = reserve1.littleEndian
        var reserve2LE = reserve2.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &reserve1LE) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: &reserve2LE) { Data($0) })
        
        // Comment (60 bytes)
        data.append(contentsOf: comment)
        
        // Hash (4 bytes)
        var hashLE = hash.littleEndian
        data.append(contentsOf: withUnsafeBytes(of: &hashLE) { Data($0) })
        
        return data
    }
}

/**
 * Represents a data block within an SPX file
 */
struct SPXDataBlock {
    var blockLength: Int32 = 0
    var gaussianCount: UInt32 = 0
    var formatID: UInt32 = 0
    var data: Data = Data()
    
    var isCompressed: Bool {
        return blockLength < 0
    }
    
    var actualDataLength: Int {
        return isCompressed ? Int(-blockLength) : Int(blockLength)
    }
    
    init() {}
    
    init(blockLength: Int32, gaussianCount: UInt32, formatID: UInt32, data: Data) {
        self.blockLength = blockLength
        self.gaussianCount = gaussianCount
        self.formatID = formatID
        self.data = data
    }
}

/**
 * Format 20: Basic data representation
 */
struct SPXBasicGaussian {
    var position: SIMD3<UInt32> = SIMD3<UInt32>(0, 0, 0) // 24-bit coordinates (3 bytes each)
    var scale: SIMD3<UInt8> = SIMD3<UInt8>(0, 0, 0)      // 8-bit per axis
    var color: SIMD4<UInt8> = SIMD4<UInt8>(0, 0, 0, 255) // RGBA, 8-bit channels
    var rotation: SIMD4<UInt8> = SIMD4<UInt8>(0, 0, 0, 255) // Quaternion, 8-bit components
    
    static let byteSize = 16 // 9 + 3 + 4 = 16 bytes per point
    
    func toSplatScenePoint() -> SplatScenePoint {
        // Decode 24-bit positions (SPX uses special encoding)
        let pos = SIMD3<Float>(
            decodeSpxPosition(position.x),
            decodeSpxPosition(position.y), 
            decodeSpxPosition(position.z)
        )
        
        // Decode 8-bit scale values (SPX uses special encoding)
        let scl = SIMD3<Float>(
            decodeSpxScale(scale.x),
            decodeSpxScale(scale.y),
            decodeSpxScale(scale.z)
        )
        
        // Convert 8-bit color to float
        let clr = SIMD3<Float>(
            Float(color.x) / 255.0,
            Float(color.y) / 255.0,
            Float(color.z) / 255.0
        )
        
        // Normalize rotation quaternion (SPX stores as 4 bytes)
        let quat = normalizeRotations(rotation.w, rotation.x, rotation.y, rotation.z)
        
        // Convert alpha to proper opacity
        let opacity = Float(color.w) / 255.0
        
        return SplatScenePoint(
            position: pos,
            color: .linearFloat(clr),
            opacity: .linearFloat(opacity),
            scale: .linearFloat(scl),
            rotation: quat
        )
    }
    
    private func decodeSpxPosition(_ value: UInt32) -> Float {
        // SPX 24-bit position decoding
        // Based on the Go implementation: cmn.DecodeSpxPositionUint24
        // Convert 24-bit unsigned to signed value and normalize
        let masked = value & 0xFFFFFF // Ensure only 24 bits
        
        // Convert to signed 24-bit value
        let signed: Int32
        if masked >= 0x800000 { // If sign bit is set
            signed = Int32(masked) - 0x1000000 // Convert to negative
        } else {
            signed = Int32(masked)
        }
        
        // Normalize to world coordinates - match Go implementation exactly  
        return Float(signed) / Float(0x800000)
    }
    
    private func decodeSpxScale(_ value: UInt8) -> Float {
        // SPX scale decoding based on Go implementation: cmn.DecodeSpxScale
        // Handle zero values specially to avoid tiny scales
        if value == 0 {
            return 0.01 // Default scale for zero values
        }
        
        let scale = exp((Float(value) / 255.0) * 12.0 - 6.0)
        return max(scale, 0.001) // Minimum scale to ensure visibility
    }
    
    private func normalizeRotations(_ w: UInt8, _ x: UInt8, _ y: UInt8, _ z: UInt8) -> simd_quatf {
        // Convert bytes to normalized floats and create quaternion
        let fw = Float(w) / 127.5 - 1.0
        let fx = Float(x) / 127.5 - 1.0
        let fy = Float(y) / 127.5 - 1.0
        let fz = Float(z) / 127.5 - 1.0
        
        return simd_quatf(ix: fx, iy: fy, iz: fz, r: fw).normalized
    }
    
    private func logit(_ x: Float) -> Float {
        let clamped = max(0.0001, min(0.9999, x))
        return log(clamped / (1.0 - clamped))
    }
}

/**
 * Format 1-3: Spherical harmonics data representations
 */
struct SPXSHGaussian {
    var baseData: SPXBasicGaussian = SPXBasicGaussian()
    var shCoefficients: [SIMD3<UInt8>] = [] // SH coefficients for RGB
    
    func toSplatScenePoint(shDegree: Int) -> SplatScenePoint {
        var point = baseData.toSplatScenePoint()
        
        // Convert SH coefficients based on degree
        let shDim = shDimForDegree(shDegree)
        var shR: [Float] = Array(repeating: 0, count: shDim)
        var shG: [Float] = Array(repeating: 0, count: shDim)
        var shB: [Float] = Array(repeating: 0, count: shDim)
        
        for i in 0..<min(shCoefficients.count, shDim) {
            shR[i] = (Float(shCoefficients[i].x) - 128.0) / 128.0
            shG[i] = (Float(shCoefficients[i].y) - 128.0) / 128.0
            shB[i] = (Float(shCoefficients[i].z) - 128.0) / 128.0
        }
        
        var shCoeffs: [SIMD3<Float>] = []
        for i in 0..<shDim {
            shCoeffs.append(SIMD3<Float>(
                i < shR.count ? shR[i] : 0.0,
                i < shG.count ? shG[i] : 0.0,
                i < shB.count ? shB[i] : 0.0
            ))
        }
        point.color = .sphericalHarmonic(shCoeffs)
        
        return point
    }
}

// MARK: - Global SPX Utility Functions

func decodeSpxPosition(_ byte1: UInt8, _ byte2: UInt8, _ byte3: UInt8) -> Float {
    // SPX 24-bit position decoding from 3 bytes
    // Based on the Go implementation: cmn.DecodeSpxPositionUint24
    let value = UInt32(byte1) | (UInt32(byte2) << 8) | (UInt32(byte3) << 16)
    
    // Convert to signed 24-bit value
    let signed: Int32
    if value >= 0x800000 { // If sign bit is set
        signed = Int32(value) - 0x1000000 // Convert to negative
    } else {
        signed = Int32(value)
    }
    
    // Normalize to world coordinates - match Go implementation exactly
    let normalized = Float(signed) / Float(0x800000)
    
    
    return normalized
}

func decodeSpxScale(_ value: UInt8) -> Float {
    // SPX scale decoding based on Go implementation: cmn.DecodeSpxScale
    // Handle zero values specially to avoid tiny scales
    if value == 0 {
        return 0.01 // Default scale for zero values
    }
    
    let scale = exp((Float(value) / 255.0) * 12.0 - 6.0)
    return max(scale, 0.001) // Minimum scale to ensure visibility
}

func normalizeRotations(_ w: UInt8, _ x: UInt8, _ y: UInt8, _ z: UInt8) -> simd_quatf {
    // Convert bytes to normalized floats and create quaternion
    let fw = Float(w) / 127.5 - 1.0
    let fx = Float(x) / 127.5 - 1.0
    let fy = Float(y) / 127.5 - 1.0
    let fz = Float(z) / 127.5 - 1.0
    
    return simd_quatf(ix: fx, iy: fy, iz: fz, r: fw).normalized
}