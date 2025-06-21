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
        
        // Validate data bounds for magic number access
        try SplatDataValidator.validateDataBounds(data: data, offset: 0, size: 3)
        let magicBytes = Array(data[0..<3])
        guard magicBytes == SPXHeader.magic else {
            throw SPXFileFormatError.invalidMagicNumber
        }
        self.magic = magicBytes
        
        // Validate bounds for version access
        try SplatDataValidator.validateDataBounds(data: data, offset: 3, size: 1)
        self.version = data[3]
        
        // Use safe data access with bounds checking
        self.splatCount = try SplatDataValidator.safeDataAccess(data: data, offset: 4, type: UInt32.self).littleEndian
        
        self.minX = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 8, type: UInt32.self).littleEndian)
        self.maxX = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 12, type: UInt32.self).littleEndian)
        self.minY = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 16, type: UInt32.self).littleEndian)
        self.maxY = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 20, type: UInt32.self).littleEndian)
        self.minZ = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 24, type: UInt32.self).littleEndian)
        self.maxZ = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 28, type: UInt32.self).littleEndian)
        
        // Validate bounding box values for NaN/infinity
        try SplatDataValidator.validateFinite(self.minX, name: "minX")
        try SplatDataValidator.validateFinite(self.maxX, name: "maxX")
        try SplatDataValidator.validateFinite(self.minY, name: "minY")
        try SplatDataValidator.validateFinite(self.maxY, name: "maxY")
        try SplatDataValidator.validateFinite(self.minZ, name: "minZ")
        try SplatDataValidator.validateFinite(self.maxZ, name: "maxZ")
        
        print("SPXHeader.init: Parsed BBox - minX: \(self.minX), maxX: \(self.maxX), minY: \(self.minY), maxY: \(self.maxY), minZ: \(self.minZ), maxZ: \(self.maxZ)")
        
        self.minTopY = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 32, type: UInt32.self).littleEndian)
        self.maxTopY = Float(bitPattern: try SplatDataValidator.safeDataAccess(data: data, offset: 36, type: UInt32.self).littleEndian)
        
        // Validate additional float values
        try SplatDataValidator.validateFinite(self.minTopY, name: "minTopY")
        try SplatDataValidator.validateFinite(self.maxTopY, name: "maxTopY")
        
        self.createDate = try SplatDataValidator.safeDataAccess(data: data, offset: 40, type: UInt32.self).littleEndian
        self.createrId = try SplatDataValidator.safeDataAccess(data: data, offset: 44, type: UInt32.self).littleEndian
        self.exclusiveId = try SplatDataValidator.safeDataAccess(data: data, offset: 48, type: UInt32.self).littleEndian
        
        // Validate bounds for flags access
        try SplatDataValidator.validateDataBounds(data: data, offset: 52, size: 4)
        self.shDegree = data[52]
        self.flag1 = data[53]
        self.flag2 = data[54]
        self.flag3 = data[55]
        
        self.reserve1 = try SplatDataValidator.safeDataAccess(data: data, offset: 56, type: UInt32.self).littleEndian
        self.reserve2 = try SplatDataValidator.safeDataAccess(data: data, offset: 60, type: UInt32.self).littleEndian
        
        // Validate bounds for comment and hash access
        try SplatDataValidator.validateDataBounds(data: data, offset: 64, size: 60)
        self.comment = Array(data[64..<124])
        self.hash = try SplatDataValidator.safeDataAccess(data: data, offset: 124, type: UInt32.self).littleEndian
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
    
    static let byteSize = 20 // 9 + 3 + 4 + 4 = 20 bytes per point
    
    func toSplatScenePoint() -> SplatScenePoint {
        // Decode 24-bit positions (SPX uses special encoding)
        let pos = SIMD3<Float>(
            decodeSpxPosition(position.x),
            decodeSpxPosition(position.y), 
            decodeSpxPosition(position.z)
        )
        
        // Decode 8-bit scale values (SPX stores in log space, convert to linear)
        let scaleXLog = decodeSpxScale(scale.x)
        let scaleYLog = decodeSpxScale(scale.y)
        let scaleZLog = decodeSpxScale(scale.z)
        
        // Convert from log space and clamp to reasonable values
        let scl = SIMD3<Float>(
            max(0.0001, min(100.0, exp(scaleXLog))),
            max(0.0001, min(100.0, exp(scaleYLog))),
            max(0.0001, min(100.0, exp(scaleZLog)))
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
        let i32 = Int32(value & 0xFFFFFF) // Ensure only 24 bits
        
        // Convert to signed 24-bit value
        let signed: Int32
        if i32 & 0x800000 > 0 { // If sign bit is set
            signed = i32 | (-0x1000000) // Convert to negative
        } else {
            signed = i32
        }
        
        // Normalize to world coordinates - match Go implementation exactly  
        return Float(signed) / 4096.0
    }
    
    private func decodeSpxScale(_ value: UInt8) -> Float {
        // SPX scale decoding based on Go implementation: cmn.DecodeSpxScale
        return Float(value) / 16.0 - 10.0
    }
    
    private func normalizeRotations(_ w: UInt8, _ x: UInt8, _ y: UInt8, _ z: UInt8) -> simd_quatf {
        // Convert bytes to normalized floats and create quaternion
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
    let i32 = Int32(byte1) | (Int32(byte2) << 8) | (Int32(byte3) << 16)
    
    // Convert to signed 24-bit value
    let signed: Int32
    if i32 & 0x800000 > 0 { // If sign bit is set
        signed = i32 | (-0x1000000) // Convert to negative
    } else {
        signed = i32
    }
    
    // Normalize to world coordinates - match Go implementation exactly
    return Float(signed) / 4096.0
}

func decodeSpxScale(_ value: UInt8) -> Float {
    // SPX scale decoding based on Go implementation: cmn.DecodeSpxScale
    return Float(value) / 16.0 - 10.0
}

func normalizeRotations(_ w: UInt8, _ x: UInt8, _ y: UInt8, _ z: UInt8) -> simd_quatf {
    // Convert bytes to normalized floats and create quaternion
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