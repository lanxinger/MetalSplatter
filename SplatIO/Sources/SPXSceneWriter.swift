import Foundation
import simd
import Compression

/**
 * Writer for SPX format Gaussian splat scenes.
 * SPX is a flexible, extensible format for 3D Gaussian Splatting models with:
 * - 128-byte header with metadata and bounding box
 * - Variable data blocks with different format types
 * - Support for gzip compression
 * - Spherical harmonics support
 */
public class SPXSceneWriter: SplatSceneWriter {
    private let compress: Bool
    private let formatID: UInt32
    private var outputURL: URL?
    private var points: [SplatScenePoint]?
    
    public init(compress: Bool = false, formatID: UInt32 = 20) {
        self.compress = compress
        self.formatID = formatID
    }
    
    public func write(_ points: [SplatScenePoint]) throws {
        self.points = points
    }
    
    public func close() throws {
        guard let outputURL = outputURL, let points = points else {
            return
        }
        
        try writeScene(points, to: outputURL)
    }
    
    public func setOutputURL(_ url: URL) {
        self.outputURL = url
    }
    
    public func writeScene(_ points: [SplatScenePoint], to url: URL) throws {
        print("SPXSceneWriter: Writing \(points.count) points to \(url.path)")
        
        // Calculate bounding box
        let boundingBox = calculateBoundingBox(points)
        
        // Create header
        var header = SPXHeader()
        header.version = 1
        header.splatCount = UInt32(points.count)
        header.minX = boundingBox.min.x
        header.maxX = boundingBox.max.x
        header.minY = boundingBox.min.y
        header.maxY = boundingBox.max.y
        header.minZ = boundingBox.min.z
        header.maxZ = boundingBox.max.z
        header.minTopY = boundingBox.min.y
        header.maxTopY = boundingBox.max.y
        
        // Set creation date in YYYYMMDD format
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd"
        if let dateInt = UInt32(dateFormatter.string(from: Date())) {
            header.createDate = dateInt
        }
        
        header.shDegree = determineSHDegree(points)
        
        // Create data block
        let blockData = try createDataBlock(points, formatID: formatID)
        
        // Prepare output data
        var outputData = header.serialize()
        outputData.append(blockData)
        
        // Apply compression if requested
        if compress {
            guard let compressed = compressData(outputData) else {
                throw SPXFileFormatError.compressionFailed
            }
            try compressed.write(to: url)
        } else {
            try outputData.write(to: url)
        }
        
        print("SPXSceneWriter: Successfully wrote SPX file")
    }
    
    // MARK: - Private Helper Methods
    
    private func calculateBoundingBox(_ points: [SplatScenePoint]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !points.isEmpty else {
            return (min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(0, 0, 0))
        }
        
        var minPos = points[0].position
        var maxPos = points[0].position
        
        for point in points {
            minPos = min(minPos, point.position)
            maxPos = max(maxPos, point.position)
        }
        
        return (min: minPos, max: maxPos)
    }
    
    private func determineSHDegree(_ points: [SplatScenePoint]) -> UInt8 {
        for point in points {
            if case .sphericalHarmonic(let shCoeffs) = point.color {
                if shCoeffs.count >= 15 {
                    return 3
                } else if shCoeffs.count >= 8 {
                    return 2
                } else if shCoeffs.count >= 3 {
                    return 1
                }
            }
        }
        return 0
    }
    
    private func createDataBlock(_ points: [SplatScenePoint], formatID: UInt32) throws -> Data {
        var blockData = Data()
        
        // Determine if we need SH data
        let needsSH = points.contains { point in
            if case .sphericalHarmonic = point.color {
                return true
            }
            return false
        }
        
        let actualFormatID = needsSH ? min(formatID, 3) : 20
        
        // Create point data based on format
        let pointData: Data
        switch actualFormatID {
        case 20:
            pointData = try createFormat20Data(points)
        case 1, 2, 3:
            pointData = try createFormatSHData(points, degree: Int(actualFormatID))
        default:
            throw SPXFileFormatError.invalidBlockFormat
        }
        
        // Create block header
        let blockLength = Int32(pointData.count)
        let compressedBlockLength = compress ? -blockLength : blockLength
        
        blockData.append(contentsOf: withUnsafeBytes(of: compressedBlockLength) { Data($0) })
        blockData.append(contentsOf: withUnsafeBytes(of: UInt32(points.count)) { Data($0) })
        blockData.append(contentsOf: withUnsafeBytes(of: actualFormatID) { Data($0) })
        
        // Add point data (compress if requested)
        if compress {
            guard let compressed = compressData(pointData) else {
                throw SPXFileFormatError.compressionFailed
            }
            blockData.append(compressed)
        } else {
            blockData.append(pointData)
        }
        
        return blockData
    }
    
    private func createFormat20Data(_ points: [SplatScenePoint]) throws -> Data {
        var data = Data()
        data.reserveCapacity(points.count * SPXBasicGaussian.byteSize)
        
        for point in points {
            let gaussian = try splatPointToBasicGaussian(point)
            data.append(encodeBasicGaussian(gaussian))
        }
        
        return data
    }
    
    private func createFormatSHData(_ points: [SplatScenePoint], degree: Int) throws -> Data {
        let shDim = shDimForDegree(degree)
        let pointSize = SPXBasicGaussian.byteSize + (shDim * 3)
        
        var data = Data()
        data.reserveCapacity(points.count * pointSize)
        
        for point in points {
            let gaussian = try splatPointToSHGaussian(point, degree: degree)
            data.append(encodeSHGaussian(gaussian, shDim: shDim))
        }
        
        return data
    }
    
    private func splatPointToBasicGaussian(_ point: SplatScenePoint) throws -> SPXBasicGaussian {
        var gaussian = SPXBasicGaussian()
        
        // Convert position to 24-bit coordinates (normalize to [0, 1] then scale)
        let normalizedPos = (point.position + SIMD3<Float>(1, 1, 1)) * 0.5 // [-1,1] to [0,1]
        gaussian.position = SIMD3<UInt32>(
            UInt32(clamp(normalizedPos.x, 0, 1) * Float(0xFFFFFF)),
            UInt32(clamp(normalizedPos.y, 0, 1) * Float(0xFFFFFF)),
            UInt32(clamp(normalizedPos.z, 0, 1) * Float(0xFFFFFF))
        )
        
        // Convert scale (exponential to linear)
        let scale = point.scale.asLinearFloat
        gaussian.scale = SIMD3<UInt8>(
            UInt8(clamp((log(scale.x) + 10.0) / 20.0 * 255.0, 0, 255)),
            UInt8(clamp((log(scale.y) + 10.0) / 20.0 * 255.0, 0, 255)),
            UInt8(clamp((log(scale.z) + 10.0) / 20.0 * 255.0, 0, 255))
        )
        
        // Convert color
        let color = extractBaseColor(point.color)
        gaussian.color = SIMD4<UInt8>(
            UInt8(clamp(color.x * 255.0, 0, 255)),
            UInt8(clamp(color.y * 255.0, 0, 255)),
            UInt8(clamp(color.z * 255.0, 0, 255)),
            UInt8(clamp(sigmoid(point.opacity.asLogitFloat) * 255.0, 0, 255))
        )
        
        // Convert rotation
        let quat = point.rotation.normalized
        gaussian.rotation = SIMD4<UInt8>(
            UInt8(clamp((quat.imag.x + 1.0) * 127.5, 0, 255)),
            UInt8(clamp((quat.imag.y + 1.0) * 127.5, 0, 255)),
            UInt8(clamp((quat.imag.z + 1.0) * 127.5, 0, 255)),
            UInt8(clamp((quat.real + 1.0) * 127.5, 0, 255))
        )
        
        return gaussian
    }
    
    private func splatPointToSHGaussian(_ point: SplatScenePoint, degree: Int) throws -> SPXSHGaussian {
        var gaussian = SPXSHGaussian()
        gaussian.baseData = try splatPointToBasicGaussian(point)
        
        // Extract SH coefficients
        let shDim = shDimForDegree(degree)
        if case .sphericalHarmonic(let shCoeffs) = point.color, shCoeffs.count > 0 {
            gaussian.shCoefficients.reserveCapacity(shDim)
            
            for i in 0..<shDim {
                let coeff = i < shCoeffs.count ? shCoeffs[i] : SIMD3<Float>(0, 0, 0)
                let r = coeff.x
                let g = coeff.y
                let b = coeff.z
                
                gaussian.shCoefficients.append(SIMD3<UInt8>(
                    UInt8(clamp((r + 1.0) * 127.5, 0, 255)),
                    UInt8(clamp((g + 1.0) * 127.5, 0, 255)),
                    UInt8(clamp((b + 1.0) * 127.5, 0, 255))
                ))
            }
        } else {
            // Fill with neutral SH values
            gaussian.shCoefficients = Array(repeating: SIMD3<UInt8>(128, 128, 128), count: shDim)
        }
        
        return gaussian
    }
    
    private func encodeBasicGaussian(_ gaussian: SPXBasicGaussian) -> Data {
        var data = Data()
        data.reserveCapacity(SPXBasicGaussian.byteSize)
        
        // Position (9 bytes - 3 bytes per coordinate)
        data.append(UInt8(gaussian.position.x & 0xFF))
        data.append(UInt8((gaussian.position.x >> 8) & 0xFF))
        data.append(UInt8((gaussian.position.x >> 16) & 0xFF))
        data.append(UInt8(gaussian.position.y & 0xFF))
        data.append(UInt8((gaussian.position.y >> 8) & 0xFF))
        data.append(UInt8((gaussian.position.y >> 16) & 0xFF))
        data.append(UInt8(gaussian.position.z & 0xFF))
        data.append(UInt8((gaussian.position.z >> 8) & 0xFF))
        data.append(UInt8((gaussian.position.z >> 16) & 0xFF))
        
        // Scale (3 bytes)
        data.append(gaussian.scale.x)
        data.append(gaussian.scale.y)
        data.append(gaussian.scale.z)
        
        // Color RGBA (4 bytes)
        data.append(gaussian.color.x)
        data.append(gaussian.color.y)
        data.append(gaussian.color.z)
        data.append(gaussian.color.w)
        
        return data
    }
    
    private func encodeSHGaussian(_ gaussian: SPXSHGaussian, shDim: Int) -> Data {
        var data = encodeBasicGaussian(gaussian.baseData)
        
        // Add SH coefficients
        for i in 0..<shDim {
            let coeff = i < gaussian.shCoefficients.count ? gaussian.shCoefficients[i] : SIMD3<UInt8>(128, 128, 128)
            data.append(coeff.x)
            data.append(coeff.y)
            data.append(coeff.z)
        }
        
        return data
    }
    
    private func extractBaseColor(_ color: SplatScenePoint.Color) -> SIMD3<Float> {
        switch color {
        case .linearFloat(let rgb):
            return rgb
        case .linearFloat256(let rgb):
            return rgb / 256.0
        case .linearUInt8(let rgb):
            return SIMD3<Float>(Float(rgb.x) / 255.0, Float(rgb.y) / 255.0, Float(rgb.z) / 255.0)
        case .sphericalHarmonic(let shCoeffs):
            return SplatScenePoint.Color.sphericalHarmonic(shCoeffs).asLinearFloat
        }
    }
    
    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
    
    private func clamp(_ value: Float, _ min: Float, _ max: Float) -> Float {
        return Swift.max(min, Swift.min(max, value))
    }
    
    private func compressData(_ data: Data) -> Data? {
        return data.withUnsafeBytes { bytes in
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
            defer { buffer.deallocate() }
            
            let compressedSize = compression_encode_buffer(
                buffer, data.count,
                bytes.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, COMPRESSION_ZLIB
            )
            
            guard compressedSize > 0 else {
                return nil
            }
            
            return Data(bytes: buffer, count: compressedSize)
        }
    }
}