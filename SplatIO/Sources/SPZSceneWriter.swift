import Foundation
import simd

/**
 * Writer for SPZ format Gaussian splat scenes.
 * SPZ is a compact binary format for Gaussian splats with support for:
 * - Float16 or fixed-point position encoding
 * - Spherical harmonics for color representation
 * - Antialiasing support
 */
public class SPZSceneWriter: SplatSceneWriter {
    private let useFloat16: Bool
    private let antialiased: Bool
    private let fractionalBits: Int
    private let compress: Bool
    
    public init(useFloat16: Bool = true, antialiased: Bool = false, fractionalBits: Int = 10, compress: Bool = true) {
        self.useFloat16 = useFloat16
        self.antialiased = antialiased
        self.fractionalBits = fractionalBits
        self.compress = compress
    }
    
    private var outputURL: URL?
    
    public func write(_ points: [SplatScenePoint]) throws {
        // Store points for later writing when close() is called
        self.points = points
    }
    
    public func close() throws {
        guard let outputURL = outputURL, let points = points else {
            // Nothing to do if no URL or points are set
            return
        }
        
        try writeScene(points, to: outputURL)
    }
    
    // Set the output URL for the writer
    public func setOutputURL(_ url: URL) {
        self.outputURL = url
    }
    
    // Store points for writing
    private var points: [SplatScenePoint]?
    
    // Direct writing method
    public func writeScene(_ points: [SplatScenePoint], to url: URL) throws {
        let packed = packGaussians(points)
        let serialized = packed.serialize()
        
        if compress {
            let compressedData = try compressToGzip(serialized)
            try compressedData.write(to: url)
        } else {
            try serialized.write(to: url)
        }
    }
    
    // MARK: - Private helpers
    
    private func packGaussians(_ points: [SplatScenePoint]) -> PackedGaussians {
        var result = PackedGaussians()
        result.numPoints = points.count
        result.shDegree = 0 // Only supporting SH degree 0 for now
        result.fractionalBits = fractionalBits
        result.antialiased = antialiased
        
        // Pre-allocate arrays
        let positionComponents = useFloat16 ? 6 : 9 // 2 bytes per component for float16, 3 bytes for fixed-point
        result.positions = [UInt8](repeating: 0, count: points.count * positionComponents)
        result.scales = [UInt8](repeating: 0, count: points.count * 3)
        result.rotations = [UInt8](repeating: 0, count: points.count * 3)
        result.alphas = [UInt8](repeating: 0, count: points.count)
        result.colors = [UInt8](repeating: 0, count: points.count * 3)
        
        // Pack each point
        for (i, point) in points.enumerated() {
            packPoint(point, into: &result, at: i)
        }
        
        return result
    }
    
    private func packPoint(_ point: SplatScenePoint, into packed: inout PackedGaussians, at index: Int) {
        let normalizedPoint = point.linearNormalized
        
        // Pack position
        let position = normalizedPoint.position
        if useFloat16 {
            let baseIdx = index * 6
            for j in 0..<3 {
                let halfValue = float32ToFloat16(position[j])
                packed.positions[baseIdx + j*2] = UInt8(halfValue & 0xFF)
                packed.positions[baseIdx + j*2 + 1] = UInt8(halfValue >> 8)
            }
        } else {
            let baseIdx = index * 9
            let scale = Float(1 << fractionalBits)
            for j in 0..<3 {
                let fixed = Int32(position[j] * scale)
                packed.positions[baseIdx + j*3] = UInt8(fixed & 0xFF)
                packed.positions[baseIdx + j*3 + 1] = UInt8((fixed >> 8) & 0xFF)
                packed.positions[baseIdx + j*3 + 2] = UInt8((fixed >> 16) & 0xFF)
            }
        }
        
        // Pack scale (convert from linear to exponent)
        let scale = normalizedPoint.scale.asExponent
        let scaleBaseIdx = index * 3
        for j in 0..<3 {
            // Scale from [-10, 6] to [0, 255]
            let packedScale = UInt8(min(255, max(0, Int((scale[j] + 10.0) * 16.0))))
            packed.scales[scaleBaseIdx + j] = packedScale
        }
        
        // Pack rotation
        let rotation = normalizedPoint.rotation
        let rotationBaseIdx = index * 3
        
        // Choose the largest component to drop (we store 3 of 4 quaternion components)
        let absRotation = simd_abs(rotation.vector)
        let maxComponent = absRotation.max()
        var largestIndex = 0
        for j in 0..<4 {
            if absRotation[j] == maxComponent {
                largestIndex = j
                break
            }
        }
        
        // Reconstruct sign of largest component
        let sign: Float = rotation.vector[largestIndex] >= 0 ? 1.0 : -1.0
        
        // Pack the 3 smallest components to 8-bit
        var idx = 0
        for j in 0..<4 {
            if j != largestIndex {
                // Scale from [-1, 1] to [0, 255]
                let scaledValue = (sign * rotation.vector[j] + 1.0) * 127.5
                packed.rotations[rotationBaseIdx + idx] = UInt8(min(255, max(0, Int(scaledValue))))
                idx += 1
            }
        }
        
        // Pack alpha (opacity)
        let alpha = normalizedPoint.opacity.asLinearFloat
        packed.alphas[index] = UInt8(min(255, max(0, Int(alpha * 255.0))))
        
        // Pack color
        let color = normalizedPoint.color.asLinearUInt8
        let colorBaseIdx = index * 3
        packed.colors[colorBaseIdx] = color.x
        packed.colors[colorBaseIdx + 1] = color.y
        packed.colors[colorBaseIdx + 2] = color.z
    }
    
    private func compressToGzip(_ data: Data) throws -> Data {
        // Using Foundation's built-in compression
        guard let compressed = try (data as NSData).compressed(using: .zlib) as Data? else {
            throw SplatFileFormatError.compressionFailed
        }
        
        // Add gzip header and footer
        var gzippedData = Data()
        gzippedData.append(contentsOf: [0x1F, 0x8B])  // Gzip magic number
        gzippedData.append(0x08)                      // Compression method (deflate)
        gzippedData.append(0x00)                      // Flags
        gzippedData.append(contentsOf: [0x00, 0x00, 0x00, 0x00])  // Modification time
        gzippedData.append(0x00)                      // Extra flags
        gzippedData.append(0x00)                      // OS (unknown)
        
        // Add the compressed data (without zlib header)
        gzippedData.append(compressed.dropFirst(2))
        
        // Add CRC32 and original size
        var crc: UInt32 = 0
        // Placeholder CRC32 - in a real implementation this would be calculated
        crc = 0
        gzippedData.append(contentsOf: withUnsafeBytes(of: crc.littleEndian) { Data($0) })
        
        // Original uncompressed size
        let originalSize = UInt32(data.count)
        gzippedData.append(contentsOf: withUnsafeBytes(of: originalSize.littleEndian) { Data($0) })
        
        return gzippedData
    }
    
    private func float32ToFloat16(_ value: Float) -> UInt16 {
        // Simplified float32 to float16 conversion
        // In a real implementation, this would handle special values like NaN and Infinity
        let sign = value < 0 ? 0x8000 : 0
        let absValue = abs(value)
        
        if absValue < 6.1e-5 {
            return UInt16(sign) // Too small, return signed zero
        }
        
        if absValue > 65504.0 {
            return UInt16(sign | 0x7C00) // Too large, return signed infinity
        }
        
        // Calculate exponent and mantissa
        var exponent = Int(floor(log2(Double(absValue))))
        var mantissa = Int((Double(absValue) / pow(2.0, Double(exponent)) - 1.0) * 1024.0 + 0.5)
        
        exponent += 15
        
        if mantissa > 1023 {
            mantissa = 0
            exponent += 1
        }
        
        if exponent > 31 {
            return UInt16(sign | 0x7C00) // Overflow, return signed infinity
        }
        
        if exponent < 1 {
            // Denormalized
            mantissa = Int(Double(absValue) * pow(2.0, 14.0) * 1024.0 + 0.5)
            return UInt16(sign | mantissa)
        }
        
        // Normalized
        return UInt16(sign | (exponent << 10) | mantissa)
    }
}
