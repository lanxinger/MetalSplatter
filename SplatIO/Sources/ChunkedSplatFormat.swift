import Foundation
import simd

/// Compressed splat format using 256-splat chunks with per-chunk quantization.
/// Achieves approximately 3.25:1 compression ratio (52 → 16 bytes per splat).
///
/// Each chunk contains:
/// - Header with min/max bounds for position, scale, and optionally color (12-18 floats)
/// - 256 packed splats (16 bytes each)
///
/// Compression scheme per splat:
/// - Position: 11-10-11 bits (32-bit total, lerp within chunk bounds)
/// - Rotation: 2-bit selector + 3×10-bit components (32-bit total)
/// - Scale: 11-10-11 bits with exponential mapping (32-bit total)
/// - Color: RGBA8 (32-bit total)
///
/// Total: 16 bytes vs 52 bytes uncompressed = 3.25:1 compression

/// Header for a chunk of 256 splats
public struct ChunkHeader: Sendable {
    /// Minimum position in the chunk (for decompression)
    public var minPosition: SIMD3<Float>

    /// Maximum position in the chunk (for decompression)
    public var maxPosition: SIMD3<Float>

    /// Minimum scale in the chunk (for exponential decompression)
    public var minScale: SIMD3<Float>

    /// Maximum scale in the chunk (for exponential decompression)
    public var maxScale: SIMD3<Float>

    /// Optional minimum color (nil = use per-splat RGBA8)
    public var minColor: SIMD4<Float>?

    /// Optional maximum color (nil = use per-splat RGBA8)
    public var maxColor: SIMD4<Float>?

    /// Number of splats in this chunk (≤256, may be less for final chunk)
    public var splatCount: UInt16

    /// Padding for alignment
    public var padding: UInt16 = 0

    public init(
        minPosition: SIMD3<Float>,
        maxPosition: SIMD3<Float>,
        minScale: SIMD3<Float>,
        maxScale: SIMD3<Float>,
        minColor: SIMD4<Float>? = nil,
        maxColor: SIMD4<Float>? = nil,
        splatCount: UInt16
    ) {
        self.minPosition = minPosition
        self.maxPosition = maxPosition
        self.minScale = minScale
        self.maxScale = maxScale
        self.minColor = minColor
        self.maxColor = maxColor
        self.splatCount = splatCount
    }

    /// Size of header in bytes (without optional color)
    public static let baseSize: Int = MemoryLayout<SIMD3<Float>>.stride * 4 + MemoryLayout<UInt16>.stride * 2

    /// Size of header in bytes (with optional color)
    public static let fullSize: Int = baseSize + MemoryLayout<SIMD4<Float>>.stride * 2
}

/// GPU-compatible header for chunk decompression
/// This matches the Metal shader structure exactly
public struct GPUChunkHeader {
    public var minPosition: SIMD3<Float>
    public var padding1: Float = 0

    public var maxPosition: SIMD3<Float>
    public var padding2: Float = 0

    public var minScale: SIMD3<Float>
    public var padding3: Float = 0

    public var maxScale: SIMD3<Float>
    public var splatCount: UInt32

    public init(from header: ChunkHeader) {
        self.minPosition = header.minPosition
        self.maxPosition = header.maxPosition
        self.minScale = header.minScale
        self.maxScale = header.maxScale
        self.splatCount = UInt32(header.splatCount)
    }
}

/// Compressed splat data (16 bytes)
public struct PackedSplat: Sendable {
    /// Position packed as 11-10-11 bits in chunk-local space
    public var positionPacked: UInt32

    /// Rotation packed as 2-bit selector + 3×10-bit components
    /// Selector indicates which quaternion component is largest
    public var rotationPacked: UInt32

    /// Scale packed as 11-10-11 bits with exponential mapping
    public var scalePacked: UInt32

    /// Color as RGBA8 (each channel 8 bits)
    public var colorPacked: UInt32

    public init(positionPacked: UInt32, rotationPacked: UInt32, scalePacked: UInt32, colorPacked: UInt32) {
        self.positionPacked = positionPacked
        self.rotationPacked = rotationPacked
        self.scalePacked = scalePacked
        self.colorPacked = colorPacked
    }
}

/// Chunk of 256 compressed splats
public struct SplatChunk: Sendable {
    public var header: ChunkHeader
    public var splats: [PackedSplat]

    public init(header: ChunkHeader, splats: [PackedSplat]) {
        self.header = header
        self.splats = splats
    }

    /// Size in bytes of this chunk
    public var sizeInBytes: Int {
        ChunkHeader.baseSize + splats.count * MemoryLayout<PackedSplat>.stride
    }
}

/// Utility for packing/unpacking splat data
public enum SplatCompression {

    /// Standard chunk size
    public static let chunkSize = 256

    /// Pack a position into 11-10-11 bits using chunk bounds
    public static func packPosition(_ position: SIMD3<Float>, min: SIMD3<Float>, max: SIMD3<Float>) -> UInt32 {
        let range = max - min
        let normalized = (position - min) / simd_max(range, SIMD3<Float>(repeating: 0.0001))
        let clamped = simd_clamp(normalized, .zero, .one)

        let x = UInt32(clamped.x * 2047.0) & 0x7FF  // 11 bits
        let y = UInt32(clamped.y * 1023.0) & 0x3FF  // 10 bits
        let z = UInt32(clamped.z * 2047.0) & 0x7FF  // 11 bits

        return (x << 21) | (y << 11) | z
    }

    /// Unpack a position from 11-10-11 bits using chunk bounds
    public static func unpackPosition(_ packed: UInt32, min: SIMD3<Float>, max: SIMD3<Float>) -> SIMD3<Float> {
        let x = Float((packed >> 21) & 0x7FF) / 2047.0
        let y = Float((packed >> 11) & 0x3FF) / 1023.0
        let z = Float(packed & 0x7FF) / 2047.0

        let range = max - min
        return min + SIMD3<Float>(x, y, z) * range
    }

    /// Pack a quaternion rotation into 2-bit selector + 3×10-bit components
    /// Uses smallest-three encoding: drops the largest component (recoverable from unit quaternion)
    public static func packRotation(_ q: simd_quatf) -> UInt32 {
        let qv = q.vector
        let absQ = simd_abs(qv)

        // Find largest component
        var largestIdx = 0
        var largestVal = absQ.x
        if absQ.y > largestVal { largestIdx = 1; largestVal = absQ.y }
        if absQ.z > largestVal { largestIdx = 2; largestVal = absQ.z }
        if absQ.w > largestVal { largestIdx = 3 }

        // Get the three smaller components (normalized to [-1, 1] range of valid quaternion components)
        // Map to [0, 1] then quantize to 10 bits
        var components: [Float] = []
        let sign: Float = qv[largestIdx] < 0 ? -1 : 1
        for i in 0..<4 where i != largestIdx {
            // Normalize to [-0.707, 0.707] range (max value for non-largest components)
            // then map to [0, 1]
            let normalized = (qv[i] * sign / 0.707 + 1.0) * 0.5
            components.append(normalized)
        }

        let a = UInt32(simd_clamp(components[0], 0, 1) * 1023.0) & 0x3FF
        let b = UInt32(simd_clamp(components[1], 0, 1) * 1023.0) & 0x3FF
        let c = UInt32(simd_clamp(components[2], 0, 1) * 1023.0) & 0x3FF

        return (UInt32(largestIdx) << 30) | (a << 20) | (b << 10) | c
    }

    /// Unpack a quaternion from 2-bit selector + 3×10-bit components
    public static func unpackRotation(_ packed: UInt32) -> simd_quatf {
        let largestIdx = Int((packed >> 30) & 0x3)
        let a = Float((packed >> 20) & 0x3FF) / 1023.0
        let b = Float((packed >> 10) & 0x3FF) / 1023.0
        let c = Float(packed & 0x3FF) / 1023.0

        // Map back from [0, 1] to [-0.707, 0.707]
        let components = [
            (a * 2.0 - 1.0) * 0.707,
            (b * 2.0 - 1.0) * 0.707,
            (c * 2.0 - 1.0) * 0.707
        ]

        // Reconstruct largest component
        let sumSq = components[0] * components[0] + components[1] * components[1] + components[2] * components[2]
        let largest = sqrt(max(1.0 - sumSq, 0.0))

        // Build quaternion
        var q = SIMD4<Float>.zero
        var j = 0
        for i in 0..<4 {
            if i == largestIdx {
                q[i] = largest
            } else {
                q[i] = components[j]
                j += 1
            }
        }

        return simd_quatf(vector: q)
    }

    /// Element-wise log for SIMD3<Float>
    private static func log3(_ v: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(log(v.x), log(v.y), log(v.z))
    }

    /// Element-wise exp for SIMD3<Float>
    private static func exp3(_ v: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(exp(v.x), exp(v.y), exp(v.z))
    }

    /// Pack scale into 11-10-11 bits with exponential mapping
    public static func packScale(_ scale: SIMD3<Float>, min: SIMD3<Float>, max: SIMD3<Float>) -> UInt32 {
        // Use log-space mapping for better precision at small scales
        let logMin = log3(simd_max(min, SIMD3<Float>(repeating: 0.0001)))
        let logMax = log3(simd_max(max, SIMD3<Float>(repeating: 0.0001)))
        let logScale = log3(simd_max(scale, SIMD3<Float>(repeating: 0.0001)))

        let logRange = logMax - logMin
        let normalized = (logScale - logMin) / simd_max(logRange, SIMD3<Float>(repeating: 0.0001))
        let clamped = simd_clamp(normalized, .zero, .one)

        let x = UInt32(clamped.x * 2047.0) & 0x7FF
        let y = UInt32(clamped.y * 1023.0) & 0x3FF
        let z = UInt32(clamped.z * 2047.0) & 0x7FF

        return (x << 21) | (y << 11) | z
    }

    /// Unpack scale from 11-10-11 bits with exponential mapping
    public static func unpackScale(_ packed: UInt32, min: SIMD3<Float>, max: SIMD3<Float>) -> SIMD3<Float> {
        let x = Float((packed >> 21) & 0x7FF) / 2047.0
        let y = Float((packed >> 11) & 0x3FF) / 1023.0
        let z = Float(packed & 0x7FF) / 2047.0

        let logMin = log3(simd_max(min, SIMD3<Float>(repeating: 0.0001)))
        let logMax = log3(simd_max(max, SIMD3<Float>(repeating: 0.0001)))

        let logRange = logMax - logMin
        let logScale = logMin + SIMD3<Float>(x, y, z) * logRange

        return exp3(logScale)
    }

    /// Pack RGBA color into 32 bits (8 bits per channel)
    public static func packColor(_ color: SIMD4<Float>) -> UInt32 {
        let r = UInt32(simd_clamp(color.x, 0, 1) * 255.0) & 0xFF
        let g = UInt32(simd_clamp(color.y, 0, 1) * 255.0) & 0xFF
        let b = UInt32(simd_clamp(color.z, 0, 1) * 255.0) & 0xFF
        let a = UInt32(simd_clamp(color.w, 0, 1) * 255.0) & 0xFF

        return (r << 24) | (g << 16) | (b << 8) | a
    }

    /// Unpack RGBA color from 32 bits
    public static func unpackColor(_ packed: UInt32) -> SIMD4<Float> {
        let r = Float((packed >> 24) & 0xFF) / 255.0
        let g = Float((packed >> 16) & 0xFF) / 255.0
        let b = Float((packed >> 8) & 0xFF) / 255.0
        let a = Float(packed & 0xFF) / 255.0

        return SIMD4<Float>(r, g, b, a)
    }
}

/// Compressor for converting full splat data to chunked format
public class SplatChunkCompressor {

    /// Compresses an array of splats into chunks
    public static func compress(
        positions: [SIMD3<Float>],
        rotations: [simd_quatf],
        scales: [SIMD3<Float>],
        colors: [SIMD4<Float>]
    ) -> [SplatChunk] {
        let splatCount = positions.count
        guard splatCount > 0 else { return [] }

        var chunks: [SplatChunk] = []
        let chunkSize = SplatCompression.chunkSize

        for chunkStart in stride(from: 0, to: splatCount, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, splatCount)
            let count = chunkEnd - chunkStart

            // Compute bounds for this chunk
            var minPos = SIMD3<Float>(repeating: .infinity)
            var maxPos = SIMD3<Float>(repeating: -.infinity)
            var minScale = SIMD3<Float>(repeating: .infinity)
            var maxScale = SIMD3<Float>(repeating: -.infinity)

            for i in chunkStart..<chunkEnd {
                minPos = simd_min(minPos, positions[i])
                maxPos = simd_max(maxPos, positions[i])
                minScale = simd_min(minScale, scales[i])
                maxScale = simd_max(maxScale, scales[i])
            }

            // Add small padding to avoid edge cases
            let posPadding = (maxPos - minPos) * 0.001
            minPos -= posPadding
            maxPos += posPadding

            let scalePadding = (maxScale - minScale) * 0.001
            minScale = simd_max(minScale - scalePadding, SIMD3<Float>(repeating: 0.0001))
            maxScale += scalePadding

            // Create header
            let header = ChunkHeader(
                minPosition: minPos,
                maxPosition: maxPos,
                minScale: minScale,
                maxScale: maxScale,
                splatCount: UInt16(count)
            )

            // Pack splats
            var packedSplats: [PackedSplat] = []
            for i in chunkStart..<chunkEnd {
                let packed = PackedSplat(
                    positionPacked: SplatCompression.packPosition(positions[i], min: minPos, max: maxPos),
                    rotationPacked: SplatCompression.packRotation(rotations[i]),
                    scalePacked: SplatCompression.packScale(scales[i], min: minScale, max: maxScale),
                    colorPacked: SplatCompression.packColor(colors[i])
                )
                packedSplats.append(packed)
            }

            chunks.append(SplatChunk(header: header, splats: packedSplats))
        }

        return chunks
    }

    /// Calculates compression ratio
    public static func compressionRatio(originalSize: Int, compressedChunks: [SplatChunk]) -> Float {
        let compressedSize = compressedChunks.reduce(0) { $0 + $1.sizeInBytes }
        guard compressedSize > 0 else { return 0 }
        return Float(originalSize) / Float(compressedSize)
    }
}
