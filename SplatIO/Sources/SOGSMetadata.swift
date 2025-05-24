import Foundation
import simd

// MARK: - SOGS Metadata Structures

public struct SOGSMetadata: Codable {
    let means: SOGSAttributeInfo
    let scales: SOGSAttributeInfo
    let quats: SOGSAttributeInfo
    let sh0: SOGSAttributeInfo
    let shN: SOGSAttributeInfo?
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        means = try container.decode(SOGSAttributeInfo.self, forKey: .means)
        scales = try container.decode(SOGSAttributeInfo.self, forKey: .scales)
        quats = try container.decode(SOGSAttributeInfo.self, forKey: .quats)
        sh0 = try container.decode(SOGSAttributeInfo.self, forKey: .sh0)
        shN = try container.decodeIfPresent(SOGSAttributeInfo.self, forKey: .shN)
    }
}

public struct SOGSAttributeInfo: Codable {
    let shape: [Int]
    let dtype: String
    let files: [String]
    let mins: [Float]?
    let maxs: [Float]?
    let encoding: String?
    let quantization: Int?
    
    // Handle both array and single value for mins/maxs in shN
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        shape = try container.decode([Int].self, forKey: .shape)
        dtype = try container.decode(String.self, forKey: .dtype)
        files = try container.decode([String].self, forKey: .files)
        encoding = try container.decodeIfPresent(String.self, forKey: .encoding)
        quantization = try container.decodeIfPresent(Int.self, forKey: .quantization)
        
        // Handle mins - can be array or single value
        if let minsArray = try? container.decode([Float].self, forKey: .mins) {
            mins = minsArray
        } else if let minsSingle = try? container.decode(Float.self, forKey: .mins) {
            mins = [minsSingle]
        } else {
            mins = nil
        }
        
        // Handle maxs - can be array or single value
        if let maxsArray = try? container.decode([Float].self, forKey: .maxs) {
            maxs = maxsArray
        } else if let maxsSingle = try? container.decode(Float.self, forKey: .maxs) {
            maxs = [maxsSingle]
        } else {
            maxs = nil
        }
    }
}

// MARK: - SOGS Data Structures

public struct SOGSCompressedData {
    let metadata: SOGSMetadata
    let means_l: WebPDecoder.DecodedImage
    let means_u: WebPDecoder.DecodedImage
    let quats: WebPDecoder.DecodedImage
    let scales: WebPDecoder.DecodedImage
    let sh0: WebPDecoder.DecodedImage
    let sh_centroids: WebPDecoder.DecodedImage?
    let sh_labels: WebPDecoder.DecodedImage?
    
    public var numSplats: Int {
        metadata.means.shape[0]
    }
    
    public var shBands: Int {
        guard metadata.shN != nil else { return 0 }
        // Calculate SH bands from the width of centroids texture
        // Based on the original implementation:
        // 192: 1 band (64 * 3), 512: 2 bands (64 * 8), 960: 3 bands (64 * 15)
        let width = sh_centroids?.width ?? 0
        switch width {
        case 192: return 1   // 64 * 3
        case 512: return 2   // 64 * 8  
        case 960: return 3   // 64 * 15
        default: return 0
        }
    }
    
    public var textureWidth: Int {
        means_l.width
    }
    
    public var textureHeight: Int {
        means_l.height
    }
}

// MARK: - SOGS Iterator for decompression

public struct SOGSIterator {
    private let data: SOGSCompressedData
    private let norm: Float = 2.0 / sqrt(2.0)
    private let SH_C0: Float = 0.28209479177387814
    
    public init(_ data: SOGSCompressedData) {
        self.data = data
    }
    
    public func readPoint(at index: Int) -> SplatScenePoint {
        // Convert linear index to 2D texture coordinates
        let textureWidth = data.textureWidth
        let x = index % textureWidth
        let y = index / textureWidth
        
        let position = readPosition(x: x, y: y)
        let rotation = readRotation(x: x, y: y) 
        let scale = readScale(x: x, y: y)
        let (color, opacity) = readColorAndOpacity(x: x, y: y)
        
        return SplatScenePoint(
            position: position,
            color: color,
            opacity: opacity,
            scale: scale,
            rotation: rotation
        )
    }
    
    private func readPosition(x: Int, y: Int) -> SIMD3<Float> {
        // Extract position from means_l and means_u textures
        let metadata = data.metadata.means
        guard let mins = metadata.mins, let maxs = metadata.maxs else {
            return SIMD3<Float>(0, 0, 0)
        }
        
        // Get pixel values from both textures
        let uPixel = WebPDecoder.getPixelFloat(from: data.means_u, x: x, y: y)
        let lPixel = WebPDecoder.getPixelFloat(from: data.means_l, x: x, y: y)
        
        // Reconstruct 16-bit values from 8-bit low and high parts
        let wx = ((uPixel.x * 255.0) * 256.0) + (lPixel.x * 255.0)
        let wy = ((uPixel.y * 255.0) * 256.0) + (lPixel.y * 255.0)
        let wz = ((uPixel.z * 255.0) * 256.0) + (lPixel.z * 255.0)
        
        // Normalize to [0,1] range
        let nx = lerp(mins[0], maxs[0], wx / 65535.0)
        let ny = lerp(mins[1], maxs[1], wy / 65535.0)
        let nz = lerp(mins[2], maxs[2], wz / 65535.0)
        
        // Apply exponential mapping as in the original SOGS implementation
        return SIMD3<Float>(
            sign(nx) * (exp(abs(nx)) - 1),
            sign(ny) * (exp(abs(ny)) - 1), 
            sign(nz) * (exp(abs(nz)) - 1)
        )
    }
    
    private func readRotation(x: Int, y: Int) -> simd_quatf {
        // Decode quaternion from packed format
        let quatPixel = WebPDecoder.getPixelFloat(from: data.quats, x: x, y: y)
        
        let a = (quatPixel.x - 0.5) * norm
        let b = (quatPixel.y - 0.5) * norm
        let c = (quatPixel.z - 0.5) * norm
        let d = sqrt(max(0, 1 - (a * a + b * b + c * c)))
        let mode = UInt32(quatPixel.w * 255.0 + 0.5) - 252
        
        // Reconstruct quaternion based on mode
        switch mode {
        case 0: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        case 1: return simd_quatf(ix: d, iy: b, iz: c, r: a)
        case 2: return simd_quatf(ix: b, iy: d, iz: c, r: a)
        case 3: return simd_quatf(ix: b, iy: c, iz: d, r: a)
        default: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        }
    }
    
    private func readScale(x: Int, y: Int) -> SplatScenePoint.Scale {
        let metadata = data.metadata.scales
        guard let mins = metadata.mins, let maxs = metadata.maxs else {
            return .exponent(SIMD3<Float>(0, 0, 0))
        }
        
        let scalePixel = WebPDecoder.getPixelFloat(from: data.scales, x: x, y: y)
        
        let sx = lerp(mins[0], maxs[0], scalePixel.x)
        let sy = lerp(mins[1], maxs[1], scalePixel.y)
        let sz = lerp(mins[2], maxs[2], scalePixel.z)
        
        return .exponent(SIMD3<Float>(sx, sy, sz))
    }
    
    private func readColorAndOpacity(x: Int, y: Int) -> (SplatScenePoint.Color, SplatScenePoint.Opacity) {
        let metadata = data.metadata.sh0
        guard let mins = metadata.mins, let maxs = metadata.maxs else {
            let color = SplatScenePoint.Color.sphericalHarmonic([SIMD3<Float>(0, 0, 0)])
            let opacity = SplatScenePoint.Opacity.linearFloat(0.5)
            return (color, opacity)
        }
        
        let sh0Pixel = WebPDecoder.getPixelFloat(from: data.sh0, x: x, y: y)
        
        // Extract values from compressed texture
        let r = lerp(mins[0], maxs[0], sh0Pixel.x)
        let g = lerp(mins[1], maxs[1], sh0Pixel.y)
        let b = lerp(mins[2], maxs[2], sh0Pixel.z)
        let a = lerp(mins[3], maxs[3], sh0Pixel.w)
        
        // Convert opacity from logit to linear
        let linearOpacity = 1.0 / (1.0 + exp(-a))
        
        // The compressed data contains SH coefficients
        // Convert to linear color using the SOGS formula: 0.5 + sh * SH_C0
        let linearR = 0.5 + r * SH_C0
        let linearG = 0.5 + g * SH_C0  
        let linearB = 0.5 + b * SH_C0
        
        // Clamp to valid color range
        let clampedR = max(0.0, min(1.0, linearR))
        let clampedG = max(0.0, min(1.0, linearG))
        let clampedB = max(0.0, min(1.0, linearB))
        
        let color = SplatScenePoint.Color.linearFloat(SIMD3<Float>(clampedR, clampedG, clampedB))
        return (color, .linearFloat(linearOpacity))
    }
    
    private func readSphericalHarmonics(x: Int, y: Int, centroids: WebPDecoder.DecodedImage, labels: WebPDecoder.DecodedImage) -> [SIMD3<Float>] {
        let metadata = data.metadata.shN
        guard let shN = metadata,
              let mins = shN.mins?[0],
              let maxs = shN.maxs?[0] else {
            return []
        }
        
        // Extract spherical harmonics palette index
        let labelPixel = WebPDecoder.getPixelFloat(from: labels, x: x, y: y)
        let t = SIMD2<Int>(Int(labelPixel.x * 255.0), Int(labelPixel.y * 255.0))
        let n = t.x + t.y * 256
        let u = (n % 64) * 15
        let v = n / 64
        
        var shCoeffs: [SIMD3<Float>] = []
        
        // Read 15 consecutive texels from the centroids texture
        for i in 0..<15 {
            let centroidPixel = WebPDecoder.getPixelFloat(from: centroids, x: u + i, y: v)
            let coeff = SIMD3<Float>(
                lerp(mins, maxs, centroidPixel.x),
                lerp(mins, maxs, centroidPixel.y),
                lerp(mins, maxs, centroidPixel.z)
            )
            shCoeffs.append(coeff)
        }
        
        return shCoeffs
    }
    
    private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        return a * (1 - t) + b * t
    }
    
    private func sign(_ value: Float) -> Float {
        return value >= 0 ? 1 : -1
    }
} 