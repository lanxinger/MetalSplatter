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
        let (baseColor, opacity, sh0Coeffs) = readColorAndOpacity(x: x, y: y)
        
        // Read additional spherical harmonics if available
        var color = baseColor
        if data.shBands > 0,
           let sh_centroids = data.sh_centroids,
           let sh_labels = data.sh_labels {
            let shCoeffs = readSphericalHarmonics(x: x, y: y, centroids: sh_centroids, labels: sh_labels)
            // Combine base SH coefficients with additional bands
            var allCoeffs = sh0Coeffs
            allCoeffs.append(contentsOf: shCoeffs)
            color = SplatScenePoint.Color.sphericalHarmonic(allCoeffs)
        }
        
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
        
        // Get pixel values from both textures as raw bytes
        let uPixel = WebPDecoder.getPixelUInt8(from: data.means_u, x: x, y: y)
        let lPixel = WebPDecoder.getPixelUInt8(from: data.means_l, x: x, y: y)
        
        // Reconstruct 16-bit values from 8-bit low and high parts using proper bit manipulation
        let wx = (UInt16(uPixel.x) << 8) | UInt16(lPixel.x)
        let wy = (UInt16(uPixel.y) << 8) | UInt16(lPixel.y)
        let wz = (UInt16(uPixel.z) << 8) | UInt16(lPixel.z)
        
        // Normalize to [0,1] range
        let nx = lerp(mins[0], maxs[0], Float(wx) / 65535.0)
        let ny = lerp(mins[1], maxs[1], Float(wy) / 65535.0)
        let nz = lerp(mins[2], maxs[2], Float(wz) / 65535.0)
        
        // Apply exponential mapping as in the original SOGS implementation
        return SIMD3<Float>(
            sign(nx) * (exp(abs(nx)) - 1),
            sign(ny) * (exp(abs(ny)) - 1), 
            sign(nz) * (exp(abs(nz)) - 1)
        )
    }
    
    private func readRotation(x: Int, y: Int) -> simd_quatf {
        // Decode quaternion from packed format - use raw integer data to match JavaScript reference exactly
        let quatPixel = WebPDecoder.getPixelUInt8(from: data.quats, x: x, y: y)
        
        let a = (Float(quatPixel.x) / 255.0 - 0.5) * norm
        let b = (Float(quatPixel.y) / 255.0 - 0.5) * norm
        let c = (Float(quatPixel.z) / 255.0 - 0.5) * norm
        let d = sqrt(max(0, 1 - (a * a + b * b + c * c)))
        let mode = UInt32(quatPixel.w) - 252  // Direct integer access - matches JavaScript exactly
        
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
    
    private func readColorAndOpacity(x: Int, y: Int) -> (SplatScenePoint.Color, SplatScenePoint.Opacity, [SIMD3<Float>]) {
        let metadata = data.metadata.sh0
        guard let mins = metadata.mins, let maxs = metadata.maxs else {
            let color = SplatScenePoint.Color.sphericalHarmonic([SIMD3<Float>(0, 0, 0)])
            let opacity = SplatScenePoint.Opacity.linearFloat(0.5)
            return (color, opacity, [SIMD3<Float>(0, 0, 0)])
        }
        
        let sh0Pixel = WebPDecoder.getPixelFloat(from: data.sh0, x: x, y: y)
        
        // Extract values from compressed texture
        let r = lerp(mins[0], maxs[0], sh0Pixel.x)
        let g = lerp(mins[1], maxs[1], sh0Pixel.y)
        let b = lerp(mins[2], maxs[2], sh0Pixel.z)
        let a = lerp(mins[3], maxs[3], sh0Pixel.w)
        
        // Convert opacity from logit to linear
        let linearOpacity = 1.0 / (1.0 + exp(-a))
        
        // Store the raw SH coefficients (DC term)
        let sh0Coeffs = [SIMD3<Float>(r, g, b)]
        
        // For now, return as spherical harmonic color with just the DC term
        // If no additional SH bands, this will be converted to linear color in the caller
        let color = data.shBands > 0 ? 
            SplatScenePoint.Color.sphericalHarmonic(sh0Coeffs) :
            SplatScenePoint.Color.linearFloat(SIMD3<Float>(
                max(0.0, min(1.0, 0.5 + r * SH_C0)),
                max(0.0, min(1.0, 0.5 + g * SH_C0)),
                max(0.0, min(1.0, 0.5 + b * SH_C0))
            ))
        
        return (color, .linearFloat(linearOpacity), sh0Coeffs)
    }
    
    private func readSphericalHarmonics(x: Int, y: Int, centroids: WebPDecoder.DecodedImage, labels: WebPDecoder.DecodedImage) -> [SIMD3<Float>] {
        let metadata = data.metadata.shN
        guard let shN = metadata,
              let mins = shN.mins?[0],
              let maxs = shN.maxs?[0] else {
            return []
        }
        
        // Extract spherical harmonics palette index - use raw integer data to match JavaScript exactly
        let labelPixel = WebPDecoder.getPixelUInt8(from: labels, x: x, y: y)
        let t = SIMD2<Int>(Int(labelPixel.x), Int(labelPixel.y))  // Direct integer access
        let n = t.x + t.y * 256  // Same as (t.x + (t.y << 8))
        let u = (n % 64) * 15
        let v = n / 64
        
        var shCoeffs: [SIMD3<Float>] = []
        
        // Read 15 consecutive texels from the centroids texture (keep float for interpolation)
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