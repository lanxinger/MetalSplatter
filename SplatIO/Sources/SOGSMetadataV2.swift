import Foundation
import simd

// MARK: - SOGS v2 Metadata Structures

public struct SOGSMetadataV2: Codable {
    public let version: Int
    public let count: Int
    public let antialias: Bool?
    
    public let means: SOGSMeansInfoV2
    public let scales: SOGSScalesInfoV2
    public let quats: SOGSQuatsInfoV2
    public let sh0: SOGSH0InfoV2
    public let shN: SOGSSHNInfoV2?
    
    public init(version: Int, count: Int, antialias: Bool?, means: SOGSMeansInfoV2, scales: SOGSScalesInfoV2, quats: SOGSQuatsInfoV2, sh0: SOGSH0InfoV2, shN: SOGSSHNInfoV2?) {
        self.version = version
        self.count = count
        self.antialias = antialias
        self.means = means
        self.scales = scales
        self.quats = quats
        self.sh0 = sh0
        self.shN = shN
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        count = try container.decode(Int.self, forKey: .count)
        antialias = try container.decodeIfPresent(Bool.self, forKey: .antialias)
        
        means = try container.decode(SOGSMeansInfoV2.self, forKey: .means)
        scales = try container.decode(SOGSScalesInfoV2.self, forKey: .scales)
        quats = try container.decode(SOGSQuatsInfoV2.self, forKey: .quats)
        sh0 = try container.decode(SOGSH0InfoV2.self, forKey: .sh0)
        shN = try container.decodeIfPresent(SOGSSHNInfoV2.self, forKey: .shN)
    }
}

public struct SOGSMeansInfoV2: Codable {
    public let mins: [Float]  // [xmin', ymin', zmin'] after log transform
    public let maxs: [Float]  // [xmax', ymax', zmax'] after log transform
    public let files: [String] // ["means_l.webp", "means_u.webp"]
    
    public init(mins: [Float], maxs: [Float], files: [String]) {
        self.mins = mins
        self.maxs = maxs
        self.files = files
    }
}

public struct SOGSScalesInfoV2: Codable {
    public let codebook: [Float] // length 256 - k-means codebook for [scale_0, scale_1, scale_2]
    public let files: [String]   // ["scales.webp"] - per-splat byte labels in RGB
    
    public init(codebook: [Float], files: [String]) {
        self.codebook = codebook
        self.files = files
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        files = try container.decode([String].self, forKey: .files)
        
        // Handle null values in codebook by replacing with 0.0
        var codebookArray: [Float] = []
        var arrayContainer = try container.nestedUnkeyedContainer(forKey: .codebook)
        while !arrayContainer.isAtEnd {
            if let value = try? arrayContainer.decode(Float.self) {
                codebookArray.append(value)
            } else {
                // Handle null by consuming the value and appending 0.0
                _ = try? arrayContainer.decodeNil()
                codebookArray.append(0.0)
            }
        }
        codebook = codebookArray
    }
}

public struct SOGSQuatsInfoV2: Codable {
    public let files: [String]   // ["quats.webp"] - orientation texture
    
    public init(files: [String]) {
        self.files = files
    }
}

public struct SOGSH0InfoV2: Codable {
    public let codebook: [Float] // length 256 - DC color codebook for [f_dc_0, f_dc_1, f_dc_2]
    public let files: [String]   // ["sh0.webp"] - per-splat byte labels in RGB; A = opacity (sigmoid*255)
    
    public init(codebook: [Float], files: [String]) {
        self.codebook = codebook
        self.files = files
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        files = try container.decode([String].self, forKey: .files)
        
        // Handle null values in codebook by replacing with 0.0
        var codebookArray: [Float] = []
        var arrayContainer = try container.nestedUnkeyedContainer(forKey: .codebook)
        while !arrayContainer.isAtEnd {
            if let value = try? arrayContainer.decode(Float.self) {
                codebookArray.append(value)
            } else {
                // Handle null by consuming the value and appending 0.0
                _ = try? arrayContainer.decodeNil()
                codebookArray.append(0.0)
            }
        }
        codebook = codebookArray
    }
}

public struct SOGSSHNInfoV2: Codable {
    public let codebook: [Float] // length 256 - 1D codebook built over SH centroids
    public let files: [String]   // ["shN_centroids.webp", "shN_labels.webp"]
    
    public init(codebook: [Float], files: [String]) {
        self.codebook = codebook
        self.files = files
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        files = try container.decode([String].self, forKey: .files)
        
        // Handle null values in codebook by replacing with 0.0
        var codebookArray: [Float] = []
        var arrayContainer = try container.nestedUnkeyedContainer(forKey: .codebook)
        while !arrayContainer.isAtEnd {
            if let value = try? arrayContainer.decode(Float.self) {
                codebookArray.append(value)
            } else {
                // Handle null by consuming the value and appending 0.0
                _ = try? arrayContainer.decodeNil()
                codebookArray.append(0.0)
            }
        }
        codebook = codebookArray
    }
}

// MARK: - SOGS v2 Compressed Data Structure

public struct SOGSCompressedDataV2 {
    public let metadata: SOGSMetadataV2
    public let means_l: WebPDecoder.DecodedImage
    public let means_u: WebPDecoder.DecodedImage
    public let quats: WebPDecoder.DecodedImage
    public let scales: WebPDecoder.DecodedImage
    public let sh0: WebPDecoder.DecodedImage
    public let sh_centroids: WebPDecoder.DecodedImage?
    public let sh_labels: WebPDecoder.DecodedImage?
    
    public init(metadata: SOGSMetadataV2, means_l: WebPDecoder.DecodedImage, means_u: WebPDecoder.DecodedImage, quats: WebPDecoder.DecodedImage, scales: WebPDecoder.DecodedImage, sh0: WebPDecoder.DecodedImage, sh_centroids: WebPDecoder.DecodedImage?, sh_labels: WebPDecoder.DecodedImage?) {
        self.metadata = metadata
        self.means_l = means_l
        self.means_u = means_u
        self.quats = quats
        self.scales = scales
        self.sh0 = sh0
        self.sh_centroids = sh_centroids
        self.sh_labels = sh_labels
    }
    
    public var numSplats: Int {
        metadata.count
    }
    
    public var hasSphericalHarmonics: Bool {
        metadata.shN != nil && sh_centroids != nil && sh_labels != nil
    }
    
    public var textureWidth: Int {
        means_l.width
    }
    
    public var textureHeight: Int {
        means_l.height
    }
}

// MARK: - SOGS v2 Iterator for decompression

public struct SOGSIteratorV2 {
    private let data: SOGSCompressedDataV2
    private let norm: Float = 2.0 / sqrt(2.0)
    private let SH_C0: Float = 0.28209479177387814
    
    // Pre-computed codebooks for performance
    private let scalesCodebook: [Float]  // 256 individual scale values
    private let sh0Codebook: [Float]     // 256 individual color values  
    private let shNCodebook: [Float]?
    
    public init(_ data: SOGSCompressedDataV2) {
        self.data = data
        
        // Pre-process scales codebook - use individual scale values
        // The codebook contains 256 individual scale values, not triplets
        let scaleCodebook = data.metadata.scales.codebook
        if scaleCodebook.count >= 256 {
            self.scalesCodebook = Array(scaleCodebook.prefix(256))
        } else {
            // Pad with zeros if needed
            var scales = scaleCodebook
            while scales.count < 256 {
                scales.append(0.0)
            }
            self.scalesCodebook = scales
        }
        
        // Pre-process sh0 codebook - use individual color values
        // The codebook contains 256 individual color values, not triplets
        let sh0CodebookData = data.metadata.sh0.codebook
        if sh0CodebookData.count >= 256 {
            self.sh0Codebook = Array(sh0CodebookData.prefix(256))
        } else {
            // Pad with zeros if needed
            var colors = sh0CodebookData
            while colors.count < 256 {
                colors.append(0.0)
            }
            self.sh0Codebook = colors
        }
        
        // Store shN codebook if available
        self.shNCodebook = data.metadata.shN?.codebook
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
        // Position decoding remains the same as v1 - uses mins/maxs normalization
        let metadata = data.metadata.means
        let mins = metadata.mins
        let maxs = metadata.maxs
        
        // Get pixel values from both textures as raw bytes
        let uPixel = WebPDecoder.getPixelUInt8(from: data.means_u, x: x, y: y)
        let lPixel = WebPDecoder.getPixelUInt8(from: data.means_l, x: x, y: y)
        
        // Reconstruct 16-bit values from 8-bit low and high parts
        let wx = (UInt16(uPixel.x) << 8) | UInt16(lPixel.x)
        let wy = (UInt16(uPixel.y) << 8) | UInt16(lPixel.y)
        let wz = (UInt16(uPixel.z) << 8) | UInt16(lPixel.z)
        
        // Normalize to [0,1] range and apply log-space denormalization
        let nx = lerp(mins[0], maxs[0], Float(wx) / 65535.0)
        let ny = lerp(mins[1], maxs[1], Float(wy) / 65535.0)
        let nz = lerp(mins[2], maxs[2], Float(wz) / 65535.0)
        
        // Apply log transform inverse: sign(v)*exp(|v|)-1
        return SIMD3<Float>(
            sign(nx) * (exp(abs(nx)) - 1),
            sign(ny) * (exp(abs(ny)) - 1), 
            sign(nz) * (exp(abs(nz)) - 1)
        )
    }
    
    private func readRotation(x: Int, y: Int) -> simd_quatf {
        // Quaternion decoding remains the same as v1
        let quatPixel = WebPDecoder.getPixelUInt8(from: data.quats, x: x, y: y)
        
        let a = (Float(quatPixel.x) / 255.0 - 0.5) * norm
        let b = (Float(quatPixel.y) / 255.0 - 0.5) * norm
        let c = (Float(quatPixel.z) / 255.0 - 0.5) * norm
        let d = sqrt(max(0, 1 - (a * a + b * b + c * c)))
        let mode = UInt32(quatPixel.w) - 252
        
        switch mode {
        case 0: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        case 1: return simd_quatf(ix: d, iy: b, iz: c, r: a)
        case 2: return simd_quatf(ix: b, iy: d, iz: c, r: a)
        case 3: return simd_quatf(ix: b, iy: c, iz: d, r: a)
        default: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        }
    }
    
    private func readScale(x: Int, y: Int) -> SplatScenePoint.Scale {
        // NEW v2: Use codebook lookup with separate indices per component
        let scalePixel = WebPDecoder.getPixelUInt8(from: data.scales, x: x, y: y)
        
        // According to spec: RGB channels contain separate indices for [scale_0, scale_1, scale_2]
        let scaleXIndex = Int(scalePixel.x)
        let scaleYIndex = Int(scalePixel.y)
        let scaleZIndex = Int(scalePixel.z)
        
        // Look up individual scale components from codebook
        let scaleX = scalesCodebook[scaleXIndex]
        let scaleY = scalesCodebook[scaleYIndex]
        let scaleZ = scalesCodebook[scaleZIndex]
        
        return .exponent(SIMD3<Float>(scaleX, scaleY, scaleZ))
    }
    
    private func readColorAndOpacity(x: Int, y: Int) -> (SplatScenePoint.Color, SplatScenePoint.Opacity) {
        // NEW v2: Use codebook lookup with separate indices per component
        let sh0Pixel = WebPDecoder.getPixelUInt8(from: data.sh0, x: x, y: y)
        
        // According to spec: RGB channels contain separate indices for [f_dc_0, f_dc_1, f_dc_2]
        let colorRIndex = Int(sh0Pixel.x)
        let colorGIndex = Int(sh0Pixel.y)
        let colorBIndex = Int(sh0Pixel.z)
        
        // Look up individual color components from codebook
        let colorR = sh0Codebook[colorRIndex]
        let colorG = sh0Codebook[colorGIndex]
        let colorB = sh0Codebook[colorBIndex]
        
        // Extract opacity directly from alpha channel (sigmoid mapped)
        let opacityValue = Float(sh0Pixel.w) / 255.0
        
        // Create base SH coefficients
        let colorTriplet = SIMD3<Float>(colorR, colorG, colorB)
        let sh0Coeffs = [colorTriplet]
        
        // Handle spherical harmonics if available
        let color: SplatScenePoint.Color
        if data.hasSphericalHarmonics {
            let additionalSH = readSphericalHarmonics(x: x, y: y)
            var allCoeffs = sh0Coeffs
            allCoeffs.append(contentsOf: additionalSH)
            color = .sphericalHarmonic(allCoeffs)
        } else {
            // Convert DC coefficients to linear color
            color = .linearFloat(SIMD3<Float>(
                max(0.0, min(1.0, 0.5 + colorTriplet.x * SH_C0)),
                max(0.0, min(1.0, 0.5 + colorTriplet.y * SH_C0)),
                max(0.0, min(1.0, 0.5 + colorTriplet.z * SH_C0))
            ))
        }
        
        return (color, .linearFloat(opacityValue))
    }
    
    private func readSphericalHarmonics(x: Int, y: Int) -> [SIMD3<Float>] {
        guard let sh_centroids = data.sh_centroids,
              let sh_labels = data.sh_labels,
              let codebook = shNCodebook else {
            return []
        }
        
        // Extract palette index from labels texture (uint16 little-endian in RG)
        let labelPixel = WebPDecoder.getPixelUInt8(from: sh_labels, x: x, y: y)
        let paletteIndex = Int(labelPixel.x) + Int(labelPixel.y) * 256
        
        // Calculate centroid texture coordinates
        let u = (paletteIndex % 64) * 15  // 15 coefficients per centroid
        let v = paletteIndex / 64
        
        var shCoeffs: [SIMD3<Float>] = []
        shCoeffs.reserveCapacity(15)
        
        // Read 15 consecutive coefficients from centroids texture
        for i in 0..<15 {
            let centroidPixel = WebPDecoder.getPixelFloat(from: sh_centroids, x: u + i, y: v)
            
            // Use codebook to decode each coefficient component
            let r = codebook[Int(centroidPixel.x * 255) % 256]
            let g = codebook[Int(centroidPixel.y * 255) % 256]
            let b = codebook[Int(centroidPixel.z * 255) % 256]
            
            shCoeffs.append(SIMD3<Float>(r, g, b))
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

// MARK: - SOGS v2 Batch Iterator for Performance

public struct SOGSBatchIteratorV2 {
    private let data: SOGSCompressedDataV2
    private let iterator: SOGSIteratorV2
    private let textureWidth: Int
    
    public init(_ data: SOGSCompressedDataV2) {
        self.data = data
        self.iterator = SOGSIteratorV2(data)
        self.textureWidth = data.textureWidth
    }
    
    /// Batch process a range of points for better performance
    public func readBatch(startIndex: Int, count: Int) -> [SplatScenePoint] {
        let endIndex = min(startIndex + count, data.numSplats)
        var points = [SplatScenePoint]()
        points.reserveCapacity(endIndex - startIndex)
        
        for index in startIndex..<endIndex {
            let point = iterator.readPoint(at: index)
            points.append(point)
        }
        
        return points
    }
}