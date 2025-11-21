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
    public let codebook: [Float]
    public let mins: [Float]?
    public let maxs: [Float]?
    public let files: [String]

    enum CodingKeys: String, CodingKey {
        case codebook
        case mins
        case maxs
        case files
    }

    public init(codebook: [Float], mins: [Float]?, maxs: [Float]?, files: [String]) {
        self.codebook = codebook
        self.mins = mins
        self.maxs = maxs
        self.files = files
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        files = try container.decode([String].self, forKey: .files)
        codebook = try decodeFloatArrayIfPresent(in: container, forKey: .codebook) ?? []
        mins = try decodeFloatArrayIfPresent(in: container, forKey: .mins)
        maxs = try decodeFloatArrayIfPresent(in: container, forKey: .maxs)
    }
}

public struct SOGSQuatsInfoV2: Codable {
    public let files: [String]   // ["quats.webp"] - orientation texture
    
    public init(files: [String]) {
        self.files = files
    }
}

public struct SOGSH0InfoV2: Codable {
    public let codebook: [Float]
    public let mins: [Float]?
    public let maxs: [Float]?
    public let files: [String]

    enum CodingKeys: String, CodingKey {
        case codebook
        case mins
        case maxs
        case files
    }

    public init(codebook: [Float], mins: [Float]?, maxs: [Float]?, files: [String]) {
        self.codebook = codebook
        self.mins = mins
        self.maxs = maxs
        self.files = files
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        files = try container.decode([String].self, forKey: .files)
        codebook = try decodeFloatArrayIfPresent(in: container, forKey: .codebook) ?? []
        mins = try decodeFloatArrayIfPresent(in: container, forKey: .mins)
        maxs = try decodeFloatArrayIfPresent(in: container, forKey: .maxs)
    }
}

public struct SOGSSHNInfoV2: Codable {
    public let count: Int?       // Palette size (entries)
    public let bands: Int?       // Number of SH bands (1...3)
    public let codebook: [Float]
    public let mins: [Float]?
    public let maxs: [Float]?
    public let files: [String]   // File names for labels / centroids textures

    enum CodingKeys: String, CodingKey {
        case count
        case bands
        case codebook
        case mins
        case maxs
        case files
    }

    public init(count: Int?, bands: Int?, codebook: [Float], mins: [Float]?, maxs: [Float]?, files: [String]) {
        self.count = count
        self.bands = bands
        self.codebook = codebook
        self.mins = mins
        self.maxs = maxs
        self.files = files
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        count = try container.decodeIfPresent(Int.self, forKey: .count)
        bands = try container.decodeIfPresent(Int.self, forKey: .bands)
        files = try container.decode([String].self, forKey: .files)
        codebook = try decodeFloatArrayIfPresent(in: container, forKey: .codebook) ?? []
        mins = try decodeFloatArrayIfPresent(in: container, forKey: .mins)
        maxs = try decodeFloatArrayIfPresent(in: container, forKey: .maxs)
    }

    /// Helper for computing the expected coefficient count per palette entry based on bands.
    public var coefficientsPerEntry: Int? {
        guard let bands else { return nil }
        switch bands {
        case 1: return 3
        case 2: return 8
        case 3: return 15
        default: return nil
        }
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
    private let sh0AlphaMin: Float?
    private let sh0AlphaMax: Float?
    private let shNCodebook: [Float]?
    private let shNMin: Float?
    private let shNMax: Float?
    private let shNCoefficientCountHint: Int?
    private let shNPaletteCountHint: Int?
    
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
        
        // Alpha is stored as a logit float (same as v1); preserve any provided range for proper sigmoid decoding
        if let mins = data.metadata.sh0.mins, mins.count > 3,
           let maxs = data.metadata.sh0.maxs, maxs.count > 3 {
            self.sh0AlphaMin = mins[3]
            self.sh0AlphaMax = maxs[3]
        } else {
            self.sh0AlphaMin = nil
            self.sh0AlphaMax = nil
        }
        
        // Store shN codebook if available
        if let shNInfo = data.metadata.shN {
            let shNCodebookData = shNInfo.codebook
            self.shNCodebook = shNCodebookData.isEmpty ? nil : Array(shNCodebookData.prefix(256))

            if let coefficientHint = shNInfo.coefficientsPerEntry, coefficientHint > 0 {
                self.shNCoefficientCountHint = coefficientHint
            } else {
                self.shNCoefficientCountHint = nil
            }

            if let count = shNInfo.count, count > 0 {
                self.shNPaletteCountHint = count
            } else {
                self.shNPaletteCountHint = nil
            }

            if let mins = shNInfo.mins?.first, let maxs = shNInfo.maxs?.first {
                self.shNMin = mins
                self.shNMax = maxs
            } else {
                self.shNMin = nil
                self.shNMax = nil
            }
        } else {
            self.shNCodebook = nil
            self.shNMin = nil
            self.shNMax = nil
            self.shNCoefficientCountHint = nil
            self.shNPaletteCountHint = nil
        }
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
        
        // Alpha channel stores logit-encoded opacity; apply optional range then sigmoid to get linear opacity
        let normalizedOpacity = Float(sh0Pixel.w) / 255.0
        let logitOpacity: Float
        if let min = sh0AlphaMin, let max = sh0AlphaMax {
            logitOpacity = lerp(min, max, normalizedOpacity)
        } else {
            logitOpacity = normalizedOpacity
        }
        let opacityValue = 1.0 / (1.0 + exp(-logitOpacity))
        
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
              let sh_labels = data.sh_labels else {
            return []
        }

        // Extract palette index from labels texture (uint16 little-endian in RG)
        let labelPixel = WebPDecoder.getPixelUInt8(from: sh_labels, x: x, y: y)
        let paletteIndex = Int(labelPixel.x) + (Int(labelPixel.y) << 8)

        // Texture stores 64 entries per row
        guard sh_centroids.width % 64 == 0,
              sh_centroids.height > 0 else {
            return []
        }

        let coefficientsPerEntry = sh_centroids.width / 64
        guard coefficientsPerEntry > 0 else {
            return []
        }

        // Derive palette count: prefer metadata when valid, otherwise fall back to texture rows
        let inferredPaletteCount = sh_centroids.height * 64
        let paletteCountHint = shNPaletteCountHint ?? 0
        let paletteCount = paletteCountHint > 0 ? min(paletteCountHint, inferredPaletteCount) : inferredPaletteCount
        guard paletteCount > 0 else { return [] }
        guard paletteIndex >= 0 && paletteIndex < paletteCount else {
            return []
        }

        // Calculate centroid texture coordinates
        let u = (paletteIndex % 64) * coefficientsPerEntry
        let v = paletteIndex / 64
        guard v < sh_centroids.height else { return [] }

        var shCoeffs: [SIMD3<Float>] = []
        shCoeffs.reserveCapacity(coefficientsPerEntry)

        for i in 0..<coefficientsPerEntry {
            let sampleX = u + i
            guard sampleX < sh_centroids.width else { break }
            let centroidPixel = WebPDecoder.getPixelUInt8(from: sh_centroids, x: sampleX, y: v)

            let normalized = SIMD3<Float>(
                Float(centroidPixel.x) / 255.0,
                Float(centroidPixel.y) / 255.0,
                Float(centroidPixel.z) / 255.0
            )

            if let minVal = shNMin, let maxVal = shNMax {
                let coeff = SIMD3<Float>(
                    lerp(minVal, maxVal, normalized.x),
                    lerp(minVal, maxVal, normalized.y),
                    lerp(minVal, maxVal, normalized.z)
                )
                shCoeffs.append(coeff)
            } else if let codebook = shNCodebook, codebook.count >= 256 {
                let rIndex = Int(centroidPixel.x)
                let gIndex = Int(centroidPixel.y)
                let bIndex = Int(centroidPixel.z)

                if rIndex < codebook.count,
                   gIndex < codebook.count,
                   bIndex < codebook.count {
                    shCoeffs.append(SIMD3<Float>(
                        codebook[rIndex],
                        codebook[gIndex],
                        codebook[bIndex]
                    ))
                } else {
                    shCoeffs.append(.zero)
                }
            } else {
                shCoeffs.append(.zero)
            }
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

@inline(__always)
fileprivate func decodeFloatArrayIfPresent<T: CodingKey>(in container: KeyedDecodingContainer<T>, forKey key: T) throws -> [Float]? {
    guard container.contains(key) else { return nil }
    var nested = try container.nestedUnkeyedContainer(forKey: key)
    var values: [Float] = []
    while !nested.isAtEnd {
        if let value = try? nested.decode(Float.self) {
            values.append(value)
        } else if (try? nested.decodeNil()) == true {
            values.append(0.0)
        }
    }
    return values
}
