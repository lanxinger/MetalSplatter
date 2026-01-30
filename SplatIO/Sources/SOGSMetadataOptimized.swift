import Foundation
import simd

// MARK: - Optimized SOGS Iterator with Batch Processing

public struct SOGSBatchIterator {
    private let data: SOGSCompressedData
    private let norm: Float = 2.0 / sqrt(2.0)
    private let SH_C0: Float = 0.28209479177387814
    
    private let textureWidth: Int
    
    public init(_ data: SOGSCompressedData) {
        self.data = data
        self.textureWidth = data.textureWidth
    }
    
    /// Batch process a range of points for better performance
    public func readBatch(startIndex: Int, count: Int) -> [SplatScenePoint] {
        let endIndex = min(startIndex + count, data.numSplats)
        var points = [SplatScenePoint]()
        points.reserveCapacity(endIndex - startIndex)
        
        // Pre-fetch metadata for batch processing
        let metadata = data.metadata
        let positionMins = metadata.means.mins ?? [0, 0, 0]
        let positionMaxs = metadata.means.maxs ?? [1, 1, 1]
        let scaleMins = metadata.scales.mins ?? [0, 0, 0]
        let scaleMaxs = metadata.scales.maxs ?? [1, 1, 1]
        let sh0Mins = metadata.sh0.mins ?? [0, 0, 0, 0]
        let sh0Maxs = metadata.sh0.maxs ?? [1, 1, 1, 1]
        
        // Process points in chunks for better cache locality
        let chunkSize = 256 // Process 256 points at a time for optimal cache usage
        var chunkStart = startIndex
        
        while chunkStart < endIndex {
            let chunkEnd = min(chunkStart + chunkSize, endIndex)
            
            // Process chunk
            for index in chunkStart..<chunkEnd {
                let x = index % textureWidth
                let y = index / textureWidth
                
                // Read position
                let position = readPositionDirect(
                    x: x, y: y,
                    mins: positionMins,
                    maxs: positionMaxs
                )
                
                // Read rotation
                let rotation = readRotationDirect(x: x, y: y)
                
                // Read scale
                let scale = readScaleDirect(
                    x: x, y: y,
                    mins: scaleMins,
                    maxs: scaleMaxs
                )
                
                // Read color and opacity
                let (color, opacity) = readColorAndOpacityDirect(
                    x: x, y: y,
                    pixelOffset: 0, // Not used anymore
                    mins: sh0Mins,
                    maxs: sh0Maxs
                )
                
                points.append(SplatScenePoint(
                    position: position,
                    color: color,
                    opacity: opacity,
                    scale: scale,
                    rotation: rotation
                ))
            }
            
            chunkStart = chunkEnd
        }
        
        return points
    }
    
    @inline(__always)
    private func readPositionDirect(x: Int, y: Int, mins: [Float], maxs: [Float]) -> SIMD3<Float> {
        // Use WebPDecoder to match original implementation exactly
        let uPixel = WebPDecoder.getPixelUInt8(from: data.means_u, x: x, y: y)
        let lPixel = WebPDecoder.getPixelUInt8(from: data.means_l, x: x, y: y)
        
        // Reconstruct 16-bit values
        let wx = (UInt16(uPixel.x) << 8) | UInt16(lPixel.x)
        let wy = (UInt16(uPixel.y) << 8) | UInt16(lPixel.y)
        let wz = (UInt16(uPixel.z) << 8) | UInt16(lPixel.z)
        
        // Normalize and apply exponential mapping
        let nx = lerp(mins[0], maxs[0], Float(wx) / 65535.0)
        let ny = lerp(mins[1], maxs[1], Float(wy) / 65535.0)
        let nz = lerp(mins[2], maxs[2], Float(wz) / 65535.0)
        
        return SIMD3<Float>(
            sign(nx) * (exp(abs(nx)) - 1),
            sign(ny) * (exp(abs(ny)) - 1),
            sign(nz) * (exp(abs(nz)) - 1)
        )
    }
    
    @inline(__always)
    private func readRotationDirect(x: Int, y: Int) -> simd_quatf {
        let quatPixel = WebPDecoder.getPixelUInt8(from: data.quats, x: x, y: y)
        let quatX = quatPixel.x
        let quatY = quatPixel.y
        let quatZ = quatPixel.z
        let quatW = quatPixel.w
        
        let a = (Float(quatX) / 255.0 - 0.5) * norm
        let b = (Float(quatY) / 255.0 - 0.5) * norm
        let c = (Float(quatZ) / 255.0 - 0.5) * norm
        let d = sqrt(max(0, 1 - (a * a + b * b + c * c)))
        let mode = UInt32(quatW) - 252
        
        switch mode {
        case 0: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        case 1: return simd_quatf(ix: d, iy: b, iz: c, r: a)
        case 2: return simd_quatf(ix: b, iy: d, iz: c, r: a)
        case 3: return simd_quatf(ix: b, iy: c, iz: d, r: a)
        default: return simd_quatf(ix: a, iy: b, iz: c, r: d)
        }
    }
    
    @inline(__always)
    private func readScaleDirect(x: Int, y: Int, mins: [Float], maxs: [Float]) -> SplatScenePoint.Scale {
        // Use WebPDecoder to match original implementation
        let scalePixel = WebPDecoder.getPixelFloat(from: data.scales, x: x, y: y)
        let scaleX = scalePixel.x
        let scaleY = scalePixel.y
        let scaleZ = scalePixel.z
        
        let sx = lerp(mins[0], maxs[0], scaleX)
        let sy = lerp(mins[1], maxs[1], scaleY)
        let sz = lerp(mins[2], maxs[2], scaleZ)
        
        return .exponent(SIMD3<Float>(sx, sy, sz))
    }
    
    @inline(__always)
    private func readColorAndOpacityDirect(x: Int, y: Int, pixelOffset: Int, mins: [Float], maxs: [Float]) -> (SplatScenePoint.Color, SplatScenePoint.Opacity) {
        // Use WebPDecoder to properly handle premultiplied alpha (matching original implementation)
        let sh0Pixel = WebPDecoder.getPixelFloat(from: data.sh0, x: x, y: y)
        
        let r = lerp(mins[0], maxs[0], sh0Pixel.x)
        let g = lerp(mins[1], maxs[1], sh0Pixel.y)
        let b = lerp(mins[2], maxs[2], sh0Pixel.z)
        let a = lerp(mins[3], maxs[3], sh0Pixel.w)
        
        // Convert opacity from logit to linear
        let linearOpacity = 1.0 / (1.0 + exp(-a))
        
        // Store the raw SH coefficients (DC term)
        let sh0Coeffs = [SIMD3<Float>(r, g, b)]
        
        // Handle spherical harmonics like the original implementation
        let color: SplatScenePoint.Color
        if data.shBands > 0,
           data.sh_centroids != nil,
           data.sh_labels != nil {
            let shCoeffs = readSphericalHarmonicsDirect(x: x, y: y)
            
            // Combine base SH with additional bands (matching original implementation)
            var allCoeffs = sh0Coeffs
            allCoeffs.append(contentsOf: shCoeffs)
            color = .sphericalHarmonic(allCoeffs)
        } else {
            // Convert to linear color if no SH bands
            color = .linearFloat(SIMD3<Float>(
                max(0.0, min(1.0, 0.5 + r * SH_C0)),
                max(0.0, min(1.0, 0.5 + g * SH_C0)),
                max(0.0, min(1.0, 0.5 + b * SH_C0))
            ))
        }
        
        return (color, .linearFloat(linearOpacity))
    }
    
    @inline(__always)
    private func readSphericalHarmonicsDirect(x: Int, y: Int) -> [SIMD3<Float>] {
        guard let shN = data.metadata.shN,
              let mins = shN.mins?[0],
              let maxs = shN.maxs?[0] else {
            return []
        }
        
        // Extract label using WebPDecoder
        let labelPixel = WebPDecoder.getPixelUInt8(from: data.sh_labels!, x: x, y: y)
        let n = Int(labelPixel.x) + Int(labelPixel.y) * 256
        let u = (n % 64) * 15
        let v = n / 64
        
        var shCoeffs: [SIMD3<Float>] = []
        shCoeffs.reserveCapacity(15)
        
        // Read 15 consecutive texels from centroids using WebPDecoder
        for i in 0..<15 {
            let centroidX = u + i
            let centroidY = v
            let centroidPixel = WebPDecoder.getPixelFloat(from: data.sh_centroids!, x: centroidX, y: centroidY)
            
            let coeff = SIMD3<Float>(
                lerp(mins, maxs, centroidPixel.x),
                lerp(mins, maxs, centroidPixel.y),
                lerp(mins, maxs, centroidPixel.z)
            )
            shCoeffs.append(coeff)
        }
        
        return shCoeffs
    }
    
    @inline(__always)
    private func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        return a * (1 - t) + b * t
    }
    
    @inline(__always)
    private func sign(_ value: Float) -> Float {
        return value >= 0 ? 1 : -1
    }
}

// MARK: - Parallel WebP Loading Extension

extension SplatSOGSSceneReader {
    /// Load WebP textures in parallel for better performance
    internal func loadCompressedDataParallel(metadata: SOGSMetadata) throws -> SOGSCompressedData {
        print("SplatSOGSSceneReader: Loading WebP textures in parallel...")
        
        // Use concurrent queue for parallel loading
        let queue = DispatchQueue(label: "sogs.webp.loading", attributes: .concurrent)
        let group = DispatchGroup()
        
        // Storage for results
        var means_l: WebPDecoder.DecodedImage?
        var means_u: WebPDecoder.DecodedImage?
        var quats: WebPDecoder.DecodedImage?
        var scales: WebPDecoder.DecodedImage?
        var sh0: WebPDecoder.DecodedImage?
        var sh_centroids: WebPDecoder.DecodedImage?
        var sh_labels: WebPDecoder.DecodedImage?
        
        // Errors from parallel operations
        var loadingErrors: [Error] = []
        let errorLock = NSLock()
        
        // Load required textures in parallel
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                means_l = try self.loadAndDecodeWebP(metadata.means.files[0])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                means_u = try self.loadAndDecodeWebP(metadata.means.files[1])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                quats = try self.loadAndDecodeWebP(metadata.quats.files[0])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                scales = try self.loadAndDecodeWebP(metadata.scales.files[0])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                sh0 = try self.loadAndDecodeWebP(metadata.sh0.files[0])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load optional SH textures if needed
        if let shN = metadata.shN, shN.files.count >= 2 {
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    let tempCentroids = try self.loadAndDecodeWebP(shN.files[0])
                    let shBands = self.calculateSHBands(width: tempCentroids.width)
                    if shBands > 0 {
                        sh_centroids = tempCentroids
                    }
                } catch {
                    errorLock.lock()
                    loadingErrors.append(error)
                    errorLock.unlock()
                }
            }
            
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    sh_labels = try self.loadAndDecodeWebP(shN.files[1])
                } catch {
                    errorLock.lock()
                    loadingErrors.append(error)
                    errorLock.unlock()
                }
            }
        }
        
        // Wait for all operations to complete
        group.wait()
        
        // Check for errors
        if !loadingErrors.isEmpty {
            throw loadingErrors.first!
        }
        
        // Verify all required textures loaded
        guard let means_l = means_l,
              let means_u = means_u,
              let quats = quats,
              let scales = scales,
              let sh0 = sh0 else {
            throw SOGSError.webpDecodingFailed("Failed to load required textures")
        }
        
        print("SplatSOGSSceneReader: Successfully loaded all WebP textures in parallel")
        
        return SOGSCompressedData(
            metadata: metadata,
            means_l: means_l,
            means_u: means_u,
            quats: quats,
            scales: scales,
            sh0: sh0,
            sh_centroids: sh_centroids,
            sh_labels: sh_labels
        )
    }
}

// MARK: - Texture Cache for Repeated Access

/// Thread-safe LRU cache for SOGS compressed texture data.
///
/// Thread Safety:
/// - All cache operations are protected by an internal NSLock.
/// - The loader closure in `getCompressedData` is called outside the lock to avoid
///   holding the lock during potentially slow I/O operations. This means concurrent
///   requests for the same uncached URL may result in duplicate loading work (but
///   the cache will still be consistent).
/// - Marked as `@unchecked Sendable` because thread safety is enforced via NSLock.
public final class SOGSTextureCache: @unchecked Sendable {
    private var cache: [URL: SOGSCompressedData] = [:]
    private let lock = NSLock()
    private let maxCacheSize = 5 // Maximum number of cached SOGS scenes
    private var accessOrder: [URL] = [] // Track LRU

    public static let shared = SOGSTextureCache()
    
    private init() {}
    
    /// Get cached compressed data or load if not cached
    public func getCompressedData(for metaURL: URL, loader: () throws -> SOGSCompressedData) throws -> SOGSCompressedData {
        // Phase 1: Check cache (locked)
        lock.lock()
        if let cached = cache[metaURL] {
            // Move to end for LRU tracking
            if let index = accessOrder.firstIndex(of: metaURL) {
                accessOrder.remove(at: index)
            }
            accessOrder.append(metaURL)
            lock.unlock()
            print("SOGSTextureCache: Using cached data for \(metaURL.lastPathComponent)")
            return cached
        }
        lock.unlock()

        // Phase 2: Load new data (unlocked - allows concurrent loading)
        // Note: If loader() throws, we simply propagate the error without touching the lock
        let compressedData = try loader()

        // Phase 3: Store in cache (locked)
        lock.lock()
        defer { lock.unlock() }

        // Re-check if another thread cached while we were loading
        if let cached = cache[metaURL] {
            print("SOGSTextureCache: Using data cached by another thread for \(metaURL.lastPathComponent)")
            return cached
        }

        // Add to cache
        cache[metaURL] = compressedData
        accessOrder.append(metaURL)

        // Evict oldest if cache is full
        if cache.count > maxCacheSize {
            if let oldest = accessOrder.first {
                cache.removeValue(forKey: oldest)
                accessOrder.removeFirst()
                print("SOGSTextureCache: Evicted \(oldest.lastPathComponent) from cache")
            }
        }

        print("SOGSTextureCache: Cached data for \(metaURL.lastPathComponent)")
        return compressedData
    }
    
    /// Clear the entire cache
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeAll()
        accessOrder.removeAll()
        print("SOGSTextureCache: Cache cleared")
    }
    
    /// Remove specific entry from cache
    public func evict(metaURL: URL) {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeValue(forKey: metaURL)
        if let index = accessOrder.firstIndex(of: metaURL) {
            accessOrder.remove(at: index)
        }
    }
}