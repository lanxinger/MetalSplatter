import Foundation
import ImageIO
import CoreGraphics

/// Optimized version of SplatSOGSSceneReader with performance improvements
public class SplatSOGSSceneReaderOptimized: SplatSceneReader, @unchecked Sendable {
    public enum SOGSError: Error {
        case invalidMetadata
        case missingFile(String)
        case webpDecodingFailed(String)
        case invalidTextureData
        case missingRequiredTextureFile(String, Int, Int)  // texture name, required count, actual count
    }
    
    private let metaURL: URL
    private let baseURL: URL
    private let useCache: Bool
    private let useParallelLoading: Bool
    private let batchSize: Int
    
    /// Initialize with options for performance tuning
    /// - Parameters:
    ///   - metaURL: URL to the meta.json file
    ///   - useCache: Whether to use texture caching (default: true)
    ///   - useParallelLoading: Whether to load WebP textures in parallel (default: true)
    ///   - batchSize: Number of points to process in each batch (default: 1024)
    public init(_ metaURL: URL, useCache: Bool = true, useParallelLoading: Bool = true, batchSize: Int = 1024) throws {
        self.metaURL = metaURL
        self.baseURL = metaURL.deletingLastPathComponent()
        self.useCache = useCache
        self.useParallelLoading = useParallelLoading
        self.batchSize = max(256, batchSize) // Minimum batch size of 256
        
        print("SplatSOGSSceneReaderOptimized: Initializing with URL: \(metaURL.path)")
        print("SplatSOGSSceneReaderOptimized: Options - Cache: \(useCache), Parallel: \(useParallelLoading), Batch: \(batchSize)")
        
        // Validate that it's a SOGS meta.json file
        let filename = metaURL.lastPathComponent.lowercased()
        guard filename == "meta.json" || filename.hasSuffix(".json") else {
            print("SplatSOGSSceneReaderOptimized: Invalid filename: \(filename)")
            throw SOGSError.invalidMetadata
        }
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        print("SplatSOGSSceneReaderOptimized: Loading SOGS metadata from \(metaURL.path)")
        
        // Load and parse metadata
        let metaData: Data
        do {
            // Handle iOS File Provider security scoped resource access
            let shouldStopAccessing = metaURL.startAccessingSecurityScopedResource()
            defer {
                if shouldStopAccessing {
                    metaURL.stopAccessingSecurityScopedResource()
                }
            }
            
            metaData = try Data(contentsOf: metaURL)
            print("SplatSOGSSceneReaderOptimized: Loaded \(metaData.count) bytes of metadata")
        } catch {
            print("SplatSOGSSceneReaderOptimized: Failed to load metadata file: \(error)")
            throw SOGSError.invalidMetadata
        }
        
        let metadata: SOGSMetadata
        do {
            metadata = try JSONDecoder().decode(SOGSMetadata.self, from: metaData)
            print("SplatSOGSSceneReaderOptimized: Successfully parsed metadata")
        } catch {
            print("SplatSOGSSceneReaderOptimized: Failed to parse metadata JSON: \(error)")
            throw SOGSError.invalidMetadata
        }
        
        print("SplatSOGSSceneReaderOptimized: Found \(metadata.means.shape[0]) splats")
        
        // Load compressed data with caching and parallel loading
        let compressedData: SOGSCompressedData
        
        if useCache {
            compressedData = try SOGSTextureCache.shared.getCompressedData(for: metaURL) {
                try self.loadCompressedDataOptimized(metadata: metadata)
            }
        } else {
            compressedData = try loadCompressedDataOptimized(metadata: metadata)
        }
        
        print("SplatSOGSSceneReaderOptimized: Successfully loaded all WebP textures")
        
        // Decompress using batch processing
        return try decompressBatchOptimized(compressedData)
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        do {
            let points = try readScene()
            delegate.didStartReading(withPointCount: UInt32(points.count))
            delegate.didRead(points: points)
            delegate.didFinishReading()
        } catch {
            delegate.didFailReading(withError: error)
        }
    }
    
    private func loadCompressedDataOptimized(metadata: SOGSMetadata) throws -> SOGSCompressedData {
        if useParallelLoading {
            return try loadCompressedDataParallel(metadata: metadata)
        } else {
            return try loadCompressedDataSequential(metadata: metadata)
        }
    }
    
    /// Sequential loading (original implementation)
    private func loadCompressedDataSequential(metadata: SOGSMetadata) throws -> SOGSCompressedData {
        print("SplatSOGSSceneReaderOptimized: Loading WebP textures sequentially...")

        // Validate required texture file counts before loading
        guard metadata.means.files.count >= 2 else {
            throw SOGSError.missingRequiredTextureFile("means", 2, metadata.means.files.count)
        }
        guard metadata.quats.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("quats", 1, metadata.quats.files.count)
        }
        guard metadata.scales.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("scales", 1, metadata.scales.files.count)
        }
        guard metadata.sh0.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("sh0", 1, metadata.sh0.files.count)
        }

        let means_l = try loadAndDecodeWebP(metadata.means.files[0])
        let means_u = try loadAndDecodeWebP(metadata.means.files[1])
        let quats = try loadAndDecodeWebP(metadata.quats.files[0])
        let scales = try loadAndDecodeWebP(metadata.scales.files[0])
        let sh0 = try loadAndDecodeWebP(metadata.sh0.files[0])
        
        var sh_centroids: WebPDecoder.DecodedImage?
        var sh_labels: WebPDecoder.DecodedImage?
        
        if let shN = metadata.shN, shN.files.count >= 2 {
            let tempCentroids = try loadAndDecodeWebP(shN.files[0])
            let shBands = calculateSHBands(width: tempCentroids.width)
            
            if shBands > 0 {
                sh_centroids = tempCentroids
                sh_labels = try loadAndDecodeWebP(shN.files[1])
                print("SplatSOGSSceneReaderOptimized: Loaded SH data with \(shBands) bands")
            }
        }
        
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
    
    /// Parallel loading for better performance
    private func loadCompressedDataParallel(metadata: SOGSMetadata) throws -> SOGSCompressedData {
        print("SplatSOGSSceneReaderOptimized: Loading WebP textures in parallel...")

        // Validate required texture file counts before loading
        guard metadata.means.files.count >= 2 else {
            throw SOGSError.missingRequiredTextureFile("means", 2, metadata.means.files.count)
        }
        guard metadata.quats.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("quats", 1, metadata.quats.files.count)
        }
        guard metadata.scales.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("scales", 1, metadata.scales.files.count)
        }
        guard metadata.sh0.files.count >= 1 else {
            throw SOGSError.missingRequiredTextureFile("sh0", 1, metadata.sh0.files.count)
        }

        let queue = DispatchQueue(label: "sogs.webp.loading", attributes: .concurrent)
        let group = DispatchGroup()
        let loadWebP = self.loadAndDecodeWebP
        let meansL = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let meansU = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let quats = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let scales = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let sh0 = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let shCentroids = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let shLabels = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let loadingErrors = LockedBox<[Error]>([])

        // Load all textures in parallel
        let loadTasks: [String] = [
            metadata.means.files[0],
            metadata.means.files[1],
            metadata.quats.files[0],
            metadata.scales.files[0],
            metadata.sh0.files[0]
        ]

        for i in 0..<loadTasks.count {
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    let decoded = try loadWebP(loadTasks[i])
                    switch i {
                    case 0: meansL.set(decoded)
                    case 1: meansU.set(decoded)
                    case 2: quats.set(decoded)
                    case 3: scales.set(decoded)
                    case 4: sh0.set(decoded)
                    default: break
                    }
                } catch {
                    loadingErrors.withValue { $0.append(error) }
                }
            }
        }

        // Load optional SH textures
        if let shN = metadata.shN, shN.files.count >= 2 {
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    let tempCentroids = try loadWebP(shN.files[0])
                    let shBands = self.calculateSHBands(width: tempCentroids.width)
                    if shBands > 0 {
                        shCentroids.set(tempCentroids)

                        let labels = try loadWebP(shN.files[1])
                        shLabels.set(labels)
                    }
                } catch {
                    loadingErrors.withValue { $0.append(error) }
                }
            }
        }

        group.wait()

        if let error = loadingErrors.get().first {
            throw error
        }
        
        guard let means_l = meansL.get(),
              let means_u = meansU.get(),
              let quats = quats.get(),
              let scales = scales.get(),
              let sh0 = sh0.get() else {
            throw SOGSError.webpDecodingFailed("Failed to load required textures")
        }
        
        print("SplatSOGSSceneReaderOptimized: Successfully loaded all WebP textures in parallel")
        
        return SOGSCompressedData(
            metadata: metadata,
            means_l: means_l,
            means_u: means_u,
            quats: quats,
            scales: scales,
            sh0: sh0,
            sh_centroids: shCentroids.get(),
            sh_labels: shLabels.get()
        )
    }
    
    /// Batch decompression for better performance
    private func decompressBatchOptimized(_ compressedData: SOGSCompressedData) throws -> [SplatScenePoint] {
        let startTime = Date()
        print("SplatSOGSSceneReaderOptimized: Decompressing \(compressedData.numSplats) splats in batches of \(batchSize)...")
        
        let iterator = SOGSBatchIterator(compressedData)
        var allPoints: [SplatScenePoint] = []
        allPoints.reserveCapacity(compressedData.numSplats)
        
        // Process in batches
        var processedCount = 0
        while processedCount < compressedData.numSplats {
            let batchCount = min(batchSize, compressedData.numSplats - processedCount)
            let batchPoints = iterator.readBatch(startIndex: processedCount, count: batchCount)
            allPoints.append(contentsOf: batchPoints)
            
            processedCount += batchCount
            
            // Progress reporting for large datasets
            if processedCount % 10000 == 0 || processedCount == compressedData.numSplats {
                let progress = Float(processedCount) / Float(compressedData.numSplats) * 100
                print(String(format: "SplatSOGSSceneReaderOptimized: Progress: %.1f%% (%d/%d)", 
                           progress, processedCount, compressedData.numSplats))
            }
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        let pointsPerSecond = Double(compressedData.numSplats) / elapsed
        print(String(format: "SplatSOGSSceneReaderOptimized: Decompressed %d points in %.2f seconds (%.0f points/sec)",
                   compressedData.numSplats, elapsed, pointsPerSecond))
        
        return allPoints
    }
    
    // Reuse existing helper methods from original implementation
    private func loadAndDecodeWebP(_ filename: String) throws -> WebPDecoder.DecodedImage {
        // Use the same implementation as the original SplatSOGSSceneReader
        // This is just a placeholder - the actual implementation should be copied
        let fileURL = baseURL.appendingPathComponent(filename)
        let webpData = try Data(contentsOf: fileURL)
        return try WebPDecoder.decode(webpData)
    }
    
    internal func calculateSHBands(width: Int) -> Int {
        switch width {
        case 192: return 1
        case 512: return 2
        case 960: return 3
        default: return 0
        }
    }
}
