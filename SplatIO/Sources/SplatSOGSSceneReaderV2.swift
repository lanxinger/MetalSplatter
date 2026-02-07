import Foundation
import ImageIO
import CoreGraphics
import Compression
import Dispatch
import ZIPFoundation

/// Thread-safe counter for use in concurrent code.
///
/// Thread Safety:
/// - All operations are protected by an internal NSLock.
/// - Marked as `@unchecked Sendable` because thread safety is enforced via NSLock.
private final class LockedCounter: @unchecked Sendable {
    private var value: Int = 0
    private let lock = NSLock()

    /// Adds to the counter and returns the new value (thread-safe)
    func add(_ amount: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        value += amount
        return value
    }
}

public class SplatSOGSSceneReaderV2: SplatSceneReader, @unchecked Sendable {
    public enum SOGSV2Error: Error {
        case invalidMetadata
        case missingFile(String)
        case webpDecodingFailed(String)
        case invalidTextureData
        case unsupportedVersion(Int)
        case zipDecodingFailed(String)
        case fileTooLarge(String)
    }
    
    private let sourceURL: URL
    private let baseURL: URL
    private var zipArchive: SOGSZipArchive?
    
    public init(_ sourceURL: URL) throws {
        self.sourceURL = sourceURL
        self.baseURL = sourceURL.deletingLastPathComponent()
        
        print("SplatSOGSSceneReaderV2: Initializing with URL: \(sourceURL.path)")
        
        let filename = sourceURL.lastPathComponent.lowercased()
        
        // Check if this is a bundled .sog file
        if filename.hasSuffix(".sog") {
            print("SplatSOGSSceneReaderV2: Detected bundled .sog format")
            self.zipArchive = try SOGSZipArchive(sourceURL)
        } else if filename == "meta.json" || filename.hasSuffix(".json") {
            print("SplatSOGSSceneReaderV2: Detected standalone meta.json format")
            self.zipArchive = nil
        } else {
            print("SplatSOGSSceneReaderV2: Invalid filename: \(filename)")
            throw SOGSV2Error.invalidMetadata
        }
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        print("SplatSOGSSceneReaderV2: Loading SOGS v2 metadata...")
        
        // Load metadata from either zip archive or standalone file
        let metaData: Data
        if let zipArchive = zipArchive {
            metaData = try zipArchive.extractFile("meta.json")
            print("SplatSOGSSceneReaderV2: Loaded metadata from zip archive")
        } else {
            metaData = try loadStandaloneFile(sourceURL)
            print("SplatSOGSSceneReaderV2: Loaded metadata from standalone file")
        }
        
        // First try to decode as v2
        let metadata: SOGSMetadataV2
        do {
            let decodedMetadata = try JSONDecoder().decode(SOGSMetadataV2.self, from: metaData)
            
            // Validate version
            guard decodedMetadata.version == 2 else {
                throw SOGSV2Error.unsupportedVersion(decodedMetadata.version)
            }
            
            metadata = decodedMetadata
            print("SplatSOGSSceneReaderV2: Successfully parsed SOGS v2 metadata")
            print("SplatSOGSSceneReaderV2: Found \(metadata.count) splats, antialias: \(metadata.antialias ?? false)")
            
        } catch let error as DecodingError {
            print("SplatSOGSSceneReaderV2: Failed to parse as v2 metadata: \(error)")
            throw SOGSV2Error.invalidMetadata
        } catch let error as SOGSV2Error {
            throw error
        } catch {
            print("SplatSOGSSceneReaderV2: Unexpected error parsing metadata: \(error)")
            throw SOGSV2Error.invalidMetadata
        }
        
        // Load and decode all WebP textures
        let compressedData = try loadCompressedDataV2(metadata: metadata)
        print("SplatSOGSSceneReaderV2: Successfully loaded all v2 WebP textures")
        
        // Decompress and convert to SplatScenePoint format
        return try decompressDataV2(compressedData)
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
    
    private func loadCompressedDataV2(metadata: SOGSMetadataV2) throws -> SOGSCompressedDataV2 {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<SOGSCompressedDataV2, Error>?
        loadCompressedDataV2Async(metadata: metadata) { loadResult in
            result = loadResult
            semaphore.signal()
        }
        semaphore.wait()
        return try result?.get() ?? { throw SOGSV2Error.webpDecodingFailed("Unknown async load failure") }()
    }

    private func loadCompressedDataV2Async(
        metadata: SOGSMetadataV2,
        completion: @escaping (Result<SOGSCompressedDataV2, Error>) -> Void
    ) {
        print("SplatSOGSSceneReaderV2: Loading v2 WebP texture files...")

        // Validate required file lists upfront
        guard metadata.means.files.count == 2 else {
            completion(.failure(SOGSV2Error.invalidMetadata))
            return
        }
        guard let quatsFilename = metadata.quats.files.first else {
            completion(.failure(SOGSV2Error.invalidMetadata))
            return
        }
        guard let scalesFilename = metadata.scales.files.first else {
            completion(.failure(SOGSV2Error.invalidMetadata))
            return
        }
        guard let sh0Filename = metadata.sh0.files.first else {
            completion(.failure(SOGSV2Error.invalidMetadata))
            return
        }

        var sanitizedMetadata = metadata
        var shCentroidsFilename: String?
        var shLabelsFilename: String?

        if let shN = metadata.shN {
            let shNHasRange = (shN.mins?.count ?? 0) > 0 && (shN.maxs?.count ?? 0) > 0
            let shNHasCodebook = shN.codebook.count >= 256

            if !(shNHasRange || shNHasCodebook) {
                print("SplatSOGSSceneReaderV2: SH metadata missing codebook/range - disabling SH data")
                sanitizedMetadata = metadataWithoutSphericalHarmonics(metadata)
            } else {
                for filename in shN.files {
                    let lower = filename.lowercased()
                    if lower.contains("centroid") {
                        shCentroidsFilename = filename
                    } else if lower.contains("label") {
                        shLabelsFilename = filename
                    }
                }

                if shCentroidsFilename == nil || shLabelsFilename == nil {
                    print("SplatSOGSSceneReaderV2: Missing SH centroid/label textures - disabling SH data")
                    sanitizedMetadata = metadataWithoutSphericalHarmonics(metadata)
                    shCentroidsFilename = nil
                    shLabelsFilename = nil
                }
            }
        }

        let shouldLoadSHData = sanitizedMetadata.shN != nil
        let resolvedSHCentroidsFilename = shCentroidsFilename
        let resolvedSHLabelsFilename = shLabelsFilename

        // Use concurrent loading for better performance
        let queue = DispatchQueue(label: "sogsv2.webp.loading", attributes: .concurrent)
        let group = DispatchGroup()
        let loadWebP = self.loadAndDecodeWebPV2
        let meansL = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let meansU = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let quats = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let scales = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let sh0 = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let shCentroids = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let shLabels = LockedBox<WebPDecoder.DecodedImage?>(nil)
        let loadingErrors = LockedBox<[Error]>([])

        // Load means_l
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                meansL.set(try loadWebP(metadata.means.files[0]))
            } catch {
                loadingErrors.withValue { $0.append(error) }
            }
        }

        // Load means_u
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                meansU.set(try loadWebP(metadata.means.files[1]))
            } catch {
                loadingErrors.withValue { $0.append(error) }
            }
        }

        // Load quats
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                quats.set(try loadWebP(quatsFilename))
            } catch {
                loadingErrors.withValue { $0.append(error) }
            }
        }

        // Load scales
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                scales.set(try loadWebP(scalesFilename))
            } catch {
                loadingErrors.withValue { $0.append(error) }
            }
        }

        // Load sh0
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                sh0.set(try loadWebP(sh0Filename))
            } catch {
                loadingErrors.withValue { $0.append(error) }
            }
        }

        // Load optional spherical harmonics textures
        if shouldLoadSHData {
            // Load centroids
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    guard let filename = resolvedSHCentroidsFilename else { throw SOGSV2Error.invalidMetadata }
                    let image = try loadWebP(filename)
                    shCentroids.set(image)
                    print("SplatSOGSSceneReaderV2: Loaded SH centroids texture")
                } catch {
                    loadingErrors.withValue { $0.append(error) }
                }
            }

            // Load labels
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    guard let filename = resolvedSHLabelsFilename else { throw SOGSV2Error.invalidMetadata }
                    let image = try loadWebP(filename)
                    shLabels.set(image)
                    print("SplatSOGSSceneReaderV2: Loaded SH labels texture")
                } catch {
                    loadingErrors.withValue { $0.append(error) }
                }
            }
        }

        group.notify(queue: queue) {
            if let error = loadingErrors.get().first {
                completion(.failure(error))
                return
            }

            let resolvedMeansL = meansL.get()
            let resolvedMeansU = meansU.get()
            let resolvedQuats = quats.get()
            let resolvedScales = scales.get()
            let resolvedSh0 = sh0.get()
            let resolvedCentroids = shCentroids.get()
            let resolvedLabels = shLabels.get()

            guard let means_l = resolvedMeansL,
                  let means_u = resolvedMeansU,
                  let quats = resolvedQuats,
                  let scales = resolvedScales,
                  let sh0 = resolvedSh0 else {
                completion(.failure(SOGSV2Error.webpDecodingFailed("Failed to load required v2 textures")))
                return
            }

            do {
                try self.validateSOGSV2Data(metadata: sanitizedMetadata,
                                            means_l: means_l,
                                            means_u: means_u,
                                            quats: quats,
                                            scales: scales,
                                            sh0: sh0,
                                            sh_centroids: resolvedCentroids,
                                            sh_labels: resolvedLabels)
            } catch {
                completion(.failure(error))
                return
            }

            print("SplatSOGSSceneReaderV2: Successfully loaded all v2 WebP textures")
            completion(.success(SOGSCompressedDataV2(
                metadata: sanitizedMetadata,
                means_l: means_l,
                means_u: means_u,
                quats: quats,
                scales: scales,
                sh0: sh0,
                sh_centroids: resolvedCentroids,
                sh_labels: resolvedLabels
            )))
        }
    }
    
    private func metadataWithoutSphericalHarmonics(_ metadata: SOGSMetadataV2) -> SOGSMetadataV2 {
        return SOGSMetadataV2(
            version: metadata.version,
            count: metadata.count,
            antialias: metadata.antialias,
            means: metadata.means,
            scales: metadata.scales,
            quats: metadata.quats,
            sh0: metadata.sh0,
            shN: nil
        )
    }
    
    private func loadAndDecodeWebPV2(_ filename: String) throws -> WebPDecoder.DecodedImage {
        print("SplatSOGSSceneReaderV2: Loading WebP file: \(filename)")
        
        let webpData: Data
        
        // Load from either zip archive or standalone file system
        if let zipArchive = zipArchive {
            webpData = try zipArchive.extractFile(filename)
            print("SplatSOGSSceneReaderV2: Loaded \(webpData.count) bytes from zip: \(filename)")
        } else {
            webpData = try loadStandaloneFile(baseURL.appendingPathComponent(filename))
            print("SplatSOGSSceneReaderV2: Loaded \(webpData.count) bytes from filesystem: \(filename)")
        }
        
        // Validate WebP signature
        if webpData.count >= 12 {
            let riffHeader = webpData.prefix(4)
            let webpSignature = webpData.dropFirst(8).prefix(4)
            print("SplatSOGSSceneReaderV2: RIFF header: \(riffHeader.map { String(format: "%02X", $0) }.joined())")
            print("SplatSOGSSceneReaderV2: WebP signature: \(webpSignature.map { String(format: "%02X", $0) }.joined())")
        }
        
        // Decode WebP data
        do {
            // Try Core Image first (iOS 14+/macOS 11+)
            let result = try WebPDecoder.decode(webpData)
            print("SplatSOGSSceneReaderV2: Successfully decoded \(filename) - \(result.width)x\(result.height), \(result.bytesPerPixel) bpp")
            return result
        } catch {
            print("SplatSOGSSceneReaderV2: Core Image decode failed for \(filename): \(error)")
            
            // Fallback to ImageIO
            do {
                let result = try WebPDecoder.decodeWithImageIO(webpData)
                print("SplatSOGSSceneReaderV2: Successfully decoded \(filename) using ImageIO")
                return result
            } catch {
                print("SplatSOGSSceneReaderV2: Both decoders failed for \(filename): \(error)")
                throw SOGSV2Error.webpDecodingFailed("Failed to decode \(filename): \(error)")
            }
        }
    }
    
    private func validateSOGSV2Data(metadata: SOGSMetadataV2,
                                    means_l: WebPDecoder.DecodedImage,
                                    means_u: WebPDecoder.DecodedImage,
                                    quats: WebPDecoder.DecodedImage,
                                    scales: WebPDecoder.DecodedImage,
                                    sh0: WebPDecoder.DecodedImage,
                                    sh_centroids: WebPDecoder.DecodedImage?,
                                    sh_labels: WebPDecoder.DecodedImage?) throws {
        let baseWidth = means_l.width
        let baseHeight = means_l.height

        func matchesBaseDimensions(_ image: WebPDecoder.DecodedImage) -> Bool {
            image.width == baseWidth && image.height == baseHeight
        }

        guard matchesBaseDimensions(means_u),
              matchesBaseDimensions(quats),
              matchesBaseDimensions(scales),
              matchesBaseDimensions(sh0) else {
            throw SOGSV2Error.invalidMetadata
        }

        guard metadata.count >= 0, metadata.count <= baseWidth * baseHeight else {
            throw SOGSV2Error.invalidMetadata
        }

        guard metadata.means.mins.count == 3,
              metadata.means.maxs.count == 3 else {
            throw SOGSV2Error.invalidMetadata
        }

        let scalesHasRange = (metadata.scales.mins?.count ?? 0) >= 3 && (metadata.scales.maxs?.count ?? 0) >= 3
        let scalesHasCodebook = metadata.scales.codebook.count >= 256
        guard scalesHasRange || scalesHasCodebook else {
            throw SOGSV2Error.invalidMetadata
        }

        let sh0HasRange = (metadata.sh0.mins?.count ?? 0) >= 4 && (metadata.sh0.maxs?.count ?? 0) >= 4
        let sh0HasCodebook = metadata.sh0.codebook.count >= 256
        guard sh0HasRange || sh0HasCodebook else {
            throw SOGSV2Error.invalidMetadata
        }

        if let shN = metadata.shN {
            if let bands = shN.bands, !(1...3).contains(bands) {
                print("SplatSOGSSceneReaderV2: Validation warning - shN.bands \(bands) outside 1...3, ignoring hint.")
            }

            if let count = shN.count, count < 0 {
                print("SplatSOGSSceneReaderV2: Validation warning - shN.count < 0, ignoring hint.")
            }

            let shNHasRange = (shN.mins?.count ?? 0) > 0 && (shN.maxs?.count ?? 0) > 0
            let shNHasCodebook = shN.codebook.count >= 256
            guard shNHasRange || shNHasCodebook else {
                print("SplatSOGSSceneReaderV2: Validation warning - shN missing codebook/range, SH decoding may be lossy")
                throw SOGSV2Error.invalidMetadata
            }

            guard let labels = sh_labels,
                  let centroids = sh_centroids,
                  labels.width > 0,
                  labels.height > 0,
                  centroids.width > 0,
                  centroids.height > 0 else {
                throw SOGSV2Error.invalidMetadata
            }

            // SH labels may be padded; require width match and enough rows for all splats
            let requiredLabelRows = (metadata.count + baseWidth - 1) / baseWidth
            guard labels.width >= baseWidth,
                  labels.height >= requiredLabelRows else {
                throw SOGSV2Error.invalidMetadata
            }

            if labels.width != baseWidth || labels.height != baseHeight {
                print("SplatSOGSSceneReaderV2: Validation warning - SH labels texture padded to \(labels.width)x\(labels.height), expected \(baseWidth)x\(baseHeight)")
            }

            guard centroids.width % 64 == 0 else {
                throw SOGSV2Error.invalidMetadata
            }

            let coefficientsFromTexture = centroids.width / 64
            guard coefficientsFromTexture > 0 else {
                throw SOGSV2Error.invalidMetadata
            }

            if let coefficientsFromMetadata = shN.coefficientsPerEntry,
               coefficientsFromMetadata != coefficientsFromTexture {
                print("SplatSOGSSceneReaderV2: Validation warning - shN bands hint (\(coefficientsFromMetadata)) does not match texture layout (\(coefficientsFromTexture)), using texture data.")
            }

            let paletteCapacity = centroids.height * 64
            guard paletteCapacity > 0 else {
                throw SOGSV2Error.invalidMetadata
            }

            if let count = shN.count, count > paletteCapacity {
                print("SplatSOGSSceneReaderV2: Validation warning - shN palette hint (\(count)) exceeds texture capacity (\(paletteCapacity)), clamping to texture.")
            }
        }
    }

    private func loadStandaloneFile(_ fileURL: URL) throws -> Data {
        // Handle iOS File Provider security scoped resource access
        let shouldStopAccessing = fileURL.startAccessingSecurityScopedResource()
        defer {
            if shouldStopAccessing {
                fileURL.stopAccessingSecurityScopedResource()
            }
        }
        
        return try Data(contentsOf: fileURL)
    }
    
    private func decompressDataV2(_ compressedData: SOGSCompressedDataV2) throws -> [SplatScenePoint] {
        let totalSplats = compressedData.numSplats
        print("SplatSOGSSceneReaderV2: Decompressing \(totalSplats) v2 splats...")
        print("SplatSOGSSceneReaderV2: Texture dimensions: \(compressedData.textureWidth)x\(compressedData.textureHeight)")
        print("SplatSOGSSceneReaderV2: Has spherical harmonics: \(compressedData.hasSphericalHarmonics)")

        guard totalSplats > 0 else {
            return []
        }

        let preferredBatchSize = 8192
        let hardwareThreads = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let maxParallelChunks = max(1, min(hardwareThreads, (totalSplats + preferredBatchSize - 1) / preferredBatchSize))

        if maxParallelChunks <= 1 || totalSplats <= preferredBatchSize {
            return decompressSequentialV2(compressedData, batchSize: preferredBatchSize)
        }

        let chunkSize = max(preferredBatchSize, (totalSplats + maxParallelChunks - 1) / maxParallelChunks)
        var chunkDescriptors: [(start: Int, count: Int)] = []
        chunkDescriptors.reserveCapacity(maxParallelChunks)
        var startIndex = 0
        while startIndex < totalSplats {
            let count = min(chunkSize, totalSplats - startIndex)
            chunkDescriptors.append((start: startIndex, count: count))
            startIndex += count
        }

        var chunkResults = Array(repeating: [SplatScenePoint](), count: chunkDescriptors.count)
        let progressCounter = LockedCounter()
        let logStep = max(1, min(preferredBatchSize * 2, totalSplats / max(1, maxParallelChunks)))

        chunkResults.withUnsafeMutableBufferPointer { buffer in
            for chunkIndex in chunkDescriptors.indices {
                let descriptor = chunkDescriptors[chunkIndex]
                let iterator = SOGSIteratorV2(compressedData)
                var localPoints: [SplatScenePoint] = []
                localPoints.reserveCapacity(descriptor.count)

                let end = descriptor.start + descriptor.count
                for idx in descriptor.start..<end {
                    localPoints.append(iterator.readPoint(at: idx))
                }

                buffer[chunkIndex] = localPoints

                let processed = progressCounter.add(descriptor.count)
                if processed == totalSplats || processed % logStep == 0 {
                    print("SplatSOGSSceneReaderV2: Batch processed \(processed)/\(totalSplats) splats")
                }
            }
        }

        var allPoints: [SplatScenePoint] = []
        allPoints.reserveCapacity(totalSplats)
        for chunk in chunkResults {
            allPoints.append(contentsOf: chunk)
        }

        print("SplatSOGSSceneReaderV2: Successfully decompressed \(allPoints.count) v2 points")
        return allPoints
    }

    private func decompressSequentialV2(_ compressedData: SOGSCompressedDataV2, batchSize: Int) -> [SplatScenePoint] {
        let batchIterator = SOGSBatchIteratorV2(compressedData)
        var allPoints: [SplatScenePoint] = []
        allPoints.reserveCapacity(compressedData.numSplats)

        let numBatches = (compressedData.numSplats + batchSize - 1) / batchSize

        for batchIndex in 0..<numBatches {
            let startIndex = batchIndex * batchSize
            let remainingPoints = compressedData.numSplats - startIndex
            let currentBatchSize = min(batchSize, remainingPoints)

            let batchPoints = batchIterator.readBatch(startIndex: startIndex, count: currentBatchSize)
            allPoints.append(contentsOf: batchPoints)

            if batchIndex % 10 == 0 || batchIndex == numBatches - 1 {
                let processed = min(startIndex + currentBatchSize, compressedData.numSplats)
                print("SplatSOGSSceneReaderV2: Batch processed \(processed)/\(compressedData.numSplats) splats")
            }
        }

        print("SplatSOGSSceneReaderV2: Successfully decompressed \(allPoints.count) v2 points")
        return allPoints
    }
}

// MARK: - Zip Archive Reader for Bundled .sog Files

private class SOGSZipArchive {
    private typealias ZipEntry = ZIPFoundation.Entry
    private let archive: Archive
    private let url: URL
    private let shouldStopAccessing: Bool
    private let archiveBackingData: Data?
    private let entriesByName: [String: ZipEntry]
    private let archiveQueue = DispatchQueue(label: "sogs.zip.archive.serial")
    
    init(_ zipURL: URL) throws {
        self.url = zipURL
        let shouldStopAccessing = zipURL.startAccessingSecurityScopedResource()
        var stopAccessOnDeinit = shouldStopAccessing
        var backingData: Data? = nil
        
        do {
            self.archive = try Archive(url: zipURL, accessMode: .read)
            self.archiveBackingData = nil
            self.shouldStopAccessing = stopAccessOnDeinit
        } catch {
            print("SOGSZipArchive: File access archive init failed: \(error). Falling back to in-memory loading.")
            do {
                backingData = try Data(contentsOf: zipURL)
                if stopAccessOnDeinit {
                    zipURL.stopAccessingSecurityScopedResource()
                    stopAccessOnDeinit = false
                }
                self.archive = try Archive(data: backingData ?? Data(), accessMode: .read)
                self.archiveBackingData = backingData
                self.shouldStopAccessing = stopAccessOnDeinit
            } catch {
                if stopAccessOnDeinit {
                    zipURL.stopAccessingSecurityScopedResource()
                }
                throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("Unable to open SOG archive: \(error)")
            }
        }
        
        print("SOGSZipArchive: Opened \(zipURL.lastPathComponent)")
        
        var entryCount = 0
        var entries: [String: ZipEntry] = [:]
        for entry in archive {
            entryCount += 1
            let compressed = entry.compressedSize
            let uncompressed = entry.uncompressedSize
            print("  - \(entry.path) (compressed: \(compressed), uncompressed: \(uncompressed))")
            entries[entry.path] = entry
        }
        self.entriesByName = entries
        print("SOGSZipArchive: Found \(entryCount) files in archive")
    }
    
    deinit {
        if shouldStopAccessing {
            url.stopAccessingSecurityScopedResource()
        }
    }
    
    /// Maximum extracted file size (512 MB) - prevents zip bombs
    private static let maxExtractedFileSize: Int = 512 * 1024 * 1024

    func extractFile(_ filename: String) throws -> Data {
        return try archiveQueue.sync(execute: { () throws -> Data in
            guard let entry = entriesByName[filename] else {
                print("SOGSZipArchive: File not found: \(filename)")
                throw SplatSOGSSceneReaderV2.SOGSV2Error.missingFile(filename)
            }

            // Pre-validate size from ZIP directory (trusted for non-malicious files)
            let uncompressedSize = entry.uncompressedSize
            if uncompressedSize > Self.maxExtractedFileSize {
                print("SOGSZipArchive: File \(filename) too large: \(uncompressedSize) bytes")
                throw SplatSOGSSceneReaderV2.SOGSV2Error.fileTooLarge(filename)
            }

            print("SOGSZipArchive: Extracting \(filename)...")
            var extractedData = Data()

            // Reserve with capped size
            let estimatedSize = entry.uncompressedSize
            let reserveSize = min(Int(min(estimatedSize, UInt64(Int.max))), Self.maxExtractedFileSize)
            if reserveSize > 0 {
                extractedData.reserveCapacity(reserveSize)
            }

            // Streaming flag for defense-in-depth (consumer callback is non-throwing)
            var tooLarge = false

            do {
                _ = try archive.extract(entry, consumer: { chunk in
                    if !tooLarge {
                        extractedData.append(chunk)
                        if extractedData.count > Self.maxExtractedFileSize {
                            tooLarge = true
                            extractedData.removeAll()  // Free memory
                        }
                    }
                })

                // Check if extraction was aborted due to size
                if tooLarge {
                    print("SOGSZipArchive: Extraction of \(filename) aborted - output too large")
                    throw SplatSOGSSceneReaderV2.SOGSV2Error.fileTooLarge(filename)
                }

                print("SOGSZipArchive: Extracted \(extractedData.count) bytes")
                return extractedData
            } catch let error as SplatSOGSSceneReaderV2.SOGSV2Error {
                throw error  // Re-throw our own errors
            } catch {
                print("SOGSZipArchive: Failed to extract \(filename): \(error)")
                throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("Failed to extract \(filename): \(error.localizedDescription)")
            }
        })
    }
}
