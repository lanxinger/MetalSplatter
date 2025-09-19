import Foundation
import ImageIO
import CoreGraphics
import Compression

public class SplatSOGSSceneReaderV2: SplatSceneReader {
    public enum SOGSV2Error: Error {
        case invalidMetadata
        case missingFile(String)
        case webpDecodingFailed(String)
        case invalidTextureData
        case unsupportedVersion(Int)
        case zipDecodingFailed(String)
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
        print("SplatSOGSSceneReaderV2: Loading v2 WebP texture files...")
        
        // Validate required file lists upfront
        guard metadata.means.files.count == 2 else {
            throw SOGSV2Error.invalidMetadata
        }
        guard let quatsFilename = metadata.quats.files.first else {
            throw SOGSV2Error.invalidMetadata
        }
        guard let scalesFilename = metadata.scales.files.first else {
            throw SOGSV2Error.invalidMetadata
        }
        guard let sh0Filename = metadata.sh0.files.first else {
            throw SOGSV2Error.invalidMetadata
        }
        
        var shCentroidsFilename: String?
        var shLabelsFilename: String?
        if let shN = metadata.shN {
            for filename in shN.files {
                let lower = filename.lowercased()
                if lower.contains("centroid") {
                    shCentroidsFilename = filename
                } else if lower.contains("label") {
                    shLabelsFilename = filename
                }
            }
            if shCentroidsFilename == nil || shLabelsFilename == nil {
                throw SOGSV2Error.invalidMetadata
            }
        }

        // Use concurrent loading for better performance
        let queue = DispatchQueue(label: "sogsv2.webp.loading", attributes: .concurrent)
        let group = DispatchGroup()

        // Storage for results
        var means_l: WebPDecoder.DecodedImage?
        var means_u: WebPDecoder.DecodedImage?
        var quats: WebPDecoder.DecodedImage?
        var scales: WebPDecoder.DecodedImage?
        var sh0: WebPDecoder.DecodedImage?
        var sh_centroids: WebPDecoder.DecodedImage?
        var sh_labels: WebPDecoder.DecodedImage?
        
        // Error handling
        var loadingErrors: [Error] = []
        let errorLock = NSLock()
        
        // Load required textures in parallel
        let requiredFiles = [
            (metadata.means.files[0], \SOGSCompressedDataV2.means_l),
            (metadata.means.files[1], \SOGSCompressedDataV2.means_u),
            (metadata.quats.files[0], \SOGSCompressedDataV2.quats),
            (metadata.scales.files[0], \SOGSCompressedDataV2.scales),
            (metadata.sh0.files[0], \SOGSCompressedDataV2.sh0)
        ]
        
        // Load means_l
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                means_l = try self.loadAndDecodeWebPV2(metadata.means.files[0])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load means_u
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                means_u = try self.loadAndDecodeWebPV2(metadata.means.files[1])
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load quats
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                quats = try self.loadAndDecodeWebPV2(quatsFilename)
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load scales
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                scales = try self.loadAndDecodeWebPV2(scalesFilename)
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load sh0
        group.enter()
        queue.async {
            defer { group.leave() }
            do {
                sh0 = try self.loadAndDecodeWebPV2(sh0Filename)
            } catch {
                errorLock.lock()
                loadingErrors.append(error)
                errorLock.unlock()
            }
        }
        
        // Load optional spherical harmonics textures
        if metadata.shN != nil {
            // Load centroids
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    guard let filename = shCentroidsFilename else { throw SOGSV2Error.invalidMetadata }
                    sh_centroids = try self.loadAndDecodeWebPV2(filename)
                    print("SplatSOGSSceneReaderV2: Loaded SH centroids texture")
                } catch {
                    errorLock.lock()
                    loadingErrors.append(error)
                    errorLock.unlock()
                }
            }
            
            // Load labels
            group.enter()
            queue.async {
                defer { group.leave() }
                do {
                    guard let filename = shLabelsFilename else { throw SOGSV2Error.invalidMetadata }
                    sh_labels = try self.loadAndDecodeWebPV2(filename)
                    print("SplatSOGSSceneReaderV2: Loaded SH labels texture")
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
            throw SOGSV2Error.webpDecodingFailed("Failed to load required v2 textures")
        }
        
        try validateSOGSV2Data(metadata: metadata,
                               means_l: means_l,
                               means_u: means_u,
                               quats: quats,
                               scales: scales,
                               sh0: sh0,
                               sh_centroids: sh_centroids,
                               sh_labels: sh_labels)

        print("SplatSOGSSceneReaderV2: Successfully loaded all v2 WebP textures")
        
        return SOGSCompressedDataV2(
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
            if let bands = shN.bands {
                guard (1...3).contains(bands) else {
                    throw SOGSV2Error.invalidMetadata
                }
            }

            if let count = shN.count {
                guard count > 0 else { throw SOGSV2Error.invalidMetadata }
            }

            let shNHasRange = (shN.mins?.count ?? 0) > 0 && (shN.maxs?.count ?? 0) > 0
            let shNHasCodebook = shN.codebook.count >= 256
            guard shNHasRange || shNHasCodebook else {
                throw SOGSV2Error.invalidMetadata
            }

            guard let labels = sh_labels,
                  let centroids = sh_centroids,
                  matchesBaseDimensions(labels) else {
                throw SOGSV2Error.invalidMetadata
            }

            let coefficientsPerEntry = shN.coefficientsPerEntry ?? (centroids.width / 64)
            guard coefficientsPerEntry > 0,
                  centroids.width % 64 == 0 else {
                throw SOGSV2Error.invalidMetadata
            }

            let expectedWidth = coefficientsPerEntry * 64
            guard centroids.width == expectedWidth else {
                throw SOGSV2Error.invalidMetadata
            }

            if let count = shN.count {
                let requiredRows = (count + 63) / 64
                guard centroids.height >= requiredRows else {
                    throw SOGSV2Error.invalidMetadata
                }
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
        print("SplatSOGSSceneReaderV2: Decompressing \(compressedData.numSplats) v2 splats...")
        print("SplatSOGSSceneReaderV2: Texture dimensions: \(compressedData.textureWidth)x\(compressedData.textureHeight)")
        print("SplatSOGSSceneReaderV2: Has spherical harmonics: \(compressedData.hasSphericalHarmonics)")
        
        var batchIterator = SOGSBatchIteratorV2(compressedData)
        var allPoints: [SplatScenePoint] = []
        allPoints.reserveCapacity(compressedData.numSplats)
        
        // Process in batches for optimal performance
        let batchSize = 8192
        let numBatches = (compressedData.numSplats + batchSize - 1) / batchSize
        
        for batchIndex in 0..<numBatches {
            let startIndex = batchIndex * batchSize
            let remainingPoints = compressedData.numSplats - startIndex
            let currentBatchSize = min(batchSize, remainingPoints)
            
            let batchPoints = batchIterator.readBatch(startIndex: startIndex, count: currentBatchSize)
            allPoints.append(contentsOf: batchPoints)
            
            // Progress logging
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
    private let zipData: Data
    private let centralDirectory: [ZipCentralDirectoryEntry]
    
    struct ZipCentralDirectoryEntry {
        let filename: String
        let compressedSize: UInt32
        let uncompressedSize: UInt32
        let localHeaderOffset: UInt32
        let compressionMethod: UInt16
    }
    
    init(_ zipURL: URL) throws {
        // Load zip file data
        let shouldStopAccessing = zipURL.startAccessingSecurityScopedResource()
        defer {
            if shouldStopAccessing {
                zipURL.stopAccessingSecurityScopedResource()
            }
        }
        
        self.zipData = try Data(contentsOf: zipURL)
        print("SOGSZipArchive: Loaded \(zipData.count) bytes from \(zipURL.lastPathComponent)")
        
        // Parse central directory
        self.centralDirectory = try Self.parseCentralDirectory(zipData)
        print("SOGSZipArchive: Found \(centralDirectory.count) files in archive")
        
        for entry in centralDirectory {
            print("  - \(entry.filename) (\(entry.uncompressedSize) bytes)")
        }
    }
    
    func extractFile(_ filename: String) throws -> Data {
        guard let entry = centralDirectory.first(where: { $0.filename == filename }) else {
            print("SOGSZipArchive: File not found: \(filename)")
            throw SplatSOGSSceneReaderV2.SOGSV2Error.missingFile(filename)
        }
        
        print("SOGSZipArchive: Extracting \(filename)...")
        print("SOGSZipArchive: Entry details - compressed: \(entry.compressedSize), uncompressed: \(entry.uncompressedSize), method: \(entry.compressionMethod)")
        
        // Read local file header
        let localHeaderOffset = Int(entry.localHeaderOffset)
        guard localHeaderOffset + 30 <= zipData.count else {
            print("SOGSZipArchive: Invalid local header offset: \(localHeaderOffset), zip size: \(zipData.count)")
            throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("Invalid local header offset")
        }
        
        // Parse local header to get filename and extra field lengths using direct byte access
        // Local file header structure:
        // 0-3: signature (0x04034b50)
        // 4-5: version needed to extract
        // 6-7: general purpose bit flag
        // 8-9: compression method
        // 10-11: last mod file time
        // 12-13: last mod file date
        // 14-17: CRC-32
        // 18-21: compressed size
        // 22-25: uncompressed size
        // 26-27: filename length
        // 28-29: extra field length
        
        let filenameLength = UInt16(zipData[localHeaderOffset + 26]) | (UInt16(zipData[localHeaderOffset + 27]) << 8)
        let extraFieldLength = UInt16(zipData[localHeaderOffset + 28]) | (UInt16(zipData[localHeaderOffset + 29]) << 8)
        
        // SOGS ZIP files use data descriptors - check general purpose bit flag from central directory
        // The SOGS writer sets bit 3 (0x8) which indicates data descriptor is present
        // This means the actual file size comes from the central directory, not local header
        
        // Calculate file data offset (skip local header, filename, and extra fields)
        let fileDataOffset = localHeaderOffset + 30 + Int(filenameLength) + Int(extraFieldLength)
        
        // Use size from central directory entry (which is accurate for SOGS format)
        // SOGS uses uncompressed storage (method=0), so compressed size == uncompressed size
        let actualFileSize = min(Int(entry.compressedSize), Int(entry.uncompressedSize))
        let fileDataEnd = fileDataOffset + actualFileSize
        
        // Validate bounds - add some buffer for potential data descriptor (16 bytes)
        guard fileDataEnd + 16 <= zipData.count else {
            throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("File data extends beyond archive")
        }
        
        // Extract the actual file data
        let fileData = zipData[fileDataOffset..<fileDataEnd]
        
        // Handle compression - SOGS always uses method 0 (uncompressed/stored)
        if entry.compressionMethod == 0 {
            // Uncompressed/stored - this is what SOGS uses
            print("SOGSZipArchive: Extracted \(fileData.count) bytes (uncompressed)")
            return Data(fileData)
        } else {
            print("SOGSZipArchive: Unsupported compression method: \(entry.compressionMethod)")
            throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("Unsupported compression method")
        }
    }
    
    private static func parseCentralDirectory(_ zipData: Data) throws -> [ZipCentralDirectoryEntry] {
        // Find end of central directory record (search backwards)
        guard let endOfCentralDir = findEndOfCentralDirectory(zipData) else {
            throw SplatSOGSSceneReaderV2.SOGSV2Error.zipDecodingFailed("End of central directory not found")
        }
        
        // Parse central directory
        let centralDirOffset = Int(endOfCentralDir.centralDirectoryOffset)
        let numEntries = Int(endOfCentralDir.totalEntries)
        
        var entries: [ZipCentralDirectoryEntry] = []
        var currentOffset = centralDirOffset
        
        for _ in 0..<numEntries {
            guard currentOffset + 46 <= zipData.count else { break }
            
            // Parse central directory entry using direct byte access for safety
            // Central directory file header structure (relevant fields):
            // 0-3: signature (0x02014b50)
            // 4: version made by
            // 5: host OS
            // 6-7: version needed to extract
            // 8-9: general purpose bit flag
            // 10-11: compression method
            // 12-13: last mod file time
            // 14-15: last mod file date
            // 16-19: CRC-32
            // 20-23: compressed size
            // 24-27: uncompressed size
            // 28-29: filename length
            // 30-31: extra field length
            // 32-33: file comment length
            // 34-35: disk number start
            // 36-37: internal file attributes
            // 38-41: external file attributes
            // 42-45: relative offset of local header
            
            // Ensure we have enough bytes for all required fields
            guard currentOffset + 45 < zipData.count else { break }
            
            let compressionMethod = UInt16(zipData[currentOffset + 10]) | (UInt16(zipData[currentOffset + 11]) << 8)
            let compressedSize = UInt32(zipData[currentOffset + 20]) |
                               (UInt32(zipData[currentOffset + 21]) << 8) |
                               (UInt32(zipData[currentOffset + 22]) << 16) |
                               (UInt32(zipData[currentOffset + 23]) << 24)
            let uncompressedSize = UInt32(zipData[currentOffset + 24]) |
                                 (UInt32(zipData[currentOffset + 25]) << 8) |
                                 (UInt32(zipData[currentOffset + 26]) << 16) |
                                 (UInt32(zipData[currentOffset + 27]) << 24)
            let filenameLength = UInt16(zipData[currentOffset + 28]) | (UInt16(zipData[currentOffset + 29]) << 8)
            let extraFieldLength = UInt16(zipData[currentOffset + 30]) | (UInt16(zipData[currentOffset + 31]) << 8)
            let commentLength = UInt16(zipData[currentOffset + 32]) | (UInt16(zipData[currentOffset + 33]) << 8)
            let localHeaderOffset = UInt32(zipData[currentOffset + 42]) |
                                  (UInt32(zipData[currentOffset + 43]) << 8) |
                                  (UInt32(zipData[currentOffset + 44]) << 16) |
                                  (UInt32(zipData[currentOffset + 45]) << 24)
            
            // Extract filename
            let filenameStart = currentOffset + 46
            let filenameEnd = filenameStart + Int(filenameLength)
            guard filenameEnd <= zipData.count else { break }
            
            let filenameData = zipData[filenameStart..<filenameEnd]
            let filename = String(data: filenameData, encoding: .utf8) ?? ""
            
            entries.append(ZipCentralDirectoryEntry(
                filename: filename,
                compressedSize: compressedSize,
                uncompressedSize: uncompressedSize,
                localHeaderOffset: localHeaderOffset,
                compressionMethod: compressionMethod
            ))
            
            // Move to next entry
            currentOffset = filenameEnd + Int(extraFieldLength) + Int(commentLength)
        }
        
        return entries
    }
    
    private static func findEndOfCentralDirectory(_ zipData: Data) -> EndOfCentralDirectory? {
        // Search backwards for end of central directory signature (0x06054b50)
        let signature: [UInt8] = [0x50, 0x4b, 0x05, 0x06]
        
        // Ensure we have enough data for the minimum record size (22 bytes)
        guard zipData.count >= 22 else {
            return nil
        }
        
        // Calculate search range - search backwards from end, but not beyond max comment size
        let minOffset = max(0, zipData.count - 65535 - 22) // 65535 is max comment size
        let maxOffset = zipData.count - 22  // Need at least 22 bytes for the record
        
        // Validate range before searching
        guard minOffset <= maxOffset else {
            return nil
        }
        
        for i in (minOffset...maxOffset).reversed() {
            // Double-check bounds before accessing data
            guard i + 4 <= zipData.count else { continue }
            
            if zipData[i..<i + 4].elementsEqual(signature) {
                // Double-check we have enough data for the full record
                guard i + 22 <= zipData.count else { continue }
                
                // Found signature, parse the record using direct byte access for safety
                // End of central directory record structure:
                // 0-3: signature (already validated)
                // 4-5: number of this disk
                // 6-7: disk where central directory starts  
                // 8-9: number of central directory records on this disk
                // 10-11: total number of central directory records
                // 12-15: size of central directory
                // 16-19: offset of central directory
                // 20-21: comment length
                
                // Read values using direct array indexing with bounds checking
                guard i + 21 < zipData.count else { continue }
                
                let totalEntries = UInt16(zipData[i + 10]) | (UInt16(zipData[i + 11]) << 8)
                let centralDirectorySize = UInt32(zipData[i + 12]) | 
                                          (UInt32(zipData[i + 13]) << 8) |
                                          (UInt32(zipData[i + 14]) << 16) |
                                          (UInt32(zipData[i + 15]) << 24)
                let centralDirectoryOffset = UInt32(zipData[i + 16]) |
                                            (UInt32(zipData[i + 17]) << 8) |
                                            (UInt32(zipData[i + 18]) << 16) |
                                            (UInt32(zipData[i + 19]) << 24)
                
                return EndOfCentralDirectory(
                    totalEntries: totalEntries,
                    centralDirectorySize: centralDirectorySize,
                    centralDirectoryOffset: centralDirectoryOffset
                )
            }
        }
        
        return nil
    }
    
    private struct EndOfCentralDirectory {
        let totalEntries: UInt16
        let centralDirectorySize: UInt32
        let centralDirectoryOffset: UInt32
    }
}
