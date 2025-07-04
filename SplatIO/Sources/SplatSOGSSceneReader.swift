import Foundation
import ImageIO
import CoreGraphics

public class SplatSOGSSceneReader: SplatSceneReader {
    public enum SOGSError: Error {
        case invalidMetadata
        case missingFile(String)
        case webpDecodingFailed(String)
        case invalidTextureData
    }
    
    private let metaURL: URL
    private let baseURL: URL
    
    public init(_ metaURL: URL) throws {
        self.metaURL = metaURL
        self.baseURL = metaURL.deletingLastPathComponent()
        
        print("SplatSOGSSceneReader: Initializing with URL: \(metaURL.path)")
        print("SplatSOGSSceneReader: Base directory: \(baseURL.path)")
        
        // Validate that it's a SOGS meta.json file
        let filename = metaURL.lastPathComponent.lowercased()
        guard filename == "meta.json" || filename.hasSuffix(".json") else {
            print("SplatSOGSSceneReader: Invalid filename: \(filename)")
            throw SOGSError.invalidMetadata
        }
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        print("SplatSOGSSceneReader: Loading SOGS metadata from \(metaURL.path)")
        
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
            print("SplatSOGSSceneReader: Loaded \(metaData.count) bytes of metadata")
        } catch {
            print("SplatSOGSSceneReader: Failed to load metadata file: \(error)")
            throw SOGSError.invalidMetadata
        }
        
        let metadata: SOGSMetadata
        do {
            metadata = try JSONDecoder().decode(SOGSMetadata.self, from: metaData)
            print("SplatSOGSSceneReader: Successfully parsed metadata")
        } catch {
            print("SplatSOGSSceneReader: Failed to parse metadata JSON: \(error)")
            throw SOGSError.invalidMetadata
        }
        
        print("SplatSOGSSceneReader: Found \(metadata.means.shape[0]) splats")
        
        // Load WebP texture files
        do {
            let compressedData = try loadCompressedData(metadata: metadata)
            print("SplatSOGSSceneReader: Successfully loaded all WebP textures")
            
            // Decompress and convert to SplatScenePoint format
            return try decompressData(compressedData)
        } catch let error as SOGSError {
            print("SplatSOGSSceneReader: SOGS-specific error: \(error)")
            throw error
        } catch {
            print("SplatSOGSSceneReader: Unexpected error during WebP loading: \(error)")
            throw SOGSError.webpDecodingFailed("Unexpected error: \(error)")
        }
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
    
    private func loadCompressedData(metadata: SOGSMetadata) throws -> SOGSCompressedData {
        print("SplatSOGSSceneReader: Loading WebP texture files...")
        
        // Load means textures
        let means_l = try loadAndDecodeWebP(metadata.means.files[0])
        let means_u = try loadAndDecodeWebP(metadata.means.files[1])
        
        // Load other required textures
        let quats = try loadAndDecodeWebP(metadata.quats.files[0])
        let scales = try loadAndDecodeWebP(metadata.scales.files[0])
        let sh0 = try loadAndDecodeWebP(metadata.sh0.files[0])
        
        // Load optional spherical harmonics textures
        var sh_centroids: WebPDecoder.DecodedImage?
        var sh_labels: WebPDecoder.DecodedImage?
        
        // Only load SH data if we have SH bands > 0
        // Calculate potential SH bands based on texture width
        if let shN = metadata.shN, shN.files.count >= 2 {
            // First load the centroids to determine SH bands
            let tempCentroids = try loadAndDecodeWebP(shN.files[0])
            let shBands = calculateSHBands(width: tempCentroids.width)
            
            if shBands > 0 {
                sh_centroids = tempCentroids
                sh_labels = try loadAndDecodeWebP(shN.files[1])
                print("SplatSOGSSceneReader: Loaded SH data with \(shBands) bands")
            } else {
                print("SplatSOGSSceneReader: Skipping SH data loading - no valid bands detected")
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
    
    private func loadAndDecodeWebP(_ filename: String) throws -> WebPDecoder.DecodedImage {
        print("SplatSOGSSceneReader: Attempting to load WebP file: \(filename)")
        
        // Try multiple approaches for finding the file on iOS File Provider
        var fileURL = baseURL.appendingPathComponent(filename)
        var fileData: Data?
        
        // List all approaches we'll try
        let urlsToTry = [
            baseURL.appendingPathComponent(filename),
            metaURL.deletingLastPathComponent().appendingPathComponent(filename)
        ]
        
        // Try each URL with proper security scoped resource handling
        for tryURL in urlsToTry {
            print("SplatSOGSSceneReader: Trying URL: \(tryURL.path)")
            
            // First start accessing the parent directory
            let parentURL = tryURL.deletingLastPathComponent()
            let parentAccess = parentURL.startAccessingSecurityScopedResource()
            defer {
                if parentAccess {
                    parentURL.stopAccessingSecurityScopedResource()
                }
            }
            
            // Then try to access the file itself
            let fileAccess = tryURL.startAccessingSecurityScopedResource()
            defer {
                if fileAccess {
                    tryURL.stopAccessingSecurityScopedResource()
                }
            }
            
            // Check if file exists with enhanced access
            if FileManager.default.fileExists(atPath: tryURL.path) {
                print("SplatSOGSSceneReader: File exists at: \(tryURL.path)")
                
                do {
                    // Try to read the file data
                    fileData = try Data(contentsOf: tryURL)
                    fileURL = tryURL
                    print("SplatSOGSSceneReader: Successfully loaded \(fileData!.count) bytes from \(filename)")
                    break
                } catch {
                    print("SplatSOGSSceneReader: Failed to read file at \(tryURL.path): \(error)")
                    // Continue to next URL
                }
            } else {
                print("SplatSOGSSceneReader: File does not exist at: \(tryURL.path)")
            }
        }
        
        guard let webpData = fileData else {
            print("SplatSOGSSceneReader: WebP file not found: \(filename)")
            print("SplatSOGSSceneReader: Tried base URL: \(baseURL.path)")
            print("SplatSOGSSceneReader: Tried meta parent: \(metaURL.deletingLastPathComponent().path)")
            
            // List directory contents for debugging
            do {
                let parentURL = baseURL
                let parentAccess = parentURL.startAccessingSecurityScopedResource()
                defer {
                    if parentAccess {
                        parentURL.stopAccessingSecurityScopedResource()
                    }
                }
                
                let contents = try FileManager.default.contentsOfDirectory(at: parentURL, includingPropertiesForKeys: nil)
                print("SplatSOGSSceneReader: Directory contents at \(parentURL.path):")
                for file in contents {
                    print("  - \(file.lastPathComponent)")
                }
            } catch {
                print("SplatSOGSSceneReader: Could not list directory contents: \(error)")
            }
            
            throw SOGSError.missingFile(filename)
        }
        
        // Validate WebP signature
        if webpData.count >= 4 {
            let riffHeader = webpData.prefix(4)
            let webpSignature = webpData.dropFirst(8).prefix(4)
            
            print("SplatSOGSSceneReader: File header: \(riffHeader.map { String(format: "%02X", $0) }.joined())")
            if webpData.count >= 12 {
                print("SplatSOGSSceneReader: WebP signature: \(webpSignature.map { String(format: "%02X", $0) }.joined())")
            }
        }
        
        do {
            // Try Core Image first (iOS 14+/macOS 11+)
            print("SplatSOGSSceneReader: Attempting Core Image decode...")
            let result = try WebPDecoder.decode(webpData)
            print("SplatSOGSSceneReader: Successfully decoded \(filename) using Core Image - \(result.width)x\(result.height), \(result.bytesPerPixel) bpp")
            return result
        } catch {
            print("SplatSOGSSceneReader: Core Image decode failed for \(filename): \(error)")
            
            // Fallback to ImageIO if Core Image fails
            do {
                print("SplatSOGSSceneReader: Attempting ImageIO decode...")
                let result = try WebPDecoder.decodeWithImageIO(webpData)
                print("SplatSOGSSceneReader: Successfully decoded \(filename) using ImageIO - \(result.width)x\(result.height), \(result.bytesPerPixel) bpp")
                return result
            } catch {
                print("SplatSOGSSceneReader: ImageIO decode also failed for \(filename): \(error)")
                throw SOGSError.webpDecodingFailed("Failed to decode \(filename) with both Core Image and ImageIO: \(error)")
            }
        }
    }
    
    private func decompressData(_ compressedData: SOGSCompressedData) throws -> [SplatScenePoint] {
        print("SplatSOGSSceneReader: Decompressing \(compressedData.numSplats) splats...")
        print("SplatSOGSSceneReader: Texture dimensions: \(compressedData.textureWidth)x\(compressedData.textureHeight)")
        print("SplatSOGSSceneReader: SH bands: \(compressedData.shBands)")
        
        let iterator = SOGSIterator(compressedData)
        var points: [SplatScenePoint] = []
        points.reserveCapacity(compressedData.numSplats)
        
        for i in 0..<compressedData.numSplats {
            let point = iterator.readPoint(at: i)
            points.append(point)
            
            // Progress logging
            if i % 10000 == 0 {
                print("SplatSOGSSceneReader: Processed \(i)/\(compressedData.numSplats) splats")
            }
        }
        
        print("SplatSOGSSceneReader: Successfully decompressed \(points.count) points")
        return points
    }
    
    private func calculateSHBands(width: Int) -> Int {
        // Based on the PlayCanvas implementation:
        // 192: 1 band (64 * 3), 512: 2 bands (64 * 8), 960: 3 bands (64 * 15)
        switch width {
        case 192: return 1
        case 512: return 2
        case 960: return 3
        default: return 0
        }
    }
}

 